#!/usr/bin/env python3
"""Exploiter-bearing league for population training (plan:
notebooks/Exploiter_League_Plan_202606.md).

Replaces the dual-population PFSP museum in pfsp.py with a single roster of
role-tagged members:

  past_main      — periodic snapshots of the training agent (PFSP curriculum)
  main_exploiter — best responses trained vs a frozen main (adversarial
                   pressure; sampled hot while their measured edge lasts)
  hof_anchor     — strongest historical members (anti-forgetting floor)

Members carry per-partner-mode PlackettLuce ratings (one set of weights plays
both modes — the old JD/CA population split stored every snapshot twice for
no benefit) and the decayed exploitation EMA (P(member outscores training))
that drives both PFSP curriculum weights and exploiter slot heat.

Opponent seats for a training table come from a 3-component mixture
(League.sample_table): PFSP win-rate curriculum over past mains / edge-weighted
exploiter slots (capped) / current-self slots. This replaces the
anchor-block + pressure-slot + support-slot + diversity-multiplier scheduling
heuristics; strategic-profile clustering is gone entirely (exploiters provide
objective-driven diversity, which is the kind that matters).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from openskill.models import PlackettLuce

from config import LeagueConfig
from ppo import PPOAgent
from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

ROLE_PAST_MAIN = "past_main"
ROLE_MAIN_EXPLOITER = "main_exploiter"
ROLE_HOF_ANCHOR = "hof_anchor"
ROLES = (ROLE_PAST_MAIN, ROLE_MAIN_EXPLOITER, ROLE_HOF_ANCHOR)

PARTNER_MODES = (PARTNER_BY_JD, PARTNER_BY_CALLED_ACE)

# Sentinel returned by League.sample_table for a "current training agent" seat;
# the driver substitutes its own frozen copy of the training agent.
SELF_PLAY = "self"


@dataclass
class MemberMeta:
    member_id: str
    role: str
    creation_time: float
    training_episodes: int
    activation: str = "swish"
    parent_id: Optional[str] = None
    # Exploiter lineage: which league generation spawned it (-1 for past mains)
    # and the paired-deal edge (settlement score/deal vs the frozen main) measured
    # at its gate — the league's empirical-exploitability record.
    generation: int = -1
    gate_edge: Optional[float] = None
    games_played: int = 0
    total_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemberMeta":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


class LeagueMember:
    """One roster entry: weights + metadata + per-mode ratings + exploitation EMA."""

    def __init__(self, agent: PPOAgent, meta: MemberMeta, ratings=None):
        self.agent = agent
        self.meta = meta
        model = PlackettLuce()
        self.ratings = ratings or {mode: model.rating() for mode in PARTNER_MODES}
        self.exploitation_win_rate_ema: float = 0.5
        self.exploitation_samples: int = 0

    @property
    def member_id(self) -> str:
        return self.meta.member_id

    @property
    def role(self) -> str:
        return self.meta.role

    def rating(self, partner_mode: int):
        return self.ratings[partner_mode]

    def skill(self) -> float:
        """Mode-averaged rating μ (for pruning / HOF selection)."""
        return float(np.mean([r.mu for r in self.ratings.values()]))

    def record_vs_training_outcome(self, result: float, alpha: float = 0.05) -> None:
        """EMA-update P(this member outscores the training agent)."""
        result = min(max(float(result), 0.0), 1.0)
        self.exploitation_win_rate_ema = (
            1.0 - alpha
        ) * self.exploitation_win_rate_ema + alpha * result
        self.exploitation_samples += 1

    def add_game_result(self, score: float) -> None:
        self.meta.games_played += 1
        self.meta.total_score += float(score)


class League:
    """Role-tagged roster with persistence, table sampling and rating updates.

    Directory layout: ``<league_dir>/members/<member_id>.pt`` + ``...json``.
    """

    def __init__(self, league_dir: str, config: LeagueConfig | None = None):
        self.config = config or LeagueConfig()
        self.league_dir = Path(league_dir)
        self.members_dir = self.league_dir / "members"
        self.members_dir.mkdir(parents=True, exist_ok=True)
        self.rating_model = PlackettLuce()
        self.members: list[LeagueMember] = []
        self._load()

    # ------------------------------------------------------------------
    # Roster
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.members)

    def by_role(self, role: str) -> list[LeagueMember]:
        return [m for m in self.members if m.role == role]

    def get(self, member_id: str) -> Optional[LeagueMember]:
        for m in self.members:
            if m.member_id == member_id:
                return m
        return None

    def add_member(
        self,
        agent: PPOAgent,
        role: str,
        training_episodes: int,
        activation: str = "swish",
        parent_id: str | None = None,
        generation: int = -1,
        gate_edge: float | None = None,
        initial_ratings=None,
        initial_ema: float | None = None,
    ) -> str:
        """Add (and immediately persist) a member. Exploiters should pass
        ``gate_edge`` and ``initial_ema`` seeded from their measured edge so
        they sample hot from the first table."""
        if role not in ROLES:
            raise ValueError(f"unknown league role: {role}")
        member_id = f"{role}_{training_episodes}_{int(time.time())}_{len(self.members)}"
        meta = MemberMeta(
            member_id=member_id,
            role=role,
            creation_time=time.time(),
            training_episodes=int(training_episodes),
            activation=activation,
            parent_id=parent_id,
            generation=int(generation),
            gate_edge=gate_edge,
        )
        member = LeagueMember(agent, meta, ratings=initial_ratings)
        if initial_ema is not None:
            member.exploitation_win_rate_ema = float(initial_ema)
        self.members.append(member)
        self._save_member(member)
        self._manage_size()
        return member_id

    def _manage_size(self) -> None:
        """Retire cold exploiters; prune past_mains over the cap.

        Pruning keeps: all HOF anchors, all active exploiters, the
        ``protect_newest`` most recent past_mains, then the highest-skill
        past_mains up to ``max_past_mains``."""
        current_gen = max((m.meta.generation for m in self.members), default=-1)
        for m in self.by_role(ROLE_MAIN_EXPLOITER):
            old = (
                current_gen - m.meta.generation
                >= self.config.exploiter_retire_generations
            )
            if old:
                m.meta.role = ROLE_PAST_MAIN
                self._save_member(m)

        past = sorted(
            self.by_role(ROLE_PAST_MAIN),
            key=lambda m: m.meta.training_episodes,
            reverse=True,
        )
        if len(past) <= self.config.max_past_mains:
            return
        protected = past[: self.config.protect_newest]
        rest = sorted(
            past[self.config.protect_newest :], key=lambda m: m.skill(), reverse=True
        )
        keep = set(
            id(m)
            for m in protected + rest[: self.config.max_past_mains - len(protected)]
        )
        for m in past:
            if id(m) not in keep:
                self._delete_member(m)

    # ------------------------------------------------------------------
    # Table sampling (plan §3.3)
    # ------------------------------------------------------------------
    def exploiter_share(self) -> float:
        """Seat probability from the hottest active exploiter's FROZEN gate edge
        (settlement score/deal). Fixed at insertion, so this can't ratchet to zero
        when the binary table EMA dips below neutral."""
        edges = [m.meta.gate_edge or 0.0 for m in self.by_role(ROLE_MAIN_EXPLOITER)]
        if not edges:
            return 0.0
        heat = max(edges) / max(self.config.exploiter_edge_full, 1e-9)
        return self.config.exploiter_seat_cap * float(np.clip(heat, 0.0, 1.0))

    def sample_table(self, partner_mode: int, rng, n_seats: int = 4) -> list:
        """Sample opponents for one training table.

        Returns a list of length ``n_seats`` whose entries are LeagueMembers
        or the SELF_PLAY sentinel (driver substitutes a frozen copy of the
        current training agent). Members are sampled without replacement."""
        p_exp = self.exploiter_share()
        p_self = self.config.self_play_share
        pool_past = self.by_role(ROLE_PAST_MAIN) + self.by_role(ROLE_HOF_ANCHOR)
        pool_exp = self.by_role(ROLE_MAIN_EXPLOITER)
        seats: list = []
        used: set[str] = set()
        for _ in range(n_seats):
            r = rng.random()
            if r < p_exp:
                pick = self._sample_exploiter(pool_exp, used, rng)
            elif r < p_exp + p_self:
                pick = SELF_PLAY
            else:
                pick = self._sample_pfsp(pool_past, used, rng)
            if pick is None:  # component empty (small league) -> fall back
                pick = self._sample_pfsp(pool_past, used, rng) or SELF_PLAY
            if isinstance(pick, LeagueMember):
                used.add(pick.member_id)
            seats.append(pick)
        return seats

    def _sample_exploiter(self, pool, used, rng) -> Optional[LeagueMember]:
        avail = [m for m in pool if m.member_id not in used]
        if not avail:
            return None
        weights = [max(m.meta.gate_edge or 0.0, 1e-3) for m in avail]
        return avail[rng.choices(range(len(avail)), weights=weights)[0]]

    def _sample_pfsp(self, pool, used, rng) -> Optional[LeagueMember]:
        avail = [m for m in pool if m.member_id not in used]
        if not avail:
            return None
        cfg = self.config
        if rng.random() < cfg.hof_floor_prob:
            hof = [m for m in avail if m.role == ROLE_HOF_ANCHOR]
            if hof:
                return hof[rng.randrange(len(hof))]
        weights = []
        for m in avail:
            x = float(m.exploitation_win_rate_ema)
            conf = min(1.0, m.exploitation_samples / cfg.pfsp_conf_scale)
            base = cfg.pfsp_variable_weight * (x * (1.0 - x)) + cfg.pfsp_hard_weight * (
                x**cfg.pfsp_hard_power
            )
            base *= 0.25 + 0.75 * conf
            w = (1.0 - cfg.pfsp_uniform_mix) * base + cfg.pfsp_uniform_mix / len(avail)
            weights.append(max(w, 1e-3))
        return avail[rng.choices(range(len(avail)), weights=weights)[0]]

    # ------------------------------------------------------------------
    # Rating + EMA updates (ported from pfsp.update_ratings_with_training,
    # per-mode ratings, no profile bookkeeping)
    # ------------------------------------------------------------------
    def update_ratings_with_training(
        self,
        partner_mode: int,
        training_rating,
        final_scores: list[float],
        training_position: int,
        opponents_by_position: dict[int, LeagueMember],
        picker_seat: Optional[int],
        partner_seat: Optional[int],
        is_leaster: bool,
    ):
        """Update this mode's ratings for one 5-seat training game; returns the
        updated training rating. Also updates members' exploitation EMA and
        game stats."""
        if not final_scores or len(final_scores) != 5:
            return training_rating
        positions = [1, 2, 3, 4, 5]
        seat_scores = {pos: final_scores[pos - 1] for pos in positions}
        seat_ratings, placeholders = {}, {}
        for pos in positions:
            if pos == training_position:
                seat_ratings[pos] = training_rating
            elif pos in opponents_by_position:
                seat_ratings[pos] = opponents_by_position[pos].rating(partner_mode)
            else:
                seat_ratings[pos] = self.rating_model.rating()
                placeholders[pos] = True

        if is_leaster or not picker_seat:
            # Free-for-all: each seat its own team; EMA vs training per-seat.
            team_positions = [[p] for p in positions]
            scores = [seat_scores[p] for p in positions]
            ema_teams = {p: [p] for p in positions}
        else:
            picker_team = [picker_seat]
            if partner_seat and partner_seat != picker_seat:
                picker_team.append(partner_seat)
            defenders = [p for p in positions if p not in picker_team]
            team_positions = [picker_team, defenders]
            scores = [
                sum(seat_scores[p] for p in picker_team),
                sum(seat_scores[p] for p in defenders),
            ]
            ema_teams = {p: picker_team for p in picker_team}
            ema_teams.update({p: defenders for p in defenders})

        try:
            new_ratings = self.rating_model.rate(
                [[seat_ratings[p] for p in team] for team in team_positions],
                scores=scores,
            )
        except ValueError as err:
            logging.warning("rating update failed: %s", err)
            return training_rating

        for team, team_ratings in zip(team_positions, new_ratings):
            for pos, new_rating in zip(team, team_ratings):
                if pos == training_position:
                    training_rating = new_rating
                elif pos in opponents_by_position and pos not in placeholders:
                    opponents_by_position[pos].ratings[partner_mode] = new_rating

        # Exploitation EMA: compare team-average scores (handles 2v3), and only
        # for members on the opposing team — a teammate's score says nothing
        # about its ability to exploit the training agent.
        training_team = ema_teams[training_position]
        team_avg = {
            pos: float(np.mean([seat_scores[p] for p in team]))
            for pos, team in ema_teams.items()
        }
        training_avg = team_avg[training_position]
        for pos, member in opponents_by_position.items():
            score = seat_scores.get(pos)
            if member is None or score is None:
                continue
            member.add_game_result(score)
            if pos in training_team:
                continue
            opp_avg = team_avg[pos]
            result = (
                1.0
                if opp_avg > training_avg
                else 0.0
                if opp_avg < training_avg
                else 0.5
            )
            member.record_vs_training_outcome(result)
        return training_rating

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _member_paths(self, member_id: str) -> tuple[Path, Path]:
        return (
            self.members_dir / f"{member_id}.pt",
            self.members_dir / f"{member_id}.json",
        )

    def _save_member(self, member: LeagueMember) -> None:
        pt, js = self._member_paths(member.member_id)
        if not pt.exists():
            member.agent.save(str(pt))
        data = member.meta.to_dict()
        data["ratings"] = {
            str(mode): {"mu": float(r.mu), "sigma": float(r.sigma)}
            for mode, r in member.ratings.items()
        }
        data["exploitation_win_rate_ema"] = float(member.exploitation_win_rate_ema)
        data["exploitation_samples"] = int(member.exploitation_samples)
        with open(js, "w") as f:
            json.dump(data, f, indent=2)

    def save(self) -> None:
        for m in self.members:
            self._save_member(m)

    def _delete_member(self, member: LeagueMember) -> None:
        for p in self._member_paths(member.member_id):
            p.unlink(missing_ok=True)
        self.members.remove(member)

    def _load(self) -> None:
        for js in sorted(self.members_dir.glob("*.json")):
            try:
                with open(js) as f:
                    data = json.load(f)
                meta = MemberMeta.from_dict(data)
                pt, _ = self._member_paths(meta.member_id)
                agent = PPOAgent(len(ACTIONS), activation=meta.activation)
                agent.load(str(pt), load_optimizers=False)
                ratings = {
                    int(mode): self.rating_model.rating(mu=rs["mu"], sigma=rs["sigma"])
                    for mode, rs in data.get("ratings", {}).items()
                }
                member = LeagueMember(agent, meta, ratings=ratings or None)
                member.exploitation_win_rate_ema = float(
                    data.get("exploitation_win_rate_ema", 0.5)
                )
                member.exploitation_samples = int(data.get("exploitation_samples", 0))
                self.members.append(member)
            except Exception as err:  # noqa: BLE001 - skip corrupt entries, keep loading
                logging.warning("failed to load league member %s: %s", js, err)

    # ------------------------------------------------------------------
    # Legacy migration (plan §4)
    # ------------------------------------------------------------------
    @classmethod
    def migrate_legacy(
        cls,
        old_population_dir: str,
        league_dir: str,
        config: LeagueConfig | None = None,
        keep_top_k: int | None = None,
    ) -> "League":
        """Ingest an old dual-population directory (jd_agents/ +
        called_ace_agents/) as past_mains.

        The old trainer saved every snapshot twice (once per mode subdir);
        copies are merged by checkpoint-file hash (fallback: matching
        training_episodes + creation second), with each copy contributing its
        mode's rating. Keeps the ``keep_top_k`` strongest (mode-max μ) plus the
        newest few; tags the strongest ``hof_quota`` as HOF anchors."""
        config = config or LeagueConfig()
        keep_top_k = keep_top_k or config.max_past_mains
        old = Path(old_population_dir)
        mode_for_subdir = {
            "jd_agents": PARTNER_BY_JD,
            "called_ace_agents": PARTNER_BY_CALLED_ACE,
        }

        merged: dict[str, dict] = {}
        for subdir, mode in mode_for_subdir.items():
            for js in sorted((old / subdir).glob("*_metadata.json")):
                with open(js) as f:
                    data = json.load(f)
                pt = js.with_name(js.name.replace("_metadata.json", ".pt"))
                if not pt.exists():
                    continue
                h = hashlib.sha256(pt.read_bytes()).hexdigest()
                key = h
                if key not in merged:
                    # Fallback merge key for non-byte-identical twin saves
                    twin = f"ep{data.get('training_episodes')}_t{int(data.get('creation_time', 0))}"
                    for k, v in merged.items():
                        if v["twin"] == twin:
                            key = k
                            break
                entry = merged.setdefault(
                    key,
                    {
                        "pt": pt,
                        "data": data,
                        "ratings": {},
                        "ema": [],
                        "twin": f"ep{data.get('training_episodes')}_t{int(data.get('creation_time', 0))}",
                    },
                )
                entry["ratings"][mode] = (
                    float(data.get("rating_mu", 25.0)),
                    float(data.get("rating_sigma", 25.0 / 3)),
                )
                entry["ema"].append(float(data.get("exploitation_win_rate_ema", 0.5)))

        entries = sorted(
            merged.values(),
            key=lambda e: max(mu for mu, _ in e["ratings"].values()),
            reverse=True,
        )
        newest = sorted(
            merged.values(),
            key=lambda e: int(e["data"].get("training_episodes", 0)),
            reverse=True,
        )[: config.protect_newest]
        chosen, seen = [], set()
        for e in newest + entries:
            if id(e) not in seen:
                chosen.append(e)
                seen.add(id(e))
            if len(chosen) >= keep_top_k:
                break

        league = cls(league_dir, config)
        model = PlackettLuce()
        hof_cutoff = sorted(
            (max(mu for mu, _ in e["ratings"].values()) for e in chosen), reverse=True
        )[: config.hof_quota]
        for e in chosen:
            data = e["data"]
            skill = max(mu for mu, _ in e["ratings"].values())
            role = (
                ROLE_HOF_ANCHOR
                if hof_cutoff
                and skill >= hof_cutoff[-1]
                and len(league.by_role(ROLE_HOF_ANCHOR)) < config.hof_quota
                else ROLE_PAST_MAIN
            )
            member_id = f"legacy_{data.get('agent_id', 'unknown')}"
            meta = MemberMeta(
                member_id=member_id,
                role=role,
                creation_time=float(data.get("creation_time", time.time())),
                training_episodes=int(data.get("training_episodes", 0)),
                activation=data.get("activation", "swish"),
                parent_id=data.get("parent_id"),
            )
            pt_dst, js_dst = league._member_paths(member_id)
            shutil.copyfile(e["pt"], pt_dst)
            agent = PPOAgent(len(ACTIONS), activation=meta.activation)
            agent.load(str(pt_dst), load_optimizers=False)
            ratings = {
                mode: model.rating(mu=mu, sigma=sigma)
                for mode, (mu, sigma) in e["ratings"].items()
            }
            for mode in PARTNER_MODES:  # twin may be missing one mode
                ratings.setdefault(mode, model.rating())
            member = LeagueMember(agent, meta, ratings=ratings)
            member.exploitation_win_rate_ema = float(np.mean(e["ema"]))
            league.members.append(member)
            league._save_member(member)
        return league

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"League ({len(self.members)} members, dir={self.league_dir})"]
        for role in ROLES:
            ms = self.by_role(role)
            if not ms:
                continue
            skills = [m.skill() for m in ms]
            emas = [m.exploitation_win_rate_ema for m in ms]
            lines.append(
                f"  {role}: n={len(ms)}  skill μ {np.mean(skills):+.1f} "
                f"[{min(skills):+.1f}, {max(skills):+.1f}]  "
                f"EMA {np.mean(emas):.2f} (max {max(emas):.2f})"
            )
        lines.append(f"  exploiter seat share: {self.exploiter_share():.2f}")
        return "\n".join(lines)
