#!/usr/bin/env python3
"""Population for league training: a single roster of role-tagged members.

  past_main   — periodic snapshots of the training agent (PFSP curriculum)
  hof_anchor  — every generation-boundary snapshot, quota by rating (the
                anti-forgetting floor)

Members carry per-partner-mode PlackettLuce ratings (one set of weights
plays both modes) and the decayed exploitation EMA (P(member outscores the
training agent)) that drives the PFSP curriculum weights.

Opponent seats for a training table come from a 2-component per-seat
mixture (League.sample_table): the current agent with ``self_play_share``,
else a PFSP draw over past mains with a HOF floor — the validated retention
regime (Learning_System_Redesign_202607 §7.9). Exploiter members, their
edge-weighted seat heat and retirement clocks left with the September 2026
redesign (Training_Program_Redesign_202609 §4.3).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from openskill.models import PlackettLuce

from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.training.config import LeagueConfig

ROLE_PAST_MAIN = "past_main"
ROLE_HOF_ANCHOR = "hof_anchor"
ROLES = (ROLE_PAST_MAIN, ROLE_HOF_ANCHOR)

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
    parent_id: Optional[str] = None
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
        parent_id: str | None = None,
        initial_ratings=None,
    ) -> str:
        """Add (and immediately persist) a member."""
        if role not in ROLES:
            raise ValueError(f"unknown league role: {role}")
        member_id = f"{role}_{training_episodes}_{int(time.time())}_{len(self.members)}"
        meta = MemberMeta(
            member_id=member_id,
            role=role,
            creation_time=time.time(),
            training_episodes=int(training_episodes),
            parent_id=parent_id,
        )
        member = LeagueMember(agent, meta, ratings=initial_ratings)
        self.members.append(member)
        self._save_member(member)
        self._manage_size()
        return member_id

    def promote_to_hof(self, member_id: str) -> None:
        """Promote a member to HOF anchor (anti-forgetting floor), enforcing
        ``hof_quota`` by demoting the lowest-skill anchor back to past_main.

        HOF anchors are immune to skill pruning and get the ``hof_floor_prob``
        seat floor in ``_sample_pfsp``. Skill-based demotion is sound because
        anchors keep playing (they sit in the PFSP pool), so their ratings
        stay live on the current scale rather than frozen at entry.
        """
        member = self.get(member_id)
        if member is None:
            raise ValueError(f"unknown league member: {member_id}")
        member.meta.role = ROLE_HOF_ANCHOR
        self._save_member(member)
        while len(self.by_role(ROLE_HOF_ANCHOR)) > self.config.hof_quota:
            weakest = min(self.by_role(ROLE_HOF_ANCHOR), key=lambda m: m.skill())
            weakest.meta.role = ROLE_PAST_MAIN
            self._save_member(weakest)
        self._manage_size()  # a demotion may push past_mains over the cap

    def _manage_size(self) -> None:
        """Prune past_mains over the cap.

        Pruning keeps: all HOF anchors, the ``protect_newest`` most recent
        past_mains, then the highest-skill past_mains up to
        ``max_past_mains``."""
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
    # Table sampling
    # ------------------------------------------------------------------
    def sample_table(self, partner_mode: int, rng, n_seats: int = 4) -> list:
        """Sample opponents for one training table.

        Returns a list of length ``n_seats`` whose entries are LeagueMembers
        or the SELF_PLAY sentinel (driver substitutes a frozen copy of the
        current training agent). Members are sampled without replacement;
        an empty or exhausted roster falls back to self seats, which is how
        the bootstrap phase (empty league) plays pure self-play."""
        p_self = self.config.self_play_share
        pool_past = self.by_role(ROLE_PAST_MAIN) + self.by_role(ROLE_HOF_ANCHOR)
        seats: list = []
        used: set[str] = set()
        for _ in range(n_seats):
            if rng.random() < p_self:
                pick = SELF_PLAY
            else:
                pick = self._sample_pfsp(pool_past, used, rng) or SELF_PLAY
            if isinstance(pick, LeagueMember):
                used.add(pick.member_id)
            seats.append(pick)
        return seats

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
                agent = load_agent(str(pt))
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
        return "\n".join(lines)
