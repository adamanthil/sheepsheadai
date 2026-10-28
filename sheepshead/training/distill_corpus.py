#!/usr/bin/env python3
"""Search corpus generator for policy iteration (CE_Teacher_Design §17.3,
§20.3; Training_Program_Redesign §4.4 step 1).

Plays frozen-theta_k self-play games (all five seats collected, per-seat
episode streams) and searches a schedule of PLAY decisions with the ISMCTS
committee (``CommitteeConfig``). Every row carries a partition label the
projection reads (``policy_iteration.py``):

  resolved   searched, and the committee's between-card Q spread beats
             its own replicate noise (shrink w > 0). Committee acting
             plays here; the §20 target is built in Stage 2.
  unresolved searched, the spread within the noise (w = 0): same
             evidence, same Stage-2 treatment (Stage 2 relabels both
             "targeted"; where the pooled advantage is ~0 the target IS
             the prior). The split never reads theta_k's policy.
  retention  search cannot speak there: bidding heads (pick/partner/bury,
             ALONE declaration included) and leaster play — anchored to
             theta_k's act-time policy. Alone-game PLAY is searched
             (the 1v4 determinization has no hidden-partner uncertainty).
  none       eligible-but-unsearched play (the p-schedule passed it
             over), forced nodes, committee failures: no policy loss,
             still in the value-regression stream.

Offline phase purity (Expert Iteration — Anthony et al. 2017; AlphaGo
Zero / AlphaZero — Silver et al. 2017/2018): nothing updates during
generation, so the expert, the acting policy and the anchor are the SAME
frozen network. Acting is the policy's own stochastic self-play (DAgger —
Ross et al. 2011) except at resolved rows of committee-acted games
(``--committee-act-frac``, 1.0 in the §20.14 recipe: +0.0055 play vs
student acting, CE_Teacher_Design §20.13 add. 16), where the argmax of
the legacy pi_gumbel readout (``tilt_summary_to_target``) replaces the
sampled card so the corpus samples the search-preferred lines. That
readout only steers the trajectory; the Stage-2 targets are built later
by ``policy_iteration.py``, and every row's anchor stays theta_k's
act-time policy whichever card was played.

Anchors are theta_k's act-time probability vectors (the ``act()`` stash)
at the true recurrent state of the realized trajectory, stored on every
row with two or more legal actions. Every SEARCHED row stores the pooled
committee evidence (``search_q`` / ``search_q_var`` / ``search_n`` /
``search_prior`` / ``search_noise_var`` / ``search_spread``; see
``pfsp_runtime.CommitteeSummary``) — row schema 3.

The search schedule (§20.10): leads at ``p_base * boost_lead`` (clipped
to ``p_max``), follows at ``p_base``; a nonzero floor keeps every eligible
class covered (annealed-bias rationale of prioritized sampling — Schaul
et al. 2016). Node telemetry (``--node-telemetry``) writes one JSONL row
per searched node (class, regime, w, gap, spread, per-replicate top-pair
Q diffs), the calibration instrument for the shrinkage noise model.

Usage:
  uv run python -m sheepshead.training.distill_corpus \\
      --ckpt runs/rc/league/final.pt --out-dir runs/rc/pi/iter1/corpus \\
      --games 2000 --workers 8 --p-base 0.5 --boost-lead 2 --p-max 1.0 \\
      --node-telemetry runs/rc/pi/iter1/corpus/nodes.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from sheepshead.agent.observation import (
    last_trick_observation_for,
    observation_for,
)
from sheepshead.game import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, TRUMP, Game
from sheepshead.rewards import RETURN_SCALE

_W: dict = {}  # per-worker state (agent, teacher, config)

# Corpus row schema version written to the manifest (3 = the rows described
# in the module docstring; 2 = the same rows with searched rows labeled
# "override" / "endorsed", read through corpus_rows.load_shard; the §17
# pilot's schema 1 is no longer produced).
ROW_SCHEMA_VERSION = 3


# --------------------------------------------------------------------------- #
# p-schedule (§17.2)
# --------------------------------------------------------------------------- #
def lead_features(game, player) -> tuple[bool, bool]:
    """(is_lead, called_suit_eligible) for a standard-game play decision.

    called_suit_eligible mirrors the calibrated-instrument definition
    (ceiling study / E9): a non-partnerish seat leading while the called
    suit is unplayed, holding both a called-suit fail card and an
    alternative."""
    is_lead = all(c == "" for c in game.history[game.current_trick])
    if not is_lead:
        return False, False
    is_partnerish = player.is_picker or player.is_partner or player.is_secret_partner
    called = game.called_card
    cs = (
        not is_partnerish
        and bool(called)
        and not game.was_called_suit_played
        and any(c not in TRUMP and c[-1] == called[-1] for c in player.hand)
        and any(c in TRUMP or c[-1] != called[-1] for c in player.hand)
    )
    return True, cs


def schedule_p(
    is_lead: bool,
    called_suit_eligible: bool,
    *,
    p_base: float,
    boost_lead: float,
    boost_cs: float,
    p_min: float,
    p_max: float,
) -> float:
    """§17.2: p = clip(p0 * b_lead^[lead] * b_cs^[cs], p_min, p_max). The
    nonzero floor keeps every eligible class covered (annealed-bias
    rationale of prioritized sampling — Schaul et al. 2016)."""
    p = p_base
    if is_lead:
        p *= boost_lead
    if called_suit_eligible:
        p *= boost_cs
    return min(max(p, p_min), p_max)


def node_class(game, player, head: str) -> str:
    """Manifest/telemetry class label: bidding head name, or
    ``{regime}|{play_cell}`` for play decisions."""
    if head != "play":
        return head
    from sheepshead.training.pfsp_runtime import play_cell

    regime = "leaster" if game.is_leaster else ("alone" if game.alone_called else "std")
    return f"{regime}|{play_cell(game, player)}"


def _replicate_top_pair_diffs(replicates, valid_actions) -> tuple[list, list]:
    """(top2_actions, per-replicate Q diffs for that pair) — the paired-
    replicate statistic behind the §17.6 noise-floor calibration (the
    §12.8 instrument, re-emitted per corpus node)."""
    acts = sorted(valid_actions)
    usable = [r for r in replicates if r["ok"] and r.get("root_q") is not None]
    pooled = {}
    for a in acts:
        obs = [float(r["root_q"][a]) for r in usable if r["root_n"].get(a, 0.0) > 0.0]
        if obs:
            pooled[a] = float(np.mean(obs))
    if len(pooled) < 2:
        return [], []
    a1, a2 = sorted(pooled, key=lambda a: pooled[a], reverse=True)[:2]
    diffs = [
        float(r["root_q"][a1]) - float(r["root_q"][a2])
        for r in usable
        if r["root_n"].get(a1, 0.0) > 0.0 and r["root_n"].get(a2, 0.0) > 0.0
    ]
    return [a1, a2], diffs


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def worker_init(init_args: dict) -> None:
    import torch

    torch.set_num_threads(int(init_args.get("torch_threads", 1)))
    routed = init_args.get("routed_encoder")
    if routed:
        # Throughput-only (§16.6: 1.50x on committee search); the shadow
        # never needs re-syncing here because the weights are frozen.
        from sheepshead.agent.compiled_encoder import enable_routed_encoder

        enable_routed_encoder(32, None, device=routed)

    from sheepshead.agent.ppo import load_agent
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    agent = load_agent(init_args["ckpt"])
    agent.stash_action_probs = True
    # Terminal-reward regime: undiscounted search backups (matches the
    # league teacher's teacher_gamma=1.0).
    agent.gamma = 1.0
    iters = int(init_args["iters"])
    # Trick-indexed committee budget (CE_Teacher_Design §20.13 addendum 29 +
    # operator amendment): lead nodes at listed tricks get their own budget
    # (t0 leads 1024, t1 leads 512 in the pinned recipe — the convention
    # cells plus the most branching / farthest-from-terminal decisions);
    # everything else runs at --iters. One teacher instance per distinct
    # budget; the trick-indexed rollout-depth schedule is the precedent.
    schedule = parse_iters_schedule(init_args.get("iters_schedule"))
    teachers = {
        budget: ISMCTSTeacher(
            agent,
            ISMCTSConfig(
                iters={h: budget for h in ("pick", "partner", "bury", "play")}
            ),
        )
        for budget in {iters, *schedule.values()}
    }
    set_worker_state(
        agent, teachers[iters], init_args, teachers=teachers, schedule=schedule
    )


def set_worker_state(agent, teacher, args: dict, teachers=None, schedule=None) -> None:
    """Install this process's corpus-generation state: the acting agent, the
    default search teacher (anything with ``search_committee`` and
    ``config``), the generation args, and optionally the per-budget teachers
    + trick-indexed schedule (``teacher_for_node``). ``worker_init`` builds
    these from a checkpoint; tests install stand-ins directly."""
    _W.clear()
    _W.update(
        {
            "agent": agent,
            "teacher": teacher,
            "teachers": teachers or {},
            "schedule": schedule or {},
            "args": args,
        }
    )


def clear_worker_state() -> None:
    _W.clear()


def parse_iters_schedule(spec) -> dict:
    """``"t0-lead:1024,t1-lead:512"`` -> ``{(0, True): 1024, (1, True): 512}``,
    keyed by (trick, is_lead). Empty/None -> {} (every node at --iters)."""
    out: dict = {}
    if not spec:
        return out
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            key, budget = item.split(":")
            trick_s, kind = key.strip().split("-")
            if not trick_s.startswith("t") or kind not in ("lead", "follow"):
                raise ValueError(item)
            out[(int(trick_s[1:]), kind == "lead")] = int(budget)
        except ValueError:
            raise SystemExit(
                f"bad --iters-schedule entry {item!r} (expected e.g. t0-lead:1024)"
            ) from None
    return out


def budget_for_node(trick: int, is_lead: bool) -> int | None:
    """The scheduled committee budget for a play node, or None for the
    default (--iters) budget."""
    return _W.get("schedule", {}).get((int(trick), bool(is_lead)))


def teacher_for_node(trick: int, is_lead: bool):
    """The committee teacher for a play node under the trick-indexed schedule."""
    budget = budget_for_node(trick, is_lead)
    teachers = _W.get("teachers") or {}
    if budget is not None and budget in teachers:
        return teachers[budget]
    return _W["teacher"]


def _search_node(
    game, player, valid_actions, forced_public, det_rng, anchor, teacher=None
):
    """Run the committee, pool it (``CommitteeSummary``) and build the legacy
    CE target (base_prior = the act-time stash, the §16.6 zero-gradient
    abstention referent). Returns
    (target_list | None, info | None, summary | None, telemetry_extras);
    the first three are all None when the committee produced no usable
    root statistics."""
    from sheepshead.training.pfsp_runtime import (
        summarize_committee,
        tilt_summary_to_target,
    )

    init_args = _W["args"]
    teacher = teacher if teacher is not None else _W["teacher"]
    rngs = [
        random.Random(det_rng.getrandbits(64))
        for _ in range(int(init_args["replicates"]))
    ]
    replicates = teacher.search_committee(
        game,
        player.position,
        list(forced_public),
        rngs,
        d_rollout=int(init_args["d_rollout"]),
    )
    summary = summarize_committee(
        replicates,
        valid_actions,
        shrink_nu=float(init_args["shrink_nu"]),
        shrink_s2_global=float(init_args["shrink_s2_global"]),
    )
    top_pair, pair_diffs = _replicate_top_pair_diffs(replicates, valid_actions)
    if summary is None:
        return None, None, None, (top_pair, pair_diffs)
    target, info = tilt_summary_to_target(
        summary,
        gumbel_c_visit=teacher.config.gumbel_c_visit,
        gumbel_c_scale=teacher.config.gumbel_c_scale,
        base_prior=anchor,
    )
    return [float(x) for x in target], info, summary, (top_pair, pair_diffs)


def play_corpus_game(task: tuple) -> dict:
    """One self-play game, all five seats collected. Returns per-seat
    episode event lists (store_episode_events schema + distill keys),
    node-telemetry rows and per-class counters."""
    import torch

    from sheepshead.ismcts import infer_head, is_private_action
    from sheepshead.rewards import (
        compute_any_unseen_trump_higher_than_hand,
        compute_known_points_rel,
        compute_seen_trump_mask,
    )

    # (game_idx, mode[, committee_act]); tests still pass the two-tuple.
    game_idx, mode = task[0], task[1]
    committee_act = bool(task[2]) if len(task) > 2 else False
    init_args = _W["args"]
    agent = _W["agent"]
    base_seed = int(init_args["seed"])
    # Per-game deterministic streams, independent of pool scheduling.
    game_rng = random.Random((base_seed << 20) ^ game_idx)
    torch.manual_seed((base_seed ^ (game_idx * 0x9E3779B1)) & 0x7FFFFFFF)
    det_rng = random.Random(game_rng.getrandbits(64))
    collect_oracle = bool(init_args["collect_oracle"])

    game = Game(partner_selection_mode=mode, seed=game_rng.randint(0, 2**31 - 1))
    agent.reset_recurrent_state()

    seat_transitions: dict[int, list] = {pos: [] for pos in range(1, 6)}
    forced_public: list[tuple[int, int]] = []
    telemetry_rows: list[dict] = []
    counts: dict[str, dict] = defaultdict(
        lambda: {"nodes": 0, "searched": 0, "resolved": 0, "unresolved": 0, "failed": 0}
    )
    gaps: list[float] = []
    committee_acted = 0

    while not game.is_done():
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            while valid_actions:
                state = observation_for(player, agent)
                action, log_prob, value = agent.act(
                    state, valid_actions, player.position
                )
                anchor = None
                probs = agent.last_action_probs
                if probs is not None:
                    acts = sorted(valid_actions)
                    a = np.clip(
                        np.array([float(probs[i - 1]) for i in acts], dtype=np.float64),
                        1e-12,
                        None,
                    )
                    anchor = a / a.sum()

                head = infer_head(valid_actions)
                cls = node_class(game, player, head)
                counts[cls]["nodes"] += 1
                dset = "none"
                target_list = None
                info = None
                summary = None
                cs_ids = None
                if len(valid_actions) >= 2:
                    searchable_play = head == "play" and not game.is_leaster
                    if not searchable_play:
                        dset = "retention"
                    else:
                        if game.alone_called:
                            is_lead, cs_elig = (
                                all(c == "" for c in game.history[game.current_trick]),
                                False,
                            )
                        else:
                            is_lead, cs_elig = lead_features(game, player)
                        called = game.called_card
                        if cs_elig and called is not None:
                            # Adherent called-suit leads for the trainer's
                            # convention telemetry (§17.4 amendment): the
                            # called card / suit-played status exist only
                            # here at generation time — def-lead and
                            # partner-lead adherence are derivable from the
                            # stored mask, called-suit is not.
                            cs_ids = [
                                a
                                for a in sorted(valid_actions)
                                if ACTIONS[a - 1].startswith("PLAY ")
                                and ACTIONS[a - 1][5:] not in TRUMP
                                and ACTIONS[a - 1][-1] == called[-1]
                            ]
                        p = schedule_p(
                            is_lead,
                            cs_elig,
                            p_base=float(init_args["p_base"]),
                            boost_lead=float(init_args["boost_lead"]),
                            boost_cs=float(init_args["boost_cs"]),
                            p_min=float(init_args["p_min"]),
                            p_max=float(init_args["p_max"]),
                        )
                        if det_rng.random() < p:
                            counts[cls]["searched"] += 1
                            target_list, info, summary, (top_pair, pair_diffs) = (
                                _search_node(
                                    game,
                                    player,
                                    valid_actions,
                                    forced_public,
                                    det_rng,
                                    anchor,
                                    teacher=teacher_for_node(
                                        game.current_trick, is_lead
                                    ),
                                )
                            )
                            if target_list is None:
                                dset = "none"
                                counts[cls]["failed"] += 1
                            elif info is not None and info["w"] > 0.0:
                                dset = "resolved"
                                counts[cls]["resolved"] += 1
                                gaps.append(info["gap"])
                            else:
                                dset = "unresolved"
                                counts[cls]["unresolved"] += 1
                            telemetry_rows.append(
                                {
                                    "game": game_idx,
                                    "class": cls,
                                    "iters": int(
                                        budget_for_node(game.current_trick, is_lead)
                                        or init_args["iters"]
                                    ),
                                    "n_valid": len(valid_actions),
                                    "w": info["w"] if info else None,
                                    "gap": info["gap"] if info else None,
                                    "spread": info["spread"] if info else None,
                                    "top_pair": top_pair,
                                    "pair_diffs": pair_diffs,
                                }
                            )
                            # Committee acting (§20.13 addendum 16: +0.0055
                            # play vs student acting): play the resolved
                            # target's argmax so the corpus samples the lines
                            # the search prefers; the stored log_prob is the
                            # act-time policy's for the acted card.
                            if (
                                committee_act
                                and dset == "resolved"
                                and target_list is not None
                                and anchor is not None
                            ):
                                acts = sorted(valid_actions)
                                acted = acts[int(np.argmax(target_list))]
                                if acted != action:
                                    committee_acted += 1
                                    action = acted
                                    log_prob = float(np.log(anchor[acts.index(acted)]))

                transition = {
                    "kind": "action",
                    "state": state,
                    "action": action,
                    "log_prob": float(log_prob),
                    "value": float(value),
                    "valid_actions": set(valid_actions),
                    "player_id": player.position,
                    "secret_partner_label": 1.0 if player.is_secret_partner else 0.0,
                    "points_label": compute_known_points_rel(player),
                    "seen_trump_mask_label": compute_seen_trump_mask(player),
                    "unseen_trump_higher_than_hand_label": (
                        compute_any_unseen_trump_higher_than_hand(player)
                    ),
                    # Distill annotations (§17.3)
                    "distill_set": dset,
                    "node_class": cls,
                    "conv_cs_ids": cs_ids,
                    "has_search_target": dset == "resolved",
                    "search_target": target_list if dset == "resolved" else None,
                    "search_gap": float(info["gap"]) if info else 0.0,
                    "search_w": float(info["w"]) if info else 0.0,
                    # Schema 2: the act-time stash on every multi-action row;
                    # forced rows (one legal action) have nothing to anchor.
                    "anchor_probs": (
                        [float(x) for x in anchor]
                        if anchor is not None and len(valid_actions) >= 2
                        else None
                    ),
                }
                if summary is not None:
                    # Schema 2: the pooled committee evidence on every
                    # searched row, w = 0 included (the §20 trainer reads
                    # these; the legacy target above is act-time readout).
                    transition.update(summary.as_row_fields())
                if collect_oracle:
                    transition["oracle_state"] = player.get_oracle_state_dict()
                seat_transitions[player.position].append(transition)

                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)

                if game.was_trick_just_completed and not game.is_done():
                    for seat in game.players:
                        obs_state = last_trick_observation_for(seat, agent)
                        agent.observe(obs_state, player_id=seat.position)
                        obs = {"kind": "observation", "state": obs_state}
                        if collect_oracle:
                            obs["oracle_state"] = (
                                seat.get_last_trick_oracle_state_dict()
                            )
                        seat_transitions[seat.position].append(obs)

                valid_actions = player.get_valid_action_ids()

    final_scores = [p.get_score() for p in game.players]
    episodes = []
    for pos in range(1, 6):
        events = seat_transitions[pos]
        score = float(final_scores[pos - 1])
        last_action = max(
            (i for i, e in enumerate(events) if e["kind"] == "action"), default=None
        )
        for i, ev in enumerate(events):
            if ev["kind"] == "action":
                # Terminal-only reward on the seat's last action (the
                # trainer's MC value target reads final_return directly).
                ev["reward"] = score / RETURN_SCALE if i == last_action else 0.0
                ev["win_label"] = 1.0 if score > 0 else 0.0
                ev["final_return_label"] = score
        episodes.append(events)

    return {
        "game": game_idx,
        "mode": "called" if mode == PARTNER_BY_CALLED_ACE else "jd",
        "committee_act": committee_act,
        "committee_acted": committee_acted,
        "is_leaster": bool(game.is_leaster),
        "alone_called": bool(game.alone_called),
        "episodes": episodes,
        "telemetry": telemetry_rows,
        "counts": dict(counts),
        "gaps": gaps,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _git_rev() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# Config keys that do not shape the rows, so a resume may change them.
# ``start_game`` is the retired index-boundary resume flag, still present in
# manifests written before ``--resume``.
RESUME_FREE_KEYS = frozenset({"workers", "torch_threads", "resume", "start_game"})


def flushed_games(out_dir: str, prior: dict) -> set[int]:
    """The game indices the prior invocation's flushed shards hold: what a
    resume must not replay. Shard entries list their games; manifests from
    before that are read from the shards' own ``games`` metadata."""
    from sheepshead.training.corpus_rows import load_shard

    done: set[int] = set()
    for entry in prior["shards"]:
        if "games" in entry:
            done.update(entry["games"])
        else:
            shard = load_shard(os.path.join(out_dir, entry["path"]))
            done.update(int(g["game"]) for g in shard["games"])
    return done


def check_resumable(prior: dict, manifest: dict) -> None:
    """Refuse to extend the prior corpus with rows a different recipe or
    checkpoint would label: the two halves would mix silently."""
    if prior.get("row_schema", 1) != manifest["row_schema"]:
        raise SystemExit(
            f"cannot resume a row-schema {prior.get('row_schema', 1)} corpus "
            f"with schema {manifest['row_schema']} rows (labels would mix)"
        )
    if prior.get("ckpt_sha256_16") != manifest["ckpt_sha256_16"]:
        raise SystemExit(
            f"cannot resume: the corpus was generated from checkpoint "
            f"{prior.get('ckpt')} (sha {prior.get('ckpt_sha256_16')}), not "
            f"{manifest['ckpt']} (sha {manifest['ckpt_sha256_16']})"
        )
    old, new = prior.get("config", {}), manifest["config"]
    differ = sorted(
        k
        for k in (old.keys() | new.keys()) - RESUME_FREE_KEYS
        if old.get(k, "<absent>") != new.get(k, "<absent>")
    )
    if differ:
        raise SystemExit(
            "cannot resume: the generation config differs from the corpus's: "
            + ", ".join(
                f"{k} {old.get(k, '<absent>')!r} -> {new.get(k, '<absent>')!r}"
                for k in differ
            )
        )


def main() -> int:
    from sheepshead.training.config import CommitteeConfig

    sc = CommitteeConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="frozen theta_k checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-games", type=int, default=200)
    ap.add_argument("--no-oracle", dest="collect_oracle", action="store_false")
    ap.add_argument("--iters", type=int, default=sc.iters)
    ap.add_argument(
        "--iters-schedule",
        default=None,
        help="trick-indexed committee budget for LEAD/FOLLOW nodes, e.g. "
        "'t0-lead:1024,t1-lead:512' (§20.13 addendum 29 + amendment: 256 does "
        "not resolve the lead conventions; t0 leads want 1024, t1 leads 512; "
        "leads are ~5.5%% of searched nodes per trick, so this costs ~+22%%). "
        "Unlisted nodes run at --iters",
    )
    ap.add_argument("--replicates", type=int, default=sc.replicates)
    ap.add_argument("--d-rollout", type=int, default=sc.d_rollout)
    ap.add_argument("--shrink-nu", type=float, default=sc.shrink_nu)
    ap.add_argument("--shrink-s2-global", type=float, default=sc.shrink_s2_global)
    # §17.2 schedule
    ap.add_argument("--p-base", type=float, default=0.10)
    ap.add_argument("--boost-lead", type=float, default=1.25)
    ap.add_argument("--boost-cs", type=float, default=1.5)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--p-max", type=float, default=0.25)
    ap.add_argument(
        "--committee-act-frac",
        type=float,
        default=1.0,
        help="fraction of games in which resolved search targets are ACTED "
        "(the committee's argmax replaces the policy's card): the corpus then "
        "samples the search-preferred lines. 1.0 = the §20.14 recipe "
        "(+0.0055 play vs student acting, CE_Teacher_Design §20.13 add. 16)",
    )
    ap.add_argument("--node-telemetry", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run in the same out-dir: play exactly "
        "the game indices its flushed shards lack, continuing its manifest "
        "and shard numbering (game-index seeding makes each index's deal "
        "and committee-act draw independent of when it is played). Refuses "
        "a corpus from another checkpoint or generation config; without a "
        "manifest it starts fresh",
    )
    ap.add_argument(
        "--routed-encoder",
        nargs="?",
        const="mps",
        default=None,
        help="route >=16-row encodes to a compiled shadow (§16.6 throughput)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    init_args = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "collect_oracle": args.collect_oracle,
        "iters": args.iters,
        "iters_schedule": args.iters_schedule,
        "replicates": args.replicates,
        "d_rollout": args.d_rollout,
        "shrink_nu": args.shrink_nu,
        "shrink_s2_global": args.shrink_s2_global,
        "p_base": args.p_base,
        "boost_lead": args.boost_lead,
        "boost_cs": args.boost_cs,
        "p_min": args.p_min,
        "p_max": args.p_max,
        "routed_encoder": args.routed_encoder,
    }

    import torch

    manifest = {
        "row_schema": ROW_SCHEMA_VERSION,
        "ckpt": args.ckpt,
        "ckpt_sha256_16": _file_sha256(args.ckpt),
        "git_rev": _git_rev(),
        "config": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "games": 0,
        "kept_games": 0,
        "episodes": 0,
        "committee_act_games": 0,
        "committee_acted_nodes": 0,
        "classes": {},
        "gap_percentiles": None,
        "shards": [],
    }
    all_gaps: list[float] = []
    class_totals: dict[str, dict] = defaultdict(
        lambda: {"nodes": 0, "searched": 0, "resolved": 0, "unresolved": 0, "failed": 0}
    )
    shard_idx = 0
    kept = 0
    done_games: set[int] = set()
    prior_path = os.path.join(args.out_dir, "manifest.json")
    if args.resume and os.path.exists(prior_path):
        # Resume: continue the prior invocation's manifest, totals and shard
        # numbering. Only whole flushed shards survive a kill; games past
        # the last flush are replayed. A shard written after the manifest's
        # last update is overwritten in place (same index).
        with open(prior_path) as f:
            prior = json.load(f)
        check_resumable(prior, manifest)
        done_games = flushed_games(args.out_dir, prior)
        for key in (
            "games",
            "kept_games",
            "episodes",
            "committee_act_games",
            "committee_acted_nodes",
        ):
            manifest[key] = prior.get(key, 0)
        manifest["shards"] = prior["shards"]
        manifest["resumes"] = prior.get("resumes", []) + [
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "games_done": len(done_games),
                "git_rev": manifest["git_rev"],
            }
        ]
        for cls, c in prior.get("classes", {}).items():
            class_totals[cls].update(c)
        shard_idx = len(prior["shards"])
        kept = prior["kept_games"]
        if prior.get("git_rev") != manifest["git_rev"]:
            print(
                f"[resume] WARNING: corpus begun at git {prior.get('git_rev')}, "
                f"resuming at {manifest['git_rev']}",
                flush=True,
            )

    # Deterministic task schedule: modes alternate by game index; the
    # committee-act draw is seeded by game index so a resumed run makes the
    # same draw for the same game.
    tasks = [
        (
            g,
            PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD,
            random.Random((args.seed << 21) ^ g).random() < args.committee_act_frac,
        )
        for g in range(args.games)
        if g not in done_games
    ]
    if done_games:
        print(
            f"[resume] {len(done_games)} games in {shard_idx} flushed shards; "
            f"playing the {len(tasks)} missing",
            flush=True,
        )

    if args.node_telemetry and done_games and os.path.exists(args.node_telemetry):
        # Keep only the telemetry of flushed games (the rest belong to games
        # whose episodes died in the unflushed buffer, and are replayed),
        # then append.
        with open(args.node_telemetry) as f:
            keep_lines = [
                line for line in f if json.loads(line).get("game") in done_games
            ]
        with open(args.node_telemetry, "w") as f:
            f.writelines(keep_lines)
        all_gaps.extend(
            row["gap"]
            for row in map(json.loads, keep_lines)
            if row.get("w") is not None and row["w"] > 0.0 and row.get("gap")
        )
        telemetry_f = open(args.node_telemetry, "a", buffering=1 << 20)
    else:
        telemetry_f = (
            open(args.node_telemetry, "w", buffering=1 << 20)
            if args.node_telemetry
            else None
        )
    shard_episodes: list = []
    shard_meta: list = []
    done = 0
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_episodes, shard_meta
        if not shard_episodes:
            return
        path = os.path.join(args.out_dir, f"corpus_{shard_idx:04d}.pt")
        torch.save({"episodes": shard_episodes, "games": shard_meta}, path + ".tmp")
        os.replace(path + ".tmp", path)
        manifest["shards"].append(
            {
                "path": os.path.basename(path),
                "episodes": len(shard_episodes),
                "games": sorted(g["game"] for g in shard_meta),
            }
        )
        shard_idx += 1
        shard_episodes = []
        shard_meta = []
        if telemetry_f is not None:
            # Telemetry durability is shard-granular: resume drops the rows
            # of unflushed games, so flushing between shards buys nothing and costs
            # one fs event per game (fseventsd fan-out to every watcher).
            telemetry_f.flush()

    def write_manifest():
        for cls, c in class_totals.items():
            manifest["classes"][cls] = dict(c)
        if all_gaps:
            qs = np.percentile(all_gaps, [10, 25, 50, 75, 90, 99])
            manifest["gap_percentiles"] = {
                p: float(v) for p, v in zip([10, 25, 50, 75, 90, 99], qs)
            }
        with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=args.workers, initializer=worker_init, initargs=(init_args,)
    ) as pool:
        for res in pool.imap_unordered(play_corpus_game, tasks, chunksize=1):
            done += 1
            manifest["games"] += 1
            kept += 1
            manifest["kept_games"] = kept
            manifest["episodes"] += len(res["episodes"])
            manifest["committee_act_games"] += int(res["committee_act"])
            manifest["committee_acted_nodes"] += int(res["committee_acted"])
            for cls, c in res["counts"].items():
                tot = class_totals[cls]
                for k in tot:
                    tot[k] += c[k]
            all_gaps.extend(res["gaps"])
            if telemetry_f is not None:
                for row in res["telemetry"]:
                    telemetry_f.write(json.dumps(row) + "\n")
            shard_episodes.extend(res["episodes"])
            shard_meta.append(
                {
                    "game": res["game"],
                    "mode": res["mode"],
                    "is_leaster": res["is_leaster"],
                    "alone_called": res["alone_called"],
                }
            )
            if len(shard_meta) >= args.shard_games:
                flush_shard()
                write_manifest()
            if done % 25 == 0 or done == len(tasks):
                searched = sum(c["searched"] for c in class_totals.values())
                resolved = sum(c["resolved"] for c in class_totals.values())
                rate = done / max(time.time() - t_start, 1e-9)
                print(
                    f"[{done}/{len(tasks)} games, {rate:.2f} g/s] "
                    f"searched {searched} resolved {resolved} "
                    f"unresolved {sum(c['unresolved'] for c in class_totals.values())} "
                    f"failed {sum(c['failed'] for c in class_totals.values())}",
                    flush=True,
                )
    flush_shard()
    write_manifest()
    if telemetry_f is not None:
        telemetry_f.close()
    print(
        f"DONE: {kept} games kept, {manifest['episodes']} episodes, "
        f"{shard_idx} shards -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
