#!/usr/bin/env python3
"""Per-node normalized policy-entropy probe (adaptive-entropy Phase 1).

Plays SAMPLED self-play (the on-policy state distribution training sees,
unlike the argmax ``greedy_health_probe``) with the agent in all five seats
and measures, at every decision node with >= 2 legal actions, the policy
entropy over the LEGAL action set normalized by its maximum:

    H_norm = -sum_a p(a) ln p(a) / ln(n_legal)   in [0, 1]

aggregated per policy head (pick / partner / bury / play). Forced moves
(n_legal == 1) are excluded — their H/H_max is 0/0 and they carry no policy
information.

This is the offline half of the Phase-1 instrumentation pair: the same
statistic is measured live at theta_old during PPO updates (see
``PPOAgent._update_minibatch``, ``stats["head_entropy_norm"]``). Sweeping it
over historical checkpoints (``entropy_backfill.py``) reconstructs the
entropy trajectory of a finished run, which serves two purposes for the
planned target-entropy controller:

* Starting targets, derived "bumplessly" from a run that worked — initialize
  the controller setpoint at the measured operating point so switch-on
  changes nothing (bumpless transfer; Astrom & Wittenmark, *Adaptive
  Control*, 2nd ed. 1995, ch. 9).
* The organic per-generation entropy decline under a fixed coefficient — the
  scale a plateau-triggered target step must exceed to be distinguishable
  from what training does anyway.

Controlling measured entropy against a target instead of hand-scheduling the
coefficient follows SAC's automatic temperature adjustment (Haarnoja et al.,
"Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905); the
discrete-action form (Christodoulou, arXiv:1910.07207) expresses targets as
a fraction of maximum entropy, which is why the statistic is normalized.
Sheepshead is imperfect-information, where equilibrium policies are
genuinely mixed and entropy regularization has convergence support (Sokota
et al., arXiv:2206.05825), so targets are meant to be annealed to a small
floor, never to zero.

Side-effect free: recurrent memories and the global ``random`` state are
snapshotted and restored; deals are re-seeded per game (CRN across
checkpoints — same deal panel for every swept policy).
"""

from __future__ import annotations

import math
import random
from typing import Dict

import numpy as np

from sheepshead import Game
from sheepshead.agent.ppo import SOFTBAND_HNORM
from sheepshead.training.training_utils import get_partner_selection_mode

PROBE_SEED = 20260728

HEADS = ("pick", "partner", "bury", "play")


def _head_of_action(agent, action_id: int) -> str:
    """Head owning a (1-based) action id, from the agent's own groups."""
    idx = action_id - 1
    for head in ("pick", "partner", "bury"):
        if idx in agent._head_index_sets[head]:
            return head
    return "play"


def _ensure_head_index_sets(agent) -> None:
    if not hasattr(agent, "_head_index_sets"):
        agent._head_index_sets = {
            head: set(agent.action_groups[head]) for head in ("pick", "partner", "bury")
        }


def _summary(vals: list) -> Dict:
    if not vals:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "softband": None,
            "rows": 0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        # Fraction of nodes above the shared soft-band threshold: the
        # decision-boundary-band gauge (mean hides band structure in a tail).
        "softband": float((arr > SOFTBAND_HNORM).mean()),
        "rows": int(arr.size),
    }


def probe_agent(agent, n_games: int = 200, seed: int = PROBE_SEED) -> Dict:
    """Sampled self-play entropy probe; returns per-head H_norm summaries.

    Deals are re-seeded per game from ``seed`` so the deal panel is common
    random numbers across checkpoints; action sampling uses a dedicated
    numpy generator so results are reproducible and the global RNG state is
    untouched (trajectories still diverge across policies after the first
    differing sample, which is the point — this measures each policy on its
    own on-policy distribution over a fixed deal set).
    """
    _ensure_head_index_sets(agent)
    rng_state = random.getstate()
    saved_mem = agent.snapshot_player_memories()
    sampler = np.random.default_rng(seed)
    per_head: Dict[str, list] = {h: [] for h in HEADS}
    try:
        for g in range(n_games):
            random.seed(seed + g)  # CRN deal panel across checkpoints
            game = Game(partner_selection_mode=get_partner_selection_mode(g))
            agent.reset_recurrent_state()
            while not game.is_done():
                for player in game.players:
                    valid = player.get_valid_action_ids()
                    while valid:
                        state = player.get_state_dict()
                        # One forward per node: yields the acting policy AND
                        # advances the recurrent memory exactly once (argmax
                        # over these probs == act(deterministic); sampling
                        # from them == act(stochastic)).
                        probs_t, _ = agent.get_action_probs_with_logits(
                            state, valid, player_id=player.position
                        )
                        ordered = sorted(valid)  # valid is a set
                        p_legal = (
                            probs_t[0][[x - 1 for x in ordered]]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        p_legal = p_legal / max(p_legal.sum(), 1e-12)
                        a = int(ordered[sampler.choice(len(ordered), p=p_legal)])
                        if len(valid) >= 2:
                            h = float(
                                -(p_legal * np.log(np.clip(p_legal, 1e-12, None))).sum()
                            )
                            per_head[_head_of_action(agent, a)].append(
                                h / math.log(len(valid))
                            )
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for seat in game.players:
                                agent.observe(
                                    seat.get_last_trick_state_dict(),
                                    player_id=seat.position,
                                )
    finally:
        agent.restore_player_memories(saved_mem)
        random.setstate(rng_state)

    result: Dict = {"games": n_games, "seed": seed, "heads": {}}
    for head in HEADS:
        result["heads"][head] = _summary(per_head[head])
    return result


__all__ = ["probe_agent", "PROBE_SEED", "HEADS"]

if __name__ == "__main__":
    import argparse

    from sheepshead.agent.ppo import load_agent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    args = ap.parse_args()
    res = probe_agent(load_agent(args.ckpt), n_games=args.games, seed=args.seed)
    for head in HEADS:
        s = res["heads"][head]
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "-"
        print(f"{head:>8}: H_norm mean {mean}  (rows {s['rows']})")
