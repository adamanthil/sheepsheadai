#!/usr/bin/env python3
"""Teacher value-add probe: does decision-time ISMCTS beat the raw policy in EV?

The missing keystone measurement of the ExIt program (Stage B left it at a
4-game smoke): distillation can only improve the policy if the search target is
*better* than the prior, and every distill experiment so far has measured bias
(falsifiers) or stability (arms) — never decision quality in points.

Method: paired duplicate deals. Each deal (Game(seed=k)) is played twice with
an identical greedy field of the baseline model in all five seats:

  RAW arm:    the probe seat plays its raw greedy policy everywhere.
  SEARCH arm: the probe seat runs the production ISMCTS teacher at every PLAY
              decision and plays argmax(pi') (visit-count argmax; tau-invariant);
              bidding stays raw greedy in both arms (production f_bid=0).
              ESS-aborted searches fall back to the raw action (as in training,
              where aborted searches don't distill).

Because every agent is deterministic-greedy, the two arms are identical until
the first searched decision that DISAGREES with the raw argmax, so the paired
delta isolates the EV effect of exactly those deviations (deals with no
deviation contribute delta == 0, a huge variance reduction). Rollout depth
follows the production schedule: full rollout for tricks <= t_full(=1), d_short
beyond.

Interpretation: paired delta > 0 -> search adds EV at decision time and
distillation has a foundation; ~0 -> at this budget ExIt has no improvement
operator (search stays a deployment/audit tool). Re-run after the exploiter
population exists (same harness, sharper world) to get the derivative: does
search value-add grow when rollout fields contain real punishment?

Usage: PYTHONPATH=. .venv/bin/python validation/teacher_value_add_probe.py \
           [-m final_pfsp_swish_ppo.pt] [--deals 300] [--iters-play 96]
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from ismcts import ISMCTSConfig, ISMCTSTeacher
from ppo import PPOAgent
from sheepshead import ACTIONS, Game
from training_utils import get_partner_selection_mode

T_FULL = 1  # production rollout-depth schedule (config.SearchConfig)
D_SHORT = 2


def _is_private(valid) -> bool:
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def _load(model: str) -> PPOAgent:
    a = PPOAgent(len(ACTIONS), activation="swish")
    a.load(model, load_optimizers=False)
    return a


def _play_deal(
    deal_seed: int,
    mode,
    seat: int,
    challenger: PPOAgent,
    field: PPOAgent,
    teacher: ISMCTSTeacher | None,
    det_rng: random.Random,
) -> tuple[float, int, int, int]:
    """Play one deal greedily; if ``teacher`` is given, the probe seat plays
    argmax(pi') at its PLAY decisions. Returns (probe-seat score, searched
    decisions, deviations from raw argmax, ESS aborts)."""
    game = Game(partner_selection_mode=mode, seed=deal_seed)
    challenger.reset_recurrent_state()
    field.reset_recurrent_state()
    forced_public = []
    n_searched = n_dev = n_abort = 0

    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                is_probe = player.position == seat
                ag = challenger if is_probe else field
                searched_action = None
                if (
                    teacher is not None
                    and is_probe
                    and game.play_started
                    and all(ACTIONS[a - 1].startswith("PLAY ") for a in valid)
                ):
                    dr = (
                        (6 - game.current_trick)
                        if game.current_trick <= T_FULL
                        else D_SHORT
                    )
                    res = teacher.search(
                        game, seat, list(forced_public), det_rng, d_rollout=dr
                    )
                    n_searched += 1
                    if res["ok"]:
                        pi = res["pi"]
                        searched_action = max(valid, key=lambda a: pi[a - 1])
                    else:
                        n_abort += 1
                # act() advances the recurrent memory for this state either
                # way; the searched action (if any) overrides its choice.
                raw_action, _, _ = ag.act(
                    player.get_state_dict(),
                    valid,
                    player.position,
                    deterministic=True,
                )
                a = raw_action
                if searched_action is not None:
                    if searched_action != raw_action:
                        n_dev += 1
                    a = searched_action
                if not _is_private(valid):
                    forced_public.append((player.position, a))
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for p in game.players:
                        ctrl = challenger if p.position == seat else field
                        ctrl.observe(
                            p.get_last_trick_state_dict(), player_id=p.position
                        )

    return float(game.players[seat - 1].get_score()), n_searched, n_dev, n_abort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="final_pfsp_swish_ppo.pt")
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--iters-play", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model} (challenger + field) ...")
    challenger = _load(args.model)
    field = _load(args.model)

    cfg = ISMCTSConfig()
    cfg.iters = dict(cfg.iters, play=args.iters_play)
    teacher = ISMCTSTeacher(challenger, cfg)
    det_rng = random.Random(args.seed + 1)

    deltas, dev_deltas = [], []
    tot_searched = tot_dev = tot_abort = 0
    t0 = time.time()

    for d in range(args.deals):
        mode = get_partner_selection_mode(d)
        seat = (d % 5) + 1
        deal_seed = args.seed * 1_000_003 + d
        raw_score, _, _, _ = _play_deal(
            deal_seed, mode, seat, challenger, field, None, det_rng
        )
        s_score, ns, nd, na = _play_deal(
            deal_seed, mode, seat, challenger, field, teacher, det_rng
        )
        delta = s_score - raw_score
        deltas.append(delta)
        tot_searched += ns
        tot_dev += nd
        tot_abort += na
        if nd > 0:
            dev_deltas.append(delta)
        if (d + 1) % 25 == 0:
            arr = np.array(deltas)
            print(
                f"  {d + 1}/{args.deals} deals ({time.time() - t0:.0f}s)  "
                f"running delta {arr.mean():+.3f} +/- "
                f"{arr.std(ddof=1) / np.sqrt(len(arr)):.3f}",
                flush=True,
            )

    arr = np.array(deltas)
    n = len(arr)
    print()
    print("=" * 72)
    print(
        f"TEACHER VALUE-ADD PROBE  ({n} paired deals, iters_play={args.iters_play}, "
        f"{time.time() - t0:.0f}s)"
    )
    print("=" * 72)
    print(
        f"  paired delta (search - raw):  {arr.mean():+.4f} +/- "
        f"{arr.std(ddof=1) / np.sqrt(n):.4f} points/deal"
    )
    print(
        f"  searched decisions:           {tot_searched} "
        f"({tot_searched / n:.1f}/deal), ESS-aborts {tot_abort} "
        f"({100 * tot_abort / max(tot_searched, 1):.1f}%)"
    )
    print(
        f"  deviations from raw argmax:   {tot_dev} "
        f"({100 * tot_dev / max(tot_searched, 1):.1f}% of searched decisions)"
    )
    if dev_deltas:
        dv = np.array(dev_deltas)
        print(
            f"  deals with >=1 deviation:     {len(dv)}/{n}  "
            f"(conditional delta {dv.mean():+.4f} +/- "
            f"{dv.std(ddof=1) / np.sqrt(len(dv)):.4f})"
        )
    print()
    print(
        "  Interpretation: delta > 0 => decision-time search beats the raw\n"
        "  policy and distillation has a foundation; ~0 => no improvement\n"
        "  operator at this budget (search stays a deployment/audit tool).\n"
        "  Field is greedy self-play; re-run with an exploiter-bearing field\n"
        "  to measure whether search value-add grows with real punishment."
    )


if __name__ == "__main__":
    main()
