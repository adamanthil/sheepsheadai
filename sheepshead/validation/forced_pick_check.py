#!/usr/bin/env python3
"""Diagnostic A — forced-pick EV vs a fixed field (run-1 post-mortem).

Localizes the ExIt PASS->leaster collapse: is it ONLY the bidding (PICK/PASS) head
that broke, or did the picker's *play* degrade too?

The bidding head is the thing that collapsed, so we take it out of the loop: the
agent under test is FORCED to become the picker every deal (it is forced to PICK and
all other seats are forced to PASS during the pick phase), then plays partner/bury/
play with its own policy against a fixed opponent field. We measure its average final
score as picker. Leasters can't occur (someone always picks), so this is pure
picker-play EV with zero bidding-policy influence.

The signal is the DELTA between the collapsed challenger and the frozen 30M reference
run through the *identical* protocol (same deals, same field, same seed -> paired):

  * delta ~ 0 (within a couple SE):  play is intact -> ONLY the bidding head collapsed.
    Fix can target the pick decision (KL-anchor / f_pick / floor) and leave play alone.
  * delta strongly negative:         the picker's play degraded too -> broad collapse;
    the distillation is hurting play, not just bidding.

Absolute EV is depressed relative to the run's natural picker_avg because we force
picks on ALL hands (including junk). Use --min-hand-strength to restrict to hands a
sane bidder would actually pick (the shaped baseline used ~7); the cross-agent delta
is the diagnostic either way.

Run from the repo root:
  PYTHONPATH=. .venv/bin/python validation/forced_pick_check.py \
      -m runs/pfsp_exit_warmstart/checkpoints/<collapsed>.pt
  # restrict to hands a sane bidder would pick:
  PYTHONPATH=. .venv/bin/python validation/forced_pick_check.py -m <ckpt> \
      -f final_pfsp_swish_ppo.pt --min-hand-strength 7
"""

from __future__ import annotations

import argparse

import numpy as np

from sheepshead.agent.ppo import load_agent
from sheepshead import ACTIONS, Game
from sheepshead.training.training_utils import (
    estimate_hand_strength_score,
    get_partner_selection_mode,
    set_all_seeds,
)

PICK_ID = ACTIONS.index("PICK") + 1
PASS_ID = ACTIONS.index("PASS") + 1


def load(model):
    a = load_agent(model)
    return a


def forced_pick_eval(agent, make_field, n_games, seed, min_strength, deterministic):
    """Force ``agent`` to be the picker every deal; play out vs the field; return
    score stats. ``make_field(mode, pos) -> (ctrl, reset_list)`` where ``ctrl(seat)``
    is the PPOAgent controlling a non-picker seat and ``reset_list`` is the agents to
    reset per game."""
    scores: list[float] = []
    strengths: list[int] = []

    for g in range(n_games):
        # Per-game seed so the deal + pool sampling are reproducible and independent
        # of how much RNG play consumes -> challenger and reference see identical
        # deals/fields (paired delta). Set BEFORE the deal and field construction.
        gseed = (seed * 1_000_003 + g) % (2**32)
        set_all_seeds(gseed)

        mode = get_partner_selection_mode(g)
        pos = (g % 5) + 1
        # Game owns its own RNG (sheepshead.py:259) — must pass seed= to fix the deal.
        game = Game(partner_selection_mode=mode, seed=gseed)
        agent.reset_recurrent_state()
        ctrl, reset_list = make_field(mode, pos)
        for fa in reset_list:
            fa.reset_recurrent_state()

        agent_strength: int | None = None
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    names = [ACTIONS[a - 1] for a in valid]
                    is_pick_decision = "PICK" in names and "PASS" in names
                    is_agent = player.position == pos

                    if is_pick_decision:
                        # Force the agent into the picker role; everyone else passes.
                        if is_agent:
                            if agent_strength is None:
                                agent_strength = estimate_hand_strength_score(
                                    player.hand
                                )
                            a = PICK_ID
                        else:
                            a = PASS_ID
                    else:
                        controller = agent if is_agent else ctrl(player.position)
                        a, _, _ = controller.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=deterministic,
                        )

                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            c = agent if seat.position == pos else ctrl(seat.position)
                            c.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )

        # By construction the agent is always the picker (never a leaster).
        if game.picker != pos:
            continue
        if min_strength is not None and (
            agent_strength is None or agent_strength < min_strength
        ):
            continue
        scores.append(game.players[pos - 1].get_score())
        strengths.append(agent_strength if agent_strength is not None else -1)

    arr = np.array(scores, dtype=float)
    n = len(arr)
    return {
        "n": n,
        "mean": float(arr.mean()) if n else float("nan"),
        "se": float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "win_rate": float((arr > 0).mean()) if n else float("nan"),
        "mean_strength": float(np.mean(strengths)) if n else float("nan"),
    }


def make_single_field(field_model):
    """One shared baseline agent in all four non-picker seats (deterministic field,
    reproducible). Keeps per-seat recurrent memory via player_id."""
    base = load(field_model)

    def make_field(mode, pos):
        return (lambda seat: base), [base]

    return make_field, f"single model: {field_model}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-m",
        "--model",
        required=True,
        nargs="+",
        help="challenger ckpt(s); pass several to sweep a checkpoint trajectory",
    )
    ap.add_argument(
        "-r",
        "--reference",
        default="final_pfsp_swish_ppo.pt",
        help="control agent run through the same protocol (the pristine 30M seed)",
    )
    ap.add_argument(
        "-f",
        "--field",
        default="final_pfsp_swish_ppo.pt",
        help="opponent field model (4 non-picker seats)",
    )
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--min-hand-strength",
        type=int,
        default=None,
        help="only count deals whose pick-time hand strength >= this (shaped used ~7)",
    )
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="sample play actions instead of greedy (greedy default = low-variance paired delta)",
    )
    args = ap.parse_args()

    deterministic = not args.stochastic
    make_field, field_desc = make_single_field(args.field)

    print("=" * 70)
    print("DIAGNOSTIC A — forced-pick EV (agent always becomes picker)")
    print(f"  field        : {field_desc}")
    print(
        f"  games        : {args.games}  | play: "
        f"{'stochastic' if args.stochastic else 'greedy'}"
        f"{f' | min hand strength >= {args.min_hand_strength}' if args.min_hand_strength is not None else ''}"
    )
    print("=" * 70)

    # Reference computed once; every challenger is scored on the SAME paired deals
    # (forced_pick_eval reseeds per game from --seed), so a multi-model sweep is
    # apples-to-apples across checkpoints.
    print(f"\nreference  ({args.reference}) ...")
    ref = load(args.reference)
    r_ref = forced_pick_eval(
        ref, make_field, args.games, args.seed, args.min_hand_strength, deterministic
    )

    def line(tag, r):
        return (
            f"  {tag:<34} mean {r['mean']:+6.3f} +/- {r['se']:5.3f}  "
            f"win {r['win_rate'] * 100:4.1f}%  (n={r['n']}, avg hand str {r['mean_strength']:.1f})"
        )

    print("\n" + "-" * 78)
    print(line(f"reference  [{args.reference.split('/')[-1]}]", r_ref))
    print("-" * 78)

    sweep = len(args.model) > 1
    for model in args.model:
        print(f"challenger ({model}) ...")
        cha = load(model)
        r_cha = forced_pick_eval(
            cha,
            make_field,
            args.games,
            args.seed,
            args.min_hand_strength,
            deterministic,
        )
        delta = r_cha["mean"] - r_ref["mean"]
        se_delta = (r_cha["se"] ** 2 + r_ref["se"] ** 2) ** 0.5
        sigma = delta / se_delta if se_delta else float("nan")
        print(line(f"[{model.split('/')[-1]}]", r_cha))
        print(
            f"  {'delta vs reference':<34} {delta:+6.3f} +/- {se_delta:5.3f}  ({sigma:+.1f} SE)"
        )
        if not sweep:
            if abs(delta) <= 2 * se_delta:
                print(
                    "  => delta ~ 0: picker PLAY is intact. The collapse is confined to\n"
                    "     the bidding head — fix the PICK/PASS decision, leave play alone."
                )
            else:
                print(
                    "  => delta strongly negative: the picker's PLAY degraded too. The\n"
                    "     distillation is hurting play — broad collapse, not just bidding."
                )
        print("-" * 78)


if __name__ == "__main__":
    main()
