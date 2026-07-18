#!/usr/bin/env python3
"""ExIt validation harness (one-off, NOT committed).

The model-evaluable half of the validation protocol, defined up front so a
from-scratch ExIt run can be compared against the frozen PPO baseline. Computes,
for a given model (deterministic-greedy unless --stochastic):

  1. Bidding-health bands (held WITHOUT the epsilon-floor controllers): PICK rate,
     ALONE rate, leaster rate, per-position pick rate. Collapse looks like
     PICK->~0/~1 or ALONE/leaster blowing up.
  2. Trick-0 defender trump-lead RATE — the original partial-obs leak: of trick-0
     defender lead decisions that HAVE both a trump and a fail option, how often
     the policy leads trump (lower = leak more corrected). Reported greedy and as
     conditional trump mass.
  3. Head-to-head vs a baseline field: challenger in a rotating seat vs 4 baseline
     opponents, mean final score (points) over rotations. Baseline-vs-baseline ~0
     is the harness sanity check.

The TRAINING-TIME diagnostics (teacher_kl, ESS-abort fraction, pg_masked_fraction,
pi' entropy) are logged in update_stats['distill'] during the run, and the PG-mask
vs additive-form A/B is a two-config training experiment — neither is a static
model eval, so they are tracked during training, not here.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from sheepshead.agent.ppo import load_agent
from sheepshead import ACTIONS, TRUMP, Game
from sheepshead.training.training_utils import get_partner_selection_mode


def load(model):
    a = load_agent(model)
    return a


# ---------------------------------------------------------------------------
# 1 + 2: self-play bidding health + trick-0 defender trump-lead rate
# ---------------------------------------------------------------------------
def selfplay_metrics(agent, n_games, seed, deterministic):
    random.seed(seed)
    picks = passes = 0
    alone = partner_decisions = 0
    leasters = 0
    pos_pick = np.zeros(6)
    pos_seen = np.zeros(6)
    t0_def_leads = 0  # trick-0 defender leads with both trump & fail
    t0_def_trump = 0  # ...that led trump (greedy)
    t0_trump_mass = []  # conditional trump prob mass at those nodes

    for g in range(n_games):
        game = Game(partner_selection_mode=get_partner_selection_mode(g))
        agent.reset_recurrent_state()
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    names = [ACTIONS[a - 1] for a in valid]
                    # Bidding-health counters.
                    if "PICK" in names or "PASS" in names:
                        pos_seen[player.position] += 1
                    # Trick-0 defender lead detection (before acting).
                    is_lead = (
                        game.play_started
                        and not game.is_leaster
                        and game.cards_played == 0
                        and game.leader == player.position
                    )
                    is_def = not (
                        player.is_picker
                        or player.is_partner
                        or player.is_secret_partner
                    )
                    has_trump = any(c in TRUMP for c in player.hand)
                    has_fail = any(c not in TRUMP for c in player.hand)
                    t0_node = (
                        is_lead
                        and game.current_trick == 0
                        and is_def
                        and has_trump
                        and has_fail
                    )
                    if t0_node:
                        # get_action_probs_with_logits advances the recurrent
                        # memory (it stores memory_out), and act() below encodes
                        # the same state again — so without a snapshot/restore
                        # the action at this node is taken from a double-encoded
                        # memory and the greedy trump-lead RATE is measured
                        # under perturbation (probe-vs-eval discrepancy found
                        # 2026-06-10: 73% vs 7% on identical weights). The mass
                        # metric was always clean (computed on the first,
                        # correctly-advanced encode). Historical greedy-rate
                        # numbers from this harness (baseline 4.8%, run-2
                        # 48.6%) carry the perturbation; mass and h2h do not.
                        saved_mem = agent.snapshot_player_memories()
                        probs, _ = agent.get_action_probs_with_logits(
                            player.get_state_dict(), valid, player_id=player.position
                        )
                        agent.restore_player_memories(saved_mem)
                        p = probs[0].detach().cpu().numpy()
                        tmass = sum(
                            float(p[a - 1])
                            for a in valid
                            if ACTIONS[a - 1].startswith("PLAY ")
                            and ACTIONS[a - 1][5:] in TRUMP
                        )
                        t0_trump_mass.append(tmass)
                        t0_def_leads += 1

                    a, _, _ = agent.act(
                        player.get_state_dict(),
                        valid,
                        player.position,
                        deterministic=deterministic,
                    )
                    name = ACTIONS[a - 1]
                    if name == "PICK":
                        picks += 1
                        pos_pick[player.position] += 1
                    elif name == "PASS":
                        passes += 1
                    elif name == "ALONE":
                        alone += 1
                        partner_decisions += 1
                    elif name == "JD PARTNER" or name.startswith("CALL "):
                        partner_decisions += 1
                    if t0_node and name.startswith("PLAY ") and name[5:] in TRUMP:
                        t0_def_trump += 1
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
        if game.is_leaster:
            leasters += 1

    pick_rate = picks / max(picks + passes, 1)
    return {
        "games": n_games,
        "pick_rate": pick_rate,
        "alone_rate": alone / max(partner_decisions, 1),
        "leaster_rate": leasters / n_games,
        "pos_pick_rate": [
            pos_pick[p] / pos_seen[p] if pos_seen[p] else float("nan")
            for p in range(1, 6)
        ],
        "t0_def_leads": t0_def_leads,
        "t0_trump_lead_rate": t0_def_trump / max(t0_def_leads, 1),
        "t0_trump_mass_mean": float(np.mean(t0_trump_mass))
        if t0_trump_mass
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# 3: head-to-head vs a baseline field
# ---------------------------------------------------------------------------
def h2h(challenger, baseline, n_games, seed):
    """Bare mixed-agent play (no shaping/profiling overhead): challenger holds a
    rotating seat vs the baseline in the other four; report the challenger's mean
    final score. Each agent keeps its own per-seat recurrent memory for the seats
    it controls."""
    random.seed(seed)
    scores = []
    for g in range(n_games):
        mode = get_partner_selection_mode(g)
        pos = (g % 5) + 1
        game = Game(partner_selection_mode=mode)
        challenger.reset_recurrent_state()
        baseline.reset_recurrent_state()
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    ag = challenger if player.position == pos else baseline
                    a, _, _ = ag.act(
                        player.get_state_dict(),
                        valid,
                        player.position,
                        deterministic=False,
                    )
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            ctrl = challenger if seat.position == pos else baseline
                            ctrl.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
        scores.append(game.players[pos - 1].get_score())
    arr = np.array(scores)
    return {
        "n": n_games,
        "mean": float(arr.mean()),
        "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="final_pfsp_swish_ppo.pt",
        help="challenger model (the ExIt agent under test)",
    )
    ap.add_argument(
        "-b",
        "--baseline",
        default="final_pfsp_swish_ppo.pt",
        help="baseline field for head-to-head",
    )
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--h2h-games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of greedy (bidding-health under the sampling policy)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading challenger {args.model} ...")
    agent = load(args.model)
    m = selfplay_metrics(
        agent, args.games, args.seed, deterministic=not args.stochastic
    )

    print("\n" + "=" * 64)
    print(
        f"BIDDING HEALTH  ({m['games']} self-play games, "
        f"{'stochastic' if args.stochastic else 'greedy'})"
    )
    print("=" * 64)
    print(f"  PICK rate (of pick decisions): {m['pick_rate'] * 100:5.1f}%")
    print(f"  ALONE rate (of partner decs):  {m['alone_rate'] * 100:5.1f}%")
    print(f"  Leaster rate (of games):       {m['leaster_rate'] * 100:5.1f}%")
    print(
        "  Per-position PICK rate:        "
        + "  ".join(f"P{p}={m['pos_pick_rate'][p - 1] * 100:.0f}%" for p in range(1, 6))
    )
    print("\n--- Trick-0 defender trump-lead (the partial-obs leak) ---")
    print(f"  qualifying nodes (trump+fail in hand): {m['t0_def_leads']}")
    print(
        f"  trump-lead rate (greedy):    {m['t0_trump_lead_rate'] * 100:5.1f}%  (lower = leak more corrected)"
    )
    print(f"  conditional trump prob mass: {m['t0_trump_mass_mean'] * 100:5.1f}%")

    if args.baseline:
        print("\n" + "=" * 64)
        print(
            f"HEAD-TO-HEAD: challenger vs baseline field ({args.h2h_games} games, rotating seat)"
        )
        print("=" * 64)
        base = load(args.baseline)
        r = h2h(agent, base, args.h2h_games, args.seed + 1)
        print(
            f"  challenger mean final score: {r['mean']:+.3f} +/- {r['se']:.3f} points/game"
        )
        print(
            "  (baseline-vs-baseline should be ~0; >0 means the challenger beats the field)"
        )


if __name__ == "__main__":
    main()
