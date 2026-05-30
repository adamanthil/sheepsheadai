#!/usr/bin/env python3
"""Diagnostic: is uniform determinization biased because it ignores the bidding
inference (the real picker self-selected a trump-rich hand)?

For trick-0 defender trump-lead states, compare the TRUE picker's trump strength
(its actual pre-bury 8 = current hand + bury) against the trump strength of the
picker's hand in uniform determinizations. If determinized pickers are
systematically weaker in trump, that explains why determinized trump-leads look
safer than they truly are (detD biased positive vs the oracle).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

import ppo
from ppo import PPOAgent
from sheepshead import ACTIONS, TRUMP, get_card_points
from gate0_determinizer import collect

DEV = ppo.device


def trump_count(cards):
    return sum(1 for c in cards if c in TRUMP)


def trump_strength(cards):
    # crude strength: #trump + fraction that are top trump (queens/jacks)
    top = sum(1 for c in cards if c[0] in ("Q", "J"))
    return trump_count(cards), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",
                    default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt")
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--max-games", type=int, default=30000)
    ap.add_argument("-K", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"Loading {args.model} ...")
    agent = PPOAgent(len(ACTIONS), activation="swish")
    agent.load(args.model, load_optimizers=False)

    print(f"Scanning for {args.states} trick-0 defender trump-lead states ...")
    states, scanned = collect(agent, args.max_games, args.states, args.seed)
    print(f"  reached {scanned} trick-0 leads; {len(states)} trump-pref states.")

    rng = random.Random(args.seed + 7)
    real_tc, real_top, det_tc, det_top = [], [], [], []
    for st in states:
        g = st["game"]
        pk = g.picker
        # true picker pre-bury 8 = current hand (6) + bury (2)
        true8 = list(g.players[pk - 1].hand) + list(g.bury)
        tc, tp = trump_strength(true8)
        real_tc.append(tc); real_top.append(tp)
        # determinized picker 8 = initial_hands[picker] + blind
        for _ in range(args.K):
            deal = g.sample_trick0_determinization(1, rng)
            eight = deal["initial_hands"][pk] + deal["blind"]
            dc, dp = trump_strength(eight)
            det_tc.append(dc); det_top.append(dp)

    real_tc = np.array(real_tc); det_tc = np.array(det_tc)
    real_top = np.array(real_top); det_top = np.array(det_top)
    print("\n" + "=" * 64)
    print("PICKER TRUMP STRENGTH: true (self-selected) vs uniform determinized")
    print("=" * 64)
    print(f"  states={len(states)}, determinizations each={args.K}")
    print(f"  trump in picker's 8 cards:")
    print(f"    TRUE picker        mean={real_tc.mean():.2f}  (SE {real_tc.std(ddof=1)/np.sqrt(len(real_tc)):.2f})")
    print(f"    DETERMINIZED picker mean={det_tc.mean():.2f}  (SE {det_tc.std(ddof=1)/np.sqrt(len(det_tc)):.2f})")
    print(f"    gap (true - det)   = {real_tc.mean() - det_tc.mean():+.2f} trump")
    print(f"  top trump (Q/J) in picker's 8:")
    print(f"    TRUE        mean={real_top.mean():.2f}")
    print(f"    DETERMINIZED mean={det_top.mean():.2f}")
    print(f"    gap         = {real_top.mean() - det_top.mean():+.2f}")
    print("\nIf TRUE >> DETERMINIZED, uniform determinization understates picker")
    print("strength -> defender trump leads look too safe -> detD biased positive.")


if __name__ == "__main__":
    main()
