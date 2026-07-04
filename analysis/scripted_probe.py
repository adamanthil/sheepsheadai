#!/usr/bin/env python3
"""Paired scripted-agent probe: hero vs ScriptedAgent on duplicate deals.

Places a checkpoint on the static, lineage-free scale defined by the
conventions ScriptedAgent (League_Run_Review_202607.md §5): the hero and the
ScriptedAgent each play the same seat on the same deal against an
all-ScriptedAgent field, and the paired score delta (hero − scripted) is the
edge. Positive = hero beats the scripted yardstick.

What this is for (and not for):
  * Sanity floor / absolute placement early in a run — the scripted agent
    never drifts, so probe values are comparable across runs and years.
  * NOT a top-of-ladder strength yardstick: the 13.65M league main cleared
    it by only +0.34±0.21 while ranking far above it vs PANEL-A. Strength
    comparisons between strong checkpoints belong in analysis/rigorous_eval.py.

Design notes:
  * paired_edge() alternates partner modes deal-by-deal (even deals = called
    ace, odd = JD) and rotates the probe seat, so one call covers both modes.
  * A fixed --seed fixes the deal set: probes with the same seed are paired
    with each other, so checkpoint-to-checkpoint deltas are CRN-paired too.
  * Default seed 31 matches the placement numbers recorded in
    League_Run_Review_202607.md (selfplay-100k −0.63±0.24 vs scripted;
    league-13.65M +0.34±0.21; 150 deals).
  * --self-check probes the ScriptedAgent against itself: the edge must be
    exactly 0 with deviating_frac 0, validating the pairing machinery.

Usage:
  PYTHONPATH=. python analysis/scripted_probe.py --ckpt final_pfsp_swish_ppo.pt
  PYTHONPATH=. python analysis/scripted_probe.py --ckpt a.pt b.pt --deals 500
  PYTHONPATH=. python analysis/scripted_probe.py --self-check
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

from scripted_agent import ScriptedAgent
from training_utils import paired_edge

PROBE_SEED = 31  # frozen: pairs new probes with the recorded placement numbers


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired scripted-agent probe")
    ap.add_argument("--ckpt", nargs="+", default=[], help="PPO checkpoint(s) to probe")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="probe ScriptedAgent vs itself (edge must be exactly 0)",
    )
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if not args.ckpt and not args.self_check:
        ap.error("provide --ckpt and/or --self-check")

    scripted = ScriptedAgent()
    heroes: list[tuple[str, object]] = []
    if args.self_check:
        heroes.append(("ScriptedAgent (self-check)", ScriptedAgent()))
    for path in args.ckpt:
        from ppo import load_agent

        agent = load_agent(path)
        heroes.append((path, agent))

    results = {"probe_seed": args.seed, "deals": args.deals, "probes": []}
    for label, hero in heroes:
        t0 = time.time()
        r = paired_edge(
            hero,
            scripted,
            scripted,
            args.deals,
            seed=args.seed,
            log_every=args.log_every,
        )
        r["hero"] = label
        results["probes"].append(r)
        print(
            f"[{label}] edge {r['edge']:+.3f} ± {r['se']:.3f} score/deal | "
            f"win {r['win_frac']:.3f} | deviating {r['deviating_frac']:.3f} | "
            f"n={r['n_deals']} | {time.time() - t0:.0f}s",
            flush=True,
        )
        if label.endswith("(self-check)") and (
            r["edge"] != 0.0 or r["deviating_frac"] != 0.0
        ):
            print("SELF-CHECK FAILED: scripted-vs-scripted edge must be exactly 0")
            return 1

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
