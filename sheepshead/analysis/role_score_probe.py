#!/usr/bin/env python3
"""Role-conditional score decomposition across a checkpoint ladder.

Question this answers (2026-07-19, stage-1 follow-up): the v2 league agent's
overall edge vs its warm start is ~flat, but its greedy pick rate fell
substantially — so if the total held, per-role performance must have shifted
(more defender hands, so defense must carry more). This probe seats each
hero in all 5 seats per CRN deal against a CONSTANT field (all four other
seats = one fixed checkpoint), records every (deal, seat) game's hero role
and score, and writes the long table so the total edge can be decomposed
into role-mix shift + within-role deltas.

Interpretation caveat (recorded here so nobody trips on it later): role is
ENDOGENOUS — a hero that picks less picks a different, presumably stronger,
subset of hands, so within-role means carry selection effects. The clean
paired comparison is per (deal, seat) cells where two heroes took the SAME
role; the analysis step does that pairing from the long table.

Usage:
  PYTHONPATH=. python -m sheepshead.analysis.role_score_probe \
    --hero sp400k=runs/.../warmstart.pt --hero league2M=runs/.../ckpt_2M.pt \
    --field runs/.../warmstart.pt --deals 400 --out-csv roles.csv
"""

from __future__ import annotations


import argparse
import csv
import time
from pathlib import Path

from sheepshead.analysis.rigorous_eval import ModelRegistry, play_hand
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

PROBE_SEED = 20260719
MODE_NAMES = {PARTNER_BY_CALLED_ACE: "called", PARTNER_BY_JD: "jd"}


def hero_role(res, seat: int) -> str:
    if res.is_leaster:
        return "leaster"
    if seat == res.picker:
        return "picker"
    if seat == res.partner:
        return "partner"
    return "defender"


def main() -> int:
    ap = argparse.ArgumentParser(description="Role-conditional score probe")
    ap.add_argument(
        "--hero",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="hero checkpoint (repeatable; same deals for every hero)",
    )
    ap.add_argument("--field", required=True, help="checkpoint filling all non-hero seats")
    ap.add_argument("--deals", type=int, default=400, help="deals per partner mode")
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    registry = ModelRegistry()
    heroes = []
    for spec in args.hero:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--hero needs LABEL=PATH, got {spec!r}")
        heroes.append((label, registry.get(Path(path))))
    field_model = registry.get(Path(args.field))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hero", "mode", "deal", "seat", "role", "score"])
        for label, hero in heroes:
            t0 = time.time()
            tallies: dict = {}
            for mode in (PARTNER_BY_CALLED_ACE, PARTNER_BY_JD):
                for d in range(args.deals):
                    deal_seed = args.seed * 1_000_003 + d
                    for seat in range(1, 6):
                        seat_to_model = {s: field_model for s in range(1, 6)}
                        seat_to_model[seat] = hero
                        res = play_hand(seat_to_model, mode, deal_seed)
                        role = hero_role(res, seat)
                        score = res.scores[seat - 1]
                        w.writerow(
                            [label, MODE_NAMES[mode], d, seat, role, f"{score:.4f}"]
                        )
                        n, tot = tallies.get(role, (0, 0.0))
                        tallies[role] = (n + 1, tot + score)
            f.flush()
            games = sum(n for n, _ in tallies.values())
            summary = "  ".join(
                f"{r}: n={n} mean={tot / n:+.3f}"
                for r, (n, tot) in sorted(tallies.items())
            )
            print(
                f"{label}: {games} games in {time.time() - t0:.0f}s | {summary}",
                flush=True,
            )
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
