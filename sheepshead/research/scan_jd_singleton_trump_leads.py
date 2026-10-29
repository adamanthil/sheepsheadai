#!/usr/bin/env python3
"""Scan simulated JD games for t0 defender leads holding a single trump
(Convention_Optimality addendum A1).

JD mode. An ELIGIBLE node is the first lead of the hand by a defender (not
the picker, not holding the JD) whose hand has exactly one trump and at least
one fail. Search says leading that lone trump is a mistake when the hand has
a fail void (the trump is its ruff), and a tie-band exception with two fail
aces and no void. The scan records the policy's choice and probabilities so
those cells can be read on the current checkpoint and replayed on /analyze.

The scanner drives the same deterministic simulation path as ``/analyze``
(``sheepshead.agent.traced_game.play_traced_game``), so every ``seed`` /
``stepIndex`` reproduces on the Analyze page (Partner Mode = JD,
Deterministic on, the same checkpoint loaded).

Usage (from repo root):

    uv run python -m sheepshead.research.scan_jd_singleton_trump_leads \
        --model runs/202609_recall_rc/pi/iter2/distill_epoch10.pt \
        --num-seeds 4000 --out runs/202609_recall_rc/t0_trump_confirm/analyze_seeds_jd_singleton.jsonl
    uv run python -m sheepshead.research.scan_jd_singleton_trump_leads \
        --summary runs/202609_recall_rc/t0_trump_confirm/analyze_seeds_jd_singleton.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import time

from sheepshead.agent.traced_game import play_traced_game
from sheepshead.game import TRUMP_SET
from sheepshead.research.scan_defender_trump_leads import (
    DEFAULT_MODEL,
    load_scan_agent,
)

PARTNER_MODE_JD = 0


def scan(model: str, seed_start: int, num_seeds: int, out: str) -> None:
    agent = load_scan_agent(model)
    t0 = time.time()
    with open(out, "w") as f:
        for seed in range(seed_start, seed_start + num_seeds):
            traced = play_traced_game(
                agent, seed=seed, partner_mode=PARTNER_MODE_JD, deterministic=True
            )
            for ad in traced.decisions:
                if not ad.action.startswith("PLAY "):
                    continue
                v = ad.view
                if int(v.get("current_trick_index", 0)) != 0 or not all(
                    c == "" for c in (v.get("current_trick") or [])
                ):
                    continue
                if v.get("is_leaster") or v.get("alone"):
                    continue
                hand = v.get("hand") or []
                seat = ad.seat
                picker = v.get("picker") or 0
                if seat == picker or "JD" in hand:
                    continue
                trumps = [c for c in hand if c in TRUMP_SET]
                fails = [c for c in hand if c not in TRUMP_SET]
                if len(trumps) != 1 or not fails:
                    continue
                probs = {
                    p.action[5:]: p.prob
                    for p in ad.inference.probabilities()
                    if p.action.startswith("PLAY ")
                }
                best_fail = max(fails, key=lambda c: probs.get(c, 0))
                led = ad.action[5:]
                rec = {
                    "seed": seed,
                    "partnerMode": PARTNER_MODE_JD,
                    "stepIndex": ad.step_index,
                    "seat": seat,
                    "seatName": ad.seat_name,
                    "picker": picker,
                    "picker_rel": ((picker - seat) % 5) + 1,
                    "hand": hand,
                    "trump": trumps[0],
                    "fails": fails,
                    "aces": sum(c[:-1] == "A" for c in fails),
                    "void": 3 - len({c[-1] for c in fails}),
                    "led": led,
                    "led_trump": led in TRUMP_SET,
                    "p_trump": probs.get(trumps[0], 0),
                    "best_fail": best_fail,
                    "p_best_fail": probs.get(best_fail, 0),
                    "final": traced.final,
                }
                f.write(json.dumps(rec, default=str) + "\n")
            if (seed - seed_start) % 500 == 499:
                print(seed + 1, f"{time.time() - t0:.0f}s", flush=True)


def summary(path: str) -> None:
    """Led-trump counts by fail void x fail aces."""
    rows = [json.loads(line) for line in open(path)]
    cells: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for r in rows:
        cells[("void" if r["void"] else "novoid", r["aces"])].append(r)
    print(f"{len(rows)} singleton-trump t0 defender leads")
    for k in sorted(cells):
        x = cells[k]
        print(
            f"  {k[0]:6s} {k[1]} aces: led trump {sum(r['led_trump'] for r in x)}/{len(x)}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint to scan")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=4000)
    ap.add_argument("--out", help="JSONL of eligible nodes")
    ap.add_argument(
        "--summary", metavar="SCAN_JSONL", help="print the cell table of a scan"
    )
    args = ap.parse_args(argv)
    if args.summary:
        summary(args.summary)
        return 0
    if not args.out:
        ap.error("--out is required to scan")
    scan(args.model, args.seed_start, args.num_seeds, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
