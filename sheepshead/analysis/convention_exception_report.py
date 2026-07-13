#!/usr/bin/env python3
"""Exception-rate readout over a counterfactual trump-lead report (E3).

Post-processes the JSON written by ``counterfactual_trump_leads`` — run
*unconditionally* for this purpose (all defender leads with both options, e.g.
``--max-trick 1 --control-ratio 1e9`` so no FAIL-PREF subsampling) — and asks
the C1 optimality question the TRUMP-PREF-conditioned studies cannot: is
"never lead trump" optimal pointwise, or a heuristic with exception classes?

Per case the true-deal-MC Δscore (trump − fail) carries a per-case SE from the
branch rollout SDs (independent branches: SE = sqrt(sdT² + sdF²) / sqrt(R)).
A case is an EXCEPTION when the trump lead is better by more than ``--epsilon``
with its CI clear of zero: Δ − z·SE > ε. Convention-supported cases satisfy
Δ + z·SE < −ε; everything else is indeterminate at this rollout budget.

Pre-registered decision rule (Convention_Optimality_202607.md): convention
*optimal-as-a-rule* if exception rate < 5% and pooled mean Δ < 0 at 2σ;
*heuristic-with-exceptions* if the rate is ≥ 5% with rung-3 agreement on
sampled exceptions (re-run those seeds through the ladder with search enabled).

Usage:

    uv run python -m sheepshead.analysis.convention_exception_report \
        runs/convention_optimality_202607/cf_trump_unconditional.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _case_rows(payload: dict) -> list[dict]:
    rows = []
    for group_key in ("trumpPref", "failPref"):
        for c in payload.get(group_key, []):
            mc_t, mc_f = c["mcTrump"], c["mcFail"]
            R = mc_t["R"]
            se = (
                math.sqrt(mc_t["leaderScoreSd"] ** 2 + mc_f["leaderScoreSd"] ** 2)
                / math.sqrt(R)
                if R > 1
                else float("inf")
            )
            rows.append(
                {
                    "seed": c["seed"],
                    "stepIndex": c["stepIndex"],
                    "trickIndex": c["trickIndex"],
                    "group": c["group"],
                    "delta": c["mcDeltaScore"],
                    "se": se,
                    "handTrumpCount": c["node"]["handTrumpCount"],
                    "numFailOptions": len(c["node"]["failLeadOptions"]),
                    "queens": sum(
                        1 for card in c["node"]["hand"] if card.startswith("Q")
                    ),
                    "trumpCard": c["node"]["bestTrumpCard"],
                    "failCard": c["node"]["bestFailCard"],
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="JSON from counterfactual_trump_leads")
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Materiality threshold in score units (default 0.05 ~ eval MDE scale).",
    )
    ap.add_argument("--z", type=float, default=2.0, help="Per-case CI width.")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    payload = json.loads(Path(args.report).read_text())
    rows = _case_rows(payload)
    if not rows:
        print("No cases in report.")
        return 1

    n = len(rows)
    deltas = [r["delta"] for r in rows]
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    se_mean = math.sqrt(var / n)
    qs = sorted(deltas)

    def q(p: float) -> float:
        return qs[min(n - 1, int(p * n))]

    exceptions = [r for r in rows if r["delta"] - args.z * r["se"] > args.epsilon]
    supported = [r for r in rows if r["delta"] + args.z * r["se"] < -args.epsilon]
    lo, hi = _wilson(len(exceptions), n)

    print(f"Cases: {n}  (Δ = trump − fail, true-deal MC, leader score)")
    print(
        f"Pooled Δ: {mean:+.3f} (SE {se_mean:.3f})  "
        f"quantiles 10/50/90: {q(0.1):+.2f} / {q(0.5):+.2f} / {q(0.9):+.2f}"
    )
    print(
        f"Convention-supported (Δ+{args.z:g}·SE < −ε): {len(supported)} "
        f"= {len(supported) / n:.1%}"
    )
    print(
        f"EXCEPTIONS (Δ−{args.z:g}·SE > +ε={args.epsilon:g}): {len(exceptions)} "
        f"= {len(exceptions) / n:.1%}  (Wilson 95%: {lo:.1%}–{hi:.1%})"
    )
    print(
        "Pre-registered read: optimal-as-a-rule if exception rate < 5% "
        "and pooled Δ < 0 at 2σ."
    )

    if exceptions:
        print("\nException characterization:")
        for key in ("handTrumpCount", "trickIndex", "queens", "numFailOptions"):
            from collections import Counter

            dist = Counter(r[key] for r in exceptions)
            base = Counter(r[key] for r in rows)
            parts = [
                f"{v}: {dist[v]}/{base[v]}" for v in sorted(dist)
            ]
            print(f"  by {key:<15}: {'  '.join(parts)}")
        print("\nTop exceptions (for ladder re-runs with search):")
        for r in sorted(exceptions, key=lambda r: r["delta"], reverse=True)[:10]:
            print(
                f"  seed={r['seed']} step={r['stepIndex']} trick={r['trickIndex'] + 1} "
                f"{r['trumpCard']} over {r['failCard']}  Δ={r['delta']:+.2f} "
                f"(SE {r['se']:.2f}, {r['handTrumpCount']}T hand)"
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {
                        "report": args.report,
                        "epsilon": args.epsilon,
                        "z": args.z,
                    },
                    "n": n,
                    "pooledDelta": mean,
                    "pooledSe": se_mean,
                    "exceptionRate": len(exceptions) / n,
                    "exceptionWilson95": [lo, hi],
                    "supportedRate": len(supported) / n,
                    "exceptions": exceptions,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
