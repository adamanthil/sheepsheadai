#!/usr/bin/env python3
"""Terminal-only learnability SNR for a convention decision class (E5).

Turns the documented "PPO can't learn small early gaps" mechanism into numbers
for a specific decision class, from a counterfactual ladder report:

  * Δ̄ — per-decision advantage of the convention action (|mean Δscore| across
    the class's cases, true-deal MC rung),
  * σ — per-rollout terminal-return SD at those nodes (RMS of the branch
    rollout SDs): the noise a terminal-only advantage estimate faces,
  * p — class visitation per training episode (from a scanner/probe report or
    given directly),
  * N ≈ (z·σ/Δ̄)² / p — episodes for the advantage sign at the class to clear
    z·SE, i.e. an order-of-magnitude "episodes to detect" under terminal-only
    reward with no variance reduction.

This is deliberately crude: it ignores GAE/critic variance reduction (which is
exactly what the oracle-critic arms change) and credit assignment across the
episode, so it bounds the *raw* signal. Pre-registered read
(Convention_Optimality_202607.md): likely learnable if N < 20% of a realistic
budget AND the critic gap has the right sign; the critic probe is separate.

Usage:

    uv run python -m sheepshead.analysis.convention_learnability \
        runs/convention_optimality_202607/cf_called_suit.json \
        --group disagree --scan-json runs/convention_optimality_202607/called_suit_scan.json \
        --budgets 1000000,30000000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _extract_cases(payload: dict, group: str) -> list[dict]:
    """Return per-case (delta, sd_a, sd_b, R) rows from either ladder layout."""
    if "groups" in payload:  # counterfactual_called_suit_leads layout
        cases = payload["groups"].get(group, [])
        return [
            {
                "delta": c["mcDeltaScore"],
                "sd_a": c["mcConv"]["leaderScoreSd"],
                "sd_b": c["mcAlt"]["leaderScoreSd"],
                "R": c["mcConv"]["R"],
            }
            for c in cases
        ]
    cases = payload.get(group, [])  # counterfactual_trump_leads layout
    return [
        {
            "delta": c["mcDeltaScore"],
            "sd_a": c["mcTrump"]["leaderScoreSd"],
            "sd_b": c["mcFail"]["leaderScoreSd"],
            "R": c["mcTrump"]["R"],
        }
        for c in cases
    ]


def _visitation_from_scan(path: str) -> float:
    """Eligible nodes per scanned game, from a scanner JSON's stats block."""
    stats = json.loads(Path(path).read_text())["stats"]
    seeds = stats.get("seedsScanned")
    eligible = stats.get("eligible", stats.get("trumpLeadCases"))
    if not seeds or eligible is None:
        raise SystemExit(f"{path}: no seedsScanned/eligible stats")
    return eligible / seeds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="JSON from a counterfactual ladder run")
    ap.add_argument(
        "--group",
        default="disagree",
        help="Case group: disagree/agree/partner (C2 layout) or "
        "trumpPref/failPref (C1 layout).",
    )
    ap.add_argument(
        "--scan-json",
        default=None,
        help="Scanner report to derive visitation p (eligible per game).",
    )
    ap.add_argument(
        "--p", type=float, default=None, help="Visitation per episode, direct."
    )
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument(
        "--budgets",
        default="1000000,30000000",
        help="Comma-separated training budgets (episodes) to compare against.",
    )
    args = ap.parse_args()

    if args.p is None and args.scan_json is None:
        ap.error("provide --p or --scan-json")
    p = args.p if args.p is not None else _visitation_from_scan(args.scan_json)

    payload = json.loads(Path(args.report).read_text())
    rows = _extract_cases(payload, args.group)
    if not rows:
        raise SystemExit(f"no cases in group '{args.group}'")

    n = len(rows)
    mean_delta = sum(r["delta"] for r in rows) / n
    # Δ orientation differs by layout: C2 groups store conv − alt (positive =
    # convention better); C1 groups store trump − fail (NEGATIVE = convention
    # better). Report the raw group Δ and derive the convention advantage
    # explicitly so the sign cannot be misread (it was, once: 2026-07-15).
    c1_layout = "groups" not in payload
    conv_adv = -mean_delta if c1_layout else mean_delta
    delta = abs(mean_delta)
    sigma = math.sqrt(
        sum(r["sd_a"] ** 2 + r["sd_b"] ** 2 for r in rows) / (2 * n)
    )
    if delta == 0:
        raise SystemExit("mean Δ is exactly 0; nothing to detect")

    snr = delta / sigma if sigma else float("inf")
    n_detect = (args.z * sigma / delta) ** 2 / p if sigma else 0.0

    print(f"group={args.group}  cases={n}")
    if c1_layout:
        print(
            f"Δ̄ raw (trump − fail): {mean_delta:+.3f} score  "
            f"⇒ convention (fail) advantage {conv_adv:+.3f}  (|Δ̄| used: {delta:.3f})"
        )
    else:
        print(
            f"Δ̄ (convention advantage, conv − alt): {conv_adv:+.3f} score  "
            f"(|Δ̄| used: {delta:.3f})"
        )
    print(f"σ (terminal noise/rollout): {sigma:.3f} score  -> per-visit SNR {snr:.4f}")
    print(f"p (visitation/episode):     {p:.4f}")
    print(
        f"N_detect ≈ (z·σ/Δ̄)²/p = ({args.z:g}·{sigma:.2f}/{delta:.3f})²/{p:.4f} "
        f"= {n_detect:,.0f} episodes"
    )
    for b in (float(x) for x in args.budgets.split(",")):
        frac = n_detect / b
        verdict = "likely learnable" if frac < 0.2 else "UNLIKELY terminal-only"
        print(f"  vs budget {b:>12,.0f}: {frac:>8.1%} of budget -> {verdict}")
    print(
        "(Raw-signal bound: ignores GAE/critic variance reduction and credit "
        "assignment; the oracle-critic comparison is the empirical test.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
