#!/usr/bin/env python3
"""C2 exception atlas over a counterfactual called-suit report (E6).

Post-processes the JSON written by ``counterfactual_called_suit_leads`` run
*unconditionally* over all eligible nodes (``--cases`` on a full
``scan_called_suit_leads`` node file) and asks the E6 question
(Convention_Optimality_202607.md): "always lead the called suit" is right on
average — where is it NOT, and what does search prefer instead?

Label rule (pre-registered): an ESS-ok search whose pi_gumbel argmax is a
non-called-suit lead marks the node an EXCEPTION. Tier A when the belief-MC
Δscore sign agrees with the verdict (neutral |Δ| < 1 per-case SE counts with
the verdict); tier B otherwise. Skipped/ESS-low nodes are reported, never
imputed.

Per pre-registered hypothesis H1–H6 the report prints exception-rate splits
with Wilson 95% CIs, plus replacement-class buckets of the search-preferred
card and the AGREE-group sanity read.

Usage:

    uv run python -m sheepshead.analysis.called_suit_exception_report \
        runs/convention_optimality_202607/cf_called_suit_atlas_2800k.json \
        --out runs/convention_optimality_202607/e6_exception_atlas.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sheepshead import FAIL, TRUMP_SET
from sheepshead.analysis.convention_exception_report import _wilson
from sheepshead.game import CARD_POINTS

FAIL_SET = set(FAIL)
FAIL_ACES = {"AS", "AH", "AC"}


def _called_suit_fail(card: str, called: str) -> bool:
    return card in FAIL_SET and card[-1] == called[-1]


def _row(case: dict) -> dict | None:
    """Flatten one counterfactual case into a labeled feature row."""
    search = case.get("search")
    if not search or not search.get("ok"):
        return None
    gum_action = search.get("gumbelAction")
    if not gum_action or not gum_action.startswith("PLAY "):
        return None
    gum_card = gum_action[5:]
    called = case["calledCard"]
    exception = not _called_suit_fail(gum_card, called)

    # Tier: does the belief-MC sign agree with the verdict? Δ = conv − alt, so
    # an exception expects Δ ≤ 0 and a conv verdict expects Δ ≥ 0. Neutral
    # (|Δ| below the per-case MC SE) counts with the verdict.
    delta_b = case.get("beliefMcDeltaScore")
    mc_conv, mc_alt = case["mcConv"], case["mcAlt"]
    r_n = mc_conv.get("R", 0)
    se = (
        math.sqrt(mc_conv["leaderScoreSd"] ** 2 + mc_alt["leaderScoreSd"] ** 2)
        / math.sqrt(r_n)
        if r_n > 1
        else float("inf")
    )
    if delta_b is None:
        tier = "B"
    elif abs(delta_b) < se:
        tier = "A"
    else:
        tier = "A" if (delta_b < 0) == exception else "B"

    hand = case["hand"]
    lead_logits = (case.get("node") or {}).get("leadLogits") or {}
    called_opts = [c for c in lead_logits if _called_suit_fail(c, called)]
    called_held = [c for c in hand if _called_suit_fail(c, called)]
    alt_card = case["altCard"]

    # Replacement class of the search-preferred card (exceptions only).
    if not exception:
        repl = "conv"
    elif gum_card in TRUMP_SET:
        repl = "trump"
    else:
        same_suit_as_alt = gum_card[-1] == alt_card[-1] and alt_card in FAIL_SET
        fat = CARD_POINTS.get(gum_card, 0) >= 10
        repl = ("same-suit-" if same_suit_as_alt else "off-suit-") + (
            "fat" if fat else "low"
        )

    return {
        "seed": case["seed"],
        "stepIndex": case["stepIndex"],
        "group": case["group"],
        "exception": exception,
        "tier": tier,
        "replacementClass": repl,
        "searchCard": gum_card,
        "convCard": case["convCard"],
        "altCard": alt_card,
        "mcDeltaScore": case["mcDeltaScore"],
        "beliefMcDeltaScore": delta_b,
        "mcSe": se,
        # Pre-registered features (H1–H6).
        "trumpVoid": (case.get("node") or {}).get("handTrumpCount", 0) == 0,
        "minDonation": min(
            (CARD_POINTS.get(c, 0) for c in (called_opts or called_held)),
            default=0,
        ),
        "sideAce": any(
            c in FAIL_ACES and not _called_suit_fail(c, called) for c in hand
        ),
        "relPos": case["relPosFromPicker"],
        "calledLen": len(called_held),
        "trickIndex": case["trickIndex"],
    }


def _split(rows: list[dict], name: str, pred) -> str:
    sel = [r for r in rows if pred(r)]
    rest = [r for r in rows if not pred(r)]
    parts = []
    for label, grp in ((name, sel), ("else", rest)):
        k = sum(r["exception"] for r in grp)
        n = len(grp)
        if n:
            lo, hi = _wilson(k, n)
            parts.append(f"{label}: {k}/{n} = {k / n:.0%} [{lo:.0%},{hi:.0%}]")
        else:
            parts.append(f"{label}: n=0")
    return "  vs  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="cf_called_suit atlas JSON (unconditional run)")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    payload = json.loads(Path(args.report).read_text())
    cases = payload["groups"]["agree"] + payload["groups"]["disagree"]
    rows = [r for r in (_row(c) for c in cases) if r is not None]
    dropped = len(cases) - len(rows)

    k = sum(r["exception"] for r in rows)
    n = len(rows)
    lo, hi = _wilson(k, n)
    print(f"Nodes labeled: {n} ({dropped} dropped: ESS-low or no search verdict)")
    print(f"POOLED exception rate: {k}/{n} = {k / n:.1%}  Wilson95 [{lo:.1%},{hi:.1%}]")
    tier_a = [r for r in rows if r["tier"] == "A"]
    print(
        f"  tier A (belief-MC agrees/neutral): {len(tier_a)}/{n}; "
        f"tier-A-only exception rate "
        f"{sum(r['exception'] for r in tier_a)}/{len(tier_a)}"
    )

    # Sanity cell: exceptions among nodes where the policy adhered (AGREE).
    for grp in ("agree", "disagree"):
        sub = [r for r in rows if r["group"] == grp]
        if sub:
            gk, gn = sum(r["exception"] for r in sub), len(sub)
            glo, ghi = _wilson(gk, gn)
            print(
                f"  {grp.upper():>8}: exceptions {gk}/{gn} = {gk / gn:.0%} "
                f"[{glo:.0%},{ghi:.0%}]"
            )

    print("\nPre-registered feature splits (exception rate, Wilson 95%):")
    print("  H1 " + _split(rows, "trump-void", lambda r: r["trumpVoid"]))
    print("  H2 " + _split(rows, "min-donation>=10", lambda r: r["minDonation"] >= 10))
    print("  H3 " + _split(rows, "side-ace", lambda r: r["sideAce"]))
    print("  H4 " + _split(rows, "pos+1/+4", lambda r: r["relPos"] in (1, 4)))
    for rel in (1, 2, 3, 4):
        sub = [r for r in rows if r["relPos"] == rel]
        sk, sn = sum(r["exception"] for r in sub), len(sub)
        rate = f"{sk / sn:.0%}" if sn else "n/a"
        print(f"       picker+{rel}: {sk}/{sn} = {rate}")
    print("  H5 " + _split(rows, "called-len>=3", lambda r: r["calledLen"] >= 3))
    print("  H6 " + _split(rows, "trick>=3", lambda r: r["trickIndex"] >= 2))
    print(
        "  H1xH6 "
        + _split(
            rows, "void-and-late", lambda r: r["trumpVoid"] and r["trickIndex"] >= 2
        )
    )

    exc = [r for r in rows if r["exception"]]
    print(f"\nReplacement classes among {len(exc)} exceptions:")
    for cls in sorted({r["replacementClass"] for r in exc}):
        c = sum(1 for r in exc if r["replacementClass"] == cls)
        print(f"  {cls:>14}: {c:3d} ({c / len(exc):.0%})")

    print("\nStrongest exceptions (belief-MC most anti-convention):")
    for r in sorted(
        exc,
        key=lambda r: (r["beliefMcDeltaScore"] is None, r["beliefMcDeltaScore"] or 0),
    )[:10]:
        print(
            f"  seed={r['seed']} step={r['stepIndex']} trick={r['trickIndex'] + 1} "
            f"pos+{r['relPos']} conv={r['convCard']} -> search {r['searchCard']} "
            f"({r['replacementClass']}) beliefΔ={r['beliefMcDeltaScore']:+.2f} "
            f"tier {r['tier']}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "meta": {"source": args.report, "labeled": n, "dropped": dropped},
                    "pooled": {"exceptions": k, "n": n, "wilson95": [lo, hi]},
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"\nWrote atlas -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
