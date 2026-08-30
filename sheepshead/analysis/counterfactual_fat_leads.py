#!/usr/bin/env python3
"""Counterfactual analysis of defender fail-lead POINT CLASS (fat vs nopoint).

Search_Teacher_Design §12.14 follow-up: the resolved-pair teacher's probe
metric tracked a class-level tilt (fewer A/10 "fat" fail leads, more 7/8/9
"nopoint") that per-card committee resolution cannot certify at ~95% of
fat-argmax defender-lead nodes. This study asks the reward channel directly:
at defender leads offering BOTH classes, is leading nopoint better than
leading fat?

For every ELIGIBLE node — a defender lead in called-ace mode (non-alone,
non-leaster, tricks 0..max-trick) whose legal FAIL leads include at least one
fat (A/10) and one nopoint (7/8/9) card — the branches are forced with the
same CRN primitives as ``counterfactual_trump_leads``:

1. single deterministic rollout, 2. paired true-deal MC, 2b. paired
belief-pool MC. (No search rung by default: search opinion is what the
reward channel is being asked to arbitrate.)

Δ is always (nopoint branch − fat branch): conv = the policy's best nopoint
fail by logit, alt = its best fat fail. Called-suit fails are EXCLUDED from
both pools while the called suit is unled — those leads belong to the C2
convention (``counterfactual_called_suit_leads``), and mixing them would
confound the class question. Groups by the policy's actual argmax:

  * FAT-ARGMAX     — argmax lead is a fat fail. The decision group: Δ > 0
                     means the taught tilt beats the policy's choice.
  * NOPOINT-ARGMAX — argmax lead is a nopoint fail. Sanity: Δ ≥ 0 expected
                     under the policy's own rollouts.
  * OTHER-ARGMAX   — argmax is trump / K / called-suit. Context cell.

Usage (from repo root):

    uv run python -m sheepshead.analysis.counterfactual_fat_leads \
        --model runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt \
        --num-seeds 2000 --max-trick 2 --rollouts 50 \
        --out runs/search_teacher_ev_202608/cf_fat_leads_8m.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import sheepshead.analysis.counterfactual_trump_leads as cf  # noqa: E402
import sheepshead.analysis.scan_defender_trump_leads as scan  # noqa: E402
from server.api.schemas import AnalyzeSimulateRequest  # noqa: E402
from sheepshead import TRUMP  # noqa: E402
from sheepshead.analysis.scan_called_suit_leads import (  # noqa: E402
    _called_suit_already_led,
    _called_suit_fail,
)

DEFAULT_MODEL = scan.DEFAULT_MODEL
PARTNER_MODE_CALLED_ACE = 1

FAT_RANKS = ("A", "10")
NOPOINT_RANKS = ("7", "8", "9")


def _lead_class(card: str, called: Optional[str], called_led: bool) -> str:
    """'fat' | 'nopoint' | 'other' for a LEAD of ``card``. Called-suit fails
    while the suit is unled are 'other' (they belong to the C2 study)."""
    if card in TRUMP:
        return "other"
    if called and not called_led and _called_suit_fail(card, called):
        return "other"
    rank = card[:-1]
    if rank in FAT_RANKS:
        return "fat"
    if rank in NOPOINT_RANKS:
        return "nopoint"
    return "other"


@dataclass
class FatCaseResult:
    seed: int
    stepIndex: int
    trickIndex: int
    seat: int
    seatName: str
    pickerSeat: int
    relPosFromPicker: int
    group: str  # "fat" | "nopoint" | "other" (argmax class)
    calledCard: Optional[str]
    convCard: str  # forced nopoint lead
    altCard: str  # forced fat lead
    hand: List[str]
    node: cf.NodeInfo
    detConv: cf.DetBranch
    detAlt: cf.DetBranch
    detDeltaPoints: int
    detDeltaScore: int
    mcConv: cf.McBranch
    mcAlt: cf.McBranch
    mcDeltaPoints: float
    mcDeltaScore: float
    mcDeltaWin: float
    beliefMcConv: Optional[cf.BeliefMcBranch] = None
    beliefMcAlt: Optional[cf.BeliefMcBranch] = None
    beliefMcDeltaPoints: Optional[float] = None
    beliefMcDeltaScore: Optional[float] = None
    beliefMcDeltaWin: Optional[float] = None


# ---------------------------------------------------------------------------
# Case detection
# ---------------------------------------------------------------------------
def _classify_spots(resp, seed: int, max_trick: int) -> List[dict]:
    spots: List[dict] = []
    for ad in resp.trace:
        if not ad.action.startswith("PLAY "):
            continue
        card = ad.action[5:]
        view = ad.view
        if not all(c == "" for c in (view.get("current_trick") or [])):
            continue  # leads only
        if view.get("is_leaster") or view.get("alone"):
            continue
        ti = int(view.get("current_trick_index", 0))
        if ti > max_trick:
            continue

        seat = ad.seat
        picker = view.get("picker") or 0
        partner = view.get("partner") or 0
        if seat == picker or seat == partner:
            continue
        if scan._is_secret_partner(view, PARTNER_MODE_CALLED_ACE):
            continue

        called = view.get("called_card")
        called_led = _called_suit_already_led(view) if called else True
        legal_leads = [
            c for v in ad.validActionIds if (c := cf._card_of(v)) is not None
        ]
        classes = {c: _lead_class(c, called, called_led) for c in legal_leads}
        if not any(v == "fat" for v in classes.values()) or not any(
            v == "nopoint" for v in classes.values()
        ):
            continue

        spots.append(
            {
                "seed": seed,
                "partnerMode": PARTNER_MODE_CALLED_ACE,
                "stepIndex": ad.stepIndex,
                "trickIndex": ti,
                "seat": seat,
                "seatName": ad.seatName,
                "pickerSeat": picker,
                "calledCard": called,
                "calledLed": called_led,
                "cardLed": card,
                "group": classes.get(card, "other"),
            }
        )
    return spots


def _find_cases(args) -> Dict[str, List[dict]]:
    scan.set_scan_model(args.model)
    groups: Dict[str, List[dict]] = {"fat": [], "nopoint": [], "other": []}
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        req = AnalyzeSimulateRequest(
            seed=seed,
            partnerMode=PARTNER_MODE_CALLED_ACE,
            deterministic=True,
            maxSteps=args.max_steps,
        )
        resp = scan.simulate_game(req)
        for spot in _classify_spots(resp, seed, args.max_trick):
            groups[spot["group"]].append(spot)

    print(
        f"Scanned {args.num_seeds} seeds (from {args.start_seed}) -> "
        f"{len(groups['fat'])} FAT-argmax, {len(groups['nopoint'])} "
        f"NOPOINT-argmax, {len(groups['other'])} OTHER-argmax dual-class "
        f"defender leads on tricks 0-{args.max_trick}"
    )
    rng = random.Random(args.subsample_seed)
    for name, spots in groups.items():
        if len(spots) > args.max_cases_per_group:
            rng.shuffle(spots)
            groups[name] = spots[: args.max_cases_per_group]
            print(f"  {name}: subsampled to {args.max_cases_per_group}")
    return groups


# ---------------------------------------------------------------------------
# Per-case analysis
# ---------------------------------------------------------------------------
def _pick_branch_cards(node: cf.NodeInfo, spot: dict) -> Optional[tuple[str, str]]:
    """(nopoint card, fat card) by policy logit, or None if the node no longer
    offers both classes on replay."""
    called = spot["calledCard"]
    called_led = spot["calledLed"]
    logits = node.leadLogits or {}
    conv_pool = {
        c: v
        for c, v in logits.items()
        if _lead_class(c, called, called_led) == "nopoint"
    }
    alt_pool = {
        c: v for c, v in logits.items() if _lead_class(c, called, called_led) == "fat"
    }
    if not conv_pool or not alt_pool:
        return None
    return (
        max(conv_pool, key=lambda c: conv_pool[c]),
        max(alt_pool, key=lambda c: alt_pool[c]),
    )


def analyze_case(agent, teacher, spot: dict, args, device) -> Optional[FatCaseResult]:
    seed = spot["seed"]
    target_step, seat = spot["stepIndex"], spot["seat"]
    det_rng = random.Random(0xFA7 ^ (seed << 8) ^ target_step)

    cap = cf._replay_to_node(
        agent,
        seed,
        PARTNER_MODE_CALLED_ACE,
        target_step,
        args.max_steps,
        device,
        teacher=None,
        det_rng=det_rng,
        iters=args.iters,
        rollout_depth=args.rollout_depth,
        min_visit_frac=args.min_visit_frac,
    )
    if cap is None:
        print(f"  ! seed={seed} step={target_step}: node not reached; skipping")
        return None
    node_game, node_mem, node, _search, forced_public = cap
    if node.argmaxCard != spot["cardLed"]:
        print(
            f"  ! seed={seed} step={target_step}: argmax {node.argmaxCard} "
            f"!= scanned {spot['cardLed']}; skipping (non-reproducing)"
        )
        return None

    cards = _pick_branch_cards(node, spot)
    if cards is None:
        print(f"  ! seed={seed} step={target_step}: branch cards unavailable; skipping")
        return None
    conv_card, alt_card = cards

    det_conv = cf._force_and_play(
        agent, node_game, node_mem, seat, conv_card, device, deterministic=True
    )
    det_alt = cf._force_and_play(
        agent, node_game, node_mem, seat, alt_card, device, deterministic=True
    )

    torch.manual_seed(0xFA7A11 ^ (seed << 4) ^ target_step)
    mc_conv = cf._mc_branch(
        agent, node_game, node_mem, seat, conv_card, args.rollouts, device
    )
    mc_alt = cf._mc_branch(
        agent, node_game, node_mem, seat, alt_card, args.rollouts, device
    )

    belief_conv = belief_alt = None
    if teacher is not None and not args.no_belief_mc:
        belief_rng = random.Random(0xFA7BE1 ^ (seed << 6) ^ target_step)
        torch.manual_seed(0xFA7BE1 ^ (seed << 4) ^ target_step)
        pool_k = args.belief_worlds if args.belief_worlds is not None else args.iters
        belief_conv, belief_alt = cf._belief_mc(
            agent,
            teacher,
            node_game,
            seat,
            forced_public,
            conv_card,
            alt_card,
            args.rollouts,
            pool_k,
            belief_rng,
            device,
        )

    return FatCaseResult(
        seed=seed,
        stepIndex=target_step,
        trickIndex=spot["trickIndex"],
        seat=seat,
        seatName=spot["seatName"],
        pickerSeat=spot["pickerSeat"],
        relPosFromPicker=(seat - spot["pickerSeat"]) % 5,
        group=spot["group"],
        calledCard=spot["calledCard"],
        convCard=conv_card,
        altCard=alt_card,
        hand=node.hand,
        node=node,
        detConv=det_conv,
        detAlt=det_alt,
        detDeltaPoints=det_conv.defenderPoints - det_alt.defenderPoints,
        detDeltaScore=det_conv.leaderScore - det_alt.leaderScore,
        mcConv=mc_conv,
        mcAlt=mc_alt,
        mcDeltaPoints=mc_conv.defenderPointsMean - mc_alt.defenderPointsMean,
        mcDeltaScore=mc_conv.leaderScoreMean - mc_alt.leaderScoreMean,
        mcDeltaWin=mc_conv.winRate - mc_alt.winRate,
        beliefMcConv=belief_conv,
        beliefMcAlt=belief_alt,
        beliefMcDeltaPoints=(
            belief_conv.defenderPointsMean - belief_alt.defenderPointsMean
            if belief_conv
            else None
        ),
        beliefMcDeltaScore=(
            belief_conv.leaderScoreMean - belief_alt.leaderScoreMean
            if belief_conv
            else None
        ),
        beliefMcDeltaWin=(
            belief_conv.winRate - belief_alt.winRate if belief_conv else None
        ),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _mean_se(xs: List[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se


def _print_group(name: str, blurb: str, results: List[FatCaseResult]) -> None:
    print("\n" + "=" * 72)
    print(f"{name}  (n = {len(results)} states)  {blurb}")
    print("=" * 72)
    if not results:
        print("  (no states)")
        return
    for label, key in (
        ("det (1 rollout)", "detDeltaScore"),
        ("true-deal MC", "mcDeltaScore"),
        ("belief-pool MC", "beliefMcDeltaScore"),
    ):
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        if not vals:
            continue
        m, se = _mean_se(vals)
        pos = sum(1 for v in vals if v > 0) / len(vals)
        print(
            f"  {label:<16}: Δscore {m:+.3f} (SE {se:.3f})  nopoint better in {pos:.0%}"
        )
    pts = [r.mcDeltaPoints for r in results]
    wins = [r.mcDeltaWin for r in results]
    m_p, se_p = _mean_se(pts)
    m_w, se_w = _mean_se(wins)
    print(
        f"  true-deal MC    : Δpts {m_p:+.2f} (SE {se_p:.2f})  "
        f"Δwin {m_w * 100:+.1f}% (SE {se_w * 100:.1f})"
    )
    for ti in (0, 1, 2):
        sub = [r.mcDeltaScore for r in results if r.trickIndex == ti]
        if sub:
            m, se = _mean_se(sub)
            print(
                f"  trick-{ti} subset  : n={len(sub)}  MC Δscore {m:+.3f} (SE {se:.3f})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-trick", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--iters", type=int, default=384)
    parser.add_argument("--min-visit-frac", type=float, default=0.01)
    parser.add_argument("--rollout-depth", type=int, default=None)
    parser.add_argument("--belief-worlds", type=int, default=192)
    parser.add_argument(
        "--no-belief-mc", action="store_true", help="Skip the belief-pool MC rung."
    )
    parser.add_argument("--max-cases-per-group", type=int, default=120)
    parser.add_argument("--subsample-seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = cf._device()
    scan.set_scan_model(args.model)
    agent = scan._cached_load_agent(args.model)

    teacher = None
    if not args.no_belief_mc:
        from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

        cfg = ISMCTSConfig()
        cfg.iters = {k: args.iters for k in cfg.iters}
        teacher = ISMCTSTeacher(agent, cfg)

    groups = _find_cases(args)

    results: Dict[str, List[FatCaseResult]] = {}
    for name in ("fat", "nopoint", "other"):
        spots = groups[name]
        print(f"\n>>> Analyzing {len(spots)} {name.upper()}-argmax case(s)")
        out: List[FatCaseResult] = []
        for i, spot in enumerate(spots):
            r = analyze_case(agent, teacher, spot, args, device)
            if r is not None:
                out.append(r)
                print(
                    f"    [{i + 1}/{len(spots)}] seed={r.seed} step={r.stepIndex} "
                    f"trick={r.trickIndex + 1} conv={r.convCard} alt={r.altCard} "
                    f"mc={r.mcDeltaScore:+.2f}",
                    flush=True,
                )
        results[name] = out

    _print_group(
        "FAT-ARGMAX (policy leads fat)",
        "decision group: Δ > 0 supports the taught tilt",
        results["fat"],
    )
    _print_group(
        "NOPOINT-ARGMAX (policy leads nopoint)",
        "sanity: Δ ≥ 0 expected",
        results["nopoint"],
    )
    _print_group(
        "OTHER-ARGMAX (policy leads trump/K/called-suit)",
        "context cell",
        results["other"],
    )
    print(
        "\nInterpretation (Δ = nopoint − fat from the LEADER's team view via "
        "leaderScore).\nFAT-ARGMAX is the decision cell: Δ > 0 at 2σ on the "
        "true-deal MC rung with sign agreement on belief-MC supports teaching "
        "the tilt; |Δ| below ~0.05 with tight SE retires the fat/nopoint probe "
        "metric as a teaching target. NOPOINT-ARGMAX Δ < 0 at 2σ would indict "
        "the measurement."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "model": args.model,
                "startSeed": args.start_seed,
                "numSeeds": args.num_seeds,
                "maxTrick": args.max_trick,
                "rollouts": args.rollouts,
                "beliefWorlds": args.belief_worlds,
                "maxCasesPerGroup": args.max_cases_per_group,
            },
            "groups": {name: [asdict(r) for r in results[name]] for name in results},
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
