#!/usr/bin/env python3
"""Counterfactual analysis of defender called-suit leads (convention C2, E2).

For every C2-ELIGIBLE node — a defender lead in called-ace mode (non-alone,
non-leaster, non-under-call) with a legal called-suit fail, a legal alternative,
and the called suit never yet led — we ask: is leading the called suit better
than the best alternative lead? Answered with the same 3-rung ladder as the
trump-lead study (``counterfactual_trump_leads``, whose primitives this reuses):

1. single deterministic rollout, 2. paired true-deal MC, 2b. paired belief-pool
MC, 3. ISMCTS search at the node (top@Q verdict).

Δ is always (convention branch − alternative branch): the convention card is the
policy's best called-suit fail by logit, the alternative is its best
non-called-suit lead. Groups (pre-registered, Convention_Optimality_202607.md):

  * AGREE    — argmax already leads the called suit (conv = argmax). Sanity:
               Δ ≥ 0 expected under the policy's own rollouts, or the forcing
               machinery is suspect.
  * DISAGREE — argmax leads something else (alt = argmax). The decision group:
               Δ > 0 here means the convention beats the policy's choice.
  * PARTNER  — mirror falsifier: the *secret partner* leads while the suit is
               unled; conv = surfacing the called card itself, alt = its best
               other lead. The leader is on the picker team, so if the method
               were rubber-stamping "called-suit leads help the leader",
               ΔleaderScore would be > 0 here too; convention theory says ≤ 0.

Note the engine makes points and scores zero-sum across teams by construction,
so the design's "zero-sum check" is vacuous and PARTNER + AGREE carry the
falsification load.

Usage (from repo root):

    uv run python -m sheepshead.analysis.counterfactual_called_suit_leads \
        --num-seeds 3200 --rollouts 50 --iters 384 \
        --out runs/convention_optimality_202607/cf_called_suit.json
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
from sheepshead.analysis.scan_called_suit_leads import (  # noqa: E402
    _called_suit_already_led,
    _called_suit_fail,
)
from server.api.schemas import AnalyzeSimulateRequest  # noqa: E402

DEFAULT_MODEL = scan.DEFAULT_MODEL
PARTNER_MODE_CALLED_ACE = 1


@dataclass
class C2CaseResult:
    seed: int
    stepIndex: int
    trickIndex: int
    seat: int
    seatName: str
    pickerSeat: int
    relPosFromPicker: int
    group: str  # "agree" | "disagree" | "partner"
    calledCard: str
    convCard: str  # forced convention lead
    altCard: str  # forced alternative lead
    hand: List[str]
    node: cf.NodeInfo
    # Single deterministic rollout (Δ = conv − alt).
    detConv: cf.DetBranch
    detAlt: cf.DetBranch
    detDeltaPoints: int
    detDeltaScore: int
    # Paired Monte-Carlo over the TRUE deal.
    mcConv: cf.McBranch
    mcAlt: cf.McBranch
    mcDeltaPoints: float
    mcDeltaScore: float
    mcDeltaWin: float
    # Paired Monte-Carlo over the BELIEF pool.
    beliefMcConv: Optional[cf.BeliefMcBranch] = None
    beliefMcAlt: Optional[cf.BeliefMcBranch] = None
    beliefMcDeltaPoints: Optional[float] = None
    beliefMcDeltaScore: Optional[float] = None
    beliefMcDeltaWin: Optional[float] = None
    search: Optional[cf.SearchOutcome] = None
    # Search verdict in C2 terms: is the top@Q lead a convention lead?
    searchTopQIsConv: Optional[bool] = None


# ---------------------------------------------------------------------------
# Case detection
# ---------------------------------------------------------------------------
def _classify_c2_spots(resp, seed: int, max_trick: int) -> tuple[List[dict], int]:
    """C2-eligible defender leads (agree/disagree) and partner-mirror leads in
    one simulated game's trace. Returns (spots, n_skipped_under_call)."""
    spots: List[dict] = []
    skipped_under = 0
    for ad in resp.trace:
        if not ad.action.startswith("PLAY "):
            continue
        card = ad.action[5:]
        view = ad.view
        if not all(c == "" for c in (view.get("current_trick") or [])):
            continue  # leads only
        if view.get("is_leaster") or view.get("alone"):
            continue
        called = view.get("called_card")
        if not called:
            continue
        ti = int(view.get("current_trick_index", 0))
        if ti > max_trick:
            continue
        if _called_suit_already_led(view):
            continue

        seat = ad.seat
        picker = view.get("picker") or 0
        partner = view.get("partner") or 0
        if seat == picker or seat == partner:
            continue

        legal_leads = [
            cf._card_of(v) for v in ad.validActionIds if cf._card_of(v) is not None
        ]

        if scan._is_secret_partner(view, PARTNER_MODE_CALLED_ACE):
            # PARTNER mirror: surfacing the called card must be legal and a
            # real choice (some alternative lead exists).
            if called in legal_leads and len(legal_leads) >= 2:
                group = "partner"
            else:
                continue
        else:
            called_opts = [c for c in legal_leads if _called_suit_fail(c, called)]
            if not called_opts or len(called_opts) == len(legal_leads):
                continue
            if view.get("called_under"):
                skipped_under += 1
                continue
            group = "agree" if _called_suit_fail(card, called) else "disagree"

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
                "cardLed": card,
                "group": group,
            }
        )
    return spots, skipped_under


def _find_cases(args) -> Dict[str, List[dict]]:
    scan.set_scan_model(args.model)
    groups: Dict[str, List[dict]] = {"agree": [], "disagree": [], "partner": []}
    skipped_under = 0
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        req = AnalyzeSimulateRequest(
            seed=seed,
            partnerMode=PARTNER_MODE_CALLED_ACE,
            deterministic=True,
            maxSteps=args.max_steps,
        )
        resp = scan.simulate_game(req)
        spots, n_under = _classify_c2_spots(resp, seed, args.max_trick)
        skipped_under += n_under
        for spot in spots:
            groups[spot["group"]].append(spot)

    print(
        f"Scanned {args.num_seeds} seeds (from {args.start_seed}) -> "
        f"{len(groups['agree'])} AGREE, {len(groups['disagree'])} DISAGREE, "
        f"{len(groups['partner'])} PARTNER-mirror C2 leads on tricks 0-{args.max_trick} "
        f"({skipped_under} under-call spots excluded)"
    )
    # Cap per group (seeded shuffle) to keep the run budget predictable.
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
    """(convention card, alternative card) by policy logit, or None if the node
    does not offer both."""
    called = spot["calledCard"]
    logits = node.leadLogits or {}
    if spot["group"] == "partner":
        if called not in logits:
            return None
        others = {c: v for c, v in logits.items() if c != called}
        if not others:
            return None
        return called, max(others, key=others.get)
    conv_pool = {c: v for c, v in logits.items() if _called_suit_fail(c, called)}
    alt_pool = {c: v for c, v in logits.items() if not _called_suit_fail(c, called)}
    if not conv_pool or not alt_pool:
        return None
    return max(conv_pool, key=conv_pool.get), max(alt_pool, key=alt_pool.get)


def analyze_case(agent, teacher, spot: dict, args, device) -> Optional[C2CaseResult]:
    seed = spot["seed"]
    target_step, seat = spot["stepIndex"], spot["seat"]
    det_rng = random.Random(0xC0FFEE ^ (seed << 8) ^ target_step)

    search_teacher = None if args.no_search else teacher
    node_game, node_mem, node, search, forced_public = cf._replay_to_node(
        agent,
        seed,
        PARTNER_MODE_CALLED_ACE,
        target_step,
        args.max_steps,
        device,
        teacher=search_teacher,
        det_rng=det_rng,
        iters=args.iters,
        rollout_depth=args.rollout_depth,
        min_visit_frac=args.min_visit_frac,
    )
    if node is None:
        print(f"  ! seed={seed} step={target_step}: node not reached; skipping")
        return None
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

    torch.manual_seed(0xA11CE ^ (seed << 4) ^ target_step)
    mc_conv = cf._mc_branch(
        agent, node_game, node_mem, seat, conv_card, args.rollouts, device
    )
    mc_alt = cf._mc_branch(
        agent, node_game, node_mem, seat, alt_card, args.rollouts, device
    )

    belief_conv = belief_alt = None
    if teacher is not None and not args.no_belief_mc:
        belief_rng = random.Random(0xBE11E ^ (seed << 6) ^ target_step)
        torch.manual_seed(0xBE11E ^ (seed << 4) ^ target_step)
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

    top_q_is_conv = None
    if search is not None:
        top_q_card = (
            search.topQAction[5:] if search.topQAction.startswith("PLAY ") else None
        )
        if spot["group"] == "partner":
            top_q_is_conv = top_q_card == spot["calledCard"]
        else:
            top_q_is_conv = top_q_card is not None and _called_suit_fail(
                top_q_card, spot["calledCard"]
            )

    return C2CaseResult(
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
        search=search,
        searchTopQIsConv=top_q_is_conv,
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


def _print_group(name: str, blurb: str, results: List[C2CaseResult]) -> None:
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
            f"  {label:<16}: Δscore {m:+.3f} (SE {se:.3f})  conv better in {pos:.0%}"
        )
    pts = [r.mcDeltaPoints for r in results]
    wins = [r.mcDeltaWin for r in results]
    m_p, se_p = _mean_se(pts)
    m_w, se_w = _mean_se(wins)
    print(f"  true-deal MC    : Δpts {m_p:+.2f} (SE {se_p:.2f})  Δwin {m_w * 100:+.1f}% (SE {se_w * 100:.1f})")
    t0 = [r for r in results if r.trickIndex == 0]
    if t0:
        m0, se0 = _mean_se([r.mcDeltaScore for r in t0])
        print(f"  trick-0 subset  : n={len(t0)}  MC Δscore {m0:+.3f} (SE {se0:.3f})")
    searched = [r for r in results if r.search is not None and r.search.ok]
    if searched:
        conv_frac = sum(1 for r in searched if r.searchTopQIsConv) / len(searched)
        agm_frac = sum(1 for r in searched if r.search.topQIsArgmax) / len(searched)
        print(
            f"  ISMCTS top@Q    : n={len(searched)}  conv {conv_frac:.0%}  "
            f"argmax {agm_frac:.0%}"
        )
    by_pos: Dict[int, List[float]] = {}
    for r in results:
        by_pos.setdefault(r.relPosFromPicker, []).append(r.mcDeltaScore)
    parts = []
    for rel in sorted(by_pos):
        m, _ = _mean_se(by_pos[rel])
        parts.append(f"picker+{rel}: {m:+.2f} (n={len(by_pos[rel])})")
    print(f"  MC Δscore by pos: {'  '.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--max-trick",
        type=int,
        default=5,
        help="Analyze eligible leads at or below this 0-based trick (default all).",
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--iters", type=int, default=512)
    parser.add_argument("--root-explore-frac", type=float, default=1.0)
    parser.add_argument("--min-visit-frac", type=float, default=0.01)
    parser.add_argument(
        "--rollout-depth",
        type=int,
        default=None,
        help="ISMCTS rollout depth in tricks (default: roll to terminal).",
    )
    parser.add_argument("--belief-worlds", type=int, default=None)
    parser.add_argument("--no-search", action="store_true", help="Skip ISMCTS.")
    parser.add_argument(
        "--no-belief-mc", action="store_true", help="Skip the belief-pool MC rung."
    )
    parser.add_argument(
        "--max-cases-per-group",
        type=int,
        default=120,
        help="Per-group case cap (seeded subsample) to bound the run budget.",
    )
    parser.add_argument("--subsample-seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = cf._device()
    scan.set_scan_model(args.model)
    agent = scan._cached_load_agent(args.model)

    teacher = None
    if not (args.no_search and args.no_belief_mc):
        from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

        cfg = ISMCTSConfig()
        cfg.iters = {k: args.iters for k in cfg.iters}
        cfg.root_explore_frac = args.root_explore_frac
        teacher = ISMCTSTeacher(agent, cfg)

    groups = _find_cases(args)

    results: Dict[str, List[C2CaseResult]] = {}
    for name in ("agree", "disagree", "partner"):
        spots = groups[name]
        print(f"\n>>> Analyzing {len(spots)} {name.upper()} case(s)")
        out: List[C2CaseResult] = []
        for i, spot in enumerate(spots):
            r = analyze_case(agent, teacher, spot, args, device)
            if r is not None:
                out.append(r)
            if (i + 1) % 20 == 0:
                print(f"    ... {i + 1}/{len(spots)}")
        results[name] = out

    _print_group(
        "AGREE (argmax = called-suit lead)",
        "sanity: Δ ≥ 0 expected",
        results["agree"],
    )
    _print_group(
        "DISAGREE (argmax ≠ called-suit lead)",
        "decision group",
        results["disagree"],
    )
    _print_group(
        "PARTNER mirror (secret partner surfaces called card)",
        "falsifier: Δscore ≤ 0 expected",
        results["partner"],
    )
    print(
        "\nInterpretation (Δ = convention − alternative, from the LEADER's team view "
        "via leaderScore; defenderPoints is always the defending team's).\n"
        "AGREE Δ<0 or PARTNER Δ>0 at 2σ would mean the measurement, not the "
        "convention, is broken. DISAGREE is the pre-registered decision cell "
        "(support: Δ > 0 at 2σ on rung 2 with sign agreement on 2b/3)."
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
                "iters": args.iters,
                "rootExploreFrac": args.root_explore_frac,
                "rolloutDepth": args.rollout_depth,
                "maxCasesPerGroup": args.max_cases_per_group,
            },
            "groups": {
                name: [asdict(r) for r in results[name]] for name in results
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
