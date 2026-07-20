#!/usr/bin/env python3
"""Counterfactual analysis of SECRET-PARTNER trump leads (convention CP, EP).

For every CP-ELIGIBLE node — the secret partner (holds the called card / JD,
not yet revealed, non-alone, non-leaster) *leads* on tricks 0..max with both a
trump and a fail lead legal — we ask: is leading trump (the partner convention)
better than the best fail lead? Δ is always (trump − fail) from the leader's
team view, so Δ > 0 supports the convention. Same rung ladder as the C1/C2
studies (``counterfactual_trump_leads``, whose primitives and CaseResult this
reuses): 1. deterministic rollout, 2. paired true-deal MC, 2b. paired
belief-pool MC, 3. ISMCTS top@Q.

Groups (pre-registered, Convention_Erosion_202607.md):

  * AGREE    — argmax already leads trump (conv = argmax). Sanity: Δ ≥ 0
               expected under the policy's own rollouts.
  * DISAGREE — argmax leads a fail (the decision group for an eroded policy).
               Δ > 0 here means the convention beats the policy's choice.
  * DEFENDER-MIRROR — falsifier: the same trump-vs-fail forcing at DEFENDER
               lead nodes (the C1 population), where trump leads are measured
               bad (C1 residual ≈ −0.13). If the machinery shows Δ > 0 here,
               the method — not the convention — is suspect.

The pooled AGREE+DISAGREE estimate (reweighted by scanned counts when groups
are capped) is the ecology's unconditional convention value: run it with
``--model`` pointed at different checkpoints to compare ecologies (each run is
self-play of that model in all 5 seats).

Usage (from repo root, rung 1 = det + true-deal MC only):

    uv run python -m sheepshead.analysis.counterfactual_partner_trump_leads \
        --model runs/league_arch_perceiver-shared-v2/checkpoints/pfsp_perceiver-shared-v2_checkpoint_2000000.pt \
        --num-seeds 800 --rollouts 50 --no-search --no-belief-mc \
        --out runs/convention_erosion_202607/cf_partner_trump_2000k.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import sheepshead.analysis.counterfactual_trump_leads as cf  # noqa: E402
import sheepshead.analysis.scan_defender_trump_leads as scan  # noqa: E402
from server.api.schemas import AnalyzeSimulateRequest  # noqa: E402
from sheepshead import TRUMP_SET  # noqa: E402

DEFAULT_MODEL = scan.DEFAULT_MODEL
GROUPS = ("agree", "disagree", "defender_mirror")


# ---------------------------------------------------------------------------
# Case detection
# ---------------------------------------------------------------------------
def _classify_cp_spots(resp, seed: int, partner_mode: int, max_trick: int) -> List[dict]:
    """CP-eligible secret-partner leads (agree/disagree) and defender-mirror
    leads in one simulated game's trace."""
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
            continue  # picker and REVEALED partner are out for both groups

        legal = [cf._card_of(v) for v in ad.validActionIds if cf._card_of(v) is not None]
        has_trump = any(c in TRUMP_SET for c in legal)
        has_fail = any(c in cf.FAIL_SET for c in legal)
        if not (has_trump and has_fail):
            continue
        led_trump = card in TRUMP_SET
        if not led_trump and card not in cf.FAIL_SET:
            continue  # UNDER token lead: neither class

        if scan._is_secret_partner(view, partner_mode):
            group = "agree" if led_trump else "disagree"
        else:
            group = "defender_mirror"

        spots.append(
            {
                "seed": seed,
                "partnerMode": partner_mode,
                "stepIndex": ad.stepIndex,
                "trickIndex": ti,
                "seat": seat,
                "seatName": ad.seatName,
                "pickerSeat": picker,
                "cardLed": card,
                "group": group,
            }
        )
    return spots


def _find_cases(args) -> tuple[Dict[str, List[dict]], Dict[str, int]]:
    scan.set_scan_model(args.model)
    groups: Dict[str, List[dict]] = {g: [] for g in GROUPS}
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        req = AnalyzeSimulateRequest(
            seed=seed,
            partnerMode=args.partner_mode,
            deterministic=True,
            maxSteps=args.max_steps,
        )
        resp = scan.simulate_game(req)
        for spot in _classify_cp_spots(resp, seed, args.partner_mode, args.max_trick):
            groups[spot["group"]].append(spot)

    scanned = {g: len(groups[g]) for g in GROUPS}
    print(
        f"Scanned {args.num_seeds} seeds (from {args.start_seed}) -> "
        f"{scanned['agree']} AGREE, {scanned['disagree']} DISAGREE partner leads, "
        f"{scanned['defender_mirror']} DEFENDER-MIRROR leads on tricks 0-{args.max_trick}"
    )
    # Cap per group (seeded shuffle) to keep the run budget predictable; the
    # scanned counts above are the reweighting denominators for pooling.
    rng = random.Random(args.subsample_seed)
    caps = {
        "agree": args.max_cases_per_group,
        "disagree": args.max_cases_per_group,
        "defender_mirror": args.max_mirror_cases,
    }
    for name, spots in groups.items():
        if len(spots) > caps[name]:
            rng.shuffle(spots)
            groups[name] = spots[: caps[name]]
            print(f"  {name}: subsampled to {caps[name]}")
    return groups, scanned


# ---------------------------------------------------------------------------
# Per-case analysis (conv = best trump lead, alt = best fail lead)
# ---------------------------------------------------------------------------
def analyze_case(agent, teacher, spot: dict, args, device) -> Optional[cf.CaseResult]:
    seed = spot["seed"]
    target_step, seat = spot["stepIndex"], spot["seat"]
    det_rng = random.Random(0xC0FFEE ^ (seed << 8) ^ target_step)

    search_teacher = None if args.no_search else teacher
    node_game, node_mem, node, search, forced_public = cf._replay_to_node(
        agent,
        seed,
        spot["partnerMode"],
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
    if node.bestTrumpCard is None or node.bestFailCard is None:
        print(f"  ! seed={seed} step={target_step}: missing a branch class; skipping")
        return None

    det_trump = cf._force_and_play(
        agent, node_game, node_mem, seat, node.bestTrumpCard, device, deterministic=True
    )
    det_fail = cf._force_and_play(
        agent, node_game, node_mem, seat, node.bestFailCard, device, deterministic=True
    )

    torch.manual_seed(0xA11CE ^ (seed << 4) ^ target_step)
    mc_trump = cf._mc_branch(
        agent, node_game, node_mem, seat, node.bestTrumpCard, args.rollouts, device
    )
    mc_fail = cf._mc_branch(
        agent, node_game, node_mem, seat, node.bestFailCard, args.rollouts, device
    )

    belief_trump = belief_fail = None
    if teacher is not None and not args.no_belief_mc:
        belief_rng = random.Random(0xBE11E ^ (seed << 6) ^ target_step)
        torch.manual_seed(0xBE11E ^ (seed << 4) ^ target_step)
        pool_k = args.belief_worlds if args.belief_worlds is not None else args.iters
        belief_trump, belief_fail = cf._belief_mc(
            agent,
            teacher,
            node_game,
            seat,
            forced_public,
            node.bestTrumpCard,
            node.bestFailCard,
            args.rollouts,
            pool_k,
            belief_rng,
            device,
        )

    return cf.CaseResult(
        seed=seed,
        partnerMode=spot["partnerMode"],
        stepIndex=target_step,
        trickIndex=spot["trickIndex"],
        seat=seat,
        seatName=spot["seatName"],
        pickerSeat=spot["pickerSeat"],
        group=spot["group"],
        hand=node.hand,
        node=node,
        detTrump=det_trump,
        detFail=det_fail,
        detDeltaPoints=det_trump.defenderPoints - det_fail.defenderPoints,
        detDeltaScore=det_trump.leaderScore - det_fail.leaderScore,
        mcTrump=mc_trump,
        mcFail=mc_fail,
        mcDeltaPoints=mc_trump.defenderPointsMean - mc_fail.defenderPointsMean,
        mcDeltaScore=mc_trump.leaderScoreMean - mc_fail.leaderScoreMean,
        mcDeltaWin=mc_trump.winRate - mc_fail.winRate,
        beliefMcTrump=belief_trump,
        beliefMcFail=belief_fail,
        beliefMcDeltaPoints=(
            belief_trump.defenderPointsMean - belief_fail.defenderPointsMean
            if belief_trump
            else None
        ),
        beliefMcDeltaScore=(
            belief_trump.leaderScoreMean - belief_fail.leaderScoreMean
            if belief_trump
            else None
        ),
        beliefMcDeltaWin=(
            belief_trump.winRate - belief_fail.winRate if belief_trump else None
        ),
        search=search,
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


def _print_group(name: str, blurb: str, results: List[cf.CaseResult]) -> None:
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
        print(f"  {label:<16}: Δscore {m:+.3f} (SE {se:.3f})  trump better in {pos:.0%}")
    m_w, se_w = _mean_se([r.mcDeltaWin for r in results])
    print(f"  true-deal MC    : Δwin {m_w * 100:+.1f}% (SE {se_w * 100:.1f})")
    t0 = [r for r in results if r.trickIndex == 0]
    if t0:
        m0, se0 = _mean_se([r.mcDeltaScore for r in t0])
        print(f"  trick-0 subset  : n={len(t0)}  MC Δscore {m0:+.3f} (SE {se0:.3f})")
    searched = [r for r in results if r.search is not None and r.search.ok]
    if searched:
        conv_frac = sum(1 for r in searched if r.search.topQIsTrump) / len(searched)
        agm_frac = sum(1 for r in searched if r.search.topQIsArgmax) / len(searched)
        print(
            f"  ISMCTS top@Q    : n={len(searched)}  trump {conv_frac:.0%}  "
            f"argmax {agm_frac:.0%}"
        )


def _pooled(results: Dict[str, List[cf.CaseResult]], scanned: Dict[str, int]) -> dict:
    """AGREE+DISAGREE pooled MC Δscore, reweighted to the scanned group mix
    (caps distort the sampled mix, so a plain mean would be biased)."""
    parts = {}
    for g in ("agree", "disagree"):
        vals = [r.mcDeltaScore for r in results[g]]
        if vals:
            m, se = _mean_se(vals)
            parts[g] = (m, se, len(vals), scanned[g])
    if not parts:
        return {}
    tot = sum(p[3] for p in parts.values())
    mean = sum(p[0] * p[3] / tot for p in parts.values())
    se = float(np.sqrt(sum((p[1] * p[3] / tot) ** 2 for p in parts.values())))
    return {"mcDeltaScore": mean, "se": se, "weights": {g: p[3] for g, p in parts.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--partner-mode", type=int, default=1, help="1=called ace, 0=JD")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--max-trick",
        type=int,
        default=2,
        help="Analyze eligible leads at or below this 0-based trick "
        "(default 0-2, the convention window; matches partner_trump_lead_probe).",
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--iters", type=int, default=512)
    parser.add_argument("--root-explore-frac", type=float, default=1.0)
    parser.add_argument("--min-visit-frac", type=float, default=0.01)
    parser.add_argument("--rollout-depth", type=int, default=None)
    parser.add_argument("--belief-worlds", type=int, default=None)
    parser.add_argument("--no-search", action="store_true", help="Skip ISMCTS.")
    parser.add_argument("--no-belief-mc", action="store_true")
    parser.add_argument("--max-cases-per-group", type=int, default=120)
    parser.add_argument(
        "--max-mirror-cases",
        type=int,
        default=60,
        help="Falsifier group cap (smaller: it only needs sign, not precision).",
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

    groups, scanned = _find_cases(args)

    results: Dict[str, List[cf.CaseResult]] = {}
    for name in GROUPS:
        spots = groups[name]
        print(f"\n>>> Analyzing {len(spots)} {name.upper()} case(s)")
        out: List[cf.CaseResult] = []
        for i, spot in enumerate(spots):
            r = analyze_case(agent, teacher, spot, args, device)
            if r is not None:
                out.append(r)
            if (i + 1) % 20 == 0:
                print(f"    ... {i + 1}/{len(spots)}")
        results[name] = out

    _print_group(
        "AGREE (argmax = trump lead as secret partner)",
        "sanity: Δ ≥ 0 expected",
        results["agree"],
    )
    _print_group(
        "DISAGREE (argmax = fail lead as secret partner)",
        "decision group for an eroded policy",
        results["disagree"],
    )
    _print_group(
        "DEFENDER-MIRROR (trump-vs-fail at defender leads)",
        "falsifier: Δ ≤ 0 expected (C1 residual ≈ −0.13)",
        results["defender_mirror"],
    )
    pooled = _pooled(results, scanned)
    if pooled:
        print(
            f"\nPOOLED partner-node convention value (scan-mix reweighted): "
            f"MC Δscore {pooled['mcDeltaScore']:+.3f} (SE {pooled['se']:.3f})"
        )
    print(
        "\nInterpretation: Δ = (best trump lead − best fail lead) from the "
        "LEADER's team view. Partner groups: Δ > 0 supports the convention in "
        "THIS model's self-play ecology. AGREE Δ<0 or DEFENDER-MIRROR Δ>0 at "
        "2σ means the measurement, not the convention, is broken."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "model": args.model,
                "partnerMode": args.partner_mode,
                "startSeed": args.start_seed,
                "numSeeds": args.num_seeds,
                "maxTrick": args.max_trick,
                "rollouts": args.rollouts,
                "iters": args.iters,
                "rootExploreFrac": args.root_explore_frac,
                "rolloutDepth": args.rollout_depth,
                "maxCasesPerGroup": args.max_cases_per_group,
                "scannedCounts": scanned,
            },
            "pooled": pooled,
            "groups": {name: [asdict(r) for r in results[name]] for name in results},
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
