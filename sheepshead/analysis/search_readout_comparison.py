#!/usr/bin/env python3
"""Search readout comparison (Search_Readout_Comparison_202607).

Question: which root READOUT should the contingency search-teacher distill —
visit counts (tau-sharpened), top@Q, regret-matching average (pi_rm), or the
Gumbel-style completed-Q target (pi_gumbel)? The June audits showed tau=1.0
counts carry ~8-10pp forced-exploration floor mass (leak injection) and that
low-frac counts are prior-dominated; tau=0.5 / frac=1.0 / top@Q are patches.
pi_rm and pi_gumbel decouple exploration from the target by construction.

At harvested lead nodes (CP-eligible secret-partner leads = re-ignition class;
defender-mirror leads = floor/control class; same scan as
``counterfactual_partner_trump_leads``), run ISMCTS under three root regimes
with a COMMON determinization RNG seed (CRN pools):

  arm A  PUCT, root_explore_frac=0.25 (training default)
             -> counts_t10, counts_t05, topq, gumbel readouts
  arm B  PUCT, root_explore_frac=1.0 (June audit / contingency recipe)
             -> topq_f100, counts_t05_f100 readouts
  arm C  RM root (root_selection="rm"), frac=0.25
             -> rm readout

  baselines: prior (masked policy over lead cards), prior argmax.

Every legal lead card gets a paired true-deal MC value v(card) = mean leader
score over R forced-lead policy rollouts from the identical node snapshot, so
the value of ANY readout policy pi is sum_a pi(a) v(a) at zero extra rollout
cost ("mixture value"), plus v(argmax pi) ("mode value" — what a tau->0
deployment realizes). Deltas are vs the prior-argmax baseline, paired per node.

Usage (from repo root):

    uv run python -m sheepshead.analysis.search_readout_comparison \
        --model runs/league_arch_perceiver-shared-v2/checkpoints/pfsp_perceiver-shared-v2_checkpoint_2000000.pt \
        --num-seeds 800 --rollouts 30 --iters 384 \
        --out runs/search_readout_202607/readout_2000k.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import sheepshead.analysis.counterfactual_trump_leads as cf  # noqa: E402
import sheepshead.analysis.counterfactual_partner_trump_leads as cfp  # noqa: E402
import sheepshead.analysis.scan_defender_trump_leads as scan  # noqa: E402
from sheepshead import ACTION_IDS, TRUMP_SET  # noqa: E402

GROUPS = cfp.GROUPS  # ("agree", "disagree", "defender_mirror")

# Root regimes; all run tau_target=1.0 so ``pi`` is the plain count
# distribution and tau-sharpening happens offline in the readout layer.
SEARCH_ARMS = {
    "A_puct_f025": dict(root_selection="puct", root_explore_frac=0.25),
    "B_puct_f100": dict(root_selection="puct", root_explore_frac=1.0),
    "C_rm_f025": dict(root_selection="rm", root_explore_frac=0.25),
}

# readout name -> (arm, extractor kind)
READOUTS = {
    "counts_t10": ("A_puct_f025", "counts", 1.0),
    "counts_t05": ("A_puct_f025", "counts", 0.5),
    "topq": ("A_puct_f025", "topq", None),
    "gumbel": ("A_puct_f025", "pi_gumbel", None),
    "counts_t05_f100": ("B_puct_f100", "counts", 0.5),
    "topq_f100": ("B_puct_f100", "topq", None),
    "rm": ("C_rm_f025", "pi_rm", None),
}


def _restrict(vec_or_map, lead_aids: List[int]) -> tuple[Dict[int, float], float]:
    """Restrict a full-action-size vector (or aid->mass map) to the lead-card
    actions and renormalize. Returns (aid->prob, dropped_mass_before_renorm)."""
    if isinstance(vec_or_map, dict):
        raw = {aid: float(vec_or_map.get(aid, 0.0)) for aid in lead_aids}
        total_all = sum(float(v) for v in vec_or_map.values())
    else:
        raw = {aid: float(vec_or_map[aid - 1]) for aid in lead_aids}
        total_all = float(np.sum(vec_or_map))
    total = sum(raw.values())
    dropped = max(total_all - total, 0.0)
    if total <= 0.0:
        n = len(lead_aids)
        return {aid: 1.0 / n for aid in lead_aids}, dropped
    return {aid: v / total for aid, v in raw.items()}, dropped


def _extract_readout(res: dict, kind: str, tau, lead_aids, min_visit_frac):
    if kind == "counts":
        n = {aid: float(res["root_n"].get(aid, 0.0)) for aid in lead_aids}
        total = sum(n.values())
        if total <= 0.0:
            return None, 0.0
        powered = {aid: v ** (1.0 / tau) for aid, v in n.items()}
        z = sum(powered.values())
        return {aid: v / z for aid, v in powered.items()}, 0.0
    if kind == "topq":
        total_n = sum(float(v) for v in res["root_n"].values())
        eligible = [
            aid
            for aid in lead_aids
            if float(res["root_n"].get(aid, 0.0)) >= min_visit_frac * total_n
        ]
        if not eligible:
            return None, 0.0
        best = max(eligible, key=lambda a: float(res["root_q"].get(a, -np.inf)))
        return {aid: (1.0 if aid == best else 0.0) for aid in lead_aids}, 0.0
    # pi_gumbel / pi_rm vectors
    vec = res.get(kind)
    if vec is None:
        return None, 0.0
    return _restrict(np.asarray(vec, dtype=np.float64), lead_aids)


def _entropy(pi: Dict[int, float]) -> float:
    return float(-sum(p * math.log(p) for p in pi.values() if p > 0.0))


def _policy_metrics(pi: Dict[int, float], vals: Dict[int, dict], aid_card) -> dict:
    mode = max(pi, key=pi.get)
    return {
        "valueMixture": float(sum(pi[a] * vals[a]["scoreMean"] for a in pi)),
        "valueMode": float(vals[mode]["scoreMean"]),
        "modeCard": aid_card[mode],
        "trumpMass": float(sum(p for a, p in pi.items() if aid_card[a] in TRUMP_SET)),
        "modeIsTrump": aid_card[mode] in TRUMP_SET,
        "entropy": _entropy(pi),
        "probs": {aid_card[a]: round(p, 6) for a, p in pi.items()},
    }


def analyze_node(agent, spot: dict, args, device) -> Optional[dict]:
    seed, step, seat = spot["seed"], spot["stepIndex"], spot["seat"]
    node_game, node_mem, node, _s, forced_public = cf._replay_to_node(
        agent, seed, spot["partnerMode"], step, args.max_steps, device, teacher=None
    )
    if node is None:
        print(f"  ! seed={seed} step={step}: node not reached; skipping")
        return None
    if node.argmaxCard != spot["cardLed"]:
        print(f"  ! seed={seed} step={step}: argmax drifted; skipping")
        return None
    lead_cards = sorted(node.leadLogits.keys())
    if len(lead_cards) < 2:
        return None
    lead_aids = [ACTION_IDS[f"PLAY {c}"] for c in lead_cards]
    aid_card = dict(zip(lead_aids, lead_cards))

    # --- per-action paired true-deal MC values (shared torch seed sequence,
    # identical node snapshot; deterministic card order) ---
    torch.manual_seed(0xA11CE ^ (seed << 4) ^ step)
    vals: Dict[int, dict] = {}
    for card, aid in zip(lead_cards, lead_aids):
        mc = cf._mc_branch(agent, node_game, node_mem, seat, card, args.rollouts, device)
        vals[aid] = {
            "card": card,
            "scoreMean": mc.leaderScoreMean,
            "scoreSd": mc.leaderScoreSd,
            "winRate": mc.winRate,
        }

    # --- three search arms, CRN determinization seed ---
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    rng_seed = 0xC0FFEE ^ (seed << 8) ^ step
    d_rollout = 6 - int(node.trickIndex)  # roll to terminal
    arm_res: Dict[str, dict] = {}
    for arm, overrides in SEARCH_ARMS.items():
        cfg = ISMCTSConfig()
        cfg.iters = {k: args.iters for k in cfg.iters}
        cfg.tau_target = 1.0
        cfg.rm_gamma = args.rm_gamma
        for k, v in overrides.items():
            setattr(cfg, k, v)
        teacher = ISMCTSTeacher(agent, cfg)
        torch.manual_seed(rng_seed & 0x7FFFFFFF)
        res = teacher.search(
            node_game,
            seat,
            list(forced_public),
            random.Random(rng_seed),
            d_rollout=d_rollout,
            seat_policies=None,
        )
        arm_res[arm] = res
    if not all(r["ok"] for r in arm_res.values()):
        print(
            f"  ! seed={seed} step={step}: ESS gate failed "
            f"({ {a: round(r['ess'], 1) for a, r in arm_res.items()} }); skipping"
        )
        return None

    # --- readouts + baselines ---
    policies: Dict[str, dict] = {}
    for name, (arm, kind, tau) in READOUTS.items():
        pi, dropped = _extract_readout(
            arm_res[arm], kind, tau, lead_aids, args.min_visit_frac
        )
        if pi is None:
            continue
        m = _policy_metrics(pi, vals, aid_card)
        m["nonLeadMassDropped"] = round(dropped, 6)
        policies[name] = m
    logits = np.array([node.leadLogits[c] for c in lead_cards], dtype=np.float64)
    z = np.exp(logits - logits.max())
    z /= z.sum()
    policies["prior"] = _policy_metrics(dict(zip(lead_aids, z)), vals, aid_card)
    argmax_aid = ACTION_IDS[f"PLAY {node.argmaxCard}"]
    policies["prior_argmax"] = _policy_metrics(
        {aid: (1.0 if aid == argmax_aid else 0.0) for aid in lead_aids}, vals, aid_card
    )

    # search-side diagnostics from arm A
    res_a = arm_res["A_puct_f025"]
    q_lead = {aid: float(res_a["root_q"].get(aid, 0.0)) for aid in lead_aids}
    q_trump = [q for a, q in q_lead.items() if aid_card[a] in TRUMP_SET]
    q_fail = [q for a, q in q_lead.items() if aid_card[a] not in TRUMP_SET]
    return {
        "seed": seed,
        "stepIndex": step,
        "group": spot["group"],
        "trickIndex": node.trickIndex,
        "seat": seat,
        "hand": node.hand,
        "leadCards": lead_cards,
        "actionValues": {aid_card[a]: v for a, v in vals.items()},
        "policies": policies,
        "ess": {arm: round(r["ess"], 2) for arm, r in arm_res.items()},
        "rootQGapTrumpMinusFail": (
            (max(q_trump) - max(q_fail)) if q_trump and q_fail else None
        ),
    }


def _mean_se(xs: List[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se


def summarize(records: List[dict]) -> dict:
    """Per group per readout: paired Δvalue vs prior_argmax (mixture and mode),
    trump mass, mode-trump rate, entropy."""
    out: dict = {}
    names = sorted({n for r in records for n in r["policies"]})
    for group in GROUPS:
        recs = [r for r in records if r["group"] == group]
        if not recs:
            continue
        gsum: dict = {"n": len(recs)}
        for name in names:
            rows = [r for r in recs if name in r["policies"]]
            if not rows:
                continue
            base = [r["policies"]["prior_argmax"]["valueMode"] for r in rows]
            mix = [r["policies"][name]["valueMixture"] for r in rows]
            mode = [r["policies"][name]["valueMode"] for r in rows]
            d_mix, se_mix = _mean_se([m - b for m, b in zip(mix, base)])
            d_mode, se_mode = _mean_se([m - b for m, b in zip(mode, base)])
            tm, tm_se = _mean_se([r["policies"][name]["trumpMass"] for r in rows])
            ent, _ = _mean_se([r["policies"][name]["entropy"] for r in rows])
            gsum[name] = {
                "n": len(rows),
                "dValueMixture": round(d_mix, 4),
                "dValueMixtureSE": round(se_mix, 4),
                "dValueMode": round(d_mode, 4),
                "dValueModeSE": round(se_mode, 4),
                "trumpMass": round(tm, 4),
                "trumpMassSE": round(tm_se, 4),
                "modeTrumpRate": round(
                    float(
                        np.mean([r["policies"][name]["modeIsTrump"] for r in rows])
                    ),
                    4,
                ),
                "entropy": round(ent, 4),
            }
        out[group] = gsum
    return out


def _print_summary(summary: dict) -> None:
    for group, gsum in summary.items():
        print(f"\n=== {group.upper()} (n={gsum['n']}) ===")
        print(
            f"  {'readout':<16} {'Δv(mix)':>9} {'SE':>6} {'Δv(mode)':>9} {'SE':>6}"
            f" {'trumpMass':>10} {'mode@T':>7} {'H':>6}"
        )
        for name, s in gsum.items():
            if name == "n":
                continue
            print(
                f"  {name:<16} {s['dValueMixture']:>+9.3f} {s['dValueMixtureSE']:>6.3f}"
                f" {s['dValueMode']:>+9.3f} {s['dValueModeSE']:>6.3f}"
                f" {s['trumpMass']:>10.3f} {s['modeTrumpRate']:>7.2f}"
                f" {s['entropy']:>6.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=cfp.DEFAULT_MODEL)
    parser.add_argument("--partner-mode", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-trick", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=30)
    parser.add_argument("--iters", type=int, default=384)
    parser.add_argument("--min-visit-frac", type=float, default=0.01)
    parser.add_argument("--rm-gamma", type=float, default=0.10)
    parser.add_argument("--max-cases-per-group", type=int, default=80)
    parser.add_argument("--max-mirror-cases", type=int, default=60)
    parser.add_argument("--subsample-seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = cf._device()
    scan.set_scan_model(args.model)
    agent = scan._cached_load_agent(args.model)

    groups, scanned = cfp._find_cases(args)
    records: List[dict] = []
    for name in GROUPS:
        spots = groups[name]
        print(f"\n>>> Analyzing {len(spots)} {name.upper()} node(s)")
        for i, spot in enumerate(spots):
            r = analyze_node(agent, spot, args, device)
            if r is not None:
                records.append(r)
            if (i + 1) % 10 == 0:
                print(f"    ... {i + 1}/{len(spots)}")

    summary = summarize(records)
    _print_summary(summary)
    print(
        "\nInterpretation: Δv = paired (readout − prior-argmax) mean leader "
        "score at the node; 'mix' = expectation under the soft target, 'mode' "
        "= committing to its argmax. Floor test: trumpMass at DEFENDER_MIRROR "
        "(counts_t10 is the known ~8-10pp offender). Re-ignition test: "
        "trumpMass / mode@T at partner nodes for an eroded checkpoint."
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
                "minVisitFrac": args.min_visit_frac,
                "rmGamma": args.rm_gamma,
                "maxCasesPerGroup": args.max_cases_per_group,
                "maxMirrorCases": args.max_mirror_cases,
                "scannedCounts": scanned,
                "searchArms": {k: dict(v) for k, v in SEARCH_ARMS.items()},
                "readouts": {k: [v[0], v[1], v[2]] for k, v in READOUTS.items()},
            },
            "summary": summary,
            "records": records,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
