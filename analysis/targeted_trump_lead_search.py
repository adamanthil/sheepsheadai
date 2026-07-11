#!/usr/bin/env python3
"""Targeted deploy-search assessment on defender trump-lead nodes (tricks 0-1).

The all-deals pilot (``tune_deploy_search.py``) found play search score-neutral on a
random deal distribution -- because the policy is strong and its suboptimality is rare
and cheap, so it dilutes into the noise. This script removes the dilution: it conditions
*exactly* on the suspect nodes -- the defender trump-leads on tricks 0-1 that the
scanner flags -- and asks the realized-strength question there:

    At a leak node, does deploy ISMCTS search change the policy's trump lead, and does
    that change improve the defender's realized game score on the actual deal?

Method (paired, per spot)
-------------------------
For each TRUMP-PREF defender lead (policy argmax leads a trump with a fail available):
  1. replay deterministically to the node (reproduces the ``/analyze`` state);
  2. run the deploy search at the node (top@Q by default) -> the search's chosen lead;
  3. play the hand to terminal twice from the identical snapshot -- once forcing the
     POLICY's lead, once forcing the SEARCH's lead -- with greedy argmax for all seats;
  4. Δ = score(search lead) - score(policy lead) for the defender, on the same deal.

Because both branches share the deal and differ only in the lead, the paired Δ is the
clean causal value of letting search override the lead. When search keeps the trump
(no change) Δ = 0 by construction, so the all-spots mean = (fix rate) x (mean Δ | fixed)
= the realized deploy value of search at the leak.

A FAIL-PREF control (policy already leads fail) is run identically as a falsifier: search
should keep fail (Δ ≈ 0); if it instead flips to trump and Δ < 0, the measurement is sound.

The search uses the DEPLOY config (low root_explore_frac, d_rollout=2 by default -- the
pilot winner), and ``--fracs`` is swept because frac is what decides whether search even
overrides the (sharp) policy prior at these nodes. NOTE: tricks 0-1 are where the critic
is weakest, so ``--depths`` also accepts ``term`` (roll to terminal, ``6 - trick``) to
test trading the early-game critic for a full rollout at exactly these nodes.

Requires the critic-load fix in ppo.py (legacy checkpoints route the value head through
critic_adapter); a random value_trunk would make the search bootstrap meaningless here.

Usage (from repo root):

    uv run python analysis/targeted_trump_lead_search.py \
        --num-seeds 3200 --partner-mode 1 \
        --fracs 0.1,0.25,0.5,1.0 --iters 384 --depths 2 \
        --out runs/targeted_trump_lead_search.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import analysis.counterfactual_trump_leads as cf  # noqa: E402
import analysis.scan_defender_trump_leads as scan  # noqa: E402

TRUMP_SET = cf.TRUMP_SET
FAIL_SET = cf.FAIL_SET
DEFAULT_MODEL = scan.DEFAULT_MODEL


@dataclass
class SpotResult:
    seed: int
    trickIndex: int
    seat: int
    handTrumpCount: int
    policyCard: str
    searchCard: str
    changed: bool
    toFail: bool
    searchOk: bool
    ess: float
    dScore: float
    dPoints: float
    dWin: float


@dataclass
class ConfigAgg:
    group: str
    select: str
    rootExploreFrac: float
    dRollout: str
    iters: int
    n: int
    nEssOk: int
    fixRate: float       # fraction where search changed the lead
    toFailRate: float    # fraction where search changed it to a fail
    dScoreAll: float     # realized deploy value over ALL spots (zeros where unchanged)
    dScoreAllSE: float
    dScoreChanged: float  # conditional on search actually changing the lead
    dScoreChangedSE: float
    nChanged: int
    dPointsAll: float
    dWinAll: float


def _se(vals: np.ndarray) -> float:
    return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


def _parse_depth(tok: str) -> Optional[int]:
    """``term`` / ``-1`` -> None (roll to terminal, 6-trick); else the int d_rollout."""
    t = tok.strip().lower()
    if t in ("term", "terminal", "-1"):
        return None
    return int(t)


def _eval_spots(
    agent, teacher, spots, *, group, frac, depth_tok, iters, max_steps,
    min_visit_frac, select, device,
) -> tuple[ConfigAgg, List[SpotResult]]:
    teacher.config.root_explore_frac = frac
    teacher.config.iters = {k: iters for k in teacher.config.iters}
    depth_req = _parse_depth(depth_tok)

    rows: List[SpotResult] = []
    for spot in spots:
        seed, pm = spot["seed"], spot["partnerMode"]
        target_step, seat = spot["stepIndex"], spot["seat"]
        depth = depth_req if depth_req is not None else 6 - spot["trickIndex"]
        node_game, node_mem, node, search, _ = cf._replay_to_node(
            agent, seed, pm, target_step, max_steps, device,
            teacher=teacher,
            det_rng=random.Random(seed * 7919 + target_step),
            iters=iters,
            rollout_depth=depth,
            min_visit_frac=min_visit_frac,
        )
        if node is None or node.argmaxCard != spot["cardLed"] or search is None:
            continue  # node not reached / non-reproducing / no search

        policy_card = node.argmaxCard
        search_card = (
            (search.topQAction if select == "q" else search.topAction)[5:]
        )
        base = cf._force_and_play(
            agent, node_game, node_mem, seat, policy_card, device, deterministic=True
        )
        treat = cf._force_and_play(
            agent, node_game, node_mem, seat, search_card, device, deterministic=True
        )
        changed = search_card != policy_card
        rows.append(
            SpotResult(
                seed=seed,
                trickIndex=spot["trickIndex"],
                seat=seat,
                handTrumpCount=node.handTrumpCount,
                policyCard=policy_card,
                searchCard=search_card,
                changed=changed,
                toFail=changed and (search_card in FAIL_SET),
                searchOk=bool(search.ok),
                ess=round(float(search.ess), 1),
                dScore=float(treat.leaderScore - base.leaderScore),
                dPoints=float(treat.defenderPoints - base.defenderPoints),
                dWin=float(treat.win - base.win),
            )
        )

    d_all = np.array([r.dScore for r in rows], dtype=float)
    chg = [r for r in rows if r.changed]
    d_chg = np.array([r.dScore for r in chg], dtype=float)
    agg = ConfigAgg(
        group=group,
        select=select,
        rootExploreFrac=frac,
        dRollout=depth_tok,
        iters=iters,
        n=len(rows),
        nEssOk=sum(1 for r in rows if r.searchOk),
        fixRate=len(chg) / len(rows) if rows else 0.0,
        toFailRate=(sum(1 for r in rows if r.toFail) / len(rows)) if rows else 0.0,
        dScoreAll=float(d_all.mean()) if len(d_all) else 0.0,
        dScoreAllSE=_se(d_all),
        dScoreChanged=float(d_chg.mean()) if len(d_chg) else 0.0,
        dScoreChangedSE=_se(d_chg),
        nChanged=len(chg),
        dPointsAll=float(np.mean([r.dPoints for r in rows])) if rows else 0.0,
        dWinAll=float(np.mean([r.dWin for r in rows])) if rows else 0.0,
    )
    return agg, rows


def _fmt(agg: ConfigAgg) -> str:
    sig = agg.dScoreAll / agg.dScoreAllSE if agg.dScoreAllSE > 0 else 0.0
    return (
        f"  [{agg.group:<10s}] {agg.select:>6s} f={agg.rootExploreFrac:<4g} dR={agg.dRollout:<4s} "
        f"it={agg.iters:<5d} n={agg.n:3d} (ESSok {agg.nEssOk:3d}) | "
        f"fix {agg.fixRate * 100:4.0f}% (→fail {agg.toFailRate * 100:4.0f}%)  | "
        f"Δscore_all {agg.dScoreAll:+5.2f} ± {agg.dScoreAllSE:.2f} ({sig:+.1f}σ)  "
        f"Δscore|chg {agg.dScoreChanged:+5.2f} (n={agg.nChanged:2d})  "
        f"Δpts {agg.dPointsAll:+4.1f}  Δwin {agg.dWinAll * 100:+4.0f}%"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-seeds", type=int, default=3200)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--partner-mode", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-trick", type=int, default=1, help="0-indexed; 1 = tricks 0 and 1")
    p.add_argument("--fracs", default="0.1,0.25,0.5,1.0")
    p.add_argument("--iters", type=int, default=384)
    p.add_argument("--depths", default="2", help="d_rollout list; 'term' = roll to terminal")
    p.add_argument("--select", default="q", help="comma list of: q, visits")
    p.add_argument("--min-visit-frac", type=float, default=0.01)
    p.add_argument("--control-ratio", type=float, default=1.0)
    p.add_argument("--control-seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = cf._device()
    agent = scan._cached_load_agent(args.model)
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    cfg = ISMCTSConfig()
    cfg.batch_size = args.batch_size
    teacher = ISMCTSTeacher(agent, cfg)

    trump_spots, fail_spots = cf._find_cases(args)
    groups = [("TRUMP-PREF", trump_spots)]
    if args.control_ratio > 0 and fail_spots:
        groups.append(("FAIL-PREF", fail_spots))

    fracs = [float(x) for x in args.fracs.split(",")]
    depths = [d for d in args.depths.split(",")]
    selects = [s.strip() for s in args.select.split(",")]
    print(
        f"\nTargeted leak assessment: tricks 0-{args.max_trick}, select top@{selects}, "
        f"iters {args.iters}.  Δ = defender score(search lead) - score(policy lead), paired.\n"
    )

    out_aggs: List[ConfigAgg] = []
    out_rows: dict = {}
    for group, spots in groups:
        for select in selects:
            for depth_tok in depths:
                for frac in fracs:
                    agg, rows = _eval_spots(
                        agent, teacher, spots,
                        group=group, frac=frac, depth_tok=depth_tok, iters=args.iters,
                        max_steps=args.max_steps, min_visit_frac=args.min_visit_frac,
                        select=select, device=device,
                    )
                    print(_fmt(agg))
                    out_aggs.append(agg)
                    out_rows[f"{group}|{select}|f{frac}|d{depth_tok}"] = [
                        asdict(r) for r in rows
                    ]
        print()

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "numSeeds": args.num_seeds,
                    "startSeed": args.start_seed,
                    "partnerMode": args.partner_mode,
                    "maxTrick": args.max_trick,
                    "select": args.select,
                    "iters": args.iters,
                    "aggregates": [asdict(a) for a in out_aggs],
                    "spots": out_rows,
                },
                indent=2,
            )
        )
        print(f"Wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
