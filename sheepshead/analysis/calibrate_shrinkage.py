#!/usr/bin/env python3
"""Shrinkage calibration gate for the CE search teacher (CE_Teacher_Design
§1.2 / §7 step 5).

Replays the archived §12.8 deflead gating study's per-node 1024/1 replicate
root-Q tables through the PRODUCTION target builder
(``pfsp_runtime.build_ce_search_target``) to (a) calibrate the global
replicate-noise constant ``SearchConfig.shrink_s2_global`` from measured
per-action replicate variance, and (b) verify the gate criteria this data
supports:

* **Abstention at noise**: nodes whose committee scatter is within the
  noise model must shrink to w = 0 (flat target = expert prior; the §12.4
  "fight without stable winner" set must produce no teaching pressure).
* **Split-committee sign stability** (§1.2 criterion c): two DISJOINT
  3-replicate committees drawn from the same node must agree on the tilt
  argmax wherever BOTH stay material — a surviving tilt whose direction
  flips across draws is exactly the non-convergent exact-card scatter the
  §12 program died on.
* **Direction at resolved nodes**: where the study's heavy reference
  (4096/terminal, self-agreeing replicates) picked a card, a MATERIAL cheap
  target should tilt the same way.

The archived reps carry root_q only, so the builder is fed uniform priors
and equal visit counts: the shrink factor w depends on neither, and the
tilt argmax under a uniform prior is the pooled-Q argmax — the quantities
under test are exact.

Estimator lineage: James-Stein shrinkage (James & Stein, 4th Berkeley
Symposium, 1961), positive-part variant (Baranchik, Stanford TR 51, 1964),
empirical-Bayes/hierarchical variance treatment as applied here (Efron &
Morris, "Data Analysis Using Stein's Estimator and Its Generalizations,"
JASA 70(350), 1975).

The fat/nopoint EV-wash and called-suit directionality criteria from §1.2
have no archived committee draws (the §12.15 EV studies recorded belief-MC
deltas, not committee Q); they fall to the attempt-11 pre-launch checklist.

Usage:
  uv run python -m sheepshead.analysis.calibrate_shrinkage \
      [--study runs/league_retention_pg_teacher/deflead_gating_study.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sheepshead.training.config import SearchConfig
from sheepshead.training.pfsp_runtime import build_ce_search_target

BUDGET = "1024/1"  # the teacher budget's replicate tables
GUMBEL_C_VISIT = 50.0
GUMBEL_C_SCALE = 0.1


def _synthetic_replicates(rep_qtabs: list[dict]) -> list[dict]:
    """Wrap raw per-replicate root-Q dicts in the SearchResult shape the
    builder consumes (uniform prior / equal visits; see module docstring)."""
    out = []
    for qt in rep_qtabs:
        acts = sorted(int(a) for a in qt)
        out.append(
            {
                "ok": True,
                "root_q": {a: float(qt[str(a)]) for a in acts},
                "root_n": {a: 1.0 for a in acts},
                "root_prior": {a: 1.0 / len(acts) for a in acts},
            }
        )
    return out


def _target_argmax(replicates, valid, s2_global, nu):
    built = build_ce_search_target(
        replicates,
        valid,
        shrink_nu=nu,
        shrink_s2_global=s2_global,
        gumbel_c_visit=GUMBEL_C_VISIT,
        gumbel_c_scale=GUMBEL_C_SCALE,
    )
    if built is None:
        return None, None
    target, info = built
    return sorted(valid)[int(np.argmax(target))], info["w"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--study",
        default="runs/league_retention_pg_teacher/deflead_gating_study.json",
    )
    ap.add_argument("--nu", type=float, default=SearchConfig().shrink_nu)
    args = ap.parse_args()

    with open(args.study) as f:
        study = json.load(f)
    rows = [r for r in study["rows"] if BUDGET in r.get("configs", {})]
    print(f"{len(rows)} nodes with {BUDGET} replicate tables")

    # ---- s2_global calibration: per-action replicate variance, pooled ----
    per_action_vars = []
    for r in rows:
        reps = [x["rootQ"] for x in r["configs"][BUDGET]["reps"] if x.get("ok")]
        if len(reps) < 2:
            continue
        for a in reps[0]:
            obs = [rep[a] for rep in reps if a in rep]
            if len(obs) >= 2:
                per_action_vars.append(float(np.var(obs, ddof=1)))
    v = np.array(per_action_vars)
    print(
        f"per-action replicate variance over {len(v)} action cells: "
        f"mean {v.mean():.3e}  median {np.median(v):.3e}  "
        f"p90 {np.quantile(v, 0.9):.3e}  (sd at mean: {np.sqrt(v.mean()):.4f} Q)"
    )
    s2_global = float(v.mean())
    print(
        f"-> calibrated shrink_s2_global = {s2_global:.3e} "
        f"(config default {SearchConfig().shrink_s2_global:.3e})"
    )

    # ---- gate criteria on the production builder ----
    flat = material = 0
    stable = unstable = both_material = 0
    ref_checked = ref_agree = 0
    by_cell: dict[str, list] = {}
    for r in rows:
        reps_raw = [x["rootQ"] for x in r["configs"][BUDGET]["reps"] if x.get("ok")]
        if len(reps_raw) < 3:
            continue
        valid = sorted(int(a) for a in reps_raw[0])
        first3 = _synthetic_replicates(reps_raw[:3])
        argmax_a, w_a = _target_argmax(first3, valid, s2_global, args.nu)
        by_cell.setdefault(r["cell"], []).append(w_a or 0.0)
        if w_a == 0.0:
            flat += 1
        elif w_a is not None:
            material += 1
        # Split-committee stability (criterion c): disjoint 3-rep draws.
        if len(reps_raw) >= 6:
            argmax_b, w_b = _target_argmax(
                _synthetic_replicates(reps_raw[3:6]), valid, s2_global, args.nu
            )
            if w_a and w_b:
                both_material += 1
                if argmax_a == argmax_b:
                    stable += 1
                else:
                    unstable += 1
        # Direction vs the self-agreeing heavy reference.
        heavy = r["configs"].get("4096/term")
        if heavy and w_a:
            rep_argmax = heavy.get("repArgmax") or []
            if len(set(rep_argmax)) == 1 and rep_argmax:
                ref_checked += 1
                if argmax_a == rep_argmax[0]:
                    ref_agree += 1

    n = flat + material
    print(
        f"\ncommittee-of-3 targets: {n} nodes -> {flat} flat (w=0, "
        f"{100 * flat / max(n, 1):.0f}%), {material} material"
    )
    for cell, ws in sorted(by_cell.items()):
        ws = np.array(ws)
        print(
            f"  {cell}: {len(ws)} nodes, flat {100 * (ws == 0).mean():.0f}%, "
            f"mean w {ws.mean():.2f}"
        )
    print(
        f"split-committee tilt-argmax stability at both-material nodes: "
        f"{stable}/{both_material} agree"
        + (f" ({unstable} flips)" if both_material else " (none both-material)")
    )
    print(
        f"direction vs self-agreeing 4096/term reference at material nodes: "
        f"{ref_agree}/{ref_checked} agree"
    )


if __name__ == "__main__":
    main()
