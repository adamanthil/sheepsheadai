#!/usr/bin/env python3
"""Mechanical statistics report for an architecture-ablation matrix.

Turns an experiment directory (produced by analysis/run_ablation_matrix.py
+ analysis/aggregate_ablation.py) into the numbers the notebook's decision
criteria need — no ad-hoc analysis required. Reads <out-dir>/results.csv
and <out-dir>/learning_curves.csv and prints, as markdown:

  1. per-arch endpoint means +/- seed-std (PANEL-A called/jd/both, scripted
     probe, wall-clock, eps/s);
  2. each arch's delta vs --baseline on PANEL-A both-modes mean, with a
     seed-level SE (std/sqrt(n) combined) — differences > ~2 SE and > the
     0.07 PANEL-A MDE are "real";
  3. the last-window slope of edge_selfplay100k per arch ("still climbing"
     means the endpoint UNDERSTATES the arch; a flat arch that ranks well
     may be an early-plateau/low-ceiling profile — see the 100k onehot-ff
     lesson in notebooks/Architecture_Ablation_202607.md).

Usage:
  PYTHONPATH=. .venv/bin/python analysis/ablation_report.py \
      --out-dir runs/size_sweep_202607 --baseline full \
      [--extra-results runs/ablation_202607/results.csv]

--extra-results merges rows from another matrix's results.csv (e.g. to
compare size-sweep variants against the original full runs at the same
episode count). Rows are matched on (arch, seed); the row with the LARGER
final_episode wins.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import statistics as st
from collections import defaultdict


def _f(v):
    try:
        return float(v)
    except TypeError, ValueError:
        return None


def read_results(path: str) -> dict:
    rows = {}
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        key = (r["arch"], str(r["seed"]))
        old = rows.get(key)
        if old is None or (_f(r.get("final_episode")) or 0) > (
            _f(old.get("final_episode")) or 0
        ):
            rows[key] = r
    return rows


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ablation matrix statistics report")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline", default="full")
    ap.add_argument("--extra-results", default=None)
    ap.add_argument("--slope-window", type=int, default=20000)
    args = ap.parse_args()

    rows = read_results(os.path.join(args.out_dir, "results.csv"))
    if args.extra_results:
        for key, r in read_results(args.extra_results).items():
            old = rows.get(key)
            if old is None or (_f(r.get("final_episode")) or 0) > (
                _f(old.get("final_episode")) or 0
            ):
                rows[key] = r
    if not rows:
        print(f"no results found under {args.out_dir}")
        return 1

    archs = sorted({a for a, _ in rows}, key=lambda a: (a != args.baseline, a))
    by_arch = defaultdict(list)
    for (a, _), r in rows.items():
        by_arch[a].append(r)

    def panel_both(r):
        c, j = _f(r.get("panel_a_called")), _f(r.get("panel_a_jd"))
        return (c + j) / 2 if c is not None and j is not None else None

    # ---- 1. endpoint means ------------------------------------------------
    print("### Per-arch endpoint means (± seed-std)\n")
    print(
        "| arch | n seeds | final ep | PANEL-A called | PANEL-A jd | "
        "PANEL-A both | scripted edge | train h | eps/s |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    both_by_arch = {}
    for a in archs:
        rs = by_arch[a]
        eps = sorted({int(_f(r.get("final_episode")) or 0) for r in rs})
        mc, sc = mean_std([_f(r.get("panel_a_called")) for r in rs])
        mj, sj = mean_std([_f(r.get("panel_a_jd")) for r in rs])
        both = [panel_both(r) for r in rs]
        mb, _ = mean_std(both)
        both_by_arch[a] = [b for b in both if b is not None]
        msc, ssc = mean_std([_f(r.get("probe_scripted_edge")) for r in rs])
        mth, _ = mean_std([_f(r.get("train_wall_h")) for r in rs])
        mes, _ = mean_std([_f(r.get("eps_per_s")) for r in rs])

        def fmt(m, s):
            return f"{m:+.3f} ± {s:.3f}" if m is not None else "—"

        print(
            f"| {a} | {len(rs)} | {'/'.join(map(str, eps))} | {fmt(mc, sc)} | "
            f"{fmt(mj, sj)} | {f'{mb:+.3f}' if mb is not None else '—'} | "
            f"{fmt(msc, ssc)} | {f'{mth:.1f}' if mth else '—'} | "
            f"{f'{mes:.1f}' if mes else '—'} |"
        )

    # ---- 2. deltas vs baseline -------------------------------------------
    base = both_by_arch.get(args.baseline, [])
    if base:
        print(f"\n### PANEL-A (both modes) delta vs `{args.baseline}`\n")
        print("| arch | delta | seed-level SE | > 2 SE? | > 0.07 MDE? |")
        print("|---|---|---|---|---|")
        bm, bs = st.mean(base), (st.stdev(base) if len(base) > 1 else 0.0)
        for a in archs:
            if a == args.baseline or not both_by_arch.get(a):
                continue
            xs = both_by_arch[a]
            d = st.mean(xs) - bm
            se = (
                (st.stdev(xs) ** 2 / len(xs) if len(xs) > 1 else 0.0)
                + bs**2 / len(base)
            ) ** 0.5
            print(
                f"| {a} | {d:+.3f} | {se:.3f} | "
                f"{'YES' if se and abs(d) > 2 * se else 'no'} | "
                f"{'YES' if abs(d) > 0.07 else 'no'} |"
            )

    # ---- 3. last-window slopes --------------------------------------------
    lc_path = os.path.join(args.out_dir, "learning_curves.csv")
    if os.path.exists(lc_path):
        by = defaultdict(dict)
        for r in csv.DictReader(open(lc_path)):
            v = _f(r.get("edge_selfplay100k"))
            if v is not None:
                by[(r["arch"], r["seed"])][int(r["episode"])] = (
                    v,
                    _f(r.get("se_selfplay100k")) or 0.0,
                )
        print(
            f"\n### Last-{args.slope_window // 1000}k slope "
            "(edge vs selfplay-100k; positive > 1 SE = still climbing, "
            "endpoint understates the arch)\n"
        )
        print("| arch | slope | paired SE | still climbing? |")
        print("|---|---|---|---|")
        for a in archs:
            deltas, ses = [], []
            for (aa, s), d in by.items():
                if aa != a or not d:
                    continue
                last = max(d)
                prev = last - args.slope_window
                if prev in d:
                    deltas.append(d[last][0] - d[prev][0])
                    ses.append((d[last][1] ** 2 + d[prev][1] ** 2) ** 0.5)
            if not deltas:
                print(f"| {a} | — | — | — |")
                continue
            m = st.mean(deltas)
            se = st.mean(ses) / (len(deltas) ** 0.5)
            print(f"| {a} | {m:+.3f} | {se:.3f} | {'YES' if m > se else 'no'} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
