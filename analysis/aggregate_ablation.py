#!/usr/bin/env python3
"""Aggregate architecture-ablation results into CSVs, plots, and a table.

Reads each run's runs/<prefix>_<arch>_s<seed>/ artifacts:
  anchored_eval.csv        -> long-format learning_curves.csv (graph-ready)
  scripted_probe.json      -> endpoint scripted edge
  trump_lead_probe.json    -> endpoint trump-lead incidence
  status/<run>.json        -> wall-clock + return codes (if orchestrated)
and the matrix-level panel_a_{called,jd}.csv gauntlet outputs (if present),
then writes into --out-dir:
  learning_curves.csv   one row per (run, eval point) — plot source of truth
  results.csv           one row per run — endpoint summary
  results_table.md      markdown results table (paste into the notebook)
  curves_<yardstick>.png  learning-curve plots (per-arch mean bold,
                          per-seed traces faint)

Safe to re-run at any time (e.g. mid-matrix for partial curves).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

YARDSTICKS = ["scripted", "selfplay100k", "final_pfsp"]
ARCH_COLORS = {
    "full": "tab:blue",
    "full-uninformed": "tab:cyan",
    "no-aux": "tab:orange",
    "no-transformer": "tab:green",
    "no-transformer-uninformed": "tab:olive",
    "onehot-ff": "tab:red",
}


def read_csv(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def load_curves(args) -> list:
    rows = []
    for seed in args.seeds:
        for arch in args.archs:
            run = f"{args.prefix}_{arch}_s{seed}"
            path = os.path.join("runs", run, "anchored_eval.csv")
            if not os.path.exists(path):
                continue
            for r in read_csv(path):
                rows.append({"run": run, "arch": arch, "seed": seed, **r})
    return rows


def write_learning_curves(rows: list, out_dir: str) -> None:
    if not rows:
        return
    cols = ["run", "arch", "seed"] + [
        k for k in rows[0] if k not in ("run", "arch", "seed")
    ]
    with open(os.path.join(out_dir, "learning_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def plot_curves(rows: list, args) -> None:
    for yard in YARDSTICKS:
        key = f"edge_{yard}"
        fig, ax = plt.subplots(figsize=(9, 6))
        for arch in args.archs:
            color = ARCH_COLORS.get(arch, None)
            per_seed = {}
            for r in rows:
                if r["arch"] != arch or not r.get(key):
                    continue
                per_seed.setdefault(r["seed"], []).append(
                    (int(r["episode"]), float(r[key]))
                )
            if not per_seed:
                continue
            for pts in per_seed.values():
                pts.sort()
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=color,
                    alpha=0.25,
                    linewidth=1,
                )
            # mean across seeds at episodes present in every seed
            common = set.intersection(*[{p[0] for p in v} for v in per_seed.values()])
            xs = sorted(common)
            means = [
                sum(dict(v)[x] for v in per_seed.values()) / len(per_seed) for x in xs
            ]
            ax.plot(xs, means, color=color, linewidth=2.2, label=arch)
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("episode")
        ax.set_ylabel(f"paired edge vs {yard} (score/deal)")
        ax.set_title(
            f"Architecture ablation — edge vs {yard} (bold = seed mean, faint = seeds)"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"curves_{yard}.png"), dpi=130)
        plt.close(fig)


def load_json(path: str):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_results(args, curve_rows: list) -> list:
    # Optional PANEL-A endpoint scores, keyed by finals filename stem
    # (rigorous_eval CSV: model_id, score_per_hand, score_se, ...).
    panel = {}
    for mode in ("called", "jd"):
        p = os.path.join(args.out_dir, f"panel_a_{mode}.csv")
        if not os.path.exists(p):
            continue
        for r in read_csv(p):
            mid = r.get("model_id", "")
            if mid and r.get("score_per_hand") not in (None, ""):
                panel.setdefault(mid, {})[mode] = (
                    float(r["score_per_hand"]),
                    float(r["score_se"]) if r.get("score_se") else None,
                )

    results = []
    for seed in args.seeds:
        for arch in args.archs:
            run = f"{args.prefix}_{arch}_s{seed}"
            run_dir = os.path.join("runs", run)
            if not os.path.isdir(run_dir):
                continue
            row = {"run": run, "arch": arch, "seed": seed}

            last = None
            for r in curve_rows:
                if r["run"] == run and (
                    last is None or int(r["episode"]) > int(last["episode"])
                ):
                    last = r
            if last:
                row["final_episode"] = int(last["episode"])
                for yard in YARDSTICKS:
                    row[f"edge_{yard}"] = _f(last, f"edge_{yard}")
                    row[f"se_{yard}"] = _f(last, f"se_{yard}")
                row["train_wall_h"] = round(float(last["train_wall_s"]) / 3600, 2)
                row["eval_wall_h"] = round(float(last["eval_wall_s"]) / 3600, 2)
                tw = float(last["train_wall_s"])
                row["eps_per_s"] = round(int(last["episode"]) / tw, 2) if tw else None

            sp = load_json(os.path.join(run_dir, "scripted_probe.json"))
            if sp and sp.get("probes"):
                row["probe_scripted_edge"] = round(sp["probes"][0]["edge"], 4)
                row["probe_scripted_se"] = round(sp["probes"][0]["se"], 4)

            tl = load_json(os.path.join(run_dir, "trump_lead_probe.json"))
            if tl:
                for mode in ("jd", "called"):
                    if mode in tl:
                        row[f"trump_lead_{mode}_pct"] = round(
                            100 * tl[mode]["lead_rate"], 2
                        )

            mid = f"{arch}__s{seed}"
            for mode in ("called", "jd"):
                if mid in panel and mode in panel[mid]:
                    score, se = panel[mid][mode]
                    row[f"panel_a_{mode}"] = round(score, 4)
                    if se is not None:
                        row[f"panel_a_{mode}_se"] = round(se, 4)

            st = load_json(os.path.join(args.out_dir, "status", f"{run}.json"))
            if st:
                row["status"] = st.get("status")
                row["train_minutes"] = st.get("train_minutes")

            results.append(row)
    return results


def write_results(results: list, out_dir: str) -> None:
    if not results:
        print("no results found")
        return
    cols = []
    for r in results:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    # Markdown table (also printed) — the notebook's final-results block.
    show = [c for c in cols if c not in ("run",)]
    lines = [
        "| " + " | ".join(show) + " |",
        "|" + "|".join("---" for _ in show) + "|",
    ]
    for r in results:
        lines.append(
            "| "
            + " | ".join("" if r.get(c) is None else str(r.get(c, "")) for c in show)
            + " |"
        )
    md = "\n".join(lines)
    with open(os.path.join(out_dir, "results_table.md"), "w") as f:
        f.write(md + "\n")
    print(md)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate ablation results")
    ap.add_argument("--out-dir", default="runs/ablation_202607")
    ap.add_argument("--prefix", default="ablate")
    ap.add_argument(
        "--archs",
        nargs="+",
        default=[
            "full",
            "full-uninformed",
            "no-aux",
            "no-transformer",
            "no-transformer-uninformed",
            "onehot-ff",
        ],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1042, 2042])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_curves(args)
    write_learning_curves(rows, args.out_dir)
    if rows:
        plot_curves(rows, args)
    results = build_results(args, rows)
    write_results(results, args.out_dir)
    print(
        f"\naggregated {len(rows)} curve points, {len(results)} runs "
        f"-> {args.out_dir}/{{learning_curves.csv,results.csv,results_table.md,curves_*.png}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
