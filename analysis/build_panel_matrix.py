"""Assemble the wide PANEL-A strength matrix for the architecture comparison.

Produces one CSV (default notebooks/panel_a_strength_matrix.csv) with:

  - one row per 25k-episode snapshot (25k .. 400k);
  - one column per architecture+seed lineage (header `<arch>_s<seed>`)
    holding the PANEL-A score_per_hand at that snapshot, averaged over
    the called and jd partner modes (the standing both-modes scalar).
    A cell is filled only when BOTH modes' panels exist for that
    checkpoint; otherwise it is left blank for later backfill;
  - supplementary columns `<arch>_s<seed>_edge_scripted` and
    `<arch>_s<seed>_edge_100k`: the trainer's 300-deal anchored-eval
    edges vs the scripted agent and the selfplay-100k reference at the
    same snapshot. These are 300-deal instruments (motivation-only per
    standing rule 2 / P5) — kept in separate clearly-named columns so
    they are never conflated with the 1000-deal panel numbers.

Sources:
  - PANEL-A: every CSV matching runs/perceiver_202607/diag/panel*.csv
    whose rows point at `runs/ablate_<arch>[400]_s<seed>/..._checkpoint_<N>.pt`.
    Finals-based rows and anchor rows never match the filepath pattern,
    so the finals taint (standing rule 5) is excluded structurally.
    Re-running after new panels land extends the matrix automatically.
  - Edges: runs/ablate_<arch>_s<seed>/anchored_eval.csv (+ the `<arch>400`
    resume dir for episodes > 200k), rows at exact 25k multiples.

Columns are emitted for every lineage that has a run directory on disk,
including ones with no checkpoint panels yet (blank scaffolding for the
planned backfill).

Usage:
    uv run python analysis/build_panel_matrix.py \
        [--out notebooks/panel_a_strength_matrix.csv] [--max-episode 400000]
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANEL_GLOB = os.path.join(REPO_ROOT, "runs/perceiver_202607/diag/panel*.csv")
RUN_GLOB = os.path.join(REPO_ROOT, "runs/ablate_*_s*")
CKPT_ROW_RE = re.compile(
    r"runs/ablate_([A-Za-z0-9-]+?)(?:400)?_s(\d+)/.*checkpoint_(\d+)\.pt"
)
RUN_DIR_RE = re.compile(r"ablate_([A-Za-z0-9-]+?)(400)?_s(\d+)$")

# Ladder/registry ordering for stable, readable column layout.
ARCH_ORDER = [
    "full",
    "full-uninformed",
    "no-aux",
    "no-transformer",
    "no-transformer-uninformed",
    "onehot-ff",
    "full-tokenread",
    "perceiver",
    "perceiver-shared",
    "perceiver-shared-v2",
    "readout-actor",
    "readout-critic",
]
SEED_ORDER = [42, 1042, 2042]


def mode_of(panel_path):
    name = os.path.basename(panel_path)
    if "called" in name:
        return "called"
    if "jd" in name:
        return "jd"
    return None


def collect_panels():
    """(arch, seed, episode, mode) -> list of score_per_hand."""
    cells = defaultdict(list)
    for path in sorted(glob.glob(PANEL_GLOB)):
        mode = mode_of(path)
        if mode is None:
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                m = CKPT_ROW_RE.search(row.get("filepath", ""))
                if not m:
                    continue
                arch, seed, ep = m.group(1), int(m.group(2)), int(m.group(3))
                cells[(arch, seed, ep, mode)].append(float(row["score_per_hand"]))
    return cells


def collect_edges():
    """(arch, seed, episode) -> (edge_scripted, edge_100k) at 25k multiples."""
    edges = {}
    for run_dir in sorted(glob.glob(RUN_GLOB)):
        m = RUN_DIR_RE.search(run_dir)
        if not m or not os.path.isdir(run_dir):
            continue
        arch, seed = m.group(1), int(m.group(3))
        path = os.path.join(run_dir, "anchored_eval.csv")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                ep = int(row["episode"])
                if ep % 25000 != 0:
                    continue
                # Resume runs (<arch>400) overlap their base run only at the
                # resume point; later files win harmlessly (identical lineage).
                edges[(arch, seed, ep)] = (
                    float(row["edge_scripted"]),
                    float(row["edge_selfplay100k"]),
                )
    return edges


def lineages():
    """All (arch, seed) lineages present on disk, in ladder order."""
    found = set()
    for run_dir in glob.glob(RUN_GLOB):
        m = RUN_DIR_RE.search(run_dir)
        if m and os.path.isdir(run_dir):
            found.add((m.group(1), int(m.group(3))))
    known = [a for a in ARCH_ORDER if any(x == a for x, _ in found)]
    extra = sorted({a for a, _ in found} - set(known))  # future archs
    ordered = []
    for arch in known + extra:
        for seed in SEED_ORDER + sorted(
            {s for a, s in found if a == arch} - set(SEED_ORDER)
        ):
            if (arch, seed) in found:
                ordered.append((arch, seed))
    return ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "notebooks/panel_a_strength_matrix.csv"),
    )
    ap.add_argument("--max-episode", type=int, default=400000)
    args = ap.parse_args()

    cells = collect_panels()
    edges = collect_edges()
    cols = lineages()

    header = ["episode"]
    header += [f"{a}_s{s}" for a, s in cols]
    header += [f"{a}_s{s}_edge_scripted" for a, s in cols]
    header += [f"{a}_s{s}_edge_100k" for a, s in cols]

    n_panel = 0
    out_rows = []
    for ep in range(25000, args.max_episode + 1, 25000):
        row = [str(ep)]
        for a, s in cols:
            called = cells.get((a, s, ep, "called"))
            jd = cells.get((a, s, ep, "jd"))
            if called and jd:
                both = sum(called) / len(called) + sum(jd) / len(jd)
                row.append(f"{both / 2:.4f}")
                n_panel += 1
            else:
                row.append("")
        for idx in (0, 1):
            for a, s in cols:
                e = edges.get((a, s, ep))
                row.append(f"{e[idx]:.4f}" if e else "")
        out_rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out_rows)

    print(f"{len(cols)} lineages x {len(out_rows)} snapshots -> {args.out}")
    print(f"  PANEL-A cells filled: {n_panel}")
    print(
        f"  edge cells filled: {sum(1 for r in out_rows for v in r[1 + len(cols) :] if v) // 2} per yardstick"
    )
    filled_archs = sorted({a for (a, s, ep, m) in cells})
    print(f"  archs with panel data: {', '.join(filled_archs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
