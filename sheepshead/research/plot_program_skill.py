"""Plot composite skill across every phase of a training program.

One figure: the program's models in order (bootstrap seed, league
generations, policy-iteration iterates) on a composite instrument, with the
reference agents drawn as horizontal bands. The composite is the
equal-weight mean of the PANEL-A and PANEL-B per-deal scores, i.e. the
mean score against the union of the two frozen panels. Both panels are read
on the same 3,996 CRN deals (``league_progress_eval.eval_endpoint``), so the
composite is formed deal by deal and its interval comes from a deal
bootstrap of the composite scores. That interval is correct under the
cross-panel deal correlation (about 0.18 here); averaging the two
published SEs in quadrature would not be.

The absolute level of the composite mixes a weak and a strong panel and
means nothing on its own. Distances on the graph are what is interpretable,
which is why every point and every reference shares the same deals.
Inputs come from ``notebooks/panel_reference.csv``; rebuild that first.

    uv run python -m sheepshead.research.plot_program_skill
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sheepshead.eval.bootstrap import (
    IntervalStat,
    bootstrap_interval,
)
from sheepshead.eval.league_progress_eval import load_endpoint

N_BOOT = 10_000
BOOT_SEED = 42

# Program phases in order; each pattern extracts the stage ordinal.
PHASES: tuple[tuple[str, str], ...] = (
    ("Bootstrap", r"bootstrap (\d+)k$|bootstrap (seed)$"),
    ("League PPO", r"league gen (\d+)$"),
    ("Policy iteration (ExIt)", r"iteration (\d+) \(theta_\d+\)$"),
)
PHASE_COLORS = ("#9e9e9e", "#1f77b4", "#d62728")

REFERENCES: tuple[tuple[str, str, str], ...] = (
    # (csv agent name, legend label, color)
    ("production 30M (final_pfsp_swish_ppo.pt)", "production 30M", "#555555"),
    (
        "iteration-11 P1 (policy_iteration_202609/iter11/distill_epoch7.pt)",
        "iteration-11 P1",
        "#8c564b",
    ),
    ("v2 release (rc_validate_v2/final/release.pt)", "v2 release (target)", "#2ca02c"),
)


@dataclass(frozen=True)
class Point:
    agent: str
    phase: int
    stage: int
    label: str
    composite: IntervalStat
    panel_a: float
    panel_b: float


def _composite(sources: dict[str, str]) -> tuple[IntervalStat, float, float]:
    a = load_endpoint(Path(sources["PANEL-A"]))
    b = load_endpoint(Path(sources["PANEL-B"]))
    if len(a.per_deal) != len(b.per_deal) or a.seed != b.seed:
        raise ValueError("PANEL-A and PANEL-B reads are not on the same deals")
    per_deal = (a.per_deal + b.per_deal) / 2.0
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(per_deal), size=(N_BOOT, len(per_deal)))
    return bootstrap_interval(per_deal, boot_idx), a.score.mean, b.score.mean


def read_sources(csv_path: Path, run: str) -> dict[str, dict[str, str]]:
    """agent -> panel -> npz path, for program rows of ``run`` and references."""
    out: dict[str, dict[str, str]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            agent = row["agent"]
            if not (agent.startswith(run) or any(agent == r[0] for r in REFERENCES)):
                continue
            out.setdefault(agent, {})[row["panel"]] = row["source"]
    return out


def program_points(sources: dict[str, dict[str, str]], run: str) -> list[Point]:
    pts: list[Point] = []
    for agent, by_panel in sources.items():
        if not agent.startswith(run) or len(by_panel) < 2:
            continue
        tail = agent[len(run) :].strip()
        for phase, (_, pattern) in enumerate(PHASES):
            m = re.search(pattern, tail)
            if m is None:
                continue
            g = next((x for x in m.groups() if x), None)
            stage = int(g) if g and g.isdigit() else 0
            label = {
                0: f"{stage}k" if stage else "seed",
                1: f"gen {stage}",
                2: f"θ{stage}",
            }[phase]
            stat, pa, pb = _composite(by_panel)
            pts.append(Point(agent, phase, stage, label, stat, pa, pb))
            break
    pts.sort(key=lambda p: (p.phase, p.stage))
    # With bootstrap-stage composites present, the seed point (the 400k
    # composite on PANEL-A, final.pt on PANEL-B) is superseded by the 400k
    # read on both panels.
    if any(p.phase == 0 and p.stage for p in pts):
        pts = [p for p in pts if not (p.phase == 0 and p.stage == 0)]
    return pts


def reference_stats(
    sources: dict[str, dict[str, str]],
) -> list[tuple[str, str, IntervalStat]]:
    out: list[tuple[str, str, IntervalStat]] = []
    for agent, label, color in REFERENCES:
        by_panel = sources.get(agent, {})
        if len(by_panel) == 2:
            out.append((label, color, _composite(by_panel)[0]))
    return out


def plot(
    pts: list[Point],
    refs: list[tuple[str, str, IntervalStat]],
    out: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(pts))
    means = np.array([p.composite.mean for p in pts])
    lo = np.array([p.composite.lo for p in pts])
    hi = np.array([p.composite.hi for p in pts])

    # Phase bands behind the series.
    bounds: list[tuple[int, int, int]] = []
    for phase in range(len(PHASES)):
        idx = [i for i, p in enumerate(pts) if p.phase == phase]
        if idx:
            bounds.append((phase, idx[0], idx[-1]))
    for phase, i0, i1 in bounds:
        ax.axvspan(i0 - 0.5, i1 + 0.5, color=PHASE_COLORS[phase], alpha=0.07, lw=0)
        ax.text(
            (i0 + i1) / 2,
            0.98,
            PHASES[phase][0],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color=PHASE_COLORS[phase],
        )

    for label, color, stat in refs:
        ax.axhspan(stat.lo, stat.hi, color=color, alpha=0.12, lw=0)
        ax.axhline(
            stat.mean, color=color, lw=1.2, ls="--", label=f"{label}  {stat.mean:+.3f}"
        )

    ax.plot(x, means, color="black", lw=1.2, zorder=3)
    for phase, i0, i1 in bounds:
        sl = slice(i0, i1 + 1)
        ax.errorbar(
            x[sl],
            means[sl],
            yerr=[means[sl] - lo[sl], hi[sl] - means[sl]],
            fmt="o",
            ms=6,
            color=PHASE_COLORS[phase],
            ecolor=PHASE_COLORS[phase],
            capsize=3,
            lw=1.3,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([p.label for p in pts])
    ax.set_ylabel(
        "composite score vs PANEL-A ∪ PANEL-B  (per hand, 95% deal bootstrap)"
    )
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    if out.suffix != ".svg":
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="notebooks/panel_reference.csv")
    ap.add_argument("--run", default="202609_recall_rc")
    ap.add_argument("--out", default="notebooks/figures/program_skill_202609.png")
    ap.add_argument(
        "--title",
        default="Program 202609_recall_rc: skill by phase (composite panel instrument)",
    )
    args = ap.parse_args(argv)
    sources = read_sources(Path(args.csv), args.run)
    pts = program_points(sources, args.run)
    refs = reference_stats(sources)
    if not pts:
        raise SystemExit(f"no program rows with both panels for run {args.run!r}")
    plot(pts, refs, Path(args.out), args.title)
    print(
        f"{'stage':<8}{'composite':>10}{'lo':>9}{'hi':>9}{'PANEL-A':>10}{'PANEL-B':>10}"
    )
    for p in pts:
        s = p.composite
        print(
            f"{p.label:<8}{s.mean:>10.4f}{s.lo:>9.4f}{s.hi:>9.4f}"
            f"{p.panel_a:>10.4f}{p.panel_b:>10.4f}"
        )
    for label, _, s in refs:
        print(f"{label:<24}{s.mean:>+.4f}  [{s.lo:+.4f}, {s.hi:+.4f}]")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
