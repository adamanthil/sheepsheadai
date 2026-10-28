"""Tables and figure for the recall-contract architecture ablation
(Architecture_Ablation_202610; runs from training/run_recall_ablation.py).

Writes, from whatever outputs exist so far (cells or reads not yet run are
blank, so this can be rerun at any point):

- ``notebooks/arch_ablation_202610.csv``: one row per (arch, seed,
  checkpoint episode): PANEL-A / PANEL-B composite reads with 95% deal
  bootstrap intervals and per-mode means, the convention battery, and the
  run-level columns (params, wall-clock, throughput, watchdog episodes).
- ``notebooks/arch_ablation_202610_deltas.csv``: paired per-deal deltas
  between each rung and the rung below it, per seed (deal bootstrap SE)
  and across seeds (seed-level SE), at every PANEL-A stage and the PANEL-B
  endpoint. Every read shares the panel's CRN deal set, so deltas are
  formed deal by deal.
- ``notebooks/figures/arch_ablation_202610_panelA.png`` (+ .svg): PANEL-A
  learning curves, seed mean per rung with the individual seeds faint.

    uv run python -m sheepshead.analysis.recall_ablation_report
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sheepshead.analysis.league_progress_eval import Endpoint, load_endpoint
from sheepshead.analysis.recall_ablation_study import (
    CONVENTION_STAGES,
    EPISODES,
    LADDER,
    PANEL_A_STAGES,
    PANEL_B_STAGES,
    SEEDS,
    STEP,
    cell_dir,
    eval_path,
    train_log,
)

NOTEBOOKS = Path("notebooks")
RESULTS_CSV = NOTEBOOKS / "arch_ablation_202610.csv"
DELTAS_CSV = NOTEBOOKS / "arch_ablation_202610_deltas.csv"
FIGURE = NOTEBOOKS / "figures" / "arch_ablation_202610_panelA.png"

PANELS = (("PANEL-A", "panelA", PANEL_A_STAGES), ("PANEL-B", "panelB", PANEL_B_STAGES))
CONVENTION_KEYS = (
    "pick_rate",
    "alone_rate",
    "leaster_rate",
    "t0_trump_lead_rate",
    "partner_trump_lead_rate",
    "called_suit_lead_rate",
    "play_logit_spread_med",
    "seen_trump_acc",
    "seen_trump_recall",
    "aux_secret_acc",
    "aux_points_mae",
    "aux_unseen_higher_acc",
)
N_BOOT = 10_000
BOOT_SEED = 42

# Reference categorical palette (dataviz skill), fixed rung order; passes
# the CVD/normal-vision checks for 7 adjacent slots. Three slots fall below
# 3:1 contrast on the surface, so every curve is direct-labeled.
RUNG_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
)
INK, INK_MUTED, SURFACE = "#1f1f1e", "#6b6a63", "#fcfcfb"


def param_count(arch: str) -> int:
    from sheepshead.agent.ppo import PPOAgent
    from sheepshead.game import ACTIONS

    agent = PPOAgent(len(ACTIONS), arch=arch)
    return sum(
        p.numel()
        for net in (agent.encoder, agent.actor, agent.critic)
        for p in net.parameters()
    )


_EP = re.compile(r"^Ep ([\d,]+) \|.*\| ([\d.]+) eps/s")


def watchdog_episodes(log: Path) -> tuple[int | None, int | None]:
    """Episodes of the first leaster-watchdog engage and the release after
    it (the last progress line before each event)."""
    if not log.exists():
        return None, None
    episode, engaged, released = 0, None, None
    with log.open(errors="replace") as f:
        for line in f:
            m = _EP.match(line)
            if m:
                episode = int(m.group(1).replace(",", ""))
            elif "watchdog ENGAGED" in line and engaged is None:
                engaged = episode
            elif (
                "watchdog released" in line and engaged is not None and released is None
            ):
                released = episode
    return engaged, released


def run_level(arch: str, seed: int) -> dict:
    path = cell_dir(arch, seed) / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    hours = status.get("wall_hours") if status.get("state") == "done" else None
    engaged, released = watchdog_episodes(train_log(arch, seed))
    return {
        "wall_hours": hours,
        "eps_per_s": EPISODES / (hours * 3600) if hours else None,
        "resumed": status.get("resumed", False),
        "reused_from": status.get("reused_from", ""),
        "watchdog_engage_ep": engaged,
        "watchdog_release_ep": released,
    }


def load_read(arch: str, seed: int, prefix: str, stage: int) -> Endpoint | None:
    path = eval_path(arch, seed, f"{prefix}_{stage}.npz")
    return load_endpoint(path) if path.exists() else None


def results_rows() -> list[dict]:
    params = {arch: param_count(arch) for arch in LADDER}
    rows = []
    for seed in SEEDS:
        for rung, arch in enumerate(LADDER, start=1):
            run = run_level(arch, seed)
            for episode in range(STEP, EPISODES + 1, STEP):
                row: dict = {
                    "rung": rung,
                    "arch": arch,
                    "seed": seed,
                    "episode": episode,
                }
                row["params"] = params[arch]
                for panel, prefix, stages in PANELS:
                    col = prefix.replace("panel", "panel_").lower()
                    e = (
                        load_read(arch, seed, prefix, episode)
                        if episode in stages
                        else None
                    )
                    row[col] = e.score.mean if e else None
                    row[f"{col}_lo"] = e.score.lo if e else None
                    row[f"{col}_hi"] = e.score.hi if e else None
                    row[f"{col}_called"] = e.mode_means.get("called") if e else None
                    row[f"{col}_jd"] = e.mode_means.get("jd") if e else None
                conv_path = eval_path(arch, seed, f"conventions_{episode}.json")
                conv = (
                    json.loads(conv_path.read_text())
                    if episode in CONVENTION_STAGES and conv_path.exists()
                    else {}
                )
                for k in CONVENTION_KEYS:
                    row[k] = conv.get(k)
                row.update(run)
                rows.append(row)
    return rows


def _paired(a: Endpoint, b: Endpoint) -> tuple[float, float]:
    """Mean per-deal delta a - b and its deal-bootstrap SE."""
    if len(a.per_deal) != len(b.per_deal) or a.hash != b.hash:
        raise ValueError("reads are not on the same CRN deal set")
    d = a.per_deal - b.per_deal
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    return float(d.mean()), float(d[idx].mean(axis=1).std(ddof=1))


def delta_rows() -> list[dict]:
    rows = []
    for panel, prefix, stages in PANELS:
        for stage in stages:
            for k in range(1, len(LADDER)):
                arch, base = LADDER[k], LADDER[k - 1]
                per_seed = []
                for seed in SEEDS:
                    a = load_read(arch, seed, prefix, stage)
                    b = load_read(base, seed, prefix, stage)
                    if a is None or b is None:
                        continue
                    delta, se = _paired(a, b)
                    per_seed.append(delta)
                    rows.append(
                        {
                            "panel": panel,
                            "episode": stage,
                            "rung": k + 1,
                            "arch": arch,
                            "base_arch": base,
                            "seed": seed,
                            "delta": delta,
                            "se": se,
                            "n_seeds": 1,
                        }
                    )
                if len(per_seed) >= 2:
                    rows.append(
                        {
                            "panel": panel,
                            "episode": stage,
                            "rung": k + 1,
                            "arch": arch,
                            "base_arch": base,
                            "seed": "mean",
                            "delta": float(np.mean(per_seed)),
                            "se": float(
                                np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))
                            ),
                            "n_seeds": len(per_seed),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if v is None else v for k, v in row.items()})
    print(f"wrote {path}: {len(rows)} rows")


def plot_panel_a(rows: list[dict], out: Path) -> bool:
    fig, ax = plt.subplots(
        figsize=(10, 5.8), constrained_layout=True, facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)
    drew = False
    for rung, arch in enumerate(LADDER, start=1):
        color = RUNG_COLORS[rung - 1]
        by_seed: dict[int, dict[int, float]] = {}
        for r in rows:
            if r["arch"] == arch and r["panel_a"] is not None:
                by_seed.setdefault(r["seed"], {})[r["episode"]] = r["panel_a"]
        if not by_seed:
            continue
        drew = True
        for series in by_seed.values():
            xs = sorted(series)
            ax.plot(
                [x / 1000 for x in xs],
                [series[x] for x in xs],
                color=color,
                lw=1,
                alpha=0.3,
            )
        stages = sorted({x for s in by_seed.values() for x in s})
        mean = [
            float(np.mean([s[x] for s in by_seed.values() if x in s])) for x in stages
        ]
        xk = [x / 1000 for x in stages]
        ax.plot(xk, mean, color=color, lw=2, marker="o", ms=5, label=f"{rung} {arch}")
        ax.annotate(
            f"{rung}",
            (xk[-1], mean[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    if not drew:
        plt.close(fig)
        return False
    ax.set_xlabel(
        "bootstrap episodes (thousands); each point a 3-checkpoint composite",
        color=INK_MUTED,
    )
    ax.set_ylabel("score per hand vs PANEL-A (seed mean; seeds faint)", color=INK_MUTED)
    ax.set_title(
        "Recall-contract architecture ladder: bootstrap learning curves", color=INK
    )
    ax.grid(axis="y", color="#e4e3dd", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED)
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    rows = results_rows()
    write_csv(RESULTS_CSV, rows)
    write_csv(DELTAS_CSV, delta_rows())
    plot_panel_a(rows, FIGURE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
