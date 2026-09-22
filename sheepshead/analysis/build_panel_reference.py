"""Assemble the long-format panel reference table (notebooks/panel_reference.csv).

One row per (agent, panel) read taken under the orchestrator's endpoint
protocol (``league_progress_eval.eval_endpoint``: 3,996 deals, PANEL_SEED,
the frozen member list of ``analysis/panels.py``). Every PANEL-A row shares
one deal hash and every PANEL-B row another, so rows within a panel are
paired reads on identical deals. Reference agents are single checkpoints
(the path repeated three times, the documented single-checkpoint form);
generation rows are the orchestrator's three-checkpoint composites.

The table exists so write-ups can cite these numbers without recomputing
them. Sources are the endpoint ``.npz`` files each read wrote; add a
``Source`` entry when a new read lands and re-run:

    uv run python -m sheepshead.analysis.build_panel_reference

Missing source files are reported and skipped so the table can be rebuilt
while reads are still running.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from sheepshead.analysis.league_progress_eval import load_endpoint

RECALL_RC = Path("runs/202609_recall_rc/program")
RETENTION = Path("runs/league_retention_pg/orchestrator")
REFS = Path("runs/202609_recall_rc/recall_compare")

COMPOSITE = "3-ckpt composite"
SINGLE = "single ckpt x3"


@dataclass(frozen=True)
class Source:
    agent: str
    lineage: str
    league_episodes: int | None
    panel: str
    protocol: str
    npz: Path


def _fresh_run() -> list[Source]:
    lineage = "perceiver-recall (fresh run 202609_recall_rc)"
    out = [
        Source(
            "202609_recall_rc bootstrap seed",
            lineage,
            0,
            "PANEL-A",
            SINGLE,
            RECALL_RC / "panel_seed.npz",
        )
    ]
    for g in range(1, 9):
        out.append(
            Source(
                f"202609_recall_rc league gen {g}",
                lineage,
                g * 1_000_000,
                "PANEL-A",
                COMPOSITE,
                RECALL_RC / f"panel_gen{g}.npz",
            )
        )
        out.append(
            Source(
                f"202609_recall_rc league gen {g}",
                lineage,
                g * 1_000_000,
                "PANEL-B",
                COMPOSITE,
                RECALL_RC / f"panelB_gen{g}.npz",
            )
        )
    return out


def _retention_run() -> list[Source]:
    lineage = "perceiver-shared-v2 (league_retention_pg)"
    out = [
        Source(
            "league_retention_pg gen 0 (v2 400k warm start)",
            lineage,
            0,
            "PANEL-A",
            SINGLE,
            RETENTION / "panel_gen0.npz",
        )
    ]
    for g in range(1, 9):
        out.append(
            Source(
                f"league_retention_pg gen {g}",
                lineage,
                g * 1_000_000,
                "PANEL-A",
                COMPOSITE,
                RETENTION / f"panel_gen{g}.npz",
            )
        )
    return out


def _references() -> list[Source]:
    refs = [
        (
            "30m",
            "production 30M (final_pfsp_swish_ppo.pt)",
            "legacy swish PFSP",
            30_000_000,
        ),
        (
            "release",
            "v2 release (rc_validate_v2/final/release.pt)",
            "perceiver-shared-v2 lineage (theta_3 + bidding phase)",
            None,
        ),
        (
            "iter11p1",
            "iteration-11 P1 (policy_iteration_202609/iter11/distill_epoch7.pt)",
            "perceiver-shared-v2 lineage (CE teacher)",
            None,
        ),
        (
            "v2_8m",
            "v2 8M (league_retention_pg checkpoint_8000000.pt)",
            "perceiver-shared-v2 (league_retention_pg)",
            8_000_000,
        ),
        (
            "v2_seed",
            "v2 400k warm start (warmstart_perceiver-shared-v2_400k.pt)",
            "perceiver-shared-v2 (league_retention_pg)",
            0,
        ),
        (
            "ours_seed",
            "202609_recall_rc bootstrap seed",
            "perceiver-recall (fresh run 202609_recall_rc)",
            0,
        ),
    ]
    out: list[Source] = []
    for tag, agent, lineage, episodes in refs:
        for panel, prefix in (("PANEL-A", "panelA"), ("PANEL-B", "panelB")):
            out.append(
                Source(
                    agent,
                    lineage,
                    episodes,
                    panel,
                    SINGLE,
                    REFS / f"{prefix}_{tag}.npz",
                )
            )
    return out


def sources() -> list[Source]:
    return _fresh_run() + _retention_run() + _references()


FIELDS = [
    "agent",
    "lineage",
    "league_episodes",
    "panel",
    "protocol",
    "deals",
    "seed",
    "deal_hash",
    "mean",
    "lo",
    "hi",
    "se",
    "called",
    "jd",
    "checkpoints",
    "source",
]


def build_rows(srcs: list[Source]) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    missing: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    for s in srcs:
        if not s.npz.exists():
            missing.append(s.npz)
            continue
        e = load_endpoint(s.npz)
        key = (s.agent, s.panel, s.protocol)
        if key in seen:
            continue  # the same read registered under two source paths
        seen.add(key)
        rows.append(
            {
                "agent": s.agent,
                "lineage": s.lineage,
                "league_episodes": ""
                if s.league_episodes is None
                else s.league_episodes,
                "panel": s.panel,
                "protocol": s.protocol,
                "deals": len(e.per_deal),
                "seed": e.seed,
                "deal_hash": e.hash[:12],
                "mean": f"{e.score.mean:.4f}",
                "lo": f"{e.score.lo:.4f}",
                "hi": f"{e.score.hi:.4f}",
                "se": f"{e.score.se:.4f}",
                "called": f"{e.mode_means['called']:.4f}",
                "jd": f"{e.mode_means['jd']:.4f}",
                "checkpoints": ";".join(Path(c).name for c in e.ckpts),
                "source": str(s.npz),
            }
        )
    return rows, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="notebooks/panel_reference.csv")
    args = ap.parse_args(argv)
    rows, missing = build_rows(sources())
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    for m in missing:
        print(f"  (no read yet) {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
