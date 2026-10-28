"""Definition of the recall-contract architecture ablation
(Architecture_Ablation_202610): its ladder, seeds, schedule and on-disk
layout.

Shared by the orchestrator (``sheepshead.training.run_recall_ablation``)
and the report (``sheepshead.analysis.recall_ablation_report``); it lives
here, like ``panels``, so the report never imports the orchestrator.
"""

from __future__ import annotations

from pathlib import Path

STUDY = "arch_ablation_202610"
OUT = Path("runs") / STUDY

# Bottom to top; each rung adds one component to the rung below it
# (Architecture_Ablation_202610 §2).
LADDER: tuple[str, ...] = (
    "onehot-recall",
    "pooled-recall-uninformed-noaux",
    "pooled-recall-noaux",
    "full-recall-noaux",
    "full-recall",
    "perceiver-recall-nobilinear",
    "perceiver-recall",
)
SEEDS: tuple[int, ...] = (42, 1042, 2042)

EPISODES = 400_000
STEP = 50_000
AUX_DET_SCALE = 1.0
NUM_WORKERS = 8

PANEL_A_STAGES: tuple[int, ...] = tuple(range(150_000, EPISODES + 1, STEP))
PANEL_B_STAGES: tuple[int, ...] = (EPISODES,)
CONVENTION_STAGES: tuple[int, ...] = tuple(range(STEP, EPISODES + 1, STEP))
CONVENTION_GAMES = 1000
PANEL_DEALS = 3996

# The release's bootstrap phase IS this cell (same recipe, aux at 1.0x:
# the scale flag postdates it, Architecture_Ablation_202610 §3).
REUSED: dict[tuple[str, int], Path] = {
    ("perceiver-recall", 42): Path("runs/202609_recall_rc/bootstrap"),
}
# runs/202609_recall_rc/orchestrator.log: "✔ bootstrap: done in 7.77 h".
REUSED_WALL_HOURS: dict[tuple[str, int], float] = {("perceiver-recall", 42): 7.77}
REUSED_TRAIN_LOG: dict[tuple[str, int], Path] = {
    ("perceiver-recall", 42): Path("runs/202609_recall_rc/program/bootstrap.log"),
}


def cell_dir(arch: str, seed: int) -> Path:
    """Where this study keeps a cell's status, log and eval outputs."""
    return OUT / f"{arch}_s{seed}"


def run_dir(arch: str, seed: int) -> Path:
    """Where the cell's checkpoints live (the release's dir for the reused cell)."""
    return REUSED.get((arch, seed), cell_dir(arch, seed))


def train_log(arch: str, seed: int) -> Path:
    return REUSED_TRAIN_LOG.get((arch, seed), cell_dir(arch, seed) / "train.log")


def checkpoint(arch: str, seed: int, episode: int) -> Path:
    return run_dir(arch, seed) / "checkpoints" / f"checkpoint_{episode}.pt"


def eval_path(arch: str, seed: int, name: str) -> Path:
    return cell_dir(arch, seed) / "eval" / name


def cells() -> list[tuple[str, int]]:
    return [(arch, seed) for seed in SEEDS for arch in LADDER]


def is_trained(arch: str, seed: int) -> bool:
    return checkpoint(arch, seed, EPISODES).exists()


def newest_checkpoint(arch: str, seed: int) -> Path | None:
    found = [
        checkpoint(arch, seed, n)
        for n in range(STEP, EPISODES + 1, STEP)
        if checkpoint(arch, seed, n).exists()
    ]
    return found[-1] if found else None
