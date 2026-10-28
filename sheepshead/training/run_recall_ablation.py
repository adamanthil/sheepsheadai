#!/usr/bin/env python3
"""Run the recall-contract architecture ablation (Architecture_Ablation_202610).

Seven additive architecture rungs x three seeds, each trained through the
202609_recall_rc bootstrap phase exactly as that run trained it
(``train_ppo --phase bootstrap``, 400k episodes, 8 workers, leaster
watchdog on, deterministic aux coefficients at 1.0x), then read on the
release's instruments: PANEL-A three-checkpoint composites every 50k from
150k, PANEL-B at 400k, and the program's convention battery at every
checkpoint. The release's own bootstrap is the (perceiver-recall, seed 42)
run; it is read, not retrained.

Training runs strictly one at a time (seed-major, so each seed's ladder
completes before the next starts) so every run sees the same machine load
and its wall-clock is comparable. Evaluation runs afterwards with ``--jobs``
single-threaded processes. Both stages are resumable: finished runs and
existing outputs are skipped, and an interrupted run resumes from its
newest checkpoint.

    uv run python -m sheepshead.training.run_recall_ablation train
    uv run python -m sheepshead.training.run_recall_ablation eval --jobs 8
    uv run python -m sheepshead.training.run_recall_ablation status
    uv run python -m sheepshead.research.recall_ablation_report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

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


def _git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "driver.log").open("a") as f:
        f.write(line + "\n")


def _read_status(arch: str, seed: int) -> dict:
    path = cell_dir(arch, seed) / "status.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _write_status(arch: str, seed: int, status: dict) -> None:
    path = cell_dir(arch, seed) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")


def train_argv(arch: str, seed: int, resume: Path | None) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "sheepshead.training.train_ppo",
        "--phase",
        "bootstrap",
        "--arch",
        arch,
        "--run-name",
        f"{STUDY}/{arch}_s{seed}",
        "--until",
        str(EPISODES),
        "--save-interval",
        str(STEP),
        "--greedy-eval-interval",
        str(STEP),
        "--greedy-eval-games",
        "200",
        "--num-workers",
        str(NUM_WORKERS),
        "--seed",
        str(seed),
        "--aux-det-scale",
        str(AUX_DET_SCALE),
    ]
    if resume is not None:
        argv += ["--resume", str(resume)]
    return argv


def record_reused() -> None:
    for (arch, seed), src in REUSED.items():
        if _read_status(arch, seed):
            continue
        _write_status(
            arch,
            seed,
            {
                "state": "done",
                "reused_from": str(src),
                "wall_hours": REUSED_WALL_HOURS[(arch, seed)],
                "train_log": str(REUSED_TRAIN_LOG[(arch, seed)]),
            },
        )


def train(_args) -> int:
    record_reused()
    git = _git_state()
    for arch, seed in cells():
        if (arch, seed) in REUSED:
            continue
        status = _read_status(arch, seed)
        if is_trained(arch, seed) and status.get("state") == "done":
            continue
        resume = newest_checkpoint(arch, seed)
        argv = train_argv(arch, seed, resume)
        segments = status.get("segments", [])
        segment = {"argv": argv, "git": git, "started": time.time()}
        segments.append(segment)
        _write_status(arch, seed, {"state": "running", "segments": segments})
        _log(
            f"train {arch} s{seed}"
            + (f" (resuming {resume.name})" if resume else "")
            + f": {' '.join(argv[1:])}"
        )
        log_path = train_log(arch, seed)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
        segment["ended"] = time.time()
        segment["exit"] = code
        hours = sum(s["ended"] - s["started"] for s in segments if "ended" in s) / 3600
        if code != 0 or not is_trained(arch, seed):
            _write_status(arch, seed, {"state": "failed", "segments": segments})
            _log(f"FAILED {arch} s{seed} (exit {code}); see {log_path}. Stopping.")
            return 1
        _write_status(
            arch,
            seed,
            {
                "state": "done",
                "segments": segments,
                "wall_hours": hours,
                "resumed": len(segments) > 1,
            },
        )
        _log(f"done {arch} s{seed} in {hours:.2f} h")
    _log("TRAINING COMPLETE")
    return 0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _panel_task(arch: str, seed: int, panel: str, stage: int, out: str) -> str:
    import torch

    from sheepshead.eval.league_progress_eval import (
        PANEL_SEED,
        eval_endpoint,
    )
    from sheepshead.eval.panels import panel_files

    torch.set_num_threads(1)
    ckpts = [
        str(checkpoint(arch, seed, n)) for n in (stage - 2 * STEP, stage - STEP, stage)
    ]
    t0 = time.time()
    e = eval_endpoint(
        ckpts,
        n_deals=PANEL_DEALS,
        seed=PANEL_SEED,
        panel_paths=panel_files(panel),
        out_npz=Path(out),
        log=False,
    )
    return (
        f"{panel} {arch} s{seed} {stage // 1000}k: {e.score.mean:+.4f} "
        f"[{e.score.lo:+.4f}, {e.score.hi:+.4f}] in {(time.time() - t0) / 60:.0f} min"
    )


def _convention_task(arch: str, seed: int, stage: int, out: str) -> str:
    """The program's convention battery (run_training_program._conventions):
    greedy self-play probes on its fixed seeds, scalar means across seeds."""
    import torch

    from sheepshead.agent.ppo import load_agent
    from sheepshead.eval.health_probe import greedy_health_probe
    from sheepshead.training.run_training_program import CONVENTION_SEEDS

    torch.set_num_threads(1)
    t0 = time.time()
    agent = load_agent(str(checkpoint(arch, seed, stage)))
    probes = [
        greedy_health_probe(agent, n_games=CONVENTION_GAMES, seed=s)
        for s in CONVENTION_SEEDS
    ]
    scalar_keys = [
        k
        for k, v in probes[0].items()
        if isinstance(v, (int, float)) and all(k in p for p in probes)
    ]
    summary: dict = {k: float(np.mean([p[k] for p in probes])) for k in scalar_keys}
    summary["probes"] = probes
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return f"conventions {arch} s{seed} {stage // 1000}k in {(time.time() - t0) / 60:.0f} min"


def eval_tasks() -> list[tuple]:
    """Every missing eval output whose checkpoints exist, longest first."""
    panels, conventions = [], []
    for arch, seed in cells():
        for panel, stages, prefix in (
            ("PANEL-A", PANEL_A_STAGES, "panelA"),
            ("PANEL-B", PANEL_B_STAGES, "panelB"),
        ):
            for stage in stages:
                out = eval_path(arch, seed, f"{prefix}_{stage}.npz")
                needed = [
                    checkpoint(arch, seed, n)
                    for n in (stage - 2 * STEP, stage - STEP, stage)
                ]
                if not out.exists() and all(p.exists() for p in needed):
                    panels.append((_panel_task, arch, seed, panel, stage, str(out)))
        for stage in CONVENTION_STAGES:
            out = eval_path(arch, seed, f"conventions_{stage}.json")
            if not out.exists() and checkpoint(arch, seed, stage).exists():
                conventions.append((_convention_task, arch, seed, stage, str(out)))
    return panels + conventions


def evaluate(args) -> int:
    tasks = eval_tasks()
    _log(f"eval: {len(tasks)} tasks on {args.jobs} processes")
    failed = 0
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = {pool.submit(fn, *rest): (fn.__name__, rest) for fn, *rest in tasks}
        for fut in as_completed(futures):
            name, rest = futures[fut]
            try:
                _log(fut.result())
            # Deliberately broad: a read can fail anywhere in model loading,
            # play or a dying worker process; each failure is logged and the
            # remaining reads of a many-hour batch still run.
            except Exception as err:  # noqa: BLE001
                failed += 1
                _log(f"eval FAILED {name}{tuple(rest[:-1])}: {err!r}")
    _log(f"EVAL COMPLETE ({failed} failed)")
    return 1 if failed else 0


def status(_args) -> int:
    for arch, seed in cells():
        st = _read_status(arch, seed).get("state", "pending")
        newest = newest_checkpoint(arch, seed)
        evals = (
            len(list((cell_dir(arch, seed) / "eval").glob("*")))
            if (cell_dir(arch, seed) / "eval").exists()
            else 0
        )
        expected = len(PANEL_A_STAGES) + len(PANEL_B_STAGES) + len(CONVENTION_STAGES)
        print(
            f"{arch:32s} s{seed:<5d} {st:8s} "
            f"newest {newest.name if newest else '-':24s} eval {evals}/{expected}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train every missing cell, one at a time")
    ev = sub.add_parser("eval", help="read every missing instrument output")
    ev.add_argument("--jobs", type=int, default=8)
    sub.add_parser("status", help="print per-cell progress")
    args = ap.parse_args(argv)
    return {"train": train, "eval": evaluate, "status": status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
