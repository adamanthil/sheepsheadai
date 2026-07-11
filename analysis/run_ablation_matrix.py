#!/usr/bin/env python3
"""Orchestrate the architecture-ablation training matrix.

Runs the approved experiment (notebooks/Architecture_Ablation_202607.md):
6 architectures x 3 seeds x 100k self-play episodes, with a bounded pool of
concurrent training subprocesses (default 8 on the 10-core dev machine, one
BLAS thread each — game logic is Python-bound, so process-level parallelism
beats intra-op threads). As each run finishes, the next queued job starts
and the run's endpoint probes fire (scripted paired probe + trump-lead
incidence probe). After the whole matrix, optionally runs the PANEL-A
gauntlet over every final checkpoint (both partner modes) and always calls
analysis/aggregate_ablation.py to produce the learning-curve CSV, results
table, and plots.

Everything is resumable: jobs whose status file says "done" are skipped, so
re-running the same command continues an interrupted matrix.

Usage (full experiment):
  PYTHONPATH=. nohup .venv/bin/python analysis/run_ablation_matrix.py \
      > runs/ablation_202607/orchestrator.out 2>&1 &

Smoke test of the whole pipeline:
  PYTHONPATH=. .venv/bin/python analysis/run_ablation_matrix.py \
      --smoke --out-dir runs/ablation_smoke --prefix smokeablate
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ARCHS = [
    "full",
    "full-uninformed",
    "no-aux",
    "no-transformer",
    "no-transformer-uninformed",
    "onehot-ff",
]
SEEDS = [42, 1042, 2042]

PANEL_A = [
    "final_pfsp_swish_ppo.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt",
    "runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt",
]

_log_lock = threading.Lock()


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir
        self.status_dir = os.path.join(self.out_dir, "status")
        os.makedirs(self.status_dir, exist_ok=True)
        self.log_path = os.path.join(self.out_dir, "orchestrator.log")
        self.env = dict(
            os.environ,
            PYTHONPATH=".",
            OMP_NUM_THREADS=str(args.threads_per_job),
            MKL_NUM_THREADS=str(args.threads_per_job),
            OPENBLAS_NUM_THREADS=str(args.threads_per_job),
            VECLIB_MAXIMUM_THREADS=str(args.threads_per_job),
        )

    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with _log_lock:
            print(line, flush=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def _run_logged(self, cmd: list, log_file: str, label: str) -> int:
        self.log(f"START {label}: {' '.join(cmd)}")
        t0 = time.time()
        with open(log_file, "a") as lf:
            lf.write(f"\n===== {datetime.now()} :: {' '.join(cmd)} =====\n")
            lf.flush()
            rc = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT, env=self.env
            ).returncode
        self.log(f"END   {label}: rc={rc} ({(time.time() - t0) / 60:.1f} min)")
        return rc

    # ------------------------------------------------------------------
    def job(self, arch: str, seed: int) -> dict:
        run = f"{self.args.prefix}_{arch}_s{seed}"
        run_dir = os.path.join("runs", run)
        os.makedirs(run_dir, exist_ok=True)
        status_path = os.path.join(self.status_dir, f"{run}.json")
        final_ckpt = os.path.join(run_dir, f"final_{arch}.pt")

        if os.path.exists(status_path):
            with open(status_path) as f:
                st = json.load(f)
            if st.get("status") == "done":
                self.log(f"SKIP  {run}: already done")
                return st

        status = {"run": run, "arch": arch, "seed": seed, "status": "running"}
        status["started"] = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()

        train_cmd = [
            sys.executable,
            "-m",
            "sheepshead.training.train_selfplay_ppo",
            "--arch",
            arch,
            "--seed",
            str(seed),
            "--episodes",
            str(self.args.episodes),
            "--run-name",
            run,
            "--anchor-eval-interval",
            str(self.args.anchor_eval_interval),
            "--anchor-eval-deals",
            str(self.args.anchor_eval_deals),
            "--save-interval",
            str(self.args.save_interval),
            "--strategic-eval-interval",
            str(self.args.episodes * 10),
        ]
        if self.args.leaster_watchdog:
            train_cmd.append("--leaster-watchdog")
        status["train_cmd"] = " ".join(train_cmd)
        rc = self._run_logged(train_cmd, os.path.join(run_dir, "train.log"), run)
        status["train_rc"] = rc
        status["train_minutes"] = round((time.time() - t0) / 60, 1)

        if rc != 0 or not os.path.exists(final_ckpt):
            status["status"] = "train_failed"
        else:
            probes_log = os.path.join(run_dir, "probes.log")
            rc1 = self._run_logged(
                [
                    sys.executable,
                    "analysis/scripted_probe.py",
                    "--ckpt",
                    final_ckpt,
                    "--deals",
                    str(self.args.scripted_deals),
                    "--out-json",
                    os.path.join(run_dir, "scripted_probe.json"),
                ],
                probes_log,
                f"{run}:scripted_probe",
            )
            rc2 = self._run_logged(
                [
                    sys.executable,
                    "analysis/trump_lead_probe.py",
                    "--ckpt",
                    final_ckpt,
                    "--deals",
                    str(self.args.trump_deals),
                    "--out-json",
                    os.path.join(run_dir, "trump_lead_probe.json"),
                ],
                probes_log,
                f"{run}:trump_lead_probe",
            )
            status["probe_rcs"] = [rc1, rc2]
            status["status"] = "done" if rc1 == 0 and rc2 == 0 else "probes_failed"

        status["total_minutes"] = round((time.time() - t0) / 60, 1)
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
        return status

    # ------------------------------------------------------------------
    def run_matrix(self) -> list:
        jobs = [(a, s) for s in self.args.seeds for a in self.args.archs]
        self.log(
            f"MATRIX: {len(jobs)} jobs ({len(self.args.archs)} archs x "
            f"{len(self.args.seeds)} seeds), {self.args.max_concurrent} concurrent, "
            f"{self.args.episodes} episodes each"
        )
        results = []
        with ThreadPoolExecutor(max_workers=self.args.max_concurrent) as pool:
            futures = {pool.submit(self.job, a, s): (a, s) for a, s in jobs}
            for fut in as_completed(futures):
                a, s = futures[fut]
                try:
                    st = fut.result()
                except Exception as e:  # never kill the matrix for one job
                    st = {"arch": a, "seed": s, "status": f"error: {e}"}
                    self.log(f"ERROR {a}_s{s}: {e}")
                results.append(st)
                done = sum(1 for r in results if r.get("status") == "done")
                self.log(f"PROGRESS: {len(results)}/{len(jobs)} finished ({done} ok)")
        return results

    # ------------------------------------------------------------------
    def collect_finals(self) -> list:
        finals_dir = os.path.join(self.out_dir, "finals")
        os.makedirs(finals_dir, exist_ok=True)
        import shutil

        finals = []
        for s in self.args.seeds:
            for a in self.args.archs:
                src = os.path.join(
                    "runs", f"{self.args.prefix}_{a}_s{s}", f"final_{a}.pt"
                )
                if os.path.exists(src):
                    dst = os.path.join(finals_dir, f"{a}__s{s}.pt")
                    shutil.copyfile(src, dst)
                    finals.append(dst)
        self.log(f"FINALS: collected {len(finals)} checkpoints into {finals_dir}")
        return finals

    def run_panel_a(self, finals: list) -> None:
        if not finals:
            self.log("PANEL-A: no finals to evaluate; skipping")
            return
        missing = [p for p in PANEL_A if not os.path.exists(p)]
        if missing:
            self.log(f"PANEL-A: anchors missing {missing}; skipping")
            return
        procs = []
        for mode in ("called", "jd"):
            cmd = [
                sys.executable,
                "analysis/rigorous_eval.py",
                "--candidates",
                *finals,
                "--anchors",
                *PANEL_A,
                "--deals",
                str(self.args.panel_deals),
                "--partner-mode",
                mode,
                "--seed",
                "42",
                "--out-csv",
                os.path.join(self.out_dir, f"panel_a_{mode}.csv"),
                "--out-plot",
                os.path.join(self.out_dir, f"panel_a_{mode}.png"),
            ]
            log_file = os.path.join(self.out_dir, f"panel_a_{mode}.log")
            self.log(f"START panel_a_{mode}: {' '.join(cmd[:6])} ... ({mode})")
            lf = open(log_file, "a")
            procs.append(
                (
                    mode,
                    time.time(),
                    lf,
                    subprocess.Popen(
                        cmd, stdout=lf, stderr=subprocess.STDOUT, env=self.env
                    ),
                )
            )
        for mode, t0, lf, p in procs:
            rc = p.wait()
            lf.close()
            self.log(
                f"END   panel_a_{mode}: rc={rc} ({(time.time() - t0) / 60:.1f} min)"
            )

    def aggregate(self) -> None:
        self._run_logged(
            [
                sys.executable,
                "analysis/aggregate_ablation.py",
                "--out-dir",
                self.out_dir,
                "--prefix",
                self.args.prefix,
                "--archs",
                *self.args.archs,
                "--seeds",
                *[str(s) for s in self.args.seeds],
            ],
            os.path.join(self.out_dir, "aggregate.log"),
            "aggregate",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Architecture ablation orchestrator")
    ap.add_argument("--archs", nargs="+", default=ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--episodes", type=int, default=100_000)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--anchor-eval-interval", type=int, default=5000)
    ap.add_argument("--anchor-eval-deals", type=int, default=300)
    ap.add_argument("--save-interval", type=int, default=25_000)
    ap.add_argument("--scripted-deals", type=int, default=500)
    ap.add_argument("--trump-deals", type=int, default=2000)
    ap.add_argument("--panel-deals", type=int, default=1000)
    ap.add_argument("--skip-panel-a", action="store_true")
    ap.add_argument(
        "--leaster-watchdog",
        action="store_true",
        help="Forward --leaster-watchdog to every trainer (always-PASS "
        "collapse guard). Regime change: only compare watchdog-on runs "
        "against watchdog-on runs — include the baseline arch in --archs "
        "rather than borrowing watchdog-off rows via --extra-results.",
    )
    ap.add_argument("--out-dir", default="runs/ablation_202607")
    ap.add_argument("--prefix", default="ablate")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end pipeline check (overrides sizes)",
    )
    args = ap.parse_args()

    if args.smoke:
        args.episodes = 60
        args.anchor_eval_interval = 50
        args.anchor_eval_deals = 10
        args.save_interval = 1_000_000
        args.scripted_deals = 20
        args.trump_deals = 40
        args.panel_deals = 20
        args.seeds = args.seeds[:1]
        args.archs = args.archs if args.archs != ARCHS else ["full", "onehot-ff"]
        args.max_concurrent = min(args.max_concurrent, 2)

    os.makedirs(args.out_dir, exist_ok=True)
    orch = Orchestrator(args)
    t0 = time.time()
    results = orch.run_matrix()
    finals = orch.collect_finals()
    if not args.skip_panel_a:
        orch.run_panel_a(finals)
    orch.aggregate()

    failed = [r for r in results if r.get("status") != "done"]
    orch.log(
        f"MATRIX COMPLETE in {(time.time() - t0) / 3600:.1f} h — "
        f"{len(results) - len(failed)} ok, {len(failed)} failed"
    )
    for r in failed:
        orch.log(f"  FAILED: {r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
