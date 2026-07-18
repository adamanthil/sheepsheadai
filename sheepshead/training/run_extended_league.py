#!/usr/bin/env python3
"""
Extended league training orchestrator with a quantitative stopping rule.

Runs the full recipe pre-registered in notebooks/Extended_League_202607.md:

  1. CALIBRATION  pick the gen-1 bidding-anchor coefficient by probing 2-3
                  candidates for ~15k episodes each (in-process run_main_phase;
                  no exploiter). Criterion: league_stopping.pick_anchor_coeff.
  2. GEN 1        train_league_ppo.py subprocess, KL-anchored to the ORIGINAL
                  resume checkpoint, league seeded from --seed-checkpoints.
  3. GEN 2..N     unanchored generations, resume-chained from each boundary
                  checkpoint (--generations 1 per invocation; the trainer's
                  absolute-episode boundary math keeps numbering/cadence).
  4. After each generation: composite PANEL-A endpoint + gen-vs-prev h2h
                  (league_progress_eval), stop-rule verdict (league_stopping),
                  telemetry (generations.csv, report.md, generations_curve.png).
  5. STOP         after two consecutive flat generations (post min-generations
                  floor) confirmed by a fresh-deal panel (seed 20260706), or at
                  the max-generations safety cap. A confirmation contradiction
                  resumes training instead.

Crash-resumable: all phases are keyed to on-disk artifacts (checkpoints,
exploitability.csv rows, .npz/.json eval outputs) plus an atomic state.json in
runs/<run>/orchestrator/, so re-invoking with the same arguments always
converges to where it left off. Exit codes: 0 stopped cleanly, 2 needs_review.

Example (fill in the ablation winner; resume a standard CHECKPOINT, not
final_*.pt — pre-2026-07-08 finals carry an out-of-spec flush update, see
the ablation notebook's standing rule 5):
  uv run python analysis/run_extended_league.py \
    --arch full --resume runs/ablate_full400_s42/full_checkpoint_400000.pt \
    --seed-checkpoints 'runs/ablate_full_s42/full_checkpoint_*.pt' \
    --run-name ext_league_202607 --critic-mode oracle

Smoke test of the whole loop in minutes:
  uv run python analysis/run_extended_league.py --smoke \
    --arch full --resume runs/ablate_full400_s42/full_checkpoint_400000.pt \
    --seed-checkpoints 'runs/ablate_full_s42/full_checkpoint_*.pt' \
    --run-name smoke_ext_league
"""

from __future__ import annotations

import os
import sys


import argparse
import csv
import glob
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from sheepshead.analysis.league_progress_eval import (
    CONFIRM_SEED,
    N_BOOT,
    PANEL_SEED,
    Endpoint,
    _endpoint_boot_idx,
    eval_endpoint,
    h2h,
    load_endpoint,
)
from sheepshead.analysis.league_stopping import (
    ProbeSummary,
    StopRuleConfig,
    confirmation_verdict,
    decide_stop,
    flat_verdict,
    pick_anchor_coeff,
    verdict_to_dict,
)
from sheepshead.analysis.panels import PANEL_A
from sheepshead.training.league_reports import (
    write_curve_png,
    write_generations_csv,
    write_report_md,
)

BASELINE_PROBE_SEED = 20260709  # greedy-health baseline of the resume ckpt
# The ALONE gate is applied relative to the resume checkpoint's own baseline:
# effective limit = max(config gate, baseline + this margin). A high-alone
# warm start (weak defender-field collaboration, which league training itself
# repairs) then never trips the halt, while a genuine regression still does.
ALONE_BASELINE_MARGIN = 5.0
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class NeedsReview(Exception):
    """Raised when the run must halt for operator judgment."""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = os.path.join("runs", args.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.orch_dir = os.path.join(self.run_dir, "orchestrator")
        os.makedirs(self.orch_dir, exist_ok=True)
        self.cfg = StopRuleConfig(
            min_generations=args.min_generations,
            max_generations=args.max_generations,
        )
        self.state = self._load_state()

    # ------------------------------------------------------------------ #
    # State / logging
    # ------------------------------------------------------------------ #
    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        with open(os.path.join(self.orch_dir, "orchestrator.log"), "a") as f:
            f.write(line + "\n")

    def _state_path(self) -> str:
        return os.path.join(self.orch_dir, "state.json")

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path()):
            with open(self._state_path()) as f:
                return json.load(f)
        return {
            "version": 1,
            "config": {
                k: v for k, v in vars(self.args).items() if not k.startswith("_")
            },
            "resume_episode": None,
            "anchor_coeff": self.args.anchor_coeff,
            "calibration": None,
            "generations": {},  # str(g) -> record
            "flat_history": [],  # flat flag per generation 1..
            "status": "running",
            "confirmation": None,
            "events": [],
        }

    def _save_state(self) -> None:
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self._state_path())

    def _event(self, msg: str) -> None:
        self.state["events"].append({"time": _now(), "msg": msg})
        self.log(msg)

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def boundary(self, g: int) -> int:
        return g * self.args.main_episodes

    def boundary_ckpt(self, g: int) -> str:
        return os.path.join(
            self.ckpt_dir,
            f"pfsp_{self.args.arch}_checkpoint_{self.boundary(g)}.pt",
        )

    def composite_ckpts(self, g: int) -> List[str]:
        """Last three checkpoints of generation g, oldest first; missing ones
        (gen-1 resume shortfall) substitute the boundary checkpoint."""
        out = []
        for back in (2, 1, 0):
            ep = self.boundary(g) - back * self.args.save_interval
            p = os.path.join(self.ckpt_dir, f"pfsp_{self.args.arch}_checkpoint_{ep}.pt")
            if not os.path.exists(p):
                self._event(
                    f"gen {g}: composite checkpoint {os.path.basename(p)} missing; "
                    "substituting the boundary checkpoint"
                )
                p = self.boundary_ckpt(g)
            out.append(p)
        return out

    def panel_npz(self, g: int) -> str:
        return os.path.join(self.orch_dir, f"panel_gen{g}.npz")

    def h2h_json(self, g: int) -> str:
        return os.path.join(self.orch_dir, f"h2h_gen{g}.json")

    def gen_record(self, g: int) -> dict:
        return self.state["generations"].setdefault(str(g), {})

    # ------------------------------------------------------------------ #
    # Pre-flight
    # ------------------------------------------------------------------ #
    def preflight(self) -> None:
        a = self.args
        if not os.path.exists(a.resume):
            raise SystemExit(f"--resume not found: {a.resume}")
        import torch

        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        ck_arch = ckpt.get("arch", "full")
        if ck_arch != a.arch:
            raise SystemExit(
                f"--arch {a.arch} but resume checkpoint records arch={ck_arch}"
            )
        del ckpt

        resume_episode = 0
        if "checkpoint_" in os.path.basename(a.resume):
            resume_episode = int(a.resume.split("_")[-1].split(".")[0])
        self.state["resume_episode"] = resume_episode
        if resume_episode > 0:
            self._event(
                f"resume parses to episode {resume_episode:,}: generation 1 will "
                f"train only {a.main_episodes - resume_episode:,} episodes "
                "(absolute boundaries). Prefer a final_<arch>.pt resume name."
            )
        if resume_episode >= a.main_episodes:
            raise SystemExit(
                "resume episode is past the first generation boundary; "
                "raise --main-episodes or rename the checkpoint"
            )

        missing = [p for p in PANEL_A if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"PANEL_A members missing: {missing}")

        if a.seed_checkpoints:
            spec = a.seed_checkpoints
            paths = (
                sorted(glob.glob(spec))
                if any(c in spec for c in "*?[")
                else sorted(glob.glob(os.path.join(spec, "*.pt")))
            )
            if not paths:
                raise SystemExit(f"--seed-checkpoints matched nothing: {spec}")
            import torch as _t

            for p in paths:
                seed_arch = _t.load(p, map_location="cpu", weights_only=False).get(
                    "arch", "full"
                )
                if seed_arch != a.arch:
                    self.log(
                        f"⚠️  seed checkpoint {p} has arch={seed_arch} (league "
                        "members are arch-aware; mixed seeding is unusual)"
                    )
            self.log(f"seed checkpoints: {len(paths)} files")

        if a.main_episodes % a.save_interval != 0:
            raise SystemExit("--main-episodes must be a multiple of --save-interval")
        if a.panel_deals % 6 != 0:
            raise SystemExit("--panel-deals must be divisible by 6 (2 modes x 3 ckpts)")
        self._save_state()

    def dry_run(self) -> None:
        self.log("DRY RUN — planned sequence:")
        self.log(
            f"  1. calibration probes at coeffs {self.args.anchor_coeffs} "
            f"({self.args.probe_episodes} eps each)"
            if self.args.anchor_coeff is None
            else f"  1. calibration SKIPPED (--anchor-coeff {self.args.anchor_coeff})"
        )
        self.log(
            f"  2. baseline endpoint: {self.args.resume} x3, "
            f"{self.args.panel_deals} deals, seed {PANEL_SEED}"
        )
        self.log(
            "  3. gen-1 trainer: "
            + " ".join(self.trainer_cmd(1, self.args.anchor_coeff or 1.0))
        )
        self.log(
            "  4. gen-g>=2 trainer: same minus seeding/anchor flags, resuming "
            f"from {os.path.basename(self.boundary_ckpt(1))}-style boundary ckpts"
        )
        self.log(f"  5. stop rule: {self.cfg}")
        self.log(f"  6. confirmation panel seed {CONFIRM_SEED}")

    # ------------------------------------------------------------------ #
    # Baseline health (shared by calibration and the per-generation halts)
    # ------------------------------------------------------------------ #
    def ensure_baseline_health(self) -> dict:
        """Greedy-health probe of the resume checkpoint, measured once."""
        if self.state.get("baseline_health"):
            return self.state["baseline_health"]
        from sheepshead.agent.ppo import load_agent
        from sheepshead.training.training_utils import greedy_health_probe

        agent = load_agent(self.args.resume)
        probe = greedy_health_probe(
            agent, n_games=self.args.baseline_probe_games, seed=BASELINE_PROBE_SEED
        )
        del agent
        self.state["baseline_health"] = {
            k: probe[k]
            for k in (
                "pick_rate",
                "alone_rate",
                "leaster_rate",
                "t0_trump_lead_rate",
                "play_logit_spread_med",
            )
        }
        self._save_state()
        self.log(
            f"baseline greedy health: pick {probe['pick_rate']:.1f}%, "
            f"alone {probe['alone_rate']:.1f}% "
            f"(effective alone limit {self._alone_limit():.1f}%)"
        )
        return self.state["baseline_health"]

    def _alone_limit(self) -> float:
        from sheepshead.training.config import PFSPHyperparams

        gate = PFSPHyperparams().greedy_gate_max_alone
        base = self.state.get("baseline_health", {}).get("alone_rate", 0.0)
        return max(gate, base + ALONE_BASELINE_MARGIN)

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def ensure_calibration(self) -> float:
        if self.state.get("anchor_coeff") is not None:
            return float(self.state["anchor_coeff"])

        from sheepshead.agent.ppo import PPOAgent, load_agent
        from sheepshead import ACTIONS
        import sheepshead.training.train_league_ppo as tlp
        from sheepshead.training.config import LeagueConfig
        from sheepshead.training.league import League

        a = self.args
        self.log(f"CALIBRATION: coeffs {a.anchor_coeffs}, {a.probe_episodes} eps each")
        baseline = self.ensure_baseline_health()

        greedy_interval = max(1, a.probe_episodes // 3)
        probes: List[ProbeSummary] = []
        cal_table = {}
        for coeff in a.anchor_coeffs:
            cal_dir = os.path.join(self.orch_dir, "calibration", f"c{coeff}")
            league_dir = os.path.join(cal_dir, "league")
            os.makedirs(cal_dir, exist_ok=True)
            progress_csv = os.path.join(cal_dir, "league_training_progress.csv")
            if not os.path.exists(progress_csv):
                self.log(f"  probe coeff={coeff} ...")
                league = League(league_dir, LeagueConfig())
                if len(league) == 0 and a.seed_checkpoints:
                    tlp._seed_league_from_checkpoints(league, a.seed_checkpoints)
                agent = PPOAgent(len(ACTIONS), critic_mode=a.critic_mode, arch=a.arch)
                agent.load(a.resume, load_optimizers=True)
                agent.set_anchor(load_agent(a.resume), coeff)
                ns = SimpleNamespace(
                    seed=a.seed,
                    schedule_horizon=a.schedule_horizon,
                    update_interval=a.update_interval,
                    save_interval=10**9,
                    snapshot_interval=10**9,
                    greedy_eval_interval=greedy_interval,
                    greedy_eval_games=a.greedy_eval_games,
                    num_workers=a.num_workers,
                    run_name=f"{a.run_name}_cal_c{coeff}",
                    critic_mode=a.critic_mode,
                    arch=a.arch,
                )
                # Worker weight-sync files land in runs/<run_name>/ — the real
                # trainer's main() creates it; the probe must too.
                os.makedirs(os.path.join("runs", ns.run_name), exist_ok=True)
                ratings = {m: league.rating_model.rating() for m in (0, 1)}
                tlp.run_main_phase(
                    agent,
                    league,
                    ratings,
                    ns,
                    start_episode=self.state["resume_episode"],
                    n_episodes=a.probe_episodes,
                    checkpoint_dir=cal_dir,
                )
                del agent
            summary = self._summarize_probe(coeff, cal_dir)
            probes.append(summary)
            cal_table[str(coeff)] = vars(summary)
            self.log(
                f"  coeff={coeff}: kl_last {summary.kl_last:.4f}, "
                f"kl_max {summary.kl_max:.4f}, violations {summary.gate_violations}, "
                f"pick {summary.final_pick_rate:.1f}%"
            )

        choice = pick_anchor_coeff(probes, baseline["pick_rate"])
        self.state["calibration"] = {
            "baseline_pick_rate": baseline["pick_rate"],
            "probes": cal_table,
            "chosen": choice.coeff,
            "qualified": choice.qualified,
            "reason": choice.reason,
        }
        self.state["anchor_coeff"] = choice.coeff
        self._save_state()
        self._event(f"calibration chose coeff={choice.coeff} ({choice.reason})")
        if not choice.qualified and not self.args.allow_calibration_fallback:
            raise NeedsReview(
                "no anchor coefficient met the calibration criteria "
                "(rerun with --allow-calibration-fallback to accept the largest)"
            )
        return choice.coeff

    def _summarize_probe(self, coeff: float, cal_dir: str) -> ProbeSummary:
        from sheepshead.training.config import PFSPHyperparams

        hp = PFSPHyperparams()
        kls: List[float] = []
        with open(os.path.join(cal_dir, "league_training_progress.csv")) as f:
            for row in csv.DictReader(f):
                v = row.get("anchor_kl", "")
                if v:
                    kls.append(float(v))
        if not kls:
            raise NeedsReview(
                f"calibration probe c{coeff} logged no anchor_kl values "
                "(is the anchor active and the trainer patched?)"
            )
        violations = 0
        final_pick = float("nan")
        alone_limit = self._alone_limit()  # baseline-relative
        greedy_csv = os.path.join(cal_dir, "greedy_health.csv")
        if os.path.exists(greedy_csv):
            with open(greedy_csv) as f:
                for row in csv.DictReader(f):
                    final_pick = float(row["pick_rate"])
                    if (
                        float(row["pick_rate"]) < hp.greedy_gate_min_pick
                        or float(row["alone_rate"]) > alone_limit
                        or float(row["t0_trump_lead_rate"])
                        > hp.greedy_gate_max_trump_lead
                        or float(row["play_logit_spread_med"])
                        < hp.greedy_gate_min_play_spread
                    ):
                        violations += 1
        return ProbeSummary(
            coeff=coeff,
            kl_last=float(np.mean(kls[-3:])),
            kl_max=float(np.max(kls)),
            gate_violations=violations,
            final_pick_rate=final_pick,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def trainer_cmd(self, g: int, anchor_coeff: Optional[float]) -> List[str]:
        a = self.args
        cmd = [
            sys.executable,
            "-m",
            "sheepshead.training.train_league_ppo",
            "--resume",
            self._resume_for(g),
            "--league-dir",
            a.league_dir,
            "--run-name",
            a.run_name,
            "--generations",
            "1",
            "--main-episodes",
            str(a.main_episodes),
            "--exploiter-episodes",
            str(a.exploiter_episodes),
            "--gate-deals",
            str(a.gate_deals),
            "--screen-deals",
            str(a.screen_deals),
            "--update-interval",
            str(a.update_interval),
            "--save-interval",
            str(a.save_interval),
            "--snapshot-interval",
            str(a.snapshot_interval),
            "--greedy-eval-interval",
            str(a.greedy_eval_interval),
            "--greedy-eval-games",
            str(a.greedy_eval_games),
            "--schedule-horizon",
            str(a.schedule_horizon),
            "--critic-mode",
            a.critic_mode,
            "--arch",
            a.arch,
            "--seed",
            str(a.seed),
            "--num-workers",
            str(a.num_workers),
        ]
        if a.leaster_watchdog:
            cmd.append("--leaster-watchdog")
        if a.smoke:
            cmd += ["--anchor-eval-ckpt", ""]
        if g == 1:
            if a.seed_checkpoints:
                cmd += ["--seed-checkpoints", a.seed_checkpoints]
            if anchor_coeff and anchor_coeff > 0:
                # Anchor to the ORIGINAL pre-registered resume checkpoint,
                # also on mid-generation crash restarts.
                cmd += [
                    "--anchor-coeff",
                    str(anchor_coeff),
                    "--anchor-ref",
                    a.resume,
                ]
        return cmd

    def _resume_for(self, g: int) -> str:
        """Latest checkpoint below the gen-g boundary (mid-generation crash
        restart), else the previous boundary / original resume."""
        lo = self.boundary(g - 1)
        hi = self.boundary(g)
        best_ep, best = -1, None
        for p in glob.glob(
            os.path.join(self.ckpt_dir, f"pfsp_{self.args.arch}_checkpoint_*.pt")
        ):
            try:
                ep = int(p.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            if lo <= ep < hi and ep > best_ep:
                best_ep, best = ep, p
        if best is not None:
            return best
        if g == 1:
            return self.args.resume
        raise NeedsReview(f"no resume checkpoint found for generation {g}")

    def _exploit_row(self, g: int) -> Optional[dict]:
        path = os.path.join(self.ckpt_dir, "exploitability.csv")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            for row in csv.DictReader(f):
                if int(row["generation"]) == g:
                    return row
        return None

    def ensure_trained(self, g: int) -> None:
        rec = self.gen_record(g)
        if os.path.exists(self.boundary_ckpt(g)) and self._exploit_row(g):
            self._record_exploiter(g)
            return
        if os.path.exists(self.boundary_ckpt(g)):
            self._event(
                f"gen {g}: boundary checkpoint exists but gate row missing; "
                "finishing exploiter phase in-process"
            )
            self._finish_exploiter_phase(g)
            self._record_exploiter(g)
            return

        anchor = self.state["anchor_coeff"] if g == 1 else None
        for attempt in (1, 2):
            cmd = self.trainer_cmd(g, anchor)
            log_path = os.path.join(self.orch_dir, f"gen{g}_train.log")
            self._event(
                f"gen {g} training (attempt {attempt}): {' '.join(cmd)} "
                f"[log: {log_path}]"
            )
            t0 = time.time()
            with open(log_path, "a") as logf:
                proc = subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONPATH="."),
                )
            rec["train_hours"] = (
                rec.get("train_hours", 0.0) + (time.time() - t0) / 3600.0
            )
            self._save_state()
            if proc.returncode == 0:
                break
            self._event(f"gen {g} trainer exited rc={proc.returncode}")
            if attempt == 2:
                raise NeedsReview(f"gen {g} trainer failed twice; see {log_path}")
        if not (os.path.exists(self.boundary_ckpt(g)) and self._exploit_row(g)):
            raise NeedsReview(
                f"gen {g} trainer finished but boundary checkpoint or gate row "
                "is missing"
            )
        self._record_exploiter(g)

    def _finish_exploiter_phase(self, g: int) -> None:
        """Replicates the trainer's post-main-phase block for the crash window
        between the boundary save and the gate row (train_league_ppo.main)."""
        import sheepshead.training.train_league_ppo as tlp
        from sheepshead.training.config import LeagueConfig
        from sheepshead.training.league import ROLE_PAST_MAIN, League

        a = self.args
        ns = SimpleNamespace(
            run_name=a.run_name,
            exploiter_episodes=a.exploiter_episodes,
            gate_deals=a.gate_deals,
            screen_deals=a.screen_deals,
            num_workers=a.num_workers,
            league_dir=a.league_dir,
            seed=a.seed,
            arch=a.arch,
            critic_mode=a.critic_mode,
        )
        episode = self.boundary(g)
        gate = tlp.run_exploiter_generation(ns, g, self.boundary_ckpt(g))
        path = os.path.join(self.ckpt_dir, "exploitability.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "generation",
                        "main_episode",
                        "edge",
                        "se",
                        "win_frac",
                        "passed",
                        "exploiter_ckpt",
                    ]
                )
            w.writerow(
                [
                    g,
                    episode,
                    f"{gate['edge']:.4f}",
                    f"{gate['se']:.4f}",
                    f"{gate['win_frac']:.3f}",
                    gate["passed"],
                    gate["exploiter_ckpt"],
                ]
            )
        league = League(a.league_dir, LeagueConfig())
        league.note_generation(g)
        if not gate["passed"]:
            snaps = [
                m
                for m in league.by_role(ROLE_PAST_MAIN)
                if m.meta.training_episodes == episode
            ]
            if snaps:
                league.promote_to_hof(snaps[-1].member_id)
                self._event(f"gen {g} survived its gate; {snaps[-1].member_id} -> HOF")

    def _record_exploiter(self, g: int) -> None:
        row = self._exploit_row(g)
        if row:
            self.gen_record(g)["exploiter"] = {
                "edge": float(row["edge"]),
                "se": float(row["se"]),
                "passed": row["passed"] in ("True", "true", "1"),
            }
            self._save_state()

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def ensure_endpoint(self, g: int) -> Endpoint:
        npz = self.panel_npz(g)
        if os.path.exists(npz):
            return load_endpoint(Path(npz))
        ckpts = [self.args.resume] * 3 if g == 0 else self.composite_ckpts(g)
        self.log(f"endpoint eval gen {g}: {self.args.panel_deals} deals ...")
        t0 = time.time()
        e = eval_endpoint(
            ckpts,
            n_deals=self.args.panel_deals,
            seed=PANEL_SEED,
            out_npz=Path(npz),
        )
        if g > 0:
            rec = self.gen_record(g)
            rec["eval_hours"] = rec.get("eval_hours", 0.0) + (time.time() - t0) / 3600.0
        return e

    def ensure_h2h(self, g: int) -> dict:
        path = self.h2h_json(g)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        prev = self.args.resume if g == 1 else self.boundary_ckpt(g - 1)
        self.log(f"h2h gen {g} vs gen {g - 1}: {self.args.h2h_deals} deals ...")
        res = h2h(self.boundary_ckpt(g), prev, n_deals=self.args.h2h_deals)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        return res

    def health_checks(self, g: int) -> None:
        """Post-generation collapse guards (the trainer only warns)."""
        if self.args.ignore_health_halt:
            return
        lo, hi = self.boundary(g - 1), self.boundary(g)
        greedy_csv = os.path.join(self.ckpt_dir, "greedy_health.csv")
        if os.path.exists(greedy_csv):
            from sheepshead.training.config import PFSPHyperparams

            hp = PFSPHyperparams()
            alone_limit = self._alone_limit()  # baseline-relative
            gates = {
                "pick": lambda r: float(r["pick_rate"]) < hp.greedy_gate_min_pick,
                "alone": lambda r: float(r["alone_rate"]) > alone_limit,
                "trump_lead": lambda r: (
                    float(r["t0_trump_lead_rate"]) > hp.greedy_gate_max_trump_lead
                ),
                "play_spread": lambda r: (
                    float(r["play_logit_spread_med"]) < hp.greedy_gate_min_play_spread
                ),
            }
            streaks = {k: 0 for k in gates}
            with open(greedy_csv) as f:
                for row in csv.DictReader(f):
                    if not (lo < int(row["episode"]) <= hi):
                        continue
                    for k, bad in gates.items():
                        streaks[k] = streaks[k] + 1 if bad(row) else 0
                        if streaks[k] >= 3:
                            raise NeedsReview(
                                f"gen {g}: >=3 consecutive greedy-health "
                                f"violations of gate '{k}'"
                            )
        progress_csv = os.path.join(self.ckpt_dir, "league_training_progress.csv")
        if os.path.exists(progress_csv):
            leasters = []
            with open(progress_csv) as f:
                for row in csv.DictReader(f):
                    if lo < int(row["episode"]) <= hi:
                        leasters.append(float(row["leaster_rate"]))
            if len(leasters) >= 40:
                start = float(np.mean(leasters[:20]))
                end = float(np.mean(leasters[-20:]))
                if end > 0.30 and end > start + 0.10:
                    raise NeedsReview(
                        f"gen {g}: leaster rate climbing toward PASS-collapse "
                        f"({start:.2f} -> {end:.2f})"
                    )

    # ------------------------------------------------------------------ #
    # Stop check
    # ------------------------------------------------------------------ #
    def stop_check(self, g: int) -> bool:
        """Verdict + decision for generation g. Returns True when the run is
        over (stop confirmed or cap reached without contradiction)."""
        rec = self.gen_record(g)
        endpoint = self.ensure_endpoint(g)
        h2h_res = self.ensure_h2h(g)
        rec["panel"] = {
            "mean": endpoint.score.mean,
            "lo": endpoint.score.lo,
            "hi": endpoint.score.hi,
            "se": endpoint.score.se,
            "modes": endpoint.mode_means,
            "trump_lead": endpoint.trump_lead,
        }
        rec["h2h"] = h2h_res

        per_deal = {
            h: load_endpoint(Path(self.panel_npz(h))).per_deal for h in range(0, g + 1)
        }
        means = {
            h: (
                self.state["generations"][str(h)]["panel"]["mean"]
                if h > 0
                else load_endpoint(Path(self.panel_npz(0))).score.mean
            )
            for h in range(0, g + 1)
        }
        boot_idx = _endpoint_boot_idx(self.args.panel_deals, N_BOOT, PANEL_SEED)
        v = flat_verdict(
            g, per_deal, means, h2h_res["edge"], h2h_res["se"], boot_idx, self.cfg
        )
        rec["verdict"] = verdict_to_dict(v)

        if len(self.state["flat_history"]) < g:
            self.state["flat_history"].append(v.flat)
        else:
            self.state["flat_history"][g - 1] = v.flat
        decision = decide_stop(self.state["flat_history"][:g], g, self.cfg)
        rec["stop_verdict"] = decision.reason
        self._save_state()
        self.write_reports()
        self._event(
            f"gen {g}: panel {endpoint.score.mean:+.4f} "
            f"[{endpoint.score.lo:+.4f},{endpoint.score.hi:+.4f}]  "
            f"h2h {h2h_res['edge']:+.3f}±{h2h_res['se']:.3f}  "
            f"flat={v.flat} streak={decision.flat_streak}  -> {decision.reason}"
        )

        if not decision.stop_candidate:
            return False
        return self.run_confirmation(g, decision.forced_by_cap)

    def run_confirmation(self, g: int, forced: bool) -> bool:
        self._event(f"CONFIRMATION after gen {g} (fresh deals, seed {CONFIRM_SEED})")
        conf: Dict[str, dict] = {}

        def conf_endpoint(h: int) -> Endpoint:
            npz = os.path.join(self.orch_dir, f"confirm_gen{h}.npz")
            if os.path.exists(npz):
                return load_endpoint(Path(npz))
            ckpts = [self.args.resume] * 3 if h == 0 else self.composite_ckpts(h)
            return eval_endpoint(
                ckpts,
                n_deals=self.args.panel_deals,
                seed=CONFIRM_SEED,
                out_npz=Path(npz),
            )

        e_g = conf_endpoint(g)
        e_gm2 = conf_endpoint(g - 2)
        best_g = max(
            (h for h in range(1, g + 1)),
            key=lambda h: self.state["generations"][str(h)]["panel"]["mean"],
        )
        conf["deploy_candidate"] = {"generation": best_g}
        if best_g not in (g, g - 2):
            e_best = conf_endpoint(best_g)
            conf["deploy_candidate"]["score"] = vars(e_best.score)
        else:
            src = e_g if best_g == g else e_gm2
            conf["deploy_candidate"]["score"] = vars(src.score)

        boot_idx = _endpoint_boot_idx(self.args.panel_deals, N_BOOT, CONFIRM_SEED)
        cv = confirmation_verdict(e_g.per_deal, e_gm2.per_deal, boot_idx, self.cfg)
        conf["gain_g_vs_gm2"] = {
            "mean": cv.stat.mean,
            "lo": cv.stat.lo,
            "hi": cv.stat.hi,
            "p_value": cv.stat.p_value,
            "contradiction": cv.contradiction,
        }
        conf["forced_by_cap"] = forced
        self.state["confirmation"] = conf

        if cv.contradiction and not forced:
            self._event(
                f"confirmation CONTRADICTS the plateau (fresh-deal gain "
                f"{cv.stat.mean:+.4f} [{cv.stat.lo:+.4f},{cv.stat.hi:+.4f}]); "
                "resetting flat streak and resuming training"
            )
            self.state["flat_history"][g - 2 : g] = [False, False]
            self._save_state()
            self.write_reports()
            return False

        self.state["status"] = "cap" if forced else "stopped"
        self._save_state()
        self.write_reports()
        self._event(
            f"STOP confirmed after generation {g} "
            f"(status={self.state['status']}); deploy candidate = gen {best_g}"
        )
        return True

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def write_reports(self) -> None:
        self._write_generations_csv()
        self._write_report_md()
        try:
            self._write_curve_png()
        except Exception as exc:  # plotting must never kill the run
            self.log(f"⚠️  curve plot failed: {exc}")

    def _write_generations_csv(self) -> None:
        write_generations_csv(
            self.orch_dir,
            self.state["generations"],
            self.state["anchor_coeff"],
            self.args.main_episodes,
            self.args.arch,
        )

    def _write_report_md(self) -> None:
        write_report_md(self.state, self.args, self.cfg, self._alone_limit(), self.orch_dir)

    def _write_curve_png(self) -> None:
        write_curve_png(self.orch_dir, self.state)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        if self.state["status"] in ("stopped", "cap"):
            self.log(
                f"run already concluded (status={self.state['status']}); see "
                f"{os.path.join(self.orch_dir, 'report.md')}"
            )
            return 0
        self.preflight()
        if self.args.dry_run:
            self.dry_run()
            return 0
        if self.state["status"] == "needs_review":
            self._event("resuming a needs_review run (operator override implied)")
            self.state["status"] = "running"
        try:
            self.ensure_baseline_health()  # anchors the relative ALONE gate
            self.ensure_calibration()
            self.ensure_endpoint(0)  # baseline
            g = 1
            while True:
                self.ensure_trained(g)
                self.health_checks(g)
                if self.stop_check(g):
                    return 0
                g += 1
        except NeedsReview as exc:
            self.state["status"] = "needs_review"
            self._event(f"NEEDS REVIEW: {exc}")
            self._save_state()
            self.write_reports()
            return 2


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extended league training with automatic stopping"
    )
    p.add_argument("--arch", default="full")
    p.add_argument(
        "--resume",
        required=True,
        help="checkpoint to resume (the ablation winner's final "
        "selfplay checkpoint; prefer a name without 'checkpoint_')",
    )
    p.add_argument(
        "--seed-checkpoints",
        default=None,
        help="glob/dir of selfplay checkpoints seeding the league",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--league-dir", default=None, help="default: runs/<run-name>/league")
    p.add_argument("--critic-mode", choices=["limited", "oracle"], default="oracle")
    p.add_argument("--main-episodes", type=int, default=1_000_000)
    p.add_argument("--exploiter-episodes", type=int, default=50_000)
    p.add_argument("--gate-deals", type=int, default=3000)
    p.add_argument("--screen-deals", type=int, default=200)
    p.add_argument("--update-interval", type=int, default=2048)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--snapshot-interval", type=int, default=50_000)
    p.add_argument("--greedy-eval-interval", type=int, default=50_000)
    p.add_argument("--greedy-eval-games", type=int, default=200)
    p.add_argument("--schedule-horizon", type=int, default=20_000_000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--leaster-watchdog",
        action="store_true",
        help="forward --leaster-watchdog to every generation's trainer "
        "(anchor-free PASS-collapse guard; see leaster_watchdog.py)",
    )
    # Calibration
    p.add_argument("--anchor-coeffs", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    p.add_argument(
        "--anchor-coeff",
        type=float,
        default=None,
        help="skip calibration and use this coefficient for gen 1",
    )
    p.add_argument("--probe-episodes", type=int, default=15_000)
    p.add_argument("--baseline-probe-games", type=int, default=400)
    p.add_argument("--allow-calibration-fallback", action="store_true")
    # Evaluation / stop rule
    p.add_argument("--panel-deals", type=int, default=4000)
    p.add_argument("--h2h-deals", type=int, default=2000)
    p.add_argument("--min-generations", type=int, default=4)
    p.add_argument("--max-generations", type=int, default=12)
    p.add_argument("--ignore-health-halt", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--smoke", action="store_true", help="minutes-long end-to-end loop check"
    )
    args = p.parse_args(argv)

    if args.smoke:
        args.main_episodes = 3000
        args.probe_episodes = 1000
        args.anchor_coeffs = [1.0]
        args.baseline_probe_games = 50
        args.greedy_eval_games = 50
        args.panel_deals = 60
        args.h2h_deals = 40
        args.exploiter_episodes = 500
        args.gate_deals = 40
        args.screen_deals = 0
        args.min_generations = 2
        args.max_generations = 3
        args.num_workers = 2
        args.save_interval = 1000
        args.snapshot_interval = 1000
        args.greedy_eval_interval = 1000
    if args.league_dir is None:
        args.league_dir = os.path.join("runs", args.run_name, "league")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    os.chdir(_REPO_ROOT)
    return Orchestrator(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
