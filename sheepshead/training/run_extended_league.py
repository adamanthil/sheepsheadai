#!/usr/bin/env python3
"""
Extended league training orchestrator with a quantitative stopping rule.

Trains the strongest agent train_league_ppo.py can produce from a selfplay
warm start, deciding for itself when learning is complete. The full recipe
(pre-registered in notebooks/Extended_League_202607.md):

  1. GEN 1        train_league_ppo.py subprocess, KL-anchored to the ORIGINAL
                  resume checkpoint (--anchor-coeff, default 1.0; guards the
                  warm-start transition against PASS-collapse), league seeded
                  from --seed-checkpoints.
  2. GEN 2..N     unanchored generations, resume-chained from each boundary
                  checkpoint (--generations 1 per invocation; the trainer's
                  absolute-episode boundary math keeps numbering/cadence).
  3. After each generation: composite fixed-panel endpoint + gen-vs-prev h2h
                  (league_progress_eval), stop-rule verdict (league_stopping),
                  telemetry (generations.csv, report.md, generations_curve.png).
  4. STOP         after two consecutive flat generations (post min-generations
                  floor) confirmed by a fresh-deal panel (seed 20260706), or at
                  the max-generations budget cap (relaunch with a higher cap to
                  continue). A confirmation contradiction resumes training.

The stopping rule measures learning completion only — it never compares
against an external target. The architecture is read from the resume
checkpoint; there is no --arch flag.

Reproducing from scratch (no artifacts from this repo's research runs):

  1. Train a selfplay warm start, e.g. ~400k episodes:
       uv run python -m sheepshead.training.train_selfplay_ppo \
         --arch perceiver-shared-v2 --run-name my_selfplay ...
  2. Run the orchestrator, seeding the league from the selfplay ladder.
     The default evaluation panel (analysis/panels.PANEL_A) references
     research-run checkpoints not shipped with the repo — pass your own
     fixed panel of >=4 checkpoints instead (e.g. your selfplay ladder;
     panel scores are relative to the field, so they are comparable only
     within one run, which is all the stopping rule needs):
       uv run python -m sheepshead.training.run_extended_league \
         --resume runs/my_selfplay/checkpoints/perceiver-shared-v2_checkpoint_400000.pt \
         --seed-checkpoints 'runs/my_selfplay/checkpoints/*.pt' \
         --panel runs/my_selfplay/checkpoints/perceiver-shared-v2_checkpoint_{100000,200000,300000,400000}.pt \
         --run-name my_league

Resume a standard CHECKPOINT, not final_*.pt — pre-2026-07-08 finals carry an
out-of-spec flush update (ablation notebook standing rule 5).

Crash-resumable: all phases are keyed to on-disk artifacts (checkpoints,
exploitability.csv rows, .npz/.json eval outputs) plus an atomic state.json in
runs/<run>/orchestrator/, so re-invoking with the same arguments always
converges to where it left off. Exit codes: 0 stopped cleanly, 2 needs_review.

Smoke test of the whole loop in minutes: add --smoke.
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
    h2h_duplicate,
    load_endpoint,
)
from sheepshead.training.league_stopping import (
    StopRuleConfig,
    confirmation_verdict,
    decide_stop,
    flat_verdict,
    resume_from_cap,
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
        # The architecture is whatever the resume checkpoint says it is —
        # it names every downstream artifact (pfsp_<arch>_checkpoint_*.pt).
        a.arch = ckpt.get("arch", "full")
        self.state["config"]["arch"] = a.arch
        del ckpt
        self.log(f"arch from resume checkpoint: {a.arch}")

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

        if len(a.panel) < 4:
            raise SystemExit(
                "--panel needs at least 4 checkpoints (each panel game seats "
                "the hero with 4 field members drawn without replacement)"
            )
        missing = [p for p in a.panel if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                f"evaluation panel members missing: {missing}\n"
                "The default panel (analysis/panels.PANEL_A) references frozen "
                "research-run checkpoints. Reproducing from scratch: pass "
                "--panel with any >=4 fixed checkpoints (e.g. your selfplay "
                "ladder) — the stopping rule only compares within one run."
            )

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
            f"  1. baseline endpoint: {self.args.resume} x3, "
            f"{self.args.panel_deals} deals, seed {PANEL_SEED}"
        )
        self.log(
            "  2. gen-1 trainer: "
            + " ".join(self.trainer_cmd(1, self.args.anchor_coeff))
        )
        self.log(
            "  3. gen-g>=2 trainer: same minus seeding/anchor flags, resuming "
            f"from {os.path.basename(self.boundary_ckpt(1))}-style boundary ckpts"
        )
        self.log(f"  4. stop rule: {self.cfg}")
        self.log(f"  5. confirmation panel seed {CONFIRM_SEED}")

    # ------------------------------------------------------------------ #
    # Baseline health (anchors the relative ALONE gate for the halts)
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
            panel_paths=tuple(self.args.panel),
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
        self.log(
            f"h2h gen {g} vs gen {g - 1} (duplicate-bridge): "
            f"{self.args.h2h_deals} deals/mode ..."
        )
        res = h2h_duplicate(
            self.boundary_ckpt(g), prev, n_deals_per_mode=self.args.h2h_deals
        )
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        return res

    def health_checks(self, g: int) -> None:
        """Post-generation health verdict, recorded once per generation.

        Greedy-gate streaks (pick/alone/trump-lead/play-spread) are
        diagnostic-grade signals from 200-game probes — the higher-powered
        endpoint instruments routinely contradict them — so they are recorded
        as warnings, never halts. Only the leaster trend halts: it targets
        the documented PASS-collapse attractor. The verdict is one-shot: a
        halt fires when the verdict is first recorded, and relaunching
        continues past it with the verdict on record (the run loop replays
        every generation, so re-checking would otherwise block forever).
        """
        rec = self.gen_record(g)
        if "health" in rec:
            return
        health: dict = {"warnings": [], "halt": None}
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
            worst = {k: 0 for k in gates}
            with open(greedy_csv) as f:
                for row in csv.DictReader(f):
                    if not (lo < int(row["episode"]) <= hi):
                        continue
                    for k, bad in gates.items():
                        streaks[k] = streaks[k] + 1 if bad(row) else 0
                        worst[k] = max(worst[k], streaks[k])
            for k, n in worst.items():
                if n >= 3:
                    health["warnings"].append(
                        f"{k}: {n} consecutive greedy-probe violations"
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
                    health["halt"] = (
                        f"leaster rate climbing toward PASS-collapse "
                        f"({start:.2f} -> {end:.2f})"
                    )
        rec["health"] = health
        self._save_state()
        for w in health["warnings"]:
            self._event(f"gen {g} health warning: {w}")
        if health["halt"]:
            if self.args.ignore_health_halt:
                self._event(
                    f"gen {g} health halt IGNORED (--ignore-health-halt): "
                    f"{health['halt']}"
                )
            else:
                raise NeedsReview(
                    f"gen {g}: {health['halt']} — verdict recorded; "
                    "relaunching continues past this halt"
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
                panel_paths=tuple(self.args.panel),
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
        if resume_from_cap(
            self.state["status"],
            len(self.state["flat_history"]),
            self.args.max_generations,
        ):
            self._event(
                f"cap raised: {len(self.state['flat_history'])} generations "
                f"recorded, max_generations now {self.args.max_generations}; "
                "resuming"
            )
            self.state["status"] = "running"
            self._save_state()
        if self.state["status"] in ("stopped", "cap"):
            hint = (
                " — relaunch with a higher --max-generations to continue"
                if self.state["status"] == "cap"
                else ""
            )
            self.log(
                f"run already concluded (status={self.state['status']}); see "
                f"{os.path.join(self.orch_dir, 'report.md')}{hint}"
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
    p.add_argument(
        "--resume",
        required=True,
        help="selfplay warm-start checkpoint to resume; the architecture is "
        "read from it (prefer a name without 'checkpoint_')",
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
    p.add_argument(
        "--anchor-coeff",
        type=float,
        default=1.0,
        help="gen-1 bidding-head KL anchor to the resume checkpoint "
        "(0 disables; 1.0 validated by the July-2026 ablation stage 1)",
    )
    p.add_argument("--baseline-probe-games", type=int, default=400)
    # Evaluation / stop rule
    p.add_argument(
        "--panel",
        nargs="+",
        default=list(PANEL_A),
        help="fixed evaluation-panel checkpoints (>=4); default is the frozen "
        "PANEL-A research anchors — reproductions without those files should "
        "pass their own selfplay ladder here",
    )
    # 3996, not 4000: must divide evenly over 2 modes x 3 composite ckpts
    p.add_argument("--panel-deals", type=int, default=3996)
    p.add_argument(
        "--h2h-deals",
        type=int,
        default=2000,
        help="deals PER MODE for the duplicate-bridge h2h (amendment 2026-07-19)",
    )
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
