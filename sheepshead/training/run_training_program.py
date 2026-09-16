#!/usr/bin/env python3
"""The release-candidate training program, end to end
(Training_Program_Redesign_202609 §4-§6): one resumable orchestrator over
the five phases, driven by a ``ProgramConfig`` that doubles as the
pre-registration artifact.

  0  bootstrap        train_ppo --phase bootstrap (shaped self-play, 400k)
  1  oracle           pretrain_oracle generate + pretrain on the bootstrap
  2  league           train_ppo --phase league, one generation per
                      invocation, judged by the marginal-value handoff rule
                      (stop_rules.py): duplicate h2h vs the previous
                      generation, fresh-deal confirmation on a miss, one
                      play-target entropy step, then handoff from the last
                      SETTLED boundary. Panel endpoint, convention battery,
                      B2 hard bounds and the review gates ride along.
  3  policy iteration distill_corpus -> policy_iteration all -> cert ->
                      train_ppo --phase bidding -> cert, per iteration,
                      until the certified gain is below 2 SE twice.
  4  final            duplicate h2h vs the reference agents, the one-time
                      exploitability audit, release.pt.

Everything is keyed to on-disk artifacts plus an atomic state.json under
runs/<run-name>/program/, so re-invoking with the same config converges to
where it left off. Exit codes: 0 finished, 2 needs operator review.

Usage:
  uv run python -m sheepshead.training.run_training_program --run-name rc_202609
  uv run python -m sheepshead.training.run_training_program --smoke --run-name _smoke
  uv run python -m sheepshead.training.run_training_program --config my.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from sheepshead.analysis.league_progress_eval import (
    CONFIRM_SEED,
    PANEL_SEED,
    eval_endpoint,
    h2h_duplicate,
    load_endpoint,
)
from sheepshead.training.entropy_controller import EntropyTargetController
from sheepshead.training.program_config import ProgramConfig
from sheepshead.training.stop_rules import (
    HandoffRuleConfig,
    IterationRuleConfig,
    decide_handoff,
    generation_verdict,
    iteration_stop,
    settled_generation,
)
from sheepshead.training.train_ppo import episode_of

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONVENTION_SEEDS = (98765, 98766, 98767, 98768)


class NeedsReview(Exception):
    """The run must halt for operator judgment."""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _hours_since(stamp: str) -> float:
    return (
        time.time() - time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    ) / 3600.0


def _fmt_hours(hours: float) -> str:
    if hours < 1.0:
        return f"{hours * 60:.0f} min"
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} d ({hours:.0f} h)"


class Program:
    def __init__(self, cfg: ProgramConfig):
        self.cfg = cfg
        self.run_dir = os.path.join("runs", cfg.run_name)
        self.program_dir = os.path.join(self.run_dir, "program")
        os.makedirs(self.program_dir, exist_ok=True)
        self.state = self._load_state()

    # ------------------------------------------------------------------ #
    # State / logging / subprocesses
    # ------------------------------------------------------------------ #
    def log(self, msg: str) -> None:
        """One timestamped line per line of ``msg`` in program.log (and on
        stdout), so a multi-line entry still greps line by line."""
        stamp = _now()
        lines = [f"[{stamp}] {part}" for part in msg.split("\n")]
        print("\n".join(lines), flush=True)
        with open(os.path.join(self.program_dir, "program.log"), "a") as f:
            f.write("\n".join(lines) + "\n")

    RULE = "=" * 72

    def banner(self, title: str, *details: str) -> None:
        """A phase boundary: a ruled header (grep for "] ==") with the facts
        the phase starts from indented beneath it."""
        self.log(self.RULE)
        self.log(title)
        for detail in details:
            self.log(f"    {detail}")
        self.log(self.RULE)

    def section(self, title: str) -> None:
        """A sub-boundary inside a phase (a generation, an iteration)."""
        self.log(f"---- {title} " + "-" * max(4, 66 - len(title)))

    def skip(self, label: str, evidence: str) -> None:
        """A stage the resume found already complete."""
        self.log(f"↷ {label}: already complete ({evidence})")

    PHASE_TITLES = {
        "bootstrap": "PHASE 0 — SHAPED SELF-PLAY BOOTSTRAP",
        "oracle": "PHASE 1 — ORACLE PRETRAINING",
        "league": "PHASE 2 — TERMINAL-ONLY LEAGUE POLICY GRADIENT",
        "policy_iteration": "PHASE 3 — SEARCH-Q POLICY ITERATION",
        "final": "PHASE 4 — FINAL CERTIFICATION",
    }

    def begin_phase(self, name: str, *details: str) -> None:
        """Enter a phase: record its first start (a resume keeps it) and
        print the phase banner."""
        self.state["phase"] = name
        times = self.state.setdefault("phase_times", {}).setdefault(name, {})
        if "started" in times:
            since = f"resumed; first started {times['started']}"
        else:
            times["started"] = _now()
            since = f"started {times['started']}"
        self.banner(self.PHASE_TITLES[name], *details, since)
        self._save_state()

    def end_phase(self, name: str) -> None:
        """Leave a phase: record the finish and print the elapsed time —
        wall clock since the first start (pauses included) and the hours
        its stage subprocesses actually ran."""
        times = self.state["phase_times"][name]
        times["finished"] = _now()
        times["wall_hours"] = _hours_since(times["started"])
        self.log(
            f"✔ {self.PHASE_TITLES[name]} complete: "
            f"{_fmt_hours(times['wall_hours'])} wall clock since {times['started']}"
            f" (stages ran {_fmt_hours(times.get('stage_hours', 0.0))})"
        )
        self._save_state()

    def phase_time_lines(self) -> list[str]:
        """One line per recorded phase for the closing banner / report."""
        lines = []
        for name in self.PHASE_TITLES:
            times = self.state.get("phase_times", {}).get(name)
            if not times or "started" not in times:
                continue
            wall = times.get("wall_hours")
            wall_s = _fmt_hours(wall) if wall is not None else "in progress"
            lines.append(
                f"{name:17s} {wall_s:>16s} wall   stages {_fmt_hours(times.get('stage_hours', 0.0)):>12s}"
                f"   {times['started']} -> {times.get('finished', '…')}"
            )
        started = self.state.get("started")
        if started:
            lines.append(
                f"{'program':17s} {_fmt_hours(_hours_since(started)):>16s} wall   since {started}"
            )
        return lines

    def _state_path(self) -> str:
        return os.path.join(self.program_dir, "state.json")

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path()):
            with open(self._state_path()) as f:
                return json.load(f)
        return {
            "version": 1,
            "status": "running",
            "phase": "bootstrap",
            "league": {"generations": {}, "step_generation": None, "handoff": None},
            "policy_iteration": {"iterations": {}, "theta": None},
            "final": {},
            "events": [],
            "started": _now(),
            "phase_times": {},
        }

    def _save_state(self) -> None:
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self._state_path())
        with open(os.path.join(self.program_dir, "config.json"), "w") as f:
            f.write(self.cfg.to_json())

    def _event(
        self, msg: str, log_msg: str | None = None, decision: bool = False
    ) -> None:
        """Record ``msg`` in the state's event list (plain, what report.md
        shows) and write ``log_msg`` (default ``msg``) to program.log;
        ``decision`` marks a line the operator would look for first."""
        self.state["events"].append({"time": _now(), "msg": msg})
        rendered = log_msg if log_msg is not None else msg
        self.log(f"★ {rendered}" if decision else rendered)
        self._save_state()

    def _run(self, label: str, cmd: list[str], log_name: str) -> None:
        log_path = os.path.join(self.program_dir, log_name)
        self._event(
            f"{label}: {' '.join(cmd)} [log: {log_path}]",
            log_msg=f"▶ {label}  (stdout -> {log_path})\n    $ {' '.join(cmd)}",
        )
        t0 = time.time()
        with open(log_path, "a") as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH="."),
            )
        hours = (time.time() - t0) / 3600.0
        times = self.state.setdefault("phase_times", {}).setdefault(
            self.state["phase"], {}
        )
        times["stage_hours"] = times.get("stage_hours", 0.0) + hours
        if proc.returncode != 0:
            raise NeedsReview(f"{label} exited rc={proc.returncode}; see {log_path}")
        self._event(
            f"{label}: done in {hours:.2f} h",
            log_msg=f"✔ {label}: done in {hours:.2f} h",
        )

    def _py(self, module: str, *args: str) -> list[str]:
        return [sys.executable, "-m", module, *args]

    def _worker_flags(
        self, device: str | None = None, compile_: str | None = None
    ) -> list[str]:
        """Worker-pool flags; a phase may override the program-level device
        and compile setting (the league generations run on MPS + compile)."""
        cfg = self.cfg
        flags = ["--num-workers", str(cfg.num_workers), "--seed", str(cfg.seed)]
        device = device or cfg.worker_device
        compile_ = compile_ or cfg.worker_compile
        if device:
            flags += ["--worker-device", device]
        if compile_:
            flags += ["--worker-compile", compile_]
        return flags

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def bootstrap_dir(self) -> str:
        return os.path.join(self.run_dir, "bootstrap")

    @property
    def bootstrap_final(self) -> str:
        return os.path.join(self.bootstrap_dir, "final.pt")

    @property
    def seeds_dir(self) -> str:
        return os.path.join(self.run_dir, "seeds")

    @property
    def oracle_init(self) -> str:
        return os.path.join(self.run_dir, "oracle", "oracle_init.pt")

    @property
    def league_dir(self) -> str:
        return os.path.join(self.run_dir, "league")

    @property
    def league_ckpt_dir(self) -> str:
        return os.path.join(self.league_dir, "checkpoints")

    def boundary(self, g: int) -> int:
        return g * self.cfg.league.generation_episodes

    def boundary_ckpt(self, g: int) -> str:
        return os.path.join(self.league_ckpt_dir, f"checkpoint_{self.boundary(g)}.pt")

    def iter_dir(self, k: int) -> str:
        return os.path.join(self.run_dir, "pi", f"iter{k}")

    # ------------------------------------------------------------------ #
    # Phase 0: bootstrap
    # ------------------------------------------------------------------ #
    def _latest_checkpoint_below(self, ckpt_dir: str, limit: int) -> Optional[str]:
        best_ep, best = -1, None
        for p in glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")):
            ep = episode_of(p)
            if ep < limit and ep > best_ep:
                best_ep, best = ep, p
        return best

    def ensure_bootstrap(self) -> None:
        cfg = self.cfg
        if os.path.exists(self.bootstrap_final):
            self.skip("bootstrap", self.bootstrap_final)
            return
        cmd = self._py(
            "sheepshead.training.train_ppo",
            "--phase",
            "bootstrap",
            "--arch",
            cfg.arch,
            "--run-name",
            f"{cfg.run_name}/bootstrap",
            "--until",
            str(cfg.bootstrap.episodes),
            "--save-interval",
            str(cfg.bootstrap.save_interval),
            "--greedy-eval-interval",
            str(cfg.bootstrap.greedy_eval_interval),
            "--greedy-eval-games",
            str(cfg.bootstrap.greedy_eval_games),
            *self._worker_flags(),
        )
        if cfg.bootstrap.update_interval:
            cmd += ["--update-interval", str(cfg.bootstrap.update_interval)]
        resume = self._latest_checkpoint_below(
            os.path.join(self.bootstrap_dir, "checkpoints"), cfg.bootstrap.episodes
        )
        if resume:
            cmd += ["--resume", resume]
        self._run("bootstrap", cmd, "bootstrap.log")
        if not os.path.exists(self.bootstrap_final):
            raise NeedsReview("bootstrap finished without final.pt")
        # Health gate only (§3.4): the attractor must have been escaped.
        greedy = os.path.join(self.bootstrap_dir, "checkpoints", "greedy_health.csv")
        if os.path.exists(greedy):
            with open(greedy) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = float(rows[-1]["leaster_rate"])
                self.state.setdefault("bootstrap", {})["final_leaster_rate"] = last
                if last > cfg.bootstrap.max_final_leaster_rate:
                    raise NeedsReview(
                        f"bootstrap ended with greedy leaster rate {last:.1f}% "
                        "(all-PASS attractor not escaped)"
                    )
        self._save_state()

    @property
    def seeds_glob(self) -> str:
        return os.path.join(self.seeds_dir, "*.pt")

    def ensure_seeds(self) -> str:
        """Materialize the generation-1 population seeds (copies of the
        bootstrap final); called right before generation 1 trains, never
        while merely rendering its command (--dry-run)."""
        os.makedirs(self.seeds_dir, exist_ok=True)
        for i in range(self.cfg.league.seed_copies):
            dst = os.path.join(self.seeds_dir, f"seed_{i}.pt")
            if not os.path.exists(dst):
                shutil.copyfile(self.bootstrap_final, dst)
        return self.seeds_glob

    # ------------------------------------------------------------------ #
    # Phase 1: oracle pretraining
    # ------------------------------------------------------------------ #
    def ensure_oracle(self) -> None:
        cfg = self.cfg
        if os.path.exists(self.oracle_init):
            self.skip("oracle pretraining", self.oracle_init)
            return
        dataset = os.path.join(self.run_dir, "oracle", "dataset.pt")
        if not os.path.exists(dataset):
            self._run(
                "oracle generate",
                self._py(
                    "sheepshead.training.pretrain_oracle",
                    "generate",
                    "--ckpt",
                    self.bootstrap_final,
                    "--episodes",
                    str(cfg.oracle.episodes),
                    "--workers",
                    str(cfg.num_workers),
                    "--gamma",
                    str(cfg.oracle.gamma),
                    "--seed",
                    str(cfg.seed),
                    "--out",
                    dataset,
                ),
                "oracle.log",
            )
        self._run(
            "oracle pretrain",
            self._py(
                "sheepshead.training.pretrain_oracle",
                "pretrain",
                "--dataset",
                dataset,
                "--max-epochs",
                str(cfg.oracle.max_epochs),
                "--patience",
                str(cfg.oracle.patience),
                "--seed",
                str(cfg.seed),
                "--out",
                self.oracle_init,
            ),
            "oracle.log",
        )

    # ------------------------------------------------------------------ #
    # Phase 2: league
    # ------------------------------------------------------------------ #
    def league_trainer_cmd(self, g: int, resume: str) -> list[str]:
        cfg = self.cfg
        cmd = self._py(
            "sheepshead.training.train_ppo",
            "--phase",
            "league",
            "--resume",
            resume,
            "--run-name",
            f"{cfg.run_name}/league",
            "--league-dir",
            os.path.join(self.league_dir, "league"),
            "--until",
            str(self.boundary(g)),
            "--save-interval",
            str(cfg.league.save_interval),
            "--snapshot-interval",
            str(cfg.league.snapshot_interval),
            "--greedy-eval-interval",
            str(cfg.league.greedy_eval_interval),
            "--greedy-eval-games",
            str(cfg.league.greedy_eval_games),
            "--entropy-play-floor",
            str(cfg.league.entropy_play_floor),
            *self._worker_flags(cfg.league.worker_device, cfg.league.worker_compile),
        )
        if cfg.league.update_interval:
            cmd += ["--update-interval", str(cfg.league.update_interval)]
        if g == 1:
            # Generation 1: seeded population, pretrained oracle, fixed
            # entropy coefficients (the controller attaches at the boundary).
            cmd += [
                "--seed-checkpoints",
                self.seeds_glob,
                "--oracle-init",
                self.oracle_init,
                "--no-entropy-controller",
            ]
        return cmd

    def _gen_record(self, g: int) -> dict:
        return self.state["league"]["generations"].setdefault(str(g), {})

    def _prev_ckpt(self, g: int) -> str:
        return self.bootstrap_final if g == 1 else self.boundary_ckpt(g - 1)

    def ensure_generation_trained(self, g: int) -> None:
        if os.path.exists(self.boundary_ckpt(g)):
            self.skip(f"league gen {g} training", self.boundary_ckpt(g))
            return
        resume = self._latest_checkpoint_below(
            self.league_ckpt_dir, self.boundary(g)
        ) or self._prev_ckpt(g)
        rec = self._gen_record(g)
        if g == 1:
            self.ensure_seeds()
        t0 = time.time()
        self._run(
            f"league gen {g}", self.league_trainer_cmd(g, resume), f"league_gen{g}.log"
        )
        rec["train_hours"] = rec.get("train_hours", 0.0) + (time.time() - t0) / 3600.0
        if not os.path.exists(self.boundary_ckpt(g)):
            raise NeedsReview(
                f"gen {g} trainer finished without its boundary checkpoint"
            )
        self._save_state()

    def _h2h(self, g: int, seed: int, tag: str) -> dict:
        path = os.path.join(self.program_dir, f"h2h_gen{g}{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        self.log(
            f"h2h gen {g} vs previous (duplicate, seed {seed}): "
            f"{self.cfg.league.h2h_deals} deals/mode ..."
        )
        res = h2h_duplicate(
            self.boundary_ckpt(g),
            self._prev_ckpt(g),
            n_deals_per_mode=self.cfg.league.h2h_deals,
            seed=seed,
        )
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        return res

    def _composite_ckpts(self, g: int) -> list[str]:
        out = []
        for back in (2, 1, 0):
            ep = self.boundary(g) - back * self.cfg.league.save_interval
            p = os.path.join(self.league_ckpt_dir, f"checkpoint_{ep}.pt")
            out.append(p if os.path.exists(p) else self.boundary_ckpt(g))
        return out

    def _panel_read(
        self, ckpts: list[str], members: list[str], label: str, npz_name: str
    ) -> Optional[dict]:
        """Anchored-gauntlet endpoint of ``ckpts`` (a 3-checkpoint composite)
        on the ``members`` field; cached in program/<npz_name>."""
        cfg = self.cfg.league
        missing = [p for p in members if not os.path.exists(p)]
        if len(members) < 4 or missing:
            self.log(f"{label} skipped: {len(members)} members, missing {missing}")
            return None
        npz = Path(self.program_dir) / npz_name
        if npz.exists():
            e = load_endpoint(npz)
        else:
            self.log(f"{label}: {cfg.panel_deals} deals ...")
            e = eval_endpoint(
                ckpts,
                n_deals=cfg.panel_deals,
                seed=PANEL_SEED,
                panel_paths=tuple(members),
                out_npz=npz,
            )
        return {
            "mean": e.score.mean,
            "lo": e.score.lo,
            "hi": e.score.hi,
            "se": e.score.se,
            "modes": e.mode_means,
            "trump_lead": e.trump_lead,
        }

    def _panel(self, g: int) -> Optional[dict]:
        """PANEL-A (the pre-registered yardstick; the gen-2 gate reads it).
        Without a configured panel the run's own bootstrap seeds serve."""
        cfg = self.cfg.league
        panel = cfg.panel or sorted(glob.glob(os.path.join(self.seeds_dir, "*.pt")))
        return self._panel_read(
            self._composite_ckpts(g), panel, f"panel gen {g}", f"panel_gen{g}.npz"
        )

    def _panel_b(self, g: int) -> Optional[dict]:
        """PANEL-B (tentative): strong-skill / cross-ecology field, recorded
        only — no gate reads it. Not run when the list is empty."""
        if not self.cfg.league.panel_b:
            return None
        return self._panel_read(
            self._composite_ckpts(g),
            self.cfg.league.panel_b,
            f"panel-B gen {g}",
            f"panelB_gen{g}.npz",
        )

    def _conventions(self, ckpt: str, label: str) -> dict:
        """Convention battery: greedy probes on fixed seeds, means across
        seeds (single reads are luck-of-phase)."""
        path = os.path.join(self.program_dir, f"conventions_{label}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        from sheepshead.agent.ppo import load_agent
        from sheepshead.training.training_utils import greedy_health_probe

        cfg = self.cfg.league
        agent = load_agent(ckpt)
        probes = [
            greedy_health_probe(agent, n_games=cfg.convention_probe_games, seed=s)
            for s in CONVENTION_SEEDS[: cfg.convention_probe_seeds]
        ]
        keys = (
            "pick_rate",
            "alone_rate",
            "leaster_rate",
            "t0_trump_lead_rate",
            "partner_trump_lead_rate",
            "called_suit_lead_rate",
            "play_logit_spread_med",
        )
        out: dict = {k: float(np.mean([p[k] for p in probes])) for k in keys}
        out["probes"] = probes
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self.log(
            f"conventions {label}: " + "  ".join(f"{k} {out[k]:.1f}" for k in keys)
        )
        return out

    def _health(self, g: int) -> None:
        """Leaster-trend halt on the generation's progress rows (the
        PASS-collapse attractor); greedy-gate streaks are warnings."""
        progress = os.path.join(self.league_ckpt_dir, "training_progress.csv")
        if not os.path.exists(progress):
            return
        lo, hi = self.boundary(g - 1), self.boundary(g)
        leasters = []
        with open(progress) as f:
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

    def judge_generation(self, g: int) -> str:
        """Evaluate generation g and return the handoff action."""
        cfg = self.cfg.league
        rule = HandoffRuleConfig(
            min_gain=cfg.h2h_min_gain,
            ci_z=cfg.h2h_ci_z,
            min_generations=cfg.min_generations,
            max_generations=cfg.max_generations,
        )
        rec = self._gen_record(g)
        if "decision" in rec:
            self.skip(f"gen {g} evaluation", f"decision {rec['decision']['action']}")
            return rec["decision"]["action"]
        self._health(g)
        primary = self._h2h(g, PANEL_SEED, "")
        verdict = generation_verdict(g, primary["edge"], primary["se"], rule)
        confirm = None
        if verdict.needs_confirmation:
            confirm = self._h2h(g, CONFIRM_SEED, "_confirm")
            verdict = generation_verdict(
                g,
                primary["edge"],
                primary["se"],
                rule,
                (confirm["edge"], confirm["se"]),
            )
        rec["h2h"] = primary
        rec["h2h_confirm"] = confirm
        rec["improving"] = verdict.improving
        rec["panel"] = self._panel(g)
        rec["panel_b"] = self._panel_b(g)
        rec["conventions"] = self._conventions(self.boundary_ckpt(g), f"gen{g}")
        conv = rec["conventions"]
        if conv["partner_trump_lead_rate"] < cfg.partner_trump_lead_min:
            raise NeedsReview(
                f"gen {g}: partner trump lead {conv['partner_trump_lead_rate']:.1f}% "
                f"below the B2 bound {cfg.partner_trump_lead_min}"
            )
        if conv["t0_trump_lead_rate"] > cfg.defender_t0_trump_lead_max:
            raise NeedsReview(
                f"gen {g}: defender t0 trump lead {conv['t0_trump_lead_rate']:.1f}% "
                f"above the B2 bound {cfg.defender_t0_trump_lead_max}"
            )
        if g == 2 and rec["panel"] is not None:
            if rec["panel"]["mean"] < self.cfg.gates.gen2_panel_min:
                self._save_state()
                raise NeedsReview(
                    f"gen-2 review gate: panel {rec['panel']['mean']:+.4f} below "
                    f"{self.cfg.gates.gen2_panel_min:+.3f}"
                )
        history = [
            bool(self.state["league"]["generations"][str(h)]["improving"])
            for h in range(1, g + 1)
        ]
        decision = decide_handoff(
            history, g, self.state["league"]["step_generation"], rule
        )
        rec["decision"] = {"action": decision.action, "reason": decision.reason}
        self._event(
            decision=True,
            msg=f"gen {g}: h2h {primary['edge']:+.4f}±{primary['se']:.4f}"
            + (
                f" (confirm {confirm['edge']:+.4f}±{confirm['se']:.4f})"
                if confirm
                else ""
            )
            + f" improving={verdict.improving}"
            + (f" panel {rec['panel']['mean']:+.4f}" if rec["panel"] else "")
            + (
                f" panel-B {panel_b['mean']:+.4f}"
                if (panel_b := rec.get("panel_b"))
                else ""
            )
            + f" -> {decision.action} ({decision.reason})",
        )
        if decision.action == "entropy_step":
            self._entropy_step(g)
        self._write_generations_csv()
        return decision.action

    def _entropy_step(self, g: int) -> None:
        path = os.path.join(self.league_ckpt_dir, "entropy_controller.json")
        if not os.path.exists(path):
            raise NeedsReview(
                f"gen {g}: entropy step requested but no controller sidecar"
            )
        ctrl = EntropyTargetController.load(path)
        moved = ctrl.step_targets()
        ctrl.save(path)
        self.state["league"]["step_generation"] = g
        self._event(
            f"gen {g}: play-target step "
            + (
                ", ".join(f"{h} {o:.3f}->{n:.3f}" for h, (o, n) in moved.items())
                if moved
                else "(targets not yet initialized; nothing to step)"
            )
        )

    def _write_generations_csv(self) -> None:
        cols = [
            "generation",
            "boundary_episode",
            "h2h_edge",
            "h2h_se",
            "confirm_edge",
            "confirm_se",
            "improving",
            "panel_mean",
            "panel_lo",
            "panel_hi",
            "panel_b_mean",
            "panel_b_lo",
            "panel_b_hi",
            "partner_trump_lead",
            "t0_trump_lead",
            "called_suit_lead",
            "pick_rate",
            "leaster_rate",
            "decision",
            "train_hours",
        ]
        gens = self.state["league"]["generations"]
        with open(
            os.path.join(self.program_dir, "generations.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for g in sorted(int(k) for k in gens):
                rec = gens[str(g)]
                if "decision" not in rec:
                    continue
                conv = rec.get("conventions") or {}
                panel = rec.get("panel") or {}
                panel_b = rec.get("panel_b") or {}
                confirm = rec.get("h2h_confirm") or {}
                w.writerow(
                    {
                        "generation": g,
                        "boundary_episode": self.boundary(g),
                        "h2h_edge": f"{rec['h2h']['edge']:.4f}",
                        "h2h_se": f"{rec['h2h']['se']:.4f}",
                        "confirm_edge": f"{confirm['edge']:.4f}" if confirm else "",
                        "confirm_se": f"{confirm['se']:.4f}" if confirm else "",
                        "improving": rec["improving"],
                        "panel_mean": f"{panel['mean']:.4f}" if panel else "",
                        "panel_lo": f"{panel['lo']:.4f}" if panel else "",
                        "panel_hi": f"{panel['hi']:.4f}" if panel else "",
                        "panel_b_mean": f"{panel_b['mean']:.4f}" if panel_b else "",
                        "panel_b_lo": f"{panel_b['lo']:.4f}" if panel_b else "",
                        "panel_b_hi": f"{panel_b['hi']:.4f}" if panel_b else "",
                        "partner_trump_lead": f"{conv.get('partner_trump_lead_rate', 0):.1f}",
                        "t0_trump_lead": f"{conv.get('t0_trump_lead_rate', 0):.2f}",
                        "called_suit_lead": f"{conv.get('called_suit_lead_rate', 0):.1f}",
                        "pick_rate": f"{conv.get('pick_rate', 0):.1f}",
                        "leaster_rate": f"{conv.get('leaster_rate', 0):.1f}",
                        "decision": rec["decision"]["action"],
                        "train_hours": f"{rec.get('train_hours', 0.0):.2f}",
                    }
                )

    def run_league(self) -> str:
        """Generations until handoff; returns theta_0 (the handoff checkpoint)."""
        league = self.state["league"]
        if league.get("handoff"):
            return league["handoff"]["checkpoint"]
        g = 1
        while True:
            self.section(
                f"league generation {g}: train to episode {self.boundary(g):,}"
            )
            self.ensure_generation_trained(g)
            action = self.judge_generation(g)
            if action == "handoff":
                settled = settled_generation(g, league["step_generation"])
                ckpt = self.boundary_ckpt(settled)
                league["handoff"] = {"generation": settled, "checkpoint": ckpt}
                self._event(
                    f"HANDOFF after gen {g}: theta_0 = gen {settled} ({ckpt})",
                    decision=True,
                )
                self._handoff_gate(ckpt)
                self._save_state()
                return ckpt
            g += 1

    def _handoff_gate(self, ckpt: str) -> None:
        ref = self.cfg.gates.handoff_reference
        if not ref or not os.path.exists(ref):
            self.log(f"handoff gate skipped (reference {ref!r} not available)")
            return
        path = os.path.join(self.program_dir, "h2h_handoff_vs_reference.json")
        if os.path.exists(path):
            with open(path) as f:
                res = json.load(f)
        else:
            res = h2h_duplicate(ckpt, ref, n_deals_per_mode=self.cfg.league.h2h_deals)
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
        self.state["league"]["handoff_gate"] = res
        lower = res["edge"] - 2.0 * res["se"]
        self._event(
            f"handoff gate vs {os.path.basename(ref)}: {res['edge']:+.4f}±{res['se']:.4f}",
            decision=True,
        )
        if lower < self.cfg.gates.handoff_h2h_lower_min:
            raise NeedsReview(
                f"handoff review gate: h2h lower bound {lower:+.4f} below "
                f"{self.cfg.gates.handoff_h2h_lower_min:+.3f}"
            )

    # ------------------------------------------------------------------ #
    # Phase 3: policy iteration
    # ------------------------------------------------------------------ #
    def _pi_stage_flags(self) -> list[str]:
        pi = self.cfg.policy_iteration
        flags = [
            "--trunk-epochs",
            str(pi.trunk_epochs),
            "--lr",
            str(pi.lr),
            "--lambda-ret",
            str(pi.lambda_ret),
            "--head-epochs",
            str(
                pi.head_epochs if pi.head_epochs is not None else pi.head_epochs_default
            ),
        ]
        for name, val in (
            ("--fit-epochs", pi.fit_epochs),
            ("--holdout-frac", pi.holdout_frac),
            ("--batch-rows", pi.batch_rows),
            ("--buffer-episodes", pi.buffer_episodes),
            ("--batch-segments", pi.batch_segments),
            ("--probe-games", pi.probe_games),
        ):
            if val is not None:
                flags += [name, str(val)]
        return flags

    def _cert_flags(self, routed: bool) -> list[str]:
        pi = self.cfg.policy_iteration
        flags = [
            "--cert-games",
            str(pi.cert_games),
            "--cert-seeds",
            str(pi.cert_seeds),
            "--h2h-deals",
            str(pi.cert_h2h_deals),
        ]
        if not (routed and pi.routed_reads):
            flags.append("--no-routed-reads")
        if self.cfg.smoke:
            flags.append("--no-bars")
        return flags

    def run_iteration(self, k: int, theta_k: str) -> tuple[str, dict]:
        """One iteration from theta_k; returns (theta_{k+1}, record)."""
        pi = self.cfg.policy_iteration
        it_dir = self.iter_dir(k)
        corpus_dir = os.path.join(it_dir, "corpus")
        os.makedirs(it_dir, exist_ok=True)
        rec = self.state["policy_iteration"]["iterations"].setdefault(str(k), {})
        rec["theta_k"] = theta_k
        self.section(f"policy iteration {k}: theta_{k - 1} = {theta_k}")
        if not os.path.exists(os.path.join(corpus_dir, "manifest.json")) or (
            self._corpus_incomplete(corpus_dir, pi.games)
        ):
            cmd = self._py(
                "sheepshead.training.distill_corpus",
                "--ckpt",
                theta_k,
                "--out-dir",
                corpus_dir,
                "--games",
                str(pi.games),
                "--workers",
                str(self.cfg.num_workers),
                "--seed",
                str(pi.corpus_seed_base + k),
                "--p-base",
                str(pi.p_base),
                "--boost-lead",
                str(pi.boost_lead),
                "--boost-cs",
                str(pi.boost_cs),
                "--p-min",
                str(pi.p_min),
                "--p-max",
                str(pi.p_max),
                "--committee-act-frac",
                str(pi.committee_act_frac),
                "--iters",
                str(pi.iters),
                "--replicates",
                str(pi.replicates),
                "--node-telemetry",
                os.path.join(corpus_dir, "nodes.jsonl"),
            )
            if pi.iters_schedule:
                cmd += ["--iters-schedule", pi.iters_schedule]
            if pi.routed_encoder:
                cmd += ["--routed-encoder", pi.routed_encoder]
            self._run(f"iter {k} corpus", cmd, f"pi_iter{k}.log")
        else:
            self.skip(f"iter {k} corpus", os.path.join(corpus_dir, "manifest.json"))
        if os.path.exists(os.path.join(it_dir, "distill_best.json")):
            self.skip(
                f"iter {k} fit/target/distill",
                os.path.join(it_dir, "distill_best.json"),
            )
        else:
            self._run(
                f"iter {k} fit/target/distill",
                self._py(
                    "sheepshead.training.policy_iteration",
                    "all",
                    "--corpus-dir",
                    corpus_dir,
                    "--ckpt",
                    theta_k,
                    "--out-dir",
                    it_dir,
                    *self._pi_stage_flags(),
                ),
                f"pi_iter{k}.log",
            )
        cert_path = os.path.join(it_dir, "cert.json")
        if os.path.exists(cert_path):
            self.skip(f"iter {k} cert", cert_path)
        else:
            self._run(
                f"iter {k} cert",
                self._py(
                    "sheepshead.training.policy_iteration",
                    "cert",
                    "--ckpt",
                    theta_k,
                    "--out-dir",
                    it_dir,
                    *self._cert_flags(routed=True),
                ),
                f"pi_iter{k}.log",
            )
        with open(cert_path) as f:
            cert = json.load(f)
        rec["cert"] = {
            k2: cert[k2]
            for k2 in (
                "candidate",
                "passed",
                "failures",
                "h2h",
                "probe_means",
                "compounding",
            )
        }
        rec["cert"]["routed"] = {
            name: {k3: v[k3] for k3 in ("edge", "se") if k3 in v}
            for name, v in cert.get("routed", {}).items()
        }
        if not cert["passed"]:
            self._save_state()
            raise NeedsReview(
                f"iteration {k} cert FAILED: {'; '.join(cert['failures'])} "
                "(WiSE-FT walk-back is an operator decision)"
            )
        candidate = cert["candidate"]
        theta_next = self.run_bidding_phase(k, candidate, it_dir, rec)
        rec["theta_next"] = theta_next
        self._save_state()
        return theta_next, rec

    def run_bidding_phase(self, k: int, candidate: str, it_dir: str, rec: dict) -> str:
        """Bidding-only PG phase (play heads pinned, oracle critic on) from
        ``candidate`` against the league population; adopted if not
        significantly worse than the candidate on the same battery.
        Returns the checkpoint to carry forward."""
        pi = self.cfg.policy_iteration
        bidding_dir = os.path.join(it_dir, "bidding")
        bidding_final = os.path.join(bidding_dir, "final.pt")
        if pi.bidding_episodes > 0 and os.path.exists(bidding_final):
            self.skip(f"iter {k} bidding phase", bidding_final)
        if pi.bidding_episodes > 0 and not os.path.exists(bidding_final):
            cmd = self._py(
                "sheepshead.training.train_ppo",
                "--phase",
                "bidding",
                "--resume",
                candidate,
                "--run-name",
                f"{self.cfg.run_name}/pi/iter{k}/bidding",
                "--league-dir",
                pi.league_dir or os.path.join(self.league_dir, "league"),
                "--until",
                str(pi.bidding_episodes),
                "--save-interval",
                str(max(pi.bidding_episodes, 1)),
                "--snapshot-interval",
                "0",
                "--greedy-eval-interval",
                "0",
                *self._worker_flags(),
            )
            if self.cfg.league.update_interval:
                cmd += ["--update-interval", str(self.cfg.league.update_interval)]
            self._run(f"iter {k} bidding phase", cmd, f"pi_iter{k}.log")
        theta_next = candidate
        if os.path.exists(bidding_final):
            bcert_dir = os.path.join(it_dir, "bidding_cert")
            os.makedirs(bcert_dir, exist_ok=True)
            bcert_path = os.path.join(bcert_dir, "cert.json")
            if os.path.exists(bcert_path):
                self.skip(f"iter {k} bidding cert", bcert_path)
            else:
                self._run(
                    f"iter {k} bidding cert",
                    self._py(
                        "sheepshead.training.policy_iteration",
                        "cert",
                        "--ckpt",
                        candidate,
                        "--out-dir",
                        bcert_dir,
                        "--candidate",
                        bidding_final,
                        *self._cert_flags(routed=False),
                    ),
                    f"pi_iter{k}.log",
                )
            with open(bcert_path) as f:
                bcert = json.load(f)
            rec["bidding_cert"] = {
                "h2h": bcert["h2h"],
                "probe_means": bcert["probe_means"],
            }
            non_inferior = bcert["h2h"]["edge"] + 2.0 * bcert["h2h"]["se"] >= 0.0
            rec["bidding_adopted"] = bool(non_inferior)
            if non_inferior:
                theta_next = bidding_final
            self._event(
                f"iter {k} bidding phase: h2h vs candidate "
                f"{bcert['h2h']['edge']:+.4f}±{bcert['h2h']['se']:.4f} -> "
                f"{'ADOPTED' if non_inferior else 'not adopted'}",
                decision=True,
            )
        rec["theta_next"] = theta_next
        self._save_state()
        return theta_next

    @staticmethod
    def _corpus_incomplete(corpus_dir: str, games: int) -> bool:
        with open(os.path.join(corpus_dir, "manifest.json")) as f:
            return int(json.load(f).get("kept_games", 0)) < games

    def run_policy_iteration(self, theta_0: str) -> str:
        pi = self.cfg.policy_iteration
        rule = IterationRuleConfig(
            se_multiple=pi.stop_se_multiple,
            flat_iterations=pi.stop_flat_iterations,
            max_iterations=pi.max_iterations,
        )
        state = self.state["policy_iteration"]
        if state.get("theta"):
            return state["theta"]
        theta = theta_0
        if self.cfg.start_phase == "policy_iteration" and pi.bidding_first:
            # Validation entry: the external theta_0 is a certified distill
            # candidate; run its bidding phase first (iteration 0), then the
            # next corpus comes from the adopted checkpoint.
            it0 = self.iter_dir(0)
            os.makedirs(it0, exist_ok=True)
            self.section(f"policy iteration 0: bidding phase on theta_0 = {theta_0}")
            rec0 = state["iterations"].setdefault("0", {"theta_k": theta_0})
            theta = self.run_bidding_phase(0, theta_0, it0, rec0)
            self._event(
                f"iteration 0 (bidding phase on theta_0): theta = {theta}",
                decision=True,
            )
        gains: list[tuple[float, float]] = []
        k = 1
        while True:
            theta, rec = self.run_iteration(k, theta)
            comp = rec["cert"].get("compounding") or rec["cert"]["h2h"]
            gains.append((comp["edge"], comp["se"]))
            self._event(
                f"iter {k}: compounding gain ({comp.get('source', 'h2h')}) "
                f"{gains[-1][0]:+.4f}±{gains[-1][1]:.4f}; full h2h "
                f"{rec['cert']['h2h']['edge']:+.4f}±{rec['cert']['h2h']['se']:.4f}; "
                f"theta_{k} = {theta}",
                decision=True,
            )
            stop, reason = iteration_stop(gains, rule)
            if stop:
                self._event(f"policy iteration STOP: {reason}", decision=True)
                state["theta"] = theta
                self._save_state()
                return theta
            k += 1

    # ------------------------------------------------------------------ #
    # Phase 4: final certification
    # ------------------------------------------------------------------ #
    def run_final(self, theta: str) -> None:
        cfg = self.cfg
        final_dir = os.path.join(self.run_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        release = os.path.join(final_dir, "release.pt")
        if not os.path.exists(release):
            shutil.copyfile(theta, release)
        rec = self.state["final"]
        rec["release"] = release
        for name, ref in cfg.final.references.items():
            if not os.path.exists(ref):
                self.log(f"final h2h vs {name} skipped ({ref} missing)")
                continue
            path = os.path.join(final_dir, f"h2h_vs_{name}.json")
            if not os.path.exists(path):
                res = h2h_duplicate(release, ref, n_deals_per_mode=cfg.final.h2h_deals)
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
            with open(path) as f:
                res = json.load(f)
            rec[f"h2h_vs_{name}"] = {"edge": res["edge"], "se": res["se"]}
            self._event(
                f"final h2h vs {name}: {res['edge']:+.4f}±{res['se']:.4f}",
                decision=True,
            )
        rec["conventions"] = self._conventions(release, "final")
        for key, members, label in (
            ("panel", cfg.league.panel, "panel final"),
            ("panel_b", cfg.league.panel_b, "panel-B final"),
        ):
            if members and key not in rec:
                rec[key] = self._panel_read(
                    [release, release, release],
                    members,
                    label,
                    f"{label.replace(' ', '_')}.npz",
                )
                if rec[key]:
                    self._event(
                        f"final {label.split()[0]}: {rec[key]['mean']:+.4f} "
                        f"[{rec[key]['lo']:+.4f}, {rec[key]['hi']:+.4f}]",
                        decision=True,
                    )
        gate_path = os.path.join(
            "runs", f"{cfg.run_name}/final/exploit", "gate_result.json"
        )
        if cfg.final.exploit_episodes > 0 and not os.path.exists(gate_path):
            self._run(
                "exploitability audit",
                self._py(
                    "sheepshead.analysis.exploitability_audit",
                    "--main-ckpt",
                    release,
                    "--run-name",
                    f"{cfg.run_name}/final/exploit",
                    "--episodes",
                    str(cfg.final.exploit_episodes),
                    "--gate-deals",
                    str(cfg.final.exploit_gate_deals),
                    "--num-workers",
                    str(cfg.num_workers),
                    "--seed",
                    str(cfg.seed),
                ),
                "final.log",
            )
        if os.path.exists(gate_path):
            with open(gate_path) as f:
                gate = json.load(f)
            rec["exploit_audit"] = {
                "edge": gate["edge"],
                "se": gate["se"],
                "passed": gate["passed"],
            }
            self._event(
                f"exploitability audit: {gate['edge']:+.4f}±{gate['se']:.4f} "
                f"({'EXPLOITABLE' if gate['passed'] else 'gate not cleared'})",
                decision=True,
            )
        self._save_state()
        self._write_report()

    def _write_report(self) -> None:
        s = self.state
        lines = [
            f"# Training program report — {self.cfg.run_name}",
            "",
            f"*Regenerated {_now()} — status: **{s['status']}***",
            "",
            "## League generations",
            "",
            "| gen | h2h vs prev | confirm | improving | panel A | panel B | partner | t0 trump | called-suit | decision |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g in sorted(int(k) for k in s["league"]["generations"]):
            rec = s["league"]["generations"][str(g)]
            if "decision" not in rec:
                continue
            conv = rec.get("conventions") or {}
            panel = rec.get("panel")
            panel_b = rec.get("panel_b")
            confirm = rec.get("h2h_confirm")
            lines.append(
                f"| {g} | {rec['h2h']['edge']:+.4f}±{rec['h2h']['se']:.4f} "
                f"| {(f'{confirm["edge"]:+.4f}±{confirm["se"]:.4f}') if confirm else '—'} "
                f"| {rec['improving']} "
                f"| {(f'{panel["mean"]:+.4f}') if panel else '—'} "
                f"| {(f'{panel_b["mean"]:+.4f}') if panel_b else '—'} "
                f"| {conv.get('partner_trump_lead_rate', 0):.1f} "
                f"| {conv.get('t0_trump_lead_rate', 0):.2f} "
                f"| {conv.get('called_suit_lead_rate', 0):.1f} "
                f"| {rec['decision']['action']} |"
            )
        if s["league"].get("handoff"):
            lines += [
                "",
                f"Handoff: generation {s['league']['handoff']['generation']} "
                f"(`{s['league']['handoff']['checkpoint']}`)",
            ]
        lines += [
            "",
            "## Policy iteration",
            "",
            "| iter | cert h2h vs theta_k | called-suit | bidding phase | theta_next |",
            "|---|---|---|---|---|",
        ]
        for k in sorted(int(x) for x in s["policy_iteration"]["iterations"]):
            rec = s["policy_iteration"]["iterations"][str(k)]
            cert = rec.get("cert")
            if not cert:
                continue
            b = rec.get("bidding_cert")
            lines.append(
                f"| {k} | {cert['h2h']['edge']:+.4f}±{cert['h2h']['se']:.4f} "
                f"| {cert['probe_means']['called_suit_lead_rate']:.1f} "
                f"| {(f'{b["h2h"]["edge"]:+.4f}±{b["h2h"]["se"]:.4f} ' + ('adopted' if rec.get('bidding_adopted') else 'not adopted')) if b else '—'} "
                f"| `{rec.get('theta_next', '')}` |"
            )
        lines += ["", "## Phase times", "", "```"]
        lines += self.phase_time_lines()
        lines += ["```", "", "## Final", ""]
        for key, val in s["final"].items():
            if key.startswith("h2h_vs_"):
                lines.append(f"- {key}: {val['edge']:+.4f}±{val['se']:.4f}")
        for key, label in (("panel", "PANEL-A"), ("panel_b", "PANEL-B")):
            val = s["final"].get(key)
            if val:
                lines.append(
                    f"- {label}: {val['mean']:+.4f} [{val['lo']:+.4f}, {val['hi']:+.4f}]"
                )
        conv = s["final"].get("conventions")
        if conv:
            lines.append(
                "- conventions (4 x 1000 greedy games): "
                f"called-suit {conv.get('called_suit_lead_rate', 0):.1f}, "
                f"partner trump {conv.get('partner_trump_lead_rate', 0):.1f}, "
                f"t0 trump {conv.get('t0_trump_lead_rate', 0):.2f}, "
                f"pick {conv.get('pick_rate', 0):.1f}, "
                f"leaster {conv.get('leaster_rate', 0):.1f}"
            )
        if s["final"].get("release"):
            lines.append(f"- release: `{s['final']['release']}`")
        if "exploit_audit" in s["final"]:
            a = s["final"]["exploit_audit"]
            lines.append(
                f"- exploitability audit: {a['edge']:+.4f}±{a['se']:.4f} "
                f"({'exploitable' if a['passed'] else 'gate not cleared'})"
            )
        if s["events"]:
            lines += ["", "## Event log", ""]
            lines += [f"- {e['time']}: {e['msg']}" for e in s["events"][-60:]]
        with open(os.path.join(self.program_dir, "report.md"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        cfg = self.cfg
        if self.state["status"] == "finished":
            self.log("program already finished; see program/report.md")
            return 0
        resumed = bool(self.state["events"])
        self.banner(
            f"TRAINING PROGRAM {cfg.run_name}" + (" (resumed)" if resumed else ""),
            f"config: {os.path.join(self.program_dir, 'config.json')}",
            f"state:  {self._state_path()} (status {self.state['status']}, "
            f"phase {self.state['phase']})",
            f"arch {cfg.arch}, seed {cfg.seed}, workers {cfg.num_workers}, "
            f"start_phase {cfg.start_phase}" + (", SMOKE" if cfg.smoke else ""),
        )
        if self.state["status"] == "needs_review":
            self._event("resuming a needs_review run (operator override implied)")
            self.state["status"] = "running"
        try:
            if cfg.start_phase == "policy_iteration":
                # Validation entry (Training_Program_Redesign §7.0 / §4.4):
                # phase 3 on an external lineage from policy_iteration.theta_0.
                theta_0 = cfg.policy_iteration.theta_0
                if not theta_0 or not os.path.exists(theta_0):
                    raise NeedsReview(
                        "start_phase=policy_iteration needs policy_iteration.theta_0"
                    )
                self._event(f"starting at policy iteration from {theta_0}")
            else:
                self.begin_phase(
                    "bootstrap",
                    f"{cfg.bootstrap.episodes:,} episodes -> {self.bootstrap_final}",
                )
                self.ensure_bootstrap()
                self.end_phase("bootstrap")
                self.begin_phase(
                    "oracle", f"{cfg.oracle.episodes:,} episodes -> {self.oracle_init}"
                )
                self.ensure_oracle()
                self.end_phase("oracle")
                self.begin_phase(
                    "league",
                    f"{cfg.league.generation_episodes:,} episodes/generation, "
                    f"{cfg.league.min_generations}-{cfg.league.max_generations} generations; "
                    f"handoff when h2h gain < {cfg.league.h2h_min_gain:+.3f} "
                    f"(CI z {cfg.league.h2h_ci_z}) twice",
                )
                theta_0 = self.run_league()
                self.end_phase("league")
            pi = cfg.policy_iteration
            self.begin_phase(
                "policy_iteration",
                f"theta_0 = {theta_0}",
                f"{pi.games:,} committee-acted games/iteration at {pi.iters} iters"
                + (f" ({pi.iters_schedule})" if pi.iters_schedule else "")
                + f", {pi.trunk_epochs} trunk epochs @ {pi.lr:g}, lambda_ret {pi.lambda_ret:g}",
                f"stop: play-only gain < {pi.stop_se_multiple:g} SE for "
                f"{pi.stop_flat_iterations} iterations, cap {pi.max_iterations}",
            )
            theta = self.run_policy_iteration(theta_0)
            self.end_phase("policy_iteration")
            self.begin_phase("final", f"release candidate = {theta}")
            self.run_final(theta)
            self.end_phase("final")
            self.state["status"] = "finished"
            self._event("PROGRAM FINISHED", decision=True)
            self._write_report()
            self.banner(
                "PROGRAM FINISHED",
                f"report: {os.path.join(self.program_dir, 'report.md')}",
                *self.phase_time_lines(),
            )
            return 0
        except NeedsReview as exc:
            self.state["status"] = "needs_review"
            self._event(f"NEEDS REVIEW: {exc}", log_msg=f"✖ NEEDS REVIEW: {exc}")
            self._save_state()
            self._write_report()
            self.banner(
                "PROGRAM STOPPED — NEEDS REVIEW",
                str(exc),
                "fix the cause and re-run the same command to resume",
                *self.phase_time_lines(),
            )
            return 2


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-name", default=None)
    p.add_argument("--config", default=None, help="ProgramConfig JSON")
    p.add_argument("--smoke", action="store_true", help="minutes-long end-to-end check")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_config(args) -> ProgramConfig:
    if args.config:
        cfg = ProgramConfig.load(args.config)
        if args.run_name:
            cfg.run_name = args.run_name
        return cfg
    if args.smoke:
        return ProgramConfig.smoke_config(args.run_name or "_smoke_program")
    if not args.run_name:
        raise SystemExit("--run-name is required (or --config / --smoke)")
    return ProgramConfig(run_name=args.run_name)


def main(argv=None) -> int:
    os.chdir(_REPO_ROOT)
    args = parse_args(argv)
    cfg = build_config(args)
    program = Program(cfg)
    if args.dry_run:
        program.log("DRY RUN — configuration:")
        program.log(cfg.to_json())
        if cfg.start_phase == "bootstrap":
            program.log(
                "gen-1 league command: "
                + " ".join(program.league_trainer_cmd(1, "<bootstrap final>"))
            )
        else:
            program.log(
                f"start_phase={cfg.start_phase}: theta_0 = "
                f"{cfg.policy_iteration.theta_0}, league_dir = "
                f"{cfg.policy_iteration.league_dir}, bidding_first = "
                f"{cfg.policy_iteration.bidding_first}"
            )
        return 0
    return program.run()


if __name__ == "__main__":
    raise SystemExit(main())
