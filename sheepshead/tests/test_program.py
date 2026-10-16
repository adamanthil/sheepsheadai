"""The program orchestrator's wiring (run_training_program.py): phase
commands carry the pre-registered flags, the config round-trips as the
pre-registration artifact, and the smoke preset covers every phase."""

import csv
import json
import os
import signal
import time

import pytest

import sheepshead.training.run_training_program as rtp
from sheepshead.training.program_config import ProgramConfig
from sheepshead.training.run_training_program import Program


def _program(tmp_path, monkeypatch, cfg=None):
    monkeypatch.chdir(tmp_path)
    return Program(cfg or ProgramConfig(run_name="t"))


def test_config_round_trips(tmp_path):
    cfg = ProgramConfig(run_name="x")
    cfg.league.max_generations = 5
    cfg.policy_iteration.games = 123
    path = tmp_path / "cfg.json"
    path.write_text(cfg.to_json())
    back = ProgramConfig.load(str(path))
    assert back.league.max_generations == 5
    assert back.policy_iteration.games == 123
    assert back.gates.gen2_panel_min == cfg.gates.gen2_panel_min
    assert json.loads(back.to_json()) == json.loads(cfg.to_json())


def test_smoke_preset_is_small_and_complete():
    cfg = ProgramConfig.smoke_config("s")
    assert cfg.smoke and cfg.bootstrap.episodes < 100
    assert cfg.league.max_generations <= 3 and cfg.policy_iteration.games <= 12
    assert cfg.policy_iteration.bidding_episodes > 0  # the bidding phase runs
    assert cfg.final.exploit_episodes > 0  # and so does the audit


def test_bidding_command_runs_the_league_greedy_probe(tmp_path, monkeypatch):
    """The bidding phase (fixed coefficients, no controller) carries the
    league's greedy-probe cadence so the sharpening of the bidding heads is
    observed during the phase, not only at the certificate; it still takes
    no population snapshots and checkpoints once, at the end."""
    p = _program(tmp_path, monkeypatch)
    cmd = p.bidding_trainer_cmd(2, "cand.pt")
    assert cmd[cmd.index("--phase") + 1] == "bidding"
    assert cmd[cmd.index("--resume") + 1] == "cand.pt"
    assert cmd[cmd.index("--greedy-eval-interval") + 1] == str(
        p.cfg.league.greedy_eval_interval
    )
    assert cmd[cmd.index("--greedy-eval-games") + 1] == str(
        p.cfg.league.greedy_eval_games
    )
    assert cmd[cmd.index("--snapshot-interval") + 1] == "0"
    assert cmd[cmd.index("--until") + 1] == str(p.cfg.policy_iteration.bidding_episodes)
    assert cmd[cmd.index("--save-interval") + 1] == str(
        p.cfg.policy_iteration.bidding_episodes
    )
    assert "/pi/iter2/bidding" in cmd[cmd.index("--run-name") + 1]


def test_league_commands_distinguish_generation_one(tmp_path, monkeypatch):
    p = _program(tmp_path, monkeypatch)
    os.makedirs(p.bootstrap_dir)
    open(p.bootstrap_final, "wb").close()
    gen1 = p.league_trainer_cmd(1, p.bootstrap_final)
    gen2 = p.league_trainer_cmd(2, p.boundary_ckpt(1))
    assert "--no-entropy-controller" in gen1 and "--no-entropy-controller" not in gen2
    assert "--oracle-init" in gen1 and "--oracle-init" not in gen2
    assert "--seed-checkpoints" in gen1 and "--seed-checkpoints" not in gen2
    assert gen1[gen1.index("--until") + 1] == str(p.cfg.league.generation_episodes)
    assert gen2[gen2.index("--until") + 1] == str(2 * p.cfg.league.generation_episodes)
    assert "--phase" in gen1 and gen1[gen1.index("--phase") + 1] == "league"
    # League generations run their workers on MPS + compile (LeagueConfig);
    # the program-level default (CPU) stays for the other phases.
    assert gen1[gen1.index("--worker-device") + 1] == p.cfg.league.worker_device
    assert gen1[gen1.index("--worker-compile") + 1] == p.cfg.league.worker_compile
    assert "--worker-device" not in p._worker_flags()
    # Rendering the command materializes nothing (--dry-run renders it
    # before any bootstrap exists); the seeds appear when generation 1
    # is about to train.
    assert not os.path.exists(p.seeds_dir)
    assert p.ensure_seeds() == gen1[gen1.index("--seed-checkpoints") + 1]
    assert len(os.listdir(p.seeds_dir)) == p.cfg.league.seed_copies


def test_dry_run_of_a_fresh_program_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rtp, "_REPO_ROOT", str(tmp_path))
    assert rtp.main(["--dry-run", "--run-name", "fresh"]) == 0
    assert not os.path.exists(os.path.join("runs", "fresh", "seeds"))
    assert not os.path.exists(os.path.join("runs", "fresh", "bootstrap"))


def test_state_is_persisted_with_the_config(tmp_path, monkeypatch):
    p = _program(tmp_path, monkeypatch)
    p._event("hello")
    assert os.path.exists(os.path.join(p.program_dir, "state.json"))
    assert os.path.exists(os.path.join(p.program_dir, "config.json"))
    again = Program(ProgramConfig(run_name="t"))
    assert again.state["events"][-1]["msg"] == "hello"


class _Stub:
    """Drive Program.run_league without training: boundary checkpoints are
    touched, h2h/panel/conventions are scripted, the controller sidecar is
    a real one so the entropy step edits real state."""

    def __init__(self, program, edges, aux_ready_from=1):
        self.p = program
        self.edges = edges  # {gen: (primary_edge, confirm_edge)}; se fixed
        self.judged = []
        # The deterministic aux heads read as converged from this generation
        # on (the battery's aux keys, §5.4); before it, they miss every bar.
        self.aux_ready_from = aux_ready_from
        os.makedirs(program.league_ckpt_dir, exist_ok=True)
        from sheepshead.training.entropy_controller import EntropyTargetController

        ctrl = EntropyTargetController(targets={"play": 0.7})
        ctrl.alphas = {"pick": 0.05, "partner": 0.05, "bury": 0.04, "play": 0.015}
        ctrl.save(os.path.join(program.league_ckpt_dir, "entropy_controller.json"))
        program.ensure_generation_trained = self.train
        program._h2h = self.h2h
        program._panel = lambda g: None
        program._panel_b = lambda g: None
        program._conventions = self.conventions

    def conventions(self, ckpt, label):
        g = int(label.replace("gen", ""))
        ready = g >= self.aux_ready_from
        return {
            "pick_rate": 30.0,
            "alone_rate": 5.0,
            "leaster_rate": 8.0,
            "t0_trump_lead_rate": 0.5,
            "partner_trump_lead_rate": 97.0,
            "called_suit_lead_rate": 45.0,
            "play_logit_spread_med": 4.0,
            "seen_trump_acc": 99.8 if ready else 90.0,
            "seen_trump_recall": 99.8 if ready else 95.0,
            "seen_trump_false_seen": 0.2 if ready else 16.0,
            "seen_trump_recall_by_trick": [60.0] + [99.5 if ready else 80.0] * 5,
            "seen_trump_false_seen_by_trick": [0.0] + [0.2 if ready else 30.0] * 5,
            "aux_secret_acc": 99.9 if ready else 97.0,
            "aux_points_exact": 99.5 if ready else 90.0,
            "aux_points_mae": 0.05 if ready else 1.5,
            "aux_unseen_higher_acc": 99.5 if ready else 95.0,
        }

    def train(self, g):
        open(self.p.boundary_ckpt(g), "wb").close()

    def h2h(self, g, seed, tag):
        self.judged.append((g, tag))
        primary, confirm = self.edges[g]
        return {"edge": confirm if tag else primary, "se": 0.012, "modes": {}}


def test_league_loop_steps_once_then_hands_off_from_a_settled_boundary(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg = ProgramConfig(run_name="loop")
    p = Program(cfg)
    # gens 1-2 pay, gen 3 misses (confirmation also misses) -> step;
    # gen 4 misses -> handoff from gen 3's boundary (the pre-step one).
    stub = _Stub(p, {1: (0.08, 0.0), 2: (0.10, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0)})
    theta_0 = p.run_league()
    assert theta_0 == p.boundary_ckpt(3)
    assert p.state["league"]["step_generation"] == 3
    actions = [
        p.state["league"]["generations"][str(g)]["decision"]["action"]
        for g in range(1, 5)
    ]
    assert actions == ["continue", "continue", "entropy_step", "handoff"]
    # The step edited the real sidecar (retain 0.75 toward the 0.28 floor).
    from sheepshead.training.entropy_controller import EntropyTargetController

    ctrl = EntropyTargetController.load(
        os.path.join(p.league_ckpt_dir, "entropy_controller.json")
    )
    assert ctrl.targets["play"] == pytest.approx(0.28 + 0.75 * (0.7 - 0.28))
    # Confirmation reads happened only on the missing generations.
    assert [j for j in stub.judged if j[1] == "_confirm"] == [
        (3, "_confirm"),
        (4, "_confirm"),
    ]
    # Resume is idempotent: a fresh Program returns the recorded handoff
    # without judging anything again.
    p2 = Program(ProgramConfig.from_dict(json.loads(cfg.to_json())))
    p2.gates = cfg.gates
    calls = []
    p2._h2h = lambda *a: calls.append(a) or {"edge": 0.0, "se": 0.0, "modes": {}}
    assert p2.run_league() == theta_0 and calls == []
    assert os.path.exists(os.path.join(p.program_dir, "generations.csv"))


def test_handoff_waits_for_the_aux_heads(tmp_path, monkeypatch):
    """The marginal-value rule would hand off after gen 4; the aux heads
    read ready only from gen 5, so gen 4 defers (continue) and gen 5 hands
    off — from its own boundary (the step was at gen 3, so gen 5 is
    settled)."""
    monkeypatch.chdir(tmp_path)
    cfg = ProgramConfig(run_name="auxwait")
    p = Program(cfg)
    _Stub(
        p,
        {1: (0.08, 0.0), 2: (0.10, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0), 5: (0.0, 0.0)},
        aux_ready_from=5,
    )
    theta_0 = p.run_league()
    gens = p.state["league"]["generations"]
    assert gens["4"]["aux_ready"] is False and gens["4"]["aux_failures"]
    assert gens["4"]["decision"]["action"] == "continue"
    assert "deferred" in gens["4"]["decision"]["reason"]
    assert gens["5"]["aux_ready"] is True
    assert gens["5"]["decision"]["action"] == "handoff"
    assert theta_0 == p.boundary_ckpt(5)
    with open(os.path.join(p.program_dir, "generations.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [r["aux_ready"] for r in rows] == ["False"] * 4 + ["True"]


def test_aux_heads_not_ready_at_the_cap_is_a_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ProgramConfig(run_name="auxcap")
    cfg.league.max_generations = 4
    p = Program(cfg)
    _Stub(p, {g: (0.08, 0.0) for g in range(1, 5)}, aux_ready_from=99)
    with pytest.raises(rtp.NeedsReview, match="aux heads not ready"):
        p.run_league()
    assert p.state["league"]["generations"]["4"]["decision"]["action"] == "review"


def test_confirmation_rescues_a_noise_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ProgramConfig(run_name="rescue")
    cfg.league.max_generations = 3
    p = Program(cfg)
    _Stub(p, {1: (0.08, 0.0), 2: (0.015, 0.05), 3: (0.0, 0.0)})
    p.run_league()
    gens = p.state["league"]["generations"]
    assert gens["2"]["improving"] is True and gens["2"]["h2h_confirm"]["edge"] == 0.05
    assert p.state["league"]["step_generation"] is None  # gen 3 = cap handoff


def test_iteration_panels_read_the_carried_iterate_as_one_checkpoint(
    tmp_path, monkeypatch
):
    """Phase 3 continues the panel series: after each iteration the iterate
    carried forward is scored on PANEL-A/B in the single-checkpoint form
    (the path in all three slots — distillation leaves no earlier
    snapshots to composite), cached as panel_iter<k> / panelB_iter<k>."""
    p = _program(tmp_path, monkeypatch)
    p.cfg.league.panel = ["a.pt", "b.pt", "c.pt", "d.pt"]
    p.cfg.league.panel_b = ["e.pt", "f.pt", "g.pt", "h.pt"]
    calls = []

    def fake_read(ckpts, members, label, npz_name):
        calls.append((tuple(ckpts), tuple(members), label, npz_name))
        return {"mean": 0.1, "lo": 0.0, "hi": 0.2, "se": 0.05, "modes": {}}

    p._panel_read = fake_read
    rec = {}
    p._iteration_panels(2, "theta2.pt", rec)
    assert rec["panel"]["mean"] == 0.1 and rec["panel_b"]["mean"] == 0.1
    assert calls == [
        (
            ("theta2.pt",) * 3,
            tuple(p.cfg.league.panel),
            "panel iter 2",
            "panel_iter2.npz",
        ),
        (
            ("theta2.pt",) * 3,
            tuple(p.cfg.league.panel_b),
            "panel-B iter 2",
            "panelB_iter2.npz",
        ),
    ]
    # Re-entry (resume) does not read again: the record is the cache key.
    p._iteration_panels(2, "theta2.pt", rec)
    assert len(calls) == 2
    # Without a PANEL-B list only PANEL-A is read.
    p.cfg.league.panel_b = []
    rec2 = {}
    p._iteration_panels(3, "theta3.pt", rec2)
    assert "panel_b" not in rec2 and calls[-1][3] == "panel_iter3.npz"


def test_stop_after_pauses_at_the_named_boundary(tmp_path, monkeypatch):
    """``--stop-after iter<k>.<stage>`` raises at that boundary only; the run
    loop records the pause (status "paused", exit 0) and a relaunch
    without the knob resumes through the ordinary skips."""
    cfg = ProgramConfig(run_name="t", stop_after="iter1.cert")
    p = _program(tmp_path, monkeypatch, cfg)
    p._check_stop(1, "corpus")
    p._check_stop(2, "cert")
    with pytest.raises(rtp.StopAfter, match="iter1.cert"):
        p._check_stop(1, "cert")
    # The run loop: phase 3 entry raises at the boundary -> paused, exit 0.
    theta = tmp_path / "theta0.pt"
    theta.write_bytes(b"x")
    cfg = ProgramConfig(run_name="t", stop_after="iter1.cert")
    cfg.start_phase = "policy_iteration"
    cfg.policy_iteration.theta_0 = str(theta)
    p = Program(cfg)

    def fake_iteration(theta_0):
        p._check_stop(1, "cert")
        return theta_0

    p.run_policy_iteration = fake_iteration
    assert p.run() == 0
    assert p.state["status"] == "paused"
    assert any("PAUSED after iter1.cert" in e["msg"] for e in p.state["events"])
    # Relaunch without the knob: status goes back to running.
    cfg2 = ProgramConfig(run_name="t")
    cfg2.start_phase = "policy_iteration"
    cfg2.policy_iteration.theta_0 = str(theta)
    p2 = Program(cfg2)
    assert p2.state["status"] == "paused"
    seen = []
    p2.run_policy_iteration = lambda t: seen.append(t) or t
    p2.run_final = lambda t: None
    assert p2.run() == 0
    assert seen == [str(theta)] and p2.state["status"] == "finished"


def test_stop_after_cli_validation():
    args = rtp.parse_args(["--run-name", "x", "--stop-after", "iter3.distill"])
    assert rtp.build_config(args).stop_after == "iter3.distill"
    assert rtp.build_config(rtp.parse_args(["--run-name", "x"])).stop_after is None
    with pytest.raises(SystemExit):
        rtp.build_config(rtp.parse_args(["--run-name", "x", "--stop-after", "cert"]))
    with pytest.raises(SystemExit):
        rtp.build_config(
            rtp.parse_args(["--run-name", "x", "--stop-after", "iter1.fit"])
        )


def test_recipe_ablation_knobs_pass_through_to_the_stages(tmp_path, monkeypatch):
    p = _program(tmp_path, monkeypatch)
    flags = p._pi_stage_flags()
    assert flags[flags.index("--cross-fit") + 1] == "5"  # the pinned recipe
    assert "--distill-holdout-frac" not in flags
    p.cfg.policy_iteration.cross_fit = None
    assert "--cross-fit" not in p._pi_stage_flags()
    p.cfg.policy_iteration.cross_fit = 5
    p.cfg.policy_iteration.distill_holdout_frac = 0.02
    flags = p._pi_stage_flags()
    assert flags[flags.index("--cross-fit") + 1] == "5"
    assert flags[flags.index("--distill-holdout-frac") + 1] == "0.02"
    assert "--model-variance" not in flags  # unset = the module default
    p.cfg.policy_iteration.model_variance = "one-sided-head"
    flags = p._pi_stage_flags()
    assert flags[flags.index("--model-variance") + 1] == "one-sided-head"


def test_a_second_orchestrator_is_refused_while_the_first_runs(tmp_path, monkeypatch):
    p = _program(tmp_path, monkeypatch)
    p._acquire_lock()
    with pytest.raises(SystemExit, match="still running"):
        Program(ProgramConfig(run_name="t")).run()
    p._release_lock()
    p2 = Program(ProgramConfig(run_name="t"))
    p2._acquire_lock()  # free again once the holder is gone
    p2._release_lock()


def test_an_orphaned_stage_keeps_the_lock(tmp_path, monkeypatch):
    """A stage that outlives its orchestrator (here: a background process
    the stage left behind, standing in for a child of a killed
    orchestrator) keeps the run locked until it exits."""
    p = _program(tmp_path, monkeypatch)
    p._acquire_lock()
    pid_file = tmp_path / "orphan.pid"
    p._run("stage", ["sh", "-c", f"sleep 30 & echo $! > {pid_file}"], "stage.log")
    p._release_lock()
    orphan = int(pid_file.read_text())
    try:
        with pytest.raises(SystemExit, match="still running"):
            Program(ProgramConfig(run_name="t"))._acquire_lock()
    finally:
        os.kill(orphan, signal.SIGTERM)
    deadline = time.time() + 10
    while True:
        p2 = Program(ProgramConfig(run_name="t"))
        try:
            p2._acquire_lock()
            break
        except SystemExit:
            if time.time() > deadline:
                raise
            time.sleep(0.05)
    p2._release_lock()


def test_a_partial_corpus_is_resumed_not_restarted(tmp_path, monkeypatch):
    p = _program(tmp_path, monkeypatch)
    corpus_dir = os.path.join(p.iter_dir(1), "corpus")
    os.makedirs(corpus_dir)
    with open(os.path.join(corpus_dir, "manifest.json"), "w") as f:
        json.dump({"kept_games": p.cfg.policy_iteration.games - 1}, f)
    cmds = []

    def fake_run(label, cmd, log_name):
        cmds.append(cmd)
        raise rtp.StopAfter(label)

    p._run = fake_run
    with pytest.raises(rtp.StopAfter, match="iter 1 corpus"):
        p.run_iteration(1, "theta.pt")
    assert "sheepshead.training.distill_corpus" in cmds[0]
    assert "--resume" in cmds[0]
