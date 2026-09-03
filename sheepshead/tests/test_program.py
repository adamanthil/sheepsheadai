"""The program orchestrator's wiring (run_training_program.py): phase
commands carry the pre-registered flags, the config round-trips as the
pre-registration artifact, and the smoke preset covers every phase."""

import json
import os

import pytest

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
    # Seeds were materialized as copies of the bootstrap final.
    assert len(os.listdir(p.seeds_dir)) == p.cfg.league.seed_copies


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

    def __init__(self, program, edges):
        self.p = program
        self.edges = edges  # {gen: (primary_edge, confirm_edge)}; se fixed
        self.judged = []
        os.makedirs(program.league_ckpt_dir, exist_ok=True)
        from sheepshead.training.entropy_controller import EntropyTargetController

        ctrl = EntropyTargetController(targets={"play": 0.7})
        ctrl.alphas = {"pick": 0.05, "partner": 0.05, "bury": 0.04, "play": 0.015}
        ctrl.save(os.path.join(program.league_ckpt_dir, "entropy_controller.json"))
        program.ensure_generation_trained = self.train
        program._h2h = self.h2h
        program._panel = lambda g: None
        program._conventions = lambda ckpt, label: {
            "pick_rate": 30.0,
            "alone_rate": 5.0,
            "leaster_rate": 8.0,
            "t0_trump_lead_rate": 0.5,
            "partner_trump_lead_rate": 97.0,
            "called_suit_lead_rate": 45.0,
            "play_logit_spread_med": 4.0,
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
    cfg.gates.handoff_reference = ""
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


def test_confirmation_rescues_a_noise_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ProgramConfig(run_name="rescue")
    cfg.league.max_generations = 3
    cfg.gates.handoff_reference = ""
    p = Program(cfg)
    _Stub(p, {1: (0.08, 0.0), 2: (0.015, 0.05), 3: (0.0, 0.0)})
    p.run_league()
    gens = p.state["league"]["generations"]
    assert gens["2"]["improving"] is True and gens["2"]["h2h_confirm"]["edge"] == 0.05
    assert p.state["league"]["step_generation"] is None  # gen 3 = cap handoff
