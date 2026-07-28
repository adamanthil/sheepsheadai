"""Target-entropy controller invariants (adaptive entropy, Phase 2).

Inner loop: bumpless target adoption, integral-feedback sign, per-update
log clamp, coefficient bounds. Outer loop: hold heads never step, anneal
heads step geometrically toward their floor and stop at min_step (at-floor
detection). Persistence: sidecar roundtrip preserves everything. Wiring:
both CLI surfaces default OFF (historical behavior).

Refs: Haarnoja et al. arXiv:1812.05905 (target-entropy temperature);
Christodoulou arXiv:1910.07207 (discrete fraction-of-max targets);
Jaderberg et al. arXiv:1711.09846 (perturbation-scale outer steps)."""

import math
from types import SimpleNamespace

from sheepshead.training.entropy_controller import (
    ALPHA_MAX,
    ALPHA_MIN,
    HEADS,
    EntropyControllerConfig,
    EntropyTargetController,
)


def _ctrl(**kwargs):
    return EntropyTargetController(config=EntropyControllerConfig(**kwargs))


def _agent(**coeffs):
    defaults = {
        "entropy_coeff_pick": 0.046,
        "entropy_coeff_partner": 0.046,
        "entropy_coeff_bury": 0.037,
        "entropy_coeff_play": 0.0138,
    }
    defaults.update(coeffs)
    return SimpleNamespace(**defaults)


MEASURED = {"pick": 0.05, "partner": 0.12, "bury": 0.16, "play": 0.75}


class TestInnerLoop:
    def test_attach_is_bumpless_in_alpha(self):
        agent = _agent()
        ctrl = _ctrl()
        ctrl.attach(agent)
        for h in HEADS:
            assert ctrl.alphas[h] == getattr(agent, f"entropy_coeff_{h}")
        # apply() writes the same values back: switch-on changes nothing.
        before = {h: getattr(agent, f"entropy_coeff_{h}") for h in HEADS}
        ctrl.apply(agent)
        assert before == {h: getattr(agent, f"entropy_coeff_{h}") for h in HEADS}

    def test_bumpless_target_adoption(self):
        ctrl = _ctrl()
        ctrl.attach(_agent())
        alphas_before = dict(ctrl.alphas)
        ctrl.observe(MEASURED)
        assert ctrl.targets == MEASURED  # first measurement becomes the target
        assert ctrl.alphas == alphas_before  # adoption step moves no alpha

    def test_feedback_sign_and_clamp(self):
        ctrl = _ctrl(eta=1.0, max_log_step=0.1)
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        a0 = dict(ctrl.alphas)
        # Entropy below target -> alpha rises; above -> falls.
        low = {h: v - 0.02 for h, v in MEASURED.items()}
        ctrl.observe(low)
        assert all(ctrl.alphas[h] > a0[h] for h in HEADS)
        a1 = dict(ctrl.alphas)
        high = {h: v + 0.05 for h, v in MEASURED.items()}
        ctrl.observe(high)
        assert all(ctrl.alphas[h] < a1[h] for h in HEADS)
        # A huge error is clamped to max_log_step per update.
        a2 = dict(ctrl.alphas)
        ctrl.observe({h: v - 5.0 for h, v in MEASURED.items()})
        for h in HEADS:
            assert ctrl.alphas[h] <= a2[h] * math.exp(0.1) + 1e-12

    def test_alpha_bounds(self):
        ctrl = _ctrl()
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        for _ in range(400):
            ctrl.observe({h: v + 0.5 for h, v in MEASURED.items()})
        assert all(ctrl.alphas[h] >= ALPHA_MIN[h] for h in HEADS)
        for _ in range(400):
            ctrl.observe({h: v - 0.5 for h, v in MEASURED.items()})
        assert all(ctrl.alphas[h] <= ALPHA_MAX[h] for h in HEADS)

    def test_missing_head_measurement_skipped(self):
        ctrl = _ctrl()
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        a0 = dict(ctrl.alphas)
        ctrl.observe({"pick": None, "play": MEASURED["play"] - 0.02})
        assert ctrl.alphas["pick"] == a0["pick"]  # no measurement, no move
        assert ctrl.alphas["play"] > a0["play"]


class TestOuterLoop:
    def test_hold_heads_never_step(self):
        ctrl = _ctrl()
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        moved = ctrl.step_targets()
        assert set(moved) == {"play"}
        for h in ("pick", "partner", "bury"):
            assert ctrl.targets[h] == MEASURED[h]
            assert ctrl.head_at_floor(h)  # holds are trivially at floor

    def test_play_step_geometry_and_floor(self):
        ctrl = _ctrl(retain=0.75, min_step=0.03)
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        floor = ctrl.config.floors["play"]
        old = ctrl.targets["play"]
        moved = ctrl.step_targets()
        o, n = moved["play"]
        assert o == old
        assert abs(n - (floor + 0.75 * (old - floor))) < 1e-12
        # The ladder terminates: once the would-be step is < min_step the
        # head reports at_floor and step_targets stops moving it.
        steps = 0
        while not ctrl.at_floor():
            assert ctrl.step_targets()
            steps += 1
            assert steps < 50
        assert ctrl.step_targets() == {}
        gap = ctrl.targets["play"] - floor
        assert 0.0 <= (1 - 0.75) * gap < 0.03
        # Backfill-scale sanity: from 0.75 with floor 0.28 the ladder is a
        # handful of generations, not dozens.
        assert 3 <= steps <= 8

    def test_uninitialized_target_not_at_floor(self):
        ctrl = _ctrl()
        assert not ctrl.head_at_floor("play")  # cannot be judged converged
        assert ctrl.step_targets() == {}  # ...but there is nothing to step


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        ctrl = _ctrl(eta=1.4, retain=0.7, min_step=0.02)
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        ctrl.step_targets()
        path = str(tmp_path / "entropy_controller.json")
        ctrl.save(path)
        back = EntropyTargetController.load(path)
        assert back.targets == ctrl.targets
        assert back.alphas == ctrl.alphas
        assert back.config == ctrl.config


class TestWiring:
    def test_trainer_defaults_off(self):
        from sheepshead.training.train_league_ppo import build_arg_parser

        args = build_arg_parser().parse_args(["--resume", "x.pt", "--league-dir", "y"])
        assert args.entropy_mode == "schedule"
        on = build_arg_parser().parse_args(
            [
                "--resume",
                "x.pt",
                "--league-dir",
                "y",
                "--entropy-mode",
                "target",
                "--entropy-target-play",
                "0.75",
            ]
        )
        assert on.entropy_mode == "target"
        assert on.entropy_target_play == 0.75
        assert on.entropy_target_pick is None
        assert on.entropy_play_floor == 0.28

    def test_orchestrator_default_on_with_opt_out(self):
        # Operator adoption 2026-07-28: adaptive entropy is the default
        # orchestrator behavior (still deferred past generation 1);
        # --no-adaptive-entropy restores the pure legacy schedule.
        from sheepshead.training.run_extended_league import parse_args

        args = parse_args(["--resume", "x.pt", "--run-name", "t", "--panel", "a.pt"])
        assert args.adaptive_entropy is True
        assert args.entropy_play_floor == 0.28
        off = parse_args(
            [
                "--resume",
                "x.pt",
                "--run-name",
                "t",
                "--panel",
                "a.pt",
                "--no-adaptive-entropy",
            ]
        )
        assert off.adaptive_entropy is False

    def test_adaptive_entropy_defers_to_generation_two(self, tmp_path, monkeypatch):
        """--adaptive-entropy leaves generation 1 on the legacy schedule
        (the validated seed-transient phase: organic hold-head sharpening
        and the high-entropy play window across the league transition) and
        enables target mode from generation 2, where bumpless attachment
        captures a settled operating point. This is what makes the single
        flag correct for from-scratch reproductions — no target derivation
        step exists in the flow."""
        monkeypatch.chdir(tmp_path)
        from sheepshead.training.run_extended_league import Orchestrator, parse_args

        args = parse_args(
            [
                "--resume",
                "seed.pt",
                "--run-name",
                "t",
                "--panel",
                "a.pt",
                "--adaptive-entropy",
            ]
        )
        args.arch = "perceiver-shared-v2"  # normally set by preflight
        orch = Orchestrator(args)
        # _resume_for(2) globs for the gen-1 boundary checkpoint by name.
        ckpt_dir = tmp_path / "runs" / "t" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "pfsp_perceiver-shared-v2_checkpoint_1000000.pt").touch()
        gen1 = " ".join(orch.trainer_cmd(1, None))
        gen2 = " ".join(orch.trainer_cmd(2, None))
        assert "--entropy-mode" not in gen1
        assert "--entropy-mode target" in gen2
        assert "--entropy-play-floor 0.28" in gen2


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
