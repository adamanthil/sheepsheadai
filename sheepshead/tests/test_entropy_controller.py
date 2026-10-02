"""Target-entropy controller invariants (one-sided hold).

Inner loop: bumpless target adoption, integral-feedback sign, per-update
LINEAR clamp, and the one-sided bound — sustained entropy-above-target
drives alpha to exactly zero and no further (a target is a floor; reward
decides everything above it). Outer loop: hold heads never step, anneal
heads step geometrically toward their floor and stop at min_step.
Persistence: sidecar roundtrip plus migration of older sidecars (log-space
gains; the retired signed controller's penalty range). Wiring: CLI
surfaces.

Refs: Haarnoja et al. arXiv:1812.05905 (target-entropy temperature);
Christodoulou arXiv:1910.07207 (discrete fraction-of-max targets);
Jaderberg et al. arXiv:1711.09846 (perturbation-scale outer steps);
notebooks/CE_Teacher_Design_202608.md §4 (inner-loop gains)."""

import math
from types import SimpleNamespace

from sheepshead.training.entropy_controller import (
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

    def test_feedback_sign_and_linear_clamp(self):
        ctrl = _ctrl()
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
        # A huge error is clamped to max_step (absolute, not multiplicative).
        a2 = dict(ctrl.alphas)
        deltas = ctrl.observe({h: v - 5.0 for h, v in MEASURED.items()})
        for h in HEADS:
            assert abs(deltas[h]) <= ctrl.config.max_step + 1e-12
            assert ctrl.alphas[h] <= a2[h] + ctrl.config.max_step + 1e-12

    def test_alpha_bounds(self):
        ctrl = _ctrl()
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        for _ in range(400):
            ctrl.observe({h: v + 0.5 for h, v in MEASURED.items()})
        assert all(ctrl.alphas[h] >= 0.0 for h in HEADS)
        for _ in range(400):
            ctrl.observe({h: v - 0.5 for h, v in MEASURED.items()})
        assert all(ctrl.alphas[h] <= ctrl.config.alpha_max + 1e-12 for h in HEADS)

    def test_sustained_above_target_rests_at_zero(self):
        """The hold is one-sided: a head that reward keeps above its target
        drives alpha to exactly zero and holds there — no entropy penalty —
        and relieving the pressure walks it back up."""
        ctrl = _ctrl()
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        assert all(ctrl.alphas[h] > 0.0 for h in HEADS)
        for _ in range(200):
            ctrl.observe({h: v + 0.5 for h, v in MEASURED.items()})
        assert all(ctrl.alphas[h] == 0.0 for h in HEADS)
        for _ in range(200):
            ctrl.observe({h: v - 0.5 for h, v in MEASURED.items()})
        assert all(ctrl.alphas[h] > 0.0 for h in HEADS)

    def test_missing_head_measurement_skipped(self):
        ctrl = _ctrl()
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        a0 = dict(ctrl.alphas)
        ctrl.observe({"pick": None, "play": MEASURED["play"] - 0.02})
        assert ctrl.alphas["pick"] == a0["pick"]  # no measurement, no move
        assert ctrl.alphas["play"] > a0["play"]

    def test_gain_matches_v1_at_calibration_point(self):
        """eta_lin=0.15 is v1's eta=1.0 response linearized at alpha=0.15
        (CE_Teacher_Design_202608.md §4). At the backfill's organic drift
        error the two agree to well within a percent."""
        ctrl = _ctrl()
        ctrl.alphas = {h: 0.15 for h in HEADS}
        ctrl.targets = {h: 0.5 for h in HEADS}
        err = 0.057
        deltas = ctrl.observe({h: 0.5 - err for h in HEADS})
        v1 = 0.15 * (math.exp(1.0 * err) - 1.0)
        for h in HEADS:
            assert abs(deltas[h] - 0.15 * err) < 1e-12
            assert abs(deltas[h] - v1) / v1 < 0.05


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
        ctrl = _ctrl(eta_lin=0.2, max_step=0.02, alpha_max=0.3, retain=0.7)
        ctrl.attach(_agent())
        ctrl.observe(MEASURED)
        for _ in range(50):
            ctrl.observe({h: v + 0.5 for h, v in MEASURED.items()})
        ctrl.step_targets()
        path = str(tmp_path / "entropy_controller.json")
        ctrl.save(path)
        back = EntropyTargetController.load(path)
        assert back.targets == ctrl.targets
        assert back.alphas == ctrl.alphas
        assert back.config == ctrl.config

    def test_legacy_sidecar_migrates(self):
        """A sidecar from the log-space controller loads: state carries
        over, the dead gains are dropped for the current defaults, and
        re-attaching an agent is still bumpless in alpha."""
        v1 = {
            "targets": {"pick": 0.05, "partner": 0.12, "bury": 0.16, "play": 0.61},
            "alphas": {
                "pick": 0.041,
                "partner": 0.052,
                "bury": 0.033,
                "play": 0.0201,
            },
            "config": {
                "eta": 1.0,
                "max_log_step": 0.1,
                "retain": 0.75,
                "min_step": 0.03,
                "anneal_heads": ["play"],
                "floors": {"play": 0.28},
            },
        }
        ctrl = EntropyTargetController.from_dict(v1)
        assert ctrl.targets == v1["targets"]
        assert ctrl.alphas == v1["alphas"]
        assert ctrl.config.floors == {"play": 0.28}
        assert ctrl.config.anneal_heads == ("play",)
        defaults = EntropyControllerConfig()
        assert ctrl.config.eta_lin == defaults.eta_lin
        assert ctrl.config.max_step == defaults.max_step
        assert ctrl.config.alpha_max == defaults.alpha_max
        # Bumpless: the stored alphas win over the agent's coefficients.
        agent = _agent()
        ctrl.attach(agent)
        for h in HEADS:
            assert getattr(agent, f"entropy_coeff_{h}") == v1["alphas"][h]

    def test_signed_sidecar_penalty_clamps_to_zero(self):
        """A sidecar from the retired signed controller may carry a
        negative coefficient (an entropy penalty) plus its penalty-floor
        and sign-flip keys. The penalty loads as zero, the keys are
        ignored, and the next save writes neither."""
        signed = {
            "targets": {"pick": 0.079, "partner": 0.057, "bury": 0.147, "play": 0.58},
            "alphas": {"pick": 0.233, "partner": -0.05, "bury": -0.05, "play": 0.009},
            "sign_flips": {"pick": 10, "partner": 10, "bury": 43, "play": 204},
            "config": {
                "eta_lin": 0.15,
                "max_step": 0.015,
                "alpha_min": -0.05,
                "alpha_max": 0.25,
                "retain": 0.75,
                "min_step": 0.03,
                "anneal_heads": ["play"],
                "floors": {"play": 0.28},
            },
        }
        ctrl = EntropyTargetController.from_dict(signed)
        assert ctrl.alphas == {
            "pick": 0.233,
            "partner": 0.0,
            "bury": 0.0,
            "play": 0.009,
        }
        assert ctrl.targets == signed["targets"]
        d = ctrl.to_dict()
        assert "sign_flips" not in d
        assert "alpha_min" not in d["config"]


class TestWiring:
    def test_trainer_controller_defaults_per_phase(self):
        # Training_Program_Redesign §4.3: the controller owns the coefficients
        # in the league phases (bumpless attach at the settled operating
        # point); the bootstrap runs its own fixed linear schedule; the
        # orchestrator opts league generation 1 out explicitly.
        from sheepshead.training.train_ppo import build_arg_parser, resolve_args

        league = build_arg_parser().parse_args(
            ["--phase", "league", "--run-name", "x", "--until", "1"]
        )
        resolve_args(league)
        assert league.entropy_controller is True
        assert league.entropy_play_floor == 0.28
        boot = build_arg_parser().parse_args(
            ["--phase", "bootstrap", "--run-name", "x", "--until", "1"]
        )
        resolve_args(boot)
        assert boot.entropy_controller is False
        off = build_arg_parser().parse_args(
            [
                "--phase",
                "league",
                "--run-name",
                "x",
                "--until",
                "1",
                "--no-entropy-controller",
            ]
        )
        resolve_args(off)
        assert off.entropy_controller is False
