#!/usr/bin/env python3
"""Stopping-rule math for the extended league run, on synthetic curves.

Every case builds deal-aligned per-deal endpoint vectors with a shared
heavy-tailed deal effect (the structure CRN pairing exploits) plus
idiosyncratic noise, then walks the generation loop exactly the way
sheepshead/training/run_extended_league.py does: flat_verdict per generation, then
decide_stop over the accumulated flat history.
"""

import numpy as np
import pytest

from sheepshead.analysis.league_stopping import (
    CalibrationChoice,
    ProbeSummary,
    StopRuleConfig,
    bootstrap_deal_indices,
    confirmation_verdict,
    decide_stop,
    flat_verdict,
    pick_anchor_coeff,
    resume_from_cap,
)

N_DEALS = 4000
N_BOOT = 2000  # plenty for test resolution, keeps the suite fast
CFG = StopRuleConfig()


def synth_endpoints(mu_by_gen, n_deals=N_DEALS, seed=0):
    """Per-deal endpoint vectors: shared heavy-tailed deal effect (sd ~1.2,
    t3-shaped like multiplier-inflated scores) + idiosyncratic noise sd 0.5."""
    rng = np.random.default_rng(seed)
    deal_effect = rng.standard_t(df=3, size=n_deals) * 1.2
    return {
        g: mu + deal_effect + rng.normal(0.0, 0.5, size=n_deals)
        for g, mu in mu_by_gen.items()
    }


def walk_generations(mu_by_gen, h2h_se=0.01, cfg=CFG, seed=0):
    """Run the orchestrator's decision loop over synthetic generations.

    Returns (verdicts, decisions) keyed by generation. Head-to-head edges are
    fed as the true consecutive-generation gap plus a little noise, with the
    given SE -- the h2h probe is a separate measurement in the real system.
    """
    per_deal = synth_endpoints(mu_by_gen, seed=seed)
    n = len(per_deal[0])
    boot_idx = bootstrap_deal_indices(n, N_BOOT, np.random.default_rng(42))
    edge_rng = np.random.default_rng(seed + 1)

    means = {0: float(per_deal[0].mean())}
    flat_history, verdicts, decisions = [], {}, {}
    for g in sorted(k for k in mu_by_gen if k >= 1):
        true_edge = mu_by_gen[g] - mu_by_gen[g - 1]
        edge = true_edge + float(edge_rng.normal(0.0, h2h_se))
        v = flat_verdict(g, per_deal, means, edge, h2h_se, boot_idx, cfg)
        means[g] = float(per_deal[g].mean())
        verdicts[g] = v
        flat_history.append(v.flat)
        decisions[g] = decide_stop(flat_history, g, cfg)
    return per_deal, verdicts, decisions


class TestFlatTruth:
    """A genuinely converged run stops at the min-generations floor."""

    def test_stops_at_floor_and_not_before(self):
        mu = {g: 0.0 for g in range(0, 7)}
        _, verdicts, decisions = walk_generations(mu)
        for g, v in verdicts.items():
            assert v.flat, f"gen {g} spuriously improving"
        for g in range(1, CFG.min_generations):
            assert not decisions[g].stop_candidate, f"stopped early at {g}"
        assert decisions[CFG.min_generations].stop_candidate
        assert not decisions[CFG.min_generations].forced_by_cap


class TestSlowClimb:
    """+0.015/generation: real but below every threshold -> stop by design,
    with the slope's statistical reality surfaced as a report caveat."""

    def test_stops_with_caveat_flag(self):
        mu = {g: 0.015 * g for g in range(0, 7)}
        _, verdicts, decisions = walk_generations(mu)
        g_stop = next(g for g in sorted(decisions) if decisions[g].stop_candidate)
        assert g_stop <= CFG.min_generations + 1
        # The paired slope sees the climb even though the rule stops.
        caveats = [
            v.slope.small_but_significant
            for v in verdicts.values()
            if v.slope is not None
        ]
        assert any(caveats), "slow climb never flagged as significant"
        for v in verdicts.values():
            if v.slope is not None:
                assert not v.slope.climbing


class TestRealLearning:
    """Clear per-generation gains keep the run alive via A and C."""

    def test_no_stop_while_climbing(self):
        mu = {g: 0.10 * g for g in range(0, 6)}
        _, verdicts, decisions = walk_generations(mu)
        for g, v in verdicts.items():
            assert not v.flat, f"gen {g} wrongly flat"
            assert v.gain.improving
            if v.slope is not None:
                assert v.slope.climbing
        assert not any(d.stop_candidate for d in decisions.values())


class TestStepJumpFalsePlateau:
    """onehot-ff lesson: flat for several generations, then a real jump."""

    def test_jump_breaks_the_streak(self):
        mu = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.15, 5: 0.15, 6: 0.15, 7: 0.15}
        _, verdicts, decisions = walk_generations(mu)
        assert verdicts[3].flat  # gens 1-3 flat
        assert not verdicts[4].flat, "the jump generation must not be flat"
        # The 3-endpoint slope window still sees the jump at gen 5 (g5 vs g3),
        # so the rule conservatively keeps the run alive one extra generation.
        assert not verdicts[5].flat
        assert verdicts[5].slope.climbing
        assert not decisions[4].stop_candidate
        assert not decisions[5].stop_candidate
        assert not decisions[6].stop_candidate  # streak = 1
        assert decisions[7].stop_candidate

    def test_confirmation_contradiction_path(self):
        """A stop candidate whose fresh-deal panel reveals a real recent gain
        must contradict (the orchestrator then resumes training)."""
        conf = synth_endpoints({0: 0.0, 1: 0.10}, seed=7)
        boot_idx = bootstrap_deal_indices(N_DEALS, N_BOOT, np.random.default_rng(1))
        cv = confirmation_verdict(conf[1], conf[0], boot_idx, CFG)
        assert cv.contradiction

        flat_conf = synth_endpoints({0: 0.0, 1: 0.0}, seed=8)
        cv2 = confirmation_verdict(flat_conf[1], flat_conf[0], boot_idx, CFG)
        assert not cv2.contradiction


class TestSafetyCap:
    def test_cap_forces_stop_even_while_improving(self):
        mu = {g: 0.10 * g for g in range(0, CFG.max_generations + 1)}
        _, _, decisions = walk_generations(mu)
        d = decisions[CFG.max_generations]
        assert d.stop_candidate
        assert d.forced_by_cap
        assert not decisions[CFG.max_generations - 1].forced_by_cap


class TestResumeFromCap:
    def test_cap_reopens_only_with_raised_cap(self):
        # Concluded at the 12-gen cap; relaunching with a higher cap resumes.
        assert resume_from_cap("cap", 12, 16)
        # Same or lower cap: still concluded, nothing to do.
        assert not resume_from_cap("cap", 12, 12)
        assert not resume_from_cap("cap", 12, 8)

    def test_confirmed_plateau_never_auto_resumes(self):
        # "stopped" is the learning-completion verdict; a raised cap must
        # not reopen it (override = deliberate flat_history reset).
        assert not resume_from_cap("stopped", 6, 16)
        assert not resume_from_cap("running", 3, 12)
        assert not resume_from_cap("needs_review", 3, 12)


class TestDecideStopValidation:
    def test_history_length_must_match_generation(self):
        with pytest.raises(ValueError):
            decide_stop([True, True], 3, CFG)


class TestPickAnchorCoeff:
    BASELINE_PICK = 30.0  # percent, greedy_health_probe units

    @staticmethod
    def probe(coeff, kl_last, kl_max, violations=0, pick=30.0):
        return ProbeSummary(
            coeff=coeff,
            kl_last=kl_last,
            kl_max=kl_max,
            gate_violations=violations,
            final_pick_rate=pick,
        )

    def test_smallest_qualifying_wins(self):
        probes = [
            self.probe(0.3, kl_last=0.20, kl_max=0.40),  # exploding KL
            self.probe(1.0, kl_last=0.03, kl_max=0.06),
            self.probe(3.0, kl_last=0.01, kl_max=0.02),
        ]
        choice = pick_anchor_coeff(probes, self.BASELINE_PICK)
        assert isinstance(choice, CalibrationChoice)
        assert choice.coeff == 1.0
        assert choice.qualified

    def test_gate_violation_disqualifies(self):
        probes = [
            self.probe(0.3, kl_last=0.02, kl_max=0.04, violations=2),
            self.probe(1.0, kl_last=0.02, kl_max=0.04),
        ]
        assert pick_anchor_coeff(probes, self.BASELINE_PICK).coeff == 1.0

    def test_pick_rate_drift_disqualifies(self):
        probes = [
            self.probe(0.3, kl_last=0.02, kl_max=0.04, pick=45.0),  # +15 pts
            self.probe(1.0, kl_last=0.02, kl_max=0.04, pick=33.0),
        ]
        assert pick_anchor_coeff(probes, self.BASELINE_PICK).coeff == 1.0

    def test_fallback_is_largest_and_flagged(self):
        probes = [
            self.probe(0.3, kl_last=0.30, kl_max=0.50),
            self.probe(1.0, kl_last=0.20, kl_max=0.40),
        ]
        choice = pick_anchor_coeff(probes, self.BASELINE_PICK)
        assert choice.coeff == 1.0
        assert not choice.qualified
        assert "fallback" in choice.reason


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
