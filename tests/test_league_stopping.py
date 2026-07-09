#!/usr/bin/env python3
"""Stopping-rule math for the extended league run, on synthetic curves.

Every case builds deal-aligned per-deal endpoint vectors with a shared
heavy-tailed deal effect (the structure CRN pairing exploits) plus
idiosyncratic noise, then walks the generation loop exactly the way
analysis/run_extended_league.py does: flat_verdict per generation, then
decide_stop over the accumulated flat history.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.league_stopping import (
    CalibrationChoice,
    ProbeSummary,
    StopRuleConfig,
    bootstrap_deal_indices,
    confirmation_verdict,
    decide_stop,
    flat_verdict,
    pick_anchor_coeff,
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


class TestFlatTruth(unittest.TestCase):
    """A genuinely converged run stops at the min-generations floor."""

    def test_stops_at_floor_and_not_before(self):
        mu = {g: 0.0 for g in range(0, 7)}
        _, verdicts, decisions = walk_generations(mu)
        for g, v in verdicts.items():
            self.assertTrue(v.flat, f"gen {g} spuriously improving")
        for g in range(1, CFG.min_generations):
            self.assertFalse(decisions[g].stop_candidate, f"stopped early at {g}")
        self.assertTrue(decisions[CFG.min_generations].stop_candidate)
        self.assertFalse(decisions[CFG.min_generations].forced_by_cap)


class TestSlowClimb(unittest.TestCase):
    """+0.015/generation: real but below every threshold -> stop by design,
    with the slope's statistical reality surfaced as a report caveat."""

    def test_stops_with_caveat_flag(self):
        mu = {g: 0.015 * g for g in range(0, 7)}
        _, verdicts, decisions = walk_generations(mu)
        g_stop = next(g for g in sorted(decisions) if decisions[g].stop_candidate)
        self.assertLessEqual(g_stop, CFG.min_generations + 1)
        # The paired slope sees the climb even though the rule stops.
        caveats = [
            v.slope.small_but_significant
            for v in verdicts.values()
            if v.slope is not None
        ]
        self.assertTrue(any(caveats), "slow climb never flagged as significant")
        for v in verdicts.values():
            if v.slope is not None:
                self.assertFalse(v.slope.climbing)


class TestRealLearning(unittest.TestCase):
    """Clear per-generation gains keep the run alive via A and C."""

    def test_no_stop_while_climbing(self):
        mu = {g: 0.10 * g for g in range(0, 6)}
        _, verdicts, decisions = walk_generations(mu)
        for g, v in verdicts.items():
            self.assertFalse(v.flat, f"gen {g} wrongly flat")
            self.assertTrue(v.gain.improving)
            if v.slope is not None:
                self.assertTrue(v.slope.climbing)
        self.assertFalse(any(d.stop_candidate for d in decisions.values()))


class TestStepJumpFalsePlateau(unittest.TestCase):
    """onehot-ff lesson: flat for several generations, then a real jump."""

    def test_jump_breaks_the_streak(self):
        mu = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.15, 5: 0.15, 6: 0.15, 7: 0.15}
        _, verdicts, decisions = walk_generations(mu)
        self.assertTrue(verdicts[3].flat)  # gens 1-3 flat
        self.assertFalse(verdicts[4].flat, "the jump generation must not be flat")
        # The 3-endpoint slope window still sees the jump at gen 5 (g5 vs g3),
        # so the rule conservatively keeps the run alive one extra generation.
        self.assertFalse(verdicts[5].flat)
        self.assertTrue(verdicts[5].slope.climbing)
        self.assertFalse(decisions[4].stop_candidate)
        self.assertFalse(decisions[5].stop_candidate)
        self.assertFalse(decisions[6].stop_candidate)  # streak = 1
        self.assertTrue(decisions[7].stop_candidate)

    def test_confirmation_contradiction_path(self):
        """A stop candidate whose fresh-deal panel reveals a real recent gain
        must contradict (the orchestrator then resumes training)."""
        conf = synth_endpoints({0: 0.0, 1: 0.10}, seed=7)
        boot_idx = bootstrap_deal_indices(N_DEALS, N_BOOT, np.random.default_rng(1))
        cv = confirmation_verdict(conf[1], conf[0], boot_idx, CFG)
        self.assertTrue(cv.contradiction)

        flat_conf = synth_endpoints({0: 0.0, 1: 0.0}, seed=8)
        cv2 = confirmation_verdict(flat_conf[1], flat_conf[0], boot_idx, CFG)
        self.assertFalse(cv2.contradiction)


class TestSafetyCap(unittest.TestCase):
    def test_cap_forces_stop_even_while_improving(self):
        mu = {g: 0.10 * g for g in range(0, CFG.max_generations + 1)}
        _, _, decisions = walk_generations(mu)
        d = decisions[CFG.max_generations]
        self.assertTrue(d.stop_candidate)
        self.assertTrue(d.forced_by_cap)
        self.assertFalse(decisions[CFG.max_generations - 1].forced_by_cap)


class TestDecideStopValidation(unittest.TestCase):
    def test_history_length_must_match_generation(self):
        with self.assertRaises(ValueError):
            decide_stop([True, True], 3, CFG)


class TestPickAnchorCoeff(unittest.TestCase):
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
        self.assertIsInstance(choice, CalibrationChoice)
        self.assertEqual(choice.coeff, 1.0)
        self.assertTrue(choice.qualified)

    def test_gate_violation_disqualifies(self):
        probes = [
            self.probe(0.3, kl_last=0.02, kl_max=0.04, violations=2),
            self.probe(1.0, kl_last=0.02, kl_max=0.04),
        ]
        self.assertEqual(pick_anchor_coeff(probes, self.BASELINE_PICK).coeff, 1.0)

    def test_pick_rate_drift_disqualifies(self):
        probes = [
            self.probe(0.3, kl_last=0.02, kl_max=0.04, pick=45.0),  # +15 pts
            self.probe(1.0, kl_last=0.02, kl_max=0.04, pick=33.0),
        ]
        self.assertEqual(pick_anchor_coeff(probes, self.BASELINE_PICK).coeff, 1.0)

    def test_fallback_is_largest_and_flagged(self):
        probes = [
            self.probe(0.3, kl_last=0.30, kl_max=0.50),
            self.probe(1.0, kl_last=0.20, kl_max=0.40),
        ]
        choice = pick_anchor_coeff(probes, self.BASELINE_PICK)
        self.assertEqual(choice.coeff, 1.0)
        self.assertFalse(choice.qualified)
        self.assertIn("fallback", choice.reason)


if __name__ == "__main__":
    unittest.main()
