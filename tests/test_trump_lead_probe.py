#!/usr/bin/env python3
"""Trump-lead incidence probe invariants (analysis/trump_lead_probe.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.trump_lead_probe import PROBE_SEED, probe_agent
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import PARTNER_BY_CALLED_ACE


class TestTrumpLeadProbe(unittest.TestCase):
    def test_scripted_hero_never_flagged_and_probe_is_deterministic(self):
        # Instrument self-check: the conventions agent never makes the
        # diagnosed lead, so its measured rate must be exactly 0 — and the
        # probe must find real opportunities (otherwise it measures nothing).
        r = probe_agent(ScriptedAgent(), n_deals=80, partner_mode=PARTNER_BY_CALLED_ACE)
        self.assertGreater(r["opportunities"], 20)
        self.assertEqual(r["trump_leads"], 0)
        self.assertEqual(r["implied_ev_per_1000_hands"], 0.0)
        # Fixed CRN deal set: an identical call reproduces identical counts.
        r2 = probe_agent(
            ScriptedAgent(), n_deals=80, partner_mode=PARTNER_BY_CALLED_ACE
        )
        self.assertEqual(r["opportunities"], r2["opportunities"])
        self.assertEqual(r["by_trick"], r2["by_trick"])

    def test_always_trump_leader_is_fully_flagged(self):
        # A hero that always leads trump when it can must be flagged on 100%
        # of opportunities: validates the detection logic end to end.
        from sheepshead import TRUMP

        trump = set(TRUMP)

        class TrumpLeader(ScriptedAgent):
            def _lead(self, state, cards):
                trumps = [c for c in cards if c in trump]
                if trumps:
                    return trumps[0]
                return super()._lead(state, cards)

        r = probe_agent(TrumpLeader(), n_deals=80, partner_mode=PARTNER_BY_CALLED_ACE)
        self.assertGreater(r["opportunities"], 20)
        self.assertEqual(r["trump_leads"], r["opportunities"])
        self.assertLess(r["implied_ev_per_1000_hands"], 0.0)
        assert PROBE_SEED == 20260702  # frozen: results are comparable forever


if __name__ == "__main__":
    unittest.main(verbosity=2)
