#!/usr/bin/env python3
"""Called-suit-lead adherence probe invariants (sheepshead/analysis/called_suit_probe.py)."""

import unittest

from sheepshead.analysis.called_suit_probe import _called_suit_fail, probe_agent
from sheepshead.analysis.trump_lead_probe import PROBE_SEED
from sheepshead.scripted_agent import ScriptedAgent, _card


class TestCalledSuitProbe(unittest.TestCase):
    def test_scripted_hero_trick0_adherence_is_total_and_deterministic(self):
        # Instrument self-check: the conventions agent leads the called suit
        # through on trick 0 by construction, so its trick-0 adherence must be
        # exactly 1.0 — and the probe must find real opportunities.
        r = probe_agent(ScriptedAgent(), n_deals=80)
        self.assertGreater(r["eligible_trick0"], 10)
        self.assertEqual(r["adherent_trick0"], r["eligible_trick0"])
        self.assertEqual(r["adherence_rate_trick0"], 1.0)
        # Fixed CRN deal set: an identical call reproduces identical counts.
        r2 = probe_agent(ScriptedAgent(), n_deals=80)
        self.assertEqual(r["eligible"], r2["eligible"])
        self.assertEqual(r["by_trick"], r2["by_trick"])

    def test_called_suit_avoider_is_fully_flagged(self):
        # A hero that never leads the called suit (when it has any other card)
        # must show 0 adherence on every node where an alternative existed:
        # validates the detection logic end to end.
        class CalledSuitAvoider(ScriptedAgent):
            def _lead(self, state, cards):
                called = _card(int(state["called_card_id"]))
                if called:
                    off = [c for c in cards if not _called_suit_fail(c, called)]
                    if off:
                        return off[0]
                return super()._lead(state, cards)

        r = probe_agent(CalledSuitAvoider(), n_deals=80)
        self.assertGreater(r["eligible"], 20)
        # Eligibility requires a non-called-suit alternative, so the avoider
        # can never be counted adherent.
        self.assertEqual(r["adherent"], 0)
        self.assertEqual(r["adherent_under"], 0)
        assert PROBE_SEED == 20260702  # frozen: results are comparable forever


if __name__ == "__main__":
    unittest.main(verbosity=2)
