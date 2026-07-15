#!/usr/bin/env python3
"""ConventionWrapper invariants (sheepshead/agent/convention_wrapper.py).

The convention probes are the measuring instruments: wrapping a deliberately
convention-violating hero must drive its measured violation rate to exactly
zero, end to end through the real engine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead.agent.convention_wrapper import (
    ConventionWrapper,
    parse_wrap_spec,
    wrap_agent,
)
from sheepshead.analysis.called_suit_probe import _called_suit_fail, probe_agent
from sheepshead.analysis.trump_lead_probe import probe_agent as probe_trump
from sheepshead.scripted_agent import ScriptedAgent, _card
from sheepshead import PARTNER_BY_CALLED_ACE, TRUMP

_TRUMP = set(TRUMP)


class TrumpLeader(ScriptedAgent):
    def _lead(self, state, cards):
        trumps = [c for c in cards if c in _TRUMP]
        if trumps:
            return trumps[0]
        return super()._lead(state, cards)


class CalledSuitAvoider(ScriptedAgent):
    def _lead(self, state, cards):
        called = _card(int(state["called_card_id"]))
        if called:
            off = [c for c in cards if not _called_suit_fail(c, called)]
            if off:
                return off[0]
        return super()._lead(state, cards)


class TestConventionWrapper(unittest.TestCase):
    def test_c1_mask_zeroes_a_trump_leader(self):
        raw = probe_trump(TrumpLeader(), n_deals=60, partner_mode=PARTNER_BY_CALLED_ACE)
        self.assertEqual(raw["trump_leads"], raw["opportunities"])  # violates freely
        wrapped = probe_trump(
            ConventionWrapper(TrumpLeader(), c1=True, c2=False),
            n_deals=60,
            partner_mode=PARTNER_BY_CALLED_ACE,
        )
        self.assertGreater(wrapped["opportunities"], 10)
        self.assertEqual(wrapped["trump_leads"], 0)

    def test_c2_force_makes_an_avoider_adherent_at_trick0(self):
        raw = probe_agent(CalledSuitAvoider(), n_deals=60)
        self.assertEqual(raw["adherent_trick0"], 0)  # violates freely
        wrapped = probe_agent(
            ConventionWrapper(CalledSuitAvoider(), c1=False, c2=True), n_deals=60
        )
        self.assertGreater(wrapped["eligible_trick0"], 5)
        self.assertEqual(wrapped["adherent_trick0"], wrapped["eligible_trick0"])
        # Trick-0 only: later-trick eligibility is unknowable from the state
        # dict, so the wrapper must NOT have forced anything there.
        self.assertEqual(
            wrapped["adherent"] - wrapped["adherent_trick0"],
            raw["adherent"] - raw["adherent_trick0"],
        )

    def test_wrap_specs(self):
        self.assertEqual(parse_wrap_spec("m.pt"), ("m.pt", None))
        self.assertEqual(parse_wrap_spec("m.pt@c1c2"), ("m.pt", "c1c2"))
        with self.assertRaises(ValueError):
            parse_wrap_spec("m.pt@bogus")
        w = wrap_agent(ScriptedAgent(), "c1")
        self.assertTrue(w.c1)
        self.assertFalse(w.c2)
        # A typo'd deploy config must fail fast, not silently no-op.
        with self.assertRaises(ValueError):
            wrap_agent(ScriptedAgent(), "c3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
