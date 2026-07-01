#!/usr/bin/env python3
"""ScriptedAgent invariants: legality, determinism, zero-sum self-play."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripted_agent import ScriptedAgent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game


class TestScriptedAgent(unittest.TestCase):
    def test_selfplay_legal_deterministic_zero_sum(self):
        ag = ScriptedAgent()
        for g in range(60):
            mode = PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD
            game = Game(partner_selection_mode=mode, seed=5_000_000 + g)
            while not game.is_done():
                for player in game.players:
                    valid = player.get_valid_action_ids()
                    while valid:
                        a, logp, val = ag.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=True,
                        )
                        # Legality and a stable repeat decision.
                        self.assertIn(a, valid)
                        a2, _, _ = ag.act(
                            player.get_state_dict(), valid, player.position
                        )
                        self.assertEqual(a, a2)
                        self.assertEqual((logp, val), (0.0, 0.0))
                        player.act(a)
                        valid = player.get_valid_action_ids()
            scores = [p.get_score() for p in game.players]
            self.assertEqual(sum(scores), 0, f"game {g}: {scores}")

    def test_defender_never_leads_trump_holding_fail(self):
        # The convention the 30M lineage leaks against: as an unrevealed
        # defender with fail in hand, the scripted agent must not lead trump.
        from sheepshead import TRUMP

        trump_set = set(TRUMP)
        ag = ScriptedAgent()
        checked = 0
        for g in range(200):
            game = Game(
                partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=7_000_000 + g
            )
            while not game.is_done():
                for player in game.players:
                    valid = player.get_valid_action_ids()
                    while valid:
                        state = player.get_state_dict()
                        a, _, _ = ag.act(state, valid, player.position)
                        is_lead = (
                            game.play_started
                            and not game.is_leaster
                            and game.leader == player.position
                            and all(int(x) == 0 for x in state["trick_card_ids"])
                        )
                        is_hidden_partner = (
                            game.called_card and game.called_card in player.hand
                        )
                        if is_lead and not (
                            player.is_picker or player.is_partner or is_hidden_partner
                        ):
                            card_played = None
                            from sheepshead import ACTION_LOOKUP

                            name = ACTION_LOOKUP[a]
                            if name.startswith("PLAY "):
                                card_played = name.split(" ", 1)[1]
                            has_fail = any(c not in trump_set for c in player.hand)
                            if card_played and has_fail:
                                self.assertNotIn(
                                    card_played, trump_set, f"game {g}: led trump"
                                )
                                checked += 1
                        player.act(a)
                        valid = player.get_valid_action_ids()
        self.assertGreater(checked, 50)  # the probe actually exercised leads


if __name__ == "__main__":
    unittest.main(verbosity=2)
