#!/usr/bin/env python3
"""Leaster scoring invariants.

Regression for a tie-break bug found during the July 2026 run review
(League_Run_Review_202607.md follow-up): ``Game.get_leaster_winner`` drew a
fresh ``rng.choice`` on every call for tied leasters, so each seat's
``get_score()`` could crown a different winner — hands scored two +4s (or
none) and stopped summing to zero, quietly contaminating every score-based
measurement (rigorous_eval, paired gates, terminal rewards) on ~tied
leasters. The winner is now drawn once and cached.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead import ACTION_IDS, Game

# Seeds whose forced-pass games (everyone passes, then plays the lowest legal
# action id) end in a leaster with MULTIPLE tied minimum-point qualifiers.
TIE_SEEDS = [53, 66, 137, 166, 192]


def _play_forced_pass_game(seed: int) -> Game:
    game = Game(partner_selection_mode=1, seed=seed)
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                a = (
                    ACTION_IDS["PASS"]
                    if ACTION_IDS["PASS"] in valid
                    else sorted(valid)[0]
                )
                player.act(a)
                valid = player.get_valid_action_ids()
    return game


class TestLeasterTieScoring(unittest.TestCase):
    def test_tied_leaster_has_one_stable_winner_and_zero_sum(self):
        for seed in TIE_SEEDS:
            game = _play_forced_pass_game(seed)
            self.assertTrue(game.is_leaster)

            # The winner must be stable across calls (each seat's get_score,
            # __str__, and any re-query must agree).
            winner = game.get_leaster_winner()
            for _ in range(25):
                self.assertEqual(game.get_leaster_winner(), winner)

            scores = [p.get_score() for p in game.players]
            self.assertEqual(sum(scores), 0, f"seed {seed}: {scores}")
            self.assertEqual(sorted(scores), [-1, -1, -1, -1, 4])
            self.assertEqual(scores.index(4) + 1, winner)

    def test_untied_leasters_also_zero_sum(self):
        # A broad sweep: every forced-pass leaster must produce exactly one
        # winner and zero-sum scores, tie or not.
        for seed in range(60):
            game = _play_forced_pass_game(seed)
            scores = [p.get_score() for p in game.players]
            self.assertEqual(sum(scores), 0, f"seed {seed}: {scores}")
            self.assertEqual(sorted(scores), [-1, -1, -1, -1, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
