#!/usr/bin/env python3
"""ScriptedAgent invariants: legality, determinism, zero-sum self-play."""

import numpy as np

from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game
from sheepshead.scripted_agent import ScriptedAgent


class TestScriptedAgent:
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
                        assert a in valid
                        a2, _, _ = ag.act(
                            player.get_state_dict(), valid, player.position
                        )
                        assert a == a2
                        assert (logp, val) == (0.0, 0.0)
                        player.act(a)
                        valid = player.get_valid_action_ids()
            scores = [p.get_score() for p in game.players]
            assert sum(scores) == 0, f"game {g}: {scores}"

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
                                assert card_played not in trump_set, (
                                    f"game {g}: led trump"
                                )
                                checked += 1
                        player.act(a)
                        valid = player.get_valid_action_ids()
        assert checked > 50  # the probe actually exercised leads


class TestTeamInference:
    def _lead_state(self, partner_mode, alone, hand_ids):
        import numpy as np

        return {
            "partner_mode": np.uint8(partner_mode),
            "is_leaster": np.uint8(0),
            "play_started": np.uint8(1),
            "current_trick": np.uint8(1),
            "alone_called": np.uint8(alone),
            "called_card_id": np.uint8(0),
            "called_under": np.uint8(0),
            "called_suit_played": np.uint8(0),
            "picker_rel": np.uint8(3),
            "partner_rel": np.uint8(0),
            "leader_rel": np.uint8(1),
            "picker_position": np.uint8(3),
            "hand_ids": np.array(hand_ids + [0] * (8 - len(hand_ids)), dtype=np.uint8),
            "blind_ids": np.zeros(2, dtype=np.uint8),
            "bury_ids": np.zeros(2, dtype=np.uint8),
            "trick_card_ids": np.zeros(5, dtype=np.uint8),
            "trick_is_picker": np.zeros(5, dtype=np.uint8),
            "trick_is_partner_known": np.zeros(5, dtype=np.uint8),
        }

    def test_defender_leads_called_suit_any_trick_until_played(self):
        # C2 across tricks: a defender holding a called-suit fail leads it
        # whenever the called suit has not yet been led — not just trick 0 —
        # and reverts to normal fail leads once it has been played.
        from sheepshead import DECK_IDS

        ag = ScriptedAgent()
        hand = [DECK_IDS["9H"], DECK_IDS["AS"], DECK_IDS["7C"]]
        state = self._lead_state(1, alone=0, hand_ids=hand)
        state["called_card_id"] = np.uint8(DECK_IDS["AH"])  # hearts called
        state["current_trick"] = np.uint8(2)
        cards = ["9H", "AS", "7C"]
        assert ag._lead(state, cards) == "9H"  # through, even at trick 2
        state["called_suit_played"] = np.uint8(1)
        assert ag._lead(state, cards) == "AS"  # back to cashing fail aces

    def test_jd_holder_is_defender_when_picker_goes_alone(self):
        # JD-mode: holding the JD marks the secret partner — but not when
        # ALONE was declared; then the JD holder is an ordinary defender and
        # must not lead trump (the exact tell the conventions forbid).
        from sheepshead import DECK_IDS

        ag = ScriptedAgent()
        jd_hand = [DECK_IDS["JD"], DECK_IDS["7C"], DECK_IDS["8S"]]
        state_partner = self._lead_state(0, alone=0, hand_ids=jd_hand)
        state_defender = self._lead_state(0, alone=1, hand_ids=jd_hand)
        assert ag._same_team(state_partner, 0, leading=True)
        assert not ag._same_team(state_defender, 0, leading=True)
        assert ag._lead(state_partner, ["JD", "7C", "8S"]) == "JD"
        assert ag._lead(state_defender, ["JD", "7C", "8S"]) != "JD"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
