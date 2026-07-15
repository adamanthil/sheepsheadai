"""Game-engine rules & contracts: frozen encodings, pure helpers, bidding flow,
and rare rule branches.

These are the regression guards for refactors of ``sheepshead.py``:

* The action space and card-id encodings are FROZEN CONTRACTS — trained model
  checkpoints bake in these exact sizes and orderings. If a test here fails
  because you reordered/extended ACTIONS or DECK, that is a breaking change for
  every saved policy, not a test to update casually.
* Pure-helper tests pin the card-power/point tables and the called-ace family
  of hand filters.
* Bidding-flow tests pin turn order and state transitions (pick, pass, leaster
  entry, partner selection, bury gating).
* Branch tests cover the under-call play rules that random playouts rarely hit.
"""

import sys

import numpy as np

from sheepshead.tests.game_test_utils import act, make_game, run_script, valid_action_names
from sheepshead import (
    ACTION_IDS,
    ACTION_LOOKUP,
    ACTIONS,
    CALLED_10S,
    CALLED_ACES,
    CALLED_PARTNER_CARDS,
    DECK,
    DECK_IDS,
    FAIL,
    PARTNER_BY_JD,
    TRUMP,
    UNDER_CARD_ID,
    UNDER_TOKEN,
    filter_by_suit,
    get_callable_cards,
    get_card_points,
    get_card_suit,
    get_leadable_called_partner_cards,
    get_playable_called_picker_cards,
    get_trick_points,
    get_trick_winner,
)


# ---------------------------------------------------------------------------
# Frozen encoding contracts (model I/O depends on these)
# ---------------------------------------------------------------------------
class TestFrozenContracts:
    def test_deck_composition(self):
        assert len(TRUMP) == 14
        assert len(FAIL) == 18
        assert DECK == TRUMP + FAIL
        assert len(set(DECK)) == 32
        assert sum(get_card_points(c) for c in DECK) == 120

    def test_deck_order_and_ids(self):
        # Trump strength order QC..7D, then fail suits C, S, H each A..7.
        assert TRUMP[0] == "QC" and TRUMP[-1] == "7D"
        assert FAIL[0] == "AC" and FAIL[-1] == "7H"
        assert DECK_IDS["QC"] == 1
        assert DECK_IDS["7D"] == 14
        assert DECK_IDS["AC"] == 15
        assert DECK_IDS["7H"] == 32
        assert sorted(DECK_IDS.values()) == list(range(1, 33))
        assert UNDER_CARD_ID == 33

    def test_action_space_frozen(self):
        # 4 fixed + 9 calls + 32 UNDER + 32 BURY + 33 PLAY (incl. PLAY UNDER)
        assert len(ACTIONS) == 110
        assert ACTIONS[:4] == ["PICK", "PASS", "ALONE", "JD PARTNER"]
        assert ACTIONS[4:10] == [f"CALL {c}" for c in CALLED_PARTNER_CARDS]
        assert ACTIONS[10:13] == [f"CALL {c} UNDER" for c in ["AC", "AS", "AH"]]
        assert ACTIONS[13:45] == [f"UNDER {c}" for c in DECK]
        assert ACTIONS[45:77] == [f"BURY {c}" for c in DECK]
        assert ACTIONS[77:109] == [f"PLAY {c}" for c in DECK]
        assert ACTIONS[109] == f"PLAY {UNDER_TOKEN}"

    def test_action_id_bijection(self):
        assert ACTION_IDS["PICK"] == 1
        assert ACTION_IDS["PASS"] == 2
        assert ACTION_IDS[f"PLAY {UNDER_TOKEN}"] == 110
        for i, a in enumerate(ACTIONS, start=1):
            assert ACTION_IDS[a] == i
            assert ACTION_LOOKUP[i] == a

    def test_called_partner_card_lists(self):
        assert CALLED_ACES == ["AC", "AS", "AH"]
        assert CALLED_10S == ["10C", "10S", "10H"]


# ---------------------------------------------------------------------------
# Pure helpers: suits, points, powers
# ---------------------------------------------------------------------------
class TestCardHelpers:
    def test_trump_cards_have_suit_t(self):
        for c in TRUMP:
            assert get_card_suit(c) == "T"

    def test_fail_cards_keep_letter_suit(self):
        assert get_card_suit("AC") == "C"
        assert get_card_suit("10S") == "S"
        assert get_card_suit("7H") == "H"
        # QD/JD/diamonds are trump, never "D"
        assert get_card_suit("AD") == "T"

    def test_card_points_table(self):
        assert get_card_points("AC") == 11
        assert get_card_points("10D") == 10
        assert get_card_points("KS") == 4
        assert get_card_points("QC") == 3
        assert get_card_points("JD") == 2
        for c in ["9C", "8S", "7H", "9D"]:
            assert get_card_points(c) == 0
        # Non-deck tokens fall back to 0 points
        assert get_card_points(UNDER_TOKEN) == 0
        assert get_card_points("") == 0

    def test_filter_by_suit(self):
        hand = ["QC", "JD", "AC", "7C", "10H", "8S"]
        assert filter_by_suit(hand, "T") == ["QC", "JD"]
        assert filter_by_suit(hand, "C") == ["AC", "7C"]
        assert filter_by_suit(hand, "H") == ["10H"]
        assert filter_by_suit(hand, "D") == []

    def test_trick_points(self):
        assert get_trick_points(["AC", "10S", "KH", "QD", "JC"]) == 30
        assert get_trick_points(["7C", "8S", "9H", "7D", "8D"]) == 0


class TestTrickWinner:
    def test_highest_trump_wins_regardless_of_position(self):
        # Suit led is hearts; QC (highest trump) played 4th.
        assert get_trick_winner(["AH", "JD", "7D", "QC", "QS"], "H") == 4

    def test_trump_beats_any_fail(self):
        assert get_trick_winner(["AC", "10C", "KC", "7D", "9C"], "C") == 4

    def test_led_fail_suit_beats_offsuit_fail(self):
        # 8H follows the heart lead; AS/AC are worth more but off-suit.
        assert get_trick_winner(["7H", "AS", "AC", "KS", "8H"], "H") == 5

    def test_led_ace_wins_without_trump(self):
        assert get_trick_winner(["AH", "10H", "KH", "9H", "8H"], "H") == 1

    def test_leader_wins_when_everyone_sloughs(self):
        assert get_trick_winner(["7S", "AH", "10H", "KC", "9H"], "S") == 1

    def test_trump_order_within_trump_lead(self):
        # QC > QS > ... > JD > AD > 10D > KD > 9D..7D
        assert get_trick_winner(["JD", "AD", "10D", "KD", "QD"], "T") == 5
        assert get_trick_winner(["AD", "JC", "9D", "8D", "7D"], "T") == 2

    def test_called_10_takes_suit_over_ace(self):
        trick = ["AH", "10H", "9H", "8H", "7H"]
        assert get_trick_winner(trick, "H", is_called_10_suit=True) == 2
        # Without the called-10 rule the ace wins.
        assert get_trick_winner(trick, "H", is_called_10_suit=False) == 1

    def test_trump_still_beats_called_10(self):
        assert get_trick_winner(["AH", "10H", "7D", "8H", "9H"], "H", True) == 3


# ---------------------------------------------------------------------------
# Called-ace hand filters
# ---------------------------------------------------------------------------
class TestCallableCards:
    def test_ace_callable_when_holding_fail_in_suit(self):
        hand = ["QC", "JD", "AD", "10D", "7C", "KD", "9D", "8D"]
        assert get_callable_cards(hand) == ["AC"]

    def test_held_ace_not_callable(self):
        hand = ["AC", "7C", "QC", "JD", "AD", "10D", "KD", "9D"]
        # AC is held; no spade/heart fails either -> under calls for S and H.
        assert get_callable_cards(hand) == ["AS UNDER", "AH UNDER"]

    def test_multiple_aces_callable(self):
        hand = ["7C", "7S", "7H", "QC", "QS", "QH", "QD", "JC"]
        assert get_callable_cards(hand) == ["AC", "AS", "AH"]

    def test_regular_call_preferred_over_under(self):
        # Club fail present (AC callable); void in spades/hearts, but the
        # regular call suppresses the under options entirely.
        hand = ["7C", "QC", "QS", "QH", "QD", "JC", "JS", "JH"]
        assert get_callable_cards(hand) == ["AC"]

    def test_all_aces_held_falls_back_to_10s(self):
        hand = ["AC", "AS", "AH", "QC", "QS", "JD", "7D", "8D"]
        assert get_callable_cards(hand) == CALLED_10S

    def test_pure_trump_hand_gets_all_unders(self):
        hand = ["QC", "QS", "QH", "QD", "JC", "JS", "JH", "JD"]
        assert get_callable_cards(hand) == ["AC UNDER", "AS UNDER", "AH UNDER"]


class TestCalledPartnerFilters:
    def test_partner_lead_restricted_to_called_card_or_offsuit(self):
        hand = ["AC", "10C", "7C", "QD", "10H", "8S"]
        leadable = get_leadable_called_partner_cards(hand, "AC")
        assert sorted(leadable) == sorted(["AC", "QD", "10H", "8S"])

    def test_picker_cannot_bury_called_card(self):
        hand = ["10H", "7H", "QD", "JC", "AD", "KD"]
        playable = get_playable_called_picker_cards(hand, "10H")
        assert "10H" not in playable
        assert "7H" in playable

    def test_picker_keeps_last_called_suit_card(self):
        hand = ["7C", "QD", "JC", "AD", "KD", "9D"]
        playable = get_playable_called_picker_cards(hand, "AC")
        assert "7C" not in playable
        assert sorted(playable) == sorted(["QD", "JC", "AD", "KD", "9D"])

    def test_picker_with_two_called_suit_cards_may_shed_one(self):
        hand = ["7C", "8C", "QD", "JC", "AD", "KD"]
        playable = get_playable_called_picker_cards(hand, "AC")
        assert "7C" in playable and "8C" in playable


# ---------------------------------------------------------------------------
# Bidding flow & turn order
# ---------------------------------------------------------------------------
HANDS_BASIC = [
    ["QC", "QS", "JD", "AD", "10D", "7C"],
    ["AC", "10C", "KC", "JC", "9S", "8S"],
    ["QH", "QD", "JS", "9D", "7S", "KS"],
    ["JH", "8D", "7D", "AS", "10S", "9C"],
    ["AH", "10H", "KH", "9H", "8H", "7H"],
]
BLIND_BASIC = ["KD", "8C"]


class TestBiddingFlow:
    def test_only_seat_one_may_open(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        assert valid_action_names(game, 1) == {"PICK", "PASS"}
        for pos in (2, 3, 4, 5):
            assert valid_action_names(game, pos) == set()

    def test_acting_out_of_turn_is_rejected_without_mutation(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        p2 = game.players[1]
        before = list(p2.hand)
        assert p2.act(ACTION_IDS["PICK"]) is False
        assert p2.hand == before
        assert game.picker == 0

    def test_pass_advances_turn(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        act(game, 1, "PASS")
        assert valid_action_names(game, 1) == set()
        assert valid_action_names(game, 2) == {"PICK", "PASS"}

    def test_pick_takes_blind_and_moves_to_partner_selection(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        act(game, 1, "PASS")
        act(game, 2, "PICK")
        p2 = game.players[1]
        assert game.picker == 2
        assert len(p2.hand) == 8
        assert set(BLIND_BASIC) <= set(p2.hand)
        # P2 holds AC and has spade fails -> only AS is callable.
        assert valid_action_names(game, 2) == {"ALONE", "CALL AS"}
        # Everyone else is locked out until the bury is complete.
        for pos in (1, 3, 4, 5):
            assert valid_action_names(game, pos) == set()

    def test_no_play_actions_until_bury_complete(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        run_script(game, [(1, "PICK"), (1, "CALL AC")])
        assert not game.play_started
        assert all(a.startswith("BURY ") for a in valid_action_names(game, 1))
        act(game, 1, "BURY 10D")
        assert not game.play_started
        act(game, 1, "BURY 7C")
        assert game.play_started
        assert game.leader == 1
        assert game.leaders[0] == 1
        assert len(game.players[0].hand) == 6

    def test_all_pass_enters_leaster(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        for pos in range(1, 5):
            act(game, pos, "PASS")
            assert not game.is_leaster
        act(game, 5, "PASS")
        assert game.is_leaster
        assert game.play_started
        assert game.leader == 1
        assert game.picker == 0
        # Leaster: seat 1 leads, anything is playable.
        assert valid_action_names(game, 1) == {f"PLAY {c}" for c in HANDS_BASIC[0]}

    def test_alone_sets_partner_to_picker(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        run_script(game, [(1, "PICK"), (1, "ALONE")])
        assert game.alone_called
        assert game.partner == 1
        assert game.called_card is None
        # No called card -> bury unrestricted over the 8-card hand.
        assert valid_action_names(game, 1) == {
            f"BURY {c}" for c in game.players[0].hand
        }

    def test_jd_mode_partner_choice(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC, partner_selection_mode=PARTNER_BY_JD)
        act(game, 1, "PICK")
        assert valid_action_names(game, 1) == {"ALONE", "JD PARTNER"}
        act(game, 1, "JD PARTNER")
        assert game.partner == 0  # not revealed until JD is played
        assert valid_action_names(game, 1) == {
            f"BURY {c}" for c in game.players[0].hand
        }

    def test_jd_mode_secret_partner_property(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC, partner_selection_mode=PARTNER_BY_JD)
        run_script(game, [(1, "PASS"), (2, "PICK"), (2, "JD PARTNER")])
        # P1 holds JD in this deal.
        assert game.players[0].is_secret_partner
        assert not game.players[2].is_secret_partner


# ---------------------------------------------------------------------------
# Under-call play branches (rarely hit by random play)
# ---------------------------------------------------------------------------
HANDS_UNDER = [
    ["QC", "JC", "AD", "10D", "AC", "AS"],
    ["AH", "10H", "KH", "QS", "JS", "7C"],
    ["QH", "QD", "JH", "9H", "8H", "7H"],
    ["10C", "KC", "9C", "8C", "10S", "KS"],
    ["9S", "8S", "7S", "8D", "7D", "JD"],
]
BLIND_UNDER = ["KD", "9D"]


def _under_game_after_bury():
    """P1 picks, is void in hearts holding AC+AS, calls AH UNDER, places 10D
    under, buries the two black aces. Hand afterwards: QC JC AD 9D KD."""
    game = make_game(HANDS_UNDER, BLIND_UNDER)
    act(game, 1, "PICK")
    assert valid_action_names(game, 1) == {"ALONE", "CALL AH UNDER"}
    act(game, 1, "CALL AH UNDER")
    assert game.is_called_under
    assert game.called_card == "AH"
    # Must place the under before burying; any of the 8 cards is eligible.
    assert valid_action_names(game, 1) == {f"UNDER {c}" for c in game.players[0].hand}
    act(game, 1, "UNDER 10D")
    assert game.under_card == "10D"
    run_script(game, [(1, "BURY AC"), (1, "BURY AS")])
    return game


class TestUnderCallBranches:
    def test_under_setup_and_lead_option(self):
        game = _under_game_after_bury()
        assert sorted(game.players[0].hand) == sorted(["QC", "JC", "AD", "9D", "KD"])
        # Leading, the picker may open the called suit with the face-down under.
        assert f"PLAY {UNDER_TOKEN}" in valid_action_names(game, 1)

    def test_playing_under_sets_called_suit_and_forces_called_card(self):
        game = _under_game_after_bury()
        act(game, 1, f"PLAY {UNDER_TOKEN}")
        assert game.current_suit == "H"
        assert game.history[0][0] == UNDER_TOKEN
        # Hand unchanged: the under card left the hand at the UNDER step.
        assert len(game.players[0].hand) == 5
        # P2 holds AH and hearts were led -> must play the called card.
        assert valid_action_names(game, 2) == {"PLAY AH"}

    def test_under_is_only_legal_play_when_void_on_called_suit_lead(self):
        game = _under_game_after_bury()
        # White-box: put P1 mid-trick on a heart lead (picker is void).
        game.current_suit = "H"
        game.leader = 2
        game.last_player = 0
        assert valid_action_names(game, 1) == {f"PLAY {UNDER_TOKEN}"}

    def test_under_not_playable_offsuit_before_last_trick(self):
        game = _under_game_after_bury()
        game.current_suit = "S"
        game.leader = 2
        game.last_player = 0
        game.current_trick = 4
        names = valid_action_names(game, 1)
        assert f"PLAY {UNDER_TOKEN}" not in names
        assert names == {f"PLAY {c}" for c in game.players[0].hand}

    def test_under_playable_offsuit_on_last_trick(self):
        game = _under_game_after_bury()
        game.current_suit = "S"
        game.leader = 2
        game.last_player = 0
        game.current_trick = 5
        names = valid_action_names(game, 1)
        assert f"PLAY {UNDER_TOKEN}" in names
        assert names == {f"PLAY {c}" for c in game.players[0].hand} | {
            f"PLAY {UNDER_TOKEN}"
        }

    def test_under_lead_unavailable_after_called_suit_played(self):
        game = _under_game_after_bury()
        act(game, 1, f"PLAY {UNDER_TOKEN}")
        run_script(
            game, [(2, "PLAY AH"), (3, "PLAY 9H"), (4, "PLAY 10S"), (5, "PLAY 7S")]
        )
        assert game.was_called_suit_played
        # P2 won the trick and leads; if it were P1, UNDER would be gone too.
        game.leader = 1
        game.last_player = 0
        assert f"PLAY {UNDER_TOKEN}" not in valid_action_names(game, 1)


# ---------------------------------------------------------------------------
# Observation dict contract
# ---------------------------------------------------------------------------
class TestStateDictContract:
    EXPECTED_KEYS = {
        "partner_mode",
        "is_leaster",
        "play_started",
        "current_trick",
        "alone_called",
        "called_card_id",
        "called_under",
        "picker_rel",
        "partner_rel",
        "leader_rel",
        "picker_position",
        "hand_ids",
        "blind_ids",
        "bury_ids",
        "trick_card_ids",
        "trick_is_picker",
        "trick_is_partner_known",
    }

    def _game_mid_trick(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        run_script(
            game,
            [
                (1, "PICK"),
                (1, "CALL AC"),
                (1, "BURY 7C"),
                (1, "BURY 10D"),
                (1, "PLAY 8C"),
                (2, "PLAY AC"),
            ],
        )
        return game

    def test_keys_shapes_dtypes(self):
        game = self._game_mid_trick()
        for player in game.players:
            obs = player.get_state_dict()
            assert set(obs.keys()) == self.EXPECTED_KEYS
            for k in ("hand_ids",):
                assert obs[k].shape == (8,) and obs[k].dtype == np.uint8
            for k in ("blind_ids", "bury_ids"):
                assert obs[k].shape == (2,) and obs[k].dtype == np.uint8
            for k in ("trick_card_ids", "trick_is_picker", "trick_is_partner_known"):
                assert obs[k].shape == (5,) and obs[k].dtype == np.uint8

    def test_relative_seats_and_privacy(self):
        game = self._game_mid_trick()
        obs5 = game.players[4].get_state_dict()
        # Picker is seat 1; from seat 5 that is one to the left -> rel 2.
        assert obs5["picker_rel"] == 2
        assert obs5["partner_rel"] == 3  # partner (seat 2) is +2 from seat 5
        assert obs5["picker_position"] == 1
        # Non-picker never sees blind/bury.
        assert not obs5["blind_ids"].any()
        assert not obs5["bury_ids"].any()
        obs1 = game.players[0].get_state_dict()
        assert obs1["picker_rel"] == 1
        assert sorted(obs1["bury_ids"].tolist()) == sorted(
            [DECK_IDS["7C"], DECK_IDS["10D"]]
        )
        assert sorted(obs1["blind_ids"].tolist()) == sorted(
            [DECK_IDS["KD"], DECK_IDS["8C"]]
        )

    def test_hand_and_trick_ids(self):
        game = self._game_mid_trick()
        p3 = game.players[2]
        obs = p3.get_state_dict()
        assert sorted(obs["hand_ids"].tolist()) == sorted(
            [DECK_IDS[c] for c in p3.hand] + [0, 0]
        )
        # Trick from seat 3's view: rel 1 = self (empty), rel 4 = seat 1 (8C),
        # rel 5 = seat 2 (AC).
        trick = obs["trick_card_ids"].tolist()
        assert trick[0] == 0
        assert trick[3] == DECK_IDS["8C"]
        assert trick[4] == DECK_IDS["AC"]
        assert obs["trick_is_picker"].tolist() == [0, 0, 0, 1, 0]
        assert obs["trick_is_partner_known"].tolist() == [0, 0, 0, 0, 1]

    def test_under_token_in_trick_ids(self):
        game = _under_game_after_bury()
        act(game, 1, f"PLAY {UNDER_TOKEN}")
        obs = game.players[2].get_state_dict()
        assert UNDER_CARD_ID in obs["trick_card_ids"].tolist()
        assert obs["called_under"] == 1
        assert obs["called_card_id"] == DECK_IDS["AH"]

    def test_oracle_dict_is_superset_with_hidden_state(self):
        game = self._game_mid_trick()
        p5 = game.players[4]
        obs = p5.get_oracle_state_dict()
        assert self.EXPECTED_KEYS < set(obs.keys())
        assert obs["opp_hand_ids"].shape == (4, 8)
        # Oracle sees the true blind/bury even for a non-picker.
        assert sorted(obs["bury_ids"].tolist()) == sorted(
            [DECK_IDS["7C"], DECK_IDS["10D"]]
        )
        # Partner already revealed (AC played); nobody still holds the called
        # card, so no seat is a secret partner.
        assert obs["secret_partner_rel"] == 0
        # Opponent hands, relative order 2..5 -> seats 1,2,3,4 from seat 5.
        seat1_hand = game.players[0].hand  # 5 cards left mid-trick
        expected_seat1 = sorted(
            [DECK_IDS[c] for c in seat1_hand] + [0] * (8 - len(seat1_hand))
        )
        assert sorted(obs["opp_hand_ids"][0].tolist()) == expected_seat1

    def test_oracle_secret_partner_before_reveal(self):
        game = make_game(HANDS_BASIC, BLIND_BASIC)
        run_script(game, [(1, "PICK"), (1, "CALL AC"), (1, "BURY 7C"), (1, "BURY 10D")])
        obs = game.players[4].get_state_dict()
        assert obs["partner_rel"] == 0
        oracle = game.players[4].get_oracle_state_dict()
        # Seat 2 holds AC: from seat 5, rel = 3.
        assert oracle["secret_partner_rel"] == 3


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
