"""Hand-verified full games against fixed deals.

Every scenario here was computed by hand (trick winners, per-trick points,
final points and scores) and scripted move-by-move through the public action
API. If a refactor of ``sheepshead.py`` changes any trick winner, point total,
partner reveal, legality set, or final score in these games, that is a rules
regression — the expected values encode the rules of the game, not the current
implementation.

Scenarios:
  A  called-ace standard game (partner forced to play called card, bury rules,
     double-on-the-bump scoring at both settings)
  A' same deal played ALONE (defenders-win multiplier applies 4x to picker)
  B  under call (UNDER lead sets called suit, under points go to the trick,
     >60 picker win at x1)
  C  called-10 (10 takes the suit over the ace; schneider x2 win)
  D  leaster (blind to first-trick winner; zero-point seats without a trick
     don't qualify; winner +4)
  E  secret-partner restrictions (lead restriction, can't fail off the called
     card until trick 6, picker keeps last called-suit card at bury)
  JD JD-mode reveal on play, and buried-JD -> picker is own partner
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_test_utils import act, make_game, run_script, sole_actor, valid_action_names
from sheepshead import (
    ACTION_IDS,
    PARTNER_BY_JD,
    UNDER_TOKEN,
    get_trick_points,
)


def play_trick(game, plays):
    """Script one trick as [(pos, card), ...] in play order."""
    run_script(game, [(pos, f"PLAY {card}") for pos, card in plays])


# ---------------------------------------------------------------------------
# Scenario A: called-ace standard game
# ---------------------------------------------------------------------------
HANDS_A = [
    ["QC", "QS", "JD", "AD", "10D", "7C"],
    ["AC", "10C", "KC", "JC", "9S", "8S"],
    ["QH", "QD", "JS", "9D", "7S", "KS"],
    ["JH", "8D", "7D", "AS", "10S", "9C"],
    ["AH", "10H", "KH", "9H", "8H", "7H"],
]
BLIND_A = ["KD", "8C"]

TRICKS_A = [
    [(1, "8C"), (2, "AC"), (3, "QD"), (4, "9C"), (5, "7H")],  # P3 trumps, 14
    [(3, "KS"), (4, "AS"), (5, "8H"), (1, "QC"), (2, "9S")],  # P1 trumps, 18
    [(1, "QS"), (2, "JC"), (3, "9D"), (4, "JH"), (5, "9H")],  # P1, 7
    [(1, "AD"), (2, "10C"), (3, "QH"), (4, "8D"), (5, "10H")],  # P3 (QH>AD), 34
    [(3, "JS"), (4, "7D"), (5, "KH"), (1, "KD"), (2, "KC")],  # P3 (JS>KD), 14
    [(3, "7S"), (4, "10S"), (5, "AH"), (1, "JD"), (2, "8S")],  # P1 trumps, 23
]


def _play_scenario_a(double_on_the_bump=True, alone=False):
    game = make_game(HANDS_A, BLIND_A, double_on_the_bump=double_on_the_bump)
    act(game, 1, "PICK")
    if alone:
        act(game, 1, "ALONE")
    else:
        # P1 holds club fails but no AC; AC is the only callable card.
        assert valid_action_names(game, 1) == {"ALONE", "CALL AC"}
        act(game, 1, "CALL AC")
    act(game, 1, "BURY 7C")
    if not alone:
        # 8C is now P1's last club: it may not be buried (picker must keep a
        # card of the called suit).
        assert "BURY 8C" not in valid_action_names(game, 1)
    act(game, 1, "BURY 10D")
    for trick in TRICKS_A:
        play_trick(game, trick)
    assert game.is_done()
    return game


class TestScenarioACalledAce:
    def test_partner_forced_to_play_called_card(self):
        game = make_game(HANDS_A, BLIND_A)
        run_script(
            game,
            [(1, "PICK"), (1, "CALL AC"), (1, "BURY 7C"), (1, "BURY 10D")],
        )
        act(game, 1, "PLAY 8C")  # picker leads the called suit
        # P2 holds AC and clubs were led: the called card is forced.
        assert valid_action_names(game, 2) == {"PLAY AC"}
        act(game, 2, "PLAY AC")
        assert game.partner == 2  # revealed the moment it hits the table

    def test_trick_winners_points_and_flow(self):
        game = _play_scenario_a()
        assert game.trick_winners == [3, 1, 1, 3, 3, 1]
        assert game.trick_points == [14, 18, 7, 34, 14, 23]
        assert game.leaders == [1, 3, 1, 1, 3, 3]
        assert game.points_taken == [48, 0, 62, 0, 0]
        assert game.was_called_suit_played
        assert game.partner == 2

    def test_final_scoring_defenders_win_with_bump(self):
        game = _play_scenario_a(double_on_the_bump=True)
        # Picker side: 48 (P1) + 0 (P2) + 10 bury = 58; defenders 62.
        assert game.get_final_picker_points() == 58
        assert game.get_final_defender_points() == 62
        assert get_trick_points(game.bury) == 10
        assert [p.get_score() for p in game.players] == [-4, -2, 2, 2, 2]

    def test_final_scoring_without_bump(self):
        game = _play_scenario_a(double_on_the_bump=False)
        assert [p.get_score() for p in game.players] == [-2, -1, 1, 1, 1]

    def test_scores_zero_sum(self):
        for bump in (True, False):
            game = _play_scenario_a(double_on_the_bump=bump)
            assert sum(p.get_score() for p in game.players) == 0

    def test_alone_variant_quadruples_picker(self):
        game = _play_scenario_a(alone=True)
        assert game.alone_called
        assert game.partner == 1
        # Same plays are legal alone; P2's AC was a free choice this time.
        # Picker keeps only own points + bury: 48 + 10 = 58 -> defenders win.
        assert game.get_final_picker_points() == 58
        assert game.get_final_defender_points() == 62
        assert [p.get_score() for p in game.players] == [-8, 2, 2, 2, 2]


# ---------------------------------------------------------------------------
# Scenario B: under call
# ---------------------------------------------------------------------------
HANDS_B = [
    ["QC", "JC", "AD", "10D", "AC", "AS"],
    ["AH", "10H", "KH", "QS", "JS", "7C"],
    ["QH", "QD", "JH", "9H", "8H", "7H"],
    ["10C", "KC", "9C", "8C", "10S", "KS"],
    ["9S", "8S", "7S", "8D", "7D", "JD"],
]
BLIND_B = ["KD", "9D"]

TRICKS_B = [
    # P1 leads the face-down under; hearts are the called suit.
    [(2, "AH"), (3, "9H"), (4, "10S"), (5, "7S")],  # + UNDER lead; P2 wins 31
    [(2, "QS"), (3, "QH"), (4, "10C"), (5, "JD"), (1, "QC")],  # P1, 21
    [(1, "JC"), (2, "JS"), (3, "QD"), (4, "KC"), (5, "8D")],  # P3 (QD>JC), 11
    [(3, "JH"), (4, "9C"), (5, "7D"), (1, "AD"), (2, "10H")],  # P3 (JH>AD), 23
    [(3, "8H"), (4, "8C"), (5, "9S"), (1, "KD"), (2, "KH")],  # P1 trumps, 8
    [(1, "9D"), (2, "7C"), (3, "7H"), (4, "KS"), (5, "8S")],  # P1, 4
]


def _play_scenario_b():
    game = make_game(HANDS_B, BLIND_B)
    act(game, 1, "PICK")
    # P1 holds AC and AS and is void in hearts: only an AH under call exists.
    assert valid_action_names(game, 1) == {"ALONE", "CALL AH UNDER"}
    run_script(
        game,
        [
            (1, "CALL AH UNDER"),
            (1, "UNDER 10D"),
            (1, "BURY AC"),
            (1, "BURY AS"),
        ],
    )
    act(game, 1, f"PLAY {UNDER_TOKEN}")
    assert game.current_suit == "H"
    for trick in TRICKS_B:
        play_trick(game, trick)
    assert game.is_done()
    return game


class TestScenarioBUnderCall:
    def test_under_trick_carries_under_points(self):
        game = _play_scenario_b()
        # Trick 1 face cards: AH(11) + 10S(10) = 21, plus the hidden 10D = 31.
        assert game.trick_points[0] == 31
        assert game.trick_winners[0] == 2
        assert game.history[0][0] == UNDER_TOKEN
        assert game.partner == 2

    def test_full_game_accounting(self):
        game = _play_scenario_b()
        assert game.trick_winners == [2, 1, 3, 3, 1, 1]
        assert game.trick_points == [31, 21, 11, 23, 8, 4]
        assert game.points_taken == [33, 31, 34, 0, 0]
        # Picker side: 33 + 31 + bury (AC+AS = 22) = 86; defenders 34.
        assert game.get_final_picker_points() == 86
        assert game.get_final_defender_points() == 34

    def test_scores_x1_win(self):
        game = _play_scenario_b()
        assert [p.get_score() for p in game.players] == [2, 1, -1, -1, -1]
        assert sum(p.get_score() for p in game.players) == 0


# ---------------------------------------------------------------------------
# Scenario C: called 10
# ---------------------------------------------------------------------------
HANDS_C = [
    ["AC", "AS", "AH", "QC", "QS", "JD"],
    ["10H", "KH", "JC", "JS", "9C", "8C"],
    ["QH", "QD", "AD", "10D", "KD", "9H"],
    ["JH", "9D", "10C", "KC", "8H", "7H"],
    ["10S", "KS", "9S", "8S", "7S", "7C"],
]
BLIND_C = ["7D", "8D"]

TRICKS_C = [
    [(1, "AH"), (2, "10H"), (3, "9H"), (4, "8H"), (5, "7S")],  # called 10 wins, 21
    [(2, "KH"), (3, "AD"), (4, "7H"), (5, "7C"), (1, "8D")],  # P3 trumps, 15
    [(3, "QH"), (4, "JH"), (5, "8S"), (1, "QS"), (2, "JC")],  # P1 (QS>QH), 10
    [(1, "QC"), (2, "JS"), (3, "QD"), (4, "9D"), (5, "9S")],  # P1, 8
    [(1, "JD"), (2, "9C"), (3, "10D"), (4, "10C"), (5, "10S")],  # P1 (JD>10D), 32
    [(1, "7D"), (2, "8C"), (3, "KD"), (4, "KC"), (5, "KS")],  # P3 (KD>7D), 12
]


def _play_scenario_c():
    game = make_game(HANDS_C, BLIND_C)
    act(game, 1, "PICK")
    # All three fail aces held -> the callable cards are the three 10s.
    assert valid_action_names(game, 1) == {
        "ALONE",
        "CALL 10C",
        "CALL 10S",
        "CALL 10H",
    }
    act(game, 1, "CALL 10H")
    # AH is P1's only heart: it must be kept for the called suit.
    assert "BURY AH" not in valid_action_names(game, 1)
    run_script(game, [(1, "BURY AC"), (1, "BURY AS")])
    for trick in TRICKS_C:
        play_trick(game, trick)
    assert game.is_done()
    return game


class TestScenarioCCalled10:
    def test_called_10_beats_ace_on_first_called_suit_trick(self):
        game = _play_scenario_c()
        # Trick 1: AH led, 10H (called) takes the suit.
        assert game.trick_winners[0] == 2
        assert game.partner == 2

    def test_full_game_accounting(self):
        game = _play_scenario_c()
        assert game.trick_winners == [2, 3, 1, 1, 1, 3]
        assert game.trick_points == [21, 15, 10, 8, 32, 12]
        assert game.points_taken == [50, 21, 27, 0, 0]
        # Picker side: 50 + 21 + bury 22 = 93 > 90 -> schneider, x2.
        assert game.get_final_picker_points() == 93
        assert game.get_final_defender_points() == 27

    def test_scores_x2_win(self):
        game = _play_scenario_c()
        assert [p.get_score() for p in game.players] == [4, 2, -2, -2, -2]


# ---------------------------------------------------------------------------
# Scenario D: leaster
# ---------------------------------------------------------------------------
HANDS_D = [
    ["QC", "AC", "10C", "KC", "9C", "8C"],
    ["QS", "AS", "10S", "KS", "9S", "8S"],
    ["QH", "AH", "10H", "KH", "9H", "8H"],
    ["QD", "10D", "KD", "9D", "8D", "7D"],
    ["JC", "JS", "JH", "JD", "7C", "7S"],
]
BLIND_D = ["AD", "7H"]

TRICKS_D = [
    [(1, "AC"), (2, "8S"), (3, "8H"), (4, "7D"), (5, "7C")],  # P4 trumps; +blind
    [(4, "QD"), (5, "JD"), (1, "QC"), (2, "QS"), (3, "QH")],  # P1 (QC), 14
    [(1, "10C"), (2, "9S"), (3, "9H"), (4, "8D"), (5, "7S")],  # P4 trumps, 10
    [(4, "9D"), (5, "JH"), (1, "KC"), (2, "KS"), (3, "KH")],  # P5 (JH>9D), 14
    [(5, "JS"), (1, "9C"), (2, "10S"), (3, "10H"), (4, "KD")],  # P5, 26
    [(5, "JC"), (1, "8C"), (2, "AS"), (3, "AH"), (4, "10D")],  # P5, 34
]


def _play_scenario_d():
    game = make_game(HANDS_D, BLIND_D)
    for pos in range(1, 6):
        act(game, pos, "PASS")
    assert game.is_leaster
    for trick in TRICKS_D:
        play_trick(game, trick)
    assert game.is_done()
    return game


class TestScenarioDLeaster:
    def test_blind_points_go_to_first_trick_winner(self):
        game = _play_scenario_d()
        # Trick 1 cards are worth 11 (AC); the blind adds AD+7H = 11 more.
        assert game.trick_points[0] == 22
        assert game.trick_winners[0] == 4

    def test_accounting(self):
        game = _play_scenario_d()
        assert game.trick_winners == [4, 1, 4, 5, 5, 5]
        assert game.trick_points == [22, 14, 10, 14, 26, 34]
        assert game.points_taken == [14, 0, 0, 32, 74]
        assert sum(game.points_taken) == 120  # blind included via trick 1

    def test_winner_must_have_taken_a_trick(self):
        game = _play_scenario_d()
        # P2 and P3 have 0 points but no trick: they do not qualify. The
        # fewest-points trick-taker is P1 with 14.
        assert game.get_leaster_winner() == 1
        assert [p.get_score() for p in game.players] == [4, -1, -1, -1, -1]
        assert sum(p.get_score() for p in game.players) == 0


# ---------------------------------------------------------------------------
# Scenario E: secret-partner restrictions
# ---------------------------------------------------------------------------
HANDS_E = [
    ["QS", "JD", "AD", "10D", "KD", "7C"],
    ["QC", "AC", "10C", "KC", "9C", "8H"],
    ["QH", "QD", "JC", "9S", "8S", "7S"],
    ["JS", "JH", "7D", "AS", "10S", "KS"],
    ["8C", "AH", "10H", "KH", "9H", "7H"],
]
BLIND_E = ["9D", "8D"]


def _play_scenario_e():
    game = make_game(HANDS_E, BLIND_E)
    run_script(game, [(1, "PICK"), (1, "CALL AC")])
    # 7C is P1's only club: it cannot be buried.
    assert "BURY 7C" not in valid_action_names(game, 1)
    run_script(game, [(1, "BURY 9D"), (1, "BURY 8D")])

    # Trick 1: P2 (secret partner) wins with QC while AC stays hidden.
    play_trick(game, [(1, "QS")])
    assert valid_action_names(game, 2) == {"PLAY QC"}  # only trump held
    play_trick(game, [(2, "QC"), (3, "JC"), (4, "7D"), (5, "7H")])
    assert game.trick_winners[0] == 2
    assert game.partner == 0  # still secret

    # Trick 2: secret partner leads. Only the called card itself or
    # off-called-suit cards are leadable — not the other clubs.
    assert valid_action_names(game, 2) == {"PLAY AC", "PLAY 8H"}
    play_trick(game, [(2, "8H"), (3, "8S"), (4, "AS"), (5, "AH"), (1, "JD")])
    assert game.trick_winners[1] == 1

    # Trick 3: trump led, P2 is void — may not fail off the called card.
    play_trick(game, [(1, "AD")])
    assert valid_action_names(game, 2) == {"PLAY 10C", "PLAY KC", "PLAY 9C"}
    play_trick(game, [(2, "10C"), (3, "QD"), (4, "JH"), (5, "9H")])
    assert game.trick_winners[2] == 3

    # Trick 4: spades led; the picker is void and holds KD, 10D, 7C — but 7C
    # is the last club and the called suit is still unplayed, so the picker
    # may not slough it.
    play_trick(game, [(3, "9S"), (4, "10S"), (5, "KH")])
    assert valid_action_names(game, 1) == {"PLAY KD", "PLAY 10D"}
    play_trick(game, [(1, "10D")])
    # P2 (secret partner) likewise may not slough the called card.
    assert valid_action_names(game, 2) == {"PLAY KC", "PLAY 9C"}
    play_trick(game, [(2, "KC")])
    assert game.trick_winners[3] == 1

    # Trick 5: trump led; with only AC + 9C left and unable to follow, the
    # restriction forces P2's single non-called card.
    play_trick(game, [(1, "KD")])
    assert valid_action_names(game, 2) == {"PLAY 9C"}
    play_trick(game, [(2, "9C"), (3, "QH"), (4, "JS"), (5, "10H")])
    assert game.trick_winners[4] == 3

    # Trick 6: last trick — the called card may finally be failed off, and the
    # picker may finally part with the last club.
    play_trick(game, [(3, "7S"), (4, "KS"), (5, "8C")])
    assert valid_action_names(game, 1) == {"PLAY 7C"}
    play_trick(game, [(1, "7C")])
    assert game.partner == 0
    assert valid_action_names(game, 2) == {"PLAY AC"}
    play_trick(game, [(2, "AC")])
    assert game.partner == 2  # revealed on the very last card
    assert game.is_done()
    return game


class TestScenarioESecretPartner:
    def test_restrictions_and_late_reveal(self):
        game = _play_scenario_e()
        # Clubs were never led, so the called suit never completed a trick.
        assert not game.was_called_suit_played
        assert game.trick_winners == [2, 1, 3, 1, 3, 4]
        assert game.trick_points == [8, 24, 26, 28, 19, 15]
        assert game.points_taken == [52, 8, 45, 15, 0]
        # Picker side: 52 + 8 + bury 0 = 60 exactly; defenders 60. Ties go to
        # the defenders (>= 60), and the loss is bumped x2.
        assert game.get_final_picker_points() == 60
        assert game.get_final_defender_points() == 60
        assert [p.get_score() for p in game.players] == [-4, -2, 2, 2, 2]


# ---------------------------------------------------------------------------
# JD mode: reveal on play, and buried JD
# ---------------------------------------------------------------------------
HANDS_JD = [
    ["QC", "QS", "QH", "QD", "7C", "7S"],
    ["AC", "10C", "KC", "9C", "8C", "AH"],
    ["JD", "JC", "JS", "JH", "10H", "KH"],
    ["AD", "10D", "KD", "9D", "8D", "7D"],
    ["AS", "10S", "KS", "9S", "8S", "9H"],
]
BLIND_JD = ["8H", "7H"]


def _min_action(player):
    return min(player.get_valid_action_ids())


def _finish_min_id(game, on_action=None):
    """Drive the game to completion, always taking the lowest legal action id.
    Deterministic given the fixed deal. Calls on_action(pos, action_id) after
    each action if provided."""
    while not game.is_done():
        pos = sole_actor(game)
        player = game.players[pos - 1]
        a = _min_action(player)
        assert player.act(a)
        if on_action:
            on_action(pos, a)


class TestJDMode:
    def test_partner_revealed_when_jd_played(self):
        game = make_game(HANDS_JD, BLIND_JD, partner_selection_mode=PARTNER_BY_JD)
        run_script(game, [(1, "PICK"), (1, "JD PARTNER")])
        run_script(game, [(1, "BURY 7C"), (1, "BURY 7S")])
        assert game.partner == 0
        assert game.players[2].is_secret_partner  # P3 holds JD

        reveal = {}

        def on_action(pos, a):
            if game.partner and "at" not in reveal:
                reveal["at"] = (pos, a)

        _finish_min_id(game, on_action)
        assert game.partner == 3
        # The reveal happened exactly on P3's JD play.
        pos, a = reveal["at"]
        assert pos == 3
        assert ACTION_IDS["PLAY JD"] == a
        # Standard sanity on the finished game.
        assert sum(game.points_taken) + get_trick_points(game.bury) == 120
        assert sum(p.get_score() for p in game.players) == 0

    def test_buried_jd_makes_picker_own_partner(self):
        # Give P1 the JD via the blind and bury it.
        hands = [
            ["QC", "QS", "QH", "QD", "7C", "7S"],
            ["AC", "10C", "KC", "9C", "8C", "AH"],
            ["JC", "JS", "JH", "10H", "KH", "9H"],
            ["AD", "10D", "KD", "9D", "8D", "7D"],
            ["AS", "10S", "KS", "9S", "8S", "8H"],
        ]
        blind = ["JD", "7H"]
        game = make_game(hands, blind, partner_selection_mode=PARTNER_BY_JD)
        run_script(game, [(1, "PICK"), (1, "JD PARTNER")])
        run_script(game, [(1, "BURY JD"), (1, "BURY 7H")])
        assert game.partner == 0
        _finish_min_id(game)
        # Nobody could play JD; at the final card the picker becomes partner.
        assert game.partner == 1
        picker_score = game.players[0].get_score()
        others = [p.get_score() for p in game.players[1:]]
        # Picker-as-own-partner earns the 4x role multiplier against four
        # defenders who each score the opposite sign.
        assert picker_score == -4 * others[0]
        assert sum([picker_score] + others) == 0
        assert len({*others}) == 1  # all four defenders score alike


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
