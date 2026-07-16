"""Characterization tests for the reward-shaping math in training_utils.py.

These pin the CURRENT numeric behavior of the shaping/reward functions used
by the trainers, computed against fully deterministic games (built the same
way as sheepshead/tests/test_game_scenarios.py, via ``make_game``). They are
not a statement that the shaping is "correct" Sheepshead strategy advice --
several of the constants here are ad-hoc human-tuned nudges. If a refactor of
training_utils.py changes one of these values, that's a behavior change to
review, not necessarily a bug.

Scenario A (called-ace) and Scenario D (leaster) reuse the hand-verified
deals from test_game_scenarios.py so trick winners/points are already
independently checked there.
"""

import sys

from sheepshead.tests.game_test_utils import act, make_game, valid_action_names
from sheepshead import ACTION_IDS, DECK, TRUMP, Game, get_trick_points
from sheepshead.training.training_utils import (
    LEASTER_FINAL_REWARD_BONUS,
    RETURN_SCALE,
    TRICK_POINT_RATIO,
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    estimate_hand_strength_score,
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    update_intermediate_rewards_for_action,
)


def deal_with_p1_hand(p1_hand):
    """Build a full 5x6+2 deal with P1's hand fixed; the remaining 26 cards
    fill the other seats and blind in deck order. Only P1's hand composition
    matters for the tests that use this (PICK/PASS/ALONE shaping)."""
    rest = [c for c in DECK if c not in p1_hand]
    assert len(rest) == 26
    hands = [p1_hand] + [rest[i * 6 : i * 6 + 6] for i in range(4)]
    blind = rest[24:26]
    return make_game(hands, blind)


def force_lead(game, position, current_trick=0):
    """Force ``position`` to be the seat on lead (cards_played==0, no suit
    established yet), bypassing having them actually win a prior trick. The
    engine's PLAY-branch shaping only reads these fields, so this is a safe,
    targeted way to reach "defender/partner/picker leads trick N" states
    without hand-scripting several legal tricks first."""
    game.leader = position
    game.last_player = position - 1
    game.cards_played = 0
    game.current_suit = ""
    game.current_trick = current_trick


# ---------------------------------------------------------------------------
# Scenario A: called-ace game (mirrors test_game_scenarios.py HANDS_A)
# ---------------------------------------------------------------------------
HANDS_A = [
    ["QC", "QS", "JD", "AD", "10D", "7C"],
    ["AC", "10C", "KC", "JC", "9S", "8S"],
    ["QH", "QD", "JS", "9D", "7S", "KS"],
    ["JH", "8D", "7D", "AS", "10S", "9C"],
    ["AH", "10H", "KH", "9H", "8H", "7H"],
]
BLIND_A = ["KD", "8C"]

TRICK_A1 = [(1, "8C"), (2, "AC"), (3, "QD"), (4, "9C"), (5, "7H")]  # P3 trumps, 14 pts


def scenario_a_post_bury():
    game = make_game(HANDS_A, BLIND_A)
    act(game, 1, "PICK")
    act(game, 1, "CALL AC")
    act(game, 1, "BURY 7C")
    act(game, 1, "BURY 10D")
    return game


def scenario_a_after_trick1():
    game = scenario_a_post_bury()
    for pos, card in TRICK_A1:
        act(game, pos, f"PLAY {card}")
    return game


# ---------------------------------------------------------------------------
# Scenario D: leaster game (mirrors test_game_scenarios.py HANDS_D)
# ---------------------------------------------------------------------------
HANDS_D = [
    ["QC", "AC", "10C", "KC", "9C", "8C"],
    ["QS", "AS", "10S", "KS", "9S", "8S"],
    ["QH", "AH", "10H", "KH", "9H", "8H"],
    ["QD", "10D", "KD", "9D", "8D", "7D"],
    ["JC", "JS", "JH", "JD", "7C", "7S"],
]
BLIND_D = ["AD", "7H"]

TRICK_D1 = [(1, "AC"), (2, "8S"), (3, "8H"), (4, "7D"), (5, "7C")]  # P4 trumps; +blind


def scenario_d_leaster():
    game = make_game(HANDS_D, BLIND_D)
    for pos in range(1, 6):
        act(game, pos, "PASS")
    assert game.is_leaster
    return game


def scenario_d_after_trick1():
    game = scenario_d_leaster()
    for pos, card in TRICK_D1:
        act(game, pos, f"PLAY {card}")
    return game


# ---------------------------------------------------------------------------
# compute_known_points_rel
# ---------------------------------------------------------------------------
class TestComputeKnownPointsRel:
    def test_zero_before_any_points_or_bury(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        assert compute_known_points_rel(game.players[0]) == [0, 0, 0, 0, 0]

    def test_picker_sees_bury_points_once_fixed(self):
        game = scenario_a_post_bury()
        assert get_trick_points(game.bury) == 10  # 7C(0) + 10D(10)
        assert compute_known_points_rel(game.players[0]) == [10, 0, 0, 0, 0]

    def test_defender_does_not_see_bury_points(self):
        game = scenario_a_post_bury()
        assert compute_known_points_rel(game.players[2]) == [0, 0, 0, 0, 0]

    def test_final_points_rel_seating_picker(self):
        game = scenario_a_after_trick1()
        for pos, card in [(3, "KS"), (4, "AS"), (5, "8H"), (1, "QC"), (2, "9S")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(1, "QS"), (2, "JC"), (3, "9D"), (4, "JH"), (5, "9H")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(1, "AD"), (2, "10C"), (3, "QH"), (4, "8D"), (5, "10H")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(3, "JS"), (4, "7D"), (5, "KH"), (1, "KD"), (2, "KC")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(3, "7S"), (4, "10S"), (5, "AH"), (1, "JD"), (2, "8S")]:
            act(game, pos, f"PLAY {card}")
        assert game.is_done()
        assert game.points_taken == [48, 0, 62, 0, 0]
        # P1 (picker, self=rel1): own 48 + bury 10, then seats 2..5 in order.
        assert compute_known_points_rel(game.players[0]) == [58, 0, 62, 0, 0]
        # P3 (defender): self=62, then P4,P5,P1,P2 -> [62,0,0,48,0].
        assert compute_known_points_rel(game.players[2]) == [62, 0, 0, 48, 0]
        # P5 (defender): self=0, then P1,P2,P3,P4 -> [0,48,0,62,0].
        assert compute_known_points_rel(game.players[4]) == [0, 48, 0, 62, 0]

    def test_leaster_never_adds_bury_points(self):
        # Full 6-trick playthrough, all-pass leaster.
        game = scenario_d_after_trick1()
        for pos, card in [(4, "QD"), (5, "JD"), (1, "QC"), (2, "QS"), (3, "QH")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(1, "10C"), (2, "9S"), (3, "9H"), (4, "8D"), (5, "7S")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(4, "9D"), (5, "JH"), (1, "KC"), (2, "KS"), (3, "KH")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(5, "JS"), (1, "9C"), (2, "10S"), (3, "10H"), (4, "KD")]:
            act(game, pos, f"PLAY {card}")
        for pos, card in [(5, "JC"), (1, "8C"), (2, "AS"), (3, "AH"), (4, "10D")]:
            act(game, pos, f"PLAY {card}")
        assert game.is_done()
        assert game.points_taken == [14, 0, 0, 32, 74]
        assert compute_known_points_rel(game.players[0]) == [14, 0, 0, 32, 74]
        # P4 (self=rel1): 32, then P5,P1,P2,P3 -> [32,74,14,0,0].
        assert compute_known_points_rel(game.players[3]) == [32, 74, 14, 0, 0]
        # P5 (self=rel1): 74, then P1,P2,P3,P4 -> [74,14,0,0,32].
        assert compute_known_points_rel(game.players[4]) == [74, 14, 0, 0, 32]


# ---------------------------------------------------------------------------
# compute_seen_trump_mask
# ---------------------------------------------------------------------------
class TestComputeSeenTrumpMask:
    def test_picker_sees_hand_and_blind_before_bury(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        # P1 hand (QC,QS,JD,AD,10D) + blind (KD) are all trump.
        expected = [0] * len(TRUMP)
        for c in ["QC", "QS", "JD", "AD", "10D", "KD"]:
            expected[TRUMP.index(c)] = 1
        assert compute_seen_trump_mask(game.players[0]) == expected

    def test_non_picker_sees_only_own_hand(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        # P2 hand (AC,10C,KC,JC,9S,8S): only JC is trump.
        expected = [0] * len(TRUMP)
        expected[TRUMP.index("JC")] = 1
        assert compute_seen_trump_mask(game.players[1]) == expected

    def test_mask_unchanged_by_burying_since_bury_still_counted_for_picker(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        before = compute_seen_trump_mask(game.players[0])
        act(game, 1, "BURY 7C")
        act(game, 1, "BURY 10D")
        assert compute_seen_trump_mask(game.players[0]) == before

    def test_non_picker_gains_public_history_after_a_trick(self):
        game = scenario_a_post_bury()
        act(game, 1, "PLAY 8C")
        act(game, 2, "PLAY AC")
        act(game, 3, "PLAY QD")
        act(game, 4, "PLAY 9C")
        act(game, 5, "PLAY 7H")
        expected = [0] * len(TRUMP)
        expected[TRUMP.index("JC")] = 1  # own hand
        expected[TRUMP.index("QD")] = 1  # played by P3 in trick 1
        assert compute_seen_trump_mask(game.players[1]) == expected

    def test_under_card_seen_by_picker_only(self):
        hands_b = [
            ["QC", "JC", "AD", "10D", "AC", "AS"],
            ["AH", "10H", "KH", "QS", "JS", "7C"],
            ["QH", "QD", "JH", "9H", "8H", "7H"],
            ["10C", "KC", "9C", "8C", "10S", "KS"],
            ["9S", "8S", "7S", "8D", "7D", "JD"],
        ]
        blind_b = ["KD", "9D"]
        game = make_game(hands_b, blind_b)
        act(game, 1, "PICK")
        act(game, 1, "CALL AH UNDER")
        act(game, 1, "UNDER 10D")
        assert game.under_card == "10D"

        expected_picker = [0] * len(TRUMP)
        for c in ["QC", "JC", "AD", "KD", "9D", "10D"]:
            expected_picker[TRUMP.index(c)] = 1
        assert compute_seen_trump_mask(game.players[0]) == expected_picker

        expected_other = [0] * len(TRUMP)
        for c in ["QS", "JS"]:
            expected_other[TRUMP.index(c)] = 1
        assert compute_seen_trump_mask(game.players[1]) == expected_other


# ---------------------------------------------------------------------------
# compute_any_unseen_trump_higher_than_hand
# ---------------------------------------------------------------------------
class TestComputeAnyUnseenTrumpHigherThanHand:
    def test_zero_when_holding_the_top_trump(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        assert "QC" in game.players[0].hand
        assert compute_any_unseen_trump_higher_than_hand(game.players[0]) == 0

    def test_one_when_hand_has_no_trump_at_all(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        p5 = game.players[4]
        assert not any(c in TRUMP for c in p5.hand)
        assert compute_any_unseen_trump_higher_than_hand(p5) == 1

    def test_one_when_a_higher_trump_remains_unseen_mid_hand(self):
        game = scenario_a_after_trick1()
        p3 = game.players[2]  # best trump QH; QC/QS still unseen
        assert compute_any_unseen_trump_higher_than_hand(p3) == 1

    def test_zero_once_every_higher_trump_is_seen(self):
        # Synthetic fixture: the function only reads player.hand and
        # game.history/blind/bury/under_card, so direct assignment (same
        # technique make_game uses for hands) gives a clean, hand-verifiable
        # edge case without scripting several legal tricks.
        game = Game(seed=0)
        player = game.players[0]
        player.hand = ["JD", "7C", "8C"]  # best trump = JD, idx 7
        game.history[0] = ["QC", "QS", "QH", "QD", "JC", ""]
        game.history[1] = ["", "", "", "", "", "JS"]
        game.history[2] = ["JH", "", "", "", "", ""]
        assert compute_seen_trump_mask(player)[:8] == [1] * 8
        assert compute_any_unseen_trump_higher_than_hand(player) == 0

    def test_one_when_exactly_one_higher_trump_remains_unseen(self):
        game = Game(seed=0)
        player = game.players[0]
        player.hand = ["JD", "7C", "8C"]
        game.history[0] = ["QC", "QS", "QH", "QD", "JC", ""]
        game.history[1] = ["", "", "", "", "", "JS"]
        # JH (idx 6) is left unseen.
        assert compute_seen_trump_mask(player)[6] == 0
        assert compute_any_unseen_trump_higher_than_hand(player) == 1


# ---------------------------------------------------------------------------
# update_intermediate_rewards_for_action: PICK/PASS hand-strength shaping
# ---------------------------------------------------------------------------
class TestUpdateIntermediateRewardsPickPass:
    def test_estimate_hand_strength_score_weights(self):
        # 3 per queen, 2 per jack, 1 per other trump; pins the scoring table
        # these shaping brackets are keyed on.
        assert estimate_hand_strength_score(["AC", "10C", "KC", "9C", "8C", "7C"]) == 0
        assert estimate_hand_strength_score(["QC", "JC", "AC", "10C", "KC", "9C"]) == 5
        assert estimate_hand_strength_score(["QC", "JC", "JS", "AC", "10C", "KC"]) == 7
        assert estimate_hand_strength_score(["QC", "QS", "JC", "AC", "10C", "KC"]) == 8

    def test_weak_hand_penalizes_pick_and_rewards_pass(self):
        game = deal_with_p1_hand(["AC", "10C", "KC", "9C", "8C", "7C"])  # score 0
        player = game.players[0]
        t_pick = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PICK"], t_pick, [])
        assert t_pick["intermediate_reward"] == -0.1
        t_pass = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PASS"], t_pass, [])
        assert t_pass["intermediate_reward"] == 0.1

    def test_mid_hand_score_5_to_6_is_neutral(self):
        game = deal_with_p1_hand(["QC", "JC", "AC", "10C", "KC", "9C"])  # score 5
        player = game.players[0]
        t_pick = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PICK"], t_pick, [])
        assert t_pick["intermediate_reward"] == 0.0
        t_pass = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PASS"], t_pass, [])
        assert t_pass["intermediate_reward"] == 0.0

    def test_score_7_lightly_favors_pick(self):
        game = deal_with_p1_hand(["QC", "JC", "JS", "AC", "10C", "KC"])  # score 7
        player = game.players[0]
        t_pick = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PICK"], t_pick, [])
        assert t_pick["intermediate_reward"] == 0.02
        t_pass = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PASS"], t_pass, [])
        assert t_pass["intermediate_reward"] == -0.02

    def test_strong_hand_rewards_pick_and_penalizes_pass(self):
        game = deal_with_p1_hand(["QC", "QS", "JC", "AC", "10C", "KC"])  # score 8
        player = game.players[0]
        t_pick = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PICK"], t_pick, [])
        assert t_pick["intermediate_reward"] == 0.15
        t_pass = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PASS"], t_pass, [])
        assert t_pass["intermediate_reward"] == -0.15

    def test_pick_weight_scales_the_shaped_reward(self):
        game = deal_with_p1_hand(["AC", "10C", "KC", "9C", "8C", "7C"])  # score 0
        player = game.players[0]
        t = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(
            game, player, ACTION_IDS["PICK"], t, [], pick_weight=2.0
        )
        assert t["intermediate_reward"] == -0.2


# ---------------------------------------------------------------------------
# update_intermediate_rewards_for_action: ALONE shaping
# ---------------------------------------------------------------------------
class TestUpdateIntermediateRewardsAlone:
    def test_alone_with_weak_hand_is_penalized(self):
        game = deal_with_p1_hand(["AC", "10C", "KC", "9C", "8C", "7C"])  # score 0
        t = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, game.players[0], ACTION_IDS["ALONE"], t, [])
        assert t["intermediate_reward"] == -0.1

    def test_alone_at_the_score_8_boundary_is_still_penalized(self):
        game = deal_with_p1_hand(["QC", "QS", "JC", "AC", "10C", "KC"])  # score 8 (<=8)
        t = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, game.players[0], ACTION_IDS["ALONE"], t, [])
        assert t["intermediate_reward"] == -0.1

    def test_alone_above_score_8_has_no_penalty(self):
        game = deal_with_p1_hand(["QC", "QS", "QH", "AC", "10C", "KC"])  # score 9
        t = {"intermediate_reward": 0.0}
        update_intermediate_rewards_for_action(game, game.players[0], ACTION_IDS["ALONE"], t, [])
        assert t["intermediate_reward"] == 0.0


# ---------------------------------------------------------------------------
# update_intermediate_rewards_for_action: BURY shaping
# ---------------------------------------------------------------------------
def _forced_trump_only_bury_game():
    # P1's post-call hand is 7 trump + a lone club (the called suit), which
    # makes get_playable_called_picker_cards exclude the club entirely: every
    # legal bury option is trump.
    init_hand = ["QC", "QS", "QH", "QD", "JC", "JS"]
    blind = ["7D", "7C"]
    rest = [c for c in DECK if c not in (init_hand + blind)]
    hands = [init_hand] + [rest[i * 6 : i * 6 + 6] for i in range(4)]
    game = make_game(hands, blind)
    act(game, 1, "PICK")
    act(game, 1, "CALL AC")
    return game


class TestUpdateIntermediateRewardsBury:
    def test_burying_trump_when_fail_available_is_penalized(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        player = game.players[0]
        valid = player.get_valid_action_ids()
        assert "BURY 8C" in valid_action_names(game, 1)  # fail still available
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["BURY QC"], t, [])
        assert t["intermediate_reward"] == -0.1

    def test_burying_fail_when_trump_available_is_rewarded(self):
        game = make_game(HANDS_A, BLIND_A)
        act(game, 1, "PICK")
        act(game, 1, "CALL AC")
        player = game.players[0]
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["BURY 8C"], t, [])
        assert t["intermediate_reward"] == 0.02

    def test_forced_trump_bury_of_a_queen_is_penalized(self):
        game = _forced_trump_only_bury_game()
        player = game.players[0]
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["BURY QC"], t, [])
        assert t["intermediate_reward"] == -0.1

    def test_forced_trump_bury_of_a_jack_is_mildly_penalized(self):
        game = _forced_trump_only_bury_game()
        player = game.players[0]
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["BURY JC"], t, [])
        assert t["intermediate_reward"] == -0.02

    def test_forced_trump_bury_of_a_plain_trump_is_unshaped(self):
        game = _forced_trump_only_bury_game()
        player = game.players[0]
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["BURY 7D"], t, [])
        assert t["intermediate_reward"] == 0.0


# ---------------------------------------------------------------------------
# update_intermediate_rewards_for_action: PLAY-lead shaping
# ---------------------------------------------------------------------------
class TestUpdateIntermediateRewardsPlayLead:
    def test_picker_leading_trump_is_nudged(self):
        game = scenario_a_post_bury()
        player = game.players[0]
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        current_trick_transitions = []
        update_intermediate_rewards_for_action(
            game, player, ACTION_IDS["PLAY QC"], t, current_trick_transitions
        )
        assert t["intermediate_reward"] == 0.03
        assert current_trick_transitions == [t]
        assert t["head_shaping_reward"] == 0.03

    def test_defender_leading_trump_with_fail_available_is_discouraged(self):
        game = scenario_a_post_bury()
        player = game.players[2]  # P3: QH,QD,JS,9D,7S,KS
        force_lead(game, 3)
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PLAY QH"], t, [])
        assert t["intermediate_reward"] == -0.06

    def test_defender_leading_called_suit_early_is_encouraged(self):
        game = scenario_a_post_bury()
        player = game.players[3]  # P4 holds 9C (called suit is clubs)
        force_lead(game, 4, current_trick=2)
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PLAY 9C"], t, [])
        assert t["intermediate_reward"] == 0.1 - 0.02 * 2

    def test_defender_leading_off_called_suit_while_holding_it_is_discouraged(self):
        game = scenario_a_post_bury()
        player = game.players[3]  # P4 holds 9C but leads AS instead
        force_lead(game, 4)
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PLAY AS"], t, [])
        assert t["intermediate_reward"] == -0.01

    def test_defender_leading_fail_with_no_called_suit_in_hand_is_nudged(self):
        game = scenario_a_post_bury()
        player = game.players[2]  # P3 holds no clubs at all
        force_lead(game, 3)
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PLAY KS"], t, [])
        assert t["intermediate_reward"] == 0.03

    def test_secret_partner_leading_trump_is_encouraged(self):
        game = scenario_a_post_bury()
        player = game.players[1]  # P2 is secret partner, still holds AC + JC
        assert player.is_secret_partner
        force_lead(game, 2)
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(game, player, ACTION_IDS["PLAY JC"], t, [])
        assert t["intermediate_reward"] == 0.08

    def test_non_lead_play_is_unshaped_but_still_tracked(self):
        game = scenario_a_post_bury()
        player = game.players[0]
        game.leader = 3  # someone else leads; this player is following
        game.cards_played = 1
        game.current_suit = "T"
        t = {"intermediate_reward": 0.0, "valid_actions": set()}
        current_trick_transitions = []
        update_intermediate_rewards_for_action(
            game, player, ACTION_IDS["PLAY QC"], t, current_trick_transitions
        )
        assert t["intermediate_reward"] == 0.0
        assert current_trick_transitions == [t]
        assert t["head_shaping_reward"] == 0.0

    def test_play_weight_scales_lead_shaping(self):
        game = scenario_a_post_bury()
        player = game.players[0]
        valid = player.get_valid_action_ids()
        t = {"intermediate_reward": 0.0, "valid_actions": valid}
        update_intermediate_rewards_for_action(
            game, player, ACTION_IDS["PLAY QC"], t, [], play_weight=0.5
        )
        assert t["intermediate_reward"] == 0.015


# ---------------------------------------------------------------------------
# handle_trick_completion
# ---------------------------------------------------------------------------
class TestHandleTrickCompletion:
    def test_non_leaster_trick_rewards_by_team(self):
        game = scenario_a_post_bury()
        current_trick_transitions = []
        transitions = {}
        for pos, card in TRICK_A1:
            player = game.players[pos - 1]
            valid = player.get_valid_action_ids()
            t = {"player": player, "intermediate_reward": 0.0, "valid_actions": valid}
            update_intermediate_rewards_for_action(
                game, player, ACTION_IDS[f"PLAY {card}"], t, current_trick_transitions
            )
            transitions[pos] = t
            assert player.act(ACTION_IDS[f"PLAY {card}"])
            completed = handle_trick_completion(game, current_trick_transitions)

        assert completed is True
        assert game.trick_winners[0] == 3
        assert game.trick_points[0] == 14
        assert game.partner == 2  # revealed: P2 played AC in this trick
        trick_reward = 14 / TRICK_POINT_RATIO

        # P1 (picker) and P2 (partner) are on the losing (non-winning) team.
        assert transitions[1]["intermediate_reward"] == -trick_reward
        assert transitions[2]["intermediate_reward"] == -trick_reward
        # P3 (winner) and the other defenders P4/P5 share the winning team.
        assert transitions[3]["intermediate_reward"] == trick_reward
        assert transitions[4]["intermediate_reward"] == trick_reward
        assert transitions[5]["intermediate_reward"] == trick_reward
        assert current_trick_transitions == []  # tracking reset for next trick

    def test_leaster_trick_only_penalizes_the_winner(self):
        game = scenario_d_leaster()
        current_trick_transitions = []
        transitions = {}
        for pos, card in TRICK_D1:
            player = game.players[pos - 1]
            valid = player.get_valid_action_ids()
            t = {"player": player, "intermediate_reward": 0.0, "valid_actions": valid}
            update_intermediate_rewards_for_action(
                game, player, ACTION_IDS[f"PLAY {card}"], t, current_trick_transitions
            )
            transitions[pos] = t
            assert player.act(ACTION_IDS[f"PLAY {card}"])
            completed = handle_trick_completion(game, current_trick_transitions)

        assert completed is True
        assert game.trick_winners[0] == 4
        assert game.trick_points[0] == 22  # includes the blind
        trick_reward = 22 / TRICK_POINT_RATIO

        assert transitions[4]["intermediate_reward"] == -trick_reward
        for pos in (1, 2, 3, 5):
            assert transitions[pos]["intermediate_reward"] == 0.0

    def test_returns_false_mid_trick(self):
        game = scenario_a_post_bury()
        act(game, 1, "PLAY 8C")
        assert handle_trick_completion(game, []) is False


# ---------------------------------------------------------------------------
# process_episode_rewards / process_terminal_rewards
# ---------------------------------------------------------------------------
class TestProcessEpisodeRewards:
    def _transitions(self, player):
        return [
            {"player": player, "intermediate_reward": 0.05},
            {"player": player, "intermediate_reward": -0.02},
            {"player": player, "intermediate_reward": 0.1},
        ]

    def test_only_the_last_transition_gets_the_final_score(self):
        game = Game(seed=0)
        player = game.players[1]  # position 2
        final_scores = [4, -8, 1, 1, 2]
        transitions = self._transitions(player)
        out = list(process_episode_rewards(transitions, final_scores, is_leaster=False))
        assert [o["reward"] for o in out[:-1]] == [0.05, -0.02]
        expected_final = final_scores[1] / RETURN_SCALE + 0.1
        assert out[-1]["reward"] == expected_final
        assert out[0]["transition"] is transitions[0]

    def test_leaster_adds_the_bonus_only_to_the_final_reward(self):
        game = Game(seed=0)
        player = game.players[1]
        final_scores = [4, -8, 1, 1, 2]
        transitions = self._transitions(player)
        out = list(process_episode_rewards(transitions, final_scores, is_leaster=True))
        assert [o["reward"] for o in out[:-1]] == [0.05, -0.02]
        expected_final = (
            final_scores[1] / RETURN_SCALE + LEASTER_FINAL_REWARD_BONUS + 0.1
        )
        assert out[-1]["reward"] == expected_final

    def test_single_step_episode_still_gets_final_reward(self):
        game = Game(seed=0)
        player = game.players[1]
        final_scores = [4, -8, 1, 1, 2]
        out = list(
            process_episode_rewards(
                [{"player": player, "intermediate_reward": 0.2}],
                final_scores,
                is_leaster=False,
            )
        )
        assert out[0]["reward"] == final_scores[1] / RETURN_SCALE + 0.2


class TestProcessTerminalRewards:
    def test_ignores_intermediate_reward_and_leaster_flag(self):
        game = Game(seed=0)
        player = game.players[1]
        final_scores = [4, -8, 1, 1, 2]
        transitions = [
            {"player": player, "intermediate_reward": 0.05},
            {"player": player, "intermediate_reward": -0.02},
            {"player": player, "intermediate_reward": 0.1},
        ]
        for is_leaster in (False, True):
            out = list(process_terminal_rewards(transitions, final_scores, is_leaster=is_leaster))
            assert [o["reward"] for o in out[:-1]] == [0.0, 0.0]
            assert out[-1]["reward"] == final_scores[1] / RETURN_SCALE

    def test_default_is_leaster_is_false_and_has_no_effect(self):
        game = Game(seed=0)
        player = game.players[1]
        final_scores = [4, -8, 1, 1, 2]
        out = list(
            process_terminal_rewards(
                [{"player": player, "intermediate_reward": 0.0}], final_scores
            )
        )
        assert out[0]["reward"] == final_scores[1] / RETURN_SCALE


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
