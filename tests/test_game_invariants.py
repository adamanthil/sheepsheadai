"""Property/fuzz tests: structural invariants that must hold for EVERY legal
game, checked over hundreds of randomly played deals in both partner modes.

Complements test_game_scenarios.py (which pins exact hand-computed games) and
test_ismcts_exit_regression.py::test_point_conservation (points + winner-range
only). Here each random game is additionally audited by an independent
re-implementation of the follow-suit rules, checked for turn-order/leader
consistency, card conservation, zero-sum scoring consistent with the score
table, and byte-identical replay determinism from the recorded action log.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead import (
    ACTION_LOOKUP,
    DECK,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    UNDER_TOKEN,
    Game,
    get_card_points,
    get_card_suit,
    get_trick_points,
)

N_GAMES = 150  # per partner mode


def play_random_game(seed, mode, double_on_the_bump=True, force_pass=False):
    """Play a full game with uniformly random legal actions, asserting the
    single-actor turn invariant at every step. Returns (game, action_log)."""
    game = Game(
        double_on_the_bump=double_on_the_bump, partner_selection_mode=mode, seed=seed
    )
    rng = random.Random(seed + 7919)
    log = []
    while not game.is_done():
        actors = [p for p in game.players if p.get_valid_action_ids()]
        assert len(actors) == 1, (
            f"seed {seed}: expected exactly one seat able to act, got "
            f"{[p.position for p in actors]}"
        )
        player = actors[0]
        valid = sorted(player.get_valid_action_ids())
        if force_pass and ACTION_LOOKUP[valid[0]] == "PICK" and len(valid) > 1:
            action = next(a for a in valid if ACTION_LOOKUP[a] == "PASS")
        else:
            action = rng.choice(valid)
        assert player.act(action) is True
        log.append((player.position, action))
    return game, log


def iter_games():
    for i in range(N_GAMES):
        for mode in (PARTNER_BY_JD, PARTNER_BY_CALLED_ACE):
            yield i, mode, play_random_game(1000 + i, mode)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
def _played_cards(game):
    return [c for trick in game.history for c in trick if c and c != UNDER_TOKEN]


def test_card_conservation_and_history_shape():
    saw_under = False
    for i, mode, (game, _) in iter_games():
        played = _played_cards(game)
        accounted = list(played) + list(game.bury)
        if game.under_card:
            accounted.append(game.under_card)
            saw_under = True
        if game.is_leaster:
            accounted += list(game.blind)
        assert sorted(accounted) == sorted(DECK), (
            f"seed {1000 + i} mode {mode}: cards not conserved"
        )
        # History is a full 6x5 grid of plays (UNDER token at most once).
        assert all(all(c for c in trick) for trick in game.history)
        under_plays = sum(1 for t in game.history for c in t if c == UNDER_TOKEN)
        assert under_plays == (1 if game.under_card else 0)
    assert saw_under, "no under-call games sampled; raise N_GAMES"


def test_leader_chain_matches_trick_winners():
    for i, mode, (game, _) in iter_games():
        assert game.leaders[0] == 1
        for t in range(5):
            assert game.leaders[t + 1] == game.trick_winners[t], (
                f"seed {1000 + i}: trick {t} winner does not lead trick {t + 1}"
            )
        assert game.current_trick == 6


def test_points_accounting():
    for i, mode, (game, _) in iter_games():
        assert sum(game.points_taken) == sum(game.trick_points)
        bury_pts = get_trick_points(game.bury)
        assert sum(game.trick_points) + bury_pts == 120
        if not game.is_leaster:
            picker_pts = game.get_final_picker_points()
            defender_pts = game.get_final_defender_points()
            assert picker_pts + defender_pts == 120
        # Per-trick points match the recorded cards (+ blind on leaster trick
        # 0, + under card on its trick).
        for t in range(6):
            expected = sum(
                get_card_points(c) for c in game.history[t] if c and c != UNDER_TOKEN
            )
            if game.is_leaster and t == 0:
                expected += get_trick_points(game.blind)
            if game.under_card and UNDER_TOKEN in game.history[t]:
                expected += get_card_points(game.under_card)
            assert game.trick_points[t] == expected


def _expected_multiplier(picker_pts, defender_pts, bump):
    if picker_pts == 120:
        m = 3
    elif picker_pts > 90:
        m = 2
    elif defender_pts == 120:
        m = -3
    elif defender_pts >= 90:
        m = -2
    elif defender_pts >= 60:
        m = -1
    else:
        m = 1
    if m < 0 and bump:
        m *= 2
    return m


def test_scores_zero_sum_and_match_score_table():
    for i, mode, (game, _) in iter_games():
        scores = [p.get_score() for p in game.players]
        assert sum(scores) == 0, f"seed {1000 + i}: scores not zero-sum"
        if game.is_leaster:
            assert sorted(scores) == [-1, -1, -1, -1, 4]
            winner = game.get_leaster_winner()
            assert scores[winner - 1] == 4
            # The winner took a trick and has minimal points among trick-takers.
            takers = {w for w in game.trick_winners}
            assert winner in takers
            assert all(
                game.points_taken[winner - 1] <= game.points_taken[w - 1]
                for w in takers
            )
        else:
            m = _expected_multiplier(
                game.get_final_picker_points(),
                game.get_final_defender_points(),
                game.is_double_on_the_bump,
            )
            for p in game.players:
                if p.position == game.picker == game.partner:
                    assert p.get_score() == 4 * m
                elif p.position == game.picker:
                    assert p.get_score() == 2 * m
                elif p.position == game.partner:
                    assert p.get_score() == m
                else:
                    assert p.get_score() == -m


def test_scores_without_bump():
    for i in range(40):
        game, _ = play_random_game(
            3000 + i, PARTNER_BY_CALLED_ACE, double_on_the_bump=False
        )
        if game.is_leaster:
            continue
        m = _expected_multiplier(
            game.get_final_picker_points(), game.get_final_defender_points(), False
        )
        picker = game.get_picker()
        expected = 4 * m if picker.is_partner else 2 * m
        assert picker.get_score() == expected


# ---------------------------------------------------------------------------
# Independent follow-suit audit
# ---------------------------------------------------------------------------
def _audit_follow_suit(game, seed):
    """Re-derive hands from initial deals and verify every play in the history
    was from-hand and followed suit whenever the seat could follow. This is an
    independent check of the trick mechanics (it does not use
    ``get_valid_actions``). Called-suit-specific restrictions are covered by
    the scripted scenarios; here we audit the universal rules."""
    hands = {p.position: list(p.initial_hand) for p in game.players}
    if game.picker:
        hands[game.picker] += list(game.blind)
        for c in game.bury:
            hands[game.picker].remove(c)
        if game.under_card:
            hands[game.picker].remove(game.under_card)

    for t in range(6):
        leader = game.leaders[t]
        order = [((leader - 1 + k) % 5) + 1 for k in range(5)]
        led_card = game.history[t][leader - 1]
        led_suit = (
            game.called_suit if led_card == UNDER_TOKEN else get_card_suit(led_card)
        )
        for seat in order:
            card = game.history[t][seat - 1]
            if card == UNDER_TOKEN:
                assert seat == game.picker, f"seed {seed}: non-picker played UNDER"
                continue
            assert card in hands[seat], (
                f"seed {seed}: trick {t} seat {seat} played {card} not in hand"
            )
            hands[seat].remove(card)
            if seat != leader and get_card_suit(card) != led_suit:
                in_suit = [
                    c for c in hands[seat] + [card] if get_card_suit(c) == led_suit
                ]
                # The one legal exception: the picker ducking a called-suit
                # lead with the face-down under instead is handled above; a
                # revoke is any remaining in-suit holding.
                assert not in_suit, (
                    f"seed {seed}: trick {t} seat {seat} revoked with {card}, "
                    f"held {in_suit} of suit {led_suit}"
                )
    for seat, rest in hands.items():
        assert rest == [], f"seed {seed}: seat {seat} finished holding {rest}"


def test_follow_suit_audit():
    for i, mode, (game, _) in iter_games():
        if game.is_leaster:
            # Leaster: hands are just the initial six, no blind pickup.
            pass
        _audit_follow_suit(game, 1000 + i)


# ---------------------------------------------------------------------------
# Determinism & replay
# ---------------------------------------------------------------------------
def test_seeded_deal_is_reproducible():
    for seed in (1, 42, 31337):
        g1 = Game(seed=seed)
        g2 = Game(seed=seed)
        assert [p.hand for p in g1.players] == [p.hand for p in g2.players]
        assert g1.blind == g2.blind


def test_replaying_action_log_reproduces_game():
    for i in range(60):
        mode = PARTNER_BY_JD if i % 2 else PARTNER_BY_CALLED_ACE
        game, log = play_random_game(5000 + i, mode)
        replay = Game(partner_selection_mode=mode, seed=5000 + i)
        for position, action in log:
            assert replay.players[position - 1].act(action) is True, (
                f"seed {5000 + i}: logged action {ACTION_LOOKUP[action]} by "
                f"P{position} rejected on replay"
            )
        assert replay.is_done()
        assert replay.history == game.history
        assert replay.points_taken == game.points_taken
        assert replay.trick_winners == game.trick_winners
        assert [p.get_score() for p in replay.players] == [
            p.get_score() for p in game.players
        ]


def test_forced_pass_games_reach_leaster():
    for i in range(20):
        game, _ = play_random_game(7000 + i, PARTNER_BY_CALLED_ACE, force_pass=True)
        assert game.is_leaster
        assert game.picker == 0
        assert game.bury == []
        assert sum(game.points_taken) == 120


# ---------------------------------------------------------------------------
# picking_hand constructor
# ---------------------------------------------------------------------------
def test_picking_hand_constructor():
    picking_hand = ["QC", "QS", "JD", "AD", "10D", "KD"]
    game = Game(picking_hand=picking_hand, picking_player=3, seed=11)
    assert game.picker == 3
    p3 = game.players[2]
    # Picker holds the requested six plus the blind.
    assert len(p3.hand) == 8
    assert set(picking_hand) <= set(p3.hand)
    assert set(game.blind) <= set(p3.hand)
    # Full deal is a partition of the deck.
    dealt = [c for p in game.players for c in p.hand]
    assert sorted(dealt + [c for c in DECK if c not in dealt]) == sorted(DECK)
    assert len(dealt) == 32  # 4x6 + picker's 8
    assert len(set(dealt)) == 32
    # Bidding is already resolved: only the picker may act (partner choice).
    actors = [p.position for p in game.players if p.get_valid_action_ids()]
    assert actors == [3]


def test_picking_hand_game_completes():
    picking_hand = ["QC", "QS", "QH", "QD", "JC", "JS"]
    game = Game(picking_hand=picking_hand, picking_player=1, seed=5)
    rng = random.Random(99)
    while not game.is_done():
        for p in game.players:
            valid = p.get_valid_action_ids()
            while valid:
                p.act(rng.choice(sorted(valid)))
                valid = p.get_valid_action_ids()
    assert sum(game.points_taken) + get_trick_points(game.bury) == 120
    assert sum(p.get_score() for p in game.players) == 0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
