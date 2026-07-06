"""Shared helpers for the Game-engine test suite.

Not collected by pytest (no ``test_`` prefix). Provides fixed-deal game
construction and a scripted-action driver so scenario tests can assert
hand-computed expectations against fully deterministic games.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead import ACTION_IDS, DECK, Game


def make_game(hands, blind, **kwargs):
    """Build a Game with exactly the given five 6-card hands and 2-card blind.

    The Game is constructed normally (seed=0) and then the deal is overridden,
    which is safe because dealing is pure list assignment in ``Game.__init__``.
    Validates the deal is a partition of the 32-card deck.
    """
    cards = [c for h in hands for c in h] + list(blind)
    assert len(hands) == 5 and all(len(h) == 6 for h in hands), "need five 6-card hands"
    assert len(blind) == 2, "blind must be 2 cards"
    assert sorted(cards) == sorted(DECK), (
        f"deal is not a deck partition (missing {set(DECK) - set(cards)}, "
        f"dup/extra {sorted(c for c in cards if cards.count(c) > 1 or c not in DECK)})"
    )
    game = Game(seed=0, **kwargs)
    for player, hand in zip(game.players, hands):
        player.initial_hand = list(hand)
        player.hand = list(hand)
    game.blind = list(blind)
    return game


def act(game, position, action_name):
    """Have the seat at ``position`` perform the named action, asserting it is
    currently legal (and that no other action path silently rejects it)."""
    player = game.players[position - 1]
    action_id = ACTION_IDS[action_name]
    valid = player.get_valid_action_ids()
    assert action_id in valid, (
        f"P{position}: {action_name!r} not legal; legal now: {sorted(a for a in valid)}"
    )
    assert player.act(action_id) is True
    return player


def run_script(game, script):
    """Run a list of (position, action_name) steps through ``act``."""
    for position, action_name in script:
        act(game, position, action_name)


def valid_action_names(game, position):
    """The set of legal action strings for a seat right now."""
    player = game.players[position - 1]
    from sheepshead import ACTION_LOOKUP

    return {ACTION_LOOKUP[a] for a in player.get_valid_action_ids()}


def sole_actor(game):
    """Position (1-5) of the unique seat with any legal action, asserting
    exactly one seat can act (the engine's turn invariant)."""
    actors = [p.position for p in game.players if p.get_valid_action_ids()]
    assert len(actors) == 1, f"expected exactly one actor, got {actors}"
    return actors[0]
