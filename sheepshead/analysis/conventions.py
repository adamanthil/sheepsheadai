"""Convention geometry shared by the probes, reports and scans.

The three predicates every convention instrument needs, defined once:
who is the secret partner, what a player could lead, and whether a card
is a fail of the called suit.
"""

from __future__ import annotations

from sheepshead import ACTION_LOOKUP, FAIL, PARTNER_BY_CALLED_ACE, TRUMP_SET, Game

FAIL_SET = frozenset(FAIL)


def is_secret_partner(game: Game, player) -> bool:
    """Whether ``player`` is the picker's (still secret) partner: holds the
    called card in called-ace mode, the Jack of Diamonds in JD mode
    (never when the picker went alone). The instruments' own predicate,
    kept separate from ``Player.is_secret_partner`` (which also answers
    False in a leaster) so calibrated probe numbers stay reproducible."""
    if game.partner_mode_flag == PARTNER_BY_CALLED_ACE:
        return bool(game.called_card) and game.called_card in player.hand
    return not game.alone_called and "JD" in player.hand


def lead_options(player) -> tuple[list[str], list[str]]:
    """(trump, fail) cards among the player's currently legal PLAY actions."""
    cards = [
        ACTION_LOOKUP[a].split(" ", 1)[1]
        for a in player.get_valid_action_ids()
        if ACTION_LOOKUP[a].startswith("PLAY ")
    ]
    return (
        [c for c in cards if c in TRUMP_SET],
        [c for c in cards if c not in TRUMP_SET],
    )


def called_suit_fail(card: str, called_card: str | None) -> bool:
    """True when ``card`` is a fail of the called card's suit (the suit letter
    is the last character for all fail cards; QC/JC etc. are trump). A hand
    with no called card has no called suit, so nothing matches."""
    return called_card is not None and card in FAIL_SET and card[-1] == called_card[-1]
