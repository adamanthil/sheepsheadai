"""Convention geometry shared by the probes, reports and scans.

The two predicates every convention instrument needs, defined once: what
a player could lead, and whether a card is a fail of the called suit.
(Who the secret partner is comes from ``Player.is_secret_partner``.)
"""

from __future__ import annotations

from sheepshead import ACTION_LOOKUP, FAIL, TRUMP_SET

FAIL_SET = frozenset(FAIL)


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
