"""What the hand page is served around a stored review."""

from __future__ import annotations

from dataclasses import replace

from server.services.hand_review import HandRecord
from server.services.review_view import partner_card
from sheepshead.game import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

CALLED = HandRecord(
    partner_mode=PARTNER_BY_CALLED_ACE,
    double_on_the_bump=False,
    doublers=False,
    hands={},
    blind=[],
    picker=2,
    alone=False,
    called_card="AH",
    under_card=None,
    bury=[],
    tricks=[],
)


def test_the_partner_is_shown_by_the_called_ace_or_the_jack_of_diamonds():
    assert partner_card(CALLED) == "AH"
    jack = replace(CALLED, partner_mode=PARTNER_BY_JD, called_card=None)
    assert partner_card(jack) == "JD"


def test_no_card_shows_a_partner_where_there_is_none():
    assert partner_card(replace(CALLED, alone=True, called_card=None)) is None
    assert partner_card(replace(CALLED, picker=None, called_card=None)) is None
