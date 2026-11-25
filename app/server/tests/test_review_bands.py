"""The bands a difference in expected score falls into."""

from __future__ import annotations

from server.services.review_bands import is_notable, standing


def test_a_decision_stands_by_the_size_of_its_difference():
    assert standing(None, forced=True) == "forced"
    assert standing(None, forced=False) == "agreed"
    assert standing(0.0, forced=False) == "even"
    assert standing(-0.2, forced=False) == "even"
    assert standing(0.3, forced=False) == "slight"
    assert standing(-0.9, forced=False) == "slight"
    assert standing(-1.0, forced=False) == "costly"
    assert standing(1.0, forced=False) == "better"


def test_a_hand_is_notable_when_its_net_bears_out_a_flagged_decision():
    # A costly decision, and the hand a full point down.
    assert is_notable(cost=1.6, gained=0.2, costly=1, better=0)
    assert is_notable(cost=0.3, gained=2.1, costly=0, better=1)
    # The same costly decision, cancelled by the smaller ones.
    assert not is_notable(cost=1.1, gained=0.6, costly=1, better=0)
    # A net that small differences added up to, with nothing flagged.
    assert not is_notable(cost=1.4, gained=0.0, costly=0, better=0)
    # A costly decision in a hand that came out ahead on its small ones.
    assert not is_notable(cost=1.0, gained=2.4, costly=1, better=0)
