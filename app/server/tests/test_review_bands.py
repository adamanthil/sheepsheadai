"""The bands a difference in expected score falls into."""

from __future__ import annotations

from server.services.review_bands import is_notable, standing, summarize


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


def test_a_hand_is_summed_up_from_its_decisions():
    summary = summarize(
        [
            (0, True, None),
            (1, False, None),
            (2, False, -1.6),
            (3, False, 0.2),
            (4, False, -0.5),
            (5, False, 1.1),
        ]
    )

    assert summary.decisions == 5
    assert (summary.agreed, summary.even, summary.slight) == (1, 1, 1)
    assert (summary.costly, summary.better) == (1, 1)
    assert (summary.cost, summary.gained, summary.net) == (2.1, 1.3, -0.8)
    # The costliest decision sums the hand up, ahead of the best gain.
    assert summary.headline_index == 2
    # A flagged decision, but the net did not come out a full point.
    assert not summary.notable


def test_the_headline_falls_to_the_best_gain_or_to_nothing():
    summary = summarize([(0, False, None), (1, False, 0.4), (2, False, 1.0)])
    assert summary.headline_index == 2
    assert summarize([(0, False, None), (1, False, 0.4)]).headline_index is None
