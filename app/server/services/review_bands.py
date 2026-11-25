"""The bands a reviewed decision falls into, by its difference in expected
score against the AI's choice.

The one definition of them, applied when a review is read and never
stored: what is kept of a decision is its difference, so moving a threshold
re-bands every review already made. The read queries
(persistence.review_reads) band in SQL built from the same two numbers and
tested against ``standing``; a hand's summary is counted up here; and the
web app is sent each decision's standing rather than the thresholds.

Pure arithmetic: no model, database or web types, so anything may import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

# Three bands for a difference in expected score, set from how often the
# two choices really ended in different scores when production hands were
# played out both ways (notebooks/Decision_Review_Design_202611.md, 11):
# under EVEN_BELOW the final score almost never moves; from FLAG_FROM it
# moves about four times in ten, nine in ten of those the predicted way;
# in between it is a real but small difference.
EVEN_BELOW = 0.25
FLAG_FROM = 1.0

Standing = Literal["forced", "agreed", "even", "slight", "costly", "better"]


def standing(difference: Optional[float], forced: bool) -> Standing:
    """How a decision stands against the AI's choice. ``difference`` is
    what the seat's choice gained (+) or cost (-), None where the two made
    the same choice; ``forced`` is a decision with one legal action."""
    if forced:
        return "forced"
    if difference is None:
        return "agreed"
    if difference <= -FLAG_FROM:
        return "costly"
    if difference >= FLAG_FROM:
        return "better"
    return "even" if abs(difference) < EVEN_BELOW else "slight"


def hand_net(cost: float, gained: float) -> float:
    """A hand's net against the AI from what its choices cost and gained,
    to the one decimal it is shown at."""
    return round(gained - cost, 1)


def is_notable(cost: float, gained: float, costly: int, better: int) -> bool:
    """Whether a reviewed hand is worth picking out of a list: one of its
    decisions cost or gained a full point, and the hand's net came out a
    full point the same way. A flagged decision that the hand's smaller
    differences cancelled does not count."""
    net = hand_net(cost, gained)
    return (costly > 0 and net <= -FLAG_FROM) or (better > 0 and net >= FLAG_FROM)


@dataclass(frozen=True)
class ReviewSummary:
    """One seat's reviewed hand in a few figures."""

    # Decisions where the seat had a choice, split by standing.
    decisions: int
    agreed: int
    even: int
    slight: int
    costly: int
    better: int
    # Expected score the seat's choices cost and gained, both >= 0; every
    # difference counts, so gained - cost is the hand's net.
    cost: float
    gained: float
    net: float
    # Index of the decision that sums the hand up: the costliest, else the
    # best gain; None when no difference reaches FLAG_FROM.
    headline_index: Optional[int]
    notable: bool


def summarize(
    decisions: Iterable[Tuple[int, bool, Optional[float]]],
) -> ReviewSummary:
    """A hand's decisions counted up. Each is (index, forced, difference),
    as ``standing`` takes them."""
    rows = list(decisions)
    standings = [standing(difference, forced) for _, forced, difference in rows]
    differences = [(difference, index) for index, _, difference in rows if difference]
    worst = min(differences, default=None)
    best = max(differences, default=None)
    headline: Optional[int] = None
    if worst is not None and worst[0] <= -FLAG_FROM:
        headline = worst[1]
    elif best is not None and best[0] >= FLAG_FROM:
        headline = best[1]
    cost = round(-sum(d for d, _ in differences if d < 0), 1)
    gained = round(sum(d for d, _ in differences if d > 0), 1)
    costly, better = standings.count("costly"), standings.count("better")
    return ReviewSummary(
        decisions=sum(s != "forced" for s in standings),
        agreed=standings.count("agreed"),
        even=standings.count("even"),
        slight=standings.count("slight"),
        costly=costly,
        better=better,
        cost=cost,
        gained=gained,
        net=hand_net(cost, gained),
        headline_index=headline,
        notable=is_notable(cost, gained, costly, better),
    )
