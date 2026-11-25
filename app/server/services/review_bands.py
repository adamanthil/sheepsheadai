"""The bands a reviewed decision falls into, by its difference in expected
score against the AI's choice.

The one definition of them. The review engine (services.hand_review) bands
decisions as it makes them, the stored reviews are banded again when read
back (persistence.review_reads, whose SQL is built from the same two
numbers and tested against ``standing``), and the web app is sent each
decision's standing rather than the thresholds.

Pure arithmetic: no model, database or web types, so anything may import it.
"""

from __future__ import annotations

from typing import Literal, Optional

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
