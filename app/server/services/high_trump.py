"""The high trump at a decision: the highest trump not yet played.

Bookkeeping on the recorded hand, not a model output. It is worked out from
what the deciding seat could know: the cards played so far, and its own
bury if it is the picker. A trump another player buried still counts as
out, since the seat cannot know it is gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Optional, Sequence, Tuple

from sheepshead.game import TRUMP


@dataclass(frozen=True)
class HighTrump:
    # None once every trump has been played.
    card: Optional[str]
    held: bool
    # Trump neither played nor in the seat's hand.
    unseen: int


def high_trump(
    hand: Collection[str],
    played: Collection[str],
    own_bury: Collection[str] = (),
) -> HighTrump:
    """``hand`` is what the seat holds, ``played`` every card played before
    its decision (finished tricks and the current one), ``own_bury`` the
    cards it buried itself."""
    gone = set(played) | set(own_bury)
    left = [card for card in TRUMP if card not in gone]
    return HighTrump(
        card=left[0] if left else None,
        held=bool(left) and left[0] in hand,
        unseen=sum(card not in hand for card in left),
    )


def played_before(
    tricks: Sequence[Sequence[Tuple[int, str]]],
    trick: Optional[int],
    on_table: Sequence[Tuple[int, str]],
) -> list[str]:
    """The cards played before a decision in ``trick`` (None: before the
    first card) with ``on_table`` already played to it."""
    if trick is None:
        return []
    earlier = [card for plays in tricks[:trick] for _, card in plays]
    return earlier + [card for _, card in on_table]
