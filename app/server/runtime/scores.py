"""Score columns: whose running total each seat's hand adds to.

Running scores count over a block of hands played by one set of players.
A seat's column follows its occupant as the deal rotates the seats, and
stays with the seat while the AI stands in for a player who dropped, timed
out, or left -- their seat is kept, so their total carries on. Only a
newcomer dealt into a seat starts a new block, from 0, at that deal; a
hand they took over mid-way stays in the block it was dealt in. Everyone
carried into the new block keeps their place in the scores table, so a
column can be read down across blocks; newcomers fill the places left.

A leaf module (it imports only the data model), so the dealing and
results code can both use it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from server.runtime.models import Table


def assign_score_columns(table: Table) -> None:
    """Give each seat its score column for the hand being dealt, opening a
    new block when someone new is dealt in. Call once the seats are final
    for the deal (after the rotation). Caller holds ``table.state_lock``."""
    columns: Dict[int, str] = {}
    newcomer = not table.score_segments
    for seat in range(1, 6):
        occ = table.seats.get(seat) or ""
        prev = table.seat_columns.get(seat)
        if occ in table.clients:
            columns[seat] = table.score_column_key(occ)
            newcomer = newcomer or columns[seat] != prev
        else:
            columns[seat] = prev or occ
    if newcomer:
        columns = {seat: _fresh_column(table, seat) for seat in range(1, 6)}
        names = table.to_public_dict()["seats"]
        dealt = [
            {"id": columns[seat], "name": names[seat] or f"Seat {seat}"}
            for seat in range(1, 6)
        ]
        previous = table.score_segments[-1]["columns"] if table.score_segments else []
        table.score_segments.append({"columns": _keep_places(previous, dealt)})
        table.running_scores = {}
    table.seat_columns = columns


def _keep_places(
    previous: List[Dict[str, Any]], dealt: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The new block's columns: each one carried over from ``previous`` at
    its old position, the rest in ``dealt``'s (seat) order in the places
    left. The deal rotates the seats every hand, so seat order alone would
    move a returning player to a different column."""
    by_id = {c["id"]: c for c in dealt}
    places: List[Dict[str, Any] | None] = [
        by_id.pop(c["id"], None) for c in previous[: len(dealt)]
    ]
    places += [None] * (len(dealt) - len(places))
    placed = {id(c) for c in places if c is not None}
    rest = iter([c for c in dealt if id(c) not in placed])
    return [c if c is not None else next(rest) for c in places]


def rotate_score_columns(table: Table) -> None:
    """Move each seat's column down a seat with its occupant, as
    games._pass_the_deal does, seat 1's into seat 5."""
    old = dict(table.seat_columns)
    table.seat_columns = {i: old[i % 5 + 1] for i in range(1, 6) if (i % 5 + 1) in old}


def _fresh_column(table: Table, seat: int) -> str:
    """A seat's column in a new block: its player's own, or, for an AI
    standing in for a player still at the table, that player's."""
    occ = table.seats.get(seat) or ""
    if occ not in table.clients:
        for conn in table.clients.values():
            if occ in (
                conn.home_occupant,
                table.reserved_ai_by_human.get(conn.client_id),
            ):
                return table.score_column_key(conn.client_id)
    return table.score_column_key(occ)
