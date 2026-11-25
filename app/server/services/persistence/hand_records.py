"""Finished hands read back from the database as replayable records.

The schema stores a hand across several tables (the deal, the seats, the
tricks); hand review wants it as one HandRecord it can step through the
engine. This is the only place that translation is made, for the server's
review worker and for the offline report script alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID

import asyncpg

from server.services.hand_review import HandRecord, recorded_actions
from server.services.persistence.pool import Querier
from sheepshead.game import ACTION_LOOKUP

GAMES_SQL = """
SELECT g.game_id, g.time_closed, t.name AS table_name, t.is_doublers,
       g.is_called_partner, g.is_double_on_the_bump, g.is_alone, g.is_leaster,
       called.code AS called_card, under.code AS under_card,
       (SELECT array_agg(c.code) FROM cardset_card cc JOIN card c USING (card_id)
         WHERE cc.cardset_id = g.blind_id) AS blind,
       (SELECT array_agg(c.code) FROM cardset_card cc JOIN card c USING (card_id)
         WHERE cc.cardset_id = g.bury_id) AS bury
FROM game g
JOIN game_table t USING (game_table_id)
LEFT JOIN card called ON called.card_id = g.called_card_id
LEFT JOIN card under ON under.card_id = g.under_card_id
WHERE g.time_closed IS NOT NULL
  AND ($1::uuid[] IS NULL OR g.game_id = ANY($1))
ORDER BY g.time_closed
"""

SEATS_SQL = """
SELECT gp.game_id, gp.position, gp.player_id, gp.name, gp.is_picker, gp.score,
       gp.is_substituted_pick, gp.ai_actions,
       (SELECT array_agg(c.code) FROM cardset_card cc JOIN card c USING (card_id)
         WHERE cc.cardset_id = gp.starting_hand_id) AS hand
FROM game_player gp
WHERE $1::uuid[] IS NULL OR gp.game_id = ANY($1)
"""

PLAYS_SQL = """
SELECT t.game_id, t.index AS trick, tc.index AS play, gp.position,
       c.code AS card, tc.is_substituted
FROM trick t
JOIN trick_card tc USING (trick_id)
JOIN game_player gp USING (game_player_id)
JOIN card c USING (card_id)
WHERE $1::uuid[] IS NULL OR t.game_id = ANY($1)
ORDER BY t.game_id, t.index, tc.index
"""


@dataclass
class StoredHand:
    game_id: UUID
    time_closed: datetime
    table: str
    record: HandRecord
    # Human seats -> display name.
    humans: Dict[int, str]
    scores: Dict[int, Optional[int]]
    # (seat, trick, card) plays and pick decisions the AI made for a human.
    substituted_plays: Set[Tuple[int, int]]
    substituted_pick: Set[int]
    # Seats with AI decisions that no flag accounts for (a call or a bury).
    unflagged_ai: Set[int]


async def load_hands(
    conn: Querier, game_ids: Optional[Sequence[UUID]] = None
) -> List[StoredHand]:
    """Every closed hand, oldest first, or just ``game_ids``."""
    only = list(game_ids) if game_ids is not None else None
    games = await conn.fetch(GAMES_SQL, only)
    seats = await conn.fetch(SEATS_SQL, only)
    plays = await conn.fetch(PLAYS_SQL, only)

    seats_by_game: Dict[Any, List[asyncpg.Record]] = {}
    for row in seats:
        seats_by_game.setdefault(row["game_id"], []).append(row)
    plays_by_game: Dict[Any, List[asyncpg.Record]] = {}
    for row in plays:
        plays_by_game.setdefault(row["game_id"], []).append(row)

    hands: List[StoredHand] = []
    for g in games:
        gid = g["game_id"]
        tricks: List[List[Tuple[int, str]]] = []
        substituted_plays: Set[Tuple[int, int]] = set()
        flagged: Dict[int, int] = {}
        for p in plays_by_game.get(gid, []):
            while len(tricks) <= p["trick"]:
                tricks.append([])
            tricks[p["trick"]].append((p["position"], p["card"]))
            if p["is_substituted"]:
                substituted_plays.add((p["position"], p["trick"]))
                flagged[p["position"]] = flagged.get(p["position"], 0) + 1
        picker: Optional[int] = None
        humans: Dict[int, str] = {}
        substituted_pick: Set[int] = set()
        unflagged_ai: Set[int] = set()
        for s in seats_by_game[gid]:
            seat = s["position"]
            if s["is_picker"]:
                picker = seat
            if s["player_id"] is None:
                continue
            humans[seat] = s["name"]
            if s["is_substituted_pick"]:
                substituted_pick.add(seat)
            accounted = flagged.get(seat, 0) + int(s["is_substituted_pick"])
            if s["ai_actions"] > accounted:
                unflagged_ai.add(seat)
        hands.append(
            StoredHand(
                game_id=gid,
                time_closed=g["time_closed"],
                table=g["table_name"],
                record=HandRecord(
                    partner_mode=1 if g["is_called_partner"] else 0,
                    double_on_the_bump=g["is_double_on_the_bump"],
                    doublers=g["is_doublers"],
                    hands={s["position"]: list(s["hand"]) for s in seats_by_game[gid]},
                    blind=list(g["blind"]),
                    picker=picker,
                    alone=bool(g["is_alone"]),
                    called_card=g["called_card"],
                    under_card=g["under_card"],
                    bury=list(g["bury"] or []),
                    tricks=tricks,
                ),
                humans=humans,
                scores={s["position"]: s["score"] for s in seats_by_game[gid]},
                substituted_plays=substituted_plays,
                substituted_pick=substituted_pick,
                unflagged_ai=unflagged_ai,
            )
        )
    return hands


def skipped_positions(hand: StoredHand) -> Set[Tuple[int, int]]:
    """(seat, position in the action sequence) of the decisions the AI made
    on a human seat's behalf."""
    skip: Set[Tuple[int, int]] = set()
    trick = 0
    plays_in_trick = 0
    for position, (seat, action_id) in enumerate(recorded_actions(hand.record)):
        action = ACTION_LOOKUP[action_id]
        if action in ("PICK", "PASS"):
            if seat in hand.substituted_pick:
                skip.add((seat, position))
        elif action.startswith("PLAY "):
            if (seat, trick) in hand.substituted_plays:
                skip.add((seat, position))
            plays_in_trick += 1
            if plays_in_trick == 5:
                trick, plays_in_trick = trick + 1, 0
    return skip
