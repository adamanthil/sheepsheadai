"""Persistence hooks 1-4: deal, pick, bury, partner.

Each hook is its own transaction and logs-then-continues on failure so that
DB errors never break live play.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

import asyncpg

from sheepshead import DECK_IDS, Game

from server.services.persistence.cards import upsert_cardset
from server.services.persistence.pool import get_ai_player_id

if TYPE_CHECKING:
    from server.runtime.tables import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook 1: persist game + game_player rows after deal
# ---------------------------------------------------------------------------


async def persist_started_game(
    pool: asyncpg.Pool,
    table: "Table",
    game: Game,
) -> None:
    """Insert game row, blind cardset, and 5 game_player rows.

    Stashes the new game_id and game_player_ids back on the Table so
    subsequent hooks can reference them.
    """
    ai_player_id = get_ai_player_id()
    game_id = str(uuid.uuid4())

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Blind cardset
                blind_card_ids = [DECK_IDS[c] for c in game.blind]
                blind_id = await upsert_cardset(conn, blind_card_ids)

                # game row (decision fields all NULL at deal time)
                is_called_partner = int(table.rules.get("partnerMode", 1)) == 1
                is_double = bool(table.rules.get("doubleOnTheBump", True))
                await conn.execute(
                    """
                    INSERT INTO game (
                        game_id, game_table_id, is_double_on_the_bump, is_called_partner,
                        blind_id, time_created, time_closed,
                        is_alone, is_leaster, called_card_id, under_card_id, bury_id
                    ) VALUES ($1, $2, $3, $4, $5, now(), NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    UUID(game_id),
                    UUID(table.id),
                    is_double,
                    is_called_partner,
                    blind_id,
                )

                # game_player rows
                new_gp_ids: Dict[int, int] = {}
                for seat in range(1, 6):
                    occ_id = table.seats.get(seat)
                    occ = table.occupants.get(occ_id) if occ_id else None
                    is_ai = bool(occ and occ.is_ai)

                    player_uuid: Optional[UUID] = None
                    gp_ai_player_id: Optional[int] = None

                    if is_ai:
                        gp_ai_player_id = ai_player_id
                    else:
                        conn_obj = table.clients.get(occ_id) if occ_id else None
                        if conn_obj and conn_obj.player_id:
                            player_uuid = UUID(conn_obj.player_id)
                        else:
                            # Fallback: mint a player row so the DB constraint is satisfied.
                            player_uuid = uuid.uuid4()
                            logger.warning(
                                "seat %s has no player_id; minting %s (table=%s)",
                                seat,
                                player_uuid,
                                table.id,
                                extra={"table_id": table.id},
                            )
                            await conn.execute(
                                """
                                INSERT INTO player (player_id, name, time_created, last_updated)
                                VALUES ($1, NULL, now(), now())
                                """,
                                player_uuid,
                            )
                            if conn_obj:
                                conn_obj.player_id = str(player_uuid)

                    # Display name snapshot
                    if occ_id and occ_id in table.clients:
                        display_name = table.clients[occ_id].display_name
                    elif occ:
                        display_name = occ.display_name
                    else:
                        display_name = f"Seat {seat}"

                    # Starting hand cardset
                    hand_card_ids = [DECK_IDS[c] for c in game.players[seat - 1].hand]
                    starting_hand_id = await upsert_cardset(conn, hand_card_ids)

                    gp_id = await conn.fetchval(
                        """
                        INSERT INTO game_player (
                            game_id, player_id, ai_player_id, name, position,
                            starting_hand_id, is_picker, is_partner, score
                        ) VALUES ($1, $2, $3, $4, $5, $6, NULL, NULL, NULL)
                        RETURNING game_player_id
                        """,
                        UUID(game_id),
                        player_uuid,
                        gp_ai_player_id,
                        display_name,
                        seat,
                        starting_hand_id,
                    )
                    new_gp_ids[seat] = gp_id

        # Only update table state after the transaction commits successfully.
        table.current_game_id = game_id
        table.game_player_ids = new_gp_ids

    except Exception:
        logger.exception(
            "persist_started_game failed (table=%s)",
            table.id,
            extra={"table_id": table.id, "game_id": game_id},
        )


# ---------------------------------------------------------------------------
# Hook 2: pick resolved
# ---------------------------------------------------------------------------


async def persist_pick_resolved(
    pool: asyncpg.Pool,
    table: "Table",
    picker: Optional[int],
    is_leaster: bool,
) -> None:
    if not table.current_game_id:
        return
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE game SET is_leaster = $1 WHERE game_id = $2",
                    is_leaster,
                    UUID(table.current_game_id),
                )
                if not is_leaster:
                    await conn.execute(
                        """
                        UPDATE game_player SET is_picker = true
                        WHERE game_id = $1 AND position = $2
                        """,
                        UUID(table.current_game_id),
                        picker,
                    )
    except Exception:
        logger.exception(
            "persist_pick_resolved failed (table=%s game=%s)",
            table.id,
            table.current_game_id,
            extra={"table_id": table.id, "game_id": table.current_game_id},
        )


# ---------------------------------------------------------------------------
# Hook 3: picker decisions locked in (bury complete)
# ---------------------------------------------------------------------------


async def persist_picker_decisions(
    pool: asyncpg.Pool,
    table: "Table",
    post: Dict[str, Any],
) -> None:
    if not table.current_game_id:
        return
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                bury_id: Optional[int] = None
                if not post["is_leaster"] and post["bury"]:
                    bury_card_ids = [DECK_IDS[c] for c in post["bury"]]
                    bury_id = await upsert_cardset(conn, bury_card_ids)

                called_card_id: Optional[int] = (
                    DECK_IDS[post["called_card"]] if post["called_card"] else None
                )
                under_card_id: Optional[int] = (
                    DECK_IDS[post["under_card"]]
                    if post["is_called_under"] and post["under_card"]
                    else None
                )
                is_alone: Optional[bool] = (
                    post["alone_called"] if not post["is_leaster"] else None
                )

                await conn.execute(
                    """
                    UPDATE game
                    SET bury_id        = $1,
                        is_alone       = $2,
                        called_card_id = $3,
                        under_card_id  = $4
                    WHERE game_id = $5
                    """,
                    bury_id,
                    is_alone,
                    called_card_id,
                    under_card_id,
                    UUID(table.current_game_id),
                )
    except Exception:
        logger.exception(
            "persist_picker_decisions failed (table=%s game=%s)",
            table.id,
            table.current_game_id,
            extra={"table_id": table.id, "game_id": table.current_game_id},
        )


# ---------------------------------------------------------------------------
# Hook 4: partner revealed
# ---------------------------------------------------------------------------


async def persist_partner_revealed(
    pool: asyncpg.Pool,
    table: "Table",
    partner: int,
) -> None:
    if not table.current_game_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE game_player SET is_partner = true
                WHERE game_id = $1 AND position = $2
                """,
                UUID(table.current_game_id),
                partner,
            )
    except Exception:
        logger.exception(
            "persist_partner_revealed failed (table=%s game=%s)",
            table.id,
            table.current_game_id,
            extra={"table_id": table.id, "game_id": table.current_game_id},
        )
