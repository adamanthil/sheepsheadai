"""Game persistence hooks (Phase 5 §5.2).

Six lifecycle hooks persist each hand's state incrementally to Postgres.
Each hook is its own transaction and logs-then-continues on failure so that
DB errors never break live play.

Hook call sites:
  1  persist_started_game    — end of start_game (after Game.__init__ deals)
  2  persist_pick_resolved   — when picker is set OR is_leaster becomes True
  3  persist_picker_decisions — when bury reaches 2 cards
  4  persist_partner_revealed — when game.partner becomes non-zero (not alone)
  5  persist_trick_completed  — when a trick resolves (was_trick_just_completed)
  6  persist_finalize_game    — when game.is_done() first returns True

Hooks 2-6 short-circuit if table.current_game_id is None (hook 1 never ran
or failed), so a partial-write hand is simply absent from the DB.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

import asyncpg

from sheepshead import DECK_IDS, UNDER_TOKEN, Game

from server.services.persistence.cards import upsert_cardset
from server.services.persistence.pool import get_ai_player_id, get_db_pool

if TYPE_CHECKING:
    from server.runtime.tables import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-state capture (call while holding game_lock, before player.act())
# ---------------------------------------------------------------------------

def capture_pre_state(game: Game) -> Dict[str, Any]:
    return {
        "picker": game.picker,
        "is_leaster": game.is_leaster,
        "bury_len": len(game.bury),
        "partner": game.partner,
        "current_trick": game.current_trick,
    }


# ---------------------------------------------------------------------------
# Hook dispatcher
# ---------------------------------------------------------------------------

async def fire_game_hooks(table: "Table", game: Game, pre: Dict[str, Any]) -> None:
    """Check state transitions after player.act() and fire appropriate hooks."""
    pool = get_db_pool()
    if pool is None:
        return
    if table.current_game_id is None:
        return  # Hook 1 never ran; skip all subsequent hooks.

    # Hook 2: pick resolved (picker set OR leaster declared)
    if not pre["picker"] and game.picker:
        await persist_pick_resolved(pool, table, game, is_leaster=False)
    elif not pre["is_leaster"] and game.is_leaster:
        await persist_pick_resolved(pool, table, game, is_leaster=True)

    # Hook 3: bury complete (second BURY action lands)
    if pre["bury_len"] < 2 and len(game.bury) == 2:
        await persist_picker_decisions(pool, table, game)

    # Hook 4: partner revealed — not for alone hands (partner == picker)
    if not pre["partner"] and game.partner and not game.alone_called:
        await persist_partner_revealed(pool, table, game)

    # Hook 5: trick completed
    if game.current_trick > pre["current_trick"]:
        trick_idx = game.current_trick - 1
        await persist_trick_completed(pool, table, game, trick_idx)

    # Hook 6: game done (fires on the same action as hook 5 for the last trick)
    if game.is_done():
        await persist_finalize_game(pool, table, game)


# ---------------------------------------------------------------------------
# Hook 0 (auxiliary): game_table bookkeeping
# ---------------------------------------------------------------------------

async def ensure_game_table(pool: asyncpg.Pool, table_id: str, table_name: str) -> None:
    """Create game_table row on first start; no-op if the row already exists."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO game_table (game_table_id, name, time_created, time_closed)
                VALUES ($1, $2, now(), NULL)
                ON CONFLICT (game_table_id) DO NOTHING
                """,
                UUID(table_id),
                table_name,
            )
    except Exception:
        logger.exception(
            "ensure_game_table failed (table=%s)", table_id,
            extra={"table_id": table_id},
        )


async def close_game_table(pool: asyncpg.Pool, table_id: str) -> None:
    """Stamp time_closed on the game_table row when the table is closed."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE game_table SET time_closed = now() WHERE game_table_id = $1",
                UUID(table_id),
            )
    except Exception:
        logger.exception(
            "close_game_table failed (table=%s)", table_id,
            extra={"table_id": table_id},
        )


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
                                seat, player_uuid, table.id,
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
            "persist_started_game failed (table=%s)", table.id,
            extra={"table_id": table.id, "game_id": game_id},
        )


# ---------------------------------------------------------------------------
# Hook 2: pick resolved
# ---------------------------------------------------------------------------

async def persist_pick_resolved(
    pool: asyncpg.Pool,
    table: "Table",
    game: Game,
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
                        game.picker,
                    )
    except Exception:
        logger.exception(
            "persist_pick_resolved failed (table=%s game=%s)",
            table.id, table.current_game_id,
            extra={"table_id": table.id, "game_id": table.current_game_id},
        )


# ---------------------------------------------------------------------------
# Hook 3: picker decisions locked in (bury complete)
# ---------------------------------------------------------------------------

async def persist_picker_decisions(
    pool: asyncpg.Pool,
    table: "Table",
    game: Game,
) -> None:
    if not table.current_game_id:
        return
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                bury_id: Optional[int] = None
                if not game.is_leaster and game.bury:
                    bury_card_ids = [DECK_IDS[c] for c in game.bury]
                    bury_id = await upsert_cardset(conn, bury_card_ids)

                called_card_id: Optional[int] = (
                    DECK_IDS[game.called_card] if game.called_card else None
                )
                under_card_id: Optional[int] = (
                    DECK_IDS[game.under_card]
                    if game.is_called_under and game.under_card
                    else None
                )
                is_alone: Optional[bool] = (
                    game.alone_called if not game.is_leaster else None
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
            table.id, table.current_game_id,
            extra={"table_id": table.id, "game_id": table.current_game_id},
        )


# ---------------------------------------------------------------------------
# Hook 4: partner revealed
# ---------------------------------------------------------------------------

async def persist_partner_revealed(
    pool: asyncpg.Pool,
    table: "Table",
    game: Game,
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
                game.partner,
            )
    except Exception:
        logger.exception(
            "persist_partner_revealed failed (table=%s game=%s)",
            table.id, table.current_game_id,
            extra={"table_id": table.id, "game_id": table.current_game_id},
        )


# ---------------------------------------------------------------------------
# Hook 5: trick completed
# ---------------------------------------------------------------------------

async def persist_trick_completed(
    pool: asyncpg.Pool,
    table: "Table",
    game: Game,
    trick_idx: int,
) -> None:
    if not table.current_game_id:
        return
    try:
        leader = game.leaders[trick_idx]
        winner = game.trick_winners[trick_idx]
        points = game.trick_points[trick_idx]
        lead_gp_id = table.game_player_ids[leader]
        winner_gp_id = table.game_player_ids[winner]

        async with pool.acquire() as conn:
            async with conn.transaction():
                trick_id = await conn.fetchval(
                    """
                    INSERT INTO trick (game_id, index, lead_player_id, winning_player_id, points)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING trick_id
                    """,
                    UUID(table.current_game_id),
                    trick_idx,
                    lead_gp_id,
                    winner_gp_id,
                    points,
                )

                rows = []
                for k in range(5):
                    seat = (leader - 1 + k) % 5 + 1  # 1-based seat
                    history_idx = (leader - 1 + k) % 5  # 0-based history index
                    card_str = game.history[trick_idx][history_idx]
                    if card_str == UNDER_TOKEN:
                        card_id = DECK_IDS[game.under_card]
                    else:
                        card_id = DECK_IDS[card_str]
                    gp_id = table.game_player_ids[seat]
                    rows.append((trick_id, card_id, gp_id, k))

                await conn.executemany(
                    """
                    INSERT INTO trick_card (trick_id, card_id, game_player_id, index)
                    VALUES ($1, $2, $3, $4)
                    """,
                    rows,
                )
    except Exception:
        logger.exception(
            "persist_trick_completed failed (table=%s game=%s trick=%s)",
            table.id, table.current_game_id, trick_idx,
            extra={
                "table_id": table.id,
                "game_id": table.current_game_id,
                "trick_idx": trick_idx,
            },
        )


# ---------------------------------------------------------------------------
# Hook 6: finalize game
# ---------------------------------------------------------------------------

async def persist_finalize_game(
    pool: asyncpg.Pool,
    table: "Table",
    game: Game,
) -> None:
    if not table.current_game_id:
        return
    game_id = table.current_game_id
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE game SET time_closed = now() WHERE game_id = $1",
                    UUID(game_id),
                )
                for seat in range(1, 6):
                    score = int(game.players[seat - 1].get_score())
                    gp_id = table.game_player_ids[seat]
                    await conn.execute(
                        "UPDATE game_player SET score = $1 WHERE game_player_id = $2",
                        score,
                        gp_id,
                    )
        # Clear so the next hand starts fresh.
        table.current_game_id = None
        table.game_player_ids = {}
    except Exception:
        logger.exception(
            "persist_finalize_game failed (table=%s game=%s)",
            table.id, game_id,
            extra={"table_id": table.id, "game_id": game_id},
        )
