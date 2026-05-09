"""Cardset persistence helpers (Phase 5 §5.3).

cardset rows are deduped via cards_hash so identical logical card sets
(blind, bury, starting hand) share a single cardset_id across all games.
"""

from __future__ import annotations

import asyncpg


async def upsert_cardset(conn: asyncpg.Connection, card_ids: list[int]) -> int:
    """Return cardset_id for the given card_ids, inserting if novel."""
    cards_hash = ",".join(str(cid) for cid in sorted(card_ids))
    row = await conn.fetchrow(
        "INSERT INTO cardset (cards_hash) VALUES ($1) "
        "ON CONFLICT (cards_hash) DO NOTHING "
        "RETURNING cardset_id",
        cards_hash,
    )
    if row is not None:
        cardset_id = row["cardset_id"]
        await conn.executemany(
            "INSERT INTO cardset_card (cardset_id, card_id) VALUES ($1, $2)",
            [(cardset_id, cid) for cid in card_ids],
        )
        return cardset_id
    return await conn.fetchval(
        "SELECT cardset_id FROM cardset WHERE cards_hash = $1",
        cards_hash,
    )
