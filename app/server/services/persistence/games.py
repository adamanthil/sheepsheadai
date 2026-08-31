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

from typing import TYPE_CHECKING, Any, Dict

from server.runtime.rules import plays_doublers
from server.services.persistence.game_table import (
    close_game_table,
    ensure_game_table,
)
from server.services.persistence.hand_play import (
    persist_finalize_game,
    persist_passed_out_game,
    persist_trick_completed,
)
from server.services.persistence.hand_setup import (
    persist_partner_revealed,
    persist_pick_resolved,
    persist_picker_decisions,
    persist_started_game,
)
from server.services.persistence.pool import get_db_pool
from server.services.persistence.snapshots import (
    capture_post_state,
    capture_pre_state,
)

if TYPE_CHECKING:
    from server.runtime.tables import Table

__all__ = [
    "capture_post_state",
    "capture_pre_state",
    "close_game_table",
    "ensure_game_table",
    "fire_game_hooks",
    "persist_finalize_game",
    "persist_partner_revealed",
    "persist_passed_out_game",
    "persist_pick_resolved",
    "persist_picker_decisions",
    "persist_started_game",
    "persist_trick_completed",
]


async def fire_game_hooks(
    table: "Table", pre: Dict[str, Any], post: Dict[str, Any]
) -> None:
    """Check state transitions and fire the appropriate hooks.

    ``pre`` and ``post`` are snapshots taken inside ``game_lock``; we never
    dereference the live ``Game`` here so that concurrent ``player.act()``
    cannot race with our DB I/O.
    """
    if table.current_game_id is None:
        return  # Hook 1 never ran; skip all subsequent hooks.
    pool = get_db_pool()

    # Hook 2: pick resolved (picker set OR leaster declared)
    if not pre["picker"] and post["picker"]:
        await persist_pick_resolved(
            pool, table, picker=post["picker"], is_leaster=False
        )
    elif not pre["is_leaster"] and post["is_leaster"]:
        if plays_doublers(table.rules):
            # Not a leaster: at a doublers table the deal is thrown in and
            # redealt at double stakes. Closing it here — rather than letting
            # the redeal race the hook — is what keeps is_leaster NULL on a
            # hand that was never played as one.
            await persist_passed_out_game(pool, table)
            return
        await persist_pick_resolved(pool, table, picker=None, is_leaster=True)

    # Hook 3: bury complete (second BURY action lands)
    if pre["bury_len"] < 2 and len(post["bury"]) == 2:
        await persist_picker_decisions(pool, table, post)

    # Hook 4: partner revealed — not for alone hands (partner == picker)
    if not pre["partner"] and post["partner"] and not post["alone_called"]:
        await persist_partner_revealed(pool, table, post["partner"])

    # Hook 5: trick completed
    if post["current_trick"] > pre["current_trick"]:
        trick_idx = post["current_trick"] - 1
        await persist_trick_completed(pool, table, post, trick_idx)

    # Hook 6: game done (fires on the same action as hook 5 for the last trick)
    if post["is_done"]:
        # Stored at the stake the hand was played for, so game_player.score
        # matches the running totals the players saw; game.score_multiplier
        # records how that stake was reached.
        multiplier = int(table.score_multiplier)
        await persist_finalize_game(
            pool, table, [s * multiplier for s in post["scores"]]
        )
