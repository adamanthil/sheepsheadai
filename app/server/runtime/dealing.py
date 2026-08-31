"""Dealing a hand at a live table: the Game, its AI, and the doublers redeal.

Shared by the REST action handler and the AI turn loop, which both have to
notice a hand that has just passed out.
"""

from __future__ import annotations

import logging

from server.config import get_settings
from server.realtime.chat import emit_doubler_redeal_message
from server.runtime.models import Table
from server.runtime.rules import plays_doublers
from server.runtime.views import _try_int
from server.services.ai_loader import load_agent
from server.services.persistence.games import (
    persist_passed_out_game,
    persist_started_game,
)
from server.services.persistence.pool import get_db_pool
from sheepshead import Game


def build_table_agent(settings, table_id: str):
    """Load the table's AI, applying the configured convention mask if any.

    SHEEPSHEAD_CONVENTION_WRAP ("", "c1", "c2", "c1c2") wraps the agent with
    the deploy-time convention mask (sheepshead/agent/convention_wrapper.py):
    convention-violating defender leads are masked, the policy still picks the
    card. Table agents only — /analyze stays raw so the research scanners
    measure the unwrapped policy. An unknown value raises (fail fast at game
    start rather than silently no-opping).
    """
    agent = load_agent(settings.sheepshead_model_path)
    if settings.sheepshead_convention_wrap:
        from sheepshead.agent.convention_wrapper import wrap_agent

        agent = wrap_agent(agent, settings.sheepshead_convention_wrap)
        logging.info(
            "Table %s AI wrapped with convention mask %s",
            table_id,
            settings.sheepshead_convention_wrap,
        )
    return agent


def new_game_for_table(table: Table) -> Game:
    """Deal a fresh hand under the table's house rules."""
    rules = table.rules or {}
    return Game(
        double_on_the_bump=bool(rules.get("doubleOnTheBump", True)),
        partner_selection_mode=_try_int(rules.get("partnerMode", 1), 1),
    )


def table_has_ai(table: Table) -> bool:
    return any(
        (occ_id and (occ := table.occupants.get(occ_id)) and occ.is_ai)
        for occ_id in table.seats.values()
    )


def refresh_table_agent(table: Table) -> None:
    """Give the table a fresh AI for a new deal, or none if every seat is human.

    A new agent per deal rather than a reset one: recurrent memory is keyed
    by seat, and carrying last deal's memory into this one would have the AI
    reading a hand nobody holds any more.
    """
    table.ai_agent = (
        build_table_agent(get_settings(), table.id) if table_has_ai(table) else None
    )


def hand_passed_out(table: Table) -> bool:
    """True when the table plays doublers and the deal just passed out.

    The engine has no doublers concept: the fifth PASS puts it straight into
    leaster mode, and that is the state this reads. It is only ever true for
    a moment, between the action landing and the redeal below replacing the
    Game — no leaster state is ever broadcast at a doublers table.
    """
    return bool(table.game and table.game.is_leaster and plays_doublers(table.rules))


async def redeal_passed_out_hand(table: Table) -> bool:
    """Throw in a passed-out doublers deal and deal the next at double stakes.

    Returns True when a redeal happened, so callers know the Game they were
    working with is gone. Caller must not hold ``game_lock``.
    """
    async with table.game_lock:
        # Re-checked under the lock: the REST handler and the AI loop can
        # both reach this on the same fifth PASS, and only one may redeal.
        if not hand_passed_out(table):
            return False
        table.score_multiplier *= 2
        table.game = new_game_for_table(table)
        refresh_table_agent(table)
        game = table.game

    pool = get_db_pool()
    # Normally a no-op: the persistence hook for the fifth PASS has already
    # closed the thrown-in deal. Kept so a redeal can never leave the old
    # game row open and orphaned when it has not.
    await persist_passed_out_game(pool, table)
    await persist_started_game(pool, table, game)
    await emit_doubler_redeal_message(table, table.score_multiplier)
    logging.info(
        "Table %s passed out; redealt at %sx",
        table.id,
        table.score_multiplier,
        extra={"table_id": table.id},
    )
    return True
