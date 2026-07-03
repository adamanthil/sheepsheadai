"""Facade for the runtime table modules.

Historically everything lived here; the implementation is now split into
``models`` (dataclasses), ``manager`` (registry/caps/pruning), and ``views``
(payload serialization). Importers keep using ``server.runtime.tables``.
"""

from __future__ import annotations

from server.runtime.manager import (
    MAX_TABLES,
    STALE_CLIENT_SECONDS,
    TableLimitError,
    TableManager,
    prune_table_state,
    tables,
)
from server.runtime.models import ClientConn, Occupant, Table
from server.runtime.views import (
    ACTION_SIZE,
    _json_default,
    _try_int,
    build_player_state,
    get_actor_seat,
    get_valid_action_ids_for_seat,
    record_hand_result,
)

__all__ = [
    "ACTION_SIZE",
    "MAX_TABLES",
    "STALE_CLIENT_SECONDS",
    "ClientConn",
    "Occupant",
    "Table",
    "TableLimitError",
    "TableManager",
    "_json_default",
    "_try_int",
    "build_player_state",
    "get_actor_seat",
    "get_valid_action_ids_for_seat",
    "prune_table_state",
    "record_hand_result",
    "tables",
]
