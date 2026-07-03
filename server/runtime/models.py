"""In-memory data model for live tables: connections, occupants, Table."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from ppo import PPOAgent
from sheepshead import Game


@dataclass
class ClientConn:
    client_id: str
    display_name: str
    seat: Optional[int] = None
    websocket: Optional[WebSocket] = None
    chat_timestamps: deque = field(default_factory=deque)
    # Long-lived cross-table identity (Phase 4). Set on /join.
    player_id: Optional[str] = None
    # Wall-clock time of the last websocket disconnect; None while connected.
    # Drives pruning of clients that never came back (prune_table_state).
    disconnected_at: Optional[float] = None


@dataclass
class Occupant:
    """Represents a player entity occupying a seat: human or AI.

    Humans use their client_id as occupant id; AIs are ephemeral UUIDs.
    """

    id: str
    display_name: str
    is_ai: bool = False


@dataclass
class Table:
    id: str
    name: str
    status: str = "open"  # open | playing | finished
    rules: Dict[str, Any] = field(default_factory=dict)
    fill_with_ai: bool = True
    host_client_id: Optional[str] = None
    # seat index 1..5 → occupant_id (humans use client_id; AIs use ephemeral uuid)
    seats: Dict[int, Optional[str]] = field(
        default_factory=lambda: {i: None for i in range(1, 6)}
    )
    # connected clients (client_id -> ClientConn)
    clients: Dict[str, ClientConn] = field(default_factory=dict)
    # all occupants by id (AI always present here; humans optional)
    occupants: Dict[str, Occupant] = field(default_factory=dict)
    # game/runtime
    game: Optional[Game] = None
    game_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Guards seat/occupant/client bookkeeping (join, seat choice, AI
    # replacement, reconnect reclaim). Distinct from game_lock so seat
    # operations never wait on AI inference. Never acquire game_lock while
    # holding this (or vice versa) — the two protect disjoint state.
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ai_agent: Optional[PPOAgent] = None
    ai_task: Optional[asyncio.Task] = None
    # background task to auto-close the table when humans disconnect
    autoclose_task: Optional[asyncio.Task] = None
    running_scores: Dict[str, int] = field(
        default_factory=dict
    )  # occupant_id -> cumulative score
    results_counted: bool = False
    # History of completed hands at this table (in chronological order)
    results_history: List[Dict[str, Any]] = field(default_factory=list)
    # Stable ordering of players for score display: occupant ids in the order of seats 1..5 at the first hand start
    initial_seat_order: List[str] = field(default_factory=list)
    # Snapshot of display names at first hand start, keyed by occupant id
    initial_names: Dict[str, str] = field(default_factory=dict)
    # Pending disconnect replacement tasks keyed by human client_id
    disconnect_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    # Reserved AI occupant id to reclaim per human client id
    reserved_ai_by_human: Dict[str, str] = field(default_factory=dict)
    # Chat log: bounded deque of chat messages (max 200 entries)
    chat_log: deque = field(default_factory=lambda: deque(maxlen=200))
    # Phase 5 persistence: game_id of the hand currently being persisted
    current_game_id: Optional[str] = None
    # Phase 5 persistence: seat (1-5) -> game_player_id (DB bigint) for current hand
    game_player_ids: Dict[int, int] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        def seat_name(occ_id: Optional[str]) -> Optional[str]:
            if not occ_id:
                return None
            if occ_id in self.clients:
                return self.clients[occ_id].display_name
            occ = self.occupants.get(occ_id)
            if occ:
                return occ.display_name
            return None

        seats_named = {i: seat_name(self.seats[i]) for i in self.seats}
        seats_ids = {i: self.seats[i] for i in self.seats}

        running_by_seat = {}
        for i in self.seats:
            occ = self.seats[i]
            running_by_seat[i] = (
                int(self.running_scores.get(occ or "", 0)) if occ else 0
            )

        seat_is_ai = {}
        for i in self.seats:
            occ_id = self.seats[i]
            occ = self.occupants.get(occ_id or "") if occ_id else None
            seat_is_ai[i] = bool(occ and occ.is_ai)

        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "rules": self.rules,
            "fillWithAI": self.fill_with_ai,
            "seats": seats_named,
            "runningBySeat": running_by_seat,
            "seatOccupants": seats_ids,
            "seatIsAI": seat_is_ai,
            "host": (
                self.clients[self.host_client_id].display_name
                if self.host_client_id and self.host_client_id in self.clients
                else None
            ),
            "resultsHistory": self.results_history,
            "initialSeatOrder": self.initial_seat_order,
            "initialNames": self.initial_names,
        }
