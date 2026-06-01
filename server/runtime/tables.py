from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket

from ppo import PPOAgent
from sheepshead import (
    ACTION_IDS,
    CARD_FULL_NAMES,
    DECK,
    Game,
    Player,
)

ACTION_SIZE = len(ACTION_IDS)


def _try_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except ValueError, TypeError:
        return default


def _json_default(obj: Any):
    """JSON serializer for numpy types used in observation dicts."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class ClientConn:
    client_id: str
    display_name: str
    seat: Optional[int] = None
    websocket: Optional[WebSocket] = None
    chat_timestamps: deque = field(default_factory=deque)
    # Long-lived cross-table identity (Phase 4). Set on /join.
    player_id: Optional[str] = None


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


class TableManager:
    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self._lock = asyncio.Lock()

    async def create_table(
        self, name: str, fill_with_ai: bool, rules: Dict[str, Any]
    ) -> Table:
        async with self._lock:
            tid = str(uuid.uuid4())
            table = Table(id=tid, name=name, fill_with_ai=fill_with_ai, rules=rules)
            self.tables[tid] = table
            return table

    def get_table(self, table_id: str) -> Table:
        if table_id not in self.tables:
            raise KeyError("table_not_found")
        return self.tables[table_id]

    def list_tables(self) -> List[Dict[str, Any]]:
        return [t.to_public_dict() for t in self.tables.values()]

    def delete_table(self, table_id: str) -> None:
        if table_id in self.tables:
            del self.tables[table_id]


tables = TableManager()


def build_player_state(player: Player) -> Dict[str, Any]:
    """Build the per-seat state payload: dict state and a small, readable view."""
    state_dict = player.get_state_dict()
    hand_cards = list(player.hand)
    blind_cards = [
        DECK[card_id - 1] for card_id in state_dict["blind_ids"] if card_id > 0
    ]
    bury_cards = [
        DECK[card_id - 1] for card_id in state_dict["bury_ids"] if card_id > 0
    ]

    hand_cards.sort(key=lambda card: DECK.index(card))
    blind_cards.sort(key=lambda card: DECK.index(card))
    bury_cards.sort(key=lambda card: DECK.index(card))

    game = player.game

    current_trick = ["", "", "", "", ""]
    if (
        hasattr(game, "play_started")
        and game.play_started
        and game.current_trick < len(game.history)
    ):
        trick_cards = game.history[game.current_trick]
        for i, card in enumerate(trick_cards):
            if card != "":
                current_trick[i] = card

    last_trick_index = int(game.current_trick) - 1
    last_trick = game.history[last_trick_index] if last_trick_index >= 0 else None
    last_trick_winner = (
        game.trick_winners[last_trick_index] if last_trick_index >= 0 else 0
    )
    last_trick_points = (
        game.trick_points[last_trick_index] if last_trick_index >= 0 else 0
    )

    is_done = bool(game.is_done())
    final_payload = None
    if is_done:
        if game.is_leaster:
            final_payload = {
                "mode": "leaster",
                "winner": game.get_leaster_winner(),
                "points_taken": game.points_taken,
                "scores": [p.get_score() for p in game.players],
            }
        else:
            final_payload = {
                "mode": "standard",
                "picker": game.picker,
                "partner": game.partner,
                "picker_score": game.get_final_picker_points(),
                "defender_score": game.get_final_defender_points(),
                "points_taken": game.points_taken,
                "scores": [p.get_score() for p in game.players],
            }

    view = {
        "player": player.position,
        "picker": game.picker,
        "partner": game.partner,
        "alone": bool(game.alone_called),
        "called_card": game.called_card,
        "called_card_display": (
            CARD_FULL_NAMES.get(game.called_card, game.called_card)
            if game.called_card
            else None
        ),
        "called_under": bool(getattr(game, "is_called_under", False)),
        "is_leaster": bool(game.is_leaster),
        "current_trick_index": int(game.current_trick),
        "current_trick": current_trick,
        "last_trick_index": last_trick_index if last_trick is not None else None,
        "last_trick": last_trick,
        "last_trick_winner": last_trick_winner,
        "last_trick_points": last_trick_points,
        "was_trick_just_completed": bool(
            getattr(game, "was_trick_just_completed", False)
        ),
        "leaders": game.leaders,
        "trick_points": game.trick_points,
        "trick_winners": game.trick_winners,
        "hand": hand_cards,
        "blind": blind_cards,
        "bury": bury_cards,
        "history": game.history,
        "is_done": is_done,
        "final": final_payload,
    }

    return {
        "state": state_dict,
        "view": view,
    }


def get_valid_action_ids_for_seat(table: Table, seat: int) -> List[int]:
    if not table.game:
        return []
    player = table.game.players[seat - 1]
    return sorted(list(player.get_valid_action_ids()))


def get_actor_seat(table: Table) -> Optional[int]:
    if not table.game:
        return None
    for seat in range(1, 6):
        if get_valid_action_ids_for_seat(table, seat):
            return seat
    return None


def record_hand_result(table: Table) -> None:
    """Tally per-seat scores into running totals and append a results-history entry.

    Idempotent: a second call for the same hand is a no-op because
    ``results_counted`` is flipped on first call. Reset by ``start_game`` /
    ``redeal`` for the next hand.
    """
    if not table.game or table.results_counted:
        return

    for i in range(1, 6):
        occ = table.seats[i]
        if not occ:
            continue
        pscore = int(table.game.players[i - 1].get_score())
        table.running_scores[occ] = table.running_scores.get(occ, 0) + pscore
    table.results_counted = True

    try:
        entry: Dict[str, Any] = {
            "hand": len(table.results_history) + 1,
            "timestamp": time.time(),
            "bySeat": {},
            "sum": 0,
        }
        pub = table.to_public_dict()
        for i in range(1, 6):
            score = int(table.game.players[i - 1].get_score())
            entry["bySeat"][i] = {
                "name": pub["seats"][i],
                "id": pub["seatOccupants"][i],
                "score": score,
            }
            entry["sum"] += score
        table.results_history.append(entry)
    except Exception:
        logging.exception("failed to record results history for table %s", table.id)
