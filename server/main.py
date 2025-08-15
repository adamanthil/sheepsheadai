import asyncio
import time
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
import logging

# Import game and agent from existing project
from sheepshead import (
    ACTION_IDS,
    ACTION_LOOKUP,
    Game,
    Player,
    get_cards_from_vector,
)
from ppo import PPOAgent
import numpy as np
from server.api.schemas import (
    CreateTableRequest,
    JoinTableRequest,
    StartGameRequest,
    UpdateTableRulesRequest,
    ActionRequest,
    SeatRequest,
    RedealRequest,
    CloseTableRequest,
)


# ------------------------------------------------------------
# Utilities and constants
# ------------------------------------------------------------


def _try_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


ACTION_SIZE = len(ACTION_IDS)


# ------------------------------------------------------------
# In-memory domain objects
# ------------------------------------------------------------


@dataclass
class ClientConn:
    client_id: str
    display_name: str
    seat: Optional[int] = None  # 1..5
    websocket: Optional[WebSocket] = None


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
    seats: Dict[int, Optional[str]] = field(default_factory=lambda: {i: None for i in range(1, 6)})
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
    running_scores: Dict[str, int] = field(default_factory=dict)  # occupant_id -> cumulative score
    results_counted: bool = False
    # History of completed hands at this table (in chronological order)
    # Each entry: { "hand": int, "timestamp": float, "bySeat": { seat: { "name": str, "score": int } }, "sum": int }
    results_history: List[Dict[str, Any]] = field(default_factory=list)
    # Stable ordering of players for score display: occupant ids in the order of seats 1..5 at the first hand start
    initial_seat_order: List[str] = field(default_factory=list)
    # Snapshot of display names at first hand start, keyed by occupant id
    initial_names: Dict[str, str] = field(default_factory=dict)

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

        # Running totals by current seat occupant
        running_by_seat = {}
        for i in self.seats:
            occ = self.seats[i]
            running_by_seat[i] = int(self.running_scores.get(occ or "", 0)) if occ else 0

        # AI flags by seat via occupants metadata
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
            "host": (self.clients[self.host_client_id].display_name if self.host_client_id and self.host_client_id in self.clients else None),
            "hostId": self.host_client_id,
            "resultsHistory": self.results_history,
            "initialSeatOrder": self.initial_seat_order,
            "initialNames": self.initial_names,
        }


class TableManager:
    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self._lock = asyncio.Lock()

    async def create_table(self, name: str, fill_with_ai: bool, rules: Dict[str, Any]) -> Table:
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


# ------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------


app = FastAPI(title="Sheepshead Realtime API")
logging.basicConfig(level=logging.INFO)

# Optional global model path via env var SHEEPSHEAD_MODEL_PATH (set by run_server.sh)
GLOBAL_MODEL_PATH: Optional[str] = os.environ.get("SHEEPSHEAD_MODEL_PATH")

# CORS configuration
if os.environ.get("ENV") == "production":
    # Production: Only allow specific domain
    cors_config = {
        "allow_origins": ["https://yourdomain.com"],  # Replace with production domain
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
else:
    # Development: Allow any host on port 3000 (localhost, .local domains, IPs, etc.)
    cors_config = {
        "allow_origin_regex": r"http://[^/]+:3000",
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }

app.add_middleware(CORSMiddleware, **cors_config)


# ------------------------------------------------------------
# Helper functions for state/view building
# ------------------------------------------------------------


def build_player_state(player: Player) -> Dict[str, Any]:
    """Build the per-seat state payload: vector and a small, readable view."""
    state_vec = player.get_state_vector()
    # Hand slice 16..47 (inclusive)
    hand_cards = get_cards_from_vector(state_vec[16:48])
    blind_cards = get_cards_from_vector(state_vec[48:80])
    bury_cards = get_cards_from_vector(state_vec[80:112])

    game = player.game
    current_trick = game.history[game.current_trick] if game.current_trick < len(game.history) else ["", "", "", "", ""]

    # Last completed trick info
    last_trick_index = int(game.current_trick) - 1
    last_trick = game.history[last_trick_index] if last_trick_index >= 0 else None
    last_trick_winner = game.trick_winners[last_trick_index] if last_trick_index >= 0 else 0
    last_trick_points = game.trick_points[last_trick_index] if last_trick_index >= 0 else 0

    # Final results if game is done
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
        "called_under": bool(getattr(game, "is_called_under", False)),
        "is_leaster": bool(game.is_leaster),
        "current_trick_index": int(game.current_trick),
        "current_trick": current_trick,
        "last_trick_index": last_trick_index if last_trick is not None else None,
        "last_trick": last_trick,
        "last_trick_winner": last_trick_winner,
        "last_trick_points": last_trick_points,
        "was_trick_just_completed": bool(getattr(game, "was_trick_just_completed", False)),
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
        "state": state_vec.astype(np.float32).tolist(),
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
    # Find first seat that currently has any valid actions
    for seat in range(1, 6):
        if get_valid_action_ids_for_seat(table, seat):
            return seat
    return None


async def broadcast_table_state(table: Table):
    """Send each connected human client their own masked state + valid actions."""
    if not table.game:
        return
    actor_seat = get_actor_seat(table)
    for cid, conn in list(table.clients.items()):
        if not conn.websocket:
            continue
        if not conn.seat:
            continue
        player = table.game.players[conn.seat - 1]
        payload = build_player_state(player)
        valid_actions = get_valid_action_ids_for_seat(table, conn.seat)
        msg = {
            "type": "state",
            "table": table.to_public_dict(),
            "yourSeat": conn.seat,
            "actorSeat": actor_seat,
            "state": payload["state"],
            "view": payload["view"],
            "valid_actions": valid_actions if conn.seat == actor_seat else [],
        }
        try:
            await conn.websocket.send_text(json.dumps(msg))
        except WebSocketDisconnect:
            conn.websocket = None
        except Exception as exc:
            logging.exception("broadcast send failed for table %s client %s: %s", table.id, cid, exc)


async def ai_observe_all(table: Table, except_seat: Optional[int] = None):
    if not table.ai_agent or not table.game:
        return
    for seat, occupant in table.seats.items():
        if not occupant:
            continue
        occ = table.occupants.get(occupant)
        if not occ or not occ.is_ai:
            continue
        if seat == except_seat:
            continue
        player = table.game.players[seat - 1]
        state = player.get_state_vector()
        valid = player.get_valid_action_ids()
        table.ai_agent.observe(state, player_id=seat, valid_actions=valid)


async def ai_take_turns(table: Table):
    """Loop AI moves until a human is the actor or the game ends.
    Avoid holding the game lock across sleeps and network IO.
    """
    if not table.game or not table.ai_agent:
        return
    while table.game and not table.game.is_done():
        # Determine if AI is the actor and select action under lock
        async with table.game_lock:
            if not table.game or not table.ai_agent:
                break
            actor = get_actor_seat(table)
            if actor is None:
                break
            occupant = table.seats.get(actor)
            if not occupant:
                break
            occ = table.occupants.get(occupant)
            if not occ or not occ.is_ai:
                # Human's turn
                break
            player = table.game.players[actor - 1]
            state = player.get_state_vector()
            valid = player.get_valid_action_ids()
            if not valid:
                break
            action_id, _, _ = table.ai_agent.act(state, valid_actions=valid, player_id=actor, deterministic=False)
            ok = player.act(int(action_id))
            if not ok:
                raise RuntimeError(
                    f"AI produced invalid action_id {action_id} for seat {actor}; valid set: {sorted(list(valid))}"
                )

        # After applying, let other AIs observe and broadcast
        await ai_observe_all(table, except_seat=actor)

        action_str = ACTION_LOOKUP.get(action_id, "")
        if isinstance(action_str, str) and action_str.startswith("PLAY "):
            await asyncio.sleep(0.5)
        await broadcast_table_state(table)
        if getattr(table.game, "was_trick_just_completed", False):
            await asyncio.sleep(3.3)
        else:
            if isinstance(action_str, str) and action_str == "PASS":
                await asyncio.sleep(0.5)

    # If game ended via AI actions, mark finished, tally results, record history, and broadcast
    if table.game and table.game.is_done():
        table.status = "finished"
        if not table.results_counted:
            for i in range(1, 6):
                occ = table.seats[i]
                if not occ:
                    continue
                pscore = table.game.players[i - 1].get_score()
                table.running_scores[occ] = table.running_scores.get(occ, 0) + int(pscore)
            table.results_counted = True
            try:
                entry: Dict[str, Any] = {"hand": len(table.results_history) + 1, "timestamp": time.time(), "bySeat": {}, "sum": 0}
                pub = table.to_public_dict()
                for i in range(1, 6):
                    name = pub["seats"][i]
                    occ_id = pub["seatOccupants"][i]
                    score = int(table.game.players[i - 1].get_score())
                    entry["bySeat"][i] = {"name": name, "id": occ_id, "score": score}
                    entry["sum"] += score
                table.results_history.append(entry)
            except Exception:
                logging.exception("failed to append results history for table %s", table.id)
        await broadcast_table_state(table)


def schedule_ai_turns(table: Table, initial_delay: float = 0.0) -> None:
    """Schedule background AI turns for a table, cancelling any prior task."""
    async def _runner():
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        try:
            await ai_take_turns(table)
        except Exception:
            logging.exception("ai_take_turns crashed for table %s", table.id)

    # Cancel previous task if running
    if table.ai_task and not table.ai_task.done():
        table.ai_task.cancel()
    table.ai_task = asyncio.create_task(_runner())


# ------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/tables")
def list_tables():
    return tables.list_tables()


@app.post("/api/tables")
async def create_table(req: CreateTableRequest):
    table = await tables.create_table(req.name, req.fillWithAI, req.rules or {})
    # Initialize running scores store (empty); AI occupants are created on demand
    return table.to_public_dict()


@app.post("/api/tables/{table_id}/join")
async def join_table(table_id: str, req: JoinTableRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if table.status != "open":
        raise HTTPException(status_code=400, detail="table_not_open")

    client_id = str(uuid.uuid4())
    # Do not auto-assign a seat; lobby waiting area allows explicit seat selection
    conn = ClientConn(client_id=client_id, display_name=req.display_name, seat=None)
    table.clients[client_id] = conn
    if not table.host_client_id:
        table.host_client_id = client_id

    # Broadcast lobby update (new client connected) and a callout-style message
    # Single broadcast including both event message and latest table snapshot
    await broadcast_table_event(table, {
        "type": "lobby_event",
        "message": f"{req.display_name} joined the table",
        "table": table.to_public_dict(),
    })

    return {
        "client_id": client_id,
        "table": table.to_public_dict(),
    }


@app.post("/api/tables/{table_id}/fill_ai")
async def fill_ai(table_id: str, req: RedealRequest | None = None):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    # Host-only: require client_id and it must match host
    client_id: Optional[str] = req.client_id if req else None
    if not client_id or not table.host_client_id or client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="only_host_can_fill_ai")

    ai_name_pool = ["Dan", "Kyle", "John", "Trevor", "Tim", "Tom"]
    for i in range(1, 6):
        if not table.seats[i]:
            occ_id = str(uuid.uuid4())
            display_name = ai_name_pool[(i - 1) % len(ai_name_pool)]
            table.occupants[occ_id] = Occupant(id=occ_id, display_name=display_name, is_ai=True)
            table.seats[i] = occ_id

    await broadcast_table_event(table, {"type": "table_update", "table": table.to_public_dict()})

    return table.to_public_dict()


## SeatRequest imported from server.api.schemas


@app.post("/api/tables/{table_id}/seat")
async def choose_seat(table_id: str, req: SeatRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if req.seat not in {1, 2, 3, 4, 5}:
        raise HTTPException(status_code=400, detail="invalid_seat")
    current = table.seats.get(req.seat)
    # Disallow displacement only if current occupant is a human
    if current and not (table.occupants.get(current).is_ai if table.occupants.get(current) else False):
        # Occupied by a human, cannot displace
        raise HTTPException(status_code=400, detail="seat_taken")
    if req.client_id not in table.clients:
        raise HTTPException(status_code=400, detail="client_not_joined")

    # Remove existing seat for this client if any
    for i in range(1, 6):
        if table.seats[i] == req.client_id:
            table.seats[i] = None
            break

    # Assign (displacing AI if present)
    table.seats[req.seat] = req.client_id
    table.clients[req.client_id].seat = req.seat

    await broadcast_table_event(table, {"type": "table_update", "table": table.to_public_dict()})

    return table.to_public_dict()


@app.post("/api/tables/{table_id}/start_waiting")
async def start_waiting(table_id: str, req: RedealRequest | None = None):
    # Host can ensure AIs fill empty seats in the waiting area but do not start game yet
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    # Host-only: require client_id and it must match host
    client_id: Optional[str] = req.client_id if req else None
    if not client_id or not table.host_client_id or client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="only_host_can_fill_ai")

    ai_name_pool = ["Dan", "Kyle", "John", "Trevor", "Tim", "Tom"]
    for i in range(1, 6):
        if not table.seats[i]:
            occ_id = str(uuid.uuid4())
            display_name = ai_name_pool[(i - 1) % len(ai_name_pool)]
            table.occupants[occ_id] = Occupant(id=occ_id, display_name=display_name, is_ai=True)
            table.seats[i] = occ_id

    await broadcast_table_event(table, {"type": "table_update", "table": table.to_public_dict()})

    return table.to_public_dict()


@app.get("/api/actions")
def get_actions():
    """Return action id to string mapping for the UI."""
    return {"action_lookup": ACTION_LOOKUP}


async def broadcast_table_event(table: Table, payload: Dict[str, Any]) -> None:
    """Broadcast any table-related event payload to all connected clients."""
    msg_txt = json.dumps(payload)
    for cid, conn in list(table.clients.items()):
        ws = conn.websocket
        if not ws:
            continue
        try:
            await ws.send_text(msg_txt)
        except WebSocketDisconnect:
            conn.websocket = None
        except Exception:
            logging.exception("broadcast_table_event send failed for table %s client %s", table.id, cid)


async def close_table(table: Table, reason: str = "closed") -> None:
    """Gracefully close a table: cancel AI, notify clients, close websockets, and remove from manager."""
    # Cancel AI/background tasks
    if table.ai_task and not table.ai_task.done():
        table.ai_task.cancel()
    if table.autoclose_task and not table.autoclose_task.done():
        table.autoclose_task.cancel()
    # Mark finished/closed for broadcast
    table.status = "finished"
    try:
        await broadcast_table_event(table, {"type": "table_closed", "reason": reason, "tableId": table.id})
    except Exception:
        pass
    # Close sockets politely
    for cid, conn in list(table.clients.items()):
        ws = conn.websocket
        if not ws:
            continue
        try:
            await ws.send_text(json.dumps({"type": "table_closed", "reason": reason, "tableId": table.id}))
            await ws.close()
        except Exception:
            conn.websocket = None
    # Finally remove from registry
    try:
        tables.delete_table(table.id)
    except Exception:
        logging.exception("failed deleting table %s", table.id)


def schedule_autoclose_if_no_humans(table: Table, delay_seconds: float = 5.0) -> None:
    """If there are no human players connected, schedule an auto-close after delay."""
    # Determine if any connected human client has a websocket
    def any_human_connected() -> bool:
        for cid, conn in table.clients.items():
            if conn.websocket is not None:
                return True
        return False

    # If any human connected, cancel pending autoclose
    if any_human_connected():
        if table.autoclose_task and not table.autoclose_task.done():
            table.autoclose_task.cancel()
        return

    async def _auto():
        try:
            await asyncio.sleep(delay_seconds)
            # Re-check before closing
            for _cid, _conn in table.clients.items():
                if _conn.websocket is not None:
                    return
            await close_table(table, reason="idle_all_disconnected")
        except asyncio.CancelledError:
            return
        except Exception:
            logging.exception("autoclose task failed for table %s", table.id)

    # Replace any existing task
    if table.autoclose_task and not table.autoclose_task.done():
        table.autoclose_task.cancel()
    table.autoclose_task = asyncio.create_task(_auto())

@app.patch("/api/tables/{table_id}/rules")
async def update_table_rules(table_id: str, req: UpdateTableRulesRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if table.status != "open":
        raise HTTPException(status_code=400, detail="game_already_started")

    # Only host can update rules
    if not req.client_id or not table.host_client_id or req.client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="only_host_can_update_rules")

    # Update the rules
    if not table.rules:
        table.rules = {}
    table.rules.update(req.rules)
    await broadcast_table_event(table, {"type": "table_update", "table": table.to_public_dict()})

    return {"status": "success", "rules": table.rules}


@app.post("/api/tables/{table_id}/close")
async def api_close_table(table_id: str, req: CloseTableRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    # Only host can close the table
    if not req.client_id or not table.host_client_id or req.client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="only_host_can_close")

    await close_table(table, reason="host_closed")
    return {"ok": True}


@app.post("/api/tables/{table_id}/start")
async def start_game(table_id: str, req: StartGameRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if table.status != "open":
        raise HTTPException(status_code=400, detail="already_started")

    # Host-only start: require a client_id and it must match host
    if not req.client_id or not table.host_client_id or req.client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="only_host_can_start")

    # Fill empty seats with AIs if configured
    if table.fill_with_ai:
        for i in range(1, 6):
            if not table.seats[i]:
                occ_id = str(uuid.uuid4())
                display_name = ["Dan", "Kyle", "John", "Trevor", "Tim", "Tom"][(i - 1) % 6]
                table.occupants[occ_id] = Occupant(id=occ_id, display_name=display_name, is_ai=True)
                table.seats[i] = occ_id

    # Must have 5 seats
    if not all(table.seats[i] for i in range(1, 6)):
        raise HTTPException(status_code=400, detail="not_enough_players")

    # Host (table creator) must be seated at one of the 5 seats
    if not table.host_client_id or all(table.seats[i] != table.host_client_id for i in range(1, 6)):
        raise HTTPException(status_code=400, detail="host_not_seated")

    # Instantiate game
    rules = table.rules or {}
    partner_mode = _try_int(rules.get("partnerMode", 1), 1)
    double_on_the_bump = bool(rules.get("doubleOnTheBump", True))

    game = Game(
        double_on_the_bump=double_on_the_bump,
        partner_selection_mode=partner_mode,
    )
    table.game = game
    table.status = "playing"
    table.results_counted = False
    # Capture initial order (once) from current seat occupants
    if not table.initial_seat_order:
        table.initial_seat_order = [str(table.seats[i] or "") for i in range(1, 6)]
        # capture names snapshot
        pub = table.to_public_dict()
        for i in range(1, 6):
            occ = table.seats[i]
            if occ:
                table.initial_names[str(occ)] = pub["seats"][i]

    # Create AI agent for this table if any AI players exist
    has_ai = any((table.occupants.get(occ_id).is_ai if occ_id and table.occupants.get(occ_id) else False) for occ_id in table.seats.values())
    if has_ai:
        from server.services.ai_loader import load_agent
        candidate_paths = [
            "pfsp_checkpoints_swish/pfsp_swish_checkpoint_200000.pth",
            "final_pfsp_swish_ppo.pth",
            "best_pfsp_swish_ppo.pth",
            "final_swish_ppo.pth",
            "best_swish_ppo.pth",
        ]
        table.ai_agent = load_agent(os.environ.get("SHEEPSHEAD_MODEL_PATH"), candidate_paths)

    # Notify lobby clients that the game has started
    await broadcast_table_event(table, {"type": "table_update", "table": table.to_public_dict()})

    # Initial broadcast and schedule AI chain after a small delay
    await broadcast_table_state(table)
    schedule_ai_turns(table, initial_delay=2.0)

    return table.to_public_dict()


@app.post("/api/tables/{table_id}/redeal")
async def redeal(table_id: str, req: RedealRequest | None = None):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    # Host-only redeal: require client_id and it must match host
    client_id: Optional[str] = req.client_id if req else None
    if not client_id or not table.host_client_id or client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="only_host_can_redeal")

    # Rotate seats clockwise (dealer becomes previous seat1 -> seat5)
    old = {i: table.seats[i] for i in range(1, 6)}
    new_map = {
        1: old[2],
        2: old[3],
        3: old[4],
        4: old[5],
        5: old[1],
    }
    for i in range(1, 6):
        table.seats[i] = new_map[i]
        # update client seat field for humans
        occ = new_map[i]
        if occ and occ in table.clients:
            table.clients[occ].seat = i

    # Reset game state; keep running_scores
    table.game = None
    table.status = "open"
    table.results_counted = False

    return table.to_public_dict()


@app.post("/api/tables/{table_id}/action")
async def post_action(table_id: str, req: ActionRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if not table.game:
        raise HTTPException(status_code=400, detail="game_not_started")

    conn = table.clients.get(req.client_id)
    if not conn or not conn.seat:
        raise HTTPException(status_code=400, detail="client_not_joined")

    actor_seat = get_actor_seat(table)
    if actor_seat != conn.seat:
        raise HTTPException(status_code=400, detail="not_your_turn")

    valid = get_valid_action_ids_for_seat(table, conn.seat)
    if req.action_id not in valid:
        raise HTTPException(status_code=400, detail="invalid_action")

    async with table.game_lock:
        player = table.game.players[conn.seat - 1]
        ok = player.act(int(req.action_id))
        if not ok:
            raise HTTPException(status_code=400, detail="apply_failed")

    await ai_observe_all(table, except_seat=conn.seat)
    await broadcast_table_state(table)
    schedule_ai_turns(table)

    # If the game has ended, ensure results are tallied and history is recorded
    if table.game and table.game.is_done():
        table.status = "finished"
        if not table.results_counted:
            # Tally running totals by occupant id
            for i in range(1, 6):
                occ = table.seats[i]
                if not occ:
                    continue
                pscore = table.game.players[i - 1].get_score()
                table.running_scores[occ] = table.running_scores.get(occ, 0) + int(pscore)
            table.results_counted = True
            # Append results history entry
            try:
                entry: Dict[str, Any] = {"hand": len(table.results_history) + 1, "timestamp": time.time(), "bySeat": {}, "sum": 0}
                pub = table.to_public_dict()
                for i in range(1, 6):
                    name = pub["seats"][i]
                    occ_id = pub["seatOccupants"][i]
                    score = int(table.game.players[i - 1].get_score())
                    entry["bySeat"][i] = {"name": name, "id": occ_id, "score": score}
                    entry["sum"] += score
                table.results_history.append(entry)
            except Exception:
                pass
        await broadcast_table_state(table)

    return {"ok": True}


# ------------------------------------------------------------
# WebSocket endpoint per table
# ------------------------------------------------------------


@app.websocket("/ws/table/{table_id}")
async def table_ws(websocket: WebSocket, table_id: str):
    await websocket.accept()
    params = websocket.query_params
    client_id = params.get("client_id")
    if not client_id:
        await websocket.close(code=4401)
        return
    try:
        table = tables.get_table(table_id)
    except KeyError:
        await websocket.close(code=4404)
        return

    if client_id not in table.clients:
        # unknown client
        await websocket.close(code=4403)
        return

    conn = table.clients[client_id]
    conn.websocket = websocket
    # On connect, cancel any pending autoclose
    try:
        schedule_autoclose_if_no_humans(table)
    except Exception:
        logging.exception("failed to manage autoclose on connect for table %s", table.id)

    # Initial state broadcast for connected client
    await broadcast_table_state(table)

    try:
        while True:
            try:
                await websocket.receive_text()
            except ValueError:
                # Ignore malformed messages and continue waiting
                logging.exception("Received malformed text over ws connection from client %s", client_id)
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup connection reference (do not free seat automatically)
        c = table.clients.get(client_id)
        if c:
            c.websocket = None
        # If no humans are connected anymore, start autoclose timer
        try:
            schedule_autoclose_if_no_humans(table)
        except Exception:
            logging.exception("failed to schedule autoclose for table %s", table.id)


# ------------------------------------------------------------
# Dev runner
# ------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=9000, reload=True)


