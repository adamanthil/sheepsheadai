"""Serialization of live game state into client-facing payloads."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from server.runtime.models import Table
from sheepshead.game import ACTION_IDS, Game
from sheepshead.views import build_player_state

ACTION_SIZE = len(ACTION_IDS)


def try_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except ValueError, TypeError:
        return default


def json_default(obj: Any):
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


# View fields everyone at the table sees: bids, played cards, a revealed
# partner, and the result. An allowlist rather than a denylist so a private
# field added to the per-seat view never leaks to spectators by default.
_PUBLIC_VIEW_KEYS = (
    "picker",
    "partner",
    "alone",
    "called_card",
    "called_card_display",
    "called_under",
    "is_leaster",
    "current_trick_index",
    "current_trick",
    "last_trick_index",
    "last_trick",
    "last_trick_winner",
    "last_trick_points",
    "was_trick_just_completed",
    "leaders",
    "trick_points",
    "trick_winners",
    "history",
    "is_done",
    "final",
)


def build_spectator_state(game: Game, score_multiplier: int = 1) -> Dict[str, Any]:
    """The state payload for an unseated client: public fields only, with
    the private card lists present but empty so it shares the seat shape."""
    view = build_player_state(game.players[0], score_multiplier)["view"]
    public: Dict[str, Any] = {k: view[k] for k in _PUBLIC_VIEW_KEYS}
    public.update(hand=[], blind=[], bury=[])
    return {
        "state": {"play_started": int(bool(game.play_started))},
        "view": public,
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

    Scores are scaled by the table's doublers stake, so a hand reached after
    two passed-out deals counts four times.

    Idempotent: a second call for the same hand is a no-op because
    ``results_counted`` is flipped on first call. Reset by ``start_game`` /
    ``redeal`` for the next hand.
    """
    if not table.game or table.results_counted:
        return

    multiplier = int(table.score_multiplier)
    for i in range(1, 6):
        occ = table.seats[i]
        if not occ:
            continue
        pscore = int(table.game.players[i - 1].get_score()) * multiplier
        table.running_scores[occ] = table.running_scores.get(occ, 0) + pscore
    table.results_counted = True
    table.hand_finished_at = time.monotonic()

    try:
        entry: Dict[str, Any] = {
            "hand": len(table.results_history) + 1,
            "timestamp": time.time(),
            "multiplier": multiplier,
            "bySeat": {},
            "sum": 0,
        }
        pub = table.to_public_dict()
        game = table.game
        # Who picked, so the scores card can mark the hands they were set.
        entry["picker"] = (
            None
            if game.is_leaster or not game.picker
            else pub["seatOccupants"][game.picker]
        )
        for i in range(1, 6):
            score = int(table.game.players[i - 1].get_score()) * multiplier
            entry["bySeat"][i] = {
                "name": pub["seats"][i],
                "id": pub["seatOccupants"][i],
                "score": score,
            }
            entry["sum"] += score
        table.results_history.append(entry)
    except Exception:
        logging.exception("failed to record results history for table %s", table.id)
