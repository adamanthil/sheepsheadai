"""Serialization of live game state into client-facing payloads."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from server.runtime.models import Table
from sheepshead import (
    ACTION_IDS,
    CARD_FULL_NAMES,
    DECK,
    Player,
)

ACTION_SIZE = len(ACTION_IDS)


def _try_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
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
