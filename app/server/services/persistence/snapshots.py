"""Game-state snapshots for the persistence hooks.

Captured while holding ``game_lock`` around ``player.act()`` so that later
DB I/O never observes state mutated by a concurrent action handler.
"""

from __future__ import annotations

from typing import Any, Dict

from sheepshead import Game


def capture_pre_state(game: Game) -> Dict[str, Any]:
    return {
        "picker": game.picker,
        "is_leaster": game.is_leaster,
        "bury_len": len(game.bury),
        "partner": game.partner,
        "current_trick": game.current_trick,
    }


def capture_post_state(game: Game) -> Dict[str, Any]:
    """Snapshot every field hooks 2-6 may read.

    Taken inside ``game_lock`` immediately after ``player.act()`` so that
    later DB I/O cannot observe state that has since been mutated by a
    concurrent action handler.
    """
    is_done = game.is_done()
    return {
        "picker": game.picker,
        "partner": game.partner,
        "is_leaster": game.is_leaster,
        "current_trick": game.current_trick,
        "bury": list(game.bury),
        "alone_called": game.alone_called,
        "called_card": game.called_card,
        "under_card": game.under_card,
        "is_called_under": game.is_called_under,
        "leaders": list(game.leaders),
        "trick_winners": list(game.trick_winners),
        "trick_points": list(game.trick_points),
        "history": [list(row) for row in game.history],
        "is_done": is_done,
        "scores": [int(p.get_score()) for p in game.players] if is_done else None,
    }
