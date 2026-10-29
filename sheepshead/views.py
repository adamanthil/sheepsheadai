"""One seat's view of a game, as the JSON payload the web client renders
and the /analyze trace (sheepshead.agent.traced_game) records per
decision.

The app's live tables and the research scanners read the same shape, so
it lives in the engine layer rather than in the app.
"""

from typing import Any, Dict

from sheepshead.game import CARD_FULL_NAMES, DECK, Player


def build_player_state(player: Player, score_multiplier: int = 1) -> Dict[str, Any]:
    """Build the per-seat state payload: dict state and a small, readable view.

    ``score_multiplier`` is the doublers stake the hand is being played for
    (see ``Table.score_multiplier``). The engine is stake-unaware, so the
    reported final scores are scaled here, at the one place they are read.
    """
    state_dict = player.get_state_dict()
    hand_cards = list(player.hand)
    # Game facts the picker knows (it saw the blind and chose the bury), read
    # off the game rather than the observation dict: whether an AGENT is
    # shown them is an architecture property (agent/observation.py), not a
    # property of the table view.
    game = player.game
    blind_cards = list(game.blind) if player.is_picker else []
    bury_cards = list(game.bury) if player.is_picker else []

    hand_cards.sort(key=lambda card: DECK.index(card))
    blind_cards.sort(key=lambda card: DECK.index(card))
    bury_cards.sort(key=lambda card: DECK.index(card))

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
        scores = [p.get_score() * score_multiplier for p in game.players]
        if game.is_leaster:
            final_payload = {
                "mode": "leaster",
                "multiplier": score_multiplier,
                "winner": game.get_leaster_winner(),
                "points_taken": game.points_taken,
                "scores": scores,
            }
        else:
            final_payload = {
                "mode": "standard",
                "multiplier": score_multiplier,
                "picker": game.picker,
                "partner": game.partner,
                "picker_score": game.get_final_picker_points(),
                "defender_score": game.get_final_defender_points(),
                "points_taken": game.points_taken,
                "scores": scores,
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
