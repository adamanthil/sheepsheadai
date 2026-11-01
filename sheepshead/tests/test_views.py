"""The per-seat view built for the web client (sheepshead.views)."""

from sheepshead.game import UNDER_TOKEN
from sheepshead.tests.game_test_utils import act, make_game, run_script
from sheepshead.views import build_player_state

HANDS = [
    ["QC", "JC", "AD", "10D", "AC", "AS"],
    ["AH", "10H", "KH", "QS", "JS", "7C"],
    ["QH", "QD", "JH", "9H", "8H", "7H"],
    ["10C", "KC", "9C", "8C", "10S", "KS"],
    ["9S", "8S", "7S", "8D", "7D", "JD"],
]
BLIND = ["KD", "9D"]


def under_called_game():
    """P1 picks, calls AH under, tucks 10D under and buries the black aces."""
    game = make_game(HANDS, BLIND)
    run_script(
        game,
        [
            (1, "PICK"),
            (1, "CALL AH UNDER"),
            (1, "UNDER 10D"),
            (1, "BURY AC"),
            (1, "BURY AS"),
        ],
    )
    return game


def view_of(game, position):
    return build_player_state(game.players[position - 1])["view"]


class TestUnderCard:
    def test_picker_sees_the_under_while_holding_it(self):
        game = under_called_game()
        assert view_of(game, 1)["under_card"] == "10D"
        # Face down for everyone else.
        assert all(view_of(game, p)["under_card"] is None for p in range(2, 6))

    def test_under_card_clears_once_played(self):
        game = under_called_game()
        act(game, 1, f"PLAY {UNDER_TOKEN}")
        assert view_of(game, 1)["under_card"] is None

    def test_no_under_without_an_under_call(self):
        game = make_game(HANDS, BLIND)
        run_script(game, [(1, "PICK"), (1, "ALONE"), (1, "BURY AC"), (1, "BURY AS")])
        assert view_of(game, 1)["under_card"] is None
