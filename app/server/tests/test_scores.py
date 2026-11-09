"""Score blocks: running totals restart only when someone new is dealt in."""

from __future__ import annotations

from server.runtime.dealing import new_game_for_table
from server.runtime.models import ClientConn, Occupant, Table
from server.runtime.occupants import give_seat_to_ai
from server.runtime.scores import assign_score_columns, rotate_score_columns
from server.runtime.seating import seat_over_ai
from server.runtime.views import record_hand_result


def _table() -> Table:
    """ "ann" (player p-ann) in seat 2, the AI in the rest, first hand dealt."""
    table = Table(id="t", name="scores")
    table.clients["ann"] = ClientConn(
        client_id="ann", display_name="Ann", seat=2, player_id="p-ann"
    )
    table.seats[2] = "ann"
    for seat in (1, 3, 4, 5):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name=f"AI {seat}", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"
    _deal(table)
    return table


def _deal(table: Table) -> None:
    table.game = new_game_for_table(table)
    table.results_counted = False
    assign_score_columns(table)


def _play_hand(table: Table) -> None:
    game = table.game
    assert game is not None
    while not game.is_done():
        for player in game.players:
            valid = sorted(player.get_valid_action_ids())
            if valid:
                assert player.act(valid[0])
    record_hand_result(table)


def _next_deal(table: Table) -> None:
    old = dict(table.seats)
    for i in range(1, 6):
        occ = table.seats[i] = old[i % 5 + 1]
        if occ in table.clients:
            table.clients[occ].seat = i
    rotate_score_columns(table)
    _deal(table)


def test_the_first_deal_opens_a_block_named_for_who_sits_down():
    table = _table()

    assert table.score_segments == [
        {
            "columns": [
                {"id": "ai1", "name": "AI 1"},
                {"id": "p-ann", "name": "Ann"},
                {"id": "ai3", "name": "AI 3"},
                {"id": "ai4", "name": "AI 4"},
                {"id": "ai5", "name": "AI 5"},
            ]
        }
    ]


def test_columns_follow_their_players_round_the_table():
    table = _table()
    _play_hand(table)
    _next_deal(table)
    _play_hand(table)

    first, second = table.results_history
    assert second["segment"] == 0
    assert first["bySeat"][2]["column"] == second["bySeat"][1]["column"] == "p-ann"
    assert table.to_public_dict()["runningBySeat"][1] == (
        first["bySeat"][2]["score"] + second["bySeat"][1]["score"]
    )


def test_the_ai_standing_in_for_a_player_who_left_keeps_their_total():
    table = _table()
    _play_hand(table)
    give_seat_to_ai(table, table.clients["ann"])
    del table.clients["ann"]
    _next_deal(table)
    _play_hand(table)

    assert len(table.score_segments) == 1
    assert table.results_history[-1]["bySeat"][1]["column"] == "p-ann"


def test_a_player_back_from_a_drop_keeps_their_block():
    table = _table()
    _play_hand(table)
    # Their socket dropped and the AI they took over from stands in.
    table.reserved_ai_by_human["ann"] = "stand-in"
    table.occupants["stand-in"] = Occupant(id="stand-in", display_name="AI", is_ai=True)
    table.seats[2] = "stand-in"
    table.clients["ann"].seat = None
    _next_deal(table)
    _play_hand(table)
    table.seats[1] = "ann"
    table.clients["ann"].seat = 1
    _next_deal(table)

    assert len(table.score_segments) == 1
    assert table.seat_columns[5] == "p-ann"


def test_a_newcomer_dealt_in_opens_a_new_block_from_zero():
    table = _table()
    _play_hand(table)
    table.clients["bob"] = ClientConn(
        client_id="bob", display_name="Bob", player_id="p-bob"
    )
    seat_over_ai(table, 4, "bob")
    # Seated between hands: Bob shows 0, not AI 4's total.
    assert table.to_public_dict()["runningBySeat"][4] == 0
    _next_deal(table)

    assert len(table.score_segments) == 2
    assert [c["name"] for c in table.score_segments[1]["columns"]] == [
        "Ann",
        "AI 3",
        "Bob",
        "AI 5",
        "AI 1",
    ]
    assert table.to_public_dict()["runningBySeat"] == {i: 0 for i in range(1, 6)}
    _play_hand(table)
    assert table.results_history[-1]["segment"] == 1


def test_a_hand_taken_over_mid_way_stays_in_the_block_it_was_dealt_in():
    table = _table()
    table.clients["bob"] = ClientConn(
        client_id="bob", display_name="Bob", player_id="p-bob"
    )
    seat_over_ai(table, 4, "bob")  # Take seat now
    _play_hand(table)

    entry = table.results_history[-1]
    assert entry["segment"] == 0
    assert entry["bySeat"][4] | {"score": 0} == {
        "name": "Bob",
        "id": "bob",
        "column": "ai4",
        "score": 0,
    }
