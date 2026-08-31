"""The doublers house rule: a passed-out deal is thrown in and redealt.

The engine has no doublers concept -- its fifth PASS goes straight to leaster
mode -- so the rule lives entirely in the server layer. These tests pin the
two halves of it: the redeal itself, and the stake it leaves behind.
"""

from __future__ import annotations

import pytest

from server.runtime.dealing import redeal_passed_out_hand
from server.runtime.models import Occupant, Table
from server.runtime.views import record_hand_result
from sheepshead import ACTION_IDS, Game


def _passed_out_table(all_pass_mode: str) -> Table:
    """A five-human table whose current deal everyone has passed on."""
    table = Table(
        id="t-doublers", name="doublers", rules={"allPassMode": all_pass_mode}
    )
    for seat in range(1, 6):
        occ_id = f"human-{seat}"
        table.occupants[occ_id] = Occupant(id=occ_id, display_name=f"P{seat}")
        table.seats[seat] = occ_id
    table.game = Game()
    for player in table.game.players:
        assert player.act(ACTION_IDS["PASS"])
    return table


@pytest.fixture
def no_persistence(monkeypatch):
    """Neutralize the DB hooks; persistence is covered by the API-flow tests."""
    import server.runtime.dealing as dealing

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(dealing, "get_db_pool", lambda: object())
    monkeypatch.setattr(dealing, "persist_passed_out_game", noop)
    monkeypatch.setattr(dealing, "persist_started_game", noop)


async def test_passed_out_deal_is_thrown_in_and_redealt(no_persistence):
    table = _passed_out_table("doublers")
    thrown_in = table.game

    assert await redeal_passed_out_hand(table) is True

    assert table.score_multiplier == 2
    assert table.game is not thrown_in
    # The client never sees the momentary leaster state the engine went into.
    assert table.game is not None and not table.game.is_leaster
    # Same seats, same order: only the cards change.
    assert table.seats == {seat: f"human-{seat}" for seat in range(1, 6)}
    assert any("redealing at 2x" in m["body"] for m in table.chat_log)


async def test_stake_doubles_again_on_a_second_pass_out(no_persistence):
    table = _passed_out_table("doublers")

    assert await redeal_passed_out_hand(table) is True
    for player in table.game.players:
        assert player.act(ACTION_IDS["PASS"])
    assert await redeal_passed_out_hand(table) is True

    assert table.score_multiplier == 4


async def test_leasters_table_keeps_playing_the_hand_out(no_persistence):
    table = _passed_out_table("leasters")
    dealt = table.game

    assert await redeal_passed_out_hand(table) is False

    assert table.game is dealt
    assert table.game is not None and table.game.is_leaster
    assert table.score_multiplier == 1


async def test_a_table_with_no_all_pass_rule_plays_leasters(no_persistence):
    """Tables predating the rule keep the behaviour they were created with."""
    table = _passed_out_table("leasters")
    table.rules = {}

    assert await redeal_passed_out_hand(table) is False
    assert table.score_multiplier == 1


def test_record_hand_result_scales_scores_by_the_stake():
    table = _passed_out_table("doublers")
    table.score_multiplier = 4
    game = table.game
    assert game is not None
    while not game.is_done():
        for player in game.players:
            valid = sorted(player.get_valid_action_ids())
            if valid:
                assert player.act(valid[0])

    record_hand_result(table)

    entry = table.results_history[-1]
    assert entry["multiplier"] == 4
    for seat in range(1, 6):
        raw = int(game.players[seat - 1].get_score())
        assert entry["bySeat"][seat]["score"] == raw * 4
        assert table.running_scores[f"human-{seat}"] == raw * 4
