#!/usr/bin/env python3
"""Refactor guards for the two episode-stream generators in
train_league_ppo.py (sequential_stream / parallel_stream):

  1. parallel_stream's dispatch-window sizing formula (avg_tx_per_game=26.0,
     capped at 256, floored at num_workers).
  2. Seat-rotation (CRN duplicate-bridge) grouping in parallel_stream: groups
     of 5 consecutive episodes share one game_seed and enumerate the 5 seat
     rotations 1..5; a fresh group draws a fresh seed. The group boundary
     must also survive a window split (parallel_stream carries rot_state on
     ctx across calls precisely for this).
  3. Structural equivalence of the grouping between sequential_stream and
     parallel_stream. The two paths draw their per-group seed from different
     RNG calls in a different order (parallel's window loop interleaves
     table sampling with pool dispatch), so exact seed equality between the
     two generators is NOT pinned here -- only the shared structural
     invariant (groups of 5, one constant seed per group, rotation
     1,2,3,4,5) is asserted for both.

These are pure refactor guards: nothing here should change behavior, only
catch it changing.
"""

import random
from types import SimpleNamespace
from typing import cast

import pytest

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import league_streams
from sheepshead.training.league import SELF_PLAY, League
from sheepshead.training.league_streams import (
    MainPhaseContext,
    TransitionCounter,
    parallel_stream,
    sequential_stream,
)
from sheepshead.training.league_worker import WorkerJob

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class _FakeLeague:
    """sample_table stub: always returns an all-self-play table, so
    stream tests never need real LeagueMember checkpoints."""

    def sample_table(self, partner_mode, rng, n_seats=4):
        return [SELF_PLAY] * n_seats


class _FakePool:
    """Records each window's job list; imap() fabricates the minimal
    result dict league_worker_play would return, without touching the
    real game engine."""

    def __init__(self):
        self.calls: list[list[WorkerJob]] = []

    def imap(self, fn, jobs):
        jobs = list(jobs)
        self.calls.append(jobs)
        for j in jobs:
            yield {
                "episode": j.episode,
                "partner_mode": j.partner_mode,
                "training_position": j.training_position,
                "episode_events": [],
                "final_scores": {},
                "training_data_single": None,
                "game_summary": {},
                "seat_to_member_id": {},
            }


def _make_ctx(tmp_path, args, start_episode, end_episode, training_agent=None):
    return MainPhaseContext(
        training_agent=cast(PPOAgent, training_agent),
        league=cast(League, _FakeLeague()),
        rng=random.Random(7),
        args=args,
        collect_oracle=False,
        weight_sync={"version": 0, "base": str(tmp_path / "weights")},
        tx_counter=TransitionCounter(),
        start_episode=start_episode,
        end_episode=end_episode,
    )


@pytest.fixture(scope="module")
def tiny_agent():
    # Real PPOAgent needed only because parallel_stream's publish_weights()
    # torch.save's the encoder/actor/critic state dicts; the lightest arch
    # keeps construction cheap. Shared read-only across tests in this file.
    return PPOAgent(len(ACTIONS), arch="onehot-ff")


class TestWindowSizing:
    """Pin parallel_stream's dispatch-window formula (lines ~599-603):
    window = max(num_workers, min(256, int(remaining_tx / 26.0) + 1))
    remaining_tx = max(1, update_interval - tx_counter.count)
    """

    def test_window_size_matches_formula_and_shrinks_near_update(
        self, tmp_path, tiny_agent
    ):
        args = SimpleNamespace(update_interval=130)
        ctx = _make_ctx(
            tmp_path, args, start_episode=0, end_episode=100, training_agent=tiny_agent
        )
        pool = _FakePool()
        gen = parallel_stream(ctx, pool, num_workers=2)

        # remaining_tx = max(1, 130 - 0) = 130; int(130/26)+1 = 6; window=max(2,6)=6.
        first_batch = [next(gen) for _ in range(6)]
        assert len(first_batch) == 6
        assert len(pool.calls) == 1
        assert [j.episode for j in pool.calls[0]] == list(range(1, 7))

        # Simulate a PPO update firing between windows (what run_main_phase
        # does): tx_counter climbs close to update_interval, so the next
        # window should shrink to just cover the remainder.
        ctx.tx_counter.count = 100
        # remaining_tx = max(1, 130-100)=30; int(30/26)+1=2; window=max(2,2)=2.
        second_batch = [next(gen) for _ in range(2)]
        assert len(second_batch) == 2
        assert len(pool.calls) == 2
        assert [j.episode for j in pool.calls[1]] == [7, 8]

    def test_window_capped_at_256(self, tmp_path, tiny_agent):
        args = SimpleNamespace(update_interval=1_000_000)
        ctx = _make_ctx(
            tmp_path, args, start_episode=0, end_episode=1000, training_agent=tiny_agent
        )
        pool = _FakePool()
        gen = parallel_stream(ctx, pool, num_workers=2)
        batch = [next(gen) for _ in range(256)]
        assert len(batch) == 256
        assert len(pool.calls[0]) == 256

    def test_window_floored_at_num_workers(self, tmp_path, tiny_agent):
        # remaining_tx = max(1, 26-0)=26; int(26/26)+1=2; min(256,2)=2.
        # num_workers=300 exceeds that -> window=max(300,2)=300, clipped to
        # the episodes actually remaining (end_episode=50).
        args = SimpleNamespace(update_interval=26)
        ctx = _make_ctx(
            tmp_path, args, start_episode=0, end_episode=50, training_agent=tiny_agent
        )
        pool = _FakePool()
        gen = parallel_stream(ctx, pool, num_workers=300)
        batch = list(gen)
        assert len(batch) == 50
        assert len(pool.calls) == 1
        assert len(pool.calls[0]) == 50


class TestSeatRotationGrouping:
    """parallel_stream's 5-episode seat-rotation grouping (CRN duplicate
    bridge): a group's game_seed and (mode, table) are fixed at the group's
    first episode and reused for rotations 2-5; a new group draws a fresh
    seed. Also checks the group survives a window boundary cut through it."""

    def _grouped_jobs(
        self, tmp_path, tiny_agent, end_episode, num_workers=1, update_interval=100_000
    ):
        args = SimpleNamespace(update_interval=update_interval, seat_rotation=True)
        ctx = _make_ctx(
            tmp_path,
            args,
            start_episode=0,
            end_episode=end_episode,
            training_agent=tiny_agent,
        )
        pool = _FakePool()
        list(parallel_stream(ctx, pool, num_workers=num_workers))
        jobs = [j for call in pool.calls for j in call]
        return jobs

    def test_groups_of_five_share_seed_and_enumerate_rotations(
        self, tmp_path, tiny_agent
    ):
        jobs = self._grouped_jobs(tmp_path, tiny_agent, end_episode=15)
        assert [j.episode for j in jobs] == list(range(1, 16))
        assert len(jobs) == 15
        groups = [jobs[i : i + 5] for i in range(0, 15, 5)]
        seeds = []
        for group in groups:
            group_seeds = {j.game_seed for j in group}
            assert len(group_seeds) == 1, "all 5 rotations share one deal seed"
            assert group_seeds.pop() is not None
            assert [j.training_position for j in group] == [1, 2, 3, 4, 5]
            seeds.append(group[0].game_seed)
        # Fresh seed per group -- collision probability over a 2**31 draw
        # is negligible, so inequality is a safe structural pin.
        assert len(set(seeds)) == len(seeds)

    def test_group_boundary_survives_a_window_split(self, tmp_path, tiny_agent):
        # update_interval sized so the first window (of 6 episodes) cuts
        # through the middle of the second 5-episode group (episodes 6-10).
        jobs = self._grouped_jobs(
            tmp_path, tiny_agent, end_episode=10, num_workers=1, update_interval=130
        )
        assert len(jobs) == 10
        second_group = jobs[5:10]
        assert {j.game_seed for j in second_group} == {second_group[0].game_seed}
        assert [j.training_position for j in second_group] == [1, 2, 3, 4, 5]
        first_group = jobs[0:5]
        assert first_group[0].game_seed != second_group[0].game_seed


class TestSequentialParallelGroupingEquivalence:
    """Both streams implement the same 5-episode grouping; pin the shared
    structural invariant (see module docstring for why exact seeds are not
    compared across the two paths)."""

    def test_sequential_grouping_matches_parallel_structure(
        self, tmp_path, tiny_agent, monkeypatch
    ):
        calls = []

        def fake_play_population_game(**kwargs):
            calls.append((kwargs["training_agent_position"], kwargs["game_seed"]))
            seat = SimpleNamespace(metadata=SimpleNamespace(agent_id="self"))
            game = object()
            return game, [], {}, None, {1: seat}

        monkeypatch.setattr(
            league_streams, "play_population_game", fake_play_population_game
        )
        monkeypatch.setattr(league_streams, "make_game_summary", lambda game: {})

        args = SimpleNamespace(seat_rotation=True)
        ctx = _make_ctx(
            tmp_path, args, start_episode=0, end_episode=15, training_agent=None
        )
        list(sequential_stream(ctx))

        assert len(calls) == 15
        groups = [calls[i : i + 5] for i in range(0, 15, 5)]
        seeds = []
        for group in groups:
            positions = [pos for pos, _ in group]
            group_seeds = {seed for _, seed in group}
            assert positions == [1, 2, 3, 4, 5]
            assert len(group_seeds) == 1
            seeds.append(group_seeds.pop())
        assert len(set(seeds)) == len(seeds)

        # Same structural shape as parallel_stream's grouping (separately
        # pinned in TestSeatRotationGrouping): groups of 5, one seed per
        # group, rotation 1..5. Cross-checked here on identical args/range.
        pargs = SimpleNamespace(update_interval=100_000, seat_rotation=True)
        pctx = _make_ctx(
            tmp_path, pargs, start_episode=0, end_episode=15, training_agent=tiny_agent
        )
        pool = _FakePool()
        list(parallel_stream(pctx, pool, num_workers=1))
        pjobs = [j for call in pool.calls for j in call]
        pgroups = [pjobs[i : i + 5] for i in range(0, 15, 5)]
        for sgroup, pgroup in zip(groups, pgroups):
            assert [pos for pos, _ in sgroup] == [j.training_position for j in pgroup]
            assert len({seed for _, seed in sgroup}) == 1
            assert len({j.game_seed for j in pgroup}) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
