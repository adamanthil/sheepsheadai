#!/usr/bin/env python3
"""Unit tests for table-level league sampling
(Learning_System_Redesign_202607: table_self_play_prob)."""

import random

import pytest

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestTableLevelSampling:
    @pytest.fixture()
    def league(self, tmp_path):
        from sheepshead.training.config import LeagueConfig
        from sheepshead.training.league import (
            ROLE_HOF_ANCHOR,
            ROLE_MAIN_EXPLOITER,
            ROLE_PAST_MAIN,
            League,
        )

        cfg = LeagueConfig()
        cfg.table_self_play_prob = 0.0
        lg = League(str(tmp_path), cfg)
        agent = PPOAgent(len(ACTIONS))
        for i in range(4):
            lg.add_member(agent, ROLE_PAST_MAIN, training_episodes=1000 * (i + 1))
        lg.add_member(agent, ROLE_HOF_ANCHOR, training_episodes=9000)
        lg.add_member(
            agent,
            ROLE_MAIN_EXPLOITER,
            training_episodes=5000,
            generation=1,
            gate_edge=0.3,
            initial_ema=0.6,
        )
        return lg

    def test_prob_one_gives_pure_self_table(self, league):
        from sheepshead.training.league import SELF_PLAY

        league.config.table_self_play_prob = 1.0
        seats = league.sample_table(0, random.Random(7))
        assert seats == [SELF_PLAY] * 4

    def test_prob_zero_seats_window_members_never_exploiters(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER, SELF_PLAY

        league.config.table_self_play_prob = 0.0
        rng = random.Random(11)
        for _ in range(50):
            seats = league.sample_table(0, rng)
            assert len(seats) == 4
            ids = set()
            for s in seats:
                assert s is not SELF_PLAY
                assert s.role != ROLE_MAIN_EXPLOITER
                ids.add(s.member_id)
            assert len(ids) == 4  # without replacement

    def test_historical_sampling_untouched_when_none(self, league):
        league.config.table_self_play_prob = None
        seats = league.sample_table(0, random.Random(3))
        assert len(seats) == 4


class TestExploiterFullTable:
    """Whole-table exploiter pressure + exploit-patched retirement
    (Learning_System_Redesign batch-λ arm amendment, 2026-07-21)."""

    @pytest.fixture()
    def league(self, tmp_path):
        from sheepshead.training.config import LeagueConfig
        from sheepshead.training.league import (
            ROLE_MAIN_EXPLOITER,
            ROLE_PAST_MAIN,
            League,
        )

        cfg = LeagueConfig()
        lg = League(str(tmp_path), cfg)
        agent = PPOAgent(len(ACTIONS))
        for i in range(4):
            lg.add_member(agent, ROLE_PAST_MAIN, training_episodes=1000 * (i + 1))
        lg.add_member(
            agent,
            ROLE_MAIN_EXPLOITER,
            training_episodes=5000,
            generation=1,
            gate_edge=0.30,  # == exploiter_edge_full -> full seat cap
            initial_ema=0.6,
        )
        return lg

    def test_full_table_is_one_exploiter_in_every_seat(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER

        league.config.exploiter_full_table = True
        league.config.exploiter_seat_cap = 1.0  # share -> 1.0, table certain
        seats = league.sample_table(0, random.Random(5))
        assert len(seats) == 4
        assert len({s.member_id for s in seats}) == 1
        assert seats[0].role == ROLE_MAIN_EXPLOITER

    def test_full_table_mode_never_seats_exploiters_per_seat(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER, SELF_PLAY

        league.config.exploiter_full_table = True
        league.config.exploiter_seat_cap = 0.0  # share 0: no exploiter tables
        rng = random.Random(11)
        for _ in range(50):
            for s in league.sample_table(0, rng):
                if s is not SELF_PLAY:
                    assert s.role != ROLE_MAIN_EXPLOITER

    def test_flag_off_keeps_per_seat_mixing(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER, SELF_PLAY

        league.config.exploiter_seat_cap = 1.0  # per-seat exploiter draws hot
        rng = random.Random(7)
        roles = set()
        for _ in range(50):
            for s in league.sample_table(0, rng):
                if s is not SELF_PLAY:
                    roles.add(s.role)
        assert ROLE_MAIN_EXPLOITER in roles

    def test_patched_retirement_demotes_on_collapsed_ema(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER, ROLE_PAST_MAIN

        (exploiter,) = league.by_role(ROLE_MAIN_EXPLOITER)
        exploiter.exploitation_win_rate_ema = 0.2
        exploiter.exploitation_samples = 500

        assert league.retire_patched_exploiters() == []  # disabled by default

        league.config.exploiter_patched_ema = 0.35
        assert league.retire_patched_exploiters() == [exploiter.member_id]
        assert exploiter.role == ROLE_PAST_MAIN

    def test_patched_retirement_needs_samples_and_low_ema(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER

        league.config.exploiter_patched_ema = 0.35
        (exploiter,) = league.by_role(ROLE_MAIN_EXPLOITER)

        exploiter.exploitation_win_rate_ema = 0.2
        exploiter.exploitation_samples = 10  # below min samples
        assert league.retire_patched_exploiters() == []

        exploiter.exploitation_samples = 500
        exploiter.exploitation_win_rate_ema = 0.6  # exploit still winning
        assert league.retire_patched_exploiters() == []
        assert exploiter.role == ROLE_MAIN_EXPLOITER
