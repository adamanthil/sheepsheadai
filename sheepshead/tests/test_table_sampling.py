#!/usr/bin/env python3
"""Unit tests for league table sampling: the per-seat PFSP/self/exploiter
mixture (the only sampling mode — the table-level composition and
whole-table exploiter variants were stripped 2026-07-28) plus
exploit-patched retirement."""

import random

import pytest

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestExploiterSeating:
    """Per-seat exploiter mixing + exploit-patched retirement."""

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

    def test_per_seat_exploiter_mixing(self, league):
        # Exploiters enter tables per-seat within the PFSP mixture — the
        # only exploiter seating mode (whole-table pressure removed
        # 2026-07-28: it deployed an uncertified 1-main-vs-4-copies mirror
        # with no evaluation analog, and seat rotation supplies the role
        # coverage it compensated for).
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
