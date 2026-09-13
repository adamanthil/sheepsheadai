#!/usr/bin/env python3
"""League roster / sampling / rating tests."""

import random
import shutil
import tempfile

import pytest
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.config import LeagueConfig
from sheepshead.training.league import (
    ROLE_HOF_ANCHOR,
    ROLE_PAST_MAIN,
    SELF_PLAY,
    League,
    LeagueMember,
)


def _agent(seed: int) -> PPOAgent:
    torch.manual_seed(seed)
    return PPOAgent(len(ACTIONS))


def _member(league: League, member_id: str) -> LeagueMember:
    """League.get is Optional; every id passed here came from add_member."""
    member = league.get(member_id)
    assert member is not None, f"member {member_id} missing from the league"
    return member


class TestLeagueRoster:
    def setup_method(self, method):
        self.dir = tempfile.mkdtemp(prefix="league_test_")

    def teardown_method(self, method):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip_persistence(self):
        league = League(self.dir)
        mid = league.add_member(_agent(1), ROLE_PAST_MAIN, training_episodes=1000)
        xid = league.add_member(_agent(2), ROLE_HOF_ANCHOR, training_episodes=2000)
        m = league.get(mid)
        m.ratings[PARTNER_BY_JD] = league.rating_model.rating(mu=30.0, sigma=2.0)
        m.exploitation_win_rate_ema = 0.61
        m.exploitation_samples = 40
        league.save()

        reloaded = League(self.dir)
        assert len(reloaded) == 2
        m2 = reloaded.get(mid)
        assert m2.role == ROLE_PAST_MAIN
        assert m2.ratings[PARTNER_BY_JD].mu == pytest.approx(30.0, abs=10**-5)
        assert m2.ratings[PARTNER_BY_CALLED_ACE].mu == pytest.approx(25.0, abs=10**-5)
        assert m2.exploitation_win_rate_ema == pytest.approx(0.61, abs=10**-6)
        assert m2.exploitation_samples == 40
        x2 = reloaded.get(xid)
        assert x2.role == ROLE_HOF_ANCHOR
        # Weights actually round-trip (not just metadata)
        p_orig = next(league.get(mid).agent.actor.parameters()).detach()
        p_load = next(m2.agent.actor.parameters()).detach()
        assert torch.allclose(p_orig, p_load)

    def test_prune_protects_newest_and_hof(self):
        cfg = LeagueConfig(max_past_mains=4, protect_newest=2, hof_quota=1)
        league = League(self.dir, cfg)
        league.add_member(_agent(99), ROLE_HOF_ANCHOR, training_episodes=0)
        ids = []
        for i in range(6):
            mid = league.add_member(
                _agent(i), ROLE_PAST_MAIN, training_episodes=(i + 1) * 1000
            )
            # Older members get higher skill so pruning pressure targets the
            # newest — protection must override skill.
            league.get(mid).ratings[PARTNER_BY_JD] = league.rating_model.rating(
                mu=50.0 - i * 5, sigma=1.0
            )
            ids.append(mid)
        past = league.by_role(ROLE_PAST_MAIN)
        assert len(past) == 4
        surviving = {m.member_id for m in past}
        assert ids[5] in surviving  # newest
        assert ids[4] in surviving  # second newest
        assert ids[0] in surviving  # highest skill
        assert len(league.by_role(ROLE_HOF_ANCHOR)) == 1  # untouched

    def test_initial_ratings_respected(self):
        # Snapshot rating inheritance (run-review F1): entries must be able to
        # join on the drifted population scale, not the mu=25 prior.
        league = League(self.dir)
        ratings = {
            PARTNER_BY_JD: league.rating_model.rating(mu=-3.0, sigma=4.0),
            PARTNER_BY_CALLED_ACE: league.rating_model.rating(mu=-5.0, sigma=4.0),
        }
        mid = league.add_member(
            _agent(7), ROLE_PAST_MAIN, training_episodes=500, initial_ratings=ratings
        )
        reloaded = League(self.dir)
        m = reloaded.get(mid)
        assert m.ratings[PARTNER_BY_JD].mu == pytest.approx(-3.0, abs=10**-5)
        assert m.ratings[PARTNER_BY_CALLED_ACE].mu == pytest.approx(-5.0, abs=10**-5)
        assert m.ratings[PARTNER_BY_CALLED_ACE].sigma == pytest.approx(4.0, abs=10**-5)

    def test_inherited_ratings_scale_and_sigma_floor(self):
        from sheepshead.training.train_ppo import inherited_ratings

        league = League(self.dir)
        training_ratings = {
            PARTNER_BY_JD: league.rating_model.rating(mu=-3.2, sigma=0.4),
            PARTNER_BY_CALLED_ACE: league.rating_model.rating(mu=-4.8, sigma=9.0),
        }
        inherited = inherited_ratings(league, training_ratings)
        default_sigma = league.rating_model.rating().sigma
        # mu carries over; a collapsed sigma is floored at half the prior so
        # the snapshot can still be re-rated as the field evolves.
        assert inherited[PARTNER_BY_JD].mu == pytest.approx(-3.2, abs=10**-5)
        assert inherited[PARTNER_BY_JD].sigma == pytest.approx(
            default_sigma / 2.0, abs=10**-5
        )
        assert inherited[PARTNER_BY_CALLED_ACE].sigma == pytest.approx(9.0, abs=10**-5)
        # Fresh objects, not aliases of the live training ratings.
        assert inherited[PARTNER_BY_JD] is not training_ratings[PARTNER_BY_JD]

    def test_promote_to_hof_quota_and_persistence(self):
        cfg = LeagueConfig(hof_quota=2, max_past_mains=10)
        league = League(self.dir, cfg)
        ids = []
        for i in range(3):
            mid = league.add_member(
                _agent(20 + i), ROLE_PAST_MAIN, training_episodes=(i + 1) * 100
            )
            league.get(mid).ratings[PARTNER_BY_JD] = league.rating_model.rating(
                mu=10.0 * i, sigma=1.0
            )
            ids.append(mid)
        for mid in ids:
            league.promote_to_hof(mid)
        # Quota enforced by demoting the lowest-skill anchor back to past_main.
        assert len(league.by_role(ROLE_HOF_ANCHOR)) == 2
        assert league.get(ids[0]).role == ROLE_PAST_MAIN
        assert league.get(ids[1]).role == ROLE_HOF_ANCHOR
        assert league.get(ids[2]).role == ROLE_HOF_ANCHOR
        with pytest.raises(ValueError):
            league.promote_to_hof("nonexistent_member")
        reloaded = League(self.dir, cfg)
        assert len(reloaded.by_role(ROLE_HOF_ANCHOR)) == 2
        assert reloaded.get(ids[0]).role == ROLE_PAST_MAIN


class TestSampling:
    def setup_method(self, method):
        self.dir = tempfile.mkdtemp(prefix="league_test_")
        self.league = League(self.dir, LeagueConfig(self_play_share=0.15))
        for i in range(6):
            self.league.add_member(_agent(i), ROLE_PAST_MAIN, training_episodes=i)
        self.league.add_member(_agent(50), ROLE_HOF_ANCHOR, training_episodes=0)

    def teardown_method(self, method):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_mixture_shares(self):
        rng = random.Random(1)
        seats = [
            s for _ in range(1500) for s in self.league.sample_table(PARTNER_BY_JD, rng)
        ]
        self_frac = sum(1 for s in seats if s == SELF_PLAY) / len(seats)
        assert self_frac == pytest.approx(0.15, abs=0.03)
        assert all(
            s == SELF_PLAY or s.role in (ROLE_PAST_MAIN, ROLE_HOF_ANCHOR) for s in seats
        )

    def test_hof_floor_draws_the_anchor(self):
        rng = random.Random(3)
        seats = [
            s for _ in range(1500) for s in self.league.sample_table(PARTNER_BY_JD, rng)
        ]
        hof_frac = sum(
            1
            for s in seats
            if isinstance(s, LeagueMember) and s.role == ROLE_HOF_ANCHOR
        ) / len(seats)
        # Forced floor (0.05 of PFSP seats) plus its ordinary PFSP share.
        assert hof_frac > 0.05 * 0.85

    def test_empty_league_is_pure_self_play(self):
        empty = League(tempfile.mkdtemp(prefix="league_empty_"))
        rng = random.Random(4)
        for _ in range(50):
            assert empty.sample_table(PARTNER_BY_JD, rng) == [SELF_PLAY] * 4

    def test_table_has_no_duplicate_members(self):
        rng = random.Random(2)
        for _ in range(300):
            table = self.league.sample_table(PARTNER_BY_JD, rng)
            ids = [s.member_id for s in table if isinstance(s, LeagueMember)]
            assert len(ids) == len(set(ids))
            assert len(table) == 4


class TestRatings:
    def setup_method(self, method):
        self.dir = tempfile.mkdtemp(prefix="league_test_")
        self.league = League(self.dir)
        self.ids = [
            self.league.add_member(_agent(i), ROLE_PAST_MAIN, training_episodes=i)
            for i in range(4)
        ]
        self.opps = {
            pos: _member(self.league, mid) for pos, mid in zip([2, 3, 4, 5], self.ids)
        }

    def teardown_method(self, method):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_team_update_and_ema_direction(self):
        tr = self.league.rating_model.rating()
        mu_before = {pos: m.rating(PARTNER_BY_JD).mu for pos, m in self.opps.items()}
        # Training in seat 1 is the picker (with partner seat 2) and LOSES.
        new_tr = self.league.update_ratings_with_training(
            partner_mode=PARTNER_BY_JD,
            training_rating=tr,
            final_scores=[-6.0, -6.0, 4.0, 4.0, 4.0],
            training_position=1,
            opponents_by_position=self.opps,
            picker_seat=1,
            partner_seat=2,
            is_leaster=False,
        )
        assert new_tr.mu < tr.mu  # training lost
        for pos in (3, 4, 5):  # defenders won
            assert self.opps[pos].rating(PARTNER_BY_JD).mu > mu_before[pos]
            # ...and their exploitation EMA rose above neutral
            assert self.opps[pos].exploitation_win_rate_ema > 0.5
        # The training agent's partner: rating moved with the picker team
        # (down), but exploitation EMA untouched (teammate result is not
        # evidence of exploiting the training agent).
        assert self.opps[2].rating(PARTNER_BY_JD).mu < mu_before[2]
        assert self.opps[2].exploitation_win_rate_ema == 0.5

    def test_mode_isolation(self):
        tr = self.league.rating_model.rating()
        ca_before = {
            pos: m.rating(PARTNER_BY_CALLED_ACE).mu for pos, m in self.opps.items()
        }
        self.league.update_ratings_with_training(
            partner_mode=PARTNER_BY_JD,
            training_rating=tr,
            final_scores=[2.0, -3.0, 2.0, -3.0, 2.0],
            training_position=1,
            opponents_by_position=self.opps,
            picker_seat=2,
            partner_seat=4,
            is_leaster=False,
        )
        for pos, m in self.opps.items():
            assert m.rating(PARTNER_BY_CALLED_ACE).mu == ca_before[pos]

    def test_leaster_free_for_all(self):
        tr = self.league.rating_model.rating()
        new_tr = self.league.update_ratings_with_training(
            partner_mode=PARTNER_BY_JD,
            training_rating=tr,
            final_scores=[4.0, -1.0, -1.0, -1.0, -1.0],
            training_position=1,
            opponents_by_position=self.opps,
            picker_seat=None,
            partner_seat=None,
            is_leaster=True,
        )
        assert new_tr.mu > tr.mu  # training won the leaster
        for m in self.opps.values():
            assert m.exploitation_win_rate_ema < 0.5
