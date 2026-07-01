#!/usr/bin/env python3
"""League roster / sampling / rating / migration tests (Exploiter_League_Plan §4)."""

import json
import os
import random
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import LeagueConfig
from league import (
    ROLE_HOF_ANCHOR,
    ROLE_MAIN_EXPLOITER,
    ROLE_PAST_MAIN,
    SELF_PLAY,
    League,
    LeagueMember,
)
from ppo import PPOAgent
from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD


def _agent(seed: int) -> PPOAgent:
    torch.manual_seed(seed)
    return PPOAgent(len(ACTIONS), activation="swish")


class TestLeagueRoster(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="league_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip_persistence(self):
        league = League(self.dir)
        mid = league.add_member(_agent(1), ROLE_PAST_MAIN, training_episodes=1000)
        xid = league.add_member(
            _agent(2),
            ROLE_MAIN_EXPLOITER,
            training_episodes=2000,
            generation=1,
            gate_edge=0.25,
            initial_ema=0.72,
        )
        m = league.get(mid)
        m.ratings[PARTNER_BY_JD] = league.rating_model.rating(mu=30.0, sigma=2.0)
        m.exploitation_win_rate_ema = 0.61
        m.exploitation_samples = 40
        league.save()

        reloaded = League(self.dir)
        self.assertEqual(len(reloaded), 2)
        m2 = reloaded.get(mid)
        self.assertEqual(m2.role, ROLE_PAST_MAIN)
        self.assertAlmostEqual(m2.ratings[PARTNER_BY_JD].mu, 30.0, places=5)
        self.assertAlmostEqual(m2.ratings[PARTNER_BY_CALLED_ACE].mu, 25.0, places=5)
        self.assertAlmostEqual(m2.exploitation_win_rate_ema, 0.61, places=6)
        self.assertEqual(m2.exploitation_samples, 40)
        x2 = reloaded.get(xid)
        self.assertEqual(x2.role, ROLE_MAIN_EXPLOITER)
        self.assertEqual(x2.meta.generation, 1)
        self.assertAlmostEqual(x2.meta.gate_edge, 0.25)
        self.assertAlmostEqual(x2.exploitation_win_rate_ema, 0.72, places=6)
        # Weights actually round-trip (not just metadata)
        p_orig = next(league.get(mid).agent.actor.parameters()).detach()
        p_load = next(m2.agent.actor.parameters()).detach()
        self.assertTrue(torch.allclose(p_orig, p_load))

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
        self.assertEqual(len(past), 4)
        surviving = {m.member_id for m in past}
        self.assertIn(ids[5], surviving)  # newest
        self.assertIn(ids[4], surviving)  # second newest
        self.assertIn(ids[0], surviving)  # highest skill
        self.assertEqual(len(league.by_role(ROLE_HOF_ANCHOR)), 1)  # untouched

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
        self.assertAlmostEqual(m.ratings[PARTNER_BY_JD].mu, -3.0, places=5)
        self.assertAlmostEqual(m.ratings[PARTNER_BY_CALLED_ACE].mu, -5.0, places=5)
        self.assertAlmostEqual(m.ratings[PARTNER_BY_CALLED_ACE].sigma, 4.0, places=5)

    def test_inherited_ratings_scale_and_sigma_floor(self):
        from train_league_ppo import _inherited_ratings

        league = League(self.dir)
        training_ratings = {
            PARTNER_BY_JD: league.rating_model.rating(mu=-3.2, sigma=0.4),
            PARTNER_BY_CALLED_ACE: league.rating_model.rating(mu=-4.8, sigma=9.0),
        }
        inherited = _inherited_ratings(league, training_ratings)
        default_sigma = league.rating_model.rating().sigma
        # mu carries over; a collapsed sigma is floored at half the prior so
        # the snapshot can still be re-rated as the field evolves.
        self.assertAlmostEqual(inherited[PARTNER_BY_JD].mu, -3.2, places=5)
        self.assertAlmostEqual(
            inherited[PARTNER_BY_JD].sigma, default_sigma / 2.0, places=5
        )
        self.assertAlmostEqual(inherited[PARTNER_BY_CALLED_ACE].sigma, 9.0, places=5)
        # Fresh objects, not aliases of the live training ratings.
        self.assertIsNot(inherited[PARTNER_BY_JD], training_ratings[PARTNER_BY_JD])

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
        self.assertEqual(len(league.by_role(ROLE_HOF_ANCHOR)), 2)
        self.assertEqual(league.get(ids[0]).role, ROLE_PAST_MAIN)
        self.assertEqual(league.get(ids[1]).role, ROLE_HOF_ANCHOR)
        self.assertEqual(league.get(ids[2]).role, ROLE_HOF_ANCHOR)
        with self.assertRaises(ValueError):
            league.promote_to_hof("nonexistent_member")
        reloaded = League(self.dir, cfg)
        self.assertEqual(len(reloaded.by_role(ROLE_HOF_ANCHOR)), 2)
        self.assertEqual(reloaded.get(ids[0]).role, ROLE_PAST_MAIN)

    def test_exploiter_retirement(self):
        # Retirement is purely age-based: a high EMA no longer saves an old
        # exploiter (that EMA-driven reprieve was the ratchet we removed).
        cfg = LeagueConfig(exploiter_retire_generations=2)
        league = League(self.dir, cfg)
        old_hot = league.add_member(
            _agent(1),
            ROLE_MAIN_EXPLOITER,
            training_episodes=0,
            generation=1,
            initial_ema=0.70,
        )
        young = league.add_member(
            _agent(3),
            ROLE_MAIN_EXPLOITER,
            training_episodes=0,
            generation=3,
            initial_ema=0.45,
        )
        self.assertEqual(league.get(old_hot).role, ROLE_PAST_MAIN)  # old -> retired
        self.assertEqual(league.get(young).role, ROLE_MAIN_EXPLOITER)  # still young

    def test_exploiter_retirement_by_generation_clock(self):
        # F5: the clock advances at every boundary (note_generation), so a
        # beaten exploiter retires after N elapsed generations even when no
        # later exploiter ever passes its gate (no insertion required).
        cfg = LeagueConfig(exploiter_retire_generations=2)
        league = League(self.dir, cfg)
        xid = league.add_member(
            _agent(1), ROLE_MAIN_EXPLOITER, training_episodes=0, generation=1
        )
        league.note_generation(2)
        self.assertEqual(league.get(xid).role, ROLE_MAIN_EXPLOITER)  # age 1
        league.note_generation(3)
        self.assertEqual(league.get(xid).role, ROLE_PAST_MAIN)  # age 2 -> retired
        # The clock persists across reloads and never runs backward.
        reloaded = League(self.dir, cfg)
        self.assertEqual(reloaded.current_generation, 3)
        reloaded.note_generation(1)
        self.assertEqual(reloaded.current_generation, 3)


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="league_test_")
        cfg = LeagueConfig(
            exploiter_seat_cap=0.30, exploiter_edge_full=0.30, self_play_share=0.15
        )
        self.league = League(self.dir, cfg)
        for i in range(6):
            self.league.add_member(_agent(i), ROLE_PAST_MAIN, training_episodes=i)
        self.league.add_member(_agent(50), ROLE_HOF_ANCHOR, training_episodes=0)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_no_exploiter_means_no_exploiter_seats(self):
        self.assertEqual(self.league.exploiter_share(), 0.0)
        rng = random.Random(0)
        seats = [
            s for _ in range(200) for s in self.league.sample_table(PARTNER_BY_JD, rng)
        ]
        self.assertFalse(
            any(
                isinstance(s, LeagueMember) and s.role == ROLE_MAIN_EXPLOITER
                for s in seats
            )
        )

    def test_mixture_shares_and_cap(self):
        # gate_edge 0.30 => 0.30/0.30 = 1.0 => full cap
        self.league.add_member(
            _agent(60),
            ROLE_MAIN_EXPLOITER,
            training_episodes=0,
            generation=1,
            gate_edge=0.30,
        )
        self.assertAlmostEqual(self.league.exploiter_share(), 0.30, places=6)
        rng = random.Random(1)
        n_tables = 1500
        seats = [
            s
            for _ in range(n_tables)
            for s in self.league.sample_table(PARTNER_BY_JD, rng)
        ]
        n = len(seats)
        exp_frac = (
            sum(
                1
                for s in seats
                if isinstance(s, LeagueMember) and s.role == ROLE_MAIN_EXPLOITER
            )
            / n
        )
        self_frac = sum(1 for s in seats if s == SELF_PLAY) / n
        # One exploiter per table max (sampled without replacement) => the
        # realized share is min(p_exp per-seat draws, 1 per table) — with one
        # exploiter the expected fraction is < 0.30/4*4 but bounded by 1/4.
        self.assertGreater(exp_frac, 0.10)
        self.assertLessEqual(exp_frac, 0.25 + 0.02)
        self.assertAlmostEqual(self_frac, 0.15, delta=0.03)

    def test_seat_share_survives_ema_collapse(self):
        # Regression: a passing exploiter whose binary table EMA decays below
        # neutral must keep its seat share, which is driven by the frozen
        # gate_edge, not the EMA (the ratchet that previously zeroed it out).
        mid = self.league.add_member(
            _agent(61),
            ROLE_MAIN_EXPLOITER,
            training_episodes=0,
            generation=1,
            gate_edge=0.15,
        )
        self.assertAlmostEqual(self.league.exploiter_share(), 0.15, places=6)
        self.league.get(mid).exploitation_win_rate_ema = 0.40  # tanked below neutral
        self.assertAlmostEqual(self.league.exploiter_share(), 0.15, places=6)

    def test_table_has_no_duplicate_members(self):
        rng = random.Random(2)
        for _ in range(300):
            table = self.league.sample_table(PARTNER_BY_JD, rng)
            ids = [s.member_id for s in table if isinstance(s, LeagueMember)]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(table), 4)


class TestRatings(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="league_test_")
        self.league = League(self.dir)
        self.ids = [
            self.league.add_member(_agent(i), ROLE_PAST_MAIN, training_episodes=i)
            for i in range(4)
        ]
        self.opps = {
            pos: self.league.get(mid) for pos, mid in zip([2, 3, 4, 5], self.ids)
        }

    def tearDown(self):
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
        self.assertLess(new_tr.mu, tr.mu)  # training lost
        for pos in (3, 4, 5):  # defenders won
            self.assertGreater(self.opps[pos].rating(PARTNER_BY_JD).mu, mu_before[pos])
            # ...and their exploitation EMA rose above neutral
            self.assertGreater(self.opps[pos].exploitation_win_rate_ema, 0.5)
        # The training agent's partner: rating moved with the picker team
        # (down), but exploitation EMA untouched (teammate result is not
        # evidence of exploiting the training agent).
        self.assertLess(self.opps[2].rating(PARTNER_BY_JD).mu, mu_before[2])
        self.assertEqual(self.opps[2].exploitation_win_rate_ema, 0.5)

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
            self.assertEqual(m.rating(PARTNER_BY_CALLED_ACE).mu, ca_before[pos])

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
        self.assertGreater(new_tr.mu, tr.mu)  # training won the leaster
        for m in self.opps.values():
            self.assertLess(m.exploitation_win_rate_ema, 0.5)


class TestMigration(unittest.TestCase):
    def setUp(self):
        self.old = tempfile.mkdtemp(prefix="legacy_pop_")
        self.new = tempfile.mkdtemp(prefix="league_mig_")
        for sub in ("jd_agents", "called_ace_agents"):
            os.makedirs(os.path.join(self.old, sub))

    def tearDown(self):
        shutil.rmtree(self.old, ignore_errors=True)
        shutil.rmtree(self.new, ignore_errors=True)

    def _write_legacy(self, sub, agent_id, agent, mode, episodes, mu, created):
        d = os.path.join(self.old, sub)
        agent.save(os.path.join(d, f"{agent_id}.pt"))
        with open(os.path.join(d, f"{agent_id}_metadata.json"), "w") as f:
            json.dump(
                {
                    "agent_id": agent_id,
                    "creation_time": created,
                    "parent_id": None,
                    "training_episodes": episodes,
                    "partner_mode": mode,
                    "activation": "swish",
                    "games_played": 10,
                    "total_score": 5.0,
                    "picker_games": 2,
                    "picker_score": 1.0,
                    "rating_mu": mu,
                    "rating_sigma": 3.0,
                    "exploitation_win_rate_ema": 0.55,
                    "exploitation_samples": 20,
                },
                f,
            )

    def test_migrate_dedups_twins_and_assigns_roles(self):
        t0 = time.time()
        # Three snapshots, each saved twice (JD + CA copies of the SAME
        # weights, as the old trainer did), with different per-mode ratings.
        for i in range(3):
            ag = _agent(i)
            self._write_legacy(
                "jd_agents",
                f"0_{1000 + i}_{i}",
                ag,
                0,
                (i + 1) * 1000,
                mu=20.0 + i * 5,
                created=t0 + i,
            )
            self._write_legacy(
                "called_ace_agents",
                f"1_{1000 + i}_{i}",
                ag,
                1,
                (i + 1) * 1000,
                mu=22.0 + i * 5,
                created=t0 + i,
            )
        cfg = LeagueConfig(hof_quota=1, protect_newest=1)
        league = League.migrate_legacy(self.old, self.new, cfg, keep_top_k=3)
        self.assertEqual(len(league), 3)  # twins merged, not 6
        # Per-mode ratings preserved from each twin
        strongest = max(league.members, key=lambda m: m.skill())
        self.assertAlmostEqual(strongest.ratings[PARTNER_BY_JD].mu, 30.0, places=4)
        self.assertAlmostEqual(
            strongest.ratings[PARTNER_BY_CALLED_ACE].mu, 32.0, places=4
        )
        self.assertEqual(strongest.role, ROLE_HOF_ANCHOR)
        self.assertEqual(len(league.by_role(ROLE_HOF_ANCHOR)), 1)
        # EMA carried over
        for m in league.members:
            self.assertAlmostEqual(m.exploitation_win_rate_ema, 0.55, places=6)
        # Round-trips through normal load
        reloaded = League(self.new)
        self.assertEqual(len(reloaded), 3)

    def test_migrate_keeps_top_k(self):
        t0 = time.time()
        for i in range(5):
            ag = _agent(i + 10)
            self._write_legacy(
                "jd_agents",
                f"0_{2000 + i}_{i}",
                ag,
                0,
                (i + 1) * 1000,
                mu=40.0 - i * 5,
                created=t0 + i,
            )
        cfg = LeagueConfig(hof_quota=1, protect_newest=1)
        league = League.migrate_legacy(self.old, self.new, cfg, keep_top_k=3)
        self.assertEqual(len(league), 3)
        episodes = sorted(m.meta.training_episodes for m in league.members)
        # Newest (5000, weakest) protected; then two strongest (1000, 2000)
        self.assertEqual(episodes, [1000, 2000, 5000])


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main(verbosity=2)
