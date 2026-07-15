#!/usr/bin/env python3
"""End-to-end smoke test for the league trainer's main phase.

Runs run_main_phase() on a tiny league for a handful of episodes in both
critic modes — the first test that exercises the actual training loop
(collection → store → update) rather than its pieces. Slow-ish (~1 min);
guards against wiring regressions like the oracle path breaking collection
or snapshots persisting the privileged critic.
"""

import glob
import os
import random
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import pytest
import torch

from sheepshead.training.league import ROLE_PAST_MAIN, League
from sheepshead.agent.ppo import PPOAgent
from sheepshead import ACTIONS
from sheepshead.training.train_league_ppo import run_main_phase

# Exercises the real training loop end to end (~15s).
pytestmark = pytest.mark.slow


class TestMainPhaseSmoke(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="league_smoke_")
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        os.makedirs(self.ckpt_dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(
        self, critic_mode: str, snapshot_interval: int, arch: str = "full"
    ) -> tuple:
        random.seed(3)
        torch.manual_seed(3)
        league = League(os.path.join(self.dir, f"league_{critic_mode}_{arch}"))
        for i in range(2):
            torch.manual_seed(10 + i)
            league.add_member(
                PPOAgent(len(ACTIONS), arch=arch), ROLE_PAST_MAIN, training_episodes=i
            )
        agent = PPOAgent(len(ACTIONS), critic_mode=critic_mode, arch=arch)
        never = 1_000_000_000
        args = SimpleNamespace(
            seed=13,
            run_name=f"_smoke_{critic_mode}_{arch}",
            critic_mode=critic_mode,
            arch=arch,
            num_workers=1,
            update_interval=30,
            schedule_horizon=never,
            save_interval=never,
            snapshot_interval=snapshot_interval,
            greedy_eval_interval=0,
            greedy_eval_games=0,
        )
        ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
        end = run_main_phase(
            agent,
            league,
            ratings,
            args,
            start_episode=0,
            n_episodes=8,
            checkpoint_dir=self.ckpt_dir,
        )
        return agent, league, end

    def test_limited_mode_runs_and_updates(self):
        agent, _, end = self._run("limited", snapshot_interval=1_000_000_000)
        self.assertEqual(end, 8)
        # >= 1 PPO update fired (Adam has per-param state after a step).
        self.assertGreater(len(agent.actor_optimizer.state), 0)
        self.assertIsNone(agent.oracle_critic)

    def test_oracle_mode_runs_updates_and_strips_snapshots(self):
        agent, league, end = self._run("oracle", snapshot_interval=5)
        self.assertEqual(end, 8)
        self.assertGreater(len(agent.actor_optimizer.state), 0)
        # The oracle itself trained too.
        self.assertGreater(len(agent.oracle_optimizer.state), 0)
        # The episode-5 snapshot joined the league WITHOUT the oracle critic.
        snaps = [
            m for m in league.by_role(ROLE_PAST_MAIN) if m.meta.training_episodes == 5
        ]
        self.assertEqual(len(snaps), 1)
        member_files = glob.glob(
            os.path.join(str(league.members_dir), f"{snaps[0].member_id}.pt")
        )
        self.assertEqual(len(member_files), 1)
        ckpt = torch.load(member_files[0], map_location="cpu")
        self.assertNotIn("oracle_state_dict", ckpt)
        self.assertIsNone(snaps[0].agent.oracle_critic)

    def test_nonfull_arch_runs_and_snapshots_carry_arch(self):
        agent, league, end = self._run("limited", snapshot_interval=5, arch="onehot-ff")
        self.assertEqual(end, 8)
        self.assertGreater(len(agent.actor_optimizer.state), 0)
        snaps = [
            m for m in league.by_role(ROLE_PAST_MAIN) if m.meta.training_episodes == 5
        ]
        self.assertEqual(len(snaps), 1)
        member_files = glob.glob(
            os.path.join(str(league.members_dir), f"{snaps[0].member_id}.pt")
        )
        ckpt = torch.load(member_files[0], map_location="cpu")
        self.assertEqual(ckpt.get("arch"), "onehot-ff")


if __name__ == "__main__":
    unittest.main(verbosity=2)
