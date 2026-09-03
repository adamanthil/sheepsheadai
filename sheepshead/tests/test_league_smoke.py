#!/usr/bin/env python3
"""End-to-end smoke of train_ppo.run_phase in all three phases.

Runs the actual training loop (collection -> per-seat storage -> update)
for a handful of episodes: the bootstrap on an empty population with
shaped rewards, the league phase with the oracle critic and snapshots
that must not carry the privileged critic, and the bidding-only phase
whose frozen play path must stay bit-identical through an update.
"""

import glob
import os
import random
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.league import ROLE_PAST_MAIN, League
from sheepshead.training.train_ppo import PHASE_SPECS, run_phase

# Exercises the real training loop end to end (~20s).
pytestmark = pytest.mark.slow

NEVER = 1_000_000_000


def _args(phase: str, run_name: str, **overrides) -> SimpleNamespace:
    base = dict(
        phase=phase,
        seed=13,
        run_name=run_name,
        num_workers=1,
        update_interval=30,
        until=8,
        save_interval=NEVER,
        snapshot_interval=0,
        greedy_eval_interval=0,
        greedy_eval_games=0,
        leaster_watchdog=False,
        entropy_controller=False,
        entropy_play_floor=0.28,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRunPhase:
    def setup_method(self, method):
        self.dir = tempfile.mkdtemp(prefix="phase_smoke_")
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        os.makedirs(self.ckpt_dir)

    def teardown_method(self, method):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _agent(self, phase: str, arch: str = "perceiver-recall") -> PPOAgent:
        random.seed(3)
        torch.manual_seed(3)
        spec = PHASE_SPECS[phase]
        agent = PPOAgent(len(ACTIONS), critic_mode=spec.critic_mode, arch=arch)
        agent.gamma = spec.gamma
        agent.set_trainable_heads(spec.trainable_heads)
        return agent

    def _run(self, phase: str, agent: PPOAgent, league: League, **overrides):
        args = _args(phase, f"_smoke_{phase}", **overrides)
        ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
        return run_phase(agent, league, ratings, args, 0, args.until, self.ckpt_dir)

    def test_bootstrap_on_empty_population(self):
        league = League(os.path.join(self.dir, "league"))
        agent = self._agent("bootstrap")
        end = self._run("bootstrap", agent, league)
        assert end == 8
        assert len(agent.actor_optimizer.state) > 0  # a PPO update fired
        assert agent.oracle_critic is None
        assert len(league) == 0  # no snapshots in the bootstrap

    def test_league_phase_updates_and_strips_oracle_from_snapshots(self):
        league = League(os.path.join(self.dir, "league"))
        for i in range(2):
            torch.manual_seed(10 + i)
            league.add_member(
                PPOAgent(len(ACTIONS), arch="perceiver-recall"),
                ROLE_PAST_MAIN,
                training_episodes=i,
            )
        agent = self._agent("league")
        end = self._run("league", agent, league, snapshot_interval=5)
        assert end == 8
        assert len(agent.actor_optimizer.state) > 0
        assert len(agent.oracle_optimizer.state) > 0  # the oracle trained too
        snaps = [
            m for m in league.by_role(ROLE_PAST_MAIN) if m.meta.training_episodes == 5
        ]
        assert len(snaps) == 1
        (member_file,) = glob.glob(
            os.path.join(str(league.members_dir), f"{snaps[0].member_id}.pt")
        )
        ckpt = torch.load(member_file, map_location="cpu")
        assert "oracle_state_dict" not in ckpt
        assert ckpt.get("arch") == "perceiver-recall"

    def test_bidding_phase_freezes_the_play_path(self):
        league = League(os.path.join(self.dir, "league"))
        agent = self._agent("bidding")
        frozen_before = {
            k: v.detach().clone()
            for k, v in agent.actor.state_dict().items()
            if k.startswith(agent.PLAY_HEAD_PREFIXES)
        }
        encoder_before = {
            k: v.detach().clone() for k, v in agent.encoder.state_dict().items()
        }
        bidding_before = {
            k: v.detach().clone()
            for k, v in agent.actor.state_dict().items()
            if not k.startswith(agent.PLAY_HEAD_PREFIXES)
        }
        end = self._run("bidding", agent, league)
        assert end == 8
        assert len(agent.actor_optimizer.state) > 0
        after = agent.actor.state_dict()
        assert all(torch.equal(frozen_before[k], after[k]) for k in frozen_before)
        enc_after = agent.encoder.state_dict()
        assert all(torch.equal(encoder_before[k], enc_after[k]) for k in encoder_before)
        assert any(not torch.equal(bidding_before[k], after[k]) for k in bidding_before)
