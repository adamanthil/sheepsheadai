"""Tests for the league worker weight-publishing protocol.

publish_weights (train_league_ppo.py) writes versioned weight files that
_league_worker_play reloads mid-run; these two halves of the protocol have
no direct test coverage. Drives both sides in-process (no multiprocessing
Pool) — publish_weights via a minimal MainPhaseContext-like namespace, and
the worker reload path by seeding the module's _LWORKER global directly, as
_league_worker_init would.
"""

import os
import tempfile
from types import SimpleNamespace

import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.league import SELF_PLAY
from sheepshead.training.train_league_ppo import (
    _LWORKER,
    _Job,
    _league_worker_play,
    publish_weights,
)

ARCH = "onehot-ff"  # cheapest registered arch: keeps agent construction fast


def _make_ctx(run_dir: str, agent: PPOAgent) -> SimpleNamespace:
    """Minimal stand-in for MainPhaseContext: publish_weights only reads
    .training_agent and .weight_sync."""
    return SimpleNamespace(
        training_agent=agent,
        weight_sync={
            "version": 0,
            "base": os.path.join(run_dir, "_league_worker_weights"),
        },
    )


class TestPublishWeights:
    def setup_method(self, method):
        self.dir = tempfile.mkdtemp(prefix="publish_weights_")

    def teardown_method(self, method):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_writes_versioned_file_atomically_and_loadable(self):
        agent = PPOAgent(len(ACTIONS), arch=ARCH)
        ctx = _make_ctx(self.dir, agent)

        publish_weights(ctx)

        assert ctx.weight_sync["version"] == 1
        path = ctx.weight_sync["base"] + "_v1.pt"
        assert os.path.exists(path)
        # No leftover tmp file from the atomic-rename step.
        assert not os.path.exists(path + ".tmp")

        payload = torch.load(path, map_location="cpu")
        assert "encoder_state_dict" in payload
        assert "actor_state_dict" in payload
        assert "critic_state_dict" in payload
        assert payload["gamma"] == agent.gamma

    def test_second_publish_increments_version(self):
        agent = PPOAgent(len(ACTIONS), arch=ARCH)
        ctx = _make_ctx(self.dir, agent)

        publish_weights(ctx)
        publish_weights(ctx)

        assert ctx.weight_sync["version"] == 2
        assert os.path.exists(ctx.weight_sync["base"] + "_v1.pt")
        assert os.path.exists(ctx.weight_sync["base"] + "_v2.pt")

    def test_old_version_gc_deletes_v_minus_2(self):
        agent = PPOAgent(len(ACTIONS), arch=ARCH)
        ctx = _make_ctx(self.dir, agent)
        base = ctx.weight_sync["base"]

        publish_weights(ctx)  # v1
        publish_weights(ctx)  # v2 (no v-2 yet: v0 never existed as a file)
        publish_weights(ctx)  # v3: should GC v1

        assert not os.path.exists(base + "_v1.pt")
        assert os.path.exists(base + "_v2.pt")
        assert os.path.exists(base + "_v3.pt")

        publish_weights(ctx)  # v4: should GC v2

        assert not os.path.exists(base + "_v2.pt")
        assert os.path.exists(base + "_v3.pt")
        assert os.path.exists(base + "_v4.pt")

    def test_gc_is_a_noop_when_stale_file_absent(self):
        # Retention-window edge case: v1's GC target (v-1) never existed, so
        # publish_weights must not raise when os.remove would fail.
        agent = PPOAgent(len(ACTIONS), arch=ARCH)
        ctx = _make_ctx(self.dir, agent)

        publish_weights(ctx)  # v1, target v-1: absent, should be silently skipped

        assert os.path.exists(ctx.weight_sync["base"] + "_v1.pt")


class TestWorkerReload:
    def setup_method(self, method):
        self.dir = tempfile.mkdtemp(prefix="publish_weights_worker_")
        _LWORKER.clear()

    def teardown_method(self, method):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)
        _LWORKER.clear()

    def _seed_worker(self, agent: PPOAgent, version: int, base: str) -> None:
        # Mirrors _league_worker_init's _LWORKER contract without spinning
        # up a Pool.
        _LWORKER.update(
            {
                "agent": agent,
                "members_dir": self.dir,
                "weight_path_base": base,
                "version": version,
                "cache": {},
            }
        )

    def test_reloads_when_job_version_is_newer(self):
        torch.manual_seed(1)
        publisher = PPOAgent(len(ACTIONS), arch=ARCH)
        ctx = _make_ctx(self.dir, publisher)
        publish_weights(ctx)  # v1: publisher's initial (random) weights

        # Mutate the publisher's weights so v2 is distinguishable from v1.
        with torch.no_grad():
            for p in publisher.actor.parameters():
                p.add_(1.0)
        publish_weights(ctx)  # v2: mutated weights

        torch.manual_seed(2)  # different init from publisher, so reload is observable
        worker_agent = PPOAgent(len(ACTIONS), arch=ARCH)
        before = {k: v.clone() for k, v in worker_agent.actor.state_dict().items()}
        self._seed_worker(worker_agent, version=1, base=ctx.weight_sync["base"])

        job = _Job(
            episode=1,
            partner_mode=0,
            training_position=1,
            opponent_ids=[SELF_PLAY, SELF_PLAY, SELF_PLAY, SELF_PLAY],
            weight_version=2,
        )

        result = _league_worker_play(job)

        assert _LWORKER["version"] == 2
        after = worker_agent.actor.state_dict()
        # At least one tensor changed: the reload actually copied new state.
        assert any(not torch.equal(before[k], after[k]) for k in before)
        # Reloaded weights match what was published as v2.
        expected = publisher.actor.state_dict()
        for k in expected:
            assert torch.equal(after[k], expected[k])
        assert result["episode"] == 1

    def test_no_reload_when_job_version_matches_current(self):
        torch.manual_seed(3)
        worker_agent = PPOAgent(len(ACTIONS), arch=ARCH)
        before = {k: v.clone() for k, v in worker_agent.actor.state_dict().items()}
        # weight_path_base points nowhere: if a reload were attempted it
        # would raise (no such file), proving the guard held.
        self._seed_worker(
            worker_agent, version=5, base=os.path.join(self.dir, "does_not_exist")
        )

        job = _Job(
            episode=7,
            partner_mode=0,
            training_position=1,
            opponent_ids=[SELF_PLAY, SELF_PLAY, SELF_PLAY, SELF_PLAY],
            weight_version=5,
        )

        result = _league_worker_play(job)

        assert _LWORKER["version"] == 5
        after = worker_agent.actor.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert result["episode"] == 7

    def test_no_reload_when_job_version_is_older(self):
        torch.manual_seed(4)
        worker_agent = PPOAgent(len(ACTIONS), arch=ARCH)
        before = {k: v.clone() for k, v in worker_agent.actor.state_dict().items()}
        self._seed_worker(
            worker_agent, version=5, base=os.path.join(self.dir, "does_not_exist")
        )

        job = _Job(
            episode=8,
            partner_mode=0,
            training_position=1,
            opponent_ids=[SELF_PLAY, SELF_PLAY, SELF_PLAY, SELF_PLAY],
            weight_version=3,  # older than current worker version (5)
        )

        _league_worker_play(job)

        assert _LWORKER["version"] == 5  # unchanged
        after = worker_agent.actor.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
