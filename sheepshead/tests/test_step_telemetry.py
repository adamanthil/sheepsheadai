"""Tests for the gen-2 boundary telemetry package: the lifetime
optimizer-step counter, the gradient-noise-scale diagnostic, oracle-only
extra epochs, and the append-only CSV schema migration."""

import csv
import os

import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import play_episodes, seed_all
from sheepshead.training.training_utils import ensure_csv_columns

SEED = 20260724
ARCH = "perceiver-shared-v2"


def _agent(critic_mode="limited"):
    seed_all(SEED)
    return PPOAgent(len(ACTIONS), critic_mode=critic_mode, arch=ARCH)


def _params_snapshot(*modules):
    return [p.detach().clone() for m in modules for p in m.parameters()]


def _params_equal(snap, *modules):
    now = [p.detach() for m in modules for p in m.parameters()]
    return all(torch.equal(a, b) for a, b in zip(snap, now))


def test_optimizer_step_counter_counts_and_persists(tmp_path):
    agent = _agent()
    assert agent.optimizer_steps_total == 0
    play_episodes(agent, 6, collect_oracle=False, seed0=SEED)
    agent.update(epochs=2, batch_size=2)
    # per-step mode: one step per non-empty minibatch per epoch
    assert agent.optimizer_steps_total > 0
    steps = agent.optimizer_steps_total

    ckpt = str(tmp_path / "agent.pt")
    agent.save(ckpt)
    fresh = _agent()
    fresh.load(ckpt, load_optimizers=False)
    assert fresh.optimizer_steps_total == steps

    # pre-counter checkpoints load as zero
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    del payload["optimizer_steps_total"]
    legacy = str(tmp_path / "legacy.pt")
    torch.save(payload, legacy)
    older = _agent()
    older.load(legacy, load_optimizers=False)
    assert older.optimizer_steps_total == 0


def test_grad_accum_steps_once_per_epoch_in_counter():
    agent = _agent()
    play_episodes(agent, 8, collect_oracle=False, seed0=SEED + 50)
    agent.update(epochs=3, batch_size=2, grad_accum=True)
    assert agent.optimizer_steps_total == 3


def test_gns_diagnostic_measures_without_touching_training_state():
    agent = _agent()
    play_episodes(agent, 10, collect_oracle=False, seed0=SEED + 100)
    agent.gns_log = True
    snap_before_update = _params_snapshot(agent.actor, agent.encoder)
    stats = agent.update(epochs=1, batch_size=2)
    assert "gns" in stats
    gns = stats["gns"]
    assert set(gns) >= {"global", "lead", "lead_rows"}
    # global estimate should be computable on a 10-episode buffer
    assert gns["global"] is None or gns["global"] > 0
    # partner-lead SNR readout rides along whenever lead rows were sampled
    if gns["lead_rows"] > 0:
        assert gns["lead_adv_std"] >= 0
        assert 0.0 <= gns["lead_trump_mass"] <= 1.0
    # diagnostic leaves no gradients behind
    assert all(
        p.grad is None or not p.grad.abs().any()
        for p in list(agent.actor.parameters()) + list(agent.encoder.parameters())
    )
    # params did change (the update itself ran)
    assert not _params_equal(snap_before_update, agent.actor, agent.encoder)


def test_gns_off_is_absent_from_stats():
    agent = _agent()
    play_episodes(agent, 4, collect_oracle=False, seed0=SEED + 150)
    stats = agent.update(epochs=1, batch_size=2)
    assert "gns" not in stats


def test_oracle_extra_epochs_touch_only_the_oracle():
    agent = _agent(critic_mode="oracle")
    play_episodes(agent, 6, collect_oracle=True, seed0=SEED + 200)
    agent.update(epochs=1, batch_size=2)
    steps_after_update = agent.optimizer_steps_total

    policy_snap = _params_snapshot(agent.actor, agent.encoder, agent.critic)
    oracle_snap = _params_snapshot(agent.oracle_critic)
    play_episodes(agent, 6, collect_oracle=True, seed0=SEED + 300)
    # run ONLY the extra-epoch pass (no main epochs)
    agent._oracle_extra_epochs(2, batch_size=2)
    assert _params_equal(policy_snap, agent.actor, agent.encoder, agent.critic)
    assert not _params_equal(oracle_snap, agent.oracle_critic)
    # oracle-only steps are not policy steps
    assert agent.optimizer_steps_total == steps_after_update
    agent.reset_storage()


def test_ensure_csv_columns_migrates_prefix_header(tmp_path):
    path = str(tmp_path / "progress.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "picker_avg"])
        w.writerow(["100", "1.2"])
        w.writerow(["200", "1.3"])
    added = ensure_csv_columns(path, ["episode", "picker_avg", "opt_steps"])
    assert added == 1
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "picker_avg", "opt_steps"]
    assert rows[1] == ["100", "1.2", ""]
    assert rows[2] == ["200", "1.3", ""]
    # idempotent
    assert ensure_csv_columns(path, ["episode", "picker_avg", "opt_steps"]) == 0


def test_ensure_csv_columns_leaves_foreign_headers_alone(tmp_path):
    path = str(tmp_path / "other.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "something_else"])
        w.writerow(["1", "x"])
    assert ensure_csv_columns(path, ["episode", "picker_avg", "opt_steps"]) == 0
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "something_else"]
    assert ensure_csv_columns(str(tmp_path / "missing.csv"), ["a"]) == 0
