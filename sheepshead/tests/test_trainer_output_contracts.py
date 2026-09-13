"""Characterization tests pinning train_ppo's output contracts: the
progress and greedy-health CSV schemas and the checkpoint key-set /
filename pattern. Values are training-dependent and not pinned."""

import csv
import os
from types import SimpleNamespace

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import train_ppo
from sheepshead.training.league import League

pytestmark = pytest.mark.slow

LIMITED_CHECKPOINT_KEYS = sorted(
    [
        "arch",
        "encoder_state_dict",
        "actor_state_dict",
        "critic_state_dict",
        "actor_optimizer",
        "critic_optimizer",
        "optimizer_steps_total",
        "gamma",
    ]
)
ORACLE_CHECKPOINT_KEYS = sorted(
    LIMITED_CHECKPOINT_KEYS
    + ["critic_mode", "oracle_state_dict", "oracle_optimizer", "oracle_aux_heads"]
)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _run(tmp_path, phase, **overrides):
    spec = train_ppo.PHASE_SPECS[phase]
    league = League(str(tmp_path / "league"))
    agent = PPOAgent(len(ACTIONS), critic_mode=spec.critic_mode, arch="onehot-ff")
    agent.set_trainable_heads(spec.trainable_heads)
    args = SimpleNamespace(
        phase=phase,
        seed=1,
        run_name="charcap",
        num_workers=1,
        update_interval=1_000_000,
        until=2,
        save_interval=1_000_000_000,
        snapshot_interval=0,
        greedy_eval_interval=0,
        greedy_eval_games=0,
        leaster_watchdog=False,
        entropy_controller=False,
        entropy_play_floor=0.28,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    end = train_ppo.run_phase(
        agent, league, ratings, args, 0, args.until, str(ckpt_dir)
    )
    return agent, ckpt_dir, end


class TestCheckpointPayload:
    @pytest.mark.parametrize(
        "phase,expected_keys",
        [("bootstrap", LIMITED_CHECKPOINT_KEYS), ("league", ORACLE_CHECKPOINT_KEYS)],
    )
    def test_checkpoint_keys_and_filename(self, tmp_path, phase, expected_keys):
        _, ckpt_dir, end = _run(tmp_path, phase, save_interval=2)
        path = ckpt_dir / f"checkpoint_{end}.pt"
        assert path.exists(), list(ckpt_dir.iterdir())
        payload = torch.load(path, map_location="cpu")
        assert sorted(payload.keys()) == expected_keys


class TestProgressCSV:
    def test_header_and_row_schema(self, tmp_path):
        _, ckpt_dir, _ = _run(tmp_path, "bootstrap", update_interval=10)
        rows = _read_csv(ckpt_dir / "training_progress.csv")
        assert rows[0] == train_ppo.PROGRESS_CSV_HEADER
        assert len(rows) >= 2
        for row in rows[1:]:
            assert len(row) == len(train_ppo.PROGRESS_CSV_HEADER)


class TestGreedyCSV:
    def test_header_and_row_schema(self, tmp_path):
        _, ckpt_dir, _ = _run(
            tmp_path, "bootstrap", greedy_eval_interval=1, greedy_eval_games=2
        )
        rows = _read_csv(ckpt_dir / "greedy_health.csv")
        assert rows[0] == train_ppo.GREEDY_CSV_HEADER
        assert len(rows) == 1 + 2
        for row in rows[1:]:
            assert len(row) == len(train_ppo.GREEDY_CSV_HEADER)


def test_cli_defaults_resolve_per_phase():
    p = train_ppo.build_arg_parser()
    boot = p.parse_args(["--phase", "bootstrap", "--run-name", "x", "--until", "1"])
    train_ppo.resolve_args(boot)
    assert boot.update_interval == 4096
    assert boot.snapshot_interval == 0
    assert boot.entropy_controller is False
    league = p.parse_args(["--phase", "league", "--run-name", "x", "--until", "1"])
    train_ppo.resolve_args(league)
    assert league.update_interval == 16_384
    assert league.snapshot_interval == 50_000
    assert league.entropy_controller is True
    assert league.league_dir == os.path.join("runs", "x", "league")
