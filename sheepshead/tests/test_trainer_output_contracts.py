"""Characterization tests pinning trainer output contracts.

train_selfplay_ppo.py and train_league_ppo.py carry near-duplicate CSV-emit
code that is about to be deduplicated; these tests run tiny real training
invocations and pin the exact schemas so a refactor cannot silently change
a header, a row's field count, or a checkpoint's key-set / filename pattern
without a test failing. Values are training-dependent and NOT pinned except
where they come from synthetic/deterministic inputs.
"""

import csv
from types import SimpleNamespace

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import train_league_ppo, train_selfplay_ppo
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
    ]
)
ORACLE_CHECKPOINT_KEYS = sorted(
    LIMITED_CHECKPOINT_KEYS
    + ["critic_mode", "oracle_state_dict", "oracle_optimizer"]
)

LEAGUE_ANCHORED_HEADER = ["episode", "edge", "se", "win_frac", "n_deals"]
LEAGUE_EXPLOITABILITY_HEADER = [
    "generation",
    "main_episode",
    "edge",
    "se",
    "win_frac",
    "passed",
    "exploiter_ckpt",
]
SELFPLAY_ANCHORED_HEADER = [
    "episode",
    "train_wall_s",
    "eval_wall_s",
    "updates_done",
    "transitions_done",
    "edge_scripted",
    "se_scripted",
    "edge_selfplay100k",
    "se_selfplay100k",
    "edge_final_pfsp",
    "se_final_pfsp",
]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


# ----------------------------------------------------------------------------
# train_league_ppo
# ----------------------------------------------------------------------------
def _run_league_main_phase(tmp_path, critic_mode, anchor_eval=None, save_interval=1_000_000_000):
    league = League(str(tmp_path / "league"))
    agent = PPOAgent(len(ACTIONS), critic_mode=critic_mode)
    args = SimpleNamespace(
        seed=1,
        run_name="charcap",
        critic_mode=critic_mode,
        arch="full",
        num_workers=1,
        update_interval=1_000_000,
        schedule_horizon=1_000_000_000,
        save_interval=save_interval,
        snapshot_interval=1_000_000_000,
        greedy_eval_interval=0,
        greedy_eval_games=0,
    )
    ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    end = train_league_ppo.run_main_phase(
        agent,
        league,
        ratings,
        args,
        start_episode=0,
        n_episodes=2,
        checkpoint_dir=str(ckpt_dir),
        anchor_eval=anchor_eval,
    )
    return agent, ckpt_dir, end


class TestLeagueAnchoredEvalCSV:
    def test_header_and_row_schema(self, tmp_path):
        ref = PPOAgent(len(ACTIONS))
        anchor_eval = {"agent": ref, "label": "ref", "interval": 1, "deals": 2}
        _, ckpt_dir, end = _run_league_main_phase(
            tmp_path, "limited", anchor_eval=anchor_eval
        )
        assert end == 2
        rows = _read_csv(ckpt_dir / "anchored_eval.csv")
        assert rows[0] == LEAGUE_ANCHORED_HEADER
        # Probe fires every episode (interval=1) => one row per episode.
        assert len(rows) == 1 + 2
        for row in rows[1:]:
            assert len(row) == len(LEAGUE_ANCHORED_HEADER)


class TestLeagueCheckpointPayload:
    @pytest.mark.parametrize(
        "critic_mode,expected_keys",
        [
            ("limited", LIMITED_CHECKPOINT_KEYS),
            ("oracle", ORACLE_CHECKPOINT_KEYS),
        ],
    )
    def test_checkpoint_keys_and_filename(self, tmp_path, critic_mode, expected_keys):
        _, ckpt_dir, end = _run_league_main_phase(
            tmp_path, critic_mode, save_interval=2
        )
        ckpt_path = ckpt_dir / f"pfsp_full_checkpoint_{end}.pt"
        assert ckpt_path.exists(), f"expected filename pattern pfsp_<arch>_checkpoint_<episode>.pt, found {list(ckpt_dir.iterdir())}"
        payload = torch.load(ckpt_path, map_location="cpu")
        assert sorted(payload.keys()) == expected_keys


class TestLeagueExploitabilityCSV:
    def test_header_and_row_schema(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resume_ckpt = tmp_path / "resume.pt"
        PPOAgent(len(ACTIONS)).save(str(resume_ckpt))

        fake_gate = {
            "edge": 0.125,
            "se": 0.0625,
            "win_frac": 0.55,
            "passed": True,
            "exploiter_ckpt": "fake_exploiter.pt",
        }
        monkeypatch.setattr(
            train_league_ppo,
            "run_exploiter_generation",
            lambda args, generation, main_ckpt: fake_gate,
        )

        argv = [
            "train_league_ppo.py",
            "--resume",
            str(resume_ckpt),
            "--league-dir",
            str(tmp_path / "league"),
            "--run-name",
            "charcap_main",
            "--generations",
            "1",
            "--main-episodes",
            "2",
            "--update-interval",
            "1000000",
            "--save-interval",
            "2",
            "--snapshot-interval",
            "1000000",
            "--greedy-eval-interval",
            "0",
            "--anchor-eval-ckpt",
            "",
            "--num-workers",
            "1",
            "--seed",
            "3",
        ]
        monkeypatch.setattr("sys.argv", argv)
        train_league_ppo.main()

        exploitability_csv = tmp_path / "runs" / "charcap_main" / "checkpoints" / "exploitability.csv"
        rows = _read_csv(exploitability_csv)
        assert rows[0] == LEAGUE_EXPLOITABILITY_HEADER
        assert rows[1] == ["1", "2", "0.1250", "0.0625", "0.550", "True", "fake_exploiter.pt"]


# ----------------------------------------------------------------------------
# train_selfplay_ppo
# ----------------------------------------------------------------------------
class TestSelfplayAnchoredEvalCSV:
    def test_header_and_row_schema(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        anchor_100k = tmp_path / "anchor_100k.pt"
        anchor_pfsp = tmp_path / "anchor_pfsp.pt"
        PPOAgent(len(ACTIONS)).save(str(anchor_100k))
        PPOAgent(len(ACTIONS)).save(str(anchor_pfsp))

        train_selfplay_ppo.train_ppo(
            num_episodes=2,
            update_interval=1_000_000,
            save_interval=2,
            strategic_eval_interval=1_000_000,
            run_name="charcap_selfplay",
            anchor_eval_interval=1,
            anchor_eval_deals=2,
            anchor_100k=str(anchor_100k),
            anchor_pfsp=str(anchor_pfsp),
            seed=5,
        )

        anchored_csv = tmp_path / "runs" / "charcap_selfplay" / "anchored_eval.csv"
        rows = _read_csv(anchored_csv)
        assert rows[0] == SELFPLAY_ANCHORED_HEADER
        # Probe fires every episode (interval=1) => one row per episode.
        assert len(rows) == 1 + 2
        for row in rows[1:]:
            assert len(row) == len(SELFPLAY_ANCHORED_HEADER)


class TestSelfplayCheckpointPayload:
    def test_checkpoint_keys_and_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        train_selfplay_ppo.train_ppo(
            num_episodes=2,
            update_interval=1_000_000,
            save_interval=2,
            strategic_eval_interval=1_000_000,
            run_name="charcap_selfplay_ckpt",
            anchor_eval_interval=0,
            seed=7,
        )

        checkpoint_dir = tmp_path / "runs" / "charcap_selfplay_ckpt"
        ckpt_path = checkpoint_dir / "full_checkpoint_2.pt"
        final_path = checkpoint_dir / "final_full.pt"
        assert ckpt_path.exists(), f"expected <arch>_checkpoint_<episode>.pt, found {list(checkpoint_dir.iterdir())}"
        assert final_path.exists(), "expected final_<arch>.pt"

        for path in (ckpt_path, final_path):
            payload = torch.load(path, map_location="cpu")
            assert sorted(payload.keys()) == LIMITED_CHECKPOINT_KEYS
