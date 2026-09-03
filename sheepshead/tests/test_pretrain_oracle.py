"""pretrain_oracle end to end at toy scale: a spawn-pool dataset from a
fresh checkpoint, then the supervised fit producing a state_dict that loads
into the production oracle network."""

import json

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.oracle import OracleValueNetwork
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import pretrain_oracle

pytestmark = pytest.mark.slow


def test_generate_then_pretrain(tmp_path):
    torch.manual_seed(0)
    ckpt = tmp_path / "seed.pt"
    PPOAgent(len(ACTIONS), arch="perceiver-recall").save(str(ckpt))
    dataset = tmp_path / "dataset.pt"
    rc = pretrain_oracle.main(
        [
            "generate",
            "--ckpt",
            str(ckpt),
            "--episodes",
            "4",
            "--workers",
            "1",
            "--gamma",
            "1.0",
            "--out",
            str(dataset),
        ]
    )
    assert rc == 0
    data = torch.load(dataset, weights_only=False)
    assert data["gamma"] == 1.0 and len(data["episodes"]) >= 4
    ep = data["episodes"][0]
    assert set(ep) == {"obs", "is_action", "g", "strata"}
    assert "opp_hand_ids" in ep["obs"][0]  # full-information observations
    assert any(ep["is_action"])
    out = tmp_path / "oracle_init.pt"
    rc = pretrain_oracle.main(
        [
            "pretrain",
            "--dataset",
            str(dataset),
            "--max-epochs",
            "1",
            "--batch-size",
            "4",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    net = OracleValueNetwork(use_aux_heads=True)
    net.load_state_dict(torch.load(out, weights_only=True), strict=True)
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert "test" in report and "head_metrics" in report
