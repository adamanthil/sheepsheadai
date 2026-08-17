"""--oracle-init overwrite guard (CE_Teacher_Design §10.2).

The flag exists for resuming PRE-oracle checkpoints; applied after a
resume that restored trained oracle weights it silently downgrades them
to the pretrain (this shipped in the attempt-9/10 launch recipes). The
guard is a tracking attribute set by PPOAgent.load plus a loud banner at
both launch-time overwrite sites.
"""

import sys

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.league_teacher import warn_if_oracle_overwrite


def test_load_sets_oracle_loaded_flag(tmp_path):
    ckpt = str(tmp_path / "oracle_ckpt.pt")
    PPOAgent(len(ACTIONS), critic_mode="oracle").save(ckpt)

    agent = PPOAgent(len(ACTIONS), critic_mode="oracle")
    assert agent.oracle_loaded_from_checkpoint is False
    agent.load(ckpt, load_optimizers=False)
    assert agent.oracle_loaded_from_checkpoint is True


def test_limited_checkpoint_leaves_flag_unset(tmp_path):
    ckpt = str(tmp_path / "limited_ckpt.pt")
    PPOAgent(len(ACTIONS), critic_mode="limited").save(ckpt)

    agent = PPOAgent(len(ACTIONS), critic_mode="oracle")
    agent.load(ckpt, load_optimizers=False)
    assert agent.oracle_loaded_from_checkpoint is False


def test_warn_fires_only_when_checkpoint_carried_oracle(capsys):
    agent = PPOAgent(len(ACTIONS), critic_mode="oracle")

    warn_if_oracle_overwrite(agent, "init.pt", "resume.pt")
    assert "OVERWRITING" not in capsys.readouterr().out

    agent.oracle_loaded_from_checkpoint = True
    warn_if_oracle_overwrite(agent, "init.pt", "resume.pt")
    out = capsys.readouterr().out
    assert "OVERWRITING" in out
    assert "init.pt" in out and "resume.pt" in out


if __name__ == "__main__":
    sys.exit(0)
