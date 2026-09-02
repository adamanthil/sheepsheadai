"""perceiver-shared-v2-bp (CE_Teacher_Design §20.9): the play pointer's
bilinear state x card term is inert at init, learns, and a v2 checkpoint
migrates onto it bit-identically."""

import numpy as np
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.analysis.migrate_arch_checkpoint import (
    max_prob_divergence,
    migrate,
)
from sheepshead.tests.ppo_test_helpers import seed_all


def _agent(arch: str, critic_mode: str = "limited") -> PPOAgent:
    seed_all(0)
    ag = PPOAgent(len(ACTIONS), critic_mode=critic_mode, arch=arch)
    ag.stash_action_probs = True
    return ag


def test_registry_has_bp_with_v2_encoder_and_zero_state_side():
    ag = _agent("perceiver-shared-v2-bp")
    assert ag.actor.bilinear_pointer
    assert float(ag.actor.pointer_U.weight.abs().max()) == 0.0
    assert float(ag.actor.pointer_V.weight.abs().max()) > 0.0
    v2 = _agent("perceiver-shared-v2")
    assert not v2.actor.bilinear_pointer
    assert type(ag.encoder) is type(v2.encoder)
    assert ag.encoder.state_dict().keys() == v2.encoder.state_dict().keys()
    extra = set(ag.actor.state_dict()) - set(v2.actor.state_dict())
    assert extra == {"pointer_U.weight", "pointer_V.weight"}


def test_migration_is_bit_identical_and_the_term_learns(tmp_path):
    # Oracle-mode source: the migration must carry the privileged critic.
    v2 = _agent("perceiver-shared-v2", critic_mode="oracle")
    src = tmp_path / "v2.pt"
    v2.save(str(src))
    bp, fresh = migrate(str(src), "perceiver-shared-v2-bp")
    assert sorted(fresh) == ["pointer_U.weight", "pointer_V.weight"]
    assert bp.oracle_critic is not None and bp.critic_mode == "oracle"
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            v2.oracle_critic.state_dict().values(),
            bp.oracle_critic.state_dict().values(),
        )
    )
    bp.stash_action_probs = True
    assert max_prob_divergence(v2, bp, games=3) == 0.0
    out = tmp_path / "bp.pt"
    bp.save(str(out))
    reloaded = load_agent(str(out), load_optimizers=False)
    assert reloaded.arch_name == "perceiver-shared-v2-bp"
    assert reloaded.oracle_critic is not None
    assert reloaded.actor.bilinear_pointer
    # A nonzero U changes play logits, and gradients reach both new tensors.
    with torch.no_grad():
        reloaded.actor.pointer_U.weight.add_(0.05)
    reloaded.stash_action_probs = True
    assert max_prob_divergence(v2, reloaded, games=2) > 1e-6
    feat = torch.randn(4, reloaded.actor._d_model, requires_grad=False)
    tok = torch.randn(4, 8, reloaded.actor._d_token)
    reloaded.actor._score_hand_pointer(feat, tok).sum().backward()
    assert reloaded.actor.pointer_U.weight.grad is not None
    assert float(reloaded.actor.pointer_U.weight.grad.abs().sum()) > 0.0
    assert float(reloaded.actor.pointer_V.weight.grad.abs().sum()) > 0.0
    assert np.isfinite(float(reloaded.actor.pointer_V.weight.grad.abs().sum()))
