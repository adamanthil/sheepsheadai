#!/usr/bin/env python3
"""Portable unit tests for the PPO loss and GAE math.

Each test feeds tiny hand-written tensors through the real loss/GAE code
and checks against values derivable on paper, with a toy 4-action space
(pick={0}, partner={1}, bury={2}, play={3} unless a test says otherwise).
Single float ops are stable to ~1e-6 across platforms, so unlike the
bit-exact sha fixtures these run everywhere -- and a failure names the
term that broke instead of just reporting a hash mismatch.

Head-index groups are parameters of _actor_critic_losses, which is what
makes the toy action space possible; the coefficients it reads from self
are pinned explicitly per test via ``configure``.
"""

import math

import numpy as np
import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent

GAMMA = 0.9
LAMBDA = 0.8

PICK_IDX = torch.tensor([0])
PARTNER_IDX = torch.tensor([1])
BURY_IDX = torch.tensor([2])
PLAY_IDX = torch.tensor([3])


@pytest.fixture(scope="module")
def agent():
    return PPOAgent(len(ACTIONS))


def configure(agent, **overrides):
    settings = {
        "entropy_coeff_pick": 0.0,
        "entropy_coeff_partner": 0.0,
        "entropy_coeff_bury": 0.0,
        "entropy_coeff_play": 0.0,
        "kl_coef": 0.0,
        "anchor_coeff": 0.0,
        "clip_epsilon_pick": 0.2,
        "clip_epsilon_partner": 0.2,
        "clip_epsilon_bury": 0.2,
        "clip_epsilon_play": 0.2,
        "value_clip_epsilon": 0.2,
    }
    settings.update(overrides)
    for name, value in settings.items():
        setattr(agent, name, value)


def call_losses(
    agent,
    probs_rows,
    actions,
    old_log_probs,
    advantages,
    values=None,
    old_values=None,
    returns=None,
):
    """Invoke _actor_critic_losses on the toy action space; per-row action
    probabilities are given directly and converted to logits."""
    probs = torch.tensor(probs_rows, dtype=torch.float32)
    n_rows = probs.size(0)
    zeros = torch.zeros(n_rows)
    return agent._actor_critic_losses(
        torch.log(probs),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(old_log_probs, dtype=torch.float32),
        zeros if old_values is None else torch.tensor(old_values),
        zeros if values is None else torch.tensor(values),
        zeros if returns is None else torch.tensor(returns),
        torch.tensor(advantages, dtype=torch.float32),
        PICK_IDX,
        PARTNER_IDX,
        BURY_IDX,
        PLAY_IDX,
    )


class TestGae1d:
    def test_terminal_episode_hand_computed(self):
        # t=2: delta = 2 + 0 - 0.3 = 1.7
        # t=1: delta = 0 + 0.9*0.3 - 0.4 = -0.13; gae = -0.13 + 0.72*1.7
        # t=0: delta = 1 + 0.9*0.4 - 0.5 = 0.86; gae = 0.86 + 0.72*1.094
        advantages, returns = PPOAgent._gae_1d(
            np.array([1.0, 0.0, 2.0]),
            np.array([0.5, 0.4, 0.3, 0.2]),
            np.array([False, False, True]),
            GAMMA,
            LAMBDA,
        )
        assert advantages.tolist() == pytest.approx([1.64768, 1.094, 1.7])
        assert returns.tolist() == pytest.approx([2.14768, 1.494, 2.0])

    def test_truncated_episode_bootstraps_from_final_value(self):
        # t=2: delta = 2 + 0.9*0.2 - 0.3 = 1.88
        advantages, _returns = PPOAgent._gae_1d(
            np.array([1.0, 0.0, 2.0]),
            np.array([0.5, 0.4, 0.3, 0.2]),
            np.array([False, False, False]),
            GAMMA,
            LAMBDA,
        )
        assert advantages.tolist() == pytest.approx([1.740992, 1.2236, 1.88])

    def test_mid_sequence_done_stops_propagation(self):
        # t=1 is terminal: no bootstrap from values[2] and no carry from t=2.
        advantages, _returns = PPOAgent._gae_1d(
            np.array([1.0, 0.0, 2.0]),
            np.array([0.5, 0.4, 0.3, 0.2]),
            np.array([False, True, False]),
            GAMMA,
            LAMBDA,
        )
        assert advantages.tolist() == pytest.approx([0.572, -0.4, 1.88])


class TestComputeGae:
    def test_writes_back_into_action_events_only(self, agent):
        agent.gamma = GAMMA
        agent.gae_lambda = LAMBDA
        agent.events = [
            {"kind": "observation"},
            {"kind": "action", "reward": 1.0, "value": 0.5, "done": False},
            {"kind": "observation"},
            {"kind": "action", "reward": 0.0, "value": 0.4, "done": False},
            {"kind": "action", "reward": 2.0, "value": 0.3, "done": True},
        ]
        try:
            advantages, returns = agent.compute_gae()
            expected_adv, expected_ret = PPOAgent._gae_1d(
                np.array([1.0, 0.0, 2.0]),
                np.array([0.5, 0.4, 0.3, 0.0]),
                np.array([False, False, True, False]),
                GAMMA,
                LAMBDA,
            )
            assert advantages.tolist() == pytest.approx(expected_adv.tolist())
            assert returns.tolist() == pytest.approx(expected_ret.tolist())
            action_events = [e for e in agent.events if e["kind"] == "action"]
            for event, adv, ret in zip(action_events, expected_adv, expected_ret):
                assert event["advantage"] == pytest.approx(adv)
                assert event["return"] == pytest.approx(ret)
            for event in agent.events:
                if event["kind"] == "observation":
                    assert "advantage" not in event
        finally:
            agent.events = []


class TestPolicyLoss:
    def test_ratio_above_clip_uses_clipped_surrogate(self, agent):
        configure(agent)
        # ratio = 0.4/0.2 = 2.0 > 1.2 with positive advantage:
        # element = -min(2.0*1, 1.2*1) = -1.2
        actor_loss, _critic, approx_kl, _ents, _anchor = call_losses(
            agent,
            probs_rows=[[0.1, 0.2, 0.3, 0.4]],
            actions=[3],
            old_log_probs=[math.log(0.2)],
            advantages=[1.0],
        )
        assert actor_loss.item() == pytest.approx(-1.2, abs=1e-5)
        assert approx_kl.item() == pytest.approx(2.0 - 1.0 - math.log(2.0), abs=1e-5)

    def test_ratio_below_clip_with_negative_advantage(self, agent):
        configure(agent)
        # ratio = 0.1/0.2 = 0.5 < 0.8 with advantage -1:
        # element = -min(0.5*-1, 0.8*-1) = 0.8
        actor_loss, _critic, _kl, _ents, _anchor = call_losses(
            agent,
            probs_rows=[[0.2, 0.3, 0.4, 0.1]],
            actions=[3],
            old_log_probs=[math.log(0.2)],
            advantages=[-1.0],
        )
        assert actor_loss.item() == pytest.approx(0.8, abs=1e-5)

    def test_per_head_weights_rebalance_row_counts(self, agent):
        configure(agent)
        # One pick row + three play rows, all at ratio 1:
        # w_pick = 4/(2*1) = 2, w_play = 4/(2*3) = 2/3
        # loss = -(2*2 + (2/3)*(1+1+1))/4 = -1.5
        probs_rows = [[0.25, 0.25, 0.25, 0.25]] * 4
        actions = [0, 3, 3, 3]
        actor_loss, _critic, _kl, _ents, _anchor = call_losses(
            agent,
            probs_rows=probs_rows,
            actions=actions,
            old_log_probs=[math.log(0.25)] * 4,
            advantages=[2.0, 1.0, 1.0, 1.0],
        )
        assert actor_loss.item() == pytest.approx(-1.5, abs=1e-5)

    def test_kl_penalty_scales_actor_loss(self, agent):
        configure(agent, kl_coef=3.0)
        expected_kl = 2.0 - 1.0 - math.log(2.0)
        actor_loss, _critic, _kl, _ents, _anchor = call_losses(
            agent,
            probs_rows=[[0.1, 0.2, 0.3, 0.4]],
            actions=[3],
            old_log_probs=[math.log(0.2)],
            advantages=[1.0],
        )
        assert actor_loss.item() == pytest.approx(-1.2 + 3.0 * expected_kl, abs=1e-5)


class TestCriticLoss:
    def test_clipped_value_loss_takes_pessimistic_max(self, agent):
        configure(agent)
        # v_clipped = 0 + clip(1.0 - 0, +-0.2) = 0.2; target 0.2:
        # max((1.0-0.2)^2, (0.2-0.2)^2) = 0.64
        _actor, critic_loss, _kl, _ents, _anchor = call_losses(
            agent,
            probs_rows=[[0.25, 0.25, 0.25, 0.25]],
            actions=[3],
            old_log_probs=[math.log(0.25)],
            advantages=[0.0],
            values=[1.0],
            old_values=[0.0],
            returns=[0.2],
        )
        assert critic_loss.item() == pytest.approx(0.64, abs=1e-6)

    def test_unclipped_error_dominates_when_larger(self, agent):
        configure(agent)
        # v_clipped = 0.2, target 0.2 -> clipped mse 0; unclipped (0.5-0.2)^2
        _actor, critic_loss, _kl, _ents, _anchor = call_losses(
            agent,
            probs_rows=[[0.25, 0.25, 0.25, 0.25]],
            actions=[3],
            old_log_probs=[math.log(0.25)],
            advantages=[0.0],
            values=[0.5],
            old_values=[0.0],
            returns=[0.2],
        )
        assert critic_loss.item() == pytest.approx(0.09, abs=1e-6)


class TestEntropyBonus:
    def test_head_entropies_normalize_within_head(self, agent):
        # Play head spans two actions with equal mass -> ln 2 after
        # renormalization; the peaked pick head contributes ~0.
        probs = torch.tensor([[0.3, 0.0, 0.35, 0.35]])
        pick_e, partner_e, bury_e, play_e = agent._head_entropies(
            probs, PICK_IDX, PARTNER_IDX, BURY_IDX, torch.tensor([2, 3])
        )
        assert play_e.item() == pytest.approx(math.log(2.0), abs=1e-5)
        assert pick_e.item() == pytest.approx(0.0, abs=1e-5)

    def test_entropy_bonus_subtracts_from_actor_loss(self, agent):
        configure(agent, entropy_coeff_play=0.5)
        # Zero advantage kills the PG term; play head is a single action so
        # its entropy is ~0 -- but the play-head slice covers only action 3,
        # whose probability renormalizes to 1. Use two play actions instead.
        probs = torch.tensor([[0.2, 0.2, 0.3, 0.3]])
        n_rows = 1
        zeros = torch.zeros(n_rows)
        actor_loss, _critic, _kl, entropies, _anchor = agent._actor_critic_losses(
            torch.log(probs),
            torch.tensor([3]),
            torch.log(torch.tensor([0.3])),
            zeros,
            zeros,
            zeros,
            torch.tensor([0.0]),
            PICK_IDX,
            PARTNER_IDX,
            BURY_IDX,
            torch.tensor([2, 3]),
        )
        _pick_e, _partner_e, _bury_e, play_e = entropies
        assert play_e.item() == pytest.approx(math.log(2.0), abs=1e-5)
        assert actor_loss.item() == pytest.approx(-0.5 * math.log(2.0), abs=1e-5)


class TestTeacherCE:
    """CE search-teacher term (CE_Teacher_Design §1.3): loss-form autograd
    checks plus the agent-level asymmetric-epoch pass over stored events."""

    def test_ce_gradient_zero_at_conformity(self):
        # The CE-toward-fixed-target gradient wrt logits is softmax - target,
        # so a policy that already matches the target earns exactly zero
        # gradient — abstention/self-retirement is a property of the TARGET,
        # not of any gate. (This replaces the removed §12 hinge machinery.)
        logits = torch.tensor([[0.4, -0.3, 1.2, 0.1]], requires_grad=True)
        target = torch.softmax(logits, dim=-1).detach()
        ce = -(target * torch.log_softmax(logits, dim=-1)).sum()
        ce.backward()
        assert logits.grad is not None
        assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-7)

    def test_ce_gradient_is_softmax_minus_target(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        target = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
        ce = -(target * torch.log_softmax(logits, dim=-1)).sum()
        ce.backward()
        expected = torch.softmax(torch.zeros(1, 4), dim=-1) - target
        assert logits.grad is not None
        assert torch.allclose(logits.grad, expected, atol=1e-6)

    @staticmethod
    def _synthetic_events(labeled_flags=(True, False)):
        """One-action episodes on real observation dicts (drawn from a
        seeded game); ``labeled_flags`` says which carry a CE target."""
        from sheepshead import PARTNER_BY_CALLED_ACE, Game

        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=3)
        player = game.players[0]
        state = player.get_state_dict()
        valid = sorted(player.get_valid_action_ids())
        events = []
        for labeled in labeled_flags:
            ev = {
                "kind": "action",
                "state": state,
                "action": valid[0],
                "log_prob": math.log(0.5),
                "value": 0.0,
                "valid_actions": set(valid),
                "reward": 0.0,
                "player_id": 1,
            }
            if labeled:
                ev["search_target"] = np.full(
                    len(valid), 1.0 / len(valid), dtype=np.float32
                )
                ev["has_search_target"] = True
            events.append(ev)
        return events, len(valid)

    def test_teacher_epochs_step_actor_only_on_labeled_rows(self, agent):
        agent.reset_storage()
        agent.teacher_coeff = 1.0
        events, n_valid = self._synthetic_events()
        for ev in events:
            agent.store_episode_events([ev])
        assert sum(1 for e in agent.events if e["has_search_target"]) == 1
        actor_before = {
            k: v.detach().clone() for k, v in agent.actor.state_dict().items()
        }
        critic_before = {
            k: v.detach().clone() for k, v in agent.critic.state_dict().items()
        }
        steps_before = agent.optimizer_steps_total
        agent.compute_gae()  # update() runs GAE before the teacher pass
        try:
            stats = agent._run_teacher_ce_epochs(teacher_epochs=2, batch_size=8)
        finally:
            agent.reset_storage()
        assert stats is not None
        assert stats["rows"] == 1 and stats["epochs"] == 2
        assert stats["ce"] > 0.0
        # KL = CE - H(target); the synthetic target is uniform over the
        # legal set. The two epochs move the policy toward the target, so
        # per-epoch KL means need a loose bound, not equality across epochs;
        # the identity holds within the accumulated means.
        assert stats["kl"] == pytest.approx(stats["ce"] - math.log(n_valid), abs=1e-4)
        # One actor-path optimizer step per epoch; critic untouched.
        assert agent.optimizer_steps_total == steps_before + 2
        actor_after = agent.actor.state_dict()
        assert any(
            not torch.equal(actor_before[k], actor_after[k]) for k in actor_before
        )
        critic_after = agent.critic.state_dict()
        assert all(
            torch.equal(critic_before[k], critic_after[k]) for k in critic_before
        )

    def test_teacher_epochs_noop_without_labels(self, agent):
        agent.reset_storage()
        events, _ = self._synthetic_events(labeled_flags=(False, False))
        for ev in events:
            agent.store_episode_events([ev])
        actor_before = {
            k: v.detach().clone() for k, v in agent.actor.state_dict().items()
        }
        steps_before = agent.optimizer_steps_total
        agent.compute_gae()
        try:
            stats = agent._run_teacher_ce_epochs(teacher_epochs=4, batch_size=8)
        finally:
            agent.reset_storage()
        assert stats is None
        assert agent.optimizer_steps_total == steps_before
        actor_after = agent.actor.state_dict()
        assert all(torch.equal(actor_before[k], actor_after[k]) for k in actor_before)
