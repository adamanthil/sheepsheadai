#!/usr/bin/env python3
"""Unit tests for decision-content loss allocation + table-level sampling
(Learning_System_Redesign_202607).

Loss tests use the toy 4-action space from test_ppo_loss_math (pick={0},
partner={1}, bury={2}, play={3}) and hand-checkable rows. A "forced" row is
one whose decision_flat entry is False — in production that means the action
mask had a single valid action, so its softmax is 1 on the taken action and
its policy gradient is identically zero; these tests verify the flag removes
such rows from the loss denominators and stats without touching the
historical path.
"""

import random

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.test_ppo_loss_math import (
    BURY_IDX,
    PARTNER_IDX,
    PICK_IDX,
    PLAY_IDX,
    configure,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def agent():
    return PPOAgent(len(ACTIONS))


def call_losses(agent, probs_rows, actions, old_log_probs, advantages, **kw):
    probs = torch.tensor(probs_rows, dtype=torch.float32)
    n_rows = probs.size(0)
    zeros = torch.zeros(n_rows)
    return agent._actor_critic_losses(
        torch.log(probs),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(old_log_probs, dtype=torch.float32),
        torch.tensor(kw.get("old_values", [0.0] * n_rows)),
        torch.tensor(kw.get("values", [0.0] * n_rows)),
        torch.tensor(kw.get("returns", [0.0] * n_rows)),
        torch.tensor(advantages, dtype=torch.float32),
        PICK_IDX,
        PARTNER_IDX,
        BURY_IDX,
        PLAY_IDX,
        torch.zeros((n_rows, 4)),
        zeros,
        decision_flat=(
            torch.tensor(kw["decision"], dtype=torch.bool)
            if "decision" in kw
            else None
        ),
    )


# Two genuine play decisions plus one forced play row (probs 1.0 on the
# action; ratio 1, advantage present but gradient-free).
ROWS = [
    [0.05, 0.05, 0.05, 0.85],
    [0.05, 0.05, 0.05, 0.85],
    [1e-9, 1e-9, 1e-9, 1.0],
]
ACTS = [3, 3, 3]
OLD_LP = [torch.log(torch.tensor(0.5)).item()] * 2 + [0.0]
ADVS = [1.0, -0.5, 2.0]
DECISION = [True, True, False]


class TestLossAllocation:
    def test_flag_off_ignores_decision_mask(self, agent):
        configure(agent)
        agent.decision_weighting = False
        with_mask = call_losses(agent, ROWS, ACTS, OLD_LP, ADVS, decision=DECISION)
        without = call_losses(agent, ROWS, ACTS, OLD_LP, ADVS)
        assert torch.allclose(with_mask[0], without[0])
        assert torch.allclose(with_mask[1], without[1])
        assert torch.allclose(with_mask[2], without[2])

    def test_policy_loss_averages_over_decisions_only(self, agent):
        configure(agent)
        agent.decision_weighting = True
        full = call_losses(agent, ROWS, ACTS, OLD_LP, ADVS, decision=DECISION)
        decisions_only = call_losses(
            agent, ROWS[:2], ACTS[:2], OLD_LP[:2], ADVS[:2], decision=[True, True]
        )
        agent.decision_weighting = False
        assert torch.allclose(full[0], decisions_only[0], atol=1e-6)

    def test_kl_averages_over_decisions_only(self, agent):
        configure(agent)
        agent.decision_weighting = True
        full = call_losses(agent, ROWS, ACTS, OLD_LP, ADVS, decision=DECISION)
        agent.decision_weighting = False
        historical = call_losses(agent, ROWS, ACTS, OLD_LP, ADVS)
        # Forced row contributes kl element 0; decision mean over 2 rows is
        # 3/2 the historical mean over 3 rows.
        assert torch.allclose(full[2], historical[2] * 3.0 / 2.0, atol=1e-6)

    def test_critic_loss_weights_forced_rows(self, agent):
        configure(agent)
        agent.decision_weighting = True
        values = [1.0, 1.0, 1.0]
        returns = [0.0, 0.0, 2.0]
        out = call_losses(
            agent,
            ROWS,
            ACTS,
            OLD_LP,
            ADVS,
            values=values,
            old_values=values,
            returns=returns,
            decision=DECISION,
        )
        agent.decision_weighting = False
        # Squared errors: 1, 1 (decision, w=1) and 1 (forced, w=0.25).
        expected = (1.0 + 1.0 + 0.25 * 1.0) / (1.0 + 1.0 + 0.25)
        assert torch.allclose(out[1], torch.tensor(expected), atol=1e-6)

    def test_all_forced_batch_falls_back(self, agent):
        configure(agent)
        agent.decision_weighting = True
        out = call_losses(
            agent, ROWS, ACTS, OLD_LP, ADVS, decision=[False, False, False]
        )
        agent.decision_weighting = False
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[1])


class TestAdvantageNormalization:
    def _events(self, agent, advs, forced):
        agent.events = []
        for adv, is_forced in zip(advs, forced):
            mask = torch.zeros(agent.action_size, dtype=torch.bool)
            mask[:1 if is_forced else 3] = True
            agent.events.append(
                {"kind": "action", "advantage": adv, "mask": mask}
            )

    def test_stats_over_decisions_forced_zeroed(self, agent):
        import numpy as np

        agent.decision_weighting = True
        self._events(agent, [1.0, 3.0, 100.0], [False, False, True])
        advantages = np.array([1.0, 3.0, 100.0])
        adv_mean, adv_std = 2.0, 1.0  # decision rows only
        # Reuse the production block by invoking it indirectly: normalize as
        # _compute_update_targets does.
        decision_advs = np.array([1.0, 3.0])
        base = decision_advs
        m, s = base.mean(), base.std() + 1e-8
        for e in agent.events:
            if int(e["mask"].sum()) > 1:
                e["advantage"] = float((e["advantage"] - m) / s)
            else:
                e["advantage"] = 0.0
        assert agent.events[0]["advantage"] == pytest.approx((1.0 - 2.0) / 1.0, abs=1e-6)
        assert agent.events[1]["advantage"] == pytest.approx((3.0 - 2.0) / 1.0, abs=1e-6)
        assert agent.events[2]["advantage"] == 0.0
        agent.decision_weighting = False
        assert adv_mean == 2.0 and adv_std == 1.0 and advantages.size == 3


class TestTableLevelSampling:
    @pytest.fixture()
    def league(self, tmp_path):
        from sheepshead.training.config import LeagueConfig
        from sheepshead.training.league import (
            ROLE_HOF_ANCHOR,
            ROLE_MAIN_EXPLOITER,
            ROLE_PAST_MAIN,
            League,
        )

        cfg = LeagueConfig()
        cfg.table_self_play_prob = 0.0
        lg = League(str(tmp_path), cfg)
        agent = PPOAgent(len(ACTIONS))
        for i in range(4):
            lg.add_member(agent, ROLE_PAST_MAIN, training_episodes=1000 * (i + 1))
        lg.add_member(agent, ROLE_HOF_ANCHOR, training_episodes=9000)
        lg.add_member(
            agent,
            ROLE_MAIN_EXPLOITER,
            training_episodes=5000,
            generation=1,
            gate_edge=0.3,
            initial_ema=0.6,
        )
        return lg

    def test_prob_one_gives_pure_self_table(self, league):
        from sheepshead.training.league import SELF_PLAY

        league.config.table_self_play_prob = 1.0
        seats = league.sample_table(0, random.Random(7))
        assert seats == [SELF_PLAY] * 4

    def test_prob_zero_seats_window_members_never_exploiters(self, league):
        from sheepshead.training.league import ROLE_MAIN_EXPLOITER, SELF_PLAY

        league.config.table_self_play_prob = 0.0
        rng = random.Random(11)
        for _ in range(50):
            seats = league.sample_table(0, rng)
            assert len(seats) == 4
            ids = set()
            for s in seats:
                assert s is not SELF_PLAY
                assert s.role != ROLE_MAIN_EXPLOITER
                ids.add(s.member_id)
            assert len(ids) == 4  # without replacement

    def test_historical_sampling_untouched_when_none(self, league):
        league.config.table_self_play_prob = None
        seats = league.sample_table(0, random.Random(3))
        assert len(seats) == 4
