"""Lead-vs-follow routing in the head-routed h2h chimera (§20.12 stall
diagnosis)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from sheepshead.eval.head_routed_h2h import (
    HeadRoutedAgent,
    is_lead_decision,
    routed_h2h,
)
from sheepshead.game import ACTION_IDS, ACTIONS

PICK = ACTION_IDS["PICK"]
PASS = ACTION_IDS["PASS"]
PLAY = [i for i, a in enumerate(ACTIONS, start=1) if a.startswith("PLAY ")][:3]


class TagAgent:
    """Returns its tag from act() and counts calls, like a PPOAgent would
    advance its recurrent memory once per act()."""

    def __init__(self, tag: str):
        self.tag = tag
        self.acts = 0
        self.resets = 0
        self.observes = 0

    def act(self, state, valid_actions, player_id, deterministic=True):
        self.acts += 1
        return (self.tag, None, None)

    def observe(self, state, player_id=None):
        self.observes += 1

    def reset_recurrent_state(self):
        self.resets += 1


def state_with_leader(rel: int) -> dict:
    return {
        "leader_rel": np.uint8(rel),
        "play_started": np.uint8(1),
        "trick_card_ids": np.zeros(5, dtype=np.uint8),
    }


class TestLeadFollowRouting:
    def test_is_lead_decision_reads_the_relative_leader(self):
        assert is_lead_decision(state_with_leader(1))
        assert not is_lead_decision(state_with_leader(3))

    def test_three_way_routing(self):
        bid, follow, lead = TagAgent("bid"), TagAgent("follow"), TagAgent("lead")
        chimera = HeadRoutedAgent(bid, follow, lead_agent=lead)
        assert chimera.act(state_with_leader(1), [PICK, PASS], 1)[0] == "bid"
        assert chimera.act(state_with_leader(1), PLAY, 1)[0] == "lead"
        assert chimera.act(state_with_leader(4), PLAY, 1)[0] == "follow"
        # every underlying agent advanced exactly once per decision
        assert (bid.acts, follow.acts, lead.acts) == (3, 3, 3)

    def test_without_lead_agent_all_play_goes_to_play_agent(self):
        bid, play = TagAgent("bid"), TagAgent("play")
        chimera = HeadRoutedAgent(bid, play)
        assert chimera.act(state_with_leader(1), PLAY, 1)[0] == "play"
        assert chimera.act(state_with_leader(2), PLAY, 1)[0] == "play"

    def test_shared_agent_is_not_advanced_twice(self):
        shared, other = TagAgent("shared"), TagAgent("other")
        chimera = HeadRoutedAgent(shared, other, lead_agent=shared)
        chimera.reset_recurrent_state()
        chimera.observe({}, player_id=2)
        assert chimera.act(state_with_leader(1), PLAY, 1)[0] == "shared"
        assert chimera.act(state_with_leader(2), PLAY, 1)[0] == "other"
        assert (shared.acts, shared.resets, shared.observes) == (2, 1, 1)
        assert (other.acts, other.resets, other.observes) == (2, 1, 1)

    def test_contract_covers_what_any_routed_agent_reads(self):
        """The harness observes through observation_for, which reads the
        contract off the acting agent (the chimera). A legacy member must
        pull the picker-memory keys in for everyone, and a member that reads
        the pre-play leader must keep it in the shared observation."""
        from sheepshead.agent.observation import (
            LEGACY_PICKER_MEMORY,
            LEGACY_SEATLESS_RECALL,
            legacy_observation,
        )

        class Legacy(TagAgent):
            legacy_observation = LEGACY_PICKER_MEMORY

        class Seatless(TagAgent):
            legacy_observation = LEGACY_SEATLESS_RECALL

        assert legacy_observation(HeadRoutedAgent(TagAgent("b"), TagAgent("p"))) is None

        seatless = legacy_observation(HeadRoutedAgent(Seatless("b"), Seatless("p")))
        assert seatless == LEGACY_SEATLESS_RECALL

        with_legacy = legacy_observation(
            HeadRoutedAgent(Legacy("b"), Seatless("p"), lead_agent=Seatless("l"))
        )
        assert with_legacy.picker_memory is True
        assert with_legacy.hides_pre_play_leader is True

        mixed = legacy_observation(HeadRoutedAgent(Seatless("b"), TagAgent("p")))
        assert mixed.picker_memory is False
        assert mixed.hides_pre_play_leader is False


CAND = "runs/policy_iteration_202609/iter11/distill_epoch7.pt"
ANCHOR = "runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt"


@pytest.mark.slow
def test_routed_h2h_runs_on_legacy_checkpoints():
    """The routed cert read on real (legacy-observation) checkpoints; the
    iteration-3 cert crashed here when the chimera was fed the clean
    observation. Skipped when the checkpoints are absent (CI)."""
    if not (os.path.exists(CAND) and os.path.exists(ANCHOR)):
        pytest.skip("real checkpoints not available")
    res = routed_h2h(
        ANCHOR, CAND, ANCHOR, n_deals_per_mode=4, n_boot=20, lead_ckpt=CAND, workers=2
    )
    assert np.isfinite(res["edge"]) and np.isfinite(res["se"])
