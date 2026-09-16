"""Lead-vs-follow routing in the head-routed h2h chimera (§20.12 stall
diagnosis)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from sheepshead import ACTION_IDS, ACTIONS
from sheepshead.analysis.head_routed_h2h import (
    HeadRoutedAgent,
    is_lead_decision,
    routed_h2h,
)

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
    return {"leader_rel": np.uint8(rel), "trick_card_ids": np.zeros(5, dtype=np.uint8)}


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

    def test_picker_memory_need_delegates_to_any_routed_agent(self):
        """The harness observes through observation_for, which reads
        needs_picker_memory off the acting agent (the chimera). A legacy
        member must pull the picker-memory keys in for everyone."""
        from sheepshead.agent.observation import needs_picker_memory

        class Legacy(TagAgent):
            needs_picker_memory = True

        class Clean(TagAgent):
            needs_picker_memory = False

        assert (
            needs_picker_memory(HeadRoutedAgent(TagAgent("b"), TagAgent("p"))) is False
        )
        assert needs_picker_memory(HeadRoutedAgent(Clean("b"), Clean("p"))) is False
        assert needs_picker_memory(HeadRoutedAgent(Clean("b"), Legacy("p"))) is True
        assert (
            needs_picker_memory(
                HeadRoutedAgent(Legacy("b"), Clean("p"), lead_agent=Clean("l"))
            )
            is True
        )


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
