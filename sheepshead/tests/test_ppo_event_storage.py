#!/usr/bin/env python3
"""Unit tests for PPOAgent.store_episode_events' raw-event -> record mapping.

This is the seam between rollout collection and the minibatch builders:
which label lands in which record field, the defaults when optional labels
are absent or None, the 1-indexed -> 0-indexed action shift, the done-flag
placement, and the oracle/search-target passthrough rules. Everything
downstream (GAE, minibatch tensors, losses) is tested against these
records, so this mapping is pinned directly with hand-built events.
"""

import pytest
import torch

from sheepshead import ACTIONS, TRUMP
from sheepshead.agent.ppo import PPOAgent


@pytest.fixture(scope="module")
def module_agent():
    return PPOAgent(len(ACTIONS))


@pytest.fixture
def agent(module_agent):
    module_agent.events = []
    return module_agent


def action_event(**overrides):
    event = {
        "kind": "action",
        "state": {"obs": 1},
        "action": 3,
        "log_prob": -0.5,
        "value": 0.25,
        "valid_actions": {1, 3},
        "reward": 1.5,
    }
    event.update(overrides)
    return event


def stored_record_and_mask(agent, index=0):
    record = dict(agent.events[index])
    return record, record.pop("mask")


def test_action_event_full_label_mapping(agent):
    state = {"obs": "decision"}
    seen_mask_label = [1] + [0] * (len(TRUMP) - 1)
    search_target = [0.0] * agent.action_size
    search_target[2] = 1.0
    agent.store_episode_events(
        [
            action_event(
                state=state,
                win_label=1,
                final_return_label=2.5,
                secret_partner_label=1,
                points_label=[1, 2, 3, 4, 5],
                seen_trump_mask_label=seen_mask_label,
                unseen_trump_higher_than_hand_label=1,
                search_target=search_target,
                has_search_target=True,
            )
        ]
    )

    record, mask = stored_record_and_mask(agent)
    assert record == {
        "kind": "action",
        "state": state,
        "action": 2,
        "reward": 1.5,
        "value": 0.25,
        "log_prob": -0.5,
        "done": True,
        "win": 1.0,
        "final_return": 2.5,
        "secret_partner": 1.0,
        "points_rel": [1.0, 2.0, 3.0, 4.0, 5.0],
        "seen_trump_mask": [1.0] + [0.0] * (len(TRUMP) - 1),
        "unseen_trump_higher_than_hand": 1.0,
        "search_target": search_target,
        "has_search_target": True,
    }
    assert record["state"] is state
    assert all(isinstance(x, float) for x in record["points_rel"])
    assert all(isinstance(x, float) for x in record["seen_trump_mask"])
    assert all(isinstance(x, float) for x in record["search_target"])

    expected_mask = torch.zeros(agent.action_size, dtype=torch.bool)
    expected_mask[0] = True
    expected_mask[2] = True
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("labels", [{}, {
    "win_label": None,
    "final_return_label": None,
    "secret_partner_label": None,
    "points_label": None,
    "seen_trump_mask_label": None,
    "unseen_trump_higher_than_hand_label": None,
    "search_target": None,
    "has_search_target": None,
}], ids=["absent", "none"])
def test_optional_labels_default_to_zeros(agent, labels):
    agent.store_episode_events([action_event(**labels)])
    record, _mask = stored_record_and_mask(agent)
    assert record["win"] == 0.0
    assert record["final_return"] == 0.0
    assert record["secret_partner"] == 0.0
    assert record["points_rel"] == [0.0] * 5
    assert record["seen_trump_mask"] == [0.0] * len(TRUMP)
    assert record["unseen_trump_higher_than_hand"] == 0.0
    assert record["search_target"] == [0.0] * agent.action_size
    assert record["has_search_target"] is False


def test_done_set_only_on_last_action_even_before_trailing_observation(agent):
    agent.store_episode_events(
        [
            action_event(),
            action_event(),
            {"kind": "observation", "state": {"obs": "post"}},
        ]
    )
    assert [e["done"] for e in agent.events if e["kind"] == "action"] == [False, True]


def test_observation_record_keeps_state_and_gets_all_ones_mask(agent):
    state = {"obs": "watching"}
    agent.store_episode_events([{"kind": "observation", "state": state}])
    record, mask = stored_record_and_mask(agent)
    assert record == {"kind": "observation", "state": state}
    assert record["state"] is state
    assert mask.dtype == torch.bool
    assert bool(mask.all())
    assert mask.shape == (agent.action_size,)


def test_oracle_state_passes_through_on_both_kinds(agent):
    oracle_state = {"oracle": True}
    agent.store_episode_events(
        [
            {"kind": "observation", "state": {}, "oracle_state": oracle_state},
            action_event(oracle_state=oracle_state),
        ]
    )
    assert agent.events[0]["oracle_state"] is oracle_state
    assert agent.events[1]["oracle_state"] is oracle_state


def test_no_oracle_key_when_not_collected(agent):
    agent.store_episode_events(
        [{"kind": "observation", "state": {}}, action_event()]
    )
    assert all("oracle_state" not in e for e in agent.events)


def test_search_target_dropped_without_confidence_flag(agent):
    confident = [1.0 / agent.action_size] * agent.action_size
    agent.store_episode_events(
        [action_event(search_target=confident, has_search_target=False)]
    )
    record, _mask = stored_record_and_mask(agent)
    assert record["search_target"] == [0.0] * agent.action_size
    assert record["has_search_target"] is False


def test_confidence_flag_dropped_without_search_target(agent):
    agent.store_episode_events([action_event(has_search_target=True)])
    record, _mask = stored_record_and_mask(agent)
    assert record["search_target"] == [0.0] * agent.action_size
    assert record["has_search_target"] is False


def test_appends_across_calls_without_clearing(agent):
    agent.store_episode_events([action_event()])
    agent.store_episode_events([action_event()])
    assert len(agent.events) == 2
    assert [e["done"] for e in agent.events] == [True, True]
