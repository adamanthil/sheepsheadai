#!/usr/bin/env python3
"""Reference-implementation equivalence for PPOAgent's minibatch builders.

``_build_minibatch_tensors`` and ``_flatten_action_steps`` are pure data
marshalling -- gather, pad, and index, no arithmetic -- so a naive
per-event Python loop must produce EXACTLY the same tensors on any
platform. Unlike the bit-exact sha fixtures (which only hold on the
machine that captured them), these tests compare two in-process
computations and therefore run everywhere.

The naive builders below are the specification: they spell out, one event
at a time, which event field lands in which (batch, time) slot, what the
padding fill values are, and which rows survive action-flattening.
"""

import pytest
import torch

from sheepshead import ACTIONS, TRUMP
from sheepshead.agent.ppo import (
    FlattenedActionSteps,
    MinibatchTensors,
    PPOAgent,
    device,
)
from sheepshead.tests.ppo_test_helpers import (
    play_episodes,
    prepare_minibatch_inputs,
    seed_all,
)

SEED = 20260718
N_EPISODES = 4
N_SYNTHETIC_SEARCH_TARGETS = 3


def _inject_search_targets(agent: PPOAgent) -> None:
    """Self-play episodes never carry ISMCTS targets; graft uniform targets
    onto a few action events so the search-target columns are exercised."""
    uniform = [1.0 / agent.action_size] * agent.action_size
    action_events = [e for e in agent.events if e["kind"] == "action"]
    for event in action_events[:N_SYNTHETIC_SEARCH_TARGETS]:
        event["search_target"] = list(uniform)
        event["has_search_target"] = True


@pytest.fixture(scope="module")
def prepared():
    seed_all(SEED)
    agent = PPOAgent(len(ACTIONS), arch="full", critic_mode="limited")
    play_episodes(agent, N_EPISODES, collect_oracle=False, seed0=SEED * 10)
    _inject_search_targets(agent)
    states, masks_t, kinds, segments = prepare_minibatch_inputs(agent)
    return agent, states, masks_t, kinds, segments


def naive_build_minibatch_tensors(agent, batch, states, masks_t, kinds):
    batch_size = len(batch)
    lengths = [seg_end - seg_start + 1 for seg_start, seg_end in batch]
    max_len = max(lengths)
    action_size = agent.action_size

    def zeros(*trailing):
        return torch.zeros(
            (batch_size, max_len, *trailing), dtype=torch.float32, device=device
        )

    states_seqs = [
        [states[i] for i in range(seg_start, seg_end + 1)]
        for seg_start, seg_end in batch
    ]
    masks = torch.ones(
        (batch_size, max_len, action_size), dtype=torch.bool, device=device
    )
    is_action = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    actions = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    old_log_probs = zeros()
    old_values = zeros()
    returns = zeros()
    advantages = zeros()
    win = zeros()
    final_returns = zeros()
    secret = zeros()
    points = zeros(5)
    seen_trump_mask = zeros(len(TRUMP))
    unseen_trump_higher = zeros()
    search_target = zeros(action_size)
    has_search = zeros()

    for b, (seg_start, seg_end) in enumerate(batch):
        for t, i in enumerate(range(seg_start, seg_end + 1)):
            masks[b, t] = masks_t[i]
            if kinds[i] != "action":
                continue
            event = agent.events[i]
            is_action[b, t] = True
            actions[b, t] = event["action"]
            old_log_probs[b, t] = event["log_prob"]
            old_values[b, t] = event["value"]
            returns[b, t] = event["return"]
            advantages[b, t] = event["advantage"]
            win[b, t] = float(event.get("win", 0.0) or 0.0)
            final_returns[b, t] = float(event.get("final_return", 0.0) or 0.0)
            secret[b, t] = float(event.get("secret_partner", 0.0) or 0.0)
            points[b, t] = torch.tensor(
                event.get("points_rel", [0.0] * 5), dtype=torch.float32
            )
            seen_trump_mask[b, t] = torch.tensor(
                event.get("seen_trump_mask") or [0.0] * len(TRUMP),
                dtype=torch.float32,
            )
            unseen_trump_higher[b, t] = float(
                event.get("unseen_trump_higher_than_hand", 0.0) or 0.0
            )
            search_target[b, t] = torch.tensor(
                event.get("search_target") or [0.0] * action_size,
                dtype=torch.float32,
            )
            has_search[b, t] = 1.0 if event.get("has_search_target") else 0.0

    return MinibatchTensors(
        states_seqs,
        masks,
        is_action,
        actions,
        old_log_probs,
        old_values,
        returns,
        advantages,
        win,
        final_returns,
        secret,
        points,
        seen_trump_mask,
        unseen_trump_higher,
        search_target,
        has_search,
    )


def naive_flatten_action_steps(is_action_bt, sources):
    action_rows = [
        (b, t)
        for b in range(is_action_bt.size(0))
        for t in range(is_action_bt.size(1))
        if is_action_bt[b, t]
    ]
    return tuple(
        torch.stack([source[b, t] for b, t in action_rows]) for source in sources
    )


def test_scenario_exercises_ragged_padding_and_search_targets(prepared):
    agent, _states, _masks_t, kinds, segments = prepared
    lengths = {seg_end - seg_start + 1 for seg_start, seg_end in segments}
    assert len(segments) >= 2
    assert len(lengths) >= 2, "all segments same length -- padding not exercised"
    assert sum(
        1 for e in agent.events if e["kind"] == "action" and e.get("has_search_target")
    ) == N_SYNTHETIC_SEARCH_TARGETS
    assert any(kind == "observation" for kind in kinds)


def test_build_minibatch_tensors_matches_naive_reference(prepared):
    agent, states, masks_t, kinds, segments = prepared
    actual = agent._build_minibatch_tensors(segments, states, masks_t, kinds)
    expected = naive_build_minibatch_tensors(agent, segments, states, masks_t, kinds)

    for actual_states, expected_states in zip(
        actual.states_seqs, expected.states_seqs
    ):
        assert len(actual_states) == len(expected_states)
        assert all(a is e for a, e in zip(actual_states, expected_states))

    for field in MinibatchTensors._fields:
        if field == "states_seqs":
            continue
        actual_tensor = getattr(actual, field)
        expected_tensor = getattr(expected, field)
        assert actual_tensor.dtype == expected_tensor.dtype, field
        assert torch.equal(actual_tensor, expected_tensor), field


FLAT_FIELD_SOURCES = {
    "logits_flat": "forward.logits_bt",
    "values_flat": "forward.values_bt",
    "actions_flat": "minibatch.actions_bt",
    "old_log_probs_flat": "minibatch.old_log_probs_bt",
    "old_value_flat": "minibatch.old_value_bt",
    "returns_flat": "minibatch.returns_bt",
    "advantages_flat": "minibatch.advantages_bt",
    "win_logits_flat": "forward.win_logits_bt",
    "returns_pred_flat": "forward.returns_pred_bt",
    "win_labels_flat": "minibatch.win_bt",
    "final_returns_labels_flat": "minibatch.final_returns_bt",
    "secret_logits_flat": "forward.secret_logits_bt",
    "secret_labels_flat": "minibatch.secret_bt",
    "seen_trump_mask_logits_flat": "forward.seen_trump_mask_logits_bt",
    "seen_trump_mask_labels_flat": "minibatch.seen_trump_mask_bt",
    "unseen_trump_higher_than_hand_logits_flat": (
        "forward.unseen_trump_higher_than_hand_logits_bt"
    ),
    "unseen_trump_higher_than_hand_labels_flat": (
        "minibatch.unseen_trump_higher_than_hand_bt"
    ),
    "search_target_flat": "minibatch.search_target_bt",
    "has_search_flat": "minibatch.has_search_bt",
    # Derived field (|valid| > 1 per action row); checked explicitly below.
    "decision_flat": None,
}


def _resolve_source(minibatch, forward, source_path):
    struct_name, field = source_path.split(".")
    return getattr({"minibatch": minibatch, "forward": forward}[struct_name], field)


def test_flatten_action_steps_matches_naive_reference(prepared):
    agent, states, masks_t, kinds, segments = prepared
    tensors = agent._build_minibatch_tensors(segments, states, masks_t, kinds)

    with torch.no_grad():
        forward = agent._forward_vectorized(tensors.states_seqs, tensors.masks_bt)

    actual = agent._flatten_action_steps(tensors, forward)
    assert actual is not None

    assert list(FLAT_FIELD_SOURCES) == list(FlattenedActionSteps._fields)
    direct_fields = [f for f, src in FLAT_FIELD_SOURCES.items() if src is not None]
    sources_in_field_order = [
        _resolve_source(tensors, forward, source_path)
        for source_path in FLAT_FIELD_SOURCES.values()
        if source_path is not None
    ]
    expected = dict(
        zip(
            direct_fields,
            naive_flatten_action_steps(tensors.is_action_bt, sources_in_field_order),
        )
    )

    n_action_rows = int(tensors.is_action_bt.sum())
    for field in direct_fields:
        actual_tensor = getattr(actual, field)
        expected_tensor = expected[field]
        assert actual_tensor.size(0) == n_action_rows, field
        assert actual_tensor.dtype == expected_tensor.dtype, field
        assert torch.equal(actual_tensor, expected_tensor), field

    # decision_flat: True exactly where the action row's mask has >1 valid.
    expected_decision = torch.stack(
        [
            tensors.masks_bt[b, t].sum() > 1
            for b in range(tensors.is_action_bt.size(0))
            for t in range(tensors.is_action_bt.size(1))
            if tensors.is_action_bt[b, t]
        ]
    )
    assert actual.decision_flat.size(0) == n_action_rows
    assert actual.decision_flat.dtype == torch.bool
    assert torch.equal(actual.decision_flat, expected_decision)
