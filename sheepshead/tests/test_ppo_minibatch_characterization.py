#!/usr/bin/env python3
"""Bit-exact characterization of PPOAgent minibatch-tensor internals.

Pins the exact, POSITIONALLY-ORDERED outputs of two private helpers used by
PPOAgent.update():

  * ``_build_minibatch_tensors`` (17-tuple): per-segment (B, T, ...) tensors
    consumed by the recurrent forward pass and the loss.
  * ``_flatten_action_steps`` (20-tuple): the same information flattened to
    action-only rows, ready for the actor/critic losses.

update() unpacks both tuples positionally, and an upcoming refactor converts
them to NamedTuples -- this fixture exists so that refactor cannot silently
reorder, drop, or rename an element. Each element is pinned by (type, shape,
dtype, sha256-of-bytes) rather than by value, so the fixture stays readable
and the check stays exact.

Capture strategy: this test calls the real PPOAgent instance methods
directly rather than monkeypatching update(). The preprocessing update()
performs before calling these two methods (GAE, advantage normalization
written back into self.events, `_prepare_training_views`,
`_segments_from_events`) is reproduced verbatim, in the same order, on the
same agent/events -- see `_prepare_minibatch_inputs`, which mirrors
update()'s code from the top of the function down to (but not including)
the epoch/minibatch loop. The one deliberate deviation is the epoch loop's
`torch.randperm` batching: this test instead passes every segment as a
single batch. That shuffle lives in update() itself, not in either target
method, and passing all segments keeps the fixture reproducible without
depending on torch's RNG consumption elsewhere in update().

Both methods are oracle-agnostic (neither reads `oracle_state` /
`value_oracle` / `return_oracle` / `critic_mode`); oracle-mode differences
live entirely in the separate `_build_oracle_minibatch` helper. So a single
config (arch="full", critic_mode="limited") is pinned; see the module
docstring note below for why an oracle-mode run is not also pinned.

Regenerate after an intentional change to these methods:

    uv run python -m sheepshead.tests.test_ppo_minibatch_characterization

Regeneration runs the capture twice in-process and refuses to write a
fixture that does not reproduce itself.
"""

import hashlib
import json
from pathlib import Path

import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import (
    play_episodes,
    prepare_minibatch_inputs,
    runtime_environment,
    seed_all,
    skip_unless_fixture_environment,
)

SEED = 20260716
N_EPISODES = 6

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ppo_minibatch_characterization.json"

BUILD_TENSORS_FIELDS = [
    "states_seqs (per-segment raw obs-dict sequences fed to the encoder)",
    "masks_bt (B,T,action_size) bool -- legal-action mask per step",
    "is_action_bt (B,T) bool -- step is an action (vs. observation) event",
    "actions_bt (B,T) long -- taken action index, -1 padding/non-action",
    "old_lp_bt (B,T) float32 -- behavior-policy log-prob at rollout time",
    "old_value_bt (B,T) float32 -- limited-critic value at rollout time",
    "returns_bt (B,T) float32 -- GAE return target (limited critic)",
    "adv_bt (B,T) float32 -- normalized advantage",
    "lengths_bt (B,) long -- true (unpadded) segment length",
    "win_bt (B,T) float32 -- win aux-head label",
    "final_ret_bt (B,T) float32 -- final-return aux-head label",
    "secret_bt (B,T) float32 -- secret-partner aux-head label",
    "points_bt (B,T,5) float32 -- relative-points aux-head label",
    "seen_trump_mask_bt (B,T,len(TRUMP)) float32 -- seen-trump-mask label",
    "unseen_trump_higher_than_hand_bt (B,T) float32 -- aux-head label",
    "search_target_bt (B,T,action_size) float32 -- ISMCTS policy target (pi')",
    "has_search_bt (B,T) float32 -- confident-search-target flag",
]

FLATTEN_ACTION_STEPS_FIELDS = [
    "logits_flat (N,action_size) float32 -- actor logits at action rows",
    "values_flat (N,) float32 -- critic value prediction",
    "actions_flat (N,) long -- taken action index",
    "old_lp_flat (N,) float32 -- behavior-policy log-prob",
    "old_value_flat (N,) float32 -- rollout-time critic value",
    "returns_flat (N,) float32 -- GAE return target",
    "adv_flat (N,) float32 -- normalized advantage",
    "win_logits_flat (N,) float32 -- win aux-head logit",
    "ret_pred_flat (N,) float32 -- return aux-head prediction",
    "win_labels_flat (N,) float32 -- win label",
    "final_ret_labels_flat (N,) float32 -- final-return label",
    "secret_logits_flat (N,) float32 -- secret-partner aux-head logit",
    "secret_labels_flat (N,) float32 -- secret-partner label",
    "mask_flat (N,action_size) bool -- legal-action mask",
    "seen_trump_mask_logits_flat (N,len(TRUMP)) float32 -- aux-head logits",
    "seen_trump_mask_labels_flat (N,len(TRUMP)) float32 -- aux-head labels",
    "unseen_trump_higher_than_hand_logits_flat (N,) float32 -- aux-head logit",
    "unseen_trump_higher_than_hand_labels_flat (N,) float32 -- aux-head label",
    "search_target_flat (N,action_size) float32 -- ISMCTS policy target (pi')",
    "has_search_flat (N,) float32 -- confident-search-target flag",
]

assert len(BUILD_TENSORS_FIELDS) == 17
assert len(FLATTEN_ACTION_STEPS_FIELDS) == 20


def _stable_repr(value) -> str:
    """JSON repr for non-tensor elements (nested obs dicts of numpy scalars/arrays)."""

    def normalize(v):
        if isinstance(v, dict):
            return {k: normalize(v2) for k, v2 in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [normalize(x) for x in v]
        if hasattr(v, "tolist"):
            return normalize(v.tolist())
        if hasattr(v, "item") and not isinstance(v, (int, float, bool, str)):
            return normalize(v.item())
        return v

    return json.dumps(normalize(value), sort_keys=True)


def _pin_element(value) -> dict:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return {
            "type": "Tensor",
            "shape": list(arr.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, list):
        # states_seqs: list of B per-segment sequences of raw obs dicts.
        return {
            "type": "list",
            "len": len(value),
            "sha256": hashlib.sha256(_stable_repr(value).encode()).hexdigest(),
        }
    raise TypeError(f"unexpected element type: {type(value)!r}")


def _pin_tuple(values, fields) -> list:
    assert len(values) == len(fields)
    return [
        {"index": i, "meaning": meaning, **_pin_element(v)}
        for i, (v, meaning) in enumerate(zip(values, fields))
    ]


def _capture() -> dict:
    torch.set_num_threads(1)
    seed_all(SEED)
    agent = PPOAgent(len(ACTIONS), arch="full", critic_mode="limited")
    play_episodes(agent, N_EPISODES, collect_oracle=False, seed0=SEED * 10)

    states, masks_t, kinds, segments = prepare_minibatch_inputs(agent)
    batch = segments  # every segment as a single batch -- see module docstring

    build_out = agent._build_minibatch_tensors(batch, states, masks_t, kinds)
    (
        states_bt,
        masks_bt,
        is_action_bt,
        actions_bt,
        old_lp_bt,
        old_value_bt,
        returns_bt,
        adv_bt,
        lengths_bt,
        win_bt,
        final_ret_bt,
        secret_bt,
        points_bt,
        seen_trump_mask_bt,
        unseen_trump_higher_than_hand_bt,
        search_target_bt,
        has_search_bt,
    ) = build_out

    with torch.no_grad():
        (
            logits_bt,
            values_bt,
            win_logits_bt,
            ret_pred_bt,
            secret_logits_bt,
            _points_pred_bt,
            seen_trump_mask_logits_bt,
            unseen_trump_higher_than_hand_logits_bt,
        ) = agent._forward_vectorized(states_bt, masks_bt, lengths_bt)

    flat_out = agent._flatten_action_steps(
        is_action_bt,
        logits_bt,
        values_bt,
        actions_bt,
        old_lp_bt,
        old_value_bt,
        returns_bt,
        adv_bt,
        win_logits_bt,
        ret_pred_bt,
        win_bt,
        final_ret_bt,
        secret_logits_bt,
        secret_bt,
        masks_bt,
        seen_trump_mask_logits_bt,
        seen_trump_mask_bt,
        unseen_trump_higher_than_hand_logits_bt,
        unseen_trump_higher_than_hand_bt,
        search_target_bt,
        has_search_bt,
    )
    assert flat_out is not None, "no action rows captured -- check episode config"

    return {
        "build_minibatch_tensors": _pin_tuple(build_out, BUILD_TENSORS_FIELDS),
        "flatten_action_steps": _pin_tuple(flat_out, FLATTEN_ACTION_STEPS_FIELDS),
    }


def _load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def test_build_minibatch_tensors_is_bit_identical():
    fixture = _load_fixture()
    skip_unless_fixture_environment(fixture)
    expected = fixture["build_minibatch_tensors"]
    actual = json.loads(json.dumps(_capture()["build_minibatch_tensors"]))
    assert actual == expected


def test_flatten_action_steps_is_bit_identical():
    fixture = _load_fixture()
    skip_unless_fixture_environment(fixture)
    expected = fixture["flatten_action_steps"]
    actual = json.loads(json.dumps(_capture()["flatten_action_steps"]))
    assert actual == expected


def test_element_counts_match_documented_arity():
    # The point of this fixture is element ORDER/COUNT surviving a refactor
    # to NamedTuples; guard the two arities directly against the docstring.
    fixture = _load_fixture()
    assert len(fixture["build_minibatch_tensors"]) == 17
    assert len(fixture["flatten_action_steps"]) == 20


def _regenerate() -> None:
    first = _capture()
    second = _capture()
    first_json = json.loads(json.dumps(first))
    second_json = json.loads(json.dumps(second))
    if first_json != second_json:
        raise SystemExit(
            "_build_minibatch_tensors/_flatten_action_steps are not "
            "deterministic in-process"
        )
    first_json["environment"] = runtime_environment()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(first_json, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    _regenerate()
