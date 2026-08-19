"""Wire-format and backend tests for sheepshead.inference.

The remote path cannot be covered by capture_search_goldens (a server-backed run
is not bit-reproducible by construction), so the guarantees that matter are
pinned here instead: the wire format round-trips losslessly in fp32, the weight
handshake actually discriminates, and the local backend still computes what the
pre-seam code computed.
"""

import numpy as np
import pytest
import torch

from sheepshead import ACTION_IDS, Game
from sheepshead.agent.ppo import PPOAgent
from sheepshead.inference import (
    LocalBackend,
    WireConfig,
    fingerprint_weights,
    masked_actor_probs,
    pack_request,
    pack_response,
    unpack_request,
    unpack_response,
)
from sheepshead.inference.protocol import request_nbytes, response_nbytes
from sheepshead.training.training_utils import set_all_seeds


@pytest.fixture(scope="module")
def agent():
    set_all_seeds(7)
    return PPOAgent(action_size=len(ACTION_IDS), arch="perceiver-shared-v2")


@pytest.fixture(scope="module")
def states():
    out = []
    for seed in (11, 12, 13, 14):
        game = Game(seed=seed)
        out.append(game.players[0].get_state_dict())
    return out


def _valid_lists(agent, states):
    return [[1, 2, 5, 9] for _ in states]


def test_request_round_trips_losslessly_in_fp32(agent, states):
    marshalled = agent.encoder.marshal_batch(states)
    memory = torch.randn(len(states), agent.encoder.d_model)
    valid = _valid_lists(agent, states)
    wire = WireConfig(half=False)

    blob = pack_request(marshalled, memory, valid, agent.action_size, True, wire)
    parsed = unpack_request(blob, agent.encoder.d_model)

    for key, tensor in marshalled.items():
        assert torch.equal(parsed["marshalled"][key], tensor), key
    assert torch.allclose(parsed["memory_in"], memory, atol=0, rtol=0)
    assert parsed["wants_critic"] is True
    assert parsed["action_size"] == agent.action_size
    # masks must reproduce the legal-action sets exactly
    for row, actions in enumerate(valid):
        recovered = {int(i) + 1 for i in torch.nonzero(parsed["masks"][row]).flatten()}
        assert recovered == set(actions)


def test_request_size_matches_the_accounting(agent, states):
    """The payload model in addendum 2 drives the whole go/no-go, so the
    predicted byte count and the real one must not drift."""
    marshalled = agent.encoder.marshal_batch(states)
    memory = torch.randn(len(states), agent.encoder.d_model)
    for half in (True, False):
        wire = WireConfig(half=half)
        blob = pack_request(
            marshalled,
            memory,
            _valid_lists(agent, states),
            agent.action_size,
            False,
            wire,
        )
        predicted = request_nbytes(
            len(states), agent.action_size, agent.encoder.d_model, wire
        )
        assert len(blob) == predicted, (half, len(blob), predicted)


def test_response_round_trips(agent, states):
    n = len(states)
    probs = np.random.rand(n, agent.action_size).astype(np.float32)
    values = np.random.rand(n).astype(np.float32)
    memory_out = np.random.rand(n, agent.encoder.d_model).astype(np.float32)
    wire = WireConfig(half=False)

    blob = pack_response(probs, values, memory_out, wire)
    assert len(blob) == response_nbytes(
        n, agent.action_size, agent.encoder.d_model, wire
    )
    got_probs, got_values, got_memory = unpack_response(
        blob, n, agent.action_size, agent.encoder.d_model, wire
    )
    assert np.array_equal(got_probs, probs)
    assert np.array_equal(got_values, values)
    assert np.array_equal(got_memory, memory_out)


def test_half_precision_stays_far_below_the_shrinkage_noise_floor(agent, states):
    """fp16 on the wire is a deliberate 2x bandwidth win. It is only acceptable
    while its quantisation stays well under the CE teacher's ~0.026 Q noise
    floor, which is what sets abstention."""
    n = len(states)
    probs = np.random.rand(n, agent.action_size).astype(np.float32)
    values = np.random.rand(n).astype(np.float32)
    memory_out = np.random.randn(n, agent.encoder.d_model).astype(np.float32)
    wire = WireConfig(half=True)
    got_probs, got_values, got_memory = unpack_response(
        pack_response(probs, values, memory_out, wire),
        n,
        agent.action_size,
        agent.encoder.d_model,
        wire,
    )
    assert np.abs(got_probs - probs).max() < 1e-3
    assert np.abs(got_values - values).max() < 1e-3
    assert np.abs(got_memory - memory_out).max() < 1e-2


def test_protocol_version_mismatch_is_rejected(agent, states):
    marshalled = agent.encoder.marshal_batch(states)
    memory = torch.zeros(len(states), agent.encoder.d_model)
    blob = bytearray(
        pack_request(
            marshalled,
            memory,
            _valid_lists(agent, states),
            agent.action_size,
            False,
            WireConfig(half=False),
        )
    )
    blob[4:6] = (99).to_bytes(2, "little")  # bump the version field
    with pytest.raises(ValueError, match="protocol version"):
        unpack_request(bytes(blob), agent.encoder.d_model)


def test_weight_fingerprint_discriminates(agent):
    """Serving stale weights raises nowhere and silently yields a plausible,
    wrong search target -- so the handshake has to actually detect it."""
    before = fingerprint_weights(agent.actor)
    assert before == fingerprint_weights(agent.actor)
    param = next(iter(agent.actor.parameters()))
    original = param.detach().clone()
    try:
        with torch.no_grad():
            param.add_(1e-6)
        assert fingerprint_weights(agent.actor) != before
    finally:
        # The agent fixture is module-scoped, so leaving it perturbed would make
        # every later test in this file depend on execution order. Restore the
        # exact bits -- add_ then sub_ does not round-trip in floating point.
        with torch.no_grad():
            param.copy_(original)
    assert fingerprint_weights(agent.actor) == before


def test_local_backend_matches_a_direct_forward(agent, states):
    """LocalBackend is meant to be the pre-seam code, moved and not changed."""
    valid = _valid_lists(agent, states)
    memory = torch.zeros(len(states), agent.encoder.d_model)

    encoded = agent.encoder.encode_batch(states, memory_in=memory, device=None)
    expected_probs = masked_actor_probs(
        agent, encoded, states, valid, agent.action_size, None
    )
    with torch.no_grad():
        expected_values = agent.critic(encoded).detach().view(-1).cpu().numpy()
    expected_memory = encoded["memory_out"].detach()

    result = LocalBackend(None).evaluate(agent, states, memory, valid, True)

    assert np.array_equal(result.probs, expected_probs.detach().cpu().numpy())
    assert np.array_equal(result.values, expected_values)
    assert torch.equal(result.memory_out, expected_memory)


def test_local_backend_skips_the_critic_when_not_asked(agent, states):
    valid = _valid_lists(agent, states)
    memory = torch.zeros(len(states), agent.encoder.d_model)
    result = LocalBackend(None).evaluate(agent, states, memory, valid, False)
    assert np.array_equal(result.values, np.zeros(len(states), dtype=np.float32))


def test_encode_tensors_matches_encode_batch(agent, states):
    """The split that lets a server consume packed arrays must not change what
    encode_batch computes."""
    memory = torch.randn(len(states), agent.encoder.d_model)
    direct = agent.encoder.encode_batch(states, memory_in=memory, device=None)
    via_split = agent.encoder.encode_tensors(
        agent.encoder.marshal_batch(states), memory_in=memory, device=None
    )
    assert set(direct) == set(via_split)
    for key, tensor in direct.items():
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, via_split[key]), key
