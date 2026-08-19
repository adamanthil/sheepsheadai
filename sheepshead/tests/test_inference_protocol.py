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
    decode_request,
    fingerprint_weights,
    masked_actor_probs,
    merge_requests,
    pack_request,
    pack_response,
    unpack_request,
    unpack_response,
)
from sheepshead.inference.protocol import request_nbytes, response_nbytes
from sheepshead.inference.server import ServedModel, run_batch, serve_round
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


def test_fused_device_unpack_matches_the_cpu_path(agent, states):
    """unpack_request(device=...) fuses the inbound transfers -- four copies
    instead of thirteen -- by slicing the contiguous uint8 region on the far
    side. It must decode to exactly what the unfused path decodes."""
    marshalled = agent.encoder.marshal_batch(states)
    memory = torch.randn(len(states), agent.encoder.d_model)
    valid = _valid_lists(agent, states)
    wire = WireConfig(half=False)
    blob = pack_request(marshalled, memory, valid, agent.action_size, True, wire)

    plain = unpack_request(blob, agent.encoder.d_model)
    fused = unpack_request(blob, agent.encoder.d_model, device=torch.device("cpu"))

    for key in plain["marshalled"]:
        assert torch.equal(fused["marshalled"][key], plain["marshalled"][key]), key
    assert torch.equal(fused["masks"], plain["masks"])
    assert torch.equal(fused["memory_in"], plain["memory_in"])


def test_pack_response_agrees_across_torch_and_numpy(agent, states):
    """The torch path concatenates on-device and crosses the bus once; the numpy
    path is the reference. They must produce identical bytes."""
    n = len(states)
    torch.manual_seed(0)
    probs = torch.rand(n, agent.action_size)
    values = torch.rand(n)
    memory_out = torch.randn(n, agent.encoder.d_model)
    for half in (True, False):
        wire = WireConfig(half=half)
        from_torch = pack_response(probs, values, memory_out, wire)
        from_numpy = pack_response(
            probs.numpy(), values.numpy(), memory_out.numpy(), wire
        )
        assert from_torch == from_numpy, half
        assert len(from_torch) == response_nbytes(
            n, agent.action_size, agent.encoder.d_model, wire
        )
        got_p, got_v, got_m = unpack_response(
            from_torch, n, agent.action_size, agent.encoder.d_model, wire
        )
        tol = 1e-3 if half else 0
        assert np.abs(got_p - probs.numpy()).max() <= tol
        assert np.abs(got_v - values.numpy()).max() <= tol
        assert np.abs(got_m - memory_out.numpy()).max() <= (1e-2 if half else 0)


def _request(agent, states, valid, wants_critic=True, half=False, seed=0):
    """A packed request frame for ``states``, decoded back to a RawRequest."""
    torch.manual_seed(seed)
    memory = torch.randn(len(states), agent.encoder.d_model)
    blob = pack_request(
        agent.encoder.marshal_batch(states),
        memory,
        valid,
        agent.action_size,
        wants_critic,
        WireConfig(half=half),
    )
    return blob, decode_request(blob, agent.encoder.d_model)


#: Deliberately ragged: equal-sized sub-batches would hide an off-by-one in the
#: row split, and production rounds are 96-1024 states with no relation to each
#: other's size.
_SHAPES = ((0, 1), (1, 3), (3, 4))


def test_merging_reproduces_the_individual_unpacks(agent, states):
    """The merged batch must be the row-concatenation of its parts. If field
    interleaving were wrong -- concatenating whole ids_flat regions instead of
    merging field by field -- one client's hand_ids would land in another's
    trick slots and the forward would still run, silently."""
    parts = [
        _request(agent, states[a:b], [[1, 2, 5, 9]] * (b - a), seed=i)
        for i, (a, b) in enumerate(_SHAPES)
    ]
    merged = merge_requests([raw for _blob, raw in parts])
    singles = [unpack_request(blob, agent.encoder.d_model) for blob, _raw in parts]

    assert merged["rows"] == [b - a for a, b in _SHAPES]
    for key in singles[0]["marshalled"]:
        expected = torch.cat([s["marshalled"][key] for s in singles], dim=0)
        assert torch.equal(merged["marshalled"][key], expected), key
    for key in ("memory_in", "masks"):
        expected = torch.cat([s[key] for s in singles], dim=0)
        assert torch.equal(merged[key], expected), key


def test_a_merged_batch_returns_each_client_its_own_rows(agent, states):
    """The whole feature is invisible to clients or it is a bug. Exact equality
    is not promised -- merged batches change GEMM tiling, which is why a
    server-backed run can never be golden-pinned -- but the divergence must stay
    far below the CE teacher's 0.026 Q shrinkage floor."""
    device = torch.device("cpu")
    valid = {0: [1, 2, 5, 9], 1: [3, 4, 7], 2: [2, 6, 8, 10, 11]}
    parts = [
        _request(agent, states[a:b], [valid[i]] * (b - a), seed=i)
        for i, (a, b) in enumerate(_SHAPES)
    ]
    batched = run_batch(
        ServedModel(agent, device), [raw for _blob, raw in parts], device
    )
    alone = [serve_round(agent, blob, device) for blob, _raw in parts]

    assert len(batched) == len(alone)
    for i, ((a, b), got, want) in enumerate(zip(_SHAPES, batched, alone)):
        assert len(got) == len(want), i
        shape = (b - a, agent.action_size, agent.encoder.d_model)
        wire = WireConfig(half=False)
        got_probs, got_values, got_memory = unpack_response(got, *shape, wire)
        want_probs, want_values, want_memory = unpack_response(want, *shape, wire)
        assert np.abs(got_probs - want_probs).max() < 1e-5, i
        assert np.abs(got_values - want_values).max() < 1e-5, i
        assert np.abs(got_memory - want_memory).max() < 1e-5, i


def test_a_client_that_did_not_ask_for_the_critic_still_gets_zeros(agent, states):
    """A merged batch runs the critic if *anyone* asked. The clients that did
    not must be unable to tell -- LocalBackend hands them zeros."""
    device = torch.device("cpu")
    wants = (True, False, True)
    parts = [
        _request(agent, states[a:b], [[1, 2, 5, 9]] * (b - a), wants_critic=w, seed=i)
        for i, ((a, b), w) in enumerate(zip(_SHAPES, wants))
    ]
    replies = run_batch(
        ServedModel(agent, device), [raw for _blob, raw in parts], device
    )
    for (a, b), want, reply in zip(_SHAPES, wants, replies):
        _probs, values, _memory = unpack_response(
            reply, b - a, agent.action_size, agent.encoder.d_model, WireConfig(False)
        )
        assert (np.abs(values) > 0).any() == want


def test_incompatible_requests_are_not_merged(agent, states):
    """Mixed wire dtypes would make the response block's column dtype ambiguous,
    so they must raise here rather than corrupt one client's reply."""
    _blob_a, fp32 = _request(agent, states, [[1, 2]] * len(states), half=False)
    _blob_b, fp16 = _request(agent, states, [[1, 2]] * len(states), half=True)
    assert fp32.key != fp16.key
    with pytest.raises(ValueError, match="different"):
        merge_requests([fp32, fp16])


def test_collect_defers_an_incompatible_request_instead_of_dropping_it(agent, states):
    """A dropped request strands a client forever: it is blocked on an Event
    only the batcher sets. The batcher must hand it back to the queue."""
    import queue as queue_module

    from sheepshead.inference.server import BatchPolicy, _collect, _Pending

    valid = [[1, 2]] * len(states)
    pendings = [
        _Pending(_request(agent, states, valid, half=half)[1])
        for half in (False, True, False)
    ]
    work = queue_module.Queue()
    for pending in pendings[1:]:
        work.put(pending)

    batch = _collect(work, pendings[0], BatchPolicy())
    assert batch == [pendings[0]]  # stopped at the fp16 request
    assert work.qsize() == 2  # and both survivors are still queued


def test_collect_respects_the_state_cap(agent, states):
    """Merged batches are capped so one huge round cannot blow up accelerator
    memory; the cap is checked before admitting, so it is a soft ceiling."""
    import queue as queue_module

    from sheepshead.inference.server import BatchPolicy, _collect, _Pending

    valid = [[1, 2]] * len(states)
    pendings = [_Pending(_request(agent, states, valid)[1]) for _ in range(4)]
    work = queue_module.Queue()
    for pending in pendings[1:]:
        work.put(pending)

    # states per request is len(states); a cap just above one request admits two
    batch = _collect(work, pendings[0], BatchPolicy(max_states=len(states) + 1))
    assert len(batch) == 2
    assert work.qsize() == 2


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


def test_batch_padding_does_not_change_any_client_result(agent, states):
    """Compiled mode rounds merged batches up to a fixed set of shapes. The pad
    rows must be invisible: they carry an all-legal mask (an all-False row
    softmaxes to NaN and would poison the real rows through nothing at all --
    but a wrong slice would), and they must be sliced off before the split."""
    device = torch.device("cpu")
    parts = [
        _request(agent, states[a:b], [[1, 2, 5, 9]] * (b - a), seed=i)
        for i, (a, b) in enumerate(_SHAPES)
    ]
    raws = [raw for _blob, raw in parts]
    plain = run_batch(ServedModel(agent, device), raws, device)
    padded = run_batch(ServedModel(agent, device, granularity=16), raws, device)

    assert [len(r) for r in padded] == [len(r) for r in plain]
    for i, ((a, b), got, want) in enumerate(zip(_SHAPES, padded, plain)):
        shape = (b - a, agent.action_size, agent.encoder.d_model)
        wire = WireConfig(half=False)
        for k, (x, y) in enumerate(
            zip(unpack_response(got, *shape, wire), unpack_response(want, *shape, wire))
        ):
            assert np.abs(x - y).max() < 1e-5, (i, k)
