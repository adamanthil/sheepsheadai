"""Tests for the encoder split and the opt-in compiled encoder.

``capture_search_goldens`` pins the search bit-exactly, which covers the eager
path completely and the compiled path not at all: compiled output differs from
eager by ~2.6e-08 by construction, so no golden can ever be captured against it.
The properties that stand in for a golden are pinned here instead -- that the
marshal/encode_tensors split does not change what ``encode_batch`` computes,
and that the padding the compiled wrapper adds to bucket its shapes is
invisible to the caller.

The compiled tests stub ``torch.compile`` out. The bug surface belongs to the
wrapper -- padding every marshalled field, padding the recurrent memory with it,
and slicing the pad rows back off -- and a wrong slice is silent: the caller
indexes results positionally against its own state list, so misaligned rows
would train on another sim's policy rather than raise. Inductor's own numerics
are measured in notebooks/Distributed_Inference_202608.md §5.5, not asserted
here, and paying its codegen on every CI run would buy nothing.
"""

import pytest
import torch

from sheepshead import ACTION_IDS, Game
from sheepshead.agent.compiled_encoder import (
    _DEFAULT_SHAPE_BUDGET,
    allow_shape_specialisation,
    disable_compiled_encoder,
    enable_compiled_encoder,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.training_utils import set_all_seeds

#: Padding cannot change a row's value mathematically -- the encoder is
#: row-independent -- but a padded batch is a differently shaped GEMM and so
#: tiles differently. This bounds that, far below the CE teacher's ~0.026 Q
#: shrinkage floor.
TILING_TOL = 1e-5


@pytest.fixture(scope="module")
def agent():
    set_all_seeds(7)
    return PPOAgent(action_size=len(ACTION_IDS), arch="perceiver-shared-v2")


@pytest.fixture(scope="module")
def states():
    return [Game(seed=seed).players[0].get_state_dict() for seed in range(11, 15)]


@pytest.fixture
def stub_compiler(monkeypatch):
    """``torch.compile`` as an identity, so the wrapper runs without codegen."""
    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    yield seen
    disable_compiled_encoder()


def _encode(agent, states, memory):
    return agent.encoder.encode_batch(states, memory_in=memory, device=None)


def test_encode_tensors_matches_encode_batch(agent, states):
    """The split that lets the compiled wrapper leave marshalling eager must not
    change what encode_batch computes."""
    memory = torch.randn(len(states), agent.encoder.d_model)
    direct = _encode(agent, states, memory)
    via_split = agent.encoder.encode_tensors(
        agent.encoder.marshal_batch(states), memory_in=memory, device=None
    )
    assert set(direct) == set(via_split)
    for key, tensor in direct.items():
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, via_split[key]), key


def test_an_exact_bucket_fit_is_bit_identical(agent, states, stub_compiler):
    """No padding, so nothing about the batch changes -- this is the control for
    the tolerance the padded case needs."""
    memory = torch.randn(len(states), agent.encoder.d_model)
    eager = _encode(agent, states, memory)

    enable_compiled_encoder(granularity=len(states))
    compiled = _encode(agent, states, memory)

    for key, tensor in eager.items():
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(compiled[key], tensor), key


@pytest.mark.parametrize("granularity", [3, 16, 32])
def test_pad_rows_are_sliced_back_off(agent, states, stub_compiler, granularity):
    """The caller indexes results positionally, so the row count and the row
    order both have to survive bucketing."""
    memory = torch.randn(len(states), agent.encoder.d_model)
    eager = _encode(agent, states, memory)

    enable_compiled_encoder(granularity=granularity)
    compiled = _encode(agent, states, memory)

    assert set(compiled) == set(eager)
    for key, tensor in eager.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        assert compiled[key].shape == tensor.shape, key
        if tensor.is_floating_point():
            assert torch.abs(compiled[key] - tensor).max() < TILING_TOL, key
        else:
            # Masks and ids are exact or they are a row-alignment bug.
            assert torch.equal(compiled[key], tensor), key


def test_padding_works_without_an_incoming_memory(agent, states, stub_compiler):
    """memory_in is optional -- encode_tensors allocates zeros for it. The pad
    path must not assume a tensor it was never handed."""
    enable_compiled_encoder(granularity=16)
    compiled = _encode(agent, states, None)
    assert compiled["memory_out"].shape == (len(states), agent.encoder.d_model)


def test_enabling_twice_does_not_pad_twice(agent, states, stub_compiler):
    """A second enable that wrapped the wrapper would pad the padding, and the
    inner slice would hand back the wrong rows."""
    enable_compiled_encoder(granularity=16)
    once = agent.encoder.encode_batch
    enable_compiled_encoder(granularity=16)
    assert agent.encoder.encode_batch.__func__ is once.__func__
    assert len(stub_compiler) == 1


def test_disable_restores_the_eager_method(agent, states, stub_compiler):
    """A process has to be able to get back to a golden-comparable state without
    restarting -- capture_search_goldens cannot run against the patch."""
    original = type(agent.encoder).encode_batch
    enable_compiled_encoder(granularity=16)
    assert type(agent.encoder).encode_batch is not original
    disable_compiled_encoder()
    assert type(agent.encoder).encode_batch is original


def test_the_shape_budget_clears_dynamos_default():
    """dynamo compiles 8 variants per function and then runs eager *without
    raising*. Bucketing leaves 14 shapes, so the default silently drops the tail
    of the size distribution -- which is how a measured 1.41x turned out to be
    1.36x with six shapes never compiled at all."""
    import torch._dynamo as dynamo

    allow_shape_specialisation()
    assert dynamo.config.recompile_limit >= 14
    assert dynamo.config.cache_size_limit >= 14
    assert _DEFAULT_SHAPE_BUDGET >= 14
