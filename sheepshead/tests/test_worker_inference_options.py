"""The opt-in worker throughput options (--worker-device / --worker-compile).

Both change results in the last bits, so what has to be pinned is that they
reach the worker at all and that they reach it *early enough*. The ordering is
the whole bug class: ``pfsp_runtime`` imports ``ismcts`` at module scope, so a
spawned worker has already imported it before its initializer runs, and
``PPOAgent`` places its networks at construction. Anything that reads a device
later than that is reading it too late.

Compilation itself is stubbed out — ``sheepshead/tests/test_search_encode_path``
covers the wrapper, and paying inductor codegen here would buy nothing.
"""

import argparse
from typing import cast

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent import ppo as ppo_module
from sheepshead.agent.compiled_encoder import disable_compiled_encoder
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
from sheepshead.training import league_worker
from sheepshead.training.league import League
from sheepshead.training.league_cli import build_arg_parser
from sheepshead.training.league_streams import MainPhaseContext

ARCH = "perceiver-shared-v2"


@pytest.fixture
def restore_globals():
    """Both options mutate process-wide state. Put it back or every later test
    in the session inherits it."""
    original_device = ppo_module.device
    yield
    ppo_module.device = original_device
    disable_compiled_encoder()


@pytest.fixture
def stub_compiler(monkeypatch):
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: seen.append(kw) or fn)
    return seen


def test_the_teacher_follows_the_networks_not_a_global(restore_globals):
    """The device is read off the agent at construction, so it cannot disagree
    with where the networks actually live -- and cannot be pinned by an import
    that happened before the process chose."""
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    teacher = ISMCTSTeacher(agent, ISMCTSConfig())
    assert teacher.device == next(agent.encoder.parameters()).device

    # A device chosen after ismcts was imported still lands: the failure this
    # guards is a worker crashing on its first encode with the networks on one
    # device and the search allocating on another.
    ppo_module.device = torch.device("cpu")
    assert ISMCTSTeacher(PPOAgent(len(ACTIONS), arch=ARCH), ISMCTSConfig()).device == (
        torch.device("cpu")
    )


def test_options_are_applied_before_the_agent_is_built(restore_globals, stub_compiler):
    """_apply_inference_options must run ahead of PPOAgent construction, which
    is what makes the device setting effective at all."""
    calls = []
    original = league_worker.PPOAgent

    class RecordingAgent(original):
        def __init__(self, *args, **kwargs):
            calls.append(ppo_module.device)
            super().__init__(*args, **kwargs)

    league_worker.PPOAgent = RecordingAgent
    try:
        league_worker.league_worker_init(
            {
                "arch": ARCH,
                "members_dir": ".",
                "weight_path_base": "unused",
                "base_seed": 0,
                "worker_device": "cpu",
                "worker_compile": "default",
                "worker_compile_granularity": 16,
            }
        )
    finally:
        league_worker.PPOAgent = original

    assert calls == [torch.device("cpu")]  # device was already set
    assert len(stub_compiler) == 1  # and the encoder was already patched


def test_absent_options_change_nothing(restore_globals, stub_compiler):
    """The default path must not touch the device or compile anything -- these
    are opt-in because goldens cannot pass against them."""
    before = ppo_module.device
    league_worker.league_worker_init(
        {
            "arch": ARCH,
            "members_dir": ".",
            "weight_path_base": "unused",
            "base_seed": 0,
        }
    )
    assert ppo_module.device is before
    assert stub_compiler == []


def test_the_flags_reach_the_pool_initargs():
    """The CLI values have to survive the trip into the spawn initargs; a
    worker cannot read the parent's argparse namespace."""
    from sheepshead.training.train_league_ppo import _spawn_worker_pool

    args = build_arg_parser().parse_args(
        [
            "--league-dir",
            ".",
            "--resume",
            "unused.pt",
            "--num-workers",
            "4",
            "--worker-device",
            "mps",
            "--worker-compile",
            "reduce-overhead",
            "--worker-compile-granularity",
            "64",
        ]
    )
    captured = {}

    class FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeContext:
        Pool = FakePool

    import sheepshead.training.train_league_ppo as trainer

    original = trainer.get_context
    trainer.get_context = lambda _name: FakeContext()
    try:
        _spawn_worker_pool(
            args, cast(League, _FakeLeague()), cast(MainPhaseContext, _FakeContext())
        )
    finally:
        trainer.get_context = original

    init_args = captured["initargs"][0]
    assert init_args["worker_device"] == "mps"
    assert init_args["worker_compile"] == "reduce-overhead"
    assert init_args["worker_compile_granularity"] == 64


def test_a_bare_worker_compile_means_default_mode():
    args = build_arg_parser().parse_args(
        ["--league-dir", ".", "--resume", "unused.pt", "--worker-compile"]
    )
    assert args.worker_compile == "default"


def test_the_options_default_to_off():
    args = build_arg_parser().parse_args(["--league-dir", ".", "--resume", "unused.pt"])
    assert args.worker_compile is None
    assert args.worker_device is None


def test_an_inert_flag_is_announced(capsys):
    """With no pool the options do nothing. Silence would be indistinguishable
    from an optimization that simply did not help."""
    from sheepshead.training.train_league_ppo import _spawn_worker_pool

    args = argparse.Namespace(
        num_workers=1, worker_compile="default", worker_device=None
    )
    assert _spawn_worker_pool(args, None, None) is None
    assert "ignored" in capsys.readouterr().out


class _FakeLeague:
    members_dir = "."


class _FakeContext:
    weight_sync = {"base": "unused"}

    class _Agent:
        gamma = 1.0

    training_agent = _Agent()


def test_routed_flag_reaches_the_pool_initargs():
    from sheepshead.training.train_league_ppo import _spawn_worker_pool

    args = build_arg_parser().parse_args(
        [
            "--league-dir",
            ".",
            "--resume",
            "unused.pt",
            "--num-workers",
            "4",
            "--worker-routed-encoder",
        ]
    )
    assert args.worker_routed_encoder == "mps"  # bare flag defaults the device
    captured = {}

    class FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeContext:
        Pool = FakePool

    import sheepshead.training.train_league_ppo as trainer

    original = trainer.get_context
    trainer.get_context = lambda _name: FakeContext()
    try:
        _spawn_worker_pool(
            args, cast(League, _FakeLeague()), cast(MainPhaseContext, _FakeContext())
        )
    finally:
        trainer.get_context = original

    assert captured["initargs"][0]["worker_routed_encoder"] == "mps"


def test_routed_worker_routes_and_stays_on_cpu(restore_globals, stub_compiler):
    """Routing must not touch the process device — the whole point is that
    everything except committee-scale encodes stays eager CPU."""
    from sheepshead.agent.compiled_encoder import disable_routed_encoder

    before = ppo_module.device
    league_worker.league_worker_init(
        {
            "arch": ARCH,
            "members_dir": ".",
            "weight_path_base": "unused",
            "base_seed": 0,
            "worker_routed_encoder": "cpu",  # cpu shadow: no MPS needed in CI
        }
    )
    try:
        assert ppo_module.device is before
        assert len(stub_compiler) == 1  # the shadow graph was compiled
    finally:
        disable_routed_encoder()


def test_routed_and_device_refuse_to_combine():
    import sys

    from sheepshead.training.train_league_ppo import main

    argv = sys.argv
    sys.argv = [
        "train_league_ppo",
        "--league-dir",
        ".",
        "--resume",
        "unused.pt",
        "--worker-routed-encoder",
        "--worker-device",
        "mps",
    ]
    try:
        with pytest.raises(SystemExit, match="mutually exclusive"):
            main()
    finally:
        sys.argv = argv
