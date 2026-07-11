#!/usr/bin/env python3
"""Per-architecture golden fixtures for the registry refactor.

Captures, for every registered architecture, three fingerprints of a
freshly seeded agent plus deterministic forward-pass outputs:

  * structural  — sha256 over sorted state_dict key names per network.
    Torch-version and platform independent; any drift means the module
    structure changed.
  * weights     — sha256 over the raw parameter/buffer bytes in state_dict
    order. Catches construction-order or RNG-consumption drift that keeps
    the key set identical. Same-machine/same-torch only (orthogonal init
    goes through LAPACK).
  * numerical   — encoder/actor/critic outputs on a fixed probe set of
    seeded game states, compared with torch.equal (bit identity, not
    allclose). Same-machine/same-torch only.

The fixtures gate the architecture-registry refactor: capture once at the
pre-refactor commit, then run --check after every structural change.

Usage:
    uv run python analysis/capture_arch_goldens.py            # write fixtures
    uv run python analysis/capture_arch_goldens.py --check    # compare
    uv run python analysis/capture_arch_goldens.py --check --arch full

tests/test_arch_golden.py imports this module and re-runs the comparison
under pytest (numerical/weight checks are skipped there when the runtime
torch version does not match the manifest).
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead.agent import architectures
from sheepshead.agent import ppo
from sheepshead.agent.ppo import PPOAgent
from sheepshead import ACTIONS, Game

SEED = 42
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "fixtures",
    "arch_golden",
)


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _advance_deterministically(game: Game, n_actions: int) -> None:
    """Advance a game by n_actions using a deterministic action rule
    (median valid action id), mirroring Game.play_random's turn loop."""
    done = 0
    while not game.is_done() and done < n_actions:
        progressed = False
        for player in game.players:
            acts = sorted(player.get_valid_action_ids())
            while acts and done < n_actions:
                player.act(acts[len(acts) // 2])
                done += 1
                progressed = True
                acts = sorted(player.get_valid_action_ids())
            if done >= n_actions:
                return
        if not progressed:
            return


def collect_probe_states() -> list:
    """Deterministic probe observations shared by every architecture:
    three fresh pre-pick states plus three mid-game states at different
    depths/seats of one deterministically advanced deal."""
    states = []
    for seed in (126, 130, 144):
        g = Game(seed=seed)
        states.append(g.players[0].get_state_dict())
    g = Game(seed=137)
    for extra, pos in ((3, 1), (5, 2), (10, 4)):
        _advance_deterministically(g, extra)
        states.append(g.players[pos].get_state_dict())
    return states


def _key_sha(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k in sorted(net.state_dict().keys()):
        h.update(k.encode())
    return h.hexdigest()


def _weight_sha(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_agent(arch: str) -> PPOAgent:
    _seed_all(SEED)
    return PPOAgent(len(ACTIONS), arch=arch)


def capture_outputs(agent: PPOAgent) -> dict:
    """Deterministic forward-pass outputs over the probe states.

    Chains memory_out -> memory_in across states so the recurrent path is
    exercised with nonzero memory, and includes a two-sequence
    encode_sequences pass for the sequence seams.
    """
    states = collect_probe_states()
    mask = torch.ones(1, len(ACTIONS), dtype=torch.bool)
    out: dict = {}
    with torch.no_grad():
        memory = None
        enc_out = None
        for i, s in enumerate(states):
            enc_out = agent.encoder.encode_batch([s], memory_in=memory)
            memory = enc_out["memory_out"]
            for k, v in enc_out.items():
                if isinstance(v, torch.Tensor):
                    out[f"s{i}.enc.{k}"] = v.detach().clone()
            hand_ids = torch.as_tensor(s["hand_ids"], dtype=torch.long).view(1, -1)
            _, logits = agent.actor.forward_with_logits(
                enc_out, mask, hand_ids, agent.encoder.card
            )
            out[f"s{i}.actor.logits"] = logits.detach().clone()
            out[f"s{i}.critic.value"] = agent.critic(enc_out).detach().clone()

        seq_out = agent.encoder.encode_sequences(
            [[states[0], states[1], states[2]], [states[3]]]
        )
        for k, v in seq_out.items():
            if isinstance(v, torch.Tensor):
                out[f"seq.enc.{k}"] = v.detach().clone()
        out["seq.critic.values"] = agent.critic.sequence_values(seq_out).detach().clone()

        if agent.critic.has_aux_heads:
            out["seq.critic.aux_features"] = (
                agent.critic.aux_sequence_features(seq_out).detach().clone()
            )
            win, exp_ret, secret, points = agent.critic.aux_predictions(enc_out)
            out["aux.predictions"] = torch.tensor(
                [win, exp_ret, secret, *points], dtype=torch.float64
            )
    return out


def capture_arch(arch: str) -> dict:
    agent = build_agent(arch)
    nets = {"encoder": agent.encoder, "actor": agent.actor, "critic": agent.critic}
    return {
        "arch": arch,
        "key_sha": {n: _key_sha(net) for n, net in nets.items()},
        "keys": {n: sorted(net.state_dict().keys()) for n, net in nets.items()},
        "weight_sha": {n: _weight_sha(net) for n, net in nets.items()},
        "meta": {
            "has_aux_heads": bool(agent.critic.has_aux_heads),
            "spec_has_aux_heads": bool(agent.arch_spec.has_aux_heads),
            "d_model": int(agent.encoder.d_model),
            "param_counts": {
                n: sum(p.numel() for p in net.parameters()) for n, net in nets.items()
            },
        },
        "outputs": capture_outputs(agent),
    }


def _fixture_path(arch: str, fixture_dir: str) -> str:
    return os.path.join(fixture_dir, f"{arch}.pt")


def manifest_path(fixture_dir: str = FIXTURE_DIR) -> str:
    return os.path.join(fixture_dir, "manifest.json")


def load_manifest(fixture_dir: str = FIXTURE_DIR) -> dict:
    with open(manifest_path(fixture_dir)) as f:
        return json.load(f)


def runtime_matches_manifest(manifest: dict) -> bool:
    """Weight/numerical goldens are only meaningful on the environment that
    captured them (LAPACK/BLAS-dependent init and kernels)."""
    return (
        manifest["torch"] == torch.__version__
        and manifest["platform"] == platform.platform()
    )


def check_arch(arch: str, fixture_dir: str = FIXTURE_DIR) -> list:
    """Recompute the fixture and return a list of mismatch descriptions
    (empty = bit-identical)."""
    golden = torch.load(_fixture_path(arch, fixture_dir), weights_only=True)
    current = capture_arch(arch)
    problems = []
    for net in ("encoder", "actor", "critic"):
        if golden["key_sha"][net] != current["key_sha"][net]:
            gk = set(golden["keys"][net])
            ck = set(current["keys"][net])
            problems.append(
                f"{net}: state_dict keys drifted "
                f"(missing={sorted(gk - ck)} added={sorted(ck - gk)})"
            )
        elif golden["weight_sha"][net] != current["weight_sha"][net]:
            problems.append(f"{net}: weight bytes drifted (RNG/construction order)")
    for field in ("has_aux_heads", "spec_has_aux_heads", "d_model", "param_counts"):
        if golden["meta"][field] != current["meta"][field]:
            problems.append(
                f"meta.{field}: {golden['meta'][field]} -> {current['meta'][field]}"
            )
    g_out, c_out = golden["outputs"], current["outputs"]
    for name in sorted(set(g_out) | set(c_out)):
        if name not in g_out:
            problems.append(f"output {name}: new (not in golden)")
        elif name not in c_out:
            problems.append(f"output {name}: gone (in golden only)")
        elif not torch.equal(g_out[name], c_out[name]):
            diff = (g_out[name].double() - c_out[name].double()).abs().max().item()
            problems.append(f"output {name}: values drifted (max abs diff {diff:g})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against existing fixtures instead of writing them",
    )
    parser.add_argument(
        "--arch",
        action="append",
        help="limit to specific architecture(s); default: all registered",
    )
    parser.add_argument("--fixture-dir", default=FIXTURE_DIR)
    args = parser.parse_args()

    if ppo.device.type != "cpu":
        print(
            f"Goldens must be captured/checked on CPU (ppo.device={ppo.device}); "
            'set CUDA_VISIBLE_DEVICES="".'
        )
        return 2
    torch.set_num_threads(1)

    archs = args.arch or architectures.available_architectures()

    if args.check:
        manifest = load_manifest(args.fixture_dir)
        if not runtime_matches_manifest(manifest):
            print(
                "Runtime does not match fixture manifest "
                f"(torch {manifest['torch']} vs {torch.__version__}, "
                f"platform {manifest['platform']} vs {platform.platform()}); "
                "weight/numerical comparisons would be meaningless here."
            )
            return 2
        failed = 0
        for arch in archs:
            problems = check_arch(arch, args.fixture_dir)
            if problems:
                failed += 1
                print(f"FAIL {arch}")
                for p in problems:
                    print(f"     {p}")
            else:
                print(f"ok   {arch}")
        print(f"\n{len(archs) - failed}/{len(archs)} architectures bit-identical")
        return 1 if failed else 0

    os.makedirs(args.fixture_dir, exist_ok=True)
    for arch in archs:
        torch.save(capture_arch(arch), _fixture_path(arch, args.fixture_dir))
        print(f"captured {arch}")
    with open(manifest_path(args.fixture_dir), "w") as f:
        json.dump(
            {
                "seed": SEED,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"wrote {len(archs)} fixtures + manifest to {args.fixture_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
