#!/usr/bin/env python3
"""Per-architecture golden fingerprints (architecture bit-identity gate).

Captures, for every registered architecture, three fingerprints of a
freshly seeded agent plus deterministic forward-pass outputs:

  * structural  — sorted state_dict key names per network, plus parameter
    counts and the aux-head flags. Torch-version and platform independent.
    Committed as sheepshead/tests/fixtures/arch_structure.json and checked
    by every pytest run.
  * weights     — sha256 over the raw parameter/buffer bytes in state_dict
    order. Catches construction-order or RNG-consumption drift that keeps
    the key set identical.
  * numerical   — encoder/actor/critic outputs on a fixed probe set of
    seeded game states, compared with torch.equal (bit identity, not
    allclose).

Weights and outputs depend on the BLAS/LAPACK the torch build links
(orthogonal init runs a QR factorization; every matmul goes through BLAS).
On macOS that is the system Accelerate framework, so an OS update changes
them with no code change. They are therefore never committed: the
``sheepshead.devtools.check_goldens`` driver captures them at a baseline
commit and at the working tree, on the same machine, and compares the two.

The capture-to-directory CLI below is what that driver runs at *older*
commits, so its interface (``--fixture-dir DIR [--arch NAME]...``, one
``<arch>.pt`` per architecture) and the fingerprint dict layout must stay
stable.

Usage:
    uv run python -m sheepshead.devtools.capture_arch_goldens --write-structure
    uv run python -m sheepshead.devtools.capture_arch_goldens --fixture-dir DIR
    uv run python -m sheepshead.devtools.check_goldens     # the bit-exact gate
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch

from sheepshead.agent import ppo
from sheepshead.agent.architectures.registry import available_architectures
from sheepshead.agent.observation import observation_for
from sheepshead.agent.ppo import PPOAgent
from sheepshead.game import ACTIONS, Game
from sheepshead.training.training_utils import set_all_seeds

SEED = 42
STRUCTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "fixtures",
    "arch_structure.json",
)
NETS = ("encoder", "actor", "critic")


def _seed_all(s: int) -> None:
    set_all_seeds(s)


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


def collect_probe_states(agent) -> list:
    """Deterministic probe observations shared by every architecture:
    three fresh pre-pick states plus three mid-game states at different
    depths/seats of one deterministically advanced deal, observed as
    ``agent`` would see them (legacy agents get the picker memory merged
    in, exactly as the goldens were captured)."""
    states = []
    for seed in (126, 130, 144):
        g = Game(seed=seed)
        states.append(observation_for(g.players[0], agent))
    g = Game(seed=137)
    for extra, pos in ((3, 1), (5, 2), (10, 4)):
        _advance_deterministically(g, extra)
        states.append(observation_for(g.players[pos], agent))
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
    states = collect_probe_states(agent)
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
        out["seq.critic.values"] = (
            agent.critic.sequence_values(seq_out).detach().clone()
        )

        if agent.critic.has_aux_heads:
            out["seq.critic.aux_features"] = (
                agent.critic.aux_sequence_features(seq_out).detach().clone()
            )
            win, exp_ret, secret, points = agent.critic.aux_predictions(enc_out)
            out["aux.predictions"] = torch.tensor(
                [win, exp_ret, secret, *points], dtype=torch.float64
            )
    return out


def structure_record(agent: PPOAgent) -> dict:
    """The platform-independent part of the fingerprint: what is committed
    to arch_structure.json and checked by the pytest suite."""
    nets = {"encoder": agent.encoder, "actor": agent.actor, "critic": agent.critic}
    return {
        "keys": {n: sorted(net.state_dict().keys()) for n, net in nets.items()},
        "meta": {
            "has_aux_heads": bool(agent.critic.has_aux_heads),
            "spec_has_aux_heads": bool(agent.arch_spec.has_aux_heads),
            "d_model": int(agent.encoder.d_model),
            "param_counts": {
                n: sum(p.numel() for p in net.parameters()) for n, net in nets.items()
            },
        },
    }


def capture_arch(arch: str) -> dict:
    agent = build_agent(arch)
    nets = {"encoder": agent.encoder, "actor": agent.actor, "critic": agent.critic}
    structure = structure_record(agent)
    return {
        "arch": arch,
        "key_sha": {n: _key_sha(net) for n, net in nets.items()},
        "keys": structure["keys"],
        "weight_sha": {n: _weight_sha(net) for n, net in nets.items()},
        "meta": structure["meta"],
        "outputs": capture_outputs(agent),
    }


def fixture_path(arch: str, fixture_dir: str) -> str:
    return os.path.join(fixture_dir, f"{arch}.pt")


def load_structure() -> dict:
    with open(STRUCTURE_PATH) as f:
        return json.load(f)


def compare_arch(golden: dict, current: dict) -> list:
    """Mismatch descriptions between two capture_arch fingerprints
    (empty = bit-identical)."""
    problems = []
    for net in NETS:
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write-structure",
        action="store_true",
        help=f"rewrite the committed structural fixture ({STRUCTURE_PATH})",
    )
    mode.add_argument(
        "--fixture-dir",
        help="capture full fingerprints (weights + outputs) into this directory",
    )
    parser.add_argument(
        "--arch",
        action="append",
        help="limit to specific architecture(s); default: all registered",
    )
    args = parser.parse_args()

    if args.write_structure:
        if args.arch:
            parser.error("--write-structure always covers every architecture")
        structure = {
            arch: structure_record(build_agent(arch))
            for arch in available_architectures()
        }
        with open(STRUCTURE_PATH, "w") as f:
            json.dump(structure, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {len(structure)} architectures to {STRUCTURE_PATH}")
        return 0

    if ppo.device.type != "cpu":
        print(
            f"Goldens must be captured on CPU (ppo.device={ppo.device}); "
            'set CUDA_VISIBLE_DEVICES="".'
        )
        return 2
    torch.set_num_threads(1)

    archs = args.arch or available_architectures()
    os.makedirs(args.fixture_dir, exist_ok=True)
    for arch in archs:
        torch.save(capture_arch(arch), fixture_path(arch, args.fixture_dir))
        print(f"captured {arch}")
    with open(os.path.join(args.fixture_dir, "manifest.json"), "w") as f:
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
