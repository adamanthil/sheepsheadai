#!/usr/bin/env python3
"""Golden fixtures for the ISMCTS teacher (search-behavior bit-identity gate).

Captures, on a fixed seeded panel of decision nodes (pick / partner / bury /
play / leaster play), the complete ``ISMCTSTeacher.search`` result under a
small matrix of config arms (PUCT vs RM root, oracle vs limited leaves,
batch size 1 vs 32, seat-policy grounding, terminal-depth rollout), plus a
pool-level fixture of the lockstep world-build log-weights.

Every quantity is compared bit-exactly (torch.equal on arrays, exact repr on
scalars/dicts) — the gate for the ISMCTS maintainability refactor: capture
once at the pre-refactor commit, then run --check after every step. RNG draw
order and torch op order are part of the pinned behavior, so any accidental
reorder fails the check even when the result is statistically equivalent.

Usage:
    uv run python -m sheepshead.analysis.capture_search_goldens            # write
    uv run python -m sheepshead.analysis.capture_search_goldens --check    # compare

Same-machine/same-torch only (like capture_arch_goldens); the manifest stamps
the environment and --check refuses to compare across a mismatch.
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

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, Game
from sheepshead.agent import ppo
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher, _is_private_action
from sheepshead.training.training_utils import set_all_seeds

SEED = 20260810
ARCH = "perceiver-shared-v2"
POOL_WORLDS = 16
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "fixtures",
    "search_golden",
)

# One search per (node, arm). Small iteration budgets keep a full run ~1-2 min
# while still exercising every code path the refactor touches.
_BASE = dict(
    iters={"pick": 8, "partner": 8, "bury": 8, "play": 8},
    det_max_tries=400,
    d_rollout=2,
)
ARMS = {
    # name: (config kwargs, search kwargs, all heads?)
    "puct_oracle_b32": (dict(_BASE, batch_size=32), {}, True),
    "puct_oracle_b1": (dict(_BASE, batch_size=1), {}, True),
    "puct_limited_b32": (
        dict(_BASE, batch_size=32, leaf_evaluator="limited"),
        {},
        True,
    ),
    "rm_oracle_b32": (dict(_BASE, batch_size=32, root_selection="rm"), {}, True),
    # Play-node-only arms: population grounding and the terminal-rollout path.
    "puct_oracle_seatpol": (dict(_BASE, batch_size=32), {"seat_policies": True}, False),
    "puct_terminal_d99": (dict(_BASE, batch_size=32), {"d_rollout": 99}, False),
}
HEADS = ("pick", "partner", "bury", "play", "leaster")


def _head(valid):
    names = [ACTIONS[a - 1] for a in valid]
    if any(n in ("PICK", "PASS") for n in names):
        return "pick"
    if any(n == "ALONE" or n == "JD PARTNER" or n.startswith("CALL ") for n in names):
        return "partner"
    if any(n.startswith("BURY ") or n.startswith("UNDER ") for n in names):
        return "bury"
    return "play"


def _drive_to_head(game, rng, want_head):
    """Random-legal play until the first decision of ``want_head``; returns
    (observer, forced_public) or None. Public actions only in forced_public
    (the teacher's replay contract). For "leaster" every seat passes and the
    node is seat 1's first leaster play."""
    pass_id = ACTIONS.index("PASS") + 1
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                if want_head == "leaster":
                    if game.is_leaster and player.position == 1:
                        return player.position, forced_public
                elif not game.is_leaster and _head(valid) == want_head:
                    return player.position, forced_public
                if want_head == "leaster" and pass_id in valid:
                    action_id = pass_id
                else:
                    action_id = rng.choice(sorted(valid))
                if not _is_private_action(action_id):
                    forced_public.append((player.position, action_id))
                player.act(action_id)
                valid = player.get_valid_action_ids()
    return None


def collect_panel():
    """One deterministic decision node per head. Game seeds are scanned in a
    fixed order so the panel is stable regardless of how many seeds fail to
    produce a node for a given head."""
    panel = {}
    for head in HEADS:
        for game_seed in range(200):
            game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=game_seed)
            out = _drive_to_head(game, random.Random(SEED + game_seed), head)
            if out is not None:
                observer, forced_public = out
                panel[head] = (game, observer, forced_public, game_seed)
                break
        else:
            raise RuntimeError(f"no {head} node found in 200 seeds")
    return panel


def build_agents():
    """The searched agent (oracle critic) plus a distinct second agent used as
    a non-observer seat policy in the grounding arm."""
    set_all_seeds(SEED)
    agent = PPOAgent(len(ACTIONS), critic_mode="oracle", arch=ARCH)
    set_all_seeds(SEED + 1)
    other = PPOAgent(len(ACTIONS), critic_mode="oracle", arch=ARCH)
    return agent, other


def _sha(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _result_record(res) -> dict:
    """Split a search result into bit-comparable tensors + an exact scalar repr."""
    arrays = {}
    for key in ("pi", "pi_gumbel", "pi_rm"):
        value = res[key]
        arrays[key] = None if value is None else torch.from_numpy(value.copy())
    scalars = repr(
        {
            "ess": res["ess"],
            "ok": res["ok"],
            "head": res["head"],
            "n_iter": res["n_iter"],
            "valid": res["valid"],
            "root_n": res["root_n"],
            "root_q": res["root_q"],
            "root_prior": res["root_prior"],
        }
    )
    return {"arrays": arrays, "scalars": scalars}


def capture() -> dict:
    agent, other = build_agents()
    panel = collect_panel()
    out: dict = {"searches": {}, "pool": {}}
    for head, (game, observer, forced_public, game_seed) in panel.items():
        for arm_name, (cfg_kwargs, search_kwargs, all_heads) in ARMS.items():
            if not all_heads and head != "play":
                continue
            teacher = ISMCTSTeacher(agent, ISMCTSConfig(**cfg_kwargs))
            kwargs = {}
            if search_kwargs.get("seat_policies"):
                kwargs["seat_policies"] = {
                    seat: other for seat in range(1, 6) if seat != observer
                }
            if "d_rollout" in search_kwargs:
                kwargs["d_rollout"] = search_kwargs["d_rollout"]
            torch.manual_seed(SEED + game_seed)
            res = teacher.search(
                game,
                observer,
                list(forced_public),
                random.Random(SEED + game_seed),
                **kwargs,
            )
            out["searches"][f"{head}/{arm_name}"] = _result_record(res)

    # Pool-level fixture: the lockstep world build's log-weights and world
    # histories on the play node, independent of tree stochastics. Guards the
    # replay refactor steps directly.
    game, observer, forced_public, game_seed = panel["play"]
    teacher = ISMCTSTeacher(agent, ISMCTSConfig(**dict(_BASE, batch_size=32)))
    teacher._rng = random.Random(SEED + game_seed)
    torch.manual_seed(SEED + game_seed)
    pool = teacher._build_pool(game, observer, list(forced_public), POOL_WORLDS)
    out["pool"] = {
        "n_worlds": len(pool),
        "log_weights": torch.tensor(
            [log_weight for _, _, log_weight in pool], dtype=torch.float64
        ),
        "history_sha": [
            _sha(np.frombuffer(repr(world.history).encode(), dtype=np.uint8))
            for world, _, _ in pool
        ],
        "fail": repr(dict(sorted(teacher.fail.items()))),
    }
    return out


def _fixture_path(fixture_dir: str) -> str:
    return os.path.join(fixture_dir, "search_golden.pt")


def manifest_path(fixture_dir: str = FIXTURE_DIR) -> str:
    return os.path.join(fixture_dir, "manifest.json")


def load_manifest(fixture_dir: str = FIXTURE_DIR) -> dict:
    with open(manifest_path(fixture_dir)) as f:
        return json.load(f)


def runtime_matches_manifest(manifest: dict) -> bool:
    """Bit-identity goldens are only meaningful on the environment that
    captured them (BLAS kernels differ across platforms/torch builds)."""
    return (
        manifest["torch"] == torch.__version__
        and manifest["platform"] == platform.platform()
    )


def _compare_record(name, golden, current, problems):
    for key in ("pi", "pi_gumbel", "pi_rm"):
        g, c = golden["arrays"][key], current["arrays"][key]
        if (g is None) != (c is None):
            problems.append(
                f"{name}.{key}: None-ness drifted ({g is None} -> {c is None})"
            )
        elif g is not None and not torch.equal(g, c):
            diff = (g.double() - c.double()).abs().max().item()
            problems.append(f"{name}.{key}: values drifted (max abs diff {diff:g})")
    if golden["scalars"] != current["scalars"]:
        problems.append(f"{name}: scalar record drifted")
        problems.append(f"  golden : {golden['scalars']}")
        problems.append(f"  current: {current['scalars']}")


def check(fixture_dir: str = FIXTURE_DIR) -> list:
    """Recompute everything and return a list of mismatch descriptions
    (empty = bit-identical)."""
    golden = torch.load(_fixture_path(fixture_dir), weights_only=True)
    current = capture()
    problems: list = []
    g_searches, c_searches = golden["searches"], current["searches"]
    for name in sorted(set(g_searches) | set(c_searches)):
        if name not in g_searches:
            problems.append(f"search {name}: new (not in golden)")
        elif name not in c_searches:
            problems.append(f"search {name}: gone (in golden only)")
        else:
            _compare_record(name, g_searches[name], c_searches[name], problems)
    g_pool, c_pool = golden["pool"], current["pool"]
    if g_pool["n_worlds"] != c_pool["n_worlds"]:
        problems.append(f"pool.n_worlds: {g_pool['n_worlds']} -> {c_pool['n_worlds']}")
    elif not torch.equal(g_pool["log_weights"], c_pool["log_weights"]):
        diff = (g_pool["log_weights"] - c_pool["log_weights"]).abs().max().item()
        problems.append(f"pool.log_weights: drifted (max abs diff {diff:g})")
    if g_pool["history_sha"] != c_pool["history_sha"]:
        problems.append("pool.history_sha: world histories drifted")
    if g_pool["fail"] != c_pool["fail"]:
        problems.append(f"pool.fail: {g_pool['fail']} -> {c_pool['fail']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the existing fixture instead of writing it",
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

    if args.check:
        manifest = load_manifest(args.fixture_dir)
        if not runtime_matches_manifest(manifest):
            print(
                "Runtime does not match fixture manifest "
                f"(torch {manifest['torch']} vs {torch.__version__}, "
                f"platform {manifest['platform']} vs {platform.platform()}); "
                "bit-identity comparison would be meaningless here."
            )
            return 2
        problems = check(args.fixture_dir)
        if problems:
            print("FAIL search goldens")
            for p in problems:
                print(f"     {p}")
            return 1
        print("ok   search goldens bit-identical")
        return 0

    os.makedirs(args.fixture_dir, exist_ok=True)
    torch.save(capture(), _fixture_path(args.fixture_dir))
    with open(manifest_path(args.fixture_dir), "w") as f:
        json.dump(
            {
                "seed": SEED,
                "arch": ARCH,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"wrote search goldens + manifest to {args.fixture_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
