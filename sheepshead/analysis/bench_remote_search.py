#!/usr/bin/env python3
"""Run one committee search locally and again against a remote inference server.

Step 2 of the build order in notebooks/Throughput_Profiling_Notes.md addendum 2
(B5): the modelled numbers there rest on an estimated ~897 blocking round-trips
per episode and an assumed per-round latency. This measures both, on the real
search, and reports what the model has to assume against what it actually costs.

It also compares the two searches' outputs. That is not a pass/fail -- a remote
run can never be bit-exact (different device, and fp16 on the wire by default)
-- but the size of the divergence matters: the CE teacher's shrinkage is
calibrated against a measured per-replicate Q noise floor of ~0.026 Q SD
(shrink_s2_global = 6.95e-4), so a divergence approaching that would decalibrate
abstention. Run with --fp32 to separate wire quantisation from device numerics.

Start the server first (on the GPU box, or locally on CPU for a smoke test):

    uv run python -m sheepshead.inference.server \
        --checkpoint runs/league_ce_teacher11/_league_worker_weights_v24.pt \
        --device cuda

Then:

    uv run python -m sheepshead.analysis.bench_remote_search \
        --checkpoint runs/league_ce_teacher11/_league_worker_weights_v24.pt \
        --host 192.168.1.50

``--clients 8`` runs eight concurrent client *processes*, which is the only way
to measure the server's cross-worker batching -- one client has nothing to merge
with, and the unbatched per-round cost is what made the prototype a regression.
The server prints its own merge factor.
"""

import argparse
import random
import sys
import time

import numpy as np
import torch

from sheepshead import ACTION_IDS, ACTION_LOOKUP, PARTNER_BY_CALLED_ACE, Game
from sheepshead.inference import LocalBackend, RemoteBackend

# Production CE-teacher search settings (training/config.py SearchConfig).
TEACHER_ITERS = 1024
TEACHER_REPLICATES = 3
TEACHER_D_ROLLOUT = 1


def find_play_node(agent, max_seeds: int = 200):
    """Replay deterministically to a mid-game PLAY decision with >= 2 legal
    actions -- the node class the CE teacher actually labels."""
    from sheepshead.ismcts import _is_private_action

    for seed in range(max_seeds):
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
        agent.reset_recurrent_state()
        forced = []
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    ordered = sorted(valid)
                    is_play = ACTION_LOOKUP.get(ordered[0], "").startswith("PLAY ")
                    action, _, _ = agent.act(
                        player.get_state_dict(),
                        ordered,
                        player_id=player.position,
                        deterministic=True,
                    )
                    if (
                        is_play
                        and len(ordered) >= 2
                        and game.current_trick in (1, 2)
                        and not game.is_leaster
                        and not game.alone_called
                    ):
                        return game, player.position, list(forced), seed
                    if not _is_private_action(action):
                        forced.append((player.position, action))
                    player.act(action)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                    valid = player.get_valid_action_ids()
    raise RuntimeError("no eligible play node found")


def run_committee(teacher, game, seat, forced, replicates, d_rollout, seed_offset=0):
    """One committee search. ``seed_offset`` desynchronizes concurrent clients:
    identical RNG would make every client issue its rounds in lockstep, which is
    the most favourable possible arrival pattern for a batching server and would
    overstate the merge factor."""
    rngs = [random.Random(1000 + seed_offset + i) for i in range(replicates)]
    start = time.perf_counter()
    results = teacher.search_committee(
        game, seat, list(forced), rngs, d_rollout=d_rollout
    )
    return results, time.perf_counter() - start


def compare(local_results, remote_results) -> None:
    print("\n=== search output divergence (local vs remote) ===")
    print(
        "  the CE shrinkage noise floor is ~0.026 Q SD; a divergence near that\n"
        "  would decalibrate abstention (see addendum 2, B4)."
    )
    for i, (loc, rem) in enumerate(zip(local_results, remote_results)):
        row = f"  replicate {i}:"
        for key in ("pi_gumbel", "q"):
            a, b = loc.get(key), rem.get(key)
            if a is None or b is None:
                continue
            arr_a = np.asarray(a, dtype=np.float64)
            arr_b = np.asarray(b, dtype=np.float64)
            finite = np.isfinite(arr_a) & np.isfinite(arr_b)
            delta = np.abs(arr_a[finite] - arr_b[finite]).max() if finite.any() else 0.0
            row += f"  max|d {key}| = {delta:.3e}"
        valid = loc["valid"]
        gum_l = loc.get("pi_gumbel")
        gum_r = rem.get("pi_gumbel")
        if gum_l is not None and gum_r is not None:
            arg_l = max(valid, key=lambda a: float(gum_l[a - 1]))
            arg_r = max(valid, key=lambda a: float(gum_r[a - 1]))
            row += (
                f"  argmax {'AGREE' if arg_l == arg_r else f'DIFFER {arg_l}/{arg_r}'}"
            )
        print(row)


def setup(opts: dict, announce: bool = True):
    """Load the agent and replay to the node every client searches."""
    from sheepshead.agent.ppo import PPOAgent
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
    from sheepshead.training.training_utils import set_all_seeds

    set_all_seeds(42)
    agent = PPOAgent(action_size=len(ACTION_IDS), arch=opts["arch"])
    if opts["checkpoint"]:
        payload = torch.load(opts["checkpoint"], map_location="cpu", weights_only=False)
        agent.load_network_states(payload, source=opts["checkpoint"])
        if announce:
            print(f"loaded {opts['checkpoint']}")

    game, seat, forced, seed = find_play_node(agent)
    if announce:
        print(f"node: seed={seed} trick={game.current_trick} seat={seat}")

    config = ISMCTSConfig(
        iters={k: opts["iters"] for k in ("pick", "partner", "bury", "play")}
    )
    return agent, ISMCTSTeacher(agent, config), game, seat, forced


def _run_client(index: int, opts: dict, barrier, results) -> None:
    """One orchestrator worker's worth of load.

    A separate process, not a thread, and that is the point: production workers
    are processes, and eight Python searches sharing one interpreter would
    contend on the GIL badly enough to become the bottleneck themselves --
    arrivals would space out and the server's merge factor would look better
    than it will ever be in production.
    """
    torch.set_num_threads(1)
    agent, teacher, game, seat, forced = setup(opts, announce=(index == 0))
    offset = 977 * index
    payload = {"index": index}

    barrier.wait()  # setup costs differ; start the measured phase together
    local = LocalBackend(torch.device("cpu"))
    teacher.backend = local
    local_results, payload["local_wall"] = run_committee(
        teacher, game, seat, forced, opts["replicates"], opts["d_rollout"], offset
    )
    payload["local_rounds"] = local.rounds
    payload["local_states"] = local.states

    if opts["host"]:
        barrier.wait()  # ... and so is the remote phase genuinely concurrent
        remote = RemoteBackend(opts["host"], opts["port"], agent, half=not opts["fp32"])
        teacher.backend = remote
        remote_results, payload["remote_wall"] = run_committee(
            teacher, game, seat, forced, opts["replicates"], opts["d_rollout"], offset
        )
        payload["remote"] = remote.report()
        remote.close()
        if index == 0:
            compare(local_results, remote_results)

    results.put(payload)


def run_fleet(opts: dict, clients: int) -> int:
    """Measure ``clients`` concurrent searches, which is what actually exercises
    cross-worker batching. A single client can never merge with anything."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(clients)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_run_client, args=(i, opts, barrier, results))
        for i in range(clients)
    ]
    print(f"\n=== {clients} concurrent clients ===")
    for proc in procs:
        proc.start()
    collected = [results.get() for _ in procs]
    for proc in procs:
        proc.join()

    collected.sort(key=lambda row: row["index"])
    local_walls = [row["local_wall"] for row in collected]
    states = sum(row["local_states"] for row in collected)
    print(
        f"\n[local]  wall max {max(local_walls):6.2f}s  mean "
        f"{sum(local_walls) / clients:6.2f}s   {states} states total   "
        f"{states / max(local_walls):8.0f} states/s"
    )
    if not opts["host"]:
        return 0

    remote_walls = [row["remote_wall"] for row in collected]
    rounds = sum(row["remote"]["rounds"] for row in collected)
    blocked = sum(row["remote"]["total_s"] for row in collected)
    print(
        f"[remote] wall max {max(remote_walls):6.2f}s  mean "
        f"{sum(remote_walls) / clients:6.2f}s   {states} states total   "
        f"{states / max(remote_walls):8.0f} states/s"
    )
    print(
        f"  rounds {rounds}   blocked on I/O {blocked:.1f}s of "
        f"{sum(remote_walls):.1f}s client wall"
    )
    print(f"  local/remote    : {max(local_walls) / max(remote_walls):8.2f}x")
    print(
        "  the merge factor is the server's to report -- read its "
        "'merge N req/batch' line."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--arch", default="perceiver-shared-v2")
    parser.add_argument("--host", default=None, help="omit to run the local path only")
    parser.add_argument("--port", type=int, default=53018)
    parser.add_argument("--iters", type=int, default=TEACHER_ITERS)
    parser.add_argument("--replicates", type=int, default=TEACHER_REPLICATES)
    parser.add_argument("--d-rollout", type=int, default=TEACHER_D_ROLLOUT)
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="fp32 on the wire, to separate quantisation from device numerics",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="run N concurrent client processes instead of one, to exercise "
        "the server's cross-worker batching. 8 matches the trainer's worker "
        "count. A single client can never merge with anything, so the "
        "default measures latency and fidelity, not the batching win",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    opts = vars(args)
    if args.clients > 1:
        return run_fleet(opts, args.clients)

    agent, teacher, game, seat, forced = setup(opts)

    local = LocalBackend(torch.device("cpu"))
    teacher.backend = local
    local_results, local_wall = run_committee(
        teacher, game, seat, forced, args.replicates, args.d_rollout
    )
    print(
        f"\n[local]  {local_wall:7.2f}s   rounds {local.rounds}   "
        f"states {local.states}   mean batch {local.states / local.rounds:.0f}"
    )
    print(f"  the model assumes ~1150 rounds per committee; measured {local.rounds}.")

    if not args.host:
        print("\n(no --host given; skipping the remote leg)")
        return 0

    remote = RemoteBackend(args.host, args.port, agent, half=not args.fp32)
    teacher.backend = remote
    remote_results, remote_wall = run_committee(
        teacher, game, seat, forced, args.replicates, args.d_rollout
    )
    stats = remote.report()
    remote.close()

    print(f"\n[remote] {remote_wall:7.2f}s   rounds {stats['rounds']}   ")
    print(f"  mean batch      : {stats['mean_batch']:8.0f} states")
    print(
        f"  round latency   : p50 {stats['p50_ms']:.2f} ms   "
        f"p90 {stats['p90_ms']:.2f} ms   p99 {stats['p99_ms']:.2f} ms"
    )
    print(f"  blocked on I/O  : {stats['total_s']:8.2f}s of {remote_wall:.2f}s wall")
    print(
        f"  payload         : {stats['mb_up']:.1f} MB up, {stats['mb_down']:.1f} MB down"
    )
    print(f"  round-trip cost : {stats['us_per_state']:8.1f} us/state")
    print(f"  local/remote    : {local_wall / remote_wall:8.2f}x")

    compare(local_results, remote_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
