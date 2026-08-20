#!/usr/bin/env python3
"""Committee-search throughput at production worker concurrency.

Times the CE teacher's committee search — the thing that dominates a teaching
generation — under the device and compilation options the trainer can actually
be run with. It exists to answer "does this configuration make the teacher
faster", which is not a question a single-process timing can answer:

* ``--clients 8`` runs eight concurrent client *processes*, matching the
  trainer's worker count. Workers contend for cores (and, on ``--device mps``,
  serialize on one GPU), so a one-process number systematically overstates any
  configuration that shares a resource.
* ``--repeats 3`` reports steady state. The first committee pays compilation,
  and Metal caches shaders across calls, so a cold MPS run reads ~50% slow. A
  training run does thousands of committees; the later repeats are the honest
  number.

Measured with both (M1 Max, 8 clients, steady state): CPU eager 55.5s,
MPS + ``--compile`` 40.7s, a 1.36x speedup. See
notebooks/Distributed_Inference_202608.md §5.5-§5.6 for the attribution and for
the shelved two-machine alternative.

Usage:

    uv run python -m sheepshead.analysis.bench_search_committee \\
        --checkpoint runs/league_ce_teacher11/_league_worker_weights_v24.pt \\
        --clients 8 --repeats 3

    uv run python -m sheepshead.analysis.bench_search_committee \\
        --checkpoint <weights> --clients 8 --repeats 3 --device mps --compile
"""

import argparse
import random
import sys
import time

import torch

from sheepshead import ACTION_IDS, ACTION_LOOKUP, PARTNER_BY_CALLED_ACE, Game

# Production CE-teacher search settings (training/config.py SearchConfig).
TEACHER_ITERS = 1024
TEACHER_REPLICATES = 3
TEACHER_D_ROLLOUT = 1


def find_play_node(agent, max_seeds: int = 200):
    """Replay deterministically to a mid-game PLAY decision with >= 2 legal
    actions -- the node class the CE teacher actually labels."""
    from sheepshead.ismcts import is_private_action

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
                    if not is_private_action(action):
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
    not how production workers arrive and would understate contention."""
    rngs = [random.Random(1000 + seed_offset + i) for i in range(replicates)]
    start = time.perf_counter()
    teacher.search_committee(game, seat, list(forced), rngs, d_rollout=d_rollout)
    return time.perf_counter() - start


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
    """One trainer worker's worth of load.

    A separate process, not a thread, and that is the point: production workers
    are processes, and N Python searches sharing one interpreter would contend
    on the GIL badly enough to become the bottleneck themselves.

    Running even a single client this way is also what makes the device
    setting reliable: ``ismcts`` binds ``DEV = ppo.device`` at import, and the
    replay/pool-build path (two thirds of the encoder work) uses it, so the
    device has to be set in a fresh interpreter before ``setup`` pulls ismcts in.
    """
    torch.set_num_threads(opts["threads"])
    from sheepshead.agent import ppo as ppo_module

    device = torch.device(opts["device"])
    ppo_module.device = device
    if opts["compile"]:
        from sheepshead.agent.compiled_encoder import enable_compiled_encoder

        enable_compiled_encoder(opts["granularity"], opts["compile"])

    agent, teacher, game, seat, forced = setup(opts, announce=(index == 0))
    for net in (agent.encoder, agent.actor, agent.critic):
        net.to(device)

    offset = 977 * index
    repeats = max(1, int(opts["repeats"]))
    walls = []
    for repeat in range(repeats):
        if repeat == repeats - 1:
            # Report the measured committee, not the warm-ups folded in.
            teacher.network_rounds = 0
            teacher.network_states = 0
        barrier.wait()  # every repeat has to be concurrent, not just the first
        walls.append(
            run_committee(
                teacher,
                game,
                seat,
                forced,
                opts["replicates"],
                opts["d_rollout"],
                offset,
            )
        )

    results.put(
        {
            "index": index,
            "walls": walls,
            "wall": min(walls[1:]) if repeats > 1 else walls[0],
            "rounds": teacher.network_rounds,
            "states": teacher.network_states,
        }
    )


def run_fleet(opts: dict, clients: int) -> int:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(clients)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_run_client, args=(i, opts, barrier, results))
        for i in range(clients)
    ]
    print(f"\n=== {clients} concurrent client{'s' if clients > 1 else ''} ===")
    for proc in procs:
        proc.start()
    collected = [results.get() for _ in procs]
    for proc in procs:
        proc.join()

    collected.sort(key=lambda row: row["index"])
    walls = [row["wall"] for row in collected]
    states = sum(row["states"] for row in collected)

    if max(1, int(opts["repeats"])) > 1:
        print(
            "\nper-repeat walls (client 0). The first pays compilation and device "
            "warm-up; the later ones are steady state and are what is reported below."
        )
        print("  " + "  ".join(f"{wall:6.2f}s" for wall in collected[0]["walls"]))

    label = opts["device"] + (
        f" compiled/{opts['compile']}" if opts["compile"] else " eager"
    )
    print(
        f"\n[{label}]  wall max {max(walls):6.2f}s  mean "
        f"{sum(walls) / clients:6.2f}s   {states / clients:.0f} states/committee/client"
        f"   {states / max(walls):8.0f} states/s"
    )
    first = collected[0]
    print(
        f"  client 0: {first['rounds']} network rounds, "
        f"{first['states'] / first['rounds']:.0f} states per round"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--arch", default="perceiver-shared-v2")
    parser.add_argument("--iters", type=int, default=TEACHER_ITERS)
    parser.add_argument("--replicates", type=int, default=TEACHER_REPLICATES)
    parser.add_argument("--d-rollout", type=int, default=TEACHER_D_ROLLOUT)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch threads per client. 1 matches a league worker; anything "
        "higher oversubscribes the cores once --clients is realistic",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=8,
        help="concurrent client processes. The default matches the trainer's "
        "worker count; 1 measures a configuration with the machine to itself, "
        "which no worker ever has",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="committees per client; the reported wall is the best of repeats "
        "2..N. Use >= 2 whenever --compile or --device mps is set: the first "
        "committee pays compilation, and Metal caches shaders across calls",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for every encode, the network rounds and the "
        "replay/pool-build path alike. 'mps' + --compile is the fastest known "
        "configuration on an M1 Max (1.36x)",
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const="default",
        default=None,
        metavar="MODE",
        help="compile the encoder (all four call sites). Opt-in: output differs "
        "from eager by ~2.6e-08, so goldens cannot pass against it. MODE is "
        "passed to torch.compile; 'reduce-overhead' selects CUDA graphs on CUDA "
        "and is a no-op on MPS",
    )
    parser.add_argument("--granularity", type=int, default=32)
    args = parser.parse_args()

    return run_fleet(vars(args), max(1, args.clients))


if __name__ == "__main__":
    sys.exit(main())
