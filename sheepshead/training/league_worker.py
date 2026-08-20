"""League worker pool: the league flavor of the pfsp_runtime worker
protocol (same versioned-weights scheme, opponents loaded from the league
members dir, SELF_PLAY seats played by the worker's own current-weights
copy) plus the weight-publishing side the main process drives.

Split out of train_league_ppo.py as pure code motion (Stage 1 of the
league-trainer maintainability refactor).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from sheepshead import ACTIONS
from sheepshead.agent.compiled_encoder import sync_routed_encoder
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.training.league import SELF_PLAY
from sheepshead.training.pfsp_runtime import make_game_summary, play_population_game


class OpponentAdapter:
    """Adapter giving a LeagueMember (or the training agent itself, for
    SELF_PLAY seats) the .agent / .metadata.agent_id surface that
    play_population_game expects of population opponents. The league keeps no
    strategic profiles, so this is the whole opponent surface it needs."""

    def __init__(self, agent: PPOAgent, agent_id: str):
        self.agent = agent
        self.metadata = SimpleNamespace(agent_id=agent_id)


@dataclass
class WorkerJob:
    episode: int
    partner_mode: int
    training_position: int
    opponent_ids: list  # member_id strings; SELF_PLAY for self seats
    weight_version: int
    collect_oracle: bool = False  # attach oracle_state to events (critic_mode=oracle)
    game_seed: int | None = None  # fixed deal for seat-rotated collection


# ----------------------------------------------------------------------------
# Worker pool (league flavor of the pfsp_runtime worker protocol: same
# versioned-weights scheme, opponents loaded from the league members dir,
# SELF_PLAY seats played by the worker's own current-weights copy).
# ----------------------------------------------------------------------------
WORKER_STATE: dict = {}


def _apply_inference_options(init_args: dict) -> None:
    """Apply the opt-in worker inference options, before anything builds a
    network or a teacher.

    Both are throughput-only and both change results in the last bits, so
    neither is ever on by default: compiled output differs from eager by
    ~2.6e-08 and ``capture_search_goldens`` cannot pass against it, and a
    non-CPU device brings its own numerics. Workers only generate episodes —
    the gates, greedy eval and boundary certs all run in the main process,
    which these never touch.

    Ordering is load-bearing. ``PPOAgent`` places its networks on the module
    global at construction, and the search teacher then follows the networks,
    so the device must be set here rather than anywhere later.

    Measured together (M1 Max, 8 workers, steady state): 1.36x on committee
    search. See notebooks/Distributed_Inference_202608.md §5.5-§5.6.
    """
    device = init_args.get("worker_device")
    if device:
        from sheepshead.agent import ppo as ppo_module

        ppo_module.device = torch.device(device)

    routed = init_args.get("worker_routed_encoder")
    compile_mode = init_args.get("worker_compile")
    if routed:
        # Routing keeps the process on CPU and ships only committee-scale
        # encode batches to a compiled shadow on the device — the measured
        # 1.50x (vs 1.09x for whole-agent MPS, §16.6). The shadow copies the
        # live weights lazily on first routed batch; league_worker_play
        # re-syncs it after every weight refresh.
        from sheepshead.agent.compiled_encoder import enable_routed_encoder

        enable_routed_encoder(
            int(init_args.get("worker_compile_granularity", 32)),
            compile_mode,
            device=routed,
        )
    elif compile_mode:
        from sheepshead.agent.compiled_encoder import enable_compiled_encoder

        enable_compiled_encoder(
            int(init_args.get("worker_compile_granularity", 32)), compile_mode
        )


def league_worker_init(init_args: dict) -> None:
    import torch as _torch

    _torch.set_num_threads(1)
    _apply_inference_options(init_args)
    agent = PPOAgent(
        len(ACTIONS),
        arch=init_args.get("arch", "full"),
        # Oracle-mode workers exist for the gated search teacher: the
        # worker's ISMCTS oracle leaves must evaluate with the SAME head the
        # main process trains (state arrives via the weight payload).
        critic_mode=init_args.get("critic_mode", "limited"),
        oracle_aux_heads=bool(init_args.get("oracle_aux_heads", False)),
    )
    seed = init_args["base_seed"] ^ (os.getpid() & 0xFFFFFFFF)
    random.seed(seed)
    WORKER_STATE.clear()
    WORKER_STATE.update(
        {
            "agent": agent,
            "members_dir": init_args["members_dir"],
            "weight_path_base": init_args["weight_path_base"],
            "version": 0,
            "cache": {},
        }
    )
    if init_args.get("teacher"):
        from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
        from sheepshead.training.config import SearchConfig

        search_config = SearchConfig(
            teacher_prob=float(
                init_args.get("teacher_prob", SearchConfig().teacher_prob)
            ),
            teacher_replicates=int(
                init_args.get("teacher_replicates", SearchConfig().teacher_replicates)
            ),
        )
        iters_per_head = int(
            init_args.get("teacher_iters", SearchConfig().teacher_iters)
        )
        # Closed-loop expert (CE_Teacher_Design §15a): the committee runs
        # on the worker's own current-weights copy. Weight refreshes in
        # league_worker_play mutate these same networks in place, so the
        # expert follows the student with at most one version of lag. The
        # engine snapshots/restores per-seat memories around each search
        # (keyed by id, the shared agent included), so sharing the rollout
        # agent is side-effect free.
        agent.gamma = float(init_args.get("teacher_gamma", 1.0))
        WORKER_STATE["teacher"] = ISMCTSTeacher(
            agent,
            ISMCTSConfig(
                iters={
                    head: iters_per_head for head in ("pick", "partner", "bury", "play")
                }
            ),
        )
        WORKER_STATE["search_config"] = search_config


def _get_cached_member(member_id: str) -> OpponentAdapter:
    cache = WORKER_STATE["cache"]
    seat = cache.get(member_id)
    if seat is None:
        # Arch-aware: members carry their architecture in checkpoint metadata
        # (legacy members without the key are the full architecture).
        agent = load_agent(os.path.join(WORKER_STATE["members_dir"], f"{member_id}.pt"))
        seat = OpponentAdapter(agent, member_id)
        cache[member_id] = seat
    return seat


def league_worker_play(job: WorkerJob) -> dict:
    import torch as _torch

    worker = WORKER_STATE
    if job.weight_version > worker["version"]:
        checkpoint = _torch.load(
            f"{worker['weight_path_base']}_v{job.weight_version}.pt", map_location="cpu"
        )
        worker["agent"].load_network_states(
            checkpoint, source=f"weights v{job.weight_version}"
        )
        worker["version"] = job.weight_version
        # No-op unless the routed encoder is enabled: the refresh above
        # mutated the live encoder in place, so its compiled shadow (which
        # the closed-loop teacher's committees run on) must follow or it
        # keeps labeling with stale weights.
        sync_routed_encoder(worker["agent"].encoder)

    opponents = [
        OpponentAdapter(worker["agent"], SELF_PLAY)
        if member_id_or_self == SELF_PLAY
        else _get_cached_member(member_id_or_self)
        for member_id_or_self in job.opponent_ids
    ]
    teacher_kwargs = {}
    if worker.get("teacher") is not None:
        teacher_kwargs = {
            "teacher": worker["teacher"],
            # Per-job stream: reproducible given the job, independent across
            # jobs (episode is unique; game_seed repeats across seat
            # rotations of one deal, so fold both in).
            "determinization_rng": random.Random(
                (job.episode << 20) ^ (job.game_seed or 0) ^ 0x5EA6C4
            ),
            "search_config": worker["search_config"],
        }
    game, episode_events, final_scores, training_data_single, pos_to_seat = (
        play_population_game(
            training_agent=worker["agent"],
            opponents=opponents,
            partner_mode=job.partner_mode,
            training_agent_position=job.training_position,
            reward_mode="terminal",
            collect_oracle=job.collect_oracle,
            game_seed=job.game_seed,
            **teacher_kwargs,
        )
    )
    return {
        "episode": job.episode,
        "partner_mode": job.partner_mode,
        "training_position": job.training_position,
        "episode_events": episode_events,
        "final_scores": final_scores,
        "training_data_single": training_data_single,
        "game_summary": make_game_summary(game),
        "seat_to_member_id": {
            pos: seat.metadata.agent_id for pos, seat in pos_to_seat.items()
        },
    }


def publish_weights(context) -> None:
    """Atomically write the training agent's current networks as the next
    versioned worker payload (encoder/actor/critic [+ oracle] + gamma) and
    delete the version-minus-2 file (workers only ever lag one version)."""
    context.weight_sync["version"] += 1
    path = f"{context.weight_sync['base']}_v{context.weight_sync['version']}.pt"
    payload = {
        "encoder_state_dict": context.training_agent.encoder.state_dict(),
        "actor_state_dict": context.training_agent.actor.state_dict(),
        "critic_state_dict": context.training_agent.critic.state_dict(),
        # Worker-side search teachers read gamma from the agent; the oracle
        # head keeps their leaf evaluation calibrated (load_network_states
        # consumes both when the worker agent is oracle-mode).
        "gamma": context.training_agent.gamma,
    }
    if context.training_agent.oracle_critic is not None:
        payload["oracle_state_dict"] = context.training_agent.oracle_critic.state_dict()
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)
    stale_path = (
        f"{context.weight_sync['base']}_v{context.weight_sync['version'] - 2}.pt"
    )
    if os.path.exists(stale_path):
        try:
            os.remove(stale_path)
        except OSError:
            pass
