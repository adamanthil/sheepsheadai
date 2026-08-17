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
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.training.league import SELF_PLAY
from sheepshead.training.league_teacher import build_frozen_expert
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


def league_worker_init(init_args: dict) -> None:
    import torch as _torch

    _torch.set_num_threads(1)
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
        # Stationary expert: the teacher wraps a frozen copy of the
        # generation-start policy, NOT the live worker agent — weight
        # refreshes in league_worker_play never touch it.
        frozen_expert = build_frozen_expert(
            init_args["teacher_resume"],
            init_args.get("critic_mode", "limited"),
            init_args.get("arch", "full"),
            bool(init_args.get("oracle_aux_heads", False)),
            init_args.get("teacher_oracle_init"),
            float(init_args.get("teacher_gamma", 1.0)),
        )
        WORKER_STATE["teacher"] = ISMCTSTeacher(
            frozen_expert,
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
