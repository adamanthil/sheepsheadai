"""CE search-teacher construction for train_league_ppo.py.

Split out of train_league_ppo.py as pure code motion (Stage 1 of the
league-trainer maintainability refactor): the frozen-expert builder, the
per-stream teacher kwargs builder, and a TeacherSettings dataclass that
centralizes the today's getattr-with-SearchConfig-default pattern shared by
build_teacher_kwargs (run_main_phase's sequential/parallel streams) and the
worker-pool initargs dict (spawned workers reconstruct their own frozen
expert from these settings). The exploiter phase passes a SimpleNamespace
args missing the teacher_* attributes entirely, so the getattr semantics
here must match exactly what the two call sites did before extraction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.config import SearchConfig


@dataclass(frozen=True)
class TeacherSettings:
    """CE search-teacher configuration resolved from args, with the
    exploiter's teacher-less SimpleNamespace defaulting to teacher-off."""

    enabled: bool
    prob: float
    replicates: int
    iters: int
    ckpt: str | None
    oracle_init: str | None

    @classmethod
    def from_args(cls, args) -> "TeacherSettings":
        return cls(
            enabled=bool(getattr(args, "teacher", False)),
            prob=float(getattr(args, "teacher_prob", SearchConfig().teacher_prob)),
            replicates=int(
                getattr(args, "teacher_replicates", SearchConfig().teacher_replicates)
            ),
            iters=int(getattr(args, "teacher_iters", SearchConfig().teacher_iters)),
            # Stationary expert: --teacher-ckpt pins the expert independently
            # of --resume so a mid-run continuation doesn't silently refreeze
            # to student weights; falls back to --resume when unset.
            ckpt=getattr(args, "teacher_ckpt", None) or getattr(args, "resume", None),
            oracle_init=getattr(args, "oracle_init", None),
        )


def warn_if_oracle_overwrite(agent: PPOAgent, oracle_init: str, resume: str) -> None:
    """Loud banner before --oracle-init clobbers a trained oracle.

    The flag exists for resuming PRE-oracle checkpoints (no
    oracle_state_dict — the warm start is the only trained init
    available). When the resume checkpoint already restored oracle
    weights, applying the init afterwards DOWNGRADES a trained critic to
    the pretrain — this silently shipped in the attempt-9/10 launch
    recipes (CE_Teacher_Design §10.2)."""
    if getattr(agent, "oracle_loaded_from_checkpoint", False):
        print(
            f"⚠️  --oracle-init {oracle_init} is OVERWRITING the trained oracle "
            f"critic restored from {resume}. This flag is meant for pre-oracle "
            "checkpoints; drop it unless the downgrade is intentional."
        )


def build_frozen_expert(
    resume: str,
    critic_mode: str,
    arch: str,
    oracle_aux_heads: bool,
    oracle_init: str | None,
    gamma: float,
) -> PPOAgent:
    """Stationary expert for the gated teacher (Search_Teacher_Design
    §12.1): a frozen copy of the generation-start policy, reconstructed
    exactly as the main agent was at resume (checkpoint + oracle
    warm-start + gamma). The teacher's searches (priors, rollout policies,
    critic leaves) all run on this snapshot, so the expert cannot chase a
    drifting student out of the E9-certified regime — DAgger's fixed
    expert (Ross et al. 2011), where attempt 7 showed the live-expert
    loop re-labels its own drift. The student's states still drive WHERE
    labels happen; only the expert's opinion is pinned."""
    frozen = PPOAgent(
        len(ACTIONS),
        critic_mode=critic_mode,
        arch=arch,
        oracle_aux_heads=oracle_aux_heads,
    )
    frozen.load(resume, load_optimizers=False)
    if oracle_init:
        warn_if_oracle_overwrite(frozen, oracle_init, resume)
        oracle_state_dict = torch.load(
            oracle_init, map_location="cpu", weights_only=True
        )
        frozen.oracle_critic.load_state_dict(oracle_state_dict, strict=True)
    frozen.gamma = gamma
    return frozen


def build_teacher_kwargs(context) -> dict:
    """play_population_game kwargs for the CE search teacher
    (CE_Teacher_Design §2), or {} when --teacher is off.

    Built once per stream: an ISMCTS teacher over a FROZEN copy of the
    generation-start policy (stationary expert) at the calibrated budget
    (--teacher-iters, d_rollout=1 per call, oracle leaves via the engine
    default) plus the emission SearchConfig."""
    settings = TeacherSettings.from_args(context.args)
    if not settings.enabled:
        return {}
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    frozen = build_frozen_expert(
        settings.ckpt,
        context.args.critic_mode,
        context.args.arch,
        getattr(context.args, "oracle_aux_heads", False),
        settings.oracle_init,
        context.training_agent.gamma,
    )
    teacher = ISMCTSTeacher(
        frozen,
        ISMCTSConfig(
            iters={head: settings.iters for head in ("pick", "partner", "bury", "play")}
        ),
    )
    search_config = SearchConfig(
        teacher_prob=settings.prob,
        teacher_replicates=settings.replicates,
    )
    return {
        "teacher": teacher,
        "determinization_rng": random.Random(context.args.seed ^ 0x5EA6C4),
        "search_config": search_config,
    }
