"""CE search-teacher construction for train_league_ppo.py.

The expert is CLOSED-LOOP (CE_Teacher_Design §15a): the committee runs on
the training network itself — the sequential stream wraps the training
agent, spawned workers wrap their own current-weights copy, which weight
refreshes mutate in place, so the expert lags the student by at most one
version. Targets are therefore one-step improvements of the CURRENT policy,
bounded-KL from it by construction. The frozen generation-start expert this
replaced (attempts 8-11) is gone, not optional: an open-loop expert
integrates the seed's one-step improvement past its validity radius, and
the label KL carries a floor set by seed-student distance — the §14/§15
failure mechanism. The cost is that the expert is never a certified
checkpoint; the boundary cert's absolute anchors are the certification.

TeacherSettings centralizes the getattr-with-SearchConfig-default pattern
shared by build_teacher_kwargs (run_main_phase's sequential/parallel
streams) and the worker-pool initargs dict. The exploiter phase passes a
SimpleNamespace args missing the teacher_* attributes entirely, so the
getattr semantics must tolerate that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

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

    @classmethod
    def from_args(cls, args) -> "TeacherSettings":
        return cls(
            enabled=bool(getattr(args, "teacher", False)),
            prob=float(getattr(args, "teacher_prob", SearchConfig().teacher_prob)),
            replicates=int(
                getattr(args, "teacher_replicates", SearchConfig().teacher_replicates)
            ),
            iters=int(getattr(args, "teacher_iters", SearchConfig().teacher_iters)),
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


def build_teacher_kwargs(context) -> dict:
    """play_population_game kwargs for the CE search teacher
    (CE_Teacher_Design §2), or {} when --teacher is off.

    Built once per stream: an ISMCTS teacher over the training agent itself
    (closed-loop expert, CE_Teacher_Design §15a; the engine
    snapshots/restores per-seat memories around each search, so sharing the
    acting agent is side-effect free), at the calibrated budget
    (--teacher-iters, d_rollout=1 per call, oracle leaves via the engine
    default) plus the emission SearchConfig."""
    settings = TeacherSettings.from_args(context.args)
    if not settings.enabled:
        return {}
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    teacher = ISMCTSTeacher(
        context.training_agent,
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
