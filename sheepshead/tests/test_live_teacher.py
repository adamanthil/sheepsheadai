"""The closed-loop teacher expert (CE_Teacher_Design §15a).

What has to be pinned is identity: the expert IS the acting agent — the
worker's current-weights copy, the sequential stream's training agent —
because weight refreshes mutate those networks in place and the teacher must
follow without any refresh plumbing of its own. A copy that merely started
equal would silently recreate the frozen expert (attempts 8-11's failure
mechanism, §14/§15) with extra steps.
"""

import argparse
from types import SimpleNamespace

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import league_worker
from sheepshead.training.league_teacher import TeacherSettings, build_teacher_kwargs

ARCH = "perceiver-shared-v2"


def test_worker_expert_is_the_rollout_agent():
    """Identity, not equality: league_worker_play's load_network_states
    mutates the worker agent's networks in place, so only the same object
    guarantees the expert follows each weight version."""
    league_worker.league_worker_init(
        {
            "arch": ARCH,
            "members_dir": ".",
            "weight_path_base": "unused",
            "base_seed": 0,
            "teacher": True,
            "teacher_gamma": 1.0,
        }
    )
    teacher = league_worker.WORKER_STATE["teacher"]
    agent = league_worker.WORKER_STATE["agent"]
    assert teacher.agent is agent
    assert teacher.agent.encoder is agent.encoder


def test_build_teacher_kwargs_wraps_the_training_agent():
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    context = SimpleNamespace(
        args=argparse.Namespace(teacher=True, seed=0),
        training_agent=agent,
    )
    kwargs = build_teacher_kwargs(context)
    assert kwargs["teacher"].agent is agent


def test_settings_tolerate_the_exploiter_namespace():
    # The exploiter phase passes a SimpleNamespace without teacher_* attrs.
    assert TeacherSettings.from_args(SimpleNamespace()).enabled is False
