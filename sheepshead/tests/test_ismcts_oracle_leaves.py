"""Oracle leaf evaluation in the ISMCTS teacher (default-on, 2026-08-10).

Truncated rollout leaves are evaluated by the privileged OracleValueNetwork
on the observer's full-information event stream (legitimate inside a
determinized world). These tests cover the arming logic, the fallback for
agents without an oracle critic, terminal-depth gating, and determinism.
"""

import random

import torch

from sheepshead import ACTION_LOOKUP, ACTIONS, PARTNER_BY_CALLED_ACE, Game
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher, _is_private_action
from sheepshead.tests.ppo_test_helpers import seed_all

SEED = 20260810
ARCH = "perceiver-shared-v2"


def _agent(critic_mode="oracle"):
    seed_all(SEED)
    return PPOAgent(len(ACTIONS), critic_mode=critic_mode, arch=ARCH)


def _to_first_play_node(agent, game_seed=11):
    """Greedy-drive a game to the first PLAY decision, returning
    (game, observer, forced_public) positioned at that decision."""
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=game_seed)
    agent.reset_recurrent_state()
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                valid_sorted = sorted(valid)
                if ACTION_LOOKUP[valid_sorted[0]].startswith("PLAY "):
                    return game, player.position, forced_public
                probs, _ = agent.get_action_probs_with_logits(
                    player.get_state_dict(), valid, player_id=player.position
                )
                aid = int(torch.argmax(probs.squeeze(0)).item()) + 1
                if aid not in valid:
                    aid = valid_sorted[0]
                if not _is_private_action(aid):
                    forced_public.append((player.position, aid))
                player.act(aid)
                if game.was_trick_just_completed and not game.is_done():
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )
                valid = player.get_valid_action_ids()
    raise AssertionError("no play node reached")


def _config(**kw):
    base = dict(
        iters={"pick": 16, "partner": 16, "bury": 16, "play": 16},
        batch_size=4,
        d_rollout=2,
    )
    base.update(kw)
    return ISMCTSConfig(**base)


def _search(agent, config, d_rollout=None, game_seed=11):
    game, obs, forced = _to_first_play_node(agent, game_seed)
    teacher = ISMCTSTeacher(agent, config)
    res = teacher.search(game, obs, forced, random.Random(7), d_rollout=d_rollout)
    return teacher, res


def test_oracle_leaves_default_on_for_oracle_agent():
    agent = _agent("oracle")
    teacher, res = _search(agent, _config())
    assert teacher._oracle_capture is True
    assert res["pi"].sum() > 0.99
    assert res["pi_gumbel"] is not None
    assert all(v == v for v in res["root_q"].values())  # finite


def test_limited_agent_falls_back_silently():
    agent = _agent("limited")
    teacher, res = _search(agent, _config())
    assert teacher._oracle_capture is False
    assert res["pi"].sum() > 0.99


def test_explicit_limited_override():
    agent = _agent("oracle")
    teacher, res = _search(agent, _config(leaf_evaluator="limited"))
    assert teacher._oracle_capture is False
    assert res["pi"].sum() > 0.99


def test_terminal_depth_skips_capture():
    agent = _agent("oracle")
    teacher, res = _search(agent, _config(), d_rollout=99)
    assert teacher._oracle_capture is False  # no critic leaves possible
    assert res["pi"].sum() > 0.99


def test_oracle_leaf_search_is_deterministic():
    agent = _agent("oracle")
    _, res1 = _search(agent, _config())
    _, res2 = _search(agent, _config())
    assert res1["root_q"] == res2["root_q"]
    assert res1["root_n"] == res2["root_n"]


def test_oracle_and_limited_leaves_differ():
    # Same tree seed, different leaf evaluator: values should diverge for a
    # freshly initialized (uncalibrated) pair of critics. Guards against the
    # oracle path silently routing back to the limited critic.
    agent = _agent("oracle")
    _, res_o = _search(agent, _config())
    _, res_l = _search(agent, _config(leaf_evaluator="limited"))
    assert res_o["root_q"] != res_l["root_q"]
