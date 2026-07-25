"""Tests for the per-seat event storage fix (league path).

The bug: play_population_game returns all collecting seats' events in one
temporally-interleaved list; storing it whole made ONE braided
multi-perspective segment per self-seat episode, so the recurrent update
forward ran a single memory across perspective switches (train/act
mismatch; non-unit PPO ratios at theta_old). store_events_by_seat restores
the selfplay trainer's per-player stream semantics."""

import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.league import SELF_PLAY
from sheepshead.training.pfsp_runtime import play_population_game
from sheepshead.training.train_league_ppo import _Seat, store_events_by_seat

SEED = 20260726
ARCH = "perceiver-shared-v2"


def _agent():
    seed_all(SEED)
    return PPOAgent(len(ACTIONS), critic_mode="limited", arch=ARCH)


def _play_self_table(agent, seed_offset=0, mode=PARTNER_BY_JD):
    opponents = [_Seat(agent, SELF_PLAY) for _ in range(4)]
    seed_all(SEED + 100 + seed_offset)
    _, events, _, _, _ = play_population_game(
        training_agent=agent,
        opponents=opponents,
        partner_mode=mode,
        training_agent_position=1,
        reward_mode="terminal",
    )
    return events


def test_self_table_episode_stores_one_segment_per_seat():
    agent = _agent()
    events = _play_self_table(agent)
    n_players = len({ev["player_id"] for ev in events})
    assert n_players == 5  # all seats collect on a self table

    store_events_by_seat(agent, events)
    kinds = [ev["kind"] for ev in agent.events]
    segments = agent._segments_from_events(kinds)
    assert len(segments) == n_players  # one coherent stream per seat

    # every segment is single-perspective
    for s, t_end in segments:
        pids = {agent.events[i]["player_id"] for i in range(s, t_end + 1)}
        assert len(pids) == 1
    agent.reset_storage()


def test_braided_storage_was_the_bug():
    # Control: the historical single-call path produces ONE braided segment
    # for the same episode.
    agent = _agent()
    events = _play_self_table(agent, seed_offset=1)
    agent.store_episode_events(events)
    kinds = [ev["kind"] for ev in agent.events]
    segments = agent._segments_from_events(kinds)
    assert len(segments) == 1
    s, t_end = segments[0]
    pids = {agent.events[i]["player_id"] for i in range(s, t_end + 1)}
    assert len(pids) == 5  # interleaved perspectives in one recurrent stream
    agent.reset_storage()


def test_hero_only_episode_is_unchanged():
    # With no SELF seats, grouping is a no-op: byte-for-byte identical
    # stored records in identical order.
    from types import SimpleNamespace

    agent = _agent()
    frozen = _agent()
    opponents = [SimpleNamespace(agent=frozen) for _ in range(4)]
    seed_all(SEED + 300)
    _, events, _, _, _ = play_population_game(
        training_agent=agent,
        opponents=opponents,
        partner_mode=PARTNER_BY_CALLED_ACE,
        training_agent_position=2,
        reward_mode="terminal",
    )
    assert len({ev["player_id"] for ev in events}) == 1

    n = store_events_by_seat(agent, events)
    grouped = list(agent.events)
    agent.reset_storage()
    agent.store_episode_events(events)
    legacy = list(agent.events)
    agent.reset_storage()

    assert n == sum(1 for ev in events if ev["kind"] == "action")
    assert len(grouped) == len(legacy)
    for a, b in zip(grouped, legacy):
        assert a.keys() == b.keys()
        assert a["kind"] == b["kind"]
        if a["kind"] == "action":
            assert a["action"] == b["action"]
            assert a["reward"] == b["reward"]
            assert a["done"] == b["done"]


def test_update_ratios_are_unit_at_theta_old_after_fix():
    """The coherence property the bug broke: re-running each stored segment
    with fresh memory must reproduce the act-time log-probs (PPO ratio = 1
    at theta_old) — true only when segments are single-perspective."""
    agent = _agent()
    events = _play_self_table(agent, seed_offset=2, mode=PARTNER_BY_CALLED_ACE)
    store_events_by_seat(agent, events)

    from sheepshead.tests.ppo_test_helpers import prepare_minibatch_inputs

    states, masks_t, kinds, segments = prepare_minibatch_inputs(agent)
    minibatch = agent._build_minibatch_tensors(segments, states, masks_t, kinds)
    with torch.no_grad():
        forward = agent._forward_vectorized(
            minibatch.states_seqs, minibatch.masks_bt
        )
    flat = agent._flatten_action_steps(minibatch, forward)
    probs = torch.softmax(flat.logits_flat, dim=-1)
    new_lp = torch.distributions.Categorical(probs.clamp(min=1e-12)).log_prob(
        flat.actions_flat
    )
    max_dev = float((new_lp - flat.old_log_probs_flat).abs().max())
    assert max_dev < 1e-4, (
        f"update-forward log-probs deviate from act-time by {max_dev}"
    )
    agent.reset_storage()
