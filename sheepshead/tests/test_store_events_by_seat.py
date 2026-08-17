"""Tests for the per-seat event storage fix (league path).

The bug: play_population_game returns all collecting seats' events in one
temporally-interleaved list; storing it whole made ONE braided
multi-perspective segment per self-seat episode, so the recurrent update
forward ran a single memory across perspective switches (train/act
mismatch; non-unit PPO ratios at theta_old). store_events_by_seat restores
the selfplay trainer's per-player stream semantics."""

import numpy as np
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.league import SELF_PLAY
from sheepshead.training.league_worker import OpponentAdapter
from sheepshead.training.pfsp_runtime import play_population_game
from sheepshead.training.reward_shaping import RETURN_SCALE
from sheepshead.training.train_league_ppo import store_events_by_seat

SEED = 20260726
ARCH = "perceiver-shared-v2"


def _agent():
    seed_all(SEED)
    return PPOAgent(len(ACTIONS), critic_mode="limited", arch=ARCH)


def _play_self_table(agent, seed_offset=0, mode=PARTNER_BY_JD):
    opponents = [OpponentAdapter(agent, SELF_PLAY) for _ in range(4)]
    seed_all(SEED + 100 + seed_offset)
    _, events, scores, _, _ = play_population_game(
        training_agent=agent,
        opponents=opponents,
        partner_mode=mode,
        training_agent_position=1,
        reward_mode="terminal",
    )
    return events, scores


def test_self_table_episode_stores_one_segment_per_seat():
    agent = _agent()
    events, _ = _play_self_table(agent)
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
    events, _ = _play_self_table(agent, seed_offset=1)
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
    events, _ = _play_self_table(agent, seed_offset=2, mode=PARTNER_BY_CALLED_ACE)
    store_events_by_seat(agent, events)

    from sheepshead.tests.ppo_test_helpers import prepare_minibatch_inputs

    states, masks_t, kinds, segments = prepare_minibatch_inputs(agent)
    minibatch = agent._build_minibatch_tensors(segments, states, masks_t, kinds)
    with torch.no_grad():
        forward = agent._forward_vectorized(minibatch.states_seqs, minibatch.masks_bt)
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


def test_terminal_reward_lands_on_every_seats_last_action():
    """The sibling bug: _finalize_rewards fed the merged multi-seat action
    list to process_terminal_rewards, whose contract is one player's
    chronological transitions — so the single terminal reward landed on the
    globally-last actor and every other collecting seat's stream was
    all-zero. Per-player grouping restores each seat's own final_score on
    its own last action."""
    agent = _agent()
    events, scores = _play_self_table(agent, seed_offset=3)
    assert sum(scores) == 0  # zero-sum sanity

    by_pid: dict[int, list] = {}
    for ev in events:
        if ev["kind"] == "action":
            by_pid.setdefault(ev["player_id"], []).append(ev)
    assert len(by_pid) == 5
    for pid, acts in by_pid.items():
        expected = scores[pid - 1] / RETURN_SCALE
        assert acts[-1]["reward"] == expected, (
            f"seat {pid}: last-action reward {acts[-1]['reward']} != "
            f"own return {expected}"
        )
        assert all(a["reward"] == 0.0 for a in acts[:-1])


def test_per_seat_returns_flow_through_storage():
    """Integration through the trainer path: after store_events_by_seat,
    the gamma=1 empirical return of every stored row equals that row's own
    seat's final score — the target the oracle/limited critics regress to
    (this was zero for non-last-actor seats before the reward fix, and the
    LAST actor's return for every braided row before the storage fix)."""
    agent = _agent()
    agent.gamma = 1.0
    events, scores = _play_self_table(agent, seed_offset=4)
    store_events_by_seat(agent, events)

    acts = [e for e in agent.events if e["kind"] == "action"]
    rew = np.array([e["reward"] for e in acts])
    dns = np.array([e["done"] for e in acts] + [False])
    zeros = np.zeros(len(acts) + 1)
    _, g_emp = agent._gae_1d(rew, zeros, dns, agent.gamma, 1.0)

    for g, e in zip(g_emp, acts):
        expected = scores[e["player_id"] - 1] / RETURN_SCALE
        assert abs(float(g) - expected) < 1e-9, (
            f"row of seat {e['player_id']}: return {g} != own {expected}"
        )
    agent.reset_storage()
