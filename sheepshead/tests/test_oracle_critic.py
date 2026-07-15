#!/usr/bin/env python3
"""Privileged (oracle) critic invariants: observation schema, event
collection, update mechanics, gradient isolation, checkpoint compatibility.

The oracle critic sees all hidden state and is used only as the GAE baseline
during training (asymmetric actor-critic / CTDE); the actor and the limited
critic must be completely unaffected when it is disabled.
"""

import os
import pickle
import unittest

import numpy as np
import pytest

from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import (
    DECK_IDS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    Game,
)

# Collects real games and runs PPO update mechanics (~10s).
pytestmark = pytest.mark.slow


def _play_and_probe(seed, mode, probe):
    """Drive a full game with ScriptedAgent, calling ``probe(game, player)``
    at every decision point (before the action is applied)."""
    agent = ScriptedAgent()
    game = Game(partner_selection_mode=mode, seed=seed)
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                probe(game, player)
                a, _, _ = agent.act(player.get_state_dict(), valid, player.position)
                player.act(a)
                valid = player.get_valid_action_ids()
    return game


class TestOracleStateSchema(unittest.TestCase):
    def test_superset_of_limited_obs_and_hidden_fields_correct(self):
        checked = {"deals": 0, "secret_partner_seen": 0, "under_seen": 0}

        def probe(game, player):
            limited = player.get_state_dict()
            oracle = player.get_oracle_state_dict()

            # Superset: every limited key present; identical except the
            # de-masked blind/bury.
            for k, v in limited.items():
                self.assertIn(k, oracle)
                if k in ("blind_ids", "bury_ids"):
                    continue
                np.testing.assert_array_equal(np.asarray(oracle[k]), np.asarray(v))

            # True blind/bury for every seat; picker's limited view already
            # equals the truth.
            def ids(cards, n):
                out = [DECK_IDS[c] for c in cards] + [0] * n
                return np.array(out[:n], dtype=np.uint8)

            np.testing.assert_array_equal(oracle["blind_ids"], ids(game.blind, 2))
            np.testing.assert_array_equal(oracle["bury_ids"], ids(game.bury, 2))
            if player.is_picker:
                np.testing.assert_array_equal(oracle["blind_ids"], limited["blind_ids"])
                np.testing.assert_array_equal(oracle["bury_ids"], limited["bury_ids"])

            # Opponent hands at relative seats 2..5.
            self.assertEqual(oracle["opp_hand_ids"].shape, (4, 8))
            for i, r in enumerate(range(2, 6)):
                abs_seat = ((player.position + r - 2) % 5) + 1
                np.testing.assert_array_equal(
                    oracle["opp_hand_ids"][i],
                    ids(game.players[abs_seat - 1].hand, 8),
                )

            # Secret partner seat matches the is_secret_partner player.
            expected = 0
            for p in game.players:
                if p.is_secret_partner:
                    expected = ((p.position - player.position) % 5) + 1
                    break
            self.assertEqual(int(oracle["secret_partner_rel"]), expected)
            if expected:
                checked["secret_partner_seen"] += 1

            # Under card and points taken.
            if game.under_card:
                self.assertEqual(
                    int(oracle["under_card_id"]), DECK_IDS[game.under_card]
                )
                checked["under_seen"] += 1
            else:
                self.assertEqual(int(oracle["under_card_id"]), 0)
            for r in range(1, 6):
                abs_seat = ((player.position + r - 2) % 5) + 1
                self.assertEqual(
                    int(oracle["points_taken_rel"][r - 1]),
                    game.points_taken[abs_seat - 1],
                )

            # Must survive the worker pickle boundary.
            pickle.dumps(oracle)

        for seed in range(20):
            for mode in (PARTNER_BY_JD, PARTNER_BY_CALLED_ACE):
                _play_and_probe(1000 + seed, mode, probe)
                checked["deals"] += 1
        # The sweep must actually exercise the interesting states.
        self.assertGreater(checked["secret_partner_seen"], 0)

    def test_initial_state_partitions_deck(self):
        # Before any action, the 5 hands plus the blind are exactly ids 1..32.
        for seed in (7, 8, 9):
            game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
            player = game.players[0]
            oracle = player.get_oracle_state_dict()
            ids = list(oracle["hand_ids"]) + list(oracle["opp_hand_ids"].flatten())
            ids += list(oracle["blind_ids"])
            real = sorted(i for i in ids if i != 0)
            self.assertEqual(real, list(range(1, 33)))

    def test_last_trick_oracle_state_matches_convention(self):
        game = Game(partner_selection_mode=PARTNER_BY_JD, seed=11)
        player = game.players[0]
        o = player.get_last_trick_oracle_state_dict()
        lim = player.get_last_trick_state_dict()
        self.assertEqual(int(o["current_trick"]), int(lim["current_trick"]))
        self.assertIn("opp_hand_ids", o)


class TestOracleReadout(unittest.TestCase):
    """Perceiver-style readout invariants (2026-07-06 refactor): no pooled
    bags anywhere between the reasoning transformer and the value trunk."""

    @classmethod
    def setUpClass(cls):
        from sheepshead.agent.oracle import OracleValueNetwork

        cls.net = OracleValueNetwork()
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=3)
        cls.obs = [p.get_oracle_state_dict() for p in game.players]

    def test_pools_and_fusion_deleted(self):
        for name in (
            "pool_hand",
            "pool_trick",
            "pool_blind",
            "pool_bury",
            "pool_opp",
            "feature_proj",
        ):
            self.assertFalse(hasattr(self.net.encoder, name), name)

    def test_encode_batch_emits_tokens_and_memory_token_recurrence(self):
        import torch

        enc = self.net.encoder
        with torch.no_grad():
            out = enc.encode_batch(self.obs)
        self.assertEqual(tuple(out["all_tokens"].shape), (5, 51, enc.d_token_dim))
        self.assertEqual(tuple(out["all_mask"].shape), (5, 51))
        # context + memory tokens are always valid keys for the readout.
        self.assertTrue(out["all_mask"][:, :2].all())
        self.assertTrue(torch.equal(out["features"], out["memory_out"]))
        # The GRU input is the post-reasoning MEMORY token, not the context.
        with torch.no_grad():
            expected = enc.memory_gru(
                out["all_tokens"][:, 1, :], torch.zeros(5, enc.d_model)
            )
        self.assertTrue(torch.allclose(out["memory_out"], expected, atol=1e-6))

    def test_forward_sequences_shape_and_readout_grad(self):
        import torch

        vals = self.net.forward_sequences([self.obs[:3], self.obs[:1]])
        self.assertEqual(tuple(vals.shape), (2, 3))
        self.assertTrue(torch.isfinite(vals).all())
        vals.sum().backward()
        self.assertIsNotNone(self.net.readout_query.grad)
        self.assertTrue(torch.any(self.net.readout_query.grad != 0))
        self.net.zero_grad()

    def test_param_groups_cover_every_parameter(self):
        groups = self.net.param_groups(base_lr=1e-3)
        grouped = sum(p.numel() for g in groups for p in g["params"])
        total = sum(p.numel() for p in self.net.parameters())
        self.assertEqual(grouped, total)


def _collect_episodes(agent, n_episodes, collect_oracle):
    """Play self-play episodes via play_population_game and store the events."""
    import random

    import torch

    from sheepshead.training.league import SELF_PLAY
    from sheepshead.training.pfsp_runtime import play_population_game
    from sheepshead.training.train_league_ppo import _Seat

    random.seed(7)
    torch.manual_seed(7)
    opponents = [_Seat(agent, SELF_PLAY) for _ in range(4)]
    all_events = []
    for ep in range(n_episodes):
        mode = PARTNER_BY_CALLED_ACE if ep % 2 == 0 else PARTNER_BY_JD
        _, events, _, _, _ = play_population_game(
            training_agent=agent,
            opponents=opponents,
            partner_mode=mode,
            training_agent_position=(ep % 5) + 1,
            reward_mode="terminal",
            collect_oracle=collect_oracle,
        )
        agent.store_episode_events(events)
        all_events.append(events)
    return all_events


class TestOracleEventCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sheepshead.agent.ppo import PPOAgent
        from sheepshead import ACTIONS

        cls.agent = PPOAgent(len(ACTIONS))  # limited: shared inference agent

    def test_collect_oracle_attaches_state_to_every_event(self):
        self.agent.reset_storage()
        episodes = _collect_episodes(self.agent, 2, collect_oracle=True)
        for events in episodes:
            self.assertGreater(len(events), 0)
            for ev in events:
                self.assertIn("oracle_state", ev)
                self.assertIn("opp_hand_ids", ev["oracle_state"])
        self.assertTrue(all("oracle_state" in e for e in self.agent.events))
        self.agent.reset_storage()

    def test_legacy_schema_unchanged_without_collect_oracle(self):
        self.agent.reset_storage()
        episodes = _collect_episodes(self.agent, 2, collect_oracle=False)
        for events in episodes:
            for ev in events:
                self.assertNotIn("oracle_state", ev)
        self.assertTrue(all("oracle_state" not in e for e in self.agent.events))
        self.agent.reset_storage()


class TestOracleUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sheepshead.agent.ppo import PPOAgent
        from sheepshead import ACTIONS

        cls.ACTIONS = ACTIONS
        cls.PPOAgent = PPOAgent
        cls.agent = PPOAgent(len(ACTIONS), critic_mode="oracle")

    def test_dual_gae_and_update_run(self):
        agent = self.agent
        agent.reset_storage()
        _collect_episodes(agent, 2, collect_oracle=True)

        # Fill + dual GAE write the oracle fields into action events.
        agent._fill_oracle_values()
        adv, ret_o = agent.compute_gae_dual()
        self.assertGreater(adv.size, 0)
        for e in agent.events:
            if e["kind"] == "action":
                self.assertIn("value_oracle", e)
                self.assertIn("return_oracle", e)
                self.assertIn("return", e)

        stats = agent.update(epochs=1, batch_size=4)
        self.assertIsNotNone(stats["oracle"])
        for key in ("ev_oracle", "ev_limited", "value_loss"):
            self.assertIn(key, stats["oracle"])
        self.assertTrue(np.isfinite(stats["oracle"]["value_loss"]))

    def test_missing_oracle_state_fails_fast(self):
        agent = self.agent
        agent.reset_storage()
        _collect_episodes(agent, 1, collect_oracle=False)
        with self.assertRaises(ValueError):
            agent.update(epochs=1, batch_size=4)
        agent.reset_storage()

    def test_limited_mode_has_no_oracle_surface(self):
        limited = self.PPOAgent(len(self.ACTIONS))
        self.assertIsNone(limited.oracle_critic)
        self.assertIsNone(limited.oracle_optimizer)
        limited.reset_storage()
        _collect_episodes(limited, 1, collect_oracle=False)
        stats = limited.update(epochs=1, batch_size=4)
        self.assertIsNone(stats["oracle"])
        self.assertNotIn("value_oracle", stats["critic_losses"])


class TestGradientIsolation(unittest.TestCase):
    def test_no_parameter_sharing_and_no_gradient_leak(self):
        import torch
        import torch.nn.functional as F

        from sheepshead.agent.ppo import PPOAgent, device
        from sheepshead import ACTIONS

        agent = PPOAgent(len(ACTIONS), critic_mode="oracle")

        # Structural isolation: zero shared parameter objects.
        policy_param_ids = {
            id(p)
            for net in (agent.encoder, agent.actor, agent.critic)
            for p in net.parameters()
        }
        for p in agent.oracle_critic.parameters():
            self.assertNotIn(id(p), policy_param_ids)

        # Behavioral isolation: an oracle loss backward reaches no
        # policy/limited-critic parameter.
        agent.reset_storage()
        _collect_episodes(agent, 1, collect_oracle=True)
        agent._fill_oracle_values()
        agent.compute_gae_dual()
        kinds = [e["kind"] for e in agent.events]
        segments = agent._segments_from_events(kinds)
        oracle_seqs, ret_bt, old_bt = agent._build_oracle_minibatch(segments, kinds)
        values_bt = agent.oracle_critic.forward_sequences(oracle_seqs, device=device)
        loss = F.mse_loss(values_bt, ret_bt)
        loss.backward()
        for net in (agent.encoder, agent.actor, agent.critic):
            for p in net.parameters():
                self.assertIsNone(p.grad)
        self.assertTrue(
            any(
                p.grad is not None and torch.any(p.grad != 0)
                for p in agent.oracle_critic.parameters()
            )
        )
        agent.reset_storage()


class TestOracleCheckpoints(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="oracle_ckpt_test_")

    def test_roundtrips(self):
        import torch

        from sheepshead.agent.ppo import PPOAgent
        from sheepshead import ACTIONS

        path = os.path.join(self.dir, "a.pt")
        oracle_agent = PPOAgent(len(ACTIONS), critic_mode="oracle")
        oracle_agent.save(path)
        ckpt = torch.load(path, map_location="cpu")
        self.assertIn("oracle_state_dict", ckpt)
        self.assertEqual(ckpt["critic_mode"], "oracle")

        # oracle -> oracle: weights equal after load.
        reloaded = PPOAgent(len(ACTIONS), critic_mode="oracle")
        reloaded.load(path)
        for k, v in oracle_agent.oracle_critic.state_dict().items():
            self.assertTrue(torch.equal(v, reloaded.oracle_critic.state_dict()[k]))

        # oracle checkpoint -> limited agent: keys ignored, no crash.
        limited = PPOAgent(len(ACTIONS))
        limited.load(path)
        self.assertIsNone(limited.oracle_critic)

        # limited checkpoint -> oracle agent: fresh-init warm start, no crash.
        lim_path = os.path.join(self.dir, "b.pt")
        limited.save(lim_path)
        ckpt_lim = torch.load(lim_path, map_location="cpu")
        self.assertNotIn("oracle_state_dict", ckpt_lim)
        warm = PPOAgent(len(ACTIONS), critic_mode="oracle")
        warm.load(lim_path)
        self.assertIsNotNone(warm.oracle_critic)

    def test_snapshot_strip_oracle(self):
        import copy

        import torch

        from sheepshead.agent.ppo import PPOAgent
        from sheepshead import ACTIONS

        agent = PPOAgent(len(ACTIONS), critic_mode="oracle")
        snap = copy.deepcopy(agent)
        snap.strip_oracle()
        self.assertIsNone(snap.oracle_critic)
        self.assertEqual(snap.critic_mode, "limited")
        path = os.path.join(self.dir, "snap.pt")
        snap.save(path)
        ckpt = torch.load(path, map_location="cpu")
        self.assertNotIn("oracle_state_dict", ckpt)
        self.assertNotIn("oracle_optimizer", ckpt)
        # The original keeps its oracle.
        self.assertIsNotNone(agent.oracle_critic)


if __name__ == "__main__":
    unittest.main(verbosity=2)
