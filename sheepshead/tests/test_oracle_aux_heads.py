"""Tests for the reconstituted-run machinery: official oracle aux heads,
checkpoint compatibility, deal-seeded games (seat rotation), and the
extended greedy probe."""

import random

import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, TRUMP
from sheepshead.agent.oracle import team_aux_labels
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.game import Game
from sheepshead.tests.ppo_test_helpers import play_episodes, seed_all
from sheepshead.training.reward_shaping import compute_seen_trump_mask
from sheepshead.training.training_utils import (
    greedy_health_probe,
    seen_trump_recall_cards,
)

SEED = 20260725
ARCH = "perceiver-shared-v2"


def _agent(**kw):
    seed_all(SEED)
    return PPOAgent(len(ACTIONS), critic_mode="oracle", arch=ARCH, **kw)


def test_headed_oracle_update_and_roundtrip(tmp_path):
    agent = _agent(oracle_aux_heads=True)
    assert agent.oracle_critic.has_aux_heads
    play_episodes(agent, 6, collect_oracle=True, seed0=SEED)
    stats = agent.update(epochs=1, batch_size=2)
    assert stats["oracle"] is not None

    ckpt = str(tmp_path / "headed.pt")
    agent.save(ckpt)
    fresh = load_agent(ckpt)
    assert fresh.oracle_critic.has_aux_heads
    sd_a = agent.oracle_critic.state_dict()
    sd_b = fresh.oracle_critic.state_dict()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


def test_limited_points_head_trains_under_oracle_aux():
    # Regression: the oracle aux losses once shadowed the limited critic's
    # ``points_loss`` before total_loss was assembled, so the limited points
    # head received zero gradient whenever oracle aux heads were on (frozen
    # for the whole league_retention_pg run through 5.3M episodes).
    agent = _agent(oracle_aux_heads=True)
    before = {
        k: v.clone()
        for k, v in agent.critic.state_dict().items()
        if k.startswith("points_head")
    }
    assert before, "limited critic should expose a points head"
    play_episodes(agent, 6, collect_oracle=True, seed0=SEED)
    agent.update(epochs=1, batch_size=2)
    after = agent.critic.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_headed_agent_warm_starts_headless_checkpoint(tmp_path):
    headless = _agent()
    ckpt = str(tmp_path / "headless.pt")
    headless.save(ckpt)
    headed = _agent(oracle_aux_heads=True)
    headed.load(ckpt, load_optimizers=False)  # heads start fresh, no error
    # non-head oracle weights copied exactly
    sd_a = headless.oracle_critic.state_dict()
    sd_b = headed.oracle_critic.state_dict()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a if not k.startswith("team_"))


def test_headless_agent_loads_headed_checkpoint(tmp_path):
    headed = _agent(oracle_aux_heads=True)
    ckpt = str(tmp_path / "headed.pt")
    headed.save(ckpt)
    headless = _agent()
    headless.load(ckpt, load_optimizers=False)  # heads dropped, no error
    sd_a = headed.oracle_critic.state_dict()
    sd_b = headless.oracle_critic.state_dict()
    assert all(
        torch.equal(sd_a[k], sd_b[k]) for k in sd_b
    )  # every non-head weight copied


def test_headless_default_is_unchanged(tmp_path):
    agent = _agent()
    assert not agent.oracle_critic.has_aux_heads
    ckpt = str(tmp_path / "plain.pt")
    agent.save(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["oracle_aux_heads"] is False
    assert not any(k.startswith("team_") for k in payload["oracle_state_dict"])


def test_team_aux_labels_shapes_and_mask():
    agent = _agent()
    play_episodes(agent, 4, collect_oracle=True, seed0=SEED + 7)
    seqs = []
    seq = []
    for ev in agent.events:
        if "oracle_state" in ev:
            seq.append(ev["oracle_state"])
    seqs.append(seq)
    T = len(seq)
    member, team, mask = team_aux_labels(seqs, 1, T, torch.device("cpu"))
    assert member.shape == (1, T, 5) and team.shape == (1, T, 2)
    if bool(mask.any()):
        bits = member[mask].sum(-1)
        assert bool(((bits == 1) | (bits == 2)).all())
        assert float(team[mask].max()) <= 1.0
    agent.reset_storage()


def test_game_seed_reproduces_deal():
    from types import SimpleNamespace

    from sheepshead.training.pfsp_runtime import play_population_game

    agent = _agent()
    opp = [
        SimpleNamespace(agent=agent, metadata=SimpleNamespace(agent_id="x"))
        for _ in range(4)
    ]
    g1, *_ = play_population_game(
        agent,
        opp,
        PARTNER_BY_CALLED_ACE,
        training_agent_position=1,
        reward_mode="terminal",
        game_seed=777,
    )
    agent.reset_storage()
    g2, *_ = play_population_game(
        agent,
        opp,
        PARTNER_BY_CALLED_ACE,
        training_agent_position=4,
        reward_mode="terminal",
        game_seed=777,
    )
    agent.reset_storage()
    assert g1.blind == g2.blind  # same shuffle => same deal


def test_greedy_probe_reports_partner_convention():
    agent = _agent()
    probe = greedy_health_probe(agent, n_games=15, seed=3)
    assert "partner_trump_lead_rate" in probe
    assert "partner_leads" in probe
    assert 0.0 <= probe["partner_trump_lead_rate"] <= 100.0
    # Convention C2 canary (defender leads the called suit while unled)
    assert "called_suit_lead_rate" in probe
    assert "called_leads" in probe
    assert 0.0 <= probe["called_suit_lead_rate"] <= 100.0


def test_greedy_probe_reports_seen_trump_recall():
    """The probe reads the aux critic's seen-trump mask at every seat's
    scored play node (not the picker's alone): accuracy over the 14 trumps,
    recall over the must-remember trumps overall and per trick, and the
    false-seen rate. Informational keys, present and bounded."""
    agent = _agent(oracle_aux_heads=True)
    assert agent.critic.has_aux_heads
    probe = greedy_health_probe(agent, n_games=6, seed=1)
    for key in (
        "seen_trump_acc",
        "seen_trump_recall",
        "seen_trump_false_seen",
        "aux_secret_acc",
        "aux_points_exact",
        "aux_unseen_higher_acc",
    ):
        assert 0.0 <= probe[key] <= 100.0
    assert probe["aux_points_mae"] >= 0.0
    for key in ("seen_trump_recall_by_trick", "seen_trump_false_seen_by_trick"):
        assert len(probe[key]) == 6
        assert all(0.0 <= r <= 100.0 for r in probe[key])
    # Every play node of every seat is scored, the single-legal last-trick
    # nodes included (play_nodes counts only the multi-legal ones).
    assert probe["seen_trump_nodes"] > probe["play_nodes"] > 0
    assert probe["seen_trump_recall_cards"] > 0


def test_greedy_probe_seen_trump_absent_without_aux_heads():
    """A critic without aux heads (the `no-aux` ablation arch) scores no
    seen-trump nodes; the keys stay present and zero."""
    seed_all(SEED)
    agent = PPOAgent(len(ACTIONS), critic_mode="limited", arch="no-aux")
    assert not agent.critic.has_aux_heads
    probe = greedy_health_probe(agent, n_games=3, seed=1)
    assert probe["seen_trump_nodes"] == 0
    assert probe["seen_trump_recall_cards"] == 0
    assert probe["seen_trump_recall"] == 0.0


def test_seen_trump_recall_cards_is_memory_only():
    """The must-remember set: trumps the seat has seen that are neither in
    its hand nor on the table now — earlier tricks' trumps for every seat,
    the bury/discarded blind for the picker; never the visible ones."""
    rng = random.Random(5)
    checked_table = checked_history = checked_bury = 0
    for g in range(40):
        game = Game(seed=g)
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    if game.play_started:
                        cards = seen_trump_recall_cards(player)
                        truth = compute_seen_trump_mask(player)
                        idx = int(game.current_trick)
                        on_table = (
                            set(game.history[idx]) if idx < len(game.history) else set()
                        )
                        for card, seen in zip(TRUMP, truth):
                            if card in player.hand or card in on_table:
                                assert card not in cards
                                checked_table += card in on_table
                            elif seen:
                                assert card in cards
                                checked_history += card not in (
                                    *player.blind,
                                    *player.bury,
                                )
                                checked_bury += card in player.bury
                            else:
                                assert card not in cards
                    player.act(rng.choice(sorted(valid)))
                    valid = player.get_valid_action_ids()
    assert checked_table and checked_history and checked_bury
