"""Tests for the reconstituted-run machinery: official oracle aux heads,
checkpoint compatibility, deal-seeded games (seat rotation), and the
extended greedy probe."""

import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE
from sheepshead.agent.oracle import team_aux_labels
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.tests.ppo_test_helpers import play_episodes, seed_all
from sheepshead.training.training_utils import greedy_health_probe

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
