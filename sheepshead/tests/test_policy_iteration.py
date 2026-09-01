"""Search-Q policy iteration (CE_Teacher_Design §20): the advantage model,
the Fay-Herriot blend, the tilt target and the staged trainer, exercised
on scripted-committee corpora from a real agent.
"""

import json
import os

import numpy as np
import pytest
import torch

from sheepshead.tests.test_distill_pipeline import _fresh_agent, _generate_game
from sheepshead.training import train_policy_iteration as tpi
from sheepshead.training.corpus_rows import (
    encode_rows,
    iter_row_batches,
    store_corpus_episodes,
)
from sheepshead.training.distill_corpus import ROW_SCHEMA_VERSION
from sheepshead.training.search_advantage import (
    AdvantageModel,
    RowTable,
    blend_advantages,
    build_row_table,
    build_tilt_target,
    estimate_residual_variance,
    fit_advantage_model,
    split_rows_by_game,
)


def _corpus(agent, game_indices):
    episodes, games = [], []
    for g in game_indices:
        res = _generate_game(agent, game_idx=g)
        episodes.extend(res["episodes"])
        games.append(
            {
                "game": g,
                "mode": res["mode"],
                "committee_act": False,
                "is_leaster": res["is_leaster"],
                "alone_called": res["alone_called"],
            }
        )
    return {"episodes": episodes, "games": games}


def test_model_scatter_matches_actor_pointer():
    """With the actor's own pointer weights the pointer-capacity model must
    reproduce the actor's play/bury/under logits at every legal hand-card
    action: same inputs, same scatter."""
    agent = _fresh_agent()
    shard = _corpus(agent, [3])
    model = AdvantageModel(agent, "pointer")
    model.pointer_Wg.load_state_dict(agent.actor.pointer_Wg.state_dict())
    model.pointer_Wt.load_state_dict(agent.actor.pointer_Wt.state_dict())
    model.pointer_v.load_state_dict(agent.actor.pointer_v.state_dict())
    agent.reset_storage()
    store_corpus_episodes(agent, shard["episodes"])
    checked = 0
    for rows in iter_row_batches(agent, batch_segments=8):
        with torch.no_grad():
            forward = agent._forward_vectorized(
                rows.minibatch.states_seqs, rows.minibatch.masks_bt
            )
            flat = agent._flatten_action_steps(rows.minibatch, forward)
            enc = encode_rows(agent, rows)
            out = model(enc)
        assert flat is not None
        for r in range(len(rows)):
            legal = enc.masks[r].bool()
            slot_actions = out[r] != 0.0
            both = legal & slot_actions
            if both.any():
                assert torch.allclose(
                    out[r][both], flat.logits_flat[r][both], atol=1e-4
                )
                checked += int(both.sum())
    assert checked > 20


def test_fit_recovers_a_token_readout_to_the_noise_floor():
    """Synthetic Stage 1: labels are a fixed linear readout of each hand
    token (centered over the legal set) plus known noise. The pointer rung
    must pull the held-out weighted MSE from the label variance down to a
    few times the noise floor, and its top card must agree with the
    (noisy) observation more often than the prior's does — the §20.4
    pooling diagnostic on a case with a known answer."""
    agent = _fresh_agent()
    shard = _corpus(agent, list(range(3, 11)))
    table = build_row_table(
        agent,
        shard["episodes"],
        shard_idx=0,
        game_indices=[g["game"] for g in shard["games"] for _ in range(5)],
    )
    assert len(table) > 100 and bool(table.has_q.all())
    torch.manual_seed(1)
    d_token = table.hand_tokens.size(-1)
    u = torch.randn(d_token) / d_token**0.5
    noise_sd = 0.01
    gen = torch.Generator().manual_seed(2)
    play_map = agent.actor._map_cid_to_play_action_index
    for r in range(len(table)):
        slot_val = 0.3 * (table.hand_tokens[r] @ u)  # (8,)
        play_idx = play_map[table.hand_ids[r].long()]
        adv = torch.zeros(agent.action_size)
        lab = torch.zeros(agent.action_size, dtype=torch.bool)
        for s in range(8):
            a = int(play_idx[s])
            if a >= 0 and bool(table.masks[r][a]):
                adv[a] = slot_val[s]
                lab[a] = True
        if lab.sum() < 2:
            lab[:] = False
        if lab.any():
            adv = adv - adv[lab].mean()
        adv[~lab] = 0.0
        adv[lab] += noise_sd * torch.randn(int(lab.sum()), generator=gen)
        table.advantage[r] = adv
        table.label_mask[r] = lab
        table.noise_var[r] = noise_sd**2
        table.has_q[r] = bool(lab.any())
    train_idx, hold_idx = split_rows_by_game(table, 0.25, seed=0)
    model = AdvantageModel(agent, "pointer")
    report = fit_advantage_model(
        model,
        table,
        train_idx,
        hold_idx,
        epochs=200,
        lr=3e-3,
        weight_decay=0.0,
        batch_rows=64,
        var_floor=1e-6,
        patience=50,
        seed=0,
        log=lambda *_: None,
    )
    first = report.epochs[0]["holdout_weighted_mse"]
    assert report.best_holdout_mse < 0.05 * first
    assert report.best_holdout_mse < 4 * report.holdout_noise_floor
    assert report.sigma_u2 >= 0.0
    pooled = report.per_class["__all__"]
    assert pooled["n"] == len(hold_idx)
    assert pooled["top_agree_model"] > pooled["top_agree_prior"]


def test_blend_and_tilt_identities():
    a_obs = torch.tensor([[0.02, -0.02, 0.0], [0.05, -0.05, 0.0]])
    a_model = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    has_q = torch.tensor([True, False])
    nv = torch.tensor([1e-4, float("nan")])
    a_hat, v_post, gamma = blend_advantages(a_obs, a_model, has_q, nv, sigma_u2=1e-4)
    assert gamma[0] == pytest.approx(0.5) and gamma[1] == 0.0
    assert torch.allclose(a_hat[0], 0.5 * a_obs[0]) and torch.all(a_hat[1] == 0.0)
    assert v_post[0] == pytest.approx(0.5e-4) and v_post[1] == pytest.approx(1e-4)
    # gamma limits: a resolved node keeps its reading, a noisy one defers.
    _, _, g = blend_advantages(
        a_obs,
        a_model,
        torch.tensor([True, True]),
        torch.tensor([1e-9, 1.0]),
        sigma_u2=1e-4,
    )
    assert g[0] > 0.999 and g[1] < 1e-3
    assert estimate_residual_variance(3e-4, 1e-4) == pytest.approx(2e-4)
    assert estimate_residual_variance(1e-5, 1e-4) > 0.0

    prior = torch.tensor([[0.5, 0.3, 0.2, 0.0]])
    legal = torch.tensor([[True, True, True, False]])
    zero = torch.zeros(1, 4)
    t, z = build_tilt_target(
        prior, zero, torch.tensor([1e-4]), legal, kappa=1.0, tilt_max=8.0
    )
    assert torch.allclose(t, prior, atol=1e-6) and torch.all(z == 0.0)
    big = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    t, z = build_tilt_target(
        prior, big, torch.tensor([1e-4]), legal, kappa=1.0, tilt_max=8.0
    )
    assert int(t.argmax()) == 0 and t[0, 0] > 0.99 and float(z.max()) == 8.0
    assert t[0, 3] == 0.0
    # kappa = one-SE edge = one nat.
    one_se = torch.tensor([[0.01, 0.0, 0.0, 0.0]])
    _, z = build_tilt_target(
        prior, one_se, torch.tensor([1e-4]), legal, kappa=1.0, tilt_max=8.0
    )
    assert z[0, 0] == pytest.approx(1.0)


def test_end_to_end_stages_on_tiny_corpus(tmp_path):
    agent = _fresh_agent()
    ckpt = tmp_path / "theta_k.pt"
    agent.save(str(ckpt))
    shard = _corpus(agent, [3, 4, 5])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    torch.save(shard, corpus_dir / "corpus_0000.pt")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "row_schema": ROW_SCHEMA_VERSION,
                "ckpt": str(ckpt),
                "shards": [{"path": "corpus_0000.pt"}],
            }
        )
    )
    n_targetable = sum(
        1
        for ep in shard["episodes"]
        for e in ep
        if e.get("kind") == "action" and e["distill_set"] in ("override", "endorsed")
    )
    out_dir = tmp_path / "iter1"
    rc = tpi.main(
        [
            "all",
            "--corpus-dir",
            str(corpus_dir),
            "--ckpt",
            str(ckpt),
            "--out-dir",
            str(out_dir),
            "--capacity",
            "pointer",
            "--fit-epochs",
            "2",
            "--batch-rows",
            "32",
            "--buffer-episodes",
            "10",
            "--batch-segments",
            "4",
            "--holdout-frac",
            "0.34",
            "--probe-games",
            "0",
            "--epochs",
            "1",
            "--no-oracle",
        ]
    )
    assert rc == 0
    table = RowTable.load(str(out_dir / "row_table.pt"))
    assert len(table) == n_targetable
    fit = json.loads((out_dir / "fit_report.json").read_text())
    assert fit["selected"] == "pointer" and np.isfinite(fit["sigma_u2"])
    targeted = torch.load(out_dir / "targeted" / "corpus_0000.pt", weights_only=False)
    relabeled = 0
    for ep in targeted["episodes"]:
        for e in ep:
            if e.get("kind") != "action" or "pi_target_source" not in e:
                continue
            relabeled += 1
            assert e["distill_set"] == "override" and e["has_search_target"]
            assert len(e["search_target"]) == len(e["valid_actions"])
            assert sum(e["search_target"]) == pytest.approx(1.0, abs=1e-5)
            assert e["search_target_legacy"] is not None
            assert e["pi_target_source"] == "blend"
            assert 0.0 <= e["pi_gamma"] <= 1.0 and e["pi_v_post"] > 0.0
    assert relabeled == n_targetable
    report = json.loads((out_dir / "target_report.json").read_text())
    assert report["rows"] == n_targetable and report["frac_z_clipped"] <= 1.0
    assert os.path.exists(out_dir / "distill_epoch1.pt")
    log = [
        json.loads(line)
        for line in (out_dir / "distill_log.jsonl").read_text().splitlines()
    ]
    train = next(r for r in log if r["kind"] == "train")
    assert train["override_rows"] > 0 and train["endorsed_rows"] == 0
    assert np.isfinite(train["override_ce"])
