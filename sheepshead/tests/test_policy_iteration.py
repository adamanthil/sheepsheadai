"""Search-Q policy iteration (CE_Teacher_Design §20): the advantage model,
the Fay-Herriot blend, the tilt target and the staged trainer, exercised
on scripted-committee corpora from a real agent.
"""

import json
import os

import numpy as np
import pytest
import torch

from sheepshead.tests.distill_test_helpers import fresh_agent, generate_game
from sheepshead.training import policy_iteration as tpi
from sheepshead.training.corpus_rows import (
    encode_rows,
    iter_row_batches,
    store_corpus_episodes,
)
from sheepshead.training.distill_corpus import ROW_SCHEMA_VERSION
from sheepshead.training.search_advantage import (
    AdvantageModel,
    RowTable,
    apply_noise_model,
    blend_advantages,
    build_row_table,
    build_tilt_target,
    class_residual_variances,
    clipped_disagreement,
    estimate_residual_variance,
    fit_advantage_model,
    fit_advantage_model_iterated,
    floor_class_variances,
    sigma_u2_rows,
    split_rows_by_game,
)


def _corpus(agent, game_indices):
    episodes, games = [], []
    for g in game_indices:
        res = generate_game(agent, game_idx=g)
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
    agent = fresh_agent()
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
    agent = fresh_agent()
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
        # Scaled so the centered signal is well above the noise floor (the
        # uncentered fit used to "recover" mostly a per-row offset).
        slot_val = 3.0 * (table.hand_tokens[r] @ u)  # (8,)
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
    # A fresh agent's prior can tie the model on this synthetic case; the
    # MSE bars above carry the recovery claim.
    assert pooled["top_agree_model"] >= pooled["top_agree_prior"]


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
    agent = fresh_agent()
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
            "--trunk-epochs",
            "1",
            "--head-epochs",
            "0",
            "--no-oracle",
        ]
    )
    assert rc == 0
    table = RowTable.load(str(out_dir / "row_table.pt"))
    assert len(table) == n_targetable
    fit = json.loads((out_dir / "fit_report.json").read_text())
    assert fit["selected"] == "pointer" and np.isfinite(fit["sigma_u2"])
    assert fit["sigma_u2_by_class"]
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
            assert e["pi_target_source"] == "blend"
            assert 0.0 <= e["pi_gamma"] <= 1.0 and e["pi_v_post"] > 0.0
            assert e["search_weight"] > 0.0
    assert relabeled == n_targetable
    report = json.loads((out_dir / "target_report.json").read_text())
    assert report["rows"] == n_targetable and report["frac_z_clipped"] <= 1.0
    assert report["clipped"]["n_eligible"] > 0
    assert all(
        report["sigma_u2_by_class"][c] >= report["sigma_u2_by_class_all_rows"][c]
        for c in report["sigma_u2_by_class"]
    )
    assert report["weight_p50"] > 0.0
    assert os.path.exists(out_dir / "distill_epoch1.pt")
    log = [
        json.loads(line)
        for line in (out_dir / "distill_log.jsonl").read_text().splitlines()
    ]
    train = next(r for r in log if r["kind"] == "train")
    assert train["override_rows"] > 0 and train["retention_rows"] > 0
    assert np.isfinite(train["override_ce"])
    best = json.loads((out_dir / "distill_best.json").read_text())
    assert best["best_epoch"] == 1  # the last epoch of the schedule


def test_class_residual_variances_shrink_toward_global():
    """§20.6: a thin cell stays near the global value, a fat cell keeps its
    own; the per-row lookup falls back to the global for unseen classes,
    and a per-row sigma_u^2 changes gamma row by row."""
    per_class = {
        "__all__": {"n": 1000, "weighted_mse": 1e-3, "noise_floor": 2e-4},
        "thin": {"n": 5, "weighted_mse": 3e-4, "noise_floor": 2.4e-4},
        "fat": {"n": 5000, "weighted_mse": 3e-4, "noise_floor": 2.4e-4},
        "floor": {"n": 100, "weighted_mse": 1e-4, "noise_floor": 2e-4},
    }
    by = class_residual_variances(per_class, global_sigma_u2=1e-3, shrink_rows=50)
    assert "__all__" not in by
    assert abs(by["thin"] - 1e-3) < abs(by["fat"] - 1e-3)
    assert by["fat"] == pytest.approx((5000 * 6e-5 + 50 * 1e-3) / 5050, rel=1e-6)
    assert by["floor"] > 0.0
    rows = sigma_u2_rows(["fat", "unknown", "thin"], by, 1e-3)
    assert rows[1] == pytest.approx(1e-3) and rows[0] == pytest.approx(by["fat"])
    a_obs = torch.zeros(3, 2)
    a_model = torch.zeros(3, 2)
    has_q = torch.tensor([True, True, True])
    nv = torch.full((3,), 2.4e-4)
    _, v_post, gamma = blend_advantages(a_obs, a_model, has_q, nv, rows)
    assert gamma[0] < gamma[2] < gamma[1]  # fat cell trusts the model most
    assert torch.allclose(v_post, gamma * nv)


def test_distill_runs_the_full_schedule_and_certifies_the_last_epoch(tmp_path):
    """The candidate is the last epoch of the schedule; held-out target KL
    is recorded (epoch 0 and every epoch) for the fidelity check but
    selects nothing. On a tiny corpus with a large coefficient the
    projection moves the policy toward the targets (holdout target KL
    falls from epoch 0)."""
    agent = fresh_agent()
    ckpt = tmp_path / "theta_k.pt"
    agent.save(str(ckpt))
    shard = _corpus(agent, [3, 4, 5])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    torch.save(shard, corpus_dir / "corpus_0000.pt")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {"row_schema": ROW_SCHEMA_VERSION, "shards": [{"path": "corpus_0000.pt"}]}
        )
    )
    out_dir = tmp_path / "iter"
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
            "--no-oracle",
            "--trunk-epochs",
            "1",
            "--head-epochs",
            "2",
            "--lambda-ce",
            "5",
            "--lr",
            "1e-3",
        ]
    )
    assert rc == 0
    fit = json.loads((out_dir / "fit_report.json").read_text())
    assert fit["sigma_u2_by_class"]
    rows = [
        json.loads(line)
        for line in (out_dir / "distill_log.jsonl").read_text().splitlines()
    ]
    hold = {r["epoch"]: r for r in rows if r["kind"] == "holdout"}
    assert 0 in hold and 1 in hold
    assert hold[1]["override_kl"] < hold[0]["override_kl"]
    best = json.loads((out_dir / "distill_best.json").read_text())
    assert best["best_epoch"] == 3  # 1 trunk + 2 head epochs: the last one
    assert best["best_epoch_by_kl"] >= 1
    assert best["checkpoint"].endswith("distill_epoch3.pt")
    log_text = (out_dir / "policy_iteration.log").read_text()
    assert "[distill epoch 1] trunk, lr" in log_text
    assert "[distill epoch 2] bilinear head only" in log_text


def test_heteroscedastic_head_and_nll():
    """§20.8: the variance head yields positive per-row sigma_u^2, the NLL
    reduces to the weighted MSE ordering when variances are equal, and a row
    with larger learned variance contributes less squared-error pressure."""
    from sheepshead.training.search_advantage import gaussian_nll_rows

    agent = fresh_agent()
    shard = _corpus(agent, [3])
    table = build_row_table(
        agent, shard["episodes"], shard_idx=0, game_indices=[3] * len(shard["episodes"])
    )
    model = AdvantageModel(agent, "adapter", heteroscedastic=True)
    idx = torch.arange(len(table))
    adv, su2 = model.forward_with_variance(table.encoded(idx), table.prior)
    assert su2 is not None and su2.shape == (len(table),) and bool((su2 > 0).all())
    # The bilinear term is zero-initialized on the state side, so at init it
    # adds nothing to the plain pointer's scores.
    torch.manual_seed(3)
    plain = AdvantageModel(agent, "adapter")
    torch.manual_seed(3)
    bil = AdvantageModel(agent, "adapter", bilinear=True)
    with torch.no_grad():
        assert torch.allclose(
            plain(table.encoded(idx)), bil(table.encoded(idx)), atol=1e-6
        )
    assert bil.bilinear and not plain.bilinear
    assert (
        AdvantageModel(agent, "adapter").forward_with_variance(table.encoded(idx))[1]
        is None
    )
    # NLL: same total variance everywhere -> loss orders like the MSE.
    pred = torch.zeros_like(table.advantage)
    nv = table.noise_var.clone()
    same = torch.full_like(nv, 1e-3)
    nll_a, mse_a = gaussian_nll_rows(
        pred, table.advantage, table.label_mask, nv, same, 1e-6
    )
    # A per-row constant is centered away (the tilt never sees one) ...
    nll_c, _ = gaussian_nll_rows(
        pred * 0 + 0.05, table.advantage, table.label_mask, nv, same, 1e-6
    )
    assert float(nll_c) == pytest.approx(float(nll_a))
    # ... a wrong ordering is not.
    wrong = -table.advantage
    nll_b, _ = gaussian_nll_rows(
        wrong, table.advantage, table.label_mask, nv, same, 1e-6
    )
    assert torch.isfinite(nll_a) and nll_a < nll_b
    assert torch.allclose(
        mse_a,
        ((table.advantage**2) * table.label_mask).sum(1)
        / table.label_mask.sum(1).clamp(min=1),
    )
    # Doubling a row's learned variance halves its squared-error term.
    su2_big = same.clone()
    su2_big[0] = 1e-3 * 3  # total 4e-3 vs 2e-3 at noise 1e-3 (ignoring the tiny floor)
    n0 = gaussian_nll_rows(
        pred[:1],
        table.advantage[:1],
        table.label_mask[:1],
        torch.full((1,), 1e-3),
        same[:1],
        0.0,
    )[0]
    n1 = gaussian_nll_rows(
        pred[:1],
        table.advantage[:1],
        table.label_mask[:1],
        torch.full((1,), 1e-3),
        su2_big[:1],
        0.0,
    )[0]
    sq = float(
        ((table.advantage[0] ** 2) * table.label_mask[0]).sum()
        / table.label_mask[0].sum()
    )
    assert float(n0) == pytest.approx(0.5 * (sq / 2e-3 + np.log(2e-3)), rel=1e-4)
    assert float(n1) == pytest.approx(0.5 * (sq / 4e-3 + np.log(4e-3)), rel=1e-4)


def test_fit_fails_loudly_on_a_corpus_without_searched_rows(tmp_path):
    """An all-retention corpus (every game a leaster, or a schedule that
    never fired) must stop the stage with a readable message rather than a
    tensor error deep in the row table."""
    agent = fresh_agent()
    ckpt = tmp_path / "theta_k.pt"
    agent.save(str(ckpt))
    shard = _corpus(agent, [3, 4])
    for ep in shard["episodes"]:
        for e in ep:
            if e.get("kind") == "action":
                e["distill_set"] = "retention"
                e.pop("search_q", None)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    torch.save(shard, corpus_dir / "corpus_0000.pt")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {"row_schema": ROW_SCHEMA_VERSION, "shards": [{"path": "corpus_0000.pt"}]}
        )
    )
    with pytest.raises(SystemExit, match="no searched play rows"):
        tpi.main(
            [
                "fit",
                "--corpus-dir",
                str(corpus_dir),
                "--ckpt",
                str(ckpt),
                "--out-dir",
                str(tmp_path / "iter"),
                "--fit-epochs",
                "1",
            ]
        )


def test_cert_stage_records_battery_and_enforces_bars(tmp_path):
    """The cert stage on a fresh agent: the battery runs (probes, the
    duplicate h2h with its leaster-hand read), the absolute bars fail for a
    random policy, and --no-bars records the same battery without failing."""
    agent = fresh_agent()
    theta = tmp_path / "theta_k.pt"
    agent.save(str(theta))
    torch.manual_seed(9)
    cand = tmp_path / "cand.pt"
    fresh_agent().save(str(cand))
    common = [
        "cert",
        "--ckpt",
        str(theta),
        "--out-dir",
        str(tmp_path / "iter"),
        "--candidate",
        str(cand),
        "--cert-games",
        "2",
        "--cert-seeds",
        "1",
        "--h2h-deals",
        "2",
    ]
    assert tpi.main(common) == 0
    cert = json.loads((tmp_path / "iter" / "cert.json").read_text())
    assert cert["bars_enforced"] is True and cert["passed"] is False
    assert cert["failures"]
    assert set(cert["h2h"]) >= {"edge", "se", "modes", "leaster"}
    assert cert["h2h"]["leaster"]["n"] >= 0
    assert set(cert["probe_means"]) >= {"partner_trump_lead_rate", "t0_trump_lead_rate"}
    assert tpi.main(common + ["--no-bars"]) == 0
    relaxed = json.loads((tmp_path / "iter" / "cert.json").read_text())
    assert relaxed["bars_enforced"] is False and relaxed["passed"] is True
    # The battery still ran and still recorded its misses (the strings are
    # not compared: the two runs draw different probe deals).
    # The h2h gate is non-inferiority (§20.14 step 5): a candidate at parity
    # with theta_k does not fail it; the convention bars still fire here.
    assert relaxed["failures"] and not any("inferior" in f for f in relaxed["failures"])
    assert relaxed["compounding"]["source"] in ("h2h", "play_only")
    assert relaxed["h2h"]["edge"] + 2.0 * relaxed["h2h"]["se"] >= 0.0


def test_fmt_probe_handles_the_per_trick_lists():
    """The distill-epoch probe line must survive the probe's list-valued
    per-trick keys (763237e); a scalar-only format string crashed the
    stage at epoch 1 on 2026-10-06."""
    line = tpi.fmt_probe(
        {"pick_rate": 35.25, "seen_trump_recall_by_trick": [100.0, 99.5, 0.0]}
    )
    assert line == "pick_rate 35.2  seen_trump_recall_by_trick 100/100/0"


def test_heteroscedastic_fit_is_one_round_and_rejects_extra_var():
    """The Fay-Herriot reweighting rounds exist for the homoscedastic
    objective; a heteroscedastic model learns the per-row variance in its
    NLL, so the iterated wrapper fits it once whatever fh_iterations says
    and the per-row GLS weight is refused rather than silently dropped."""
    agent = fresh_agent()
    shard = _corpus(agent, [3, 4])
    table = build_row_table(
        agent,
        shard["episodes"],
        shard_idx=0,
        game_indices=[g["game"] for g in shard["games"] for _ in range(5)],
    )
    train_idx, hold_idx = split_rows_by_game(table, 0.5, seed=0)
    lines = []
    kwargs: dict = dict(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        batch_rows=64,
        patience=5,
        seed=0,
    )
    _, report = fit_advantage_model_iterated(
        lambda: AdvantageModel(agent, "pointer", heteroscedastic=True),
        table,
        train_idx,
        hold_idx,
        fh_iterations=2,
        class_shrink_rows=50.0,
        var_floor=1e-6,
        log=lines.append,
        **kwargs,
    )
    assert sum("FH round" in ln for ln in lines) == 1
    assert any("one round" in ln for ln in lines)
    assert report.sigma_u2_by_class
    lines.clear()
    fit_advantage_model_iterated(
        lambda: AdvantageModel(agent, "pointer"),
        table,
        train_idx,
        hold_idx,
        fh_iterations=2,
        class_shrink_rows=50.0,
        var_floor=1e-6,
        log=lines.append,
        **kwargs,
    )
    assert sum("FH round" in ln for ln in lines) == 2
    with pytest.raises(ValueError, match="homoscedastic"):
        fit_advantage_model(
            AdvantageModel(agent, "pointer", heteroscedastic=True),
            table,
            train_idx,
            hold_idx,
            var_floor=1e-6,
            log=lambda *_: None,
            extra_var=torch.full((len(table),), 1e-4),
            **kwargs,
        )


def test_variance_floor_and_clipped_disagreement():
    """The blend's per-class sigma_u^2 never sits below the held-out
    estimate (the all-rows value collapses to the shrinkage prior where
    the model's training residual is under the noise floor), and the
    over-confidence read counts, on clipped rows only, the target's top
    card against the committee's and the prior's."""
    assert floor_class_variances(
        {"a": 3e-6, "b": 2e-3, "c": 1e-4}, {"a": 1.3e-5, "b": 1e-3}
    ) == {"a": 1.3e-5, "b": 2e-3, "c": 1e-4}

    adv = torch.tensor(
        [
            [0.10, 0.00, -0.10, 0.0],  # committee confident: 0 wins by 0.10
            [0.01, 0.00, -0.01, 0.0],  # near tie
            [0.00, 0.05, -0.05, 0.0],  # committee confident: 1 wins
            [0.00, 0.05, 0.00, 0.0],  # not clipped
            [0.00, 0.00, 0.00, 0.0],  # single labeled action: ineligible
        ]
    )
    lab = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    prior = torch.tensor(
        [
            [0.6, 0.3, 0.1, 0.0],
            [0.3, 0.6, 0.1, 0.0],
            [0.3, 0.6, 0.1, 0.0],
            [0.3, 0.6, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [0.99, 0.01, 0.0, 0.0],  # agrees with committee and prior
            [0.01, 0.99, 0.0, 0.0],  # disagrees with committee (a tie), agrees prior
            [0.99, 0.01, 0.0, 0.0],  # disagrees with a confident committee AND prior
            [0.3, 0.6, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    table = RowTable(
        features=torch.zeros(5, 2),
        hand_tokens=torch.zeros(5, 8, 2),
        hand_ids=torch.zeros(5, 8, dtype=torch.long),
        masks=lab.clone(),
        advantage=adv,
        label_mask=lab,
        noise_var=torch.full((5,), 1e-4),  # SE 0.01
        prior=prior,
        has_q=torch.ones(5, dtype=torch.bool),
        node_class=["x"] * 5,
        refs=[(0, 0, i) for i in range(5)],
        game=torch.zeros(5, dtype=torch.long),
        noise_var_raw=torch.full((5,), 1e-4),
        noise_q_var=torch.full((5, 4), 1e-4),
        noise_s2c=torch.full((5, 4), float("nan")),
        noise_nobs=torch.zeros(5, 4, dtype=torch.long),
        noise_var_action=torch.full((5, 4), 1e-4),
    )
    z_abs = torch.tensor([8.0, 8.0, 8.0, 2.0, 8.0])
    out = clipped_disagreement(target, z_abs, table, tilt_max=8.0)
    assert out["n_eligible"] == 4 and out["n_clipped"] == 3
    assert out["n_clipped_committee_confident"] == 2  # rows 0 and 2
    assert out["clipped_disagree_committee"] == pytest.approx(2 / 3)
    assert out["clipped_disagree_committee_confident"] == pytest.approx(1 / 2)
    assert out["clipped_disagree_prior"] == pytest.approx(1 / 3)
    empty = clipped_disagreement(target, torch.zeros(5), table, tilt_max=8.0)
    assert empty["n_clipped"] == 0 and empty["clipped_disagree_committee"] is None


def _noise_table(raw, label_counts, q_var_rows, s2c_rows=None, nobs_rows=None):
    R, A = len(raw), 4
    lab = torch.zeros(R, A, dtype=torch.bool)
    for i, k in enumerate(label_counts):
        lab[i, :k] = True
    q_var = torch.tensor(q_var_rows)
    s2c = torch.full((R, A), float("nan"))
    nobs = torch.zeros(R, A, dtype=torch.long)
    if s2c_rows is not None:
        s2c = torch.tensor(s2c_rows)
        nobs = torch.tensor(nobs_rows)
    return RowTable(
        features=torch.zeros(R, 2),
        hand_tokens=torch.zeros(R, 8, 2),
        hand_ids=torch.zeros(R, 8, dtype=torch.long),
        masks=lab.clone(),
        advantage=torch.zeros(R, A),
        label_mask=lab,
        noise_var=torch.tensor(raw),
        prior=torch.full((R, A), 0.25),
        has_q=torch.tensor([k >= 1 for k in label_counts]),
        node_class=["x"] * R,
        refs=[(0, 0, i) for i in range(R)],
        game=torch.zeros(R, dtype=torch.long),
        noise_var_raw=torch.tensor(raw),
        noise_q_var=q_var,
        noise_s2c=s2c,
        noise_nobs=nobs,
        noise_var_action=q_var.clone(),
    )


def test_noise_model_centered_independent_raw():
    """raw puts the corpus row value on every action; independent is the
    exact centering under independent per-card noise, var(y_a) = s_a^2
    (1 - 2/n) + sum_b s_b^2 / n^2; centered uses the measured across-
    replicate variance of the centered Q, shrunk toward the df-weighted
    corpus mean over the replicate count, and falls back to independent
    per action where the field is missing. The row value is the mean over
    the labeled actions."""
    nan = float("nan")
    t = _noise_table(
        raw=[4e-4, 4e-4, 4e-4],
        label_counts=[2, 4, 4],
        q_var_rows=[
            [3e-4, 5e-4, nan, nan],
            [4e-4, 4e-4, 4e-4, 4e-4],
            [1e-4, 2e-4, 3e-4, 4e-4],
        ],
        s2c_rows=[
            [1e-4, 3e-4, nan, nan],  # two actions, 3 replicates each
            [nan, nan, nan, nan],  # no centered field -> independent
            [2e-4, nan, nan, nan],  # one action with 2 replicates
        ],
        nobs_rows=[[3, 3, 0, 0], [0, 0, 0, 0], [2, 1, 0, 0]],
    )
    apply_noise_model(t, "raw", 4.0)
    assert torch.allclose(t.noise_var, torch.tensor([4e-4, 4e-4, 4e-4]))
    assert torch.allclose(t.noise_var_action[1], torch.full((4,), 4e-4))
    assert torch.isnan(t.noise_var_action[0, 2])
    apply_noise_model(t, "independent", 4.0)
    # n = 2: var(y_a) = s_a^2 (1 - 1) + (s_a^2 + s_b^2) / 4 = (s_a^2 + s_b^2) / 4
    assert torch.allclose(t.noise_var_action[0, :2], torch.full((2,), 2e-4))
    # n = 4, equal noise: s^2 (1 - 2/4) + 4 s^2 / 16 = 0.75 s^2
    assert torch.allclose(t.noise_var_action[1], torch.full((4,), 3e-4))
    ind2 = [q * (1 - 0.5) + 1e-3 / 16 for q in (1e-4, 2e-4, 3e-4, 4e-4)]
    assert torch.allclose(t.noise_var_action[2], torch.tensor(ind2))
    assert float(t.noise_var[2]) == pytest.approx(sum(ind2) / 4)
    st = apply_noise_model(t, "centered", nu := 4.0)
    assert st["rows_centered"] == 2 and st["rows_independent"] == 1
    g = (2 * 1e-4 + 2 * 3e-4 + 1 * 2e-4) / 5  # df-weighted corpus mean
    assert st["s2_centered_global"] == pytest.approx(g)
    blend = lambda s2, n: (nu * g + (n - 1) * s2) / (nu + n - 1) / n  # noqa: E731
    assert float(t.noise_var_action[0, 0]) == pytest.approx(blend(1e-4, 3))
    assert float(t.noise_var_action[0, 1]) == pytest.approx(blend(3e-4, 3))
    assert float(t.noise_var[1]) == pytest.approx(3e-4)  # fallback row
    assert float(t.noise_var_action[2, 0]) == pytest.approx(blend(2e-4, 2))
    assert float(t.noise_var_action[2, 1]) == pytest.approx(ind2[1])  # per action
    with pytest.raises(ValueError):
        apply_noise_model(t, "bogus", 4.0)


def test_noise_pool_by_class_targets_the_class_mean():
    """Class pooling shrinks each card toward its telemetry class's df-
    weighted centered variance (itself shrunk toward the corpus-wide mean
    by NOISE_POOL_DF), so a quiet class no longer inherits the loud one's
    noise; corpus pooling (the 2026-10-07 recipe) gives every card the
    same target. A single-class table makes the two agree."""
    from sheepshead.training.search_advantage import NOISE_POOL_DF

    nan = float("nan")
    t = _noise_table(
        raw=[4e-4] * 4,
        label_counts=[2, 2, 2, 2],
        q_var_rows=[[3e-4, 3e-4, nan, nan]] * 4,
        s2c_rows=[
            [1e-4, 1e-4, nan, nan],  # quiet class, two rows
            [1e-4, 1e-4, nan, nan],
            [9e-4, 9e-4, nan, nan],  # loud class, two rows
            [9e-4, 9e-4, nan, nan],
        ],
        nobs_rows=[[3, 3, 0, 0]] * 4,
    )
    t.node_class = ["quiet", "quiet", "loud", "loud"]
    nu = 4.0
    st_corpus = apply_noise_model(t, "centered", nu, pool="corpus")
    g = 5e-4  # df-weighted corpus mean (equal df everywhere)
    assert st_corpus["s2_centered_global"] == pytest.approx(g)
    assert st_corpus["s2_centered_by_class"] == {}
    corpus_quiet = float(t.noise_var[0])
    st = apply_noise_model(t, "centered", nu, pool="class")
    df_c = 2 * 2 * 2  # two rows x two cards x 2 df
    pooled = {
        c: (df_c * m + NOISE_POOL_DF * g) / (df_c + NOISE_POOL_DF)
        for c, m in (("quiet", 1e-4), ("loud", 9e-4))
    }
    assert st["s2_centered_by_class"] == pytest.approx(pooled)
    blend = lambda s2, tgt: (nu * tgt + 2 * s2) / (nu + 2) / 3  # noqa: E731
    assert float(t.noise_var[0]) == pytest.approx(blend(1e-4, pooled["quiet"]))
    assert float(t.noise_var[2]) == pytest.approx(blend(9e-4, pooled["loud"]))
    assert float(t.noise_var[0]) < corpus_quiet < float(t.noise_var[2])
    t.node_class = ["x"] * 4
    apply_noise_model(t, "centered", nu, pool="class")
    assert float(t.noise_var[0]) == pytest.approx(corpus_quiet)
    with pytest.raises(ValueError):
        apply_noise_model(t, "centered", nu, pool="bogus")


def test_nll_accepts_per_action_noise():
    from sheepshead.training.search_advantage import gaussian_nll_rows

    pred = torch.tensor([[0.1, -0.1, 0.0]])
    adv = torch.tensor([[0.2, -0.2, 0.0]])
    lab = torch.tensor([[True, True, False]])
    su2 = torch.tensor([1e-4])
    row = gaussian_nll_rows(pred, adv, lab, torch.tensor([2e-4]), su2, 0.0)[0]
    same = gaussian_nll_rows(
        pred, adv, lab, torch.tensor([[2e-4, 2e-4, 9.0]]), su2, 0.0
    )[0]
    assert float(row) == pytest.approx(float(same))  # off-label noise ignored
    loud = gaussian_nll_rows(
        pred, adv, lab, torch.tensor([[2e-4, 2e-3, 9.0]]), su2, 0.0
    )[0]
    assert float(loud) != pytest.approx(float(row))


def test_fixed_epochs_runs_every_epoch_and_keeps_the_last():
    agent = fresh_agent()
    shard = _corpus(agent, [3, 4])
    table = build_row_table(
        agent,
        shard["episodes"],
        shard_idx=0,
        game_indices=[g["game"] for g in shard["games"] for _ in range(5)],
    )
    train_idx, hold_idx = split_rows_by_game(table, 0.5, seed=0)
    model = AdvantageModel(agent, "pointer")
    report = fit_advantage_model(
        model,
        table,
        train_idx,
        hold_idx,
        epochs=3,
        lr=1e-3,
        weight_decay=0.0,
        batch_rows=64,
        var_floor=1e-6,
        patience=1,  # would stop at once if early stopping were on
        seed=0,
        log=lambda *_: None,
        fixed_epochs=True,
    )
    assert report.best_epoch == 3 and len(report.epochs) == 3


def test_cross_fit_predicts_every_row_from_a_model_that_never_saw_its_game(
    monkeypatch,
):
    from sheepshead.training import search_advantage as sa

    agent = fresh_agent()
    shard = _corpus(agent, [3, 4, 5, 6])
    table = build_row_table(
        agent,
        shard["episodes"],
        shard_idx=0,
        game_indices=[g["game"] for g in shard["games"] for _ in range(5)],
    )
    seen = []
    real_fit = sa.fit_advantage_model

    def spy(model, tbl, train_idx, holdout_idx, **kw):
        seen.append(
            (set(tbl.game[train_idx].tolist()), set(tbl.game[holdout_idx].tolist()))
        )
        assert kw["fixed_epochs"] is True
        return real_fit(model, tbl, train_idx, holdout_idx, **kw)

    monkeypatch.setattr(sa, "fit_advantage_model", spy)
    a_oof, su2_oof, fold, states = sa.cross_fit_predictions(
        lambda: AdvantageModel(agent, "pointer", heteroscedastic=True),
        table,
        folds=2,
        epochs=1,
        var_floor=1e-6,
        seed=0,
        log=lambda *_: None,
        lr=1e-3,
        weight_decay=0.0,
        batch_rows=64,
        patience=99,
    )
    assert len(seen) == 2 and len(states) == 2
    for train_games, pred_games in seen:
        assert train_games and pred_games and not (train_games & pred_games)
    assert set(fold.tolist()) == {0, 1}
    # Every row of a game sits in one fold, and every row got a prediction
    # centered over its legal set.
    for g in set(table.game.tolist()):
        assert len(set(fold[table.game == g].tolist())) == 1
    assert a_oof.shape == table.advantage.shape
    legal = table.masks.bool()
    assert torch.allclose(
        (a_oof * legal).sum(dim=1), torch.zeros(len(table)), atol=1e-5
    )
    assert su2_oof is not None and bool((su2_oof > 0).all())
    with pytest.raises(ValueError):
        sa.cross_fit_predictions(
            lambda: AdvantageModel(agent, "pointer"),
            table,
            folds=1,
            epochs=1,
            var_floor=1e-6,
            seed=0,
            lr=1e-3,
            weight_decay=0.0,
            batch_rows=64,
            patience=99,
        )


def test_cross_fit_and_distill_holdout_through_the_stages(tmp_path):
    """--cross-fit K steers the target stage to the out-of-fold predictions
    (no floor, the fit report's cross_fit sigma_u2) and --distill-holdout-
    frac sets the distill split independently of the fit's."""
    agent = fresh_agent()
    ckpt = tmp_path / "theta_k.pt"
    agent.save(str(ckpt))
    shard = _corpus(agent, [3, 4, 5, 6])
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
            "0.25",
            "--cross-fit",
            "2",
            "--distill-holdout-frac",
            "0.5",
            "--probe-games",
            "0",
            "--trunk-epochs",
            "1",
            "--head-epochs",
            "0",
            "--no-oracle",
        ]
    )
    assert rc == 0
    fit = json.loads((out_dir / "fit_report.json").read_text())
    assert fit["cross_fit"]["folds"] == 2 and fit["cross_fit"]["epochs"] >= 1
    assert np.isfinite(fit["cross_fit"]["sigma_u2"])
    assert (out_dir / tpi.OOF_FILE).exists()
    report = json.loads((out_dir / "target_report.json").read_text())
    assert report["cross_fit"] == 2
    assert report["sigma_u2"] == pytest.approx(fit["cross_fit"]["sigma_u2"])
    log = (out_dir / "policy_iteration.log").read_text()
    assert "no floor" in log and "[cross-fit] 2 folds" in log
    assert "holdout 50.0%" in log
    assert "10 train / 10 holdout episodes" in log
    # Re-running the fit without cross-fitting removes the stale predictions.
    rc = tpi.main(
        [
            "fit",
            "--corpus-dir",
            str(corpus_dir),
            "--ckpt",
            str(ckpt),
            "--out-dir",
            str(out_dir),
            "--capacity",
            "pointer",
            "--fit-epochs",
            "1",
            "--batch-rows",
            "32",
            "--holdout-frac",
            "0.25",
        ]
    )
    assert rc == 0 and not (out_dir / tpi.OOF_FILE).exists()
