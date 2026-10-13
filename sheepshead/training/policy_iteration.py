#!/usr/bin/env python3
"""Search-Q regularized policy iteration (CE_Teacher_Design §20; the standing
recipe of §20.9-§20.11; Training_Program_Redesign §4.4): one offline
iteration from a frozen theta_k corpus to a certified candidate.

    fit      Stage 1 + 1b. Encode every searched play row through the
             frozen theta_k and fit the pooled advantage model — a twin of
             the play pointer (with the bilinear state x card term and a
             heteroscedastic residual head) on theta_k's frozen features,
             by Fay-Herriot iterated weighted least squares with held-out
             early stopping. Writes row_table.pt, advantage_model.pt,
             fit_report.json.
    target   Stage 2. Per-class residual variance, the Fay-Herriot blend of
             observed and pooled advantages, the posterior-SE-tempered
             tilt of theta_k's prior, and posterior-precision CE weights;
             writes the TARGETED CORPUS (search_target / search_weight on
             every searched play row) and target_report.json.
    distill  Stage 3. PG-off supervised projection: --trunk-epochs at --lr
             (everything trains), then --head-epochs bilinear-only epochs
             at --head-lr with the encoder frozen. Retention KL to theta_k
             on bidding-head and leaster-play rows; value / aux / oracle
             regression on every row. Checkpoints + greedy probes per
             epoch; the candidate is the LAST epoch of the schedule
             (distill_best.json) — held-out target KL is logged as a
             fidelity check and selects nothing (CE_Teacher_Design §20.13
             add. 31: the KL-best epoch left +0.003 of certified play on
             the table).
    cert     The adoption battery: n=1000 greedy probes on 4 fixed seeds,
             duplicate h2h vs theta_k (with the leaster-hand paired score),
             the pre-registered bars; writes cert.json.
    all      fit -> target -> distill (cert is run on the chosen epoch).

Literature (per stage): Fay & Herriot 1979 and Efron & Morris 1975 (the
small-area estimator behind Stage 1b); Kendall & Gal 2017 (the
heteroscedastic head); Vieillard et al. 2020, Peng et al. 2019, Nair et
al. 2020, Wang et al. 2020 (the advantage-weighted mirror-descent step of
Stage 2); Anthony et al. 2017 and Silver et al. 2017/2018 (the phase-pure
expert-iteration loop and its CE projection); Li & Hoiem 2016 (retention
KL on heads search cannot speak to); Kumar et al. 2022 (separating trunk
and head epochs).

Usage:
  uv run python -m sheepshead.training.policy_iteration all \\
      --corpus-dir runs/rc/pi/iter1/corpus --ckpt runs/rc/league/final.pt \\
      --out-dir runs/rc/pi/iter1
  uv run python -m sheepshead.training.policy_iteration cert \\
      --ckpt runs/rc/league/final.pt --out-dir runs/rc/pi/iter1 \\
      --candidate runs/rc/pi/iter1/distill_epoch6.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from sheepshead.agent import ppo as ppo_module
from sheepshead.agent.ppo import load_agent
from sheepshead.game import ACTIONS, TRUMP
from sheepshead.training.corpus_rows import load_shard, shard_paths
from sheepshead.training.reward_shaping import RETURN_SCALE
from sheepshead.training.search_advantage import (
    CAPACITIES,
    CE_WEIGHTS,
    NOISE_MODELS,
    NOISE_POOLS,
    TILT_SCALES,
    AdvantageModel,
    FitReport,
    RowTable,
    apply_noise_model,
    build_row_table,
    ce_weights,
    class_residual_variances,
    clipped_disagreement,
    cross_fit_predictions,
    estimate_residual_variance,
    evaluate_rows,
    fit_advantage_model_iterated,
    floor_class_variances,
    format_model_diagnostics,
    model_push_diagnostics,
    sigma_u2_rows,
    split_rows_by_game,
    targets_for_table,
    tilt_scale,
)
from sheepshead.training.training_utils import greedy_health_probe

TARGETED_SUBDIR = "targeted"
OOF_FILE = "oof_predictions.pt"

# Partition codes for the per-row policy-loss channel (0 = no policy loss).
# targeted  = searched play rows with a Stage-2 target (CE toward it);
# retention = bidding-head and leaster-play rows (KL to theta_k's act-time
#             policy — search cannot speak there, so the projection must
#             not move them: Learning without Forgetting, Li & Hoiem 2016).
SET_CODES = {"none": 0, "targeted": 1, "retention": 3}

# Adoption bars for ``cert`` (Training_Program_Redesign §4.4 / §20.9).
CERT_SEEDS = (98765, 98766, 98767, 98768)
CERT_GAMES = 1000
CERT_H2H_DEALS = 8000
CERT_BARS = {
    "partner_trump_lead_min": 96.5,
    # Operator decision 2026-10-09: the D8k lineage certified at 0.0-0.6%
    # because its theta_0 was already there; the recall lineage's theta_0
    # leads trump at trick 0 in 3.4-4.1% of greedy defender leads and its
    # first distill read 1.7%, trending down. The bar guards against a
    # regression toward the seed's 7%, not for the finished-run value.
    "t0_trump_lead_max": 4.0,
    "play_logit_spread_min": 3.6,
    # bidding-only route (§20.14 step 6): flag the iteration if the bidding
    # heads drifted below this at 2 SE.
    "bidding_route_min": -0.003,
}


def _log_to(path: str):
    f = open(path, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()

    return log


def _freeze(agent) -> None:
    for net in (agent.encoder, agent.actor, agent.critic):
        for p in net.parameters():
            p.requires_grad_(False)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_corpus(corpus_dir: str) -> tuple[list[dict], dict]:
    """(shards, manifest); each shard is ``{"episodes", "games"}``."""
    shards = [load_shard(p) for p in shard_paths(corpus_dir)]
    with open(os.path.join(corpus_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return shards, manifest


def game_indices_of(shard: dict) -> list[int]:
    games = shard["games"]
    if len(shard["episodes"]) != 5 * len(games):
        raise SystemExit("shard episode count is not 5 x games")
    return [g["game"] for g in games for _ in range(5)]


def load_episodes(corpus_dir: str) -> list:
    """All episodes from every shard, in shard order."""
    episodes = []
    for p in shard_paths(corpus_dir):
        episodes.extend(load_shard(p)["episodes"])
    return episodes


def split_by_game(episodes: list, holdout_frac: float, seed: int) -> tuple[list, list]:
    """(train, holdout) split at GAME granularity: every corpus game
    contributes five per-seat episodes (contiguous in shard order) that
    share one deal and one outcome, so the shuffle and the cut both happen
    over 5-episode groups. Seed-deterministic."""
    if len(episodes) % 5 != 0:
        raise SystemExit(
            f"corpus episode count {len(episodes)} is not a multiple of 5 "
            "(all-seat collection guarantees 5 per game)"
        )
    games = [episodes[i : i + 5] for i in range(0, len(episodes), 5)]
    random.Random(seed).shuffle(games)
    n_holdout_games = int(len(games) * holdout_frac)
    holdout = [ep for game in games[:n_holdout_games] for ep in game]
    train = [ep for game in games[n_holdout_games:] for ep in game]
    return train, holdout


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def stage_fit(args) -> FitReport:
    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    agent = load_agent(args.ckpt)
    _freeze(agent)
    shards, manifest = load_corpus(args.corpus_dir)
    if manifest.get("row_schema", 1) < 2:
        raise SystemExit("corpus needs row schema 2")

    table_path = os.path.join(args.out_dir, "row_table.pt")
    if os.path.exists(table_path) and not args.rebuild_table:
        table = RowTable.load(table_path)
        log(f"[fit] loaded row table {table_path}: {len(table)} rows")
    else:
        t0 = time.time()
        pieces = []
        for shard_idx, shard in enumerate(shards):
            try:
                piece = build_row_table(
                    agent,
                    shard["episodes"],
                    shard_idx=shard_idx,
                    game_indices=game_indices_of(shard),
                    buffer_episodes=args.buffer_episodes,
                    batch_segments=args.batch_segments,
                )
            except ValueError:  # a shard of nothing but retention rows
                log(f"[fit] encoded shard {shard_idx}: 0 targetable rows")
                continue
            pieces.append(piece)
            log(f"[fit] encoded shard {shard_idx}: {len(piece)} targetable rows")
        try:
            table = RowTable.concat(pieces)
        except ValueError as err:
            raise SystemExit(
                "the corpus has no searched play rows (every game a leaster, or "
                "the search schedule never fired) — nothing to fit"
            ) from err
        table.save(table_path)
        log(
            f"[fit] row table: {len(table)} targetable rows, "
            f"{int(table.has_q.sum())} with Q ({(time.time() - t0) / 60:.1f} min)"
        )
    _log_noise_model(table, args, log)
    train_idx, hold_idx = split_rows_by_game(table, args.holdout_frac, args.seed)
    log(f"[fit] split: {len(train_idx)} train / {len(hold_idx)} holdout rows (by game)")

    torch.manual_seed(args.seed)
    model, report = fit_advantage_model_iterated(
        lambda: AdvantageModel(
            agent, args.capacity, heteroscedastic=True, bilinear=True
        ),
        table,
        train_idx,
        hold_idx,
        fh_iterations=args.fh_iterations,
        class_shrink_rows=args.class_shrink_rows,
        var_floor=args.var_floor,
        log=log,
        epochs=args.fit_epochs,
        lr=args.fit_lr,
        weight_decay=args.weight_decay,
        batch_rows=args.batch_rows,
        patience=args.patience,
        seed=args.seed,
    )
    torch.save(model.state_dict(), os.path.join(args.out_dir, "advantage_model.pt"))
    log(
        f"[fit {args.capacity}] best epoch {report.best_epoch}: holdout wMSE "
        f"{report.best_holdout_mse:.3e} vs noise floor "
        f"{report.holdout_noise_floor:.3e}; sigma_u2 {report.sigma_u2:.3e}"
    )
    oof_path = os.path.join(args.out_dir, OOF_FILE)
    if args.cross_fit > 0:
        # §22.5 item 2: the held-out fit above chose the epoch count; the
        # fold models run exactly that many and predict only games they
        # never saw. sigma_u^2 (global and per class) then comes from
        # honest residuals on EVERY row.
        t0 = time.time()
        a_oof, su2_oof, fold, states = cross_fit_predictions(
            lambda: AdvantageModel(
                agent, args.capacity, heteroscedastic=True, bilinear=True
            ),
            table,
            folds=args.cross_fit,
            epochs=max(report.best_epoch, 1),
            var_floor=args.var_floor,
            seed=args.seed,
            log=log,
            lr=args.fit_lr,
            weight_decay=args.weight_decay,
            batch_rows=args.batch_rows,
            patience=args.fit_epochs + 1,
        )
        torch.save(
            {
                "a_model": a_oof,
                "sigma_u2_head": su2_oof,
                "fold": fold,
                "models": states,
            },
            oof_path,
        )
        ev = evaluate_rows(
            model,
            table,
            torch.arange(len(table)),
            var_floor=args.var_floor,
            oof=(a_oof, su2_oof),
        )
        su2_oof_global = estimate_residual_variance(
            ev.residual_sq_mean, ev.noise_var_mean
        )
        report.cross_fit = {
            "folds": args.cross_fit,
            "epochs": max(report.best_epoch, 1),
            "sigma_u2": su2_oof_global,
            "per_class": ev.per_class,
            "minutes": (time.time() - t0) / 60,
        }
        log(
            f"[cross-fit] {args.cross_fit} folds x {report.cross_fit['epochs']} epochs "
            f"({report.cross_fit['minutes']:.1f} min): out-of-fold wMSE "
            f"{ev.per_class['__all__']['weighted_mse']:.3e} vs floor "
            f"{ev.per_class['__all__']['noise_floor']:.3e}; sigma_u2 "
            f"{su2_oof_global:.3e} (held-out fit {report.sigma_u2:.3e}); top-agree "
            f"model {ev.per_class['__all__']['top_agree_model']:.3f} prior "
            f"{ev.per_class['__all__']['top_agree_prior']:.3f}"
        )
    elif os.path.exists(oof_path):
        os.remove(oof_path)  # a stale cross-fit must not steer the target stage
    for cls in sorted(report.per_class):
        r = report.per_class[cls]
        if r["n"] >= 100 and cls != "__all__":
            log(
                f"    {cls:28s} n={r['n']:5d} wMSE {r['weighted_mse']:.2e} "
                f"floor {r['noise_floor']:.2e} top-agree model "
                f"{r['top_agree_model']:.3f} prior {r['top_agree_prior']:.3f}"
            )
    with open(os.path.join(args.out_dir, "fit_report.json"), "w") as f:
        f.write(
            json.dumps(
                {"selected": args.capacity, **json.loads(report.to_json())}, indent=2
            )
        )
    return report


def _log_noise_model(table: RowTable, args, log) -> dict:
    """Set the table's working noise variance (search_advantage.apply_noise_model)
    and say what it did; fit and target must agree, so both call this."""
    st = apply_noise_model(
        table, args.noise_model, args.noise_shrink_nu, pool=args.noise_pool
    )
    if st["mode"] == "centered" and st.get("rows_centered"):
        by_class = st["s2_centered_by_class"]
        pooled = (
            f"{len(by_class)} class targets {min(by_class.values()):.2e}-"
            f"{max(by_class.values()):.2e} around the corpus-wide"
            if by_class
            else "corpus-wide target"
        )
        log(
            f"[noise] centered-label noise on {st['rows_centered']} rows "
            f"({pooled} centered s2 {st['s2_centered_global']:.3e}, nu "
            f"{args.noise_shrink_nu}); {st['rows_independent']} rows without the "
            f"field use the per-card independence formula; label noise mean "
            f"{st['noise_mean']:.3e} vs raw {st['noise_raw_mean']:.3e}"
        )
    elif st["mode"] == "centered":
        log(
            "[noise] corpus predates the centered field: every row uses the "
            f"per-card independence formula; label noise mean {st['noise_mean']:.3e} "
            f"vs raw {st['noise_raw_mean']:.3e}"
        )
    else:
        log(
            f"[noise] model {st['mode']}: label noise mean {st['noise_mean']:.3e} "
            f"vs raw {st['noise_raw_mean']:.3e}"
        )
    return st


# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
def stage_target(args) -> dict:
    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    agent = load_agent(args.ckpt)
    _freeze(agent)
    with open(os.path.join(args.out_dir, "fit_report.json")) as f:
        fit = json.load(f)
    model = AdvantageModel(agent, fit["selected"], heteroscedastic=True, bilinear=True)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "advantage_model.pt")))
    table = RowTable.load(os.path.join(args.out_dir, "row_table.pt"))
    _log_noise_model(table, args, log)
    oof = None
    oof_path = os.path.join(args.out_dir, OOF_FILE)
    if os.path.exists(oof_path) and fit.get("cross_fit"):
        d = torch.load(oof_path, map_location="cpu", weights_only=False)
        oof = (d["a_model"], d["sigma_u2_head"])
    if oof is not None:
        # §22.5 item 2: out-of-fold predictions on every row, so the
        # residuals are honest everywhere and the per-class sigma_u^2 needs
        # no floor; the blend's a_model is the one that never saw the row.
        sigma_u2 = float(fit["cross_fit"]["sigma_u2"])
        ev = evaluate_rows(
            model, table, torch.arange(len(table)), var_floor=args.var_floor, oof=oof
        )
        su2_all_rows = class_residual_variances(
            ev.per_class, sigma_u2, args.class_shrink_rows
        )
        su2_holdout = class_residual_variances(
            fit["per_class"], sigma_u2, args.class_shrink_rows
        )
        su2_by_class = su2_all_rows
        log(
            f"[target] capacity {fit['selected']}, cross-fit "
            f"{fit['cross_fit']['folds']} folds: out-of-fold per-class sigma_u2 "
            f"(global {sigma_u2:.3e}; {len(su2_by_class)} classes, no floor), "
            f"kappa {args.kappa}, tilt_max {args.tilt_max}"
        )
    else:
        sigma_u2 = float(fit["sigma_u2"])
        # §20.6: per-class residual variance re-estimated on EVERY row with Q
        # (train + holdout; the held-out cells are thin and the count
        # shrinkage would pin gamma near the global), shrunk toward the global
        # value — floored at the held-out estimate (fit_report per_class), the
        # model's true out-of-sample error on the class
        # (floor_class_variances).
        ev = evaluate_rows(
            model, table, torch.arange(len(table)), var_floor=args.var_floor
        )
        su2_all_rows = class_residual_variances(
            ev.per_class, sigma_u2, args.class_shrink_rows
        )
        su2_holdout = class_residual_variances(
            fit["per_class"], sigma_u2, args.class_shrink_rows
        )
        su2_by_class = floor_class_variances(su2_all_rows, su2_holdout)
        log(
            f"[target] capacity {fit['selected']}, per-class sigma_u2 "
            f"(global {sigma_u2:.3e}; {len(su2_by_class)} classes, "
            f"{sum(su2_by_class[c] > su2_all_rows[c] for c in su2_by_class)} floored "
            f"at the held-out estimate), kappa {args.kappa}, tilt_max {args.tilt_max}"
        )
    su2 = sigma_u2_rows(table.node_class, su2_by_class, sigma_u2)
    for cls in sorted(su2_by_class):
        r = ev.per_class.get(cls)
        if r and r["n"] >= 100:
            g = su2_by_class[cls] / (su2_by_class[cls] + r["noise_floor"])
            log(
                f"    {cls:28s} n={r['n']:5d} sigma_u2 {su2_by_class[cls]:.2e} "
                f"(all-rows {su2_all_rows[cls]:.2e} held-out "
                f"{su2_holdout.get(cls, float('nan')):.2e}) gamma {g:.2f}"
            )
    eta = float(args.eta) if args.tilt_scale == "global" else None
    if eta is not None:
        log(
            f"[target] tilt scale global: eta {eta:.4g} advantage units for every "
            f"row (kappa ignored); CE weights {args.weights}"
        )
    elif args.weights != "precision":
        log(f"[target] CE weights {args.weights}")
    built = targets_for_table(
        model,
        table,
        sigma_u2=su2,
        kappa=args.kappa,
        tilt_max=args.tilt_max,
        oof=oof,
        eta=eta,
    )

    # Arm_D_Review_202610 §8: is the model's part of each target supported
    # by the search labels? Diagnostic only; nothing below reads it.
    model_diag = model_push_diagnostics(
        table,
        built["a_model"],
        built["gamma"],
        tilt_scale(built["v_post"], kappa=args.kappa, eta=eta),
        table.noise_var,
        tilt_max=args.tilt_max,
        out_of_fold=oof is not None,
    )

    shards, manifest = load_corpus(args.corpus_dir)
    z_abs = built["z"].abs().amax(dim=1)
    overconf = clipped_disagreement(
        built["target"], z_abs, table, tilt_max=args.tilt_max
    )
    # CE weights (ce_weights): §20.9 posterior precision 1 / v_post, mean-
    # normalized and capped, or uniform (Arm_D_Review §6: precision weights
    # put half the dose on trick-4 rows under an honest noise model).
    weights = ce_weights(built["v_post"], args.weights, args.weight_max)
    kl_prior = (
        built["target"]
        * (
            torch.log(built["target"].clamp(min=1e-12))
            - torch.log(table.prior.clamp(min=1e-12))
        )
    ).sum(dim=1)
    for r, (shard_idx, ep_idx, ev_idx) in enumerate(table.refs):
        event = shards[shard_idx]["episodes"][ep_idx][ev_idx]
        valid = sorted(event["valid_actions"])
        t = built["target"][r]
        event["search_target"] = [float(t[a - 1]) for a in valid]
        event["has_search_target"] = True
        event["distill_set"] = "targeted"
        event["pi_target_source"] = "blend" if bool(table.has_q[r]) else "model"
        event["pi_gamma"] = float(built["gamma"][r])
        event["pi_v_post"] = float(built["v_post"][r])
        event["pi_z_max"] = float(z_abs[r])
        event["pi_kl_to_prior"] = float(kl_prior[r])
        event["search_weight"] = float(weights[r])

    out_dir = os.path.join(args.out_dir, TARGETED_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    for shard_idx, shard in enumerate(shards):
        path = os.path.join(out_dir, f"corpus_{shard_idx:04d}.pt")
        torch.save(shard, path + ".tmp")
        os.replace(path + ".tmp", path)
    report = {
        "rows": len(table),
        "rows_with_q": int(table.has_q.sum()),
        "sigma_u2": sigma_u2,
        "kappa": args.kappa,
        "tilt_scale": args.tilt_scale,
        "eta": eta,
        "tilt_max": args.tilt_max,
        "weights": args.weights,
        "weight_max": args.weight_max,
        "weight_p50": float(weights.median()),
        "weight_p90": float(weights.quantile(0.9)),
        "weight_max_observed": float(weights.max()),
        "gamma_p50_rows_with_q": float(built["gamma"][table.has_q].median())
        if bool(table.has_q.any())
        else None,
        "kl_target_prior_p50": float(kl_prior.median()),
        "kl_target_prior_p90": float(kl_prior.quantile(0.9)),
        "z_max_p50": float(z_abs.median()),
        "z_max_p90": float(z_abs.quantile(0.9)),
        "frac_z_clipped": float((z_abs >= args.tilt_max - 1e-6).float().mean()),
        "sigma_u2_by_class": su2_by_class,
        "sigma_u2_by_class_all_rows": su2_all_rows,
        "sigma_u2_by_class_holdout": su2_holdout,
        "noise_model": args.noise_model,
        "cross_fit": fit.get("cross_fit", {}).get("folds", 0) if oof is not None else 0,
        "clipped": overconf,
        "model_diagnostics": model_diag,
    }
    by_class: dict[str, list] = {}
    for r, cls in enumerate(table.node_class):
        by_class.setdefault(cls, []).append(r)
    report["per_class"] = {}
    for cls, idx in by_class.items():
        if len(idx) < 50:
            continue
        sub = torch.tensor(idx)
        report["per_class"][cls] = {
            "n": len(idx),
            "kl_p50": float(kl_prior[sub].median()),
            "z_max_p50": float(z_abs[sub].median()),
            "gamma_p50": float(built["gamma"][sub].median()),
            "clipped": clipped_disagreement(
                built["target"][sub],
                z_abs[sub],
                table.subset(sub),
                tilt_max=args.tilt_max,
            ),
        }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                **manifest,
                "targeted_from": os.path.abspath(args.corpus_dir),
                "policy_iteration_target": report,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.out_dir, "target_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(
        f"[target] wrote {len(shards)} targeted shards: KL(target||prior) p50 "
        f"{report['kl_target_prior_p50']:.4f} p90 {report['kl_target_prior_p90']:.4f}; "
        f"|z| p50 {report['z_max_p50']:.2f} p90 {report['z_max_p90']:.2f}; "
        f"clipped {100 * report['frac_z_clipped']:.1f}%; weight p50/p90 "
        f"{report['weight_p50']:.2f}/{report['weight_p90']:.2f}"
    )
    log(
        "[target] clipped rows (near one-hot targets): top card != committee's "
        + _pct(overconf["clipped_disagree_committee"])
        + f" of {overconf['n_clipped']}; where the committee is confident "
        + "(gap > 2 SE) "
        + _pct(overconf["clipped_disagree_committee_confident"])
        + f" of {overconf['n_clipped_committee_confident']}; != prior's "
        + _pct(overconf["clipped_disagree_prior"])
    )
    for cls in sorted(report["per_class"]):
        c = report["per_class"][cls]["clipped"]
        if c["n_clipped"] >= 50:
            log(
                f"    {cls:28s} clipped {c['n_clipped']:5d} disagree "
                f"{_pct(c['clipped_disagree_committee'])} confident "
                f"{_pct(c['clipped_disagree_committee_confident'])} "
                f"(n={c['n_clipped_committee_confident']})"
            )
    for line in format_model_diagnostics(model_diag, RETURN_SCALE):
        log(line)
    return report


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- #
# distill: rows, channels, losses
# --------------------------------------------------------------------------- #
def fmt_probe(probe: dict) -> str:
    """One log line of a greedy_health_probe dict: scalars to one decimal,
    the per-trick lists (seen-trump recall / false-seen, 763237e) joined
    with slashes."""
    parts = []
    for k, v in sorted(probe.items()):
        if isinstance(v, (list, tuple)):
            parts.append(f"{k} " + "/".join(f"{x:.0f}" for x in v))
        else:
            parts.append(f"{k} {v:.1f}")
    return "  ".join(parts)


def densify(probs, valid_actions, action_size: int) -> list:
    """Distribution over sorted(valid) -> fixed-width vector (zeros off the
    legal set), mirroring the target densification in store_episode_events."""
    dense = [0.0] * action_size
    if probs is not None:
        for action_id, p in zip(sorted(valid_actions), probs):
            dense[action_id - 1] = float(p)
    return dense


def store_episodes(agent, episodes: list) -> None:
    """Store corpus episodes into the agent's event buffer and annotate the
    stored action records with the distill channels (partition code, CE
    weight, act-time anchor, telemetry class) and the supervised value
    targets: return = final_score / RETURN_SCALE at every action row (MC
    target, terminal gamma = 1)."""
    for ep in episodes:
        start = len(agent.events)
        agent.store_episode_events(ep)
        src_actions = iter([e for e in ep if e["kind"] == "action"])
        for rec in agent.events[start:]:
            if rec["kind"] != "action":
                continue
            src = next(src_actions)
            rec["distill_set"] = SET_CODES.get(src.get("distill_set", "none"), 0)
            rec["search_weight"] = src.get("search_weight")
            rec["node_class"] = src.get("node_class", "")
            rec["conv_cs_ids"] = src.get("conv_cs_ids")
            rec["anchor_probs"] = densify(
                src.get("anchor_probs"), src["valid_actions"], agent.action_size
            )
            ret = rec["final_return"] / RETURN_SCALE
            rec["return"] = ret
            rec["advantage"] = 0.0
            rec["return_oracle"] = ret
            rec["value_oracle"] = 0.0


def flat_channels(agent, batch, kinds):
    """(set_flat, weight_flat, anchor_flat) aligned with the agent's
    ``_flatten_action_steps`` row order (same per-segment padding, same
    is-action selection). Rows without an explicit weight get 1."""
    device = ppo_module.device
    lengths = []
    set_list, weight_list, anchor_list, is_act_list = [], [], [], []
    for seg_start, seg_end in batch:
        ev_range = range(seg_start, seg_end + 1)
        lengths.append(seg_end - seg_start + 1)
        sets, weights, anchors, is_act = [], [], [], []
        for i in ev_range:
            ev = agent.events[i]
            action = kinds[i] == "action"
            is_act.append(action)
            sets.append(float(ev.get("distill_set", 0)) if action else 0.0)
            w = ev.get("search_weight") if action else None
            weights.append(float(w) if w is not None else 1.0)
            anchor = ev.get("anchor_probs") if action else None
            anchors.append(
                torch.tensor(
                    anchor or [0.0] * agent.action_size,
                    dtype=torch.float32,
                    device=device,
                )
            )
        set_list.append(torch.tensor(sets, dtype=torch.float32, device=device))
        weight_list.append(torch.tensor(weights, dtype=torch.float32, device=device))
        anchor_list.append(torch.stack(anchors, dim=0))
        is_act_list.append(torch.tensor(is_act, dtype=torch.bool, device=device))
    pad = agent._pad_to_bt
    flat_mask = pad(is_act_list, lengths, False).view(-1)
    set_flat = pad(set_list, lengths, 0.0).view(-1)[flat_mask]
    weight_flat = pad(weight_list, lengths, 1.0).view(-1)[flat_mask]
    anchor_bt = pad(anchor_list, lengths, 0.0)
    anchor_flat = anchor_bt.view(-1, anchor_bt.size(-1))[flat_mask]
    return set_flat, weight_flat, anchor_flat


_IS_PLAY = [name.startswith("PLAY ") for name in ACTIONS]
_IS_TRUMP_PLAY = [name.startswith("PLAY ") and name[5:] in TRUMP for name in ACTIONS]


def convention_rows(agent, batch, kinds):
    """Per-action-row convention-telemetry annotations in flatten order: a
    list of ``(conv_name, trick, tracked_action_id_set)`` tuples per row
    (empty when ineligible). def_trump_lead: standard-game defender lead
    holding both classes, tracked = trump leads (lower = better);
    partner_trump_lead: partner lead holding both, tracked = trump;
    called_suit_lead: the adherent ids the generator stored."""
    rows = []
    for seg_start, seg_end in batch:
        for i in range(seg_start, seg_end + 1):
            if kinds[i] != "action":
                continue
            ev = agent.events[i]
            cls = ev.get("node_class", "") or ""
            entries = []
            if cls.startswith("std|") and cls.endswith("-lead"):
                trick = int(cls.split("|")[1].split("-")[0][1:])
                mask = ev["mask"]
                valid = [a + 1 for a in range(len(mask)) if bool(mask[a])]
                trump_plays = {a for a in valid if _IS_TRUMP_PLAY[a - 1]}
                fail_plays = {a for a in valid if _IS_PLAY[a - 1]} - trump_plays
                both = bool(trump_plays and fail_plays)
                if "-defender-" in cls:
                    if both:
                        entries.append(("def_trump_lead", trick, trump_plays))
                    if ev.get("conv_cs_ids"):
                        entries.append(
                            ("called_suit_lead", trick, set(ev["conv_cs_ids"]))
                        )
                elif "-partner-" in cls and both:
                    entries.append(("partner_trump_lead", trick, trump_plays))
            rows.append(entries)
    return rows


def accumulate_conventions(conv_counts, conv_rows, logits_flat):
    """Fold one minibatch's greedy convention behavior into
    ``conv_counts[(name, trick)] = [eligible, led_tracked_class]``."""
    greedy = logits_flat.argmax(dim=-1)
    for row_idx, entries in enumerate(conv_rows):
        for name, trick, tracked in entries:
            bin_ = conv_counts.setdefault((name, trick), [0, 0])
            bin_[0] += 1
            bin_[1] += int(int(greedy[row_idx].item()) + 1 in tracked)


def convention_report(conv_counts) -> dict:
    out = {}
    for name in ("def_trump_lead", "partner_trump_lead", "called_suit_lead"):
        pooled = [0, 0]
        t0 = [0, 0]
        for (n, trick), (elig, led) in conv_counts.items():
            if n != name:
                continue
            pooled[0] += elig
            pooled[1] += led
            if trick == 0:
                t0[0] += elig
                t0[1] += led
        if pooled[0]:
            out[f"{name}_rate"] = 100.0 * pooled[1] / pooled[0]
            out[f"{name}_n"] = pooled[0]
        if t0[0]:
            out[f"t0_{name}_rate"] = 100.0 * t0[1] / t0[0]
            out[f"t0_{name}_n"] = t0[0]
    return out


def kd_kl(anchor_flat: torch.Tensor, logits_flat: torch.Tensor) -> torch.Tensor:
    """Per-row KL(anchor || pi) over the legal set. The anchor is a masked
    probability vector (zeros off-legal); its softmax over the legal set
    equals the reference's masked-logit softmax, so storing probabilities
    loses nothing (Hinton et al. 2015)."""
    legal = anchor_flat > 0.0
    log_a = torch.where(
        legal,
        torch.log(anchor_flat.clamp(min=1e-12)),
        torch.full_like(anchor_flat, -1e9),
    )
    p_ref = F.softmax(log_a, dim=-1)
    logp_cur = F.log_softmax(logits_flat, dim=-1)
    log_p_ref = torch.log(p_ref.clamp(min=1e-12))
    return (p_ref * (log_p_ref - logp_cur)).sum(dim=-1)


def distill_losses(agent, minibatch, forward, flat, dchan, args):
    """Total loss + telemetry for one minibatch: weighted CE on targeted
    rows, retention KL on retention rows, value / aux / oracle regression
    on every row."""
    set_flat, weight_flat, anchor_flat = dchan
    stats = {}
    logp = F.log_softmax(flat.logits_flat, dim=-1)
    zero = flat.logits_flat.new_zeros(())

    tg = set_flat == SET_CODES["targeted"]
    if tg.any():
        target = flat.search_target_flat[tg]
        ce = -(target * logp[tg]).sum(dim=-1)
        target_loss = (weight_flat[tg] * ce).mean()
        with torch.no_grad():
            ent = -(target.clamp(min=1e-12) * target.clamp(min=1e-12).log()).sum(-1)
            stats["target_ce"] = float(ce.mean())
            stats["target_kl"] = float((ce - ent).mean())
            stats["target_weight"] = float(weight_flat[tg].mean())
    else:
        target_loss = zero
    stats["target_rows"] = int(tg.sum())

    rt = set_flat == SET_CODES["retention"]
    if rt.any():
        retention_loss = kd_kl(anchor_flat[rt], flat.logits_flat[rt]).mean()
        stats["retention_kl"] = float(retention_loss.detach())
    else:
        retention_loss = zero
    stats["retention_rows"] = int(rt.sum())

    value_loss = F.mse_loss(flat.values_flat, flat.returns_flat)
    stats["value_mse"] = float(value_loss.detach())
    total = (
        args.lambda_ce * target_loss
        + args.lambda_ret * retention_loss
        + agent.value_loss_coeff * value_loss
    )
    if agent.critic.has_aux_heads:
        win_loss = F.binary_cross_entropy_with_logits(
            flat.win_logits_flat, flat.win_labels_flat
        )
        return_loss = F.smooth_l1_loss(
            flat.returns_pred_flat / RETURN_SCALE,
            flat.final_returns_labels_flat / RETURN_SCALE,
        )
        secret_loss = F.binary_cross_entropy_with_logits(
            flat.secret_logits_flat, flat.secret_labels_flat
        )
        points_pred = forward.points_pred_bt.view(-1, forward.points_pred_bt.size(-1))[
            minibatch.is_action_bt.view(-1)
        ]
        points_lbl = minibatch.points_bt.view(-1, minibatch.points_bt.size(-1))[
            minibatch.is_action_bt.view(-1)
        ]
        points_loss = F.smooth_l1_loss(
            points_pred / ppo_module.POINTS_SCALE, points_lbl / ppo_module.POINTS_SCALE
        )
        seen_loss = F.binary_cross_entropy_with_logits(
            flat.seen_trump_mask_logits_flat, flat.seen_trump_mask_labels_flat
        )
        unseen_loss = F.binary_cross_entropy_with_logits(
            flat.unseen_trump_higher_than_hand_logits_flat,
            flat.unseen_trump_higher_than_hand_labels_flat,
        )
        total = total + (
            agent.win_loss_coeff * win_loss
            + agent.return_loss_coeff * return_loss
            + agent.secret_loss_coeff * secret_loss
            + agent.points_loss_coeff * points_loss
            + agent.seen_trump_mask_loss_coeff * seen_loss
            + agent.unseen_trump_higher_than_hand_loss_coeff * unseen_loss
        )
    return total, stats


def oracle_loss_for_batch(agent, batch, kinds, minibatch):
    """Plain-MSE oracle value regression toward the MC return (+ oracle aux
    losses); None when the agent has no oracle or the rows carry no oracle
    states."""
    if agent.oracle_critic is None:
        return None
    if any(
        "oracle_state" not in agent.events[i] for s, e in batch for i in range(s, e + 1)
    ):
        return None
    oracle_seqs, returns_oracle_bt, _ = agent._build_oracle_minibatch(batch, kinds)
    values_bt, trunk_bt = agent.oracle_critic.forward_sequences_full(
        oracle_seqs, device=ppo_module.device
    )
    flat_idx = minibatch.is_action_bt.view(-1)
    loss = F.mse_loss(
        values_bt.reshape(-1)[flat_idx], returns_oracle_bt.reshape(-1)[flat_idx]
    )
    if agent.oracle_critic.has_aux_heads:
        membership, points = agent.oracle_critic.aux_losses(
            trunk_bt, oracle_seqs, minibatch.is_action_bt
        )
        loss = loss + (
            agent.oracle_membership_coeff * membership
            + agent.oracle_points_coeff * points
        )
    return loss


def run_epoch(agent, episodes, args, train: bool):
    """One pass over ``episodes`` in buffer-sized chunks; returns the
    row-weighted mean telemetry and the step count."""
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    conv_counts: dict = {}
    steps = 0
    order = list(range(len(episodes)))
    if train:
        random.shuffle(order)
    for chunk_start in range(0, len(order), args.buffer_episodes):
        chunk = [
            episodes[i] for i in order[chunk_start : chunk_start + args.buffer_episodes]
        ]
        agent.reset_storage()
        store_episodes(agent, chunk)
        states, masks_t, kinds = agent._prepare_training_views()
        segments = agent._segments_from_events(kinds)
        seg_order = list(range(len(segments)))
        if train:
            random.shuffle(seg_order)
        for mb_start in range(0, len(seg_order), args.batch_segments):
            batch = [
                segments[i]
                for i in seg_order[mb_start : mb_start + args.batch_segments]
            ]
            minibatch = agent._build_minibatch_tensors(batch, states, masks_t, kinds)
            with torch.set_grad_enabled(train):
                forward = agent._forward_vectorized(
                    minibatch.states_seqs, minibatch.masks_bt
                )
                flat = agent._flatten_action_steps(minibatch, forward)
                if flat is None:
                    continue
                dchan = flat_channels(agent, batch, kinds)
                accumulate_conventions(
                    conv_counts,
                    convention_rows(agent, batch, kinds),
                    flat.logits_flat.detach(),
                )
                total, stats = distill_losses(
                    agent, minibatch, forward, flat, dchan, args
                )
                o_loss = (
                    oracle_loss_for_batch(agent, batch, kinds, minibatch)
                    if args.train_oracle
                    else None
                )
                if o_loss is not None:
                    total = total + agent.oracle_value_loss_coeff * o_loss
                    stats["oracle_loss"] = float(o_loss.detach())
            if train:
                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                if agent.oracle_optimizer is not None:
                    agent.oracle_optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), agent.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    agent.encoder.parameters(), agent.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    agent.critic.parameters(), agent.max_grad_norm
                )
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()
                if o_loss is not None and agent.oracle_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(
                        agent.oracle_critic.parameters(), agent.max_grad_norm
                    )
                    agent.oracle_optimizer.step()
                agent.optimizer_steps_total += 1
            steps += 1
            row_weights = {
                "target_ce": stats["target_rows"],
                "target_kl": stats["target_rows"],
                "target_weight": stats["target_rows"],
                "retention_kl": stats["retention_rows"],
            }
            for key in (
                "target_ce",
                "target_kl",
                "target_weight",
                "retention_kl",
                "value_mse",
                "oracle_loss",
            ):
                if key in stats:
                    w = row_weights.get(key, 1)
                    if w:
                        totals[key] = totals.get(key, 0.0) + stats[key] * w
                        weights[key] = weights.get(key, 0.0) + w
            for key in ("target_rows", "retention_rows"):
                totals[key] = totals.get(key, 0.0) + stats[key]
                weights[key] = 1.0
    agent.reset_storage()
    out = {k: totals[k] / max(weights.get(k, 1.0), 1e-9) for k in totals}
    out.update(convention_report(conv_counts))
    return out, steps


def fmt_stats(stats: dict) -> str:
    parts = []
    for k in (
        "target_ce",
        "target_kl",
        "target_weight",
        "retention_kl",
        "value_mse",
        "oracle_loss",
    ):
        if k in stats:
            parts.append(f"{k} {stats[k]:.4f}")
    parts.append(
        "rows tg/rt {:.0f}/{:.0f}".format(
            stats.get("target_rows", 0), stats.get("retention_rows", 0)
        )
    )
    for k in (
        "t0_def_trump_lead_rate",
        "def_trump_lead_rate",
        "partner_trump_lead_rate",
        "t0_called_suit_lead_rate",
        "called_suit_lead_rate",
    ):
        if k in stats:
            parts.append(f"{k} {stats[k]:.1f} (n={stats[k[:-5] + '_n']:.0f})")
    return "  ".join(parts)


def stage_distill(args) -> list[str]:
    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    targeted_dir = os.path.join(args.out_dir, TARGETED_SUBDIR)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = load_agent(args.ckpt)
    if not getattr(agent.actor, "bilinear_pointer", False):
        raise SystemExit(
            "the standing projection trains the play pointer's bilinear term; "
            f"{args.ckpt} ({agent.arch_name}) has none"
        )
    episodes = load_episodes(targeted_dir)
    frac = (
        args.distill_holdout_frac
        if args.distill_holdout_frac is not None
        else args.holdout_frac
    )
    train_eps, holdout = split_by_game(episodes, frac, args.seed)
    log(
        f"[distill] {len(train_eps)} train / {len(holdout)} holdout episodes "
        f"(holdout {100 * frac:.1f}%; the held-out target KL is a fidelity read, "
        "it selects nothing)"
    )
    log_path = os.path.join(args.out_dir, "distill_log.jsonl")
    saved = []
    with open(log_path, "a") as log_f:

        def log_row(row: dict) -> None:
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

        log_row({"kind": "config", **{k: str(v) for k, v in vars(args).items()}})
        best_kl, best_epoch = float("inf"), 0
        if holdout:
            init_stats, _ = run_epoch(agent, holdout, args, train=False)
            best_kl = float(init_stats.get("target_kl", float("inf")))
            log(f"[distill epoch 0] holdout: {fmt_stats(init_stats)}")
            log_row({"kind": "holdout", "epoch": 0, **init_stats})
        n_epochs = args.trunk_epochs + args.head_epochs
        for epoch in range(1, n_epochs + 1):
            # Trunk epochs: everything trains at --lr. Head epochs: encoder
            # frozen and only the bilinear pointer tensors train, at
            # --head-lr (the §20.9 recipe: EV from the trunk pass, the
            # target realization from the head phase, in that order so
            # nothing erodes the head's work).
            head_phase = epoch > args.trunk_epochs
            for prm in agent.encoder.parameters():
                prm.requires_grad_(not head_phase)
            for name, prm in agent.actor.named_parameters():
                prm.requires_grad_(
                    (not head_phase) or name.startswith(("pointer_U", "pointer_V"))
                )
            agent.set_learning_rates(
                actor_lr=args.head_lr if head_phase else args.lr, critic_lr=args.lr
            )
            log(
                f"[distill epoch {epoch}] "
                + (
                    f"bilinear head only, lr {args.head_lr}"
                    if head_phase
                    else f"trunk, lr {args.lr}"
                )
            )
            t0 = time.time()
            train_stats, steps = run_epoch(agent, train_eps, args, train=True)
            log(
                f"[distill epoch {epoch}] train ({steps} steps, "
                f"{(time.time() - t0) / 60:.1f} min): {fmt_stats(train_stats)}"
            )
            log_row({"kind": "train", "epoch": epoch, **train_stats})
            if holdout:
                hold_stats, _ = run_epoch(agent, holdout, args, train=False)
                log(f"[distill epoch {epoch}] holdout: {fmt_stats(hold_stats)}")
                log_row({"kind": "holdout", "epoch": epoch, **hold_stats})
                kl = float(hold_stats.get("target_kl", float("inf")))
                if kl < best_kl:
                    best_kl, best_epoch = kl, epoch
            if args.probe_games:
                probe = greedy_health_probe(agent, n_games=args.probe_games, seed=0)
                log(f"[distill epoch {epoch}] probe: " + fmt_probe(probe))
                log_row({"kind": "probe", "epoch": epoch, **probe})
            ckpt_path = os.path.join(args.out_dir, f"distill_epoch{epoch}.pt")
            agent.save(ckpt_path)
            saved.append(ckpt_path)
            log(f"[distill epoch {epoch}] saved {ckpt_path}")
        # The candidate is the last epoch of the pinned schedule (§20.14
        # step 4, amended add. 31). Held-out target KL is a fidelity
        # average over every targeted row and does not track EV (add. 14,
        # 20); selecting on it dropped the epochs that carried the gain.
        chosen = n_epochs
        with open(os.path.join(args.out_dir, "distill_best.json"), "w") as f:
            json.dump(
                {
                    "best_epoch": chosen,
                    "best_epoch_by_kl": best_epoch,
                    "holdout_target_kl": best_kl,
                    "checkpoint": os.path.join(
                        args.out_dir, f"distill_epoch{chosen}.pt"
                    ),
                },
                f,
            )
        log(
            f"[distill] candidate epoch {chosen} "
            f"(holdout target KL best: {best_epoch} at {best_kl:.4f})"
        )
    return saved


# --------------------------------------------------------------------------- #
# cert
# --------------------------------------------------------------------------- #
def stage_cert(args) -> dict:
    """The adoption battery on ``--candidate`` (default: distill_best.json's
    epoch) against theta_k (``--ckpt``): greedy probes at n=CERT_GAMES on
    the fixed CERT_SEEDS, the duplicate h2h vs theta_k with its leaster-hand
    paired score, and the pre-registered bars. Writes cert.json."""
    from sheepshead.analysis.league_progress_eval import h2h_duplicate

    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    candidate = args.candidate
    if candidate is None:
        with open(os.path.join(args.out_dir, "distill_best.json")) as f:
            candidate = json.load(f)["checkpoint"]
    if not candidate:
        raise SystemExit("no candidate checkpoint (distill_best.json names none)")
    agent = load_agent(candidate)
    probes = []
    for seed in CERT_SEEDS[: args.cert_seeds]:
        t0 = time.time()
        probe = greedy_health_probe(agent, n_games=args.cert_games, seed=seed)
        probes.append(probe)
        log(
            f"[cert] probe seed {seed} ({(time.time() - t0) / 60:.0f} min): "
            + "  ".join(
                f"{k} {probe[k]:.1f}"
                for k in (
                    "called_suit_lead_rate",
                    "t0_trump_lead_rate",
                    "partner_trump_lead_rate",
                    "pick_rate",
                    "leaster_rate",
                    "alone_rate",
                    "play_logit_spread_med",
                )
            )
        )
    means = {
        k: float(np.mean([p[k] for p in probes]))
        for k in (
            "called_suit_lead_rate",
            "t0_trump_lead_rate",
            "partner_trump_lead_rate",
            "pick_rate",
            "leaster_rate",
            "alone_rate",
            "play_logit_spread_med",
        )
    }
    t0 = time.time()
    h2h = h2h_duplicate(candidate, args.ckpt, n_deals_per_mode=args.h2h_deals)
    log(
        f"[cert] h2h vs theta_k ({(time.time() - t0) / 60:.0f} min): edge "
        f"{h2h['edge']:+.4f} se {h2h['se']:.4f} (called "
        f"{h2h['modes']['called']['edge']:+.4f} / jd {h2h['modes']['jd']['edge']:+.4f}); "
        f"leaster hands {h2h['leaster']['edge']:+.4f} se {h2h['leaster']['se']:.4f} "
        f"(n={h2h['leaster']['n']})"
    )
    routed = {}
    if getattr(args, "routed_reads", True):
        from sheepshead.analysis.head_routed_h2h import routed_h2h

        # §20.14 step 5: the play-only route (bidding from theta_k) is the
        # compounding statistic — it strips the +-0.005 bidding variance the
        # trunk epochs add — and the bidding-only route is the drift guard.
        for name, bid, play in (
            ("play_only", args.ckpt, candidate),
            ("bidding_only", candidate, args.ckpt),
        ):
            t0 = time.time()
            res = routed_h2h(
                bid, play, args.ckpt, n_deals_per_mode=args.h2h_deals, lead_ckpt=play
            )
            routed[name] = {
                k: res[k] for k in ("edge", "se", "modes", "per_deal") if k in res
            }
            log(
                f"[cert] routed {name} vs theta_k ({(time.time() - t0) / 60:.0f} min): "
                f"{res['edge']:+.4f} se {res['se']:.4f}"
            )
    failures = []
    # Adoption gate (operator decision 2026-09-12): NON-INFERIORITY on the
    # full checkpoint — the per-iteration gain (~+0.003) sits inside the
    # 8000-deal SE, so positivity at 2 SE would reject every real step;
    # compounding is judged by the program-level slope of the play-only
    # route (stop_rules.iteration_stop), conventions are guards.
    if h2h["edge"] + 2.0 * h2h["se"] < 0.0:
        failures.append(f"h2h vs theta_k inferior at 2 SE ({h2h['edge']:+.4f})")
    if "bidding_only" in routed:
        b = routed["bidding_only"]
        if b["edge"] + 2.0 * b["se"] < CERT_BARS["bidding_route_min"]:
            failures.append(f"bidding drift {b['edge']:+.4f} (route guard)")
    if means["partner_trump_lead_rate"] < CERT_BARS["partner_trump_lead_min"]:
        failures.append(f"partner trump lead {means['partner_trump_lead_rate']:.1f}")
    if means["t0_trump_lead_rate"] > CERT_BARS["t0_trump_lead_max"]:
        failures.append(f"t0 defender trump lead {means['t0_trump_lead_rate']:.1f}")
    if means["play_logit_spread_med"] < CERT_BARS["play_logit_spread_min"]:
        failures.append(f"play logit spread {means['play_logit_spread_med']:.2f}")
    # --no-bars (validation runs only): record the battery, enforce nothing.
    enforced = not getattr(args, "no_bars", False)
    result = {
        "candidate": candidate,
        "theta_k": args.ckpt,
        "passed": (not failures) or not enforced,
        "bars_enforced": enforced,
        "failures": failures,
        "probes": probes,
        "probe_means": means,
        "h2h": h2h,
        "routed": routed,
        # The compounding statistic for stop_rules: play-only route when
        # read, else the full h2h.
        "compounding": {
            "edge": routed["play_only"]["edge"]
            if "play_only" in routed
            else h2h["edge"],
            "se": routed["play_only"]["se"] if "play_only" in routed else h2h["se"],
            "source": "play_only" if "play_only" in routed else "h2h",
        },
        "bars": CERT_BARS,
    }
    with open(os.path.join(args.out_dir, "cert.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(
        "[cert] "
        + ("PASS" if not failures else "FAIL: " + "; ".join(failures))
        + ("" if enforced else " (bars not enforced)")
    )
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("fit", "target", "distill", "cert", "all"))
    ap.add_argument("--corpus-dir", default=None, help="schema-2 corpus")
    ap.add_argument("--ckpt", required=True, help="theta_k, the corpus's generator")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--buffer-episodes", type=int, default=250)
    ap.add_argument("--batch-segments", type=int, default=32)
    fit = ap.add_argument_group("fit")
    fit.add_argument("--capacity", default="adapter", choices=CAPACITIES)
    fit.add_argument("--fit-epochs", type=int, default=200)
    fit.add_argument("--fit-lr", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=1e-2)
    fit.add_argument("--batch-rows", type=int, default=1024)
    fit.add_argument("--patience", type=int, default=25)
    fit.add_argument("--var-floor", type=float, default=1e-6)
    fit.add_argument("--fh-iterations", type=int, default=2)
    fit.add_argument("--class-shrink-rows", type=float, default=50.0)
    fit.add_argument("--rebuild-table", action="store_true")
    fit.add_argument(
        "--cross-fit",
        type=int,
        default=0,
        metavar="K",
        help="K-fold cross-fitting (§22.5): fold models at the held-out fit's "
        "epoch count predict only games they never saw; 0 = off",
    )
    fit.add_argument(
        "--noise-model",
        default="centered",
        choices=NOISE_MODELS,
        help="label noise: centered (measured, 2026-10-07), independent "
        "(raw * (1 - 1/n)), raw (the recipe before)",
    )
    fit.add_argument(
        "--noise-shrink-nu",
        type=float,
        default=4.0,
        help="shrinkage weight of the pooled centered variance per action",
    )
    fit.add_argument(
        "--noise-pool",
        default="class",
        choices=NOISE_POOLS,
        help="shrinkage target of the centered noise: the row's telemetry "
        "class mean (default) or the corpus-wide mean (the 2026-10-07 recipe, "
        "which overstated quiet classes' noise up to 25x)",
    )
    tgt = ap.add_argument_group("target")
    tgt.add_argument(
        "--kappa", type=float, default=1.0, help="tilt temperature in posterior SEs"
    )
    tgt.add_argument(
        "--tilt-scale",
        default="global",
        choices=TILT_SCALES,
        help="tilt temperature: global (default; the pinned recipe since the "
        "2026-10-10 arm V) = one --eta in advantage units for every row, the "
        "KL-regularized improvement step (Arm_D_Review §6.4); posterior = "
        "kappa * posterior SE per row (§20.9, the recipe before)",
    )
    tgt.add_argument(
        "--eta",
        type=float,
        default=0.0097,
        help="global tilt temperature in advantage units (--tilt-scale global); "
        "0.0097 = the dose calibration that matches the §20.9 recipe's "
        "KL(target||prior) p50 on the recall run's iteration-1 corpus",
    )
    tgt.add_argument("--tilt-max", type=float, default=8.0, help="|z| clip (nats)")
    tgt.add_argument(
        "--weights",
        default="uniform",
        choices=CE_WEIGHTS,
        help="CE weights per targeted row: uniform (default; arm V) or "
        "precision = 1 / v_post capped (§20.9, the recipe before)",
    )
    tgt.add_argument("--weight-max", type=float, default=5.0, help="CE weight cap")
    dst = ap.add_argument_group("distill")
    # §20.14 step 4 (pinned 2026-09-12): six trunk epochs at 3e-5, then
    # bilinear-only head epochs at 1e-3; retention KL x10.
    dst.add_argument("--trunk-epochs", type=int, default=6)
    dst.add_argument("--head-epochs", type=int, default=4)
    dst.add_argument("--lr", type=float, default=3e-5)
    dst.add_argument("--head-lr", type=float, default=1e-3)
    dst.add_argument("--lambda-ce", type=float, default=1.0)
    dst.add_argument("--lambda-ret", type=float, default=10.0)
    dst.add_argument("--no-oracle", dest="train_oracle", action="store_false")
    dst.add_argument(
        "--distill-holdout-frac",
        type=float,
        default=None,
        help="game-level holdout of the distill (default: --holdout-frac); "
        "the held-out target KL selects nothing since add. 31",
    )
    dst.add_argument("--probe-games", type=int, default=500)
    crt = ap.add_argument_group("cert")
    crt.add_argument("--candidate", default=None)
    crt.add_argument("--cert-seeds", type=int, default=len(CERT_SEEDS))
    crt.add_argument("--cert-games", type=int, default=CERT_GAMES)
    crt.add_argument("--h2h-deals", type=int, default=CERT_H2H_DEALS)
    crt.add_argument(
        "--no-routed-reads",
        dest="routed_reads",
        action="store_false",
        help="skip the head-routed play-only / bidding-only h2h reads",
    )
    crt.add_argument(
        "--no-bars",
        action="store_true",
        help="record the battery without enforcing the adoption bars "
        "(pipeline validation only — never for a real iteration)",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.tilt_scale == "global" and args.eta is None:
        raise SystemExit("--tilt-scale global needs --eta")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage in ("fit", "target", "distill", "all") and not args.corpus_dir:
        raise SystemExit(f"stage {args.stage} needs --corpus-dir")
    if args.stage in ("fit", "all"):
        stage_fit(args)
    if args.stage in ("target", "all"):
        stage_target(args)
    if args.stage in ("distill", "all"):
        stage_distill(args)
    if args.stage == "cert":
        stage_cert(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
