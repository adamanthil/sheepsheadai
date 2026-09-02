#!/usr/bin/env python3
"""Search-Q regularized policy iteration (CE_Teacher_Design §20): one
offline iteration from a frozen theta_k corpus to a distilled candidate.

    fit      Stage 1 + 1b. Encode every searched play row through the
             frozen theta_k, fit the pooled advantage model (one or more
             capacity rungs, selected by held-out weighted MSE), estimate
             the Fay-Herriot residual variance. Writes row_table.pt, one
             advantage_<capacity>.pt + fit_report_<capacity>.json per rung,
             and advantage_model.pt / fit_report.json for the selection.
    target   Stage 2. Blend observed and pooled advantages per row, tilt
             theta_k's prior by the posterior z-score, and write the
             TARGETED CORPUS: the input shards with every searched play row
             re-labeled (search_target := the §20 target, distill_set :=
             "override"; the pi_gumbel target is kept as
             search_target_legacy). Inspectable, reproducible, and in the
             exact schema the projection consumes.
    distill  Stage 3. PG-off supervised projection of the targeted corpus
             onto the policy — train_distill's loss loop (CE on override
             rows at uniform weight, KD anchor on retention rows,
             value/aux/oracle regression everywhere; §17.4) driven with the
             §20 defaults: one epoch, lambda_ce = lambda_ret = 1, omega
             fixed at 1 because the evidence weighting now lives in the
             target itself. Checkpoints + greedy probes per epoch.
    all      fit -> target -> distill.

The cert battery (n=1000 x 4 seeds + duplicate h2h vs theta_k) and the
WiSE-FT walk-back (``analysis/interpolate_checkpoints.py``) run after
``distill`` exactly as for the §17 arms.

Usage:
  uv run python -m sheepshead.training.train_policy_iteration all \\
      --corpus-dir runs/distill_corpus_q_202608_recovered \\
      --ckpt runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt \\
      --out-dir runs/policy_iteration_202609/iter1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

from sheepshead.agent.ppo import load_agent
from sheepshead.training import train_distill
from sheepshead.training.corpus_rows import load_shard, shard_paths
from sheepshead.training.search_advantage import (
    CAPACITIES,
    AdvantageModel,
    FitReport,
    RowTable,
    build_row_table,
    class_residual_variances,
    evaluate_rows,
    fit_advantage_model_iterated,
    fit_advantage_model_live,
    sigma_u2_rows,
    split_rows_by_game,
    targets_for_table,
)
from sheepshead.training.training_utils import greedy_health_probe

TARGETED_SUBDIR = "targeted"


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


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def stage_fit(args) -> FitReport:
    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    agent = load_agent(args.ckpt)
    _freeze(agent)
    shards, manifest = load_corpus(args.corpus_dir)
    if manifest.get("row_schema", 1) < 2:
        raise SystemExit("corpus needs row schema 2 (run recover_search_q first)")

    table_path = os.path.join(args.out_dir, "row_table.pt")
    if os.path.exists(table_path) and not args.rebuild_table:
        table = RowTable.load(table_path)
        log(f"[fit] loaded row table {table_path}: {len(table)} rows")
    else:
        t0 = time.time()
        pieces = []
        for shard_idx, shard in enumerate(shards):
            pieces.append(
                build_row_table(
                    agent,
                    shard["episodes"],
                    shard_idx=shard_idx,
                    game_indices=game_indices_of(shard),
                    buffer_episodes=args.buffer_episodes,
                    batch_segments=args.batch_segments,
                )
            )
            log(f"[fit] encoded shard {shard_idx}: {len(pieces[-1])} targetable rows")
        table = RowTable.concat(pieces)
        table.save(table_path)
        log(
            f"[fit] row table: {len(table)} targetable rows, "
            f"{int(table.has_q.sum())} with Q ({(time.time() - t0) / 60:.1f} min)"
        )
    train_idx, hold_idx = split_rows_by_game(table, args.holdout_frac, args.seed)
    log(f"[fit] split: {len(train_idx)} train / {len(hold_idx)} holdout rows (by game)")

    capacities = CAPACITIES if args.capacity == "all" else (args.capacity,)
    reports: dict[str, FitReport] = {}
    for cap in capacities:
        torch.manual_seed(args.seed)
        model = AdvantageModel(
            agent, cap, heteroscedastic=args.heteroscedastic, bilinear=args.bilinear
        )
        if cap == "trunk":
            report = fit_advantage_model_live(
                model,
                agent,
                shards,
                table,
                train_idx,
                hold_idx,
                epochs=args.fit_epochs,
                lr=args.fit_lr,
                weight_decay=args.weight_decay,
                buffer_episodes=args.buffer_episodes,
                batch_segments=args.batch_segments,
                var_floor=args.var_floor,
                patience=args.patience,
                seed=args.seed,
                log=log,
            )
        else:
            model, report = fit_advantage_model_iterated(
                lambda cap=cap: AdvantageModel(
                    agent,
                    cap,
                    heteroscedastic=args.heteroscedastic,
                    bilinear=args.bilinear,
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
        reports[cap] = report
        torch.save(
            model.state_dict(), os.path.join(args.out_dir, f"advantage_{cap}.pt")
        )
        with open(os.path.join(args.out_dir, f"fit_report_{cap}.json"), "w") as f:
            f.write(report.to_json())
        log(
            f"[fit {cap}] best epoch {report.best_epoch}: holdout wMSE "
            f"{report.best_holdout_mse:.3e} vs noise floor "
            f"{report.holdout_noise_floor:.3e}; sigma_u2 {report.sigma_u2:.3e}"
        )
        for cls in sorted(report.per_class):
            r = report.per_class[cls]
            if r["n"] >= 100 and cls != "__all__":
                log(
                    f"    {cls:28s} n={r['n']:5d} wMSE {r['weighted_mse']:.2e} "
                    f"floor {r['noise_floor']:.2e} top-agree model "
                    f"{r['top_agree_model']:.3f} prior {r['top_agree_prior']:.3f}"
                )
    best = min(reports, key=lambda c: reports[c].best_holdout_mse)
    log(f"[fit] SELECTED capacity {best} (held-out weighted MSE)")
    selected = reports[best]
    with open(os.path.join(args.out_dir, "fit_report.json"), "w") as f:
        f.write(
            json.dumps(
                {
                    "selected": best,
                    "heteroscedastic": bool(args.heteroscedastic),
                    "bilinear": bool(args.bilinear),
                    **json.loads(selected.to_json()),
                },
                indent=2,
            )
        )
    os.replace(
        os.path.join(args.out_dir, f"advantage_{best}.pt"),
        os.path.join(args.out_dir, "advantage_model.pt"),
    )
    torch.save(
        torch.load(os.path.join(args.out_dir, "advantage_model.pt")),
        os.path.join(args.out_dir, f"advantage_{best}.pt"),
    )
    return selected


# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
def stage_target(args) -> dict:
    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    agent = load_agent(args.ckpt)
    _freeze(agent)
    with open(os.path.join(args.out_dir, "fit_report.json")) as f:
        fit = json.load(f)
    model = AdvantageModel(
        agent,
        fit["selected"],
        heteroscedastic=bool(fit.get("heteroscedastic", False)),
        bilinear=bool(fit.get("bilinear", False)),
    )
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "advantage_model.pt")))
    table = RowTable.load(os.path.join(args.out_dir, "row_table.pt"))
    sigma_u2 = float(fit["sigma_u2"]) if args.sigma_u2 is None else args.sigma_u2
    node_variance = False
    if args.variance_mode == "node":
        if not model.heteroscedastic:
            raise SystemExit("--variance-mode node needs a fit with --heteroscedastic")
        node_variance = True
        su2 = sigma_u2
        log(
            f"[target] capacity {fit['selected']}, per-NODE sigma_u2 from the "
            f"heteroscedastic head (global {sigma_u2:.3e}), kappa {args.kappa}, "
            f"tilt_max {args.tilt_max}"
        )
    elif args.variance_mode == "class":
        # §20.6: per-class residual variance, shrunk toward the global.
        if args.variance_rows == "all":
            # Re-estimate on EVERY row with Q (train + holdout). The held-out
            # cells are thin (~100 rows at t0 leads) and the count shrinkage
            # then pins gamma near the global value; the fit shows little
            # train/holdout gap, so the train residual is a mild under-
            # estimate at ten times the rows (§20.6 arm 3).
            ev = evaluate_rows(
                model, table, torch.arange(len(table)), var_floor=args.var_floor
            )
            su2_by_class: dict[str, float] = class_residual_variances(
                ev.per_class, sigma_u2, args.class_shrink_rows
            )
        else:
            su2_by_class = fit.get("sigma_u2_by_class") or {}
            if not su2_by_class:
                raise SystemExit("fit_report.json has no sigma_u2_by_class; refit")
        su2 = sigma_u2_rows(table.node_class, su2_by_class, sigma_u2)
        log(
            f"[target] capacity {fit['selected']}, per-class sigma_u2 from "
            f"{args.variance_rows} rows (global {sigma_u2:.3e}; "
            f"{len(su2_by_class)} classes), kappa {args.kappa}, tilt_max {args.tilt_max}"
        )
        for cls in sorted(su2_by_class):
            r = (ev.per_class if args.variance_rows == "all" else fit["per_class"]).get(
                cls
            )
            if r and r["n"] >= 100:
                g = su2_by_class[cls] / (su2_by_class[cls] + r["noise_floor"])
                log(
                    f"    {cls:28s} n={r['n']:5d} sigma_u2 {su2_by_class[cls]:.2e} gamma {g:.2f}"
                )
    else:
        su2 = sigma_u2
        log(
            f"[target] capacity {fit['selected']}, global sigma_u2 {sigma_u2:.3e}, "
            f"kappa {args.kappa}, tilt_max {args.tilt_max}"
        )
    built = targets_for_table(
        model,
        table,
        sigma_u2=su2,
        kappa=args.kappa,
        tilt_max=args.tilt_max,
        node_variance=node_variance,
    )

    shards, manifest = load_corpus(args.corpus_dir)
    action_size = agent.action_size
    z_abs = built["z"].abs().amax(dim=1)
    # §20.9 posterior-precision CE weights: 1 / v_post, mean-normalized over
    # the targeted rows so the dose is unchanged and only its allocation
    # moves, capped so no row dominates.
    weights = None
    if args.weight_mode == "precision":
        prec = 1.0 / built["v_post"].clamp(min=1e-9)
        weights = (prec / prec.mean()).clamp(max=args.weight_max)
        weights = weights / weights.mean()
    kl_prior = (
        built["target"]
        * (
            torch.log(built["target"].clamp(min=1e-12))
            - torch.log(table.prior.clamp(min=1e-12))
        )
    ).sum(dim=1)
    for r, (shard_idx, ep_idx, ev_idx) in enumerate(table.refs):
        ev = shards[shard_idx]["episodes"][ep_idx][ev_idx]
        valid = sorted(ev["valid_actions"])
        t = built["target"][r]
        ev["search_target_legacy"] = ev.get("search_target")
        ev["search_target"] = [float(t[a - 1]) for a in valid]
        ev["has_search_target"] = True
        ev["distill_set"] = "override"
        ev["pi_target_source"] = "blend" if bool(table.has_q[r]) else "model"
        ev["pi_gamma"] = float(built["gamma"][r])
        ev["pi_v_post"] = float(built["v_post"][r])
        ev["pi_z_max"] = float(z_abs[r])
        ev["pi_kl_to_prior"] = float(kl_prior[r])
        ev["search_weight"] = float(weights[r]) if weights is not None else None
        assert len(ev["search_target"]) == len(valid) and len(t) == action_size

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
        "variance_mode": args.variance_mode,
        "weight_mode": args.weight_mode,
        "weight_p50": float(weights.median()) if weights is not None else None,
        "weight_p90": float(weights.quantile(0.9)) if weights is not None else None,
        "kappa": args.kappa,
        "tilt_max": args.tilt_max,
        "gamma_p50_rows_with_q": float(built["gamma"][table.has_q].median())
        if bool(table.has_q.any())
        else None,
        "kl_target_prior_p50": float(kl_prior.median()),
        "kl_target_prior_p90": float(kl_prior.quantile(0.9)),
        "z_max_p50": float(z_abs.median()),
        "z_max_p90": float(z_abs.quantile(0.9)),
        "frac_z_clipped": float((z_abs >= args.tilt_max - 1e-6).float().mean()),
    }
    by_class: dict[str, list] = {}
    for r, cls in enumerate(table.node_class):
        by_class.setdefault(cls, []).append(r)
    report["per_class"] = {
        cls: {
            "n": len(idx),
            "kl_p50": float(kl_prior[idx].median()),
            "z_max_p50": float(z_abs[idx].median()),
            "gamma_p50": float(built["gamma"][idx].median()),
        }
        for cls, idx in by_class.items()
        if len(idx) >= 50
    }
    manifest = {
        **manifest,
        "targeted_from": os.path.abspath(args.corpus_dir),
        "policy_iteration_target": report,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "target_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(
        f"[target] wrote {len(shards)} targeted shards: KL(target||prior) p50 "
        f"{report['kl_target_prior_p50']:.4f} p90 {report['kl_target_prior_p90']:.4f}; "
        f"|z| p50 {report['z_max_p50']:.2f} p90 {report['z_max_p90']:.2f}; "
        f"clipped {100 * report['frac_z_clipped']:.1f}%"
    )
    return report


# --------------------------------------------------------------------------- #
# distill
# --------------------------------------------------------------------------- #
def distill_args(args) -> argparse.Namespace:
    """The train_distill loss-loop configuration for the §20 projection.
    omega_max = 1 makes the AWR weight identically 1 (the evidence weight
    lives in the target now); no endorsed rows exist, so lambda_end is
    inert; the §17.12-§17.14 ablation switches stay off."""
    return argparse.Namespace(
        lambda_ce=args.lambda_ce,
        lambda_end=0.0,
        lambda_ret=args.lambda_ret,
        beta=0.03,
        omega_max=1.0,
        kd_tau=args.kd_tau,
        train_oracle=args.train_oracle,
        buffer_episodes=args.buffer_episodes,
        batch_segments=args.batch_segments,
        gap_floor=0.0,
        stop_grad_value=False,
        no_value_aux=False,
        recomputed_anchors=False,
    )


def stage_distill(args) -> list[str]:
    log = _log_to(os.path.join(args.out_dir, "policy_iteration.log"))
    targeted_dir = args.targeted_dir or os.path.join(args.out_dir, TARGETED_SUBDIR)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = load_agent(args.ckpt)
    agent.set_learning_rates(actor_lr=args.lr, critic_lr=args.lr)
    episodes = train_distill.load_shards(targeted_dir)
    train_eps, holdout = train_distill.split_by_game(
        episodes, args.holdout_frac, args.seed
    )
    log(
        f"[distill] {len(train_eps)} train / {len(holdout)} holdout episodes from {targeted_dir}"
    )
    dargs = distill_args(args)
    log_path = os.path.join(args.out_dir, "distill_log.jsonl")
    saved = []
    with open(log_path, "a") as log_f:

        def log_row(row: dict) -> None:
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

        log_row({"kind": "config", **{k: str(v) for k, v in vars(args).items()}})
        # Held-out target KL before any update: the projection's starting
        # point and the reference for the stop rule (§20.6: iteration 1
        # ended with this number unchanged).
        best_kl = float("inf")
        best_epoch = 0
        # Epochs with the encoder frozen (§20.9 head-first): either the
        # first N (--freeze-encoder-epochs) or an explicit list
        # (--freeze-epochs "2,3"), e.g. trunk epoch first for EV, then the
        # head phase last so nothing erodes it (arm 5d).
        frozen_epochs = set(range(1, args.freeze_encoder_epochs + 1))
        if args.freeze_epochs:
            frozen_epochs = {int(x) for x in args.freeze_epochs.split(",") if x.strip()}
        if holdout:
            init_stats, _ = train_distill.run_epoch(agent, holdout, dargs, train=False)
            best_kl = float(init_stats.get("override_kl", float("inf")))
            log(f"[distill epoch 0] holdout: {train_distill.fmt_stats(init_stats)}")
            log_row({"kind": "holdout", "epoch": 0, **init_stats})
        for epoch in range(1, args.epochs + 1):
            # §20.9 head-first projection (LP-FT, Kumar et al. 2022): the
            # encoder is frozen for the first --freeze-encoder-epochs at
            # --head-lr, then everything trains at --lr.
            freeze = epoch in frozen_epochs
            for prm in agent.encoder.parameters():
                prm.requires_grad_(not freeze)
            agent.set_learning_rates(
                actor_lr=args.head_lr if freeze else args.lr, critic_lr=args.lr
            )
            if frozen_epochs:
                log(
                    f"[distill epoch {epoch}] encoder "
                    f"{'FROZEN, actor lr ' + str(args.head_lr) if freeze else 'unfrozen, lr ' + str(args.lr)}"
                )
            t0 = time.time()
            train_stats, steps = train_distill.run_epoch(
                agent, train_eps, dargs, train=True
            )
            log(
                f"[distill epoch {epoch}] train ({steps} steps, "
                f"{(time.time() - t0) / 60:.1f} min): {train_distill.fmt_stats(train_stats)}"
            )
            log_row({"kind": "train", "epoch": epoch, **train_stats})
            improved = True
            if holdout:
                hold_stats, _ = train_distill.run_epoch(
                    agent, holdout, dargs, train=False
                )
                log(
                    f"[distill epoch {epoch}] holdout: {train_distill.fmt_stats(hold_stats)}"
                )
                log_row({"kind": "holdout", "epoch": epoch, **hold_stats})
                kl = float(hold_stats.get("override_kl", float("inf")))
                improved = kl < best_kl * (1.0 - args.kl_min_improve)
                if improved:
                    best_kl, best_epoch = kl, epoch
            if args.probe_games:
                probe = greedy_health_probe(agent, n_games=args.probe_games, seed=0)
                log(
                    f"[distill epoch {epoch}] probe: "
                    + "  ".join(f"{k} {v:.1f}" for k, v in sorted(probe.items()))
                )
                log_row({"kind": "probe", "epoch": epoch, **probe})
            ckpt_path = os.path.join(args.out_dir, f"distill_epoch{epoch}.pt")
            agent.save(ckpt_path)
            saved.append(ckpt_path)
            log(f"[distill epoch {epoch}] saved {ckpt_path}")
            if args.kl_stop and not improved:
                log(
                    f"[distill] KL stop rule: holdout target KL did not improve by "
                    f">= {100 * args.kl_min_improve:.0f}% over the best ({best_kl:.4f}, "
                    f"epoch {best_epoch}); stopping after epoch {epoch}"
                )
                break
        if holdout:
            with open(os.path.join(args.out_dir, "distill_best.json"), "w") as f:
                json.dump({"best_epoch": best_epoch, "holdout_override_kl": best_kl}, f)
            log(
                f"[distill] best epoch by holdout target KL: {best_epoch} ({best_kl:.4f})"
            )
    return saved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("fit", "target", "distill", "all"))
    ap.add_argument(
        "--corpus-dir", required=True, help="schema-2 corpus (or recovered)"
    )
    ap.add_argument("--ckpt", required=True, help="theta_k, the corpus's generator")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--buffer-episodes", type=int, default=250)
    ap.add_argument("--batch-segments", type=int, default=32)
    # Stage 1
    fit = ap.add_argument_group("fit")
    fit.add_argument("--capacity", default="all", choices=(*CAPACITIES, "all"))
    fit.add_argument("--fit-epochs", type=int, default=30)
    fit.add_argument("--fit-lr", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=1e-2)
    fit.add_argument("--batch-rows", type=int, default=1024)
    fit.add_argument("--patience", type=int, default=4)
    fit.add_argument(
        "--var-floor",
        type=float,
        default=1e-6,
        help="added to every row's noise variance in the 1/var weight (Q^2)",
    )
    fit.add_argument("--rebuild-table", action="store_true")
    fit.add_argument(
        "--bilinear",
        action="store_true",
        help="add a bilinear state x card term to the pointer (§20.8): scores a "
        "card attribute conditionally on the state",
    )
    fit.add_argument(
        "--heteroscedastic",
        action="store_true",
        help="fit a per-row log-variance head by Gaussian NLL (§20.8): rows are "
        "standardized by their own learned residual scale, no cell taxonomy",
    )
    fit.add_argument(
        "--fh-iterations",
        type=int,
        default=2,
        help="Fay-Herriot iterated-WLS rounds for the frozen rungs (§20.6 arm 3): "
        "round 1 weights 1/noise_var, later rounds 1/(noise_var + cell sigma_u^2)",
    )
    # Stage 2
    tgt = ap.add_argument_group("target")
    tgt.add_argument(
        "--kappa", type=float, default=1.0, help="tilt temperature in posterior SEs"
    )
    tgt.add_argument("--tilt-max", type=float, default=8.0, help="|z| clip (nats)")
    tgt.add_argument(
        "--sigma-u2", type=float, default=None, help="override the fitted sigma_u^2"
    )
    tgt.add_argument(
        "--variance-mode",
        choices=("class", "global", "node"),
        default="class",
        help="residual variance per telemetry cell (§20.6, default), one global "
        "value, or per node from the heteroscedastic head (§20.8)",
    )
    tgt.add_argument(
        "--weight-mode",
        choices=("none", "precision"),
        default="none",
        help="per-row CE weight for the projection: none (uniform) or the "
        "posterior precision 1/v_post, mean-normalized (§20.9)",
    )
    tgt.add_argument("--weight-max", type=float, default=5.0)
    tgt.add_argument(
        "--variance-rows",
        choices=("holdout", "all"),
        default="all",
        help="rows the per-class residual variance is estimated on (§20.6 arm 3: "
        "all rows with Q, so thin cells are not pinned to the global by shrinkage)",
    )
    tgt.add_argument(
        "--class-shrink-rows",
        type=float,
        default=50.0,
        help="row-count weight of the global value when shrinking per-class variances",
    )
    # Stage 3
    dst = ap.add_argument_group("distill")
    dst.add_argument("--targeted-dir", default=None, help="default <out-dir>/targeted")
    dst.add_argument("--epochs", type=int, default=1)
    dst.add_argument("--lr", type=float, default=1e-4)
    dst.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="head-first projection: epochs with the encoder frozen (§20.9)",
    )
    dst.add_argument(
        "--head-lr", type=float, default=1e-3, help="actor lr while frozen"
    )
    dst.add_argument(
        "--freeze-epochs",
        default="",
        help='explicit comma list of epochs with the encoder frozen (e.g. "2,3"); '
        "overrides --freeze-encoder-epochs",
    )
    dst.add_argument("--lambda-ce", type=float, default=1.0)
    dst.add_argument("--lambda-ret", type=float, default=1.0)
    dst.add_argument("--kd-tau", type=float, default=1.0)
    dst.add_argument("--no-oracle", dest="train_oracle", action="store_false")
    dst.add_argument("--probe-games", type=int, default=500)
    dst.add_argument(
        "--kl-stop",
        action="store_true",
        help="stop when the held-out KL(target||policy) stops improving (§20.6)",
    )
    dst.add_argument(
        "--kl-min-improve",
        type=float,
        default=0.02,
        help="relative improvement in held-out target KL an epoch must deliver",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage in ("fit", "all"):
        stage_fit(args)
    if args.stage in ("target", "all"):
        stage_target(args)
    if args.stage in ("distill", "all"):
        stage_distill(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
