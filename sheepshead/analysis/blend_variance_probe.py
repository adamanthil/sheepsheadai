#!/usr/bin/env python3
"""The Fay-Herriot blend's model-error variance, per class vs where the model
is actually wrong (Arm_D_Review §9.4, §9.7).

The target stage blends each row's search advantage with the AdvantageModel's
out-of-fold prediction, gamma = sigma_u^2 / (sigma_u^2 + noise), with
sigma_u^2 estimated per node class. A cell where the model is biased (the
called-suit over-trump) has a much larger error than its class, so the class
value hands the model too much of the blend there. Reads, all on one finished
iteration (row table, OOF predictions, target report), no refit:

    fh-terms     E[r^2], E[noise], sigma_u^2 per defender-follow class vs the
                 over-trump cell, and the gamma each implies (§9.4)
    variants     the production class variance vs three ONE-SIDED per-row
                 alternatives, sigma_u^2(row) = max(estimate, class value):
                 the heteroscedastic head; a cross-fitted second-stage
                 variance regression (gradient-boosted trees on the frozen
                 state features, no class or game-specific input); and that
                 regression after cross-fitted isotonic recalibration. Per
                 variant: the cell's gamma and over-trump target mass, how
                 much of the table moves, and the §8 model-push read (§9.7)
    conventions  whether the variants weaken the t0/t1 convention targets
                 (called-suit lead mass, defender trump-lead mass) and how
                 they move picker rows

The saved ``row_table.pt`` holds the RAW corpus noise; everything here applies
the centered noise model first, as the target stage does (§8.1 / §8.8).

Usage (from repo root):

    uv run python -m sheepshead.analysis.blend_variance_probe fh-terms \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl
    uv run python -m sheepshead.analysis.blend_variance_probe variants \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl \
        --regression-cache runs/202609_recall_rc/overtrump_probe/variance_regression_oof.pt \
        --json runs/202609_recall_rc/overtrump_probe/variance_regression_diag.json
    uv run python -m sheepshead.analysis.blend_variance_probe conventions \
        --regression-cache runs/202609_recall_rc/overtrump_probe/variance_regression_oof.pt
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from sheepshead.game import ACTIONS, TRUMP_POWER
from sheepshead.rewards import RETURN_SCALE
from sheepshead.training.corpus_rows import load_shard, shard_paths
from sheepshead.training.search_advantage import (
    SIGMA_U2_FLOOR,
    RowTable,
    apply_noise_model,
    blend_advantages,
    build_tilt_target,
    fold_rows_by_game,
    model_push_diagnostics,
    sigma_u2_rows,
    tilt_scale,
)

DEFAULT_ITER = "runs/202609_recall_rc/pi/iter3"
FOLDS, SEED = 5, 0  # the target stage's cross-fit folds
NOISE_SHRINK_NU = 4.0  # policy_iteration --noise-shrink-nu default


@dataclass
class IterationTerms:
    """One finished iteration's blend inputs, as its target stage saw them."""

    iter_dir: str
    table: RowTable  # working (centered-model) noise applied
    a_model: torch.Tensor  # (R, A) out-of-fold
    su2_head: torch.Tensor  # (R,) out-of-fold heteroscedastic head
    su2_cls: torch.Tensor  # (R,) the class value the blend used
    eta: float
    tilt_max: float
    kappa: float


def load_iteration(iter_dir: str) -> IterationTerms:
    table = RowTable.load(os.path.join(iter_dir, "row_table.pt"))
    apply_noise_model(table, "centered", NOISE_SHRINK_NU, pool="class")
    oof = torch.load(
        os.path.join(iter_dir, "oof_predictions.pt"),
        weights_only=False,
        map_location="cpu",
    )
    with open(os.path.join(iter_dir, "target_report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(iter_dir, "fit_report.json")) as f:
        fit = json.load(f)
    su2_cls = sigma_u2_rows(
        table.node_class, rep["sigma_u2_by_class"], float(fit["cross_fit"]["sigma_u2"])
    )
    return IterationTerms(
        iter_dir=iter_dir,
        table=table,
        a_model=oof["a_model"],
        su2_head=oof["sigma_u2_head"].float(),
        su2_cls=su2_cls,
        eta=float(rep["eta"]),
        tilt_max=float(rep["tilt_max"]),
        kappa=float(rep["kappa"]),
    )


def build_targets(
    terms: IterationTerms, su2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(target, gamma, v_post) with per-row model-error variance ``su2``."""
    t = terms.table
    a_hat, v_post, gamma = blend_advantages(
        t.advantage, terms.a_model, t.has_q, t.noise_var, su2
    )
    target, _ = build_tilt_target(
        t.prior,
        a_hat,
        v_post,
        t.masks.bool(),
        kappa=terms.kappa,
        tilt_max=terms.tilt_max,
        eta=terms.eta,
    )
    return target, gamma, v_post


def stored_gamma(terms: IterationTerms) -> torch.Tensor:
    """The production gamma written onto each row's shard event."""
    g = torch.zeros(len(terms.table))
    by_shard: dict[int, list[int]] = collections.defaultdict(list)
    for r, (s, _, _) in enumerate(terms.table.refs):
        by_shard[s].append(r)
    paths = shard_paths(os.path.join(terms.iter_dir, "targeted"))
    for s, rr in by_shard.items():
        episodes = load_shard(paths[s])["episodes"]
        for r in rr:
            _, e, k = terms.table.refs[r]
            g[r] = float(episodes[e][k]["pi_gamma"])
    return g


# --------------------------------------------------------------------------- #
# Second-stage variance regression
# --------------------------------------------------------------------------- #
def excess_error(terms: IterationTerms) -> np.ndarray:
    """Per row, mean squared OOF residual over the labeled cards minus the
    working noise: unbiased for sigma_u^2 at the row (E[r^2] = sigma_u^2 +
    noise), very noisy per row."""
    t = terms.table
    lab = t.label_mask
    r2 = ((t.advantage - terms.a_model) ** 2 * lab).sum(1) / lab.sum(1).clamp(min=1)
    return (r2 - t.noise_var).numpy().astype(np.float64)


def regression_features(table: RowTable) -> np.ndarray:
    """Frozen state features plus two generic row statistics; no class, no
    game-specific input."""
    legal = table.masks.bool()
    prior_max = (
        torch.where(legal, table.prior, torch.zeros_like(table.prior)).max(1).values
    )
    return np.concatenate(
        [
            table.features.numpy(),
            legal.sum(1, keepdim=True).float().numpy(),
            prior_max.unsqueeze(1).numpy(),
        ],
        axis=1,
    )


def variance_trees(seed: int) -> HistGradientBoostingRegressor:
    """The regressor: the 400-row minimum leaf is the shrinkage (no region
    gets its own variance unless 400 rows back it)."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=400,
        l2_regularization=1.0,
        # "auto" = on above 10,000 rows: always the case for these tables
        # (the stubs type the parameter as a string).
        early_stopping="auto",
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def crossfit_regression(
    x: np.ndarray, y: np.ndarray, fold: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Out-of-fold predictions: each fold's from trees fit on the others."""
    pred = np.zeros(len(y))
    for i in np.unique(fold):
        t0 = time.time()
        tr = fold != i
        m = variance_trees(seed + int(i)).fit(x[tr], y[tr])
        pred[~tr] = m.predict(x[~tr])
        print(
            f"  [variance fold {int(i) + 1}] {m.n_iter_} trees, {time.time() - t0:.0f}s",
            flush=True,
        )
    return np.maximum(pred, SIGMA_U2_FLOOR)


def crossfit_isotonic(pred: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Recalibrate ``pred`` by a monotone fit of y on the OTHER folds' rows,
    so no row's own label sets its calibration."""
    out = np.zeros(len(y))
    for i in np.unique(fold):
        tr = fold != i
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            pred[tr], y[tr]
        )
        out[~tr] = ir.predict(pred[~tr])
    return np.maximum(out, SIGMA_U2_FLOOR)


def nested_variance(
    x: np.ndarray,
    y: np.ndarray,
    games: np.ndarray,
    train: np.ndarray,
    inner_folds: int = 4,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic-calibrated variance estimates for the rows in ``train``,
    computed from ``train`` alone (inner game folds). Used for fit weights,
    where an outer fold's weights must not see that fold's residuals.
    Returns (row indices, estimates)."""
    idx = np.nonzero(train)[0]
    ug = np.unique(games[idx])
    np.random.default_rng(seed).shuffle(ug)
    inner_of = {g: i % inner_folds for i, g in enumerate(ug.tolist())}
    inner = np.array([inner_of[g] for g in games[idx].tolist()])
    pred = np.zeros(len(idx))
    for j in range(inner_folds):
        m = variance_trees(seed + 10 * j).fit(x[idx[inner != j]], y[idx[inner != j]])
        pred[inner == j] = m.predict(x[idx[inner == j]])
    return idx, crossfit_isotonic(pred, y[idx], inner)


def regression_estimates(
    terms: IterationTerms, cache: str | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(raw, isotonic) one-per-row variance estimates, cross-fitted over the
    stage's game folds. ``cache`` holds the raw predictions (the slow part)."""
    y = excess_error(terms)
    fold = fold_rows_by_game(terms.table, FOLDS, SEED).numpy()
    if cache and os.path.exists(cache):
        raw = torch.load(cache).numpy().astype(np.float64)
    else:
        raw = crossfit_regression(regression_features(terms.table), y, fold)
        if cache:
            torch.save(torch.tensor(raw, dtype=torch.float32), cache)
    iso = crossfit_isotonic(raw, y, fold)
    for name, est in (("raw", raw), ("isotonic", iso)):
        q = np.quantile(est, np.linspace(0, 1, 11))
        print(
            f"\nregression ({name}, out-of-fold): predicted decile mean vs measured E[r^2] - noise"
        )
        for lo, hi in zip(q[:-1], q[1:]):
            s = (est >= lo) & (est <= hi)
            if s.any():
                print(
                    f"   predicted {est[s].mean():.2e}   measured {y[s].mean():.2e}   n {s.sum()}"
                )
    return torch.tensor(raw, dtype=torch.float32), torch.tensor(
        iso, dtype=torch.float32
    )


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def overtrump_cell(
    table: RowTable, rows_path: str, iteration: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row indices of the over-trump decisions (``called_suit_overtrump``
    rows of ``iteration``) and a dense (n, A) over-trump mask."""
    ref_idx = {tuple(r): i for i, r in enumerate(table.refs)}
    with open(rows_path, "rb") as f:
        rows = [
            r for r in pickle.load(f) if r["it"] == iteration and r["key"] in ref_idx
        ]
    idx = torch.tensor([ref_idx[r["key"]] for r in rows])
    over = torch.zeros(len(rows), table.masks.size(1), dtype=torch.bool)
    for j, r in enumerate(rows):
        for a, o in zip(r["valid"], r["over"]):
            over[j, a - 1] = bool(o)
    return idx, over


def group_of(node_class: str) -> str:
    """§8's reporting groups: defender leads kept by trick."""
    mode, rest = node_class.split("|", 1)
    t, role, kind = rest.split("-")
    return f"{mode} {role} {kind}" + (
        f" {t}" if (role, kind) == ("defender", "lead") else ""
    )


IS_TRUMP_PLAY = torch.tensor(
    [a.startswith("PLAY ") and a[5:] in TRUMP_POWER for a in ACTIONS]
)


def called_suit_mask(terms: IterationTerms, classes: tuple[str, ...]) -> torch.Tensor:
    """(R, A) legal called-suit leads (the corpus's ``conv_cs_ids``) on rows
    of ``classes``; False elsewhere."""
    mask = torch.zeros_like(terms.table.masks, dtype=torch.bool)
    by_shard: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for r, (s, e, v) in enumerate(terms.table.refs):
        if terms.table.node_class[r] in classes:
            by_shard[s].append((r, e, v))
    paths = sorted(glob.glob(os.path.join(terms.iter_dir, "targeted", "corpus_*.pt")))
    for s, items in by_shard.items():
        d = torch.load(paths[s], weights_only=False, map_location="cpu")
        for r, e, v in items:
            for a in d["episodes"][e][v].get("conv_cs_ids") or []:
                mask[r, a - 1] = True
        del d
    return mask


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def cmd_fh_terms(terms: IterationTerms, rows_path: str, iteration: str) -> None:
    t = terms.table
    lab = t.label_mask
    r2 = ((t.advantage - terms.a_model) ** 2 * lab).sum(1) / lab.sum(1).clamp(min=1)
    nv = t.noise_var
    cell, over = overtrump_cell(t, rows_path, iteration)
    prefers_over = cell[(t.prior[cell] * over).sum(1) > 0.5]
    by: dict[str, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(t.node_class):
        if t.has_q[i] and c.startswith("std|") and "defender-follow" in c:
            by[c].append(i)
    print("units: Q^2 (score = 12 Q)")
    print(
        f"{'set':30s} {'n':>6s} {'E[r^2]':>9s} {'E[noise]':>9s} {'sigma_u2':>9s} {'class s_u2':>10s} {'gamma':>6s} {'gamma if s_u2=set':>17s}"
    )

    def line(label: str, ix: torch.Tensor) -> None:
        loc = float(r2[ix].mean() - nv[ix].mean())
        g = (terms.su2_cls[ix] / (terms.su2_cls[ix] + nv[ix])).mean()
        g_loc = (loc / (loc + nv[ix])).mean() if loc > 0 else float("nan")
        print(
            f"{label:30s} {len(ix):6d} {float(r2[ix].mean()):9.2e} {float(nv[ix].mean()):9.2e} {loc:9.2e} "
            f"{float(terms.su2_cls[ix].mean()):10.2e} {float(g):6.2f} {float(g_loc):17.2f}"
        )

    for c in sorted(by):
        line(c, torch.tensor(by[c]))
    line("over-trump cell", cell)
    line("  ...policy prefers over-trump", prefers_over)


def cmd_variants(
    terms: IterationTerms,
    rows_path: str,
    iteration: str,
    cache: str | None,
    out_json: str | None,
) -> None:
    t = terms.table
    raw, iso = regression_estimates(terms, cache)
    variants = {
        "class (production)": terms.su2_cls,
        "one-sided head": torch.maximum(terms.su2_head, terms.su2_cls),
        "one-sided regression": torch.maximum(raw, terms.su2_cls),
        "one-sided isotonic": torch.maximum(iso, terms.su2_cls),
    }
    built = {k: build_targets(terms, v) for k, v in variants.items()}
    t0_, g0_, _ = built["class (production)"]
    print(
        f"\nproduction gamma vs stored pi_gamma: max abs diff {float((g0_ - stored_gamma(terms)).abs().max()):.2e}"
    )

    y = excess_error(terms)
    cell, over = overtrump_cell(t, rows_path, iteration)
    n_cell = len(cell)
    print(
        f"\nover-trump cell (n {n_cell}): measured E[r^2] - noise {y[cell.numpy()].mean():.2e};  class "
        f"{float(terms.su2_cls[cell].mean()):.2e}  head {float(terms.su2_head[cell].mean()):.2e}  "
        f"regression {float(raw[cell].mean()):.2e}  isotonic {float(iso[cell].mean()):.2e};  "
        f"prior over-trump mass {float(t.prior[cell][over].sum()) / n_cell:.3f}"
    )
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for r, c in enumerate(t.node_class):
        groups[group_of(c)].append(r)

    diags = {}
    for name, (tgt, g, v) in built.items():
        kl = (
            tgt.clamp(min=1e-12)
            * (tgt.clamp(min=1e-12).log() - t0_.clamp(min=1e-12).log())
        ).sum(1)
        dg = g - g0_
        print(f"\n== {name}")
        print(
            f"   cell: gamma {float(g[cell].mean()):.2f}  over-trump target mass {float(tgt[cell][over].sum()) / n_cell:.3f}"
        )
        print(
            f"   table: gamma mean {float(g.mean()):.3f}; rows gamma +0.05 {100 * float((dg >= 0.05).float().mean()):.1f}%, "
            f"+0.2 {100 * float((dg >= 0.2).float().mean()):.1f}%; KL vs production p90 {float(kl.quantile(0.9)):.1e} "
            f"p99 {float(kl.quantile(0.99)):.1e}; top card changed {100 * float((tgt.argmax(1) != t0_.argmax(1)).float().mean()):.2f}%"
        )
        print(
            "   gamma by group: "
            + "  ".join(
                f"{k} {float(g[torch.tensor(ix)].mean()):.3f}"
                for k, ix in sorted(groups.items())
                if len(ix) >= 300
            )
        )
        d = model_push_diagnostics(
            t,
            terms.a_model,
            g,
            tilt_scale(v, kappa=terms.kappa, eta=terms.eta),
            t.noise_var,
            tilt_max=terms.tilt_max,
            out_of_fold=True,
        )
        diags[name] = d
        p = d["push"]

        def fz(x: dict) -> str:
            return f"{RETURN_SCALE * x['mean']:+.4f} (z {x['z']:+.1f})"

        print(
            f"   §8 all: model-only {fz(p['model_gain'])}  claims {RETURN_SCALE * p['model_claim']:+.4f}  "
            f"marginal {fz(p['marginal'])}  target {fz(p['full'])}"
        )
    if out_json:
        with open(out_json, "w") as f:
            json.dump(diags, f, indent=2)

    print("\n§8 target value by group (game points per decision)")
    print(f"{'group':26s} {'n':>6s} " + " ".join(f"{k:>22s}" for k in diags))
    agg: dict[str, dict[str, list[float]]] = {
        k: collections.defaultdict(lambda: [0.0, 0.0]) for k in diags
    }
    for k, d in diags.items():
        for c, v in d["per_class"].items():
            p = v["push"]
            n = p.get("n", 0)
            if n and p["full"].get("mean") is not None:
                agg[k][group_of(c)][0] += n * p["full"]["mean"]
                agg[k][group_of(c)][1] += n
    first = next(iter(diags))
    for gname in sorted(agg[first]):
        n = agg[first][gname][1]
        if n >= 300:
            print(
                f"{gname:26s} {int(n):6d} "
                + " ".join(
                    f"{RETURN_SCALE * agg[k][gname][0] / agg[k][gname][1]:+22.4f}"
                    for k in diags
                )
            )


def cmd_conventions(terms: IterationTerms, cache: str | None) -> None:
    t = terms.table
    legal = t.masks.bool()
    raw, iso = regression_estimates(terms, cache)
    del raw
    variants = {
        "class": terms.su2_cls,
        "head": torch.maximum(terms.su2_head, terms.su2_cls),
        "isotonic": torch.maximum(iso, terms.su2_cls),
    }
    built = {k: build_targets(terms, v)[:2] for k, v in variants.items()}
    lead_classes = ("std|t0-defender-lead", "std|t1-defender-lead")
    cs_mask = called_suit_mask(terms, lead_classes)
    trump_mask = legal & IS_TRUMP_PLAY.unsqueeze(0)
    fail_mask = legal & ~IS_TRUMP_PLAY.unsqueeze(0)
    for cls in lead_classes:
        rows = torch.tensor([r for r, c in enumerate(t.node_class) if c == cls])
        cs_rows = rows[cs_mask[rows].any(1)]
        tf_rows = rows[trump_mask[rows].any(1) & fail_mask[rows].any(1)]
        print(
            f"\n{cls}: called-suit rows {len(cs_rows)}, trump-or-fail rows {len(tf_rows)}"
        )
        print(
            f"   called-suit mass: prior {float((t.prior[cs_rows] * cs_mask[cs_rows]).sum(1).mean()):.3f}  "
            + "  ".join(
                f"{k} {float((tg[cs_rows] * cs_mask[cs_rows]).sum(1).mean()):.3f}"
                for k, (tg, _) in built.items()
            )
            + "   | argmax called: "
            + "  ".join(
                f"{k} {100 * float(cs_mask[cs_rows].gather(1, tg[cs_rows].argmax(1, keepdim=True)).float().mean()):.1f}%"
                for k, (tg, _) in built.items()
            )
        )
        print(
            f"   defender trump-lead mass: prior {float((t.prior[tf_rows] * trump_mask[tf_rows]).sum(1).mean()):.4f}  "
            + "  ".join(
                f"{k} {float((tg[tf_rows] * trump_mask[tf_rows]).sum(1).mean()):.4f}"
                for k, (tg, _) in built.items()
            )
        )
        print(
            "   gamma: "
            + "  ".join(
                f"{k} {float(g[rows].mean()):.3f}" for k, (_, g) in built.items()
            )
        )
    print("\ngamma and target KL vs class, picker groups")
    base = built["class"][0]
    for grp in ("picker-follow", "picker-lead"):
        for mode in ("std", "alone"):
            rows = torch.tensor(
                [
                    r
                    for r, c in enumerate(t.node_class)
                    if c.startswith(mode + "|") and c.endswith(grp)
                ]
            )
            parts = []
            for k, (tg, g) in built.items():
                kl = (
                    tg[rows].clamp(min=1e-12)
                    * (
                        tg[rows].clamp(min=1e-12).log()
                        - base[rows].clamp(min=1e-12).log()
                    )
                ).sum(1)
                parts.append(
                    f"{k} γ {float(g[rows].mean()):.3f} (KL {float(kl.mean()):.4f})"
                )
            print(f"   {mode} {grp:14s} n {len(rows):6d}  " + "  ".join(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fh-terms", "variants", "conventions"):
        p = sub.add_parser(name)
        p.add_argument("--iter-dir", default=DEFAULT_ITER)
        if name != "conventions":
            p.add_argument(
                "--rows", required=True, help="called_suit_overtrump rows pickle"
            )
            p.add_argument(
                "--iteration",
                default=None,
                help="rows' iteration tag (default: basename of --iter-dir)",
            )
        if name != "fh-terms":
            p.add_argument(
                "--regression-cache",
                default=None,
                help="raw regression predictions (.pt), reused if present",
            )
        if name == "variants":
            p.add_argument(
                "--json", default=None, help="write the §8 diagnostics per variant here"
            )
    args = ap.parse_args(argv)
    terms = load_iteration(args.iter_dir)
    iteration = getattr(args, "iteration", None) or os.path.basename(
        os.path.normpath(args.iter_dir)
    )
    if args.cmd == "fh-terms":
        cmd_fh_terms(terms, args.rows, iteration)
    elif args.cmd == "variants":
        cmd_variants(terms, args.rows, iteration, args.regression_cache, args.json)
    else:
        cmd_conventions(terms, args.regression_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
