#!/usr/bin/env python3
"""Can the AdvantageModel learn a cell it gets wrong, given more weight on
it? A cross-fit harness for fit-side weighting (Arm_D_Review §9.6, §9.7).

Mirrors a finished iteration's cross-fit exactly (its encoder's frozen
features, the centered noise model, 5 game folds, seed 0, fixed epochs,
adapter + heteroscedastic + bilinear) with one change to each fold's TRAINING
index; every row then has an out-of-fold prediction from a model that never
saw its game. Diagnostic only, not a recipe change. Modes:

    oversample  repeat the over-trump cell's training rows k times
                (k = 1 reproduces the stage's own cross-fit)
    reweight    RELATIVE residual-driven weights for every row,
                w = clip(sigma_hat^2(row) / sigma_u^2(class), 1, cap), with
                sigma_hat^2 from the second-stage variance regression
                (``blend_variance_probe``). Leakage-free: for outer fold i the
                regression and its isotonic calibration are fit on the
                training folds alone, so no weight used to train fold i's
                model saw fold i's residuals. Weights act by stochastic
                duplication; epochs are scaled so the arm takes the
                baseline's optimizer steps

Reads, per arm: global out-of-fold wMSE (and per class against the first
arm); the over-trump cell and two look-alike controls the weighting must not
distort; the t0/t1 convention cells (called-suit lead, defender trump lead).
Each arm's predictions are cached (``--cache-dir``) and reused if present.

Usage (from repo root):

    uv run python -m sheepshead.analysis.oversample_crossfit oversample --k 1 5 20 \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl \
        --controls runs/202609_recall_rc/overtrump_probe/rows_all_fail.pkl \
        --cache-dir runs/202609_recall_rc/overtrump_probe
    uv run python -m sheepshead.analysis.oversample_crossfit reweight --caps 4 10 \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl \
        --controls runs/202609_recall_rc/overtrump_probe/rows_all_fail.pkl \
        --cache-dir runs/202609_recall_rc/overtrump_probe
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from sheepshead.agent.ppo import load_agent
from sheepshead.analysis.blend_variance_probe import (
    DEFAULT_ITER,
    FOLDS,
    IS_TRUMP_PLAY,
    SEED,
    IterationTerms,
    called_suit_mask,
    excess_error,
    load_iteration,
    nested_variance,
    regression_features,
)
from sheepshead.analysis.called_suit_overtrump import load_rows
from sheepshead.rewards import RETURN_SCALE
from sheepshead.training.search_advantage import (
    AdvantageModel,
    evaluate_rows,
    fit_advantage_model,
    fold_rows_by_game,
    index_batches,
    predict_rows,
)

FIT = {
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "batch_rows": 1024,
}  # policy_iteration fit defaults
DEFAULT_ENCODER = "runs/202609_recall_rc/pi/iter2/bidding/final.pt"


class Harness:
    """One iteration's table, folds and encoder, and the fold-model fit."""

    def __init__(self, iter_dir: str, encoder_ckpt: str):
        self.terms: IterationTerms = load_iteration(iter_dir)
        self.agent = load_agent(encoder_ckpt)
        with open(os.path.join(iter_dir, "fit_report.json")) as f:
            fit = json.load(f)
        self.capacity = fit["selected"]
        self.epochs = int(fit["cross_fit"]["epochs"])
        self.fold = fold_rows_by_game(self.terms.table, FOLDS, SEED)

    def new_model(self) -> AdvantageModel:
        return AdvantageModel(
            self.agent, self.capacity, heteroscedastic=True, bilinear=True
        )

    def fit_fold(self, i: int, train_idx: torch.Tensor, epochs: int) -> AdvantageModel:
        table = self.terms.table
        pred = torch.nonzero(self.fold == i).squeeze(1)
        torch.manual_seed(SEED + 1000 + i)
        model = self.new_model()
        fit_advantage_model(
            model,
            table,
            train_idx,
            pred,
            epochs=epochs,
            var_floor=1e-6,
            seed=SEED + i,
            log=lambda *_: None,
            fixed_epochs=True,
            patience=epochs + 1,
            **FIT,
        )
        return model

    def predict_fold(self, i: int, model: AdvantageModel, a_oof: torch.Tensor) -> None:
        pred = torch.nonzero(self.fold == i).squeeze(1)
        model.eval()
        with torch.no_grad():
            for b in index_batches(pred, 2048, False, random.Random(0)):
                a_oof[b] = predict_rows(model, self.terms.table, b)[0]

    def oversample(self, cell_idx: torch.Tensor, k: int) -> torch.Tensor:
        a_oof = torch.zeros_like(self.terms.table.advantage)
        for i in range(FOLDS):
            train = torch.nonzero(self.fold != i).squeeze(1)
            extra = cell_idx[self.fold[cell_idx] != i].repeat(k - 1)
            model = self.fit_fold(i, torch.cat([train, extra]), self.epochs)
            self.predict_fold(i, model, a_oof)
        return a_oof

    def reweight(self, cap: float) -> torch.Tensor:
        table = self.terms.table
        x = regression_features(table)
        y = excess_error(self.terms)
        games = table.game.numpy()
        su2_cls = self.terms.su2_cls.numpy()
        fold = self.fold.numpy()
        a_oof = torch.zeros_like(table.advantage)
        for i in range(FOLDS):
            t0 = time.time()
            idx, s2 = nested_variance(x, y, games, fold != i)
            w = np.clip(s2 / su2_cls[idx], 1.0, cap)
            rng = np.random.default_rng(SEED + i)
            reps = np.floor(w).astype(int) + (rng.random(len(w)) < (w - np.floor(w)))
            train = torch.tensor(np.repeat(idx, reps), dtype=torch.long)
            epochs = max(1, round(self.epochs * len(idx) / len(train)))
            model = self.fit_fold(i, train, epochs)
            self.predict_fold(i, model, a_oof)
            print(
                f"  [cap {cap:g} fold {i + 1}/{FOLDS}] rows weighted >1 {100 * float((w > 1).mean()):.1f}%  "
                f"mean w {float(w.mean()):.2f}  table x{len(train) / len(idx):.2f}  epochs {epochs}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        return a_oof


def _overtrump_verdict(
    rows: list[dict], a: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """(model, search): policy-preferred over-trump minus policy-preferred
    other play, score."""
    m, s = [], []
    for r in rows:
        o, p = r["over"], r["prior"]
        bo, bn = int(np.argmax(np.where(o, p, -1))), int(np.argmax(np.where(~o, p, -1)))
        am = a[r["i"]][[v - 1 for v in r["valid"]]].numpy()
        m.append(RETURN_SCALE * (am[bo] - am[bn]))
        s.append(RETURN_SCALE * (r["q"][bo] - r["q"][bn]))
    return np.array(m), np.array(s)


def _class_verdict(
    h: Harness, rows: torch.Tensor, good: torch.Tensor, a: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """(model, search): policy-preferred `good` card minus policy-preferred
    other card, score."""
    t = h.terms.table
    legal = t.masks.bool()[rows]
    p = t.prior[rows]
    g = good[rows] & legal
    o = ~good[rows] & legal
    bg = torch.where(g, p, torch.full_like(p, -1)).argmax(1, keepdim=True)
    bo = torch.where(o, p, torch.full_like(p, -1)).argmax(1, keepdim=True)
    m = RETURN_SCALE * (a[rows].gather(1, bg) - a[rows].gather(1, bo)).squeeze(1)
    s = RETURN_SCALE * (
        t.advantage[rows].gather(1, bg) - t.advantage[rows].gather(1, bo)
    ).squeeze(1)
    return m.numpy(), s.numpy()


def run(args) -> None:
    h = Harness(args.iter_dir, args.encoder_ckpt)
    table = h.terms.table
    iteration = os.path.basename(os.path.normpath(args.iter_dir))
    ref_idx = {tuple(r): i for i, r in enumerate(table.refs)}

    def attach(rows: list[dict]) -> list[dict]:
        rows = [r for r in rows if r["key"] in ref_idx and r["q"] is not None]
        for r in rows:
            r["i"] = ref_idx[r["key"]]
        return rows

    cell = attach(load_rows(args.rows, iteration))
    controls = attach(
        [r for r in load_rows(args.controls, iteration) if not r["led_called"]]
    )
    cell_idx = torch.tensor(sorted({r["i"] for r in cell}))
    lead_classes = ("std|t0-defender-lead", "std|t1-defender-lead")
    legal = table.masks.bool()
    cs_mask = called_suit_mask(h.terms, lead_classes)
    trump_mask = legal & IS_TRUMP_PLAY.unsqueeze(0)
    conventions = {}
    for cls in lead_classes:
        rows = torch.tensor([r for r, c in enumerate(table.node_class) if c == cls])
        conventions[f"{cls[4:6]} called-suit lead"] = (
            rows[cs_mask[rows].any(1)],
            cs_mask,
        )
        both = trump_mask[rows].any(1) & (
            legal[rows] & ~IS_TRUMP_PLAY.unsqueeze(0)
        ).any(1)
        conventions[f"{cls[4:6]} defender trump lead"] = (rows[both], trump_mask)

    arms = (
        [(f"k={k}", f"oversample_k{k}.pt", k) for k in args.k]
        if args.cmd == "oversample"
        else [(f"relative cap {c:g}", f"reweight_cap{c:g}.pt", c) for c in args.caps]
    )
    if args.cmd == "reweight" and args.baseline:
        arms.insert(0, ("baseline (k=1)", "oversample_k1.pt", 1))
    per_class = {}
    for name, fname, param in arms:
        path = os.path.join(args.cache_dir, fname)
        if os.path.exists(path):
            a = torch.load(path, weights_only=False)["a_oof"]
        else:
            print(f"\nfitting {name}", flush=True)
            t0 = time.time()
            if fname.startswith("oversample"):
                a = h.oversample(cell_idx, int(param))
            else:
                a = h.reweight(float(param))
            torch.save({"a_oof": a}, path)
            print(f"  ({(time.time() - t0) / 60:.1f} min)")
        ev = evaluate_rows(
            h.new_model(),
            table,
            torch.arange(len(table)),
            var_floor=1e-6,
            oof=(a, None),
        )
        pc = ev.per_class
        per_class[name] = pc
        print(
            f"\n== {name}: out-of-fold wMSE all {pc['__all__']['weighted_mse']:.4e}  top-agree model {pc['__all__']['top_agree_model']:.3f}"
        )
        for label, rows in (
            ("over-trump cell", cell),
            ("  ...policy over-trumps", [r for r in cell if r["arg_over"]]),
            ("  ...holds a schmear", [r for r in cell if r["max_fail_pts"] >= 10]),
            (
                "control: other suit, picker behind",
                [r for r in controls if r["picker_after"]],
            ),
            (
                "control: other suit, picker played",
                [r for r in controls if not r["picker_after"]],
            ),
        ):
            m, s = _overtrump_verdict(rows, a)
            print(
                f"   {label:36s} n {len(rows):4d}  model {m.mean():+.3f}  search {s.mean():+.3f}  "
                f"sign agree {100 * np.mean(np.sign(m) == np.sign(s)):3.0f}%  corr {np.corrcoef(m, s)[0, 1]:.2f}"
            )
        for label, (rows, mask) in conventions.items():
            m, s = _class_verdict(h, rows, mask, a)
            print(
                f"   {label:36s} n {len(rows):4d}  model {m.mean():+.3f}  search {s.mean():+.3f}  corr {np.corrcoef(m, s)[0, 1]:.2f}"
            )

    names = list(per_class)
    if len(names) > 1:
        first = per_class[names[0]]
        print(f"\nper-class out-of-fold wMSE vs {names[0]} (classes with n >= 1000)")
        print(
            f"{'class':28s} {'n':>6s} {names[0]:>14s} "
            + " ".join(f"{n:>16s}" for n in names[1:])
        )
        for c in sorted(first):
            if c == "__all__" or first[c]["n"] < 1000:
                continue
            b = first[c]["weighted_mse"]
            print(
                f"{c:28s} {first[c]['n']:6d} {b:14.3e} "
                + " ".join(
                    f"{100 * (per_class[n][c]['weighted_mse'] / b - 1):+15.1f}%"
                    for n in names[1:]
                )
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("oversample", "reweight"):
        p = sub.add_parser(name)
        p.add_argument("--iter-dir", default=DEFAULT_ITER)
        p.add_argument(
            "--encoder-ckpt",
            default=DEFAULT_ENCODER,
            help="the checkpoint --iter-dir's row table was encoded with",
        )
        p.add_argument(
            "--rows", required=True, help="called_suit_overtrump rows pickle (the cell)"
        )
        p.add_argument(
            "--controls",
            required=True,
            help="called_suit_overtrump --all-fail rows pickle",
        )
        p.add_argument(
            "--cache-dir",
            required=True,
            help="per-arm out-of-fold predictions, reused if present",
        )
        if name == "oversample":
            p.add_argument("--k", nargs="+", type=int, default=[1, 5, 20])
        else:
            p.add_argument("--caps", nargs="+", type=float, default=[4.0, 10.0])
            p.add_argument(
                "--no-baseline",
                dest="baseline",
                action="store_false",
                help="skip the k=1 baseline arm",
            )
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
