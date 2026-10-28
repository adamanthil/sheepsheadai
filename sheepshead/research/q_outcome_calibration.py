#!/usr/bin/env python3
"""Absolute search Q against realized outcomes, and the win probability an
over-trump costs (Arm_D_Review §9.3).

Search Q is trusted for DIFFERENCES between cards (that is all the distill
targets use), but its absolute level is optimistic: rollouts see perfect-
information worlds and search-improved continuations. This module measures
the bias per trick (Q of the action taken vs the final score) on the called-
ace defender follows of one corpus, calibrates Q -> P(defenders win) by
binning, and reads the over-trump decisions (``called_suit_overtrump``) as a
change in win probability.

Scoring is double on the bump: a defender win pays +2/+4/+6 and a loss costs
-1/-2/-3, so Q = 0 is about a one-in-three win chance, not one in two.

Usage (from repo root):

    uv run python -m sheepshead.research.q_outcome_calibration extract \
        --iter-dir runs/202609_recall_rc/pi/iter3 \
        --out runs/202609_recall_rc/overtrump_probe/calib.pkl
    uv run python -m sheepshead.research.q_outcome_calibration report \
        --calib runs/202609_recall_rc/overtrump_probe/calib.pkl \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections.abc import Callable

import numpy as np
import torch

from sheepshead.research.called_suit_overtrump import (
    POINT_TRUMPS,
    load_rows,
)
from sheepshead.rewards import RETURN_SCALE

CALIB_BINS = 40  # equal-count bins of Q for the P(win) curve


def extract(iter_dir: str, max_shards: int | None = None) -> np.ndarray:
    """(Q of the action taken, max Q, realized score, trick) per searched
    called-ace defender follow; Q in score units."""
    out = []
    paths = sorted(glob.glob(os.path.join(iter_dir, "targeted", "corpus_*.pt")))
    for path in paths[:max_shards]:
        d = torch.load(path, weights_only=False, map_location="cpu")
        for ep in d["episodes"]:
            for ev in ep:
                nc = str(ev.get("node_class"))
                if (
                    not nc.startswith("std|")
                    or "defender-follow" not in nc
                    or ev.get("search_q") is None
                ):
                    continue
                if int(ev["state"]["partner_mode"]) != 1:
                    continue
                valid = sorted(ev["valid_actions"])
                q = np.asarray(ev["search_q"])
                out.append(
                    (
                        RETURN_SCALE * float(q[valid.index(ev["action"])]),
                        RETURN_SCALE * float(q.max()),
                        float(ev["final_return_label"]),
                        int(ev["state"]["current_trick"]),
                    )
                )
        del d
    return np.array(out)


def win_curve(calib: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Monotone binned P(defenders win | Q) from the calibration set."""
    qt, win = calib[:, 0], (calib[:, 2] > 0).astype(float)
    edges = np.quantile(qt, np.linspace(0, 1, CALIB_BINS + 1))
    mids, rates = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (qt >= a) & (qt <= b)
        mids.append(qt[sel].mean())
        rates.append(win[sel].mean())
    mid = np.array(mids)
    pw = np.maximum.accumulate(np.array(rates))

    def pwin(q: np.ndarray) -> np.ndarray:
        return np.interp(q, mid, pw)

    return pwin


def report(calib: np.ndarray, rows_path: str | None) -> None:
    qt, ret, trick = calib[:, 0], calib[:, 2], calib[:, 3]
    print(
        f"calibration set: {len(calib)} rows; mean Q(taken) {qt.mean():+.3f}  mean realized {ret.mean():+.3f}  win {np.mean(ret > 0):.3f}"
    )
    for t in sorted(set(trick.astype(int).tolist())):
        m = trick == t
        print(
            f"  trick {t}: n {m.sum():5d}  Q {qt[m].mean():+.3f}  realized {ret[m].mean():+.3f}  bias {qt[m].mean() - ret[m].mean():+.3f}"
        )
    pwin = win_curve(calib)
    print(
        "Q (score) -> P(defenders win):  "
        + "  ".join(
            f"{q:+.1f}:{float(pwin(np.array(q))):.2f}"
            for q in (-2, -1, -0.5, 0, 0.5, 1, 1.5, 2)
        )
    )
    if rows_path is None:
        return

    rows = [r for r in load_rows(rows_path) if r["q"] is not None]
    for r in rows:
        o, p, q = r["over"], r["prior"], RETURN_SCALE * r["q"]
        bo = int(np.argmax(np.where(o, p, -1)))
        bn = int(np.argmax(np.where(~o, p, -1)))
        r["q_over"], r["q_alt"] = q[bo], q[bn]
        r["pw_over"], r["pw_alt"] = float(pwin(q[bo])), float(pwin(q[bn]))

    def summ(label: str, x: list[dict]) -> None:
        if len(x) < 8:
            return
        qa = np.array([r["q_alt"] for r in x])
        qo = np.array([r["q_over"] for r in x])
        pa = np.array([r["pw_alt"] for r in x])
        po = np.array([r["pw_over"] for r in x])
        print(
            f"  {label:34s} n {len(x):3d}  Q alt {qa.mean():+.2f}  Q over {qo.mean():+.2f}  "
            f"P(win) alt {pa.mean():.2f} -> over {po.mean():.2f} (Δ {np.mean(po - pa):+.3f})  "
            f"alt>0: {100 * np.mean(qa > 0):.0f}%  win->loss {100 * np.mean((qa > 0) & (qo <= 0)):.0f}% / "
            f"loss->win {100 * np.mean((qa <= 0) & (qo > 0)):.0f}%"
        )

    chosen = [r for r in rows if r["arg_over"]]
    print("\nover-trump decisions")
    summ("all", rows)
    summ("policy over-trumps", chosen)
    summ(
        "  schmear in hand (not AD/10D)",
        [
            r
            for r in chosen
            if r["max_fail_pts"] >= 10 and r["over_card"] not in POINT_TRUMPS
        ],
    )
    summ(
        "  10D winning, Q/J over, has fail",
        [
            r
            for r in chosen
            if r["winner"] == "10D"
            and r["over_card"][0] in "QJ"
            and r["max_fail_pts"] >= 0
        ],
    )
    print("by where the defenders stand (Q of the alternative), policy over-trumps")
    for lo, hi, label in (
        (-9, -1, "clearly losing (<-1)"),
        (-1, 0, "leaning losing"),
        (0, 1, "leaning winning"),
        (1, 9, "clearly winning (>1)"),
    ):
        summ(label, [r for r in chosen if lo <= r["q_alt"] < hi])
    print("realized outcomes (corpus action actually taken; confounded, sanity only)")
    for label, x in (
        ("took over-trump", [r for r in rows if r["took_over"]]),
        ("took other", [r for r in rows if not r["took_over"]]),
    ):
        print(
            f"  {label:18s} n {len(x):3d}  realized win {np.mean([r['ret'] > 0 for r in x]):.2f}  "
            f"mean score {np.mean([r['ret'] for r in x]):+.2f}  Q-implied P(win) of taken "
            f"{np.mean([r['pw_over'] if r['took_over'] else r['pw_alt'] for r in x]):.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="corpus Q-vs-outcome calibration set -> pickle")
    ex.add_argument("--iter-dir", default="runs/202609_recall_rc/pi/iter3")
    ex.add_argument("--max-shards", type=int, default=None)
    ex.add_argument("--out", required=True)
    rp = sub.add_parser("report", help="bias by trick, P(win) curve, over-trump cost")
    rp.add_argument("--calib", required=True)
    rp.add_argument("--rows", default=None, help="called_suit_overtrump rows pickle")
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        calib = extract(args.iter_dir, args.max_shards)
        with open(args.out, "wb") as f:
            pickle.dump(calib, f)
        print(f"{len(calib)} rows -> {args.out}")
        return 0
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)
    report(calib, args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
