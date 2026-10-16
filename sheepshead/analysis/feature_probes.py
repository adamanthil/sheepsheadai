#!/usr/bin/env python3
"""Linear probes: is "the trick is already secured" in the frozen features
the AdvantageModel reads? (Arm_D_Review §9.4)

Over the called-ace follow rows of one iteration's row table (the encoder of
the checkpoint that iteration's corpus was generated from), held out by game:

    led_called  the called suit was led this trick
    secured     ... and a known defender is winning it with a trump, which
                the picker side cannot beat (picker must follow, partner must
                play the ace)

A high held-out AUC means the information is already linearly readable, so
an auxiliary head that classifies it would add nothing the trunk lacks.

Usage (from repo root):

    uv run python -m sheepshead.analysis.feature_probes \
        --iter-dir runs/202609_recall_rc/pi/iter3
"""

from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sheepshead.game import DECK, TRUMP_POWER, get_card_suit, get_trick_winner

UNDER_CARD_ID = 33
HOLDOUT_FRAC = 0.25


def label_rows(iter_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(row indices, led_called, secured, is_defender) over called-ace
    follow rows."""
    t = torch.load(
        os.path.join(iter_dir, "row_table.pt"), weights_only=False, map_location="cpu"
    )
    shards = sorted(glob.glob(os.path.join(iter_dir, "targeted", "corpus_*.pt")))
    by_shard: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for i, (s, e, v) in enumerate(t["refs"]):
        c = t["node_class"][i]
        if c.startswith("std|") and "follow" in c:
            by_shard[s].append((i, e, v))
    idx, led_called, secured, defender = [], [], [], []
    for s, items in by_shard.items():
        d = torch.load(shards[s], weights_only=False, map_location="cpu")
        for i, e, v in items:
            st = d["episodes"][e][v]["state"]
            if int(st["partner_mode"]) != 1 or int(st["called_card_id"]) == 0:
                continue
            cs = get_card_suit(DECK[int(st["called_card_id"]) - 1])
            leader = int(st["leader_rel"])
            tc = [int(x) for x in st["trick_card_ids"]]
            if tc[leader - 1] in (0, UNDER_CARD_ID):
                continue
            led = DECK[tc[leader - 1] - 1]
            lc = led not in TRUMP_POWER and get_card_suit(led) == cs
            sec = False
            if lc:
                order = [(leader - 1 + k) % 5 for k in range(5)]
                played = [j for j in order if tc[j] != 0]
                seq = [
                    "UNDER" if tc[j] == UNDER_CARD_ID else DECK[tc[j] - 1]
                    for j in played
                ]
                w = played[get_trick_winner(seq, cs) - 1]
                sec = (
                    seq[played.index(w)] in TRUMP_POWER
                    and not st["trick_is_picker"][w]
                    and not st["trick_is_partner_known"][w]
                )
            idx.append(i)
            led_called.append(lc)
            secured.append(sec)
            defender.append("defender" in t["node_class"][i])
        del d
    return np.array(idx), np.array(led_called), np.array(secured), np.array(defender)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iter-dir", default="runs/202609_recall_rc/pi/iter3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    idx, led_called, secured, defender = label_rows(args.iter_dir)
    t = torch.load(
        os.path.join(args.iter_dir, "row_table.pt"),
        weights_only=False,
        map_location="cpu",
    )
    x = t["features"][idx].numpy()
    game = t["game"][idx].numpy()
    ug = np.unique(game)
    hold = set(
        np.random.default_rng(args.seed)
        .choice(ug, size=int(len(ug) * HOLDOUT_FRAC), replace=False)
        .tolist()
    )
    te = np.array([g in hold for g in game.tolist()])
    mu, sd = x[~te].mean(0), x[~te].std(0) + 1e-6
    xs = (x - mu) / sd
    print(f"called-ace follow rows: {len(idx)}; held-out {te.sum()} (by game)")

    def probe(name: str, y: np.ndarray, sub: np.ndarray, sub_name: str) -> None:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(xs[sub & ~te], y[sub & ~te])
        p = clf.predict_proba(xs[sub & te])[:, 1]
        print(
            f"  {name:11s} {sub_name:36s} base rate {y[sub].mean():.3f}  held-out AUC {roc_auc_score(y[sub & te], p):.3f}"
        )

    every = np.ones(len(idx), bool)
    for name, y in (("led_called", led_called), ("secured", secured)):
        probe(name, y, every, "all follows")
        probe(name, y, defender, "defender follows")
    probe(
        "secured", secured, led_called & defender, "| called suit led, defender follows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
