#!/usr/bin/env python3
"""Test arm: a one-cell label correction for the called-suit over-trump
(Arm_D_Review §9.12 option 1; NOT a recipe).

The hindsight referee (§9.11) measured the 256-iteration search label as
under-penalizing the over-trump at trick-0 defender follows by ≈ +0.20
score (search minus referee, +0.198 ± 0.086 on theta_4's residual nodes).
This tool takes an iteration's built ``row_table.pt``, finds the rows that
are §9.2 decisions at trick 0 from the stored observation (called-ace game,
called suit led, a teammate defender winning with a trump, the actor void
in the called suit with a higher trump and another legal play), and
subtracts the bias from the over-trump cards' labels. The corrected table
is written to an arm directory so ``policy_iteration fit/target/distill``
run on it unchanged (``stage_fit`` loads an existing ``row_table.pt``).

The label variance is left as is (the correction's SE² would add ≈ 25% to
those rows' working noise; omitted for the test arm and noted).

Usage (from repo root):

    uv run python -m sheepshead.analysis.overtrump_label_bias \\
        --iter-dir runs/202609_recall_rc/pi/iter4 \\
        --out-dir runs/202609_recall_rc/pi/iter4_ablations/t0_label_bias \\
        --bias-score 0.198
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any

from sheepshead.game import ACTIONS, DECK, TRUMP_POWER, get_card_suit, get_trick_winner
from sheepshead.training.corpus_rows import load_shard, shard_paths
from sheepshead.training.reward_shaping import RETURN_SCALE
from sheepshead.training.search_advantage import RowTable

PLAY_COL = {ACTIONS.index(f"PLAY {c}"): c for c in DECK}  # column -> card


def overtrump_columns(state: dict[str, Any], legal_cols: list[int]) -> list[int] | None:
    """The §9.2 decision at this observation: the legal columns that are
    over-trumps, or None when the node is not a decision."""
    if (
        int(state["partner_mode"]) != 1
        or int(state["alone_called"])
        or int(state["is_leaster"])
    ):
        return None
    called_id = int(state["called_card_id"])
    if not called_id or int(state["picker_rel"]) == 1:
        return None
    called = DECK[called_id - 1]
    cs = get_card_suit(called)
    leader = int(state["leader_rel"])
    if not leader:
        return None
    ids = [int(x) for x in state["trick_card_ids"]]
    order = [((leader - 1 + k) % 5) + 1 for k in range(5)]
    played = [r for r in order if 0 < ids[r - 1] <= 32]
    # The leader's slot is empty when the actor leads, and 33 is the UNDER
    # token: neither is a called-suit lead.
    if not played or played[0] != leader:
        return None
    led = DECK[ids[leader - 1] - 1]
    if led in TRUMP_POWER or get_card_suit(led) != cs:
        return None
    seq = [DECK[ids[r - 1] - 1] for r in played]
    w = played[get_trick_winner(seq, cs) - 1]
    wcard = DECK[ids[w - 1] - 1]
    if wcard not in TRUMP_POWER:
        return None
    if int(state["trick_is_picker"][w - 1]) or int(
        state["trick_is_partner_known"][w - 1]
    ):
        return None
    hand = [DECK[int(x) - 1] for x in state["hand_ids"] if 0 < int(x) <= 32]
    if any(c not in TRUMP_POWER and get_card_suit(c) == cs for c in hand):
        return None
    over = [
        j
        for j in legal_cols
        if j in PLAY_COL
        and PLAY_COL[j] in TRUMP_POWER
        and TRUMP_POWER[PLAY_COL[j]] > TRUMP_POWER[wcard]
    ]
    other = [j for j in legal_cols if j in PLAY_COL and j not in over]
    if not over or not other:
        return None
    return over


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--iter-dir", required=True, help="iteration with row_table.pt and corpus/"
    )
    ap.add_argument(
        "--out-dir", required=True, help="arm directory for the corrected table"
    )
    ap.add_argument(
        "--bias-score",
        type=float,
        required=True,
        help="search minus referee at the over-trump, score per hand (subtracted)",
    )
    ap.add_argument("--trick", type=int, default=0, help="only decisions at this trick")
    args = ap.parse_args(argv)

    table = RowTable.load(os.path.join(args.iter_dir, "row_table.pt"))
    paths = shard_paths(os.path.join(args.iter_dir, "corpus"))
    by_shard: dict[int, list[int]] = collections.defaultdict(list)
    for r, (s, _, _) in enumerate(table.refs):
        by_shard[s].append(r)
    bias = args.bias_score / RETURN_SCALE
    legal = table.masks.bool()
    tagged: list[tuple[int, list[int]]] = []
    classes: collections.Counter[str] = collections.Counter()
    for s, rows in sorted(by_shard.items()):
        episodes = load_shard(paths[s])["episodes"]
        for r in rows:
            _, e, k = table.refs[r]
            ev = episodes[e][k]
            st = ev["state"]
            if int(st["current_trick"]) != args.trick:
                continue
            cols = legal[r].nonzero()[:, 0].tolist()
            over = overtrump_columns(st, cols)
            if over is None:
                continue
            tagged.append((r, over))
            classes[table.node_class[r]] += 1
    shifted_cards: collections.Counter[str] = collections.Counter()
    labeled = 0
    for r, over in tagged:
        for j in over:
            if bool(table.label_mask[r, j]):
                table.advantage[r, j] -= bias
                labeled += 1
                shifted_cards[PLAY_COL[j]] += 1
    os.makedirs(args.out_dir, exist_ok=True)
    table.save(os.path.join(args.out_dir, "row_table.pt"))
    report = {
        "iter_dir": args.iter_dir,
        "bias_score": args.bias_score,
        "bias_advantage_units": bias,
        "trick": args.trick,
        "rows": len(table),
        "rows_tagged": len(tagged),
        "labels_shifted": labeled,
        "classes": dict(classes),
        "shifted_cards": dict(shifted_cards),
    }
    with open(os.path.join(args.out_dir, "label_bias.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
