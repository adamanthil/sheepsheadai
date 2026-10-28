#!/usr/bin/env python3
"""Defender over-trumps a fellow defender on the called suit (Arm_D_Review §9).

Called-ace mode. When the called suit is led, the picker must follow suit and
the secret partner must play the called ace, so once a defender trumps the
lead only another defender can beat it. A following defender who is void in
the suit and holds a higher trump has a choice: over-trump its teammate or
play something else (a schmear, a low fail, a lower trump). Over-trumping
spends a trump to take a trick the team already has.

A DECISION is a defender follow where the called suit was led, a known
defender is winning the trick with a trump, the actor is void in the led suit,
and it holds at least one over-trump AND at least one other legal play.
``extract --all-fail`` widens this to any fail lead (partner revealed when the
lead is not the called suit): the look-alike where covering can protect the
trick, because the picker may trump in.

Rows come from the policy-iteration corpora (``pi/iterN/targeted``), with the
committee's search Q (256 iterations x 3 replicates at follows), the act-time
prior and iteration N's out-of-fold AdvantageModel prediction. Every value is
in game score per decision (Q x RETURN_SCALE).

Subcommands:

    extract   corpus rows -> pickle (rows.pkl / rows_all_fail.pkl)
    value     search verdict by situation; where the policy over-trumps;
              the operator's 10D case (§9.3)
    targets   how far the stored distill targets move the over-trump, and
              whether the model agrees with confident search (§9.4)
    bleed     the same decision on called-suit vs other fail leads (§9.4)

Usage (from repo root):

    uv run python -m sheepshead.analysis.called_suit_overtrump extract \
        --out runs/202609_recall_rc/overtrump_probe/rows.pkl
    uv run python -m sheepshead.analysis.called_suit_overtrump extract --all-fail \
        --out runs/202609_recall_rc/overtrump_probe/rows_all_fail.pkl
    uv run python -m sheepshead.analysis.called_suit_overtrump value \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl
    uv run python -m sheepshead.analysis.called_suit_overtrump targets \
        --rows runs/202609_recall_rc/overtrump_probe/rows.pkl
    uv run python -m sheepshead.analysis.called_suit_overtrump bleed \
        --rows runs/202609_recall_rc/overtrump_probe/rows_all_fail.pkl
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import pickle
from collections.abc import Callable, Iterable

import numpy as np
import torch

from sheepshead.game import (
    ACTIONS,
    CARD_POINTS,
    DECK,
    TRUMP_POWER,
    get_card_suit,
    get_trick_winner,
)
from sheepshead.rewards import RETURN_SCALE

DEFAULT_RUN = "runs/202609_recall_rc"
DEFAULT_ITERATIONS = ("iter1", "iter2", "iter3")
UNDER_CARD_ID = 33  # observation id of the face-down under card
DIAMOND_TRUMPS = ("AD", "10D", "KD", "9D", "8D", "7D")
POINT_TRUMPS = ("AD", "10D")  # an over-trump that is itself a 10/11-point card


def _corpus_paths(iter_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(iter_dir, "targeted", "corpus_*.pt")))


def _decision(ev: dict, all_fail: bool) -> dict | None:
    """The over-trump decision features of one corpus event, or None."""
    nc = str(ev.get("node_class"))
    if not nc.startswith("std|") or "defender-follow" not in nc:
        return None
    st = ev["state"]
    if int(st["partner_mode"]) != 1 or int(st["called_card_id"]) == 0:
        return None
    if len(ev["valid_actions"]) < 2:
        return None
    called = DECK[int(st["called_card_id"]) - 1]
    cs = get_card_suit(called)
    leader = int(st["leader_rel"])
    tc = [int(x) for x in st["trick_card_ids"]]
    if tc[leader - 1] in (0, UNDER_CARD_ID):
        return None
    led = DECK[tc[leader - 1] - 1]
    if led in TRUMP_POWER:
        return None
    led_suit = get_card_suit(led)
    led_called = led_suit == cs
    if not all_fail and not led_called:
        return None
    order = [(leader - 1 + k) % 5 for k in range(5)]  # rel indices in play order
    played = [i for i in order if tc[i] != 0]
    cards = {
        i: ("UNDER" if tc[i] == UNDER_CARD_ID else DECK[tc[i] - 1]) for i in played
    }
    seq = [cards[i] for i in played]
    w = played[get_trick_winner(seq, led_suit) - 1]
    wcard = cards[w]
    if wcard not in TRUMP_POWER:
        return None
    if st["trick_is_picker"][w] or st["trick_is_partner_known"][w]:
        return None
    if int(st["partner_rel"]) == w + 1:
        return None
    # Off the called suit the winner is a known defender only once the
    # partner is revealed (an unrevealed partner could be the trumper).
    partner_known = any(st["trick_is_partner_known"]) or int(st["partner_rel"]) != 0
    if not led_called and not partner_known:
        return None
    valid = sorted(ev["valid_actions"])
    vc = [ACTIONS[a - 1][5:] for a in valid]
    if any(get_card_suit(c) == led_suit and c not in TRUMP_POWER for c in vc):
        return None  # following suit: no choice of trump
    over = np.array(
        [c in TRUMP_POWER and TRUMP_POWER[c] > TRUMP_POWER[wcard] for c in vc]
    )
    if not over.any() or over.all():
        return None
    after = [i for i in order if tc[i] == 0 and i != 0]
    hand = [DECK[int(x) - 1] for x in st["hand_ids"] if int(x)]
    return {
        "trick": int(st["current_trick"]),
        "under": int(st["called_under"]),
        "led_called": led_called,
        "cards": vc,
        "over": over,
        "trump": np.array([c in TRUMP_POWER for c in vc]),
        "valid": valid,
        "winner": wcard,
        "ace_played": any(cards[j] == called for j in played),
        "picker_after": any(st["trick_is_picker"][j] for j in after),
        "n_after": len(after),
        "pos_in_trick": len(played) + 1,
        "trick_pts": sum(CARD_POINTS.get(c, 0) for c in seq),
        "hand": hand,
        "ntrump": sum(c in TRUMP_POWER for c in hand),
        "prel": int(st["picker_rel"]),
    }


def extract_rows(
    run_dir: str,
    iterations: Iterable[str],
    *,
    all_fail: bool = False,
    max_shards: int | None = None,
) -> list[dict]:
    """Every decision row in the iterations' targeted corpora, with the act-
    time prior, the committee's Q / variance (None where unsearched) and the
    iteration's out-of-fold AdvantageModel advantages over the legal set."""
    rows: list[dict] = []
    for it in iterations:
        base = os.path.join(run_dir, "pi", it)
        tab = torch.load(
            os.path.join(base, "row_table.pt"), weights_only=False, map_location="cpu"
        )
        oof = torch.load(
            os.path.join(base, "oof_predictions.pt"),
            weights_only=False,
            map_location="cpu",
        )["a_model"].numpy()
        idx = {tuple(r): i for i, r in enumerate(tab["refs"])}
        n_it = 0
        for si, path in enumerate(_corpus_paths(base)[:max_shards]):
            d = torch.load(path, weights_only=False, map_location="cpu")
            for ei, ep in enumerate(d["episodes"]):
                for vi, ev in enumerate(ep):
                    dec = _decision(ev, all_fail)
                    if dec is None:
                        continue
                    i = idx.get((si, ei, vi))
                    q = ev.get("search_q")
                    qv = ev.get("search_q_var")
                    rows.append(
                        {
                            **dec,
                            "it": it,
                            "key": (si, ei, vi),
                            "shard": path,
                            "ep": ei,
                            "ev": vi,
                            "prior": np.array(ev["anchor_probs"]),
                            "action": int(ev["action"]),
                            "q": np.array(q) if q is not None else None,
                            "qv": np.array(qv) if qv is not None else None,
                            "am": (
                                oof[i][np.where(tab["masks"][i].numpy())[0]]
                                if i is not None
                                else None
                            ),
                            "ret": ev.get("final_return_label"),
                        }
                    )
                    n_it += 1
            del d
        print(f"{it}: {n_it} decisions", flush=True)
    return rows


def _alt_class(card: str) -> str:
    if card in TRUMP_POWER:
        return "undertrump"
    if CARD_POINTS.get(card, 0) >= 10:
        return "schmear"
    return "K" if card[0] == "K" else "low fail"


def enrich(rows: list[dict]) -> list[dict]:
    """Add the derived contrasts every read uses, in place. ``d`` = policy-
    preferred over-trump minus policy-preferred other play (score), so neither
    side is chosen on the Q being measured; ``d_best`` = against search's best
    other play (winner's curse favours it)."""
    for r in rows:
        o, p = r["over"], r["prior"]
        bo = int(np.argmax(np.where(o, p, -1)))
        bn = int(np.argmax(np.where(~o, p, -1)))
        r["over_card"] = r["cards"][bo]
        r["p_over"] = float(p[o].sum())
        r["arg_over"] = bool(o[p.argmax()])
        r["max_fail_pts"] = max(
            [CARD_POINTS.get(c, 0) for c in r["cards"] if c not in TRUMP_POWER],
            default=-1,
        )
        r["took_over"] = bool(o[r["valid"].index(r["action"])])
        if r["q"] is None:
            continue
        q, v = r["q"], r["qv"]
        bqn = int(np.argmax(np.where(~o, q, -9)))
        r["d"] = RETURN_SCALE * (q[bo] - q[bn])
        r["d_best"] = RETURN_SCALE * (q[bo] - q[bqn])
        r["se"] = RETURN_SCALE * np.sqrt(v[bo] + v[bn])
        r["s_over"] = bool(o[q.argmax()])
        r["alt"] = _alt_class(r["cards"][bqn])
        am = r["am"]
        r["dm"] = RETURN_SCALE * (am[bo] - am[bn]) if am is not None else np.nan
    return rows


def load_rows(path: str, iteration: str | None = None) -> list[dict]:
    """Rows written by ``extract``, enriched; optionally one iteration only."""
    with open(path, "rb") as f:
        rows = pickle.load(f)
    if iteration is not None:
        rows = [r for r in rows if r["it"] == iteration]
    return enrich(rows)


def winner_class(card: str) -> str:
    return "Q" if card[0] == "Q" else "J" if card[0] == "J" else "diamond"


def show(label: str, rows: list[dict], min_searched: int = 8) -> None:
    """One summary line: policy rate, and the search / model verdicts."""
    n = len(rows)
    if not n:
        return
    line = (
        f"  {label:34s} n {n:4d}  policy over-trumps "
        f"{100 * np.mean([x['arg_over'] for x in rows]):4.0f}%  "
        f"p(over) {np.mean([x['p_over'] for x in rows]):.2f}"
    )
    q = [x for x in rows if x["q"] is not None]
    if len(q) >= min_searched:
        d = np.array([x["d"] for x in q])
        line += (
            f" | searched {len(q):3d}: over-alt {d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}"
            f"  vs best alt {np.mean([x['d_best'] for x in q]):+.3f}"
            f"  search over {100 * np.mean([x['s_over'] for x in q]):3.0f}%"
            f"  model {np.nanmean([x['dm'] for x in q]):+.3f}"
            f"  z>2 {sum(x['d'] > 2 * x['se'] for x in q)} z<-2 {sum(x['d'] < -2 * x['se'] for x in q)}"
        )
    elif q:
        line += f" | searched {len(q)}"
    print(line)


def show_groups(
    title: str, rows: list[dict], key: Callable[[dict], object], min_searched: int = 8
) -> None:
    groups: dict[object, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print(f"-- {title}")
    for k in sorted(groups, key=str):
        show(str(k), groups[k], min_searched)


def cmd_value(rows: list[dict]) -> None:
    """§9.3: search verdict by situation, then where the policy over-trumps."""
    show("ALL", rows)
    for it in sorted({r["it"] for r in rows}):
        show(it, [r for r in rows if r["it"] == it])
    show_groups("picker still to play", rows, lambda r: r["picker_after"])
    show_groups(
        "picker after x n after", rows, lambda r: (r["picker_after"], r["n_after"])
    )
    show_groups("ace played yet", rows, lambda r: r["ace_played"])
    show_groups("winning trump", rows, lambda r: winner_class(r["winner"]))
    show_groups("trick", rows, lambda r: r["trick"])
    show_groups("under", rows, lambda r: r["under"])
    show_groups("my trump count", rows, lambda r: min(r["ntrump"], 4))
    show_groups(
        "fail pts available to throw",
        rows,
        lambda r: (
            "none"
            if r["max_fail_pts"] < 0
            else "0-4"
            if r["max_fail_pts"] < 10
            else "10+"
        ),
    )
    searched = [r for r in rows if r["q"] is not None]
    show_groups(
        "search's best alternative (searched only)", searched, lambda r: r["alt"]
    )

    chosen = [r for r in searched if r["arg_over"]]
    print("\n== where the policy over-trumps")
    show("policy over-trumps", chosen)
    show("policy doesn't", [r for r in searched if not r["arg_over"]])
    show_groups(
        "  by picker after x n after",
        chosen,
        lambda r: (r["picker_after"], r["n_after"]),
        6,
    )
    show_groups("  by my over-trump card", chosen, _over_card_class, 6)
    show_groups(
        "  by fail pts available",
        chosen,
        lambda r: (
            "none"
            if r["max_fail_pts"] < 0
            else "0-4"
            if r["max_fail_pts"] < 10
            else "10+"
        ),
        6,
    )
    print("-- picker already played (nothing can protect the trick either way)")
    safe = [r for r in chosen if not r["picker_after"]]
    for label, keep in (
        (
            "schmear in hand, not AD/10D",
            lambda r: r["max_fail_pts"] >= 10 and r["over_card"] not in POINT_TRUMPS,
        ),
        (
            "no schmear, not AD/10D",
            lambda r: (
                0 <= r["max_fail_pts"] < 10 and r["over_card"] not in POINT_TRUMPS
            ),
        ),
        ("over-trump is AD/10D", lambda r: r["over_card"] in POINT_TRUMPS),
    ):
        show(label, [r for r in safe if keep(r)], 6)
    show("picker still to play", [r for r in chosen if r["picker_after"]], 6)

    print("\n== operator's case: a diamond winning, Q/J over-trump")
    base = [
        r
        for r in searched
        if r["winner"] in DIAMOND_TRUMPS and r["over_card"][0] in "QJ"
    ]
    for label, keep in (
        (
            "holds fail K",
            lambda r: any(c[0] == "K" and c not in TRUMP_POWER for c in r["cards"]),
        ),
        ("holds fail A/10", lambda r: r["max_fail_pts"] >= 10),
        ("holds only low fail", lambda r: 0 <= r["max_fail_pts"] < 4),
        ("any fail", lambda r: r["max_fail_pts"] >= 0),
    ):
        x = [r for r in base if keep(r)]
        show(f"{label} (all)", x)
        show(f"{label} (policy over-trumps)", [r for r in x if r["arg_over"]])
    tend = [r for r in base if r["winner"] == "10D" and r["max_fail_pts"] >= 0]
    show("10D winning, Q/J over, any fail", tend)
    show("  policy over-trumps", [r for r in tend if r["arg_over"]])


def _over_card_class(r: dict) -> str:
    c = r["over_card"]
    if c[0] in "QJ":
        return c[0]
    return "AD/10D" if c in POINT_TRUMPS else "low D"


def _attach_targets(rows: list[dict]) -> None:
    """Read each row's stored distill target and gamma from its shard."""
    by_shard: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_shard[r["shard"]].append(r)
    for path, rr in by_shard.items():
        d = torch.load(path, weights_only=False, map_location="cpu")
        for r in rr:
            ev = d["episodes"][r["ep"]][r["ev"]]
            r["dset"] = ev["distill_set"]
            t = ev.get("search_target")
            r["tgt"] = np.array(t) if t is not None else None
            r["gamma"] = ev.get("pi_gamma")
        del d


def cmd_targets(rows: list[dict]) -> None:
    """§9.4: how the stored targets move the over-trump, and whether the
    model agrees with search where search is confident."""
    _attach_targets(rows)
    for it in sorted({r["it"] for r in rows}):
        x = [r for r in rows if r["it"] == it and r["tgt"] is not None]
        print(
            f"{it}: n {len(x)}  prior over {np.mean([r['p_over'] for r in x]):.2f}  "
            f"target over {np.mean([r['tgt'][r['over']].sum() for r in x]):.2f}  "
            f"gamma {np.mean([r['gamma'] or 0 for r in x]):.2f}"
        )
    q = [r for r in rows if r["q"] is not None and r["tgt"] is not None]
    sig = [r for r in q if r["d"] < -2 * r["se"]]
    print(
        f"\nsearched {len(q)}; search says over-trump worse at z<-2: {len(sig)} ({100 * len(sig) / len(q):.0f}%)"
    )
    for label, x in (("all searched", q), ("search z<-2", sig)):
        dm = np.array([r["dm"] for r in x])
        po = np.array([r["p_over"] for r in x])
        to = np.array([r["tgt"][r["over"]].sum() for r in x])
        print(
            f"  {label:14s} n {len(x)}  model agrees over<alt {100 * np.mean(dm < 0):.0f}%  "
            f"gamma {np.mean([r['gamma'] or 0 for r in x]):.2f}  prior over {po.mean():.2f} -> target {to.mean():.2f}"
        )
    for label, x in (
        ("z<-2, model says over-trump better", [r for r in sig if r["dm"] > 0]),
        ("z<-2, model agrees", [r for r in sig if r["dm"] <= 0]),
    ):
        if x:
            print(
                f"  {label:36s} n {len(x)}  prior {np.mean([r['p_over'] for r in x]):.2f} -> "
                f"target {np.mean([r['tgt'][r['over']].sum() for r in x]):.2f}  search {np.mean([r['d'] for r in x]):+.2f}  "
                f"model {np.mean([r['dm'] for r in x]):+.2f}"
            )
    chosen = [r for r in q if r["arg_over"]]
    print(
        f"\nwhere the policy over-trumps: n {len(chosen)}  search {np.mean([r['d'] for r in chosen]):+.3f}  "
        f"model {np.nanmean([r['dm'] for r in chosen]):+.3f}  gamma {np.mean([r['gamma'] or 0 for r in chosen]):.2f}"
    )


def cmd_bleed(rows: list[dict]) -> None:
    """§9.4: the same decision on called-suit vs other fail leads."""
    for label, led_called in (("called suit led", True), ("other fail led", False)):
        print("==", label)
        for pa in (True, False):
            x = [
                r
                for r in rows
                if r["led_called"] == led_called and r["picker_after"] == pa
            ]
            show(f"picker after {pa}", x)
            show(
                "  winner diamond, Q/J over",
                [
                    r
                    for r in x
                    if r["winner"] in DIAMOND_TRUMPS and r["over_card"][0] in "QJ"
                ],
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="corpus decision rows -> pickle")
    ex.add_argument("--run-dir", default=DEFAULT_RUN)
    ex.add_argument("--iterations", nargs="+", default=list(DEFAULT_ITERATIONS))
    ex.add_argument(
        "--all-fail",
        action="store_true",
        help="any fail lead, not only the called suit",
    )
    ex.add_argument(
        "--max-shards", type=int, default=None, help="per iteration (smoke runs)"
    )
    ex.add_argument("--out", required=True)
    for name in ("value", "targets", "bleed"):
        p = sub.add_parser(name)
        p.add_argument("--rows", required=True, help="pickle written by extract")
        p.add_argument("--iteration", default=None, help="restrict to one iteration")
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        rows = extract_rows(
            args.run_dir,
            args.iterations,
            all_fail=args.all_fail,
            max_shards=args.max_shards,
        )
        with open(args.out, "wb") as f:
            pickle.dump(rows, f)
        print(f"{len(rows)} rows -> {args.out}")
        return 0
    rows = load_rows(args.rows, args.iteration)
    {"value": cmd_value, "targets": cmd_targets, "bleed": cmd_bleed}[args.cmd](rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
