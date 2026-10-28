#!/usr/bin/env python3
"""Where is leading the called suit (convention C2) NOT optimal, at the
current policy level? (Convention_Optimality addendum A2)

Rows: every called-ace defender lead in the policy-iteration corpora with a
legal called-suit lead while the suit is still unled (the corpus's
``conv_cs_ids``), with the committee's search Q (t0 leads 1024 iterations,
t1 512, later 256; x 3 replicates), the act-time prior, the stored distill
target and the out-of-fold AdvantageModel prediction. The contrast is the
policy-preferred called-suit card minus the policy-preferred other lead
(score), which avoids the winner's curse of comparing search's best on each
side; "best-vs-best" is printed beside it.

Subcommands:

    extract  corpus rows -> pickle
    report   the verdict by hand feature, the crosses, the late-trick
             trump-void singleton exception, early per-node exceptions, and
             prior vs target movement by iteration
    nodes    the late-exception and early-tie nodes to re-search
             (``research_corpus_nodes search``)
    confirm  the corpus labels vs the re-search on those nodes

Usage (from repo root):

    uv run python -m sheepshead.analysis.called_suit_lead_cells extract \
        --out runs/202609_recall_rc/cs_lead_confirm/csrows.pkl
    uv run python -m sheepshead.analysis.called_suit_lead_cells report \
        --rows runs/202609_recall_rc/cs_lead_confirm/csrows.pkl
    uv run python -m sheepshead.analysis.called_suit_lead_cells nodes \
        --rows runs/202609_recall_rc/cs_lead_confirm/csrows.pkl \
        --out runs/202609_recall_rc/cs_lead_confirm/nodes.json
    uv run python -m sheepshead.analysis.called_suit_lead_cells confirm \
        --rows runs/202609_recall_rc/cs_lead_confirm/csrows.pkl \
        --results runs/202609_recall_rc/cs_lead_confirm/search_4096.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
from collections.abc import Callable

import numpy as np
import torch

from sheepshead.analysis.research_corpus_nodes import load_results
from sheepshead.game import ACTIONS, TRUMP
from sheepshead.rewards import RETURN_SCALE

CARD_POINTS_BY_RANK = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2}
TRUMP_SET = set(TRUMP)
DEFAULT_RUN = "runs/202609_recall_rc"


def extract(run_dir: str, iterations: list[str]) -> list[dict]:
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
        shards = sorted(glob.glob(os.path.join(base, "targeted", "corpus_*.pt")))
        for si, path in enumerate(shards):
            d = torch.load(path, weights_only=False, map_location="cpu")
            for ei, ep in enumerate(d["episodes"]):
                for vi, ev in enumerate(ep):
                    if not ev.get("conv_cs_ids") or ev.get("search_q") is None:
                        continue
                    nc = str(ev["node_class"])
                    if "defender-lead" not in nc:
                        continue
                    rows.append(_row(ev, it, nc, (si, ei, vi), path, idx, tab, oof))
            del d
        print(it, sum(r["it"] == it for r in rows), flush=True)
    return rows


def _row(
    ev: dict,
    it: str,
    nc: str,
    key: tuple[int, int, int],
    path: str,
    idx: dict,
    tab: dict,
    oof: np.ndarray,
) -> dict:
    st = ev["state"]
    valid = sorted(ev["valid_actions"])
    cards = [ACTIONS[a - 1][5:] for a in valid]
    cs_ids = set(ev["conv_cs_ids"])
    cs = np.array([a in cs_ids for a in valid])
    istr = np.array([c in TRUMP_SET for c in cards])
    i = idx.get(key)
    am = oof[i][np.where(tab["masks"][i].numpy())[0]] if i is not None else None
    csuit = cards[int(np.argmax(cs))][-1]
    fails = [c for c, t in zip(cards, istr) if not t]
    suits = collections.Counter(c[-1] for c in fails)
    side = {s: n for s, n in suits.items() if s != csuit}
    csc = [c for c in fails if c[-1] == csuit]
    target = ev.get("search_target")
    return {
        "it": it,
        "cls": nc,
        "trick": int(st["current_trick"]),
        "prel": int(st["picker_rel"]),
        "under": int(st["called_under"]),
        "cards": cards,
        "cs": cs,
        "istr": istr,
        "q": np.array(ev["search_q"]),
        "qv": np.array(ev["search_q_var"]),
        "prior": np.array(ev["anchor_probs"]),
        "target": np.array(target) if target is not None else None,
        "am": am,
        "action": ev["action"],
        "valid": valid,
        "ret": ev.get("final_return_label"),
        "csuit": csuit,
        "ntrump": int(istr.sum()),
        "cs_len": len(csc),
        "cs_pts": sum(CARD_POINTS_BY_RANK.get(c[:-1], 0) for c in csc),
        "cs_low": any(c[:-1] in ("7", "8", "9") for c in csc),
        "cs_fat": any(c[:-1] in ("A", "10") for c in csc),
        "side_aces": sum(c[:-1] == "A" for c in fails if c[-1] != csuit),
        "side_max": max(side.values()) if side else 0,
        "side_void": 2 - len(side),
        "key": key,
        "shard": path,
        "ep": key[1],
        "ev": key[2],
    }


def enrich(rows: list[dict]) -> list[dict]:
    """Derived contrasts, in place (score units)."""
    for r in rows:
        cs, p, q = r["cs"], r["prior"], r["q"]
        bc = int(np.argmax(np.where(cs, p, -1)))
        ba = int(np.argmax(np.where(~cs, p, -1)))
        bqa = int(np.argmax(np.where(~cs, q, -9)))
        bqc = int(np.argmax(np.where(cs, q, -9)))
        r["bc"], r["ba"] = bc, ba
        r["d"] = RETURN_SCALE * (q[bc] - q[ba])
        r["d_best"] = RETURN_SCALE * (q[bqc] - q[bqa])
        r["se"] = RETURN_SCALE * np.sqrt(r["qv"][bc] + r["qv"][ba])
        r["arg_cs"] = bool(cs[p.argmax()])
        r["p_cs"] = float(p[cs].sum())
        r["s_cs"] = bool(cs[q.argmax()])
        r["t_cs"] = float(r["target"][cs].sum()) if r["target"] is not None else np.nan
        r["dm"] = (
            RETURN_SCALE * (r["am"][bc] - r["am"][ba])
            if r["am"] is not None
            else np.nan
        )
        a = r["cards"][bqa]
        rank = "low" if a[:-1] in ("7", "8", "9") else "K" if a[:-1] == "K" else "fat"
        r["alt_cls"] = (
            "trump"
            if r["istr"][bqa]
            else ("side-" if a[-1] != r["csuit"] else "?") + rank
        )
        r["cs_rank"] = next(
            c[:-1]
            for c, t in zip(r["cards"], r["istr"])
            if c[-1] == r["csuit"] and not t
        )
    return rows


def load_rows(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return enrich(pickle.load(f))


def show(label: str, rows: list[dict], min_n: int = 15) -> None:
    if len(rows) < min_n:
        if rows:
            print(f"  {label:34s} n {len(rows):4d}")
        return
    d = np.array([x["d"] for x in rows])
    print(
        f"  {label:34s} n {len(rows):4d}  pol cs {100 * np.mean([x['arg_cs'] for x in rows]):4.0f}%  "
        f"srch cs {100 * np.mean([x['s_cs'] for x in rows]):4.0f}%  cs-alt {d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}  "
        f"bvb {np.mean([x['d_best'] for x in rows]):+.3f}  model {np.nanmean([x['dm'] for x in rows]):+.3f}  "
        f"cs<alt {100 * np.mean(d < 0):3.0f}%"
    )


def cross(
    title: str, rows: list[dict], *keys: Callable[[dict], object], min_n: int = 15
) -> None:
    groups: dict[tuple[object, ...], list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[tuple(k(r) for k in keys)].append(r)
    print("==", title)
    for v in sorted(groups, key=str):
        show(str(v[0]) if len(v) == 1 else str(v), groups[v], min_n)


def _pts_bucket(r: dict) -> str:
    return "0" if r["cs_pts"] == 0 else "1-9" if r["cs_pts"] < 10 else "10+"


def _rank_bucket(r: dict) -> str:
    return "low" if r["cs_rank"] in ("7", "8", "9") else r["cs_rank"]


def report(rows: list[dict]) -> None:
    std = [r for r in rows if r["cls"].startswith("std")]
    print("classes", collections.Counter(r["cls"] for r in rows).most_common())
    show("ALL std", std)
    for it in sorted({r["it"] for r in std}):
        show(it, [r for r in std if r["it"] == it])
    features: dict[str, Callable[[dict], object]] = {
        "trick": lambda r: r["trick"],
        "picker_rel": lambda r: r["prel"],
        "under": lambda r: r["under"],
        "ntrump": lambda r: min(r["ntrump"], 4),
        "cs_len": lambda r: min(r["cs_len"], 3),
        "cs_pts": _pts_bucket,
        "cs_low/fat": lambda r: (r["cs_low"], r["cs_fat"]),
        "side_max": lambda r: min(r["side_max"], 4),
        "side_aces": lambda r: r["side_aces"],
        "side_void": lambda r: r["side_void"],
        "alt class (search best alt)": lambda r: r["alt_cls"],
    }
    for name, key in features.items():
        cross(f"by {name}", std, key)

    def trick_bucket(r: dict) -> int:
        return min(r["trick"], 4)

    def trumps(r: dict) -> int:
        return min(r["ntrump"], 2)

    early = [r for r in std if r["trick"] <= 1]
    void_early = [r for r in early if r["ntrump"] == 0]
    cross("trick x ntrump(0,1,2+)", std, trick_bucket, trumps)
    cross("t0-1 x ntrump x cs_len", early, trumps, lambda r: min(r["cs_len"], 2))
    cross(
        "t0-1 trump-void: cs_pts x side_max>=3",
        void_early,
        _pts_bucket,
        lambda r: r["side_max"] >= 3,
    )
    cross("t0-1 trump-void: picker_rel", void_early, lambda r: r["prel"])
    cross("t0-1 trump-void: under", void_early, lambda r: r["under"])
    cross(
        "t0-1 1 trump, cs_len 1: cs_pts x side_max>=3",
        [r for r in early if r["ntrump"] == 1 and r["cs_len"] == 1],
        _pts_bucket,
        lambda r: r["side_max"] >= 3,
    )
    cross("t0-1 under x ntrump", early, lambda r: r["under"], trumps)
    cross(
        "t2+ x ntrump x cs_len",
        [r for r in std if r["trick"] >= 2],
        lambda r: min(r["trick"], 3),
        trumps,
        lambda r: min(r["cs_len"], 2),
    )

    cross(
        "singleton called card rank x trick bucket",
        [r for r in std if r["cs_len"] == 1],
        _rank_bucket,
        lambda r: "0-2" if r["trick"] <= 2 else "3+",
    )
    late = [r for r in std if r["trick"] >= 3 and r["ntrump"] == 0 and r["cs_len"] == 1]
    cross("late void singleton: alt class", late, lambda r: r["alt_cls"])
    cross("late void singleton: called rank", late, _rank_bucket)
    cross("late void singleton: picker_rel", late, lambda r: r["prel"])
    cross("late void singleton: under", late, lambda r: r["under"])
    cross("late void singleton: hand size", late, lambda r: len(r["cards"]))

    early2 = [r for r in std if r["trick"] <= 2]
    z = np.array([r["d"] / max(r["se"], 1e-6) for r in early2])
    print(
        f"tricks 0-2: {len(early2)} nodes, z<-2: {(z < -2).sum()} ({100 * (z < -2).mean():.1f}%), z>2: {(z > 2).sum()}"
    )

    print("\nprior -> target movement by iteration")
    for label, keep in (
        (
            "late void single",
            lambda r: r["trick"] >= 3 and r["ntrump"] == 0 and r["cs_len"] == 1,
        ),
        (
            "  hand 2",
            lambda r: (
                r["trick"] >= 3
                and r["ntrump"] == 0
                and r["cs_len"] == 1
                and len(r["cards"]) == 2
            ),
        ),
        ("early (t0-2)", lambda r: r["trick"] <= 2),
    ):
        for it in sorted({r["it"] for r in std}):
            x = [r for r in std if keep(r) and r["it"] == it]
            print(
                f"  {label:18s} {it} n {len(x):4d}  prior cs mass {np.mean([r['p_cs'] for r in x]):.2f}  "
                f"target cs mass {np.nanmean([r['t_cs'] for r in x]):.2f}  search argmax cs {100 * np.mean([r['s_cs'] for r in x]):.0f}%  "
                f"d {np.mean([r['d'] for r in x]):+.3f}"
            )


def select_nodes(rows: list[dict], iteration: str) -> list[dict]:
    """The late trump-void singleton exception and the early tie cell, for
    re-search (shard/ep/ev index the iteration's targeted corpus)."""
    out = []
    for r in rows:
        if not r["cls"].startswith("std") or r["it"] != iteration:
            continue
        late = r["trick"] >= 3 and r["ntrump"] == 0 and r["cs_len"] == 1
        tie = (
            r["trick"] <= 1
            and r["ntrump"] <= 1
            and r["cs_len"] == 1
            and r["cs_pts"] == 0
            and r["side_max"] >= 3
        )
        if late or tie:
            si, ei, vi = r["key"]
            out.append(
                {
                    "shard": si,
                    "ep": ei,
                    "ev": vi,
                    "cell": "late_void" if late else "early_tie",
                    "trick": r["trick"],
                    "prel": r["prel"],
                    "hand_size": len(r["cards"]),
                }
            )
    out.sort(key=lambda n: (n["cell"] != "late_void", n["shard"], n["ep"]))
    return out


def confirm(rows: list[dict], results: str, budget: str, iteration: str) -> None:
    """Corpus label vs the re-search, same contrast, on the searched nodes."""
    by_key = {r["key"]: r for r in rows if r["it"] == iteration}
    res, errors = load_results(results, budget)
    out = []
    for s in res:
        r = by_key[(s["shard"], s["ep"], s["ev"])]
        h = s["search"][budget]
        if [ACTIONS[a - 1][5:] for a in h["actions"]] != r["cards"]:
            raise ValueError(
                f"node {r['key']}: re-search legal set differs from the corpus's"
            )
        cs = r["cs"]
        qh, vh = np.array(h["q"]), np.array(h["qvar"])
        bqa = int(np.argmax(np.where(~cs, qh, -9)))
        out.append(
            {
                **r,
                "cell": s["cell"],
                "lo": r["d"],
                "hi": RETURN_SCALE * (qh[r["bc"]] - qh[r["ba"]]),
                "hi_best": RETURN_SCALE * (qh[r["bc"]] - qh[bqa]),
                "hi_se": RETURN_SCALE * np.sqrt(vh[r["bc"]] + vh[r["ba"]]),
                "hi_arg_cs": bool(cs[int(np.argmax(qh))]),
            }
        )
    print(f"{len(out)} nodes read, {errors} unusable")

    def line(label: str, x: list[dict]) -> None:
        if len(x) < 3:
            if x:
                print(f"  {label:30s} n {len(x)}")
            return
        lo, hi, hb = (np.array([r[k] for r in x]) for k in ("lo", "hi", "hi_best"))

        def se(a: np.ndarray) -> float:
            return float(a.std(ddof=1) / np.sqrt(len(a)))

        print(
            f"  {label:30s} n {len(x):3d}  corpus {lo.mean():+.3f}±{se(lo):.3f}  {budget} {hi.mean():+.3f}±{se(hi):.3f}  "
            f"vs {budget} best alt {hb.mean():+.3f}  {budget} top card called {100 * np.mean([r['hi_arg_cs'] for r in x]):.0f}%  "
            f"policy argmax called {100 * np.mean([r['arg_cs'] for r in x]):.0f}%  z<-2 {sum(r['hi'] < -2 * r['hi_se'] for r in x)}"
        )

    for cell in ("late_void", "early_tie"):
        x = [r for r in out if r["cell"] == cell]
        print("==", cell)
        line("all", x)
        for name, key in (
            ("hand size", lambda r: len(r["cards"])),
            ("picker_rel", lambda r: r["prel"]),
            ("trick", lambda r: r["trick"]),
            ("ntrump", lambda r: r["ntrump"]),
            ("called rank", _rank_bucket),
            ("corpus alt class", lambda r: r["alt_cls"]),
        ):
            groups: dict[object, list[dict]] = collections.defaultdict(list)
            for r in x:
                groups[key(r)].append(r)
            for v in sorted(groups, key=str):
                line(f"{name} {v}", groups[v])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--run-dir", default=DEFAULT_RUN)
    ex.add_argument("--iterations", nargs="+", default=["iter1", "iter2"])
    ex.add_argument("--out", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--rows", required=True)
    nd = sub.add_parser("nodes")
    nd.add_argument("--rows", required=True)
    nd.add_argument(
        "--iteration",
        default="iter2",
        help="nodes index this iteration's targeted corpus",
    )
    nd.add_argument("--out", required=True)
    cf = sub.add_parser("confirm")
    cf.add_argument("--rows", required=True)
    cf.add_argument(
        "--results", required=True, help="research_corpus_nodes search --out JSONL"
    )
    cf.add_argument("--budget", default="4096")
    cf.add_argument("--iteration", default="iter2")
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        rows = extract(args.run_dir, args.iterations)
        with open(args.out, "wb") as f:
            pickle.dump(rows, f)
        print(f"{len(rows)} rows -> {args.out}")
        return 0
    rows = load_rows(args.rows)
    if args.cmd == "report":
        report(rows)
    elif args.cmd == "nodes":
        nodes = select_nodes(rows, args.iteration)
        with open(args.out, "w") as f:
            json.dump(nodes, f)
        print(
            f"{len(nodes)} nodes ({sum(n['cell'] == 'late_void' for n in nodes)} late_void) -> {args.out}"
        )
    else:
        confirm(rows, args.results, args.budget, args.iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
