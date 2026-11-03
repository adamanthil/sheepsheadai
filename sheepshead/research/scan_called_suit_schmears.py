#!/usr/bin/env python3
"""Scan simulated games for defenders schmearing into the hidden partner's
called-suit trick (notebooks/Called_Suit_Schmear_202611.md).

Called-ace mode. A CALLED-SUIT decision is a play where the called suit was
led, the called card is still out (partner hidden), and the defender to act
is void in the suit and holds a choice of point values. Rules decide the
trick for the seats after the actor: the picker must follow and the partner
must play the called card, so neither can trump. With no trump in the trick
and no defender left to act, the hidden partner wins for certain and every
point the actor adds goes to the picking team. With a defender's trump
already winning, the defenders win for certain and a schmear is right.

A VISIBLE-LOSS decision (both modes, the control) is a defender's play as
the last seat when the picker or the revealed partner is winning and no
legal card wins the trick.

Games are played argmax (the ``/analyze`` path, ``deterministic=True``), so
every recorded ``seed`` / ``stepIndex`` reproduces on the Analyze page with
the same checkpoint. Each decision also records the deployed probability of
an A/10 play under the manifest's deploy sampling.

Usage (from repo root):

    uv run python -m sheepshead.research.scan_called_suit_schmears \
        --model runs/202609_recall_rc/final/release.pt --num-seeds 40000 \
        --out runs/202609_recall_rc/schmear_probe/scan_theta6.jsonl
    uv run python -m sheepshead.research.scan_called_suit_schmears \
        --summary runs/202609_recall_rc/schmear_probe/scan_theta6.jsonl
    uv run python -m sheepshead.research.scan_called_suit_schmears \
        --trend theta_0=.../scan_theta0.jsonl theta_6=.../scan_theta6.jsonl \
        --max-seed 20000
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from sheepshead.agent.sampling import DeploySampling
from sheepshead.agent.traced_game import Decision, play_traced_game
from sheepshead.eval.fetch_models import load_manifest
from sheepshead.game import (
    CARD_POINTS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP_POWER,
    UNDER_TOKEN,
    get_card_suit,
    get_trick_winner,
)
from sheepshead.research.scan_defender_trump_leads import load_scan_agent

DEFAULT_MODEL = "runs/202609_recall_rc/final/release.pt"
DEPLOY_KEY = "recall-rc-theta6"

Row = dict[str, Any]


def deploy_sampler(key: str = DEPLOY_KEY) -> DeploySampling:
    """The manifest's deploy sampling for ``key`` (per-head T and band)."""
    s = load_manifest()["artifacts"][key]["sampling"]
    return DeploySampling(
        temperature={h: float(s[h]) for h in ("pick", "partner", "bury", "play")},
        band=float(s["band"]),
    )


def _record(ad: Decision, seed: int, mode: int, sampler: DeploySampling) -> Row | None:
    if not ad.action.startswith("PLAY "):
        return None
    v = ad.view
    picker = v.get("picker")
    if not picker or v.get("alone") or v.get("is_leaster") or ad.seat == picker:
        return None
    partner = v.get("partner") or 0
    hand = [c for c in v["hand"] if c]
    called = v.get("called_card")
    if ad.seat == partner or (called and called in hand):
        return None  # the partner, revealed or hidden
    if mode == PARTNER_BY_JD and "JD" in hand:
        return None
    ti = int(v["current_trick_index"])
    trick = v["current_trick"] or []
    leader = v["leaders"][ti]
    order = [((leader - 1 + k) % 5) + 1 for k in range(5)]
    played = [s for s in order if trick[s - 1]]
    if not played:
        return None
    n_after = 4 - len(played)
    led = trick[leader - 1]
    if led == UNDER_TOKEN:
        suit = get_card_suit(called)  # the under card is a called-suit card
    else:
        suit = "T" if led in TRUMP_POWER else get_card_suit(led)
    # The picker's under card is hidden; it never wins -> an off-suit 7.
    seq = [
        ("7S" if suit == "C" else "7C") if trick[s - 1] == UNDER_TOKEN else trick[s - 1]
        for s in played
    ]
    called10 = bool(
        called
        and called.startswith("10")
        and suit == get_card_suit(called)
        and not partner
    )
    winner = played[get_trick_winner(seq, suit, called10) - 1]
    inf = ad.inference
    legal_ids = sorted(inf.valid_actions)
    probs = {
        p.action[5:]: p.prob
        for p in inf.probabilities()
        if p.action.startswith("PLAY ")
    }
    legal = list(probs)
    pts = {c: CARD_POINTS.get(c, 0) for c in legal}
    min_pts = min(pts.values())
    if len(legal) < 2 or max(pts.values()) == min_pts:
        return None  # no point choice to make

    if (
        mode == PARTNER_BY_CALLED_ACE
        and called
        and not partner
        and suit != "T"
        and suit == get_card_suit(called)
        and n_after >= 1
        and not any(get_card_suit(c) == suit for c in hand if c not in TRUMP_POWER)
    ):
        kind = "called_suit"
    elif n_after == 0 and (winner == picker or (partner and winner == partner)):
        if any(get_trick_winner([*seq, c], suit, called10) == 5 for c in legal):
            return None
        kind = "visible_loss"
    else:
        return None

    log_probs = torch.log(inf.action_probs[0].detach().cpu().clamp(min=1e-30)).numpy()
    dep_full = sampler.distribution(log_probs, legal_ids)
    dep = {
        p.action[5:]: float(dep_full[p.action_id - 1])
        for p in inf.probabilities()
        if p.action.startswith("PLAY ")
    }
    argmax = max(probs, key=lambda c: probs[c])
    card = ad.action[5:]
    return {
        "kind": kind,
        "mode": mode,
        "seed": seed,
        "stepIndex": ad.step_index,
        "seat": ad.seat,
        "trick": ti,
        "n_after": n_after,
        # defenders still to act; on a called-suit decision the hidden
        # partner is always among the n_after seats, the picker may be
        "def_after": n_after - 1 - (picker not in played)
        if kind == "called_suit"
        else 0,
        "trump_in_trick": any(c in TRUMP_POWER for c in seq),
        "has_trump": any(c in TRUMP_POWER for c in hand),
        "led": led,
        "winner": trick[winner - 1],
        "hand": hand,
        "legal": legal,
        "min_pts": min_pts,
        "played": card,
        "argmax": argmax,
        "probs": probs,
        "p_dep_a10": sum(dep[c] for c in legal if pts[c] >= 10),
    }


def _scan_chunk(args: tuple[list[int], int, str]) -> list[Row]:
    seeds, mode, model = args
    torch.set_num_threads(1)
    agent = load_scan_agent(model)
    sampler = deploy_sampler()
    out: list[Row] = []
    for seed in seeds:
        traced = play_traced_game(
            agent, seed=seed, partner_mode=mode, deterministic=True
        )
        for ad in traced.decisions:
            r = _record(ad, seed, mode, sampler)
            if r is not None:
                out.append(r)
        out.append({"kind": "_game", "seed": seed, "mode": mode})
    return out


def scan(
    model: str, seed_start: int, num_seeds: int, mode: int, workers: int, out: str
) -> None:
    """Write one JSON line per decision plus one ``_game`` line per deal."""
    seeds = list(range(seed_start, seed_start + num_seeds))
    n_chunks = workers * 4
    chunks = [(seeds[i::n_chunks], mode, model) for i in range(n_chunks)]
    with mp.get_context("spawn").Pool(workers) as pool, open(out, "w") as f:
        for i, rows in enumerate(pool.imap_unordered(_scan_chunk, chunks)):
            for r in sorted(rows, key=lambda r: (r["seed"], r.get("stepIndex", -1))):
                f.write(json.dumps(r) + "\n")
            print(f"chunk {i + 1}/{n_chunks}", file=sys.stderr, flush=True)


def _load(path: str, max_seed: int | None = None) -> tuple[list[Row], int]:
    rows = [json.loads(line) for line in open(path)]
    if max_seed is not None:
        rows = [r for r in rows if r["seed"] < max_seed]
    games = sum(r["kind"] == "_game" for r in rows)
    return [r for r in rows if r["kind"] != "_game"], games


def _a10(c: str) -> bool:
    return CARD_POINTS.get(c, 0) >= 10


def schmear_choice(r: Row) -> bool:
    """Called-suit decision with no trump in hand, holding an A/10 fail and a
    cheaper fail: the population whose A/10 play the cells measure."""
    return (
        r["kind"] == "called_suit"
        and not r["has_trump"]
        and any(_a10(c) for c in r["legal"])
        and r["min_pts"] < 10
    )


def pure_fail_discard(r: Row) -> bool:
    """Visible-loss decision whose legal set is all fail, holding an A/10 and
    a 0-point fail."""
    return (
        r["kind"] == "visible_loss"
        and not any(c in TRUMP_POWER for c in r["legal"])
        and r["min_pts"] == 0
        and any(_a10(c) for c in r["legal"])
    )


def _cell(rows: list[Row]) -> str:
    n = len(rows)
    if not n:
        return f"{'n/a':>28s}"
    k = sum(_a10(r["argmax"]) for r in rows)
    se = 100 * np.sqrt(max(k / n * (1 - k / n), 1e-12) / n)
    dep = 100 * float(np.mean([r["p_dep_a10"] for r in rows]))
    return f"{100 * k / n:5.1f} ± {se:3.1f} ({dep:4.1f}) n {n:4d}"


CELLS: list[tuple[str, Callable[[Row], bool]]] = [
    (
        "WRONG: no trump in, 0 defenders left",
        lambda r: schmear_choice(r) and not r["trump_in_trick"] and r["def_after"] == 0,
    ),
    (
        "unclear: no trump in, >= 1 defender left",
        lambda r: schmear_choice(r) and not r["trump_in_trick"] and r["def_after"] >= 1,
    ),
    (
        "RIGHT: defender trump winning, 0 left",
        lambda r: schmear_choice(r) and r["trump_in_trick"] and r["def_after"] == 0,
    ),
    ("control: visible loss, pure-fail discard", pure_fail_discard),
]


def summary(path: str) -> None:
    """Incidence tables of one scan: argmax A/10 % ± SE (deployed %) n."""
    rows, games = _load(path)
    print(f"{path}: {games} deals; cells = argmax A/10 % ± SE (deployed %) n")
    for label, keep in CELLS:
        print(f"  {label:42s} {_cell([r for r in rows if keep(r)])}")

    def no_trump(r: Row) -> bool:
        return schmear_choice(r) and not r["trump_in_trick"]

    print("\nno trump in the trick, by defenders still to act and trick:")
    for t in [None, *range(5)]:
        for d in (0, 1, 2):
            x = [
                r
                for r in rows
                if no_trump(r) and r["def_after"] == d and t in (None, r["trick"])
            ]
            if len(x) >= 10:
                where = "all tricks" if t is None else f"trick {t}"
                print(f"  {where:10s} defenders left {d}  {_cell(x)}")
    wrong = [r for r in rows if CELLS[0][1](r)]
    print(
        "\nWRONG-cell A/10 cards (argmax):",
        collections.Counter(
            r["argmax"] for r in wrong if _a10(r["argmax"])
        ).most_common(),
    )
    print("WRONG-cell examples (Analyze: Called Ace, Deterministic):")
    for r in [r for r in wrong if _a10(r["argmax"])][:8]:
        top = sorted(r["probs"].items(), key=lambda kv: -kv[1])[:3]
        print(
            f"  seed {r['seed']} step {r['stepIndex']} trick {r['trick']} led {r['led']}"
            f" hand {r['hand']} -> {r['argmax']} {[(c, round(p, 3)) for c, p in top]}"
        )


def trend(specs: list[str], max_seed: int | None) -> None:
    """One row per ``label=path`` scan, the CELLS columns side by side."""
    print(
        "cells = argmax A/10 % ± SE (deployed %) n"
        + (f"; seeds < {max_seed}" if max_seed else "")
    )
    print(f"{'':10s}" + "".join(f"{label[:28]:30s}" for label, _ in CELLS))
    for spec in specs:
        label, path = spec.split("=", 1)
        rows, _ = _load(path, max_seed)
        print(
            f"{label:10s}"
            + "".join(f"{_cell([r for r in rows if k(r)]):30s}" for _, k in CELLS)
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint to scan")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=20000)
    ap.add_argument(
        "--partner-mode",
        type=int,
        default=PARTNER_BY_CALLED_ACE,
        help=f"{PARTNER_BY_CALLED_ACE} called-ace, {PARTNER_BY_JD} JD",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="JSONL of decisions (one per line)")
    ap.add_argument("--summary", metavar="SCAN_JSONL", help="tables of one scan")
    ap.add_argument("--trend", nargs="+", metavar="LABEL=SCAN_JSONL")
    ap.add_argument("--max-seed", type=int, help="trend: only seeds below this")
    args = ap.parse_args(argv)
    if args.summary:
        summary(args.summary)
        return 0
    if args.trend:
        trend(args.trend, args.max_seed)
        return 0
    if not args.out:
        ap.error("--out is required to scan")
    scan(
        args.model,
        args.seed_start,
        args.num_seeds,
        args.partner_mode,
        args.workers,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
