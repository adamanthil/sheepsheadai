#!/usr/bin/env python3
"""Scan simulated games for defenders over-trumping a defender on the called
suit (Arm_D_Review §9.2).

Called-ace mode. A DECISION is a play where the called suit was led, a
defender (not the picker, not the revealed partner) is winning the trick with
a trump, and the defender to act is void in the suit, holds a trump that beats
the winner AND has another legal play (a fail, or a lower trump). Rules make
the trick safe: the picker must follow the called suit and the partner must
play the ace, so only another defender can take it. The rate is how often
the policy's chosen card is an over-trump.

The scanner drives the same deterministic simulation path as ``/analyze``
(``server.services.analyze.simulate_game``), so every recorded ``seed`` /
``stepIndex`` reproduces on the Analyze page (Partner Mode = Called Ace,
Deterministic on, the same checkpoint loaded).

Usage (from repo root):

    uv run python -m sheepshead.analysis.scan_called_suit_overtrumps \
        --model runs/202609_recall_rc/pi/iter3/distill_epoch10.pt \
        --num-seeds 4000 --out runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl
    uv run python -m sheepshead.analysis.scan_called_suit_overtrumps \
        --summary runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from server.api.schemas import AnalyzeSimulateRequest
from server.services.analyze import simulate_game
from sheepshead.analysis.scan_defender_trump_leads import DEFAULT_MODEL, load_scan_agent
from sheepshead.game import CARD_POINTS, TRUMP_POWER, get_card_suit, get_trick_winner

PARTNER_MODE_CALLED_ACE = 1


def scan(model: str, seed_start: int, num_seeds: int, out: str) -> dict:
    """Write one JSON line per decision; return the game counts (the rate's
    per-hand denominator)."""
    agent = load_scan_agent(model)
    games = std_games = 0
    t0 = time.time()
    with open(out, "w") as f:
        for seed in range(seed_start, seed_start + num_seeds):
            resp = simulate_game(
                AnalyzeSimulateRequest(
                    seed=seed, partnerMode=PARTNER_MODE_CALLED_ACE, deterministic=True
                ),
                agent=agent,
            )
            games += 1
            final = resp.final or {}
            if final.get("mode") == "standard" and not resp.trace[-1].view.get("alone"):
                std_games += 1
            for ad in resp.trace:
                rec = _decision(ad, final, seed)
                if rec is not None:
                    f.write(json.dumps(rec, default=str) + "\n")
            if (seed - seed_start) % 500 == 499:
                print(seed + 1, f"{time.time() - t0:.0f}s", flush=True)
    counts = {"games": games, "standard_called_games": std_games}
    with open(out + ".counts.json", "w") as f:
        json.dump(counts, f)
    print(json.dumps(counts))
    return counts


def _decision(ad, final: dict, seed: int) -> dict | None:
    if not ad.action.startswith("PLAY "):
        return None
    v = ad.view
    called, picker = v.get("called_card"), v.get("picker")
    if not called or v.get("alone") or v.get("is_leaster") or ad.seat == picker:
        return None
    ti = int(v["current_trick_index"])
    trick = v["current_trick"] or []
    leader = v["leaders"][ti]
    order = [((leader - 1 + k) % 5) + 1 for k in range(5)]
    played = [s for s in order if trick[s - 1]]
    if not played:
        return None
    cs = get_card_suit(called)
    led = trick[leader - 1]
    if led in TRUMP_POWER or get_card_suit(led) != cs:
        return None
    seq = [trick[s - 1] for s in played]
    w = played[get_trick_winner(seq, cs) - 1]
    wcard = trick[w - 1]
    if wcard not in TRUMP_POWER or w == picker or w == v.get("partner"):
        return None
    hand = v["hand"]
    if any(c not in TRUMP_POWER and get_card_suit(c) == cs for c in hand):
        return None  # must follow the called suit: no choice
    probs = {
        p.action[5:]: p.prob for p in ad.probabilities if p.action.startswith("PLAY ")
    }
    over = [
        c for c in probs if c in TRUMP_POWER and TRUMP_POWER[c] > TRUMP_POWER[wcard]
    ]
    other = [c for c in probs if c not in over]
    if not over or not other:
        return None
    card = ad.action[5:]
    fails = [c for c in hand if c not in TRUMP_POWER]
    return {
        "seed": seed,
        "stepIndex": ad.stepIndex,
        "seat": ad.seat,
        "seatName": ad.seatName,
        "trick": ti,
        "winner": wcard,
        "winner_seat": w,
        "picker": picker,
        "n_after": 5 - len(played) - 1,
        "hand": hand,
        "played": card,
        "overtrumped": card in over,
        "p_over": sum(probs[c] for c in over),
        "best_other": max(other, key=lambda c: probs[c]),
        "max_fail_pts": max((CARD_POINTS.get(c, 0) for c in fails), default=-1),
        "final": final,
    }


def summary(path: str) -> None:
    """§9.2 incidence tables from a scan file (and its .counts.json)."""
    rows = [json.loads(line) for line in open(path)]
    try:
        with open(path + ".counts.json") as f:
            std_games = json.load(f)["standard_called_games"]
    except FileNotFoundError:
        std_games = None
    n = len(rows)
    k = sum(r["overtrumped"] for r in rows)

    def se(k: int, n: int) -> float:
        return float(np.sqrt(k / n * (1 - k / n) / n))

    tricks: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for r in rows:
        tricks[(r["seed"], r["trick"])].append(r)
    kt = sum(any(x["overtrumped"] for x in v) for v in tricks.values())
    print(
        f"decisions {n} in {len(tricks)} tricks (decisions per trick {dict(collections.Counter(len(v) for v in tricks.values()))})"
    )
    if std_games:
        print(
            f"standard called-ace games {std_games}: {100 * n / std_games:.1f} decisions and {100 * k / std_games:.1f} over-trumps per 100"
        )
    print(
        f"over-trumped {k}/{n} = {100 * k / n:.1f}% ± {100 * se(k, n):.1f}   mean p(over) {np.mean([r['p_over'] for r in rows]):.2f}"
    )
    print(
        f"tricks with >= 1 over-trump {kt}/{len(tricks)} = {100 * kt / len(tricks):.1f}%"
    )

    def cell(label: str, keep) -> None:
        x = [r for r in rows if keep(r)]
        kk = sum(r["overtrumped"] for r in x)
        if x:
            print(
                f"  {label:42s} n {len(x):4d}  over-trumped {100 * kk / len(x):5.1f}% ± {100 * se(kk, len(x)):.1f}  p(over) {np.mean([r['p_over'] for r in x]):.2f}"
            )

    has_fail = [r for r in rows if r["max_fail_pts"] >= 0]
    clear = [
        r for r in has_fail if r["overtrumped"] and r["played"] not in ("AD", "10D")
    ]
    print(
        f"  {'has a fail; over-trump not AD/10D (clear)':42s} n {len(has_fail):4d}  {100 * len(clear) / len(has_fail):.1f}%"
    )
    cell("holds 10/A fail (schmear)", lambda r: r["max_fail_pts"] >= 10)
    cell("holds K, no 10/A", lambda r: 4 <= r["max_fail_pts"] < 10)
    cell("only low fail", lambda r: 0 <= r["max_fail_pts"] < 4)
    cell("no fail (only trump choices)", lambda r: r["max_fail_pts"] < 0)
    cell(
        "winner is a diamond",
        lambda r: r["winner"][-1] == "D" and r["winner"][0] not in "QJ",
    )
    cell("winner is a jack", lambda r: r["winner"][0] == "J")
    cell("winner is a queen", lambda r: r["winner"][0] == "Q")
    cell(
        "10D winning, has a fail",
        lambda r: r["winner"] == "10D" and r["max_fail_pts"] >= 0,
    )
    for a in range(4):
        cell(f"{a} players after me", lambda r, a=a: r["n_after"] == a)
    for t in range(5):
        cell(f"trick {t}", lambda r, t=t: r["trick"] == t)
    print(
        "over-trump cards:",
        collections.Counter(
            r["played"] for r in rows if r["overtrumped"]
        ).most_common(),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint to scan")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=4000)
    ap.add_argument("--out", help="JSONL of decisions (one per line)")
    ap.add_argument(
        "--summary", metavar="SCAN_JSONL", help="print the incidence tables of a scan"
    )
    args = ap.parse_args(argv)
    if args.summary:
        summary(args.summary)
        return 0
    if not args.out:
        ap.error("--out is required to scan")
    scan(args.model, args.seed_start, args.num_seeds, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
