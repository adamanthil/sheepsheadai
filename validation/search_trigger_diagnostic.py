#!/usr/bin/env python3
"""Selective-search trigger diagnostic: where should deploy-time ISMCTS fire?

The value-add probe established that decision-time search is worth ~+0.10
pts/deal at 384 iters but deviates from the raw policy at only ~2.4% of PLAY
decisions — i.e. almost every search just confirms the policy at a flat 4-10
s/decision cost. This diagnostic measures whether a CHEAP, policy-only trigger
(top-1 margin, entropy, trick index, forced-move filter) can concentrate the
search budget on the decisions that actually move EV, so the deployed agent
plays the policy instantly on confident nodes and only pays for search on the
genuinely uncertain ones (the human pattern).

Method: one greedy pass per paired-style deal. The probe seat plays its raw
greedy POLICY everywhere (the no-search deployment trajectory). At every PLAY
decision it ALSO runs the production ISMCTS teacher and records, per node:

  - policy features (no search needed at deploy): top-1 margin (p1 - p2),
    entropy over legal actions, top-1 prob, trick index, #legal actions;
  - whether search's argmax DEVIATES from the policy argmax;
  - the value of that deviation by search's OWN root Q: q_gap =
    RETURN_SCALE * (root_q[search_argmax] - root_q[policy_argmax]). This is
    search's self-assessed EV of following it instead of the policy at this
    node, in card-points; it is 0 when search agrees with the policy. Summed
    over a node-class it is the marginal EV of searching that class. (Search's
    self-assessment is cross-validated at the aggregate by the value-add
    probe's realized game-level delta; this gives the dense per-node signal a
    paired realized-delta probe cannot.)

Offline it sweeps each trigger and reports the curve: trigger rate (= fraction
of PLAY decisions searched = the latency cost) vs fraction of total q_gap EV
captured. The headline is the smallest trigger rate that keeps >=80/90% of the
EV, the policy-feature threshold that achieves it, and the implied average
latency. Forced moves (one legal action) are reported as a free skip.

Usage:
  PYTHONPATH=. .venv/bin/python validation/search_trigger_diagnostic.py \
      [-m final_pfsp_swish_ppo.pt] [--deals 150] [--iters-play 384] \
      [--out /tmp/search_trigger_rows.csv]
  # re-analyze a saved run without re-searching:
  PYTHONPATH=. .venv/bin/python validation/search_trigger_diagnostic.py \
      --analyze-only /tmp/search_trigger_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
import torch

from ismcts import ISMCTSConfig, ISMCTSTeacher
from ppo import PPOAgent
from sheepshead import ACTIONS, Game
from training_utils import RETURN_SCALE, get_partner_selection_mode

T_FULL = 1  # production rollout-depth schedule (config.SearchConfig)
D_SHORT = 2

FIELDS = [
    "deal",
    "trick",
    "n_valid",
    "margin",
    "entropy",
    "top1p",
    "searched",
    "deviated",
    "q_gap",
    "ess",
    "ok",
]


def _is_private(valid) -> bool:
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def _load(model: str) -> PPOAgent:
    a = PPOAgent(len(ACTIONS))
    a.load(model, load_optimizers=False)
    return a


def _policy_features(
    probs: np.ndarray, valid: list[int]
) -> tuple[int, float, float, float]:
    """(policy_argmax, margin=p1-p2, entropy over legal, top1 prob)."""
    ps = np.array([probs[a - 1] for a in valid], dtype=np.float64)
    order = np.argsort(ps)[::-1]
    pol_arg = valid[order[0]]
    p1 = float(ps[order[0]])
    p2 = float(ps[order[1]]) if len(ps) > 1 else 0.0
    s = ps.sum()
    pn = ps / s if s > 0 else ps
    nz = pn[pn > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return pol_arg, p1 - p2, entropy, p1


def _run_deal(deal_seed, mode, seat, agent, field, teacher, det_rng, rows, deal_idx):
    """Play one deal; probe seat plays POLICY-GREEDY everywhere; at each probe
    PLAY decision run search and append a feature row. Returns probe-seat score."""
    game = Game(partner_selection_mode=mode, seed=deal_seed)
    agent.reset_recurrent_state()
    field.reset_recurrent_state()
    forced_public = []

    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                is_probe = player.position == seat
                is_play = game.play_started and all(
                    ACTIONS[a - 1].startswith("PLAY ") for a in valid
                )
                if is_probe and is_play:
                    vlist = list(valid)
                    # Policy distribution (this forward advances the probe's
                    # recurrent memory exactly once — do NOT also call act()).
                    probs_t, _ = agent.get_action_probs_with_logits(
                        player.get_state_dict(), valid, player_id=player.position
                    )
                    probs = probs_t[0].detach().cpu().numpy()
                    pol_arg, margin, entropy, top1p = _policy_features(probs, vlist)
                    row = {
                        "deal": deal_idx,
                        "trick": game.current_trick,
                        "n_valid": len(valid),
                        "margin": margin,
                        "entropy": entropy,
                        "top1p": top1p,
                        "searched": 0,
                        "deviated": 0,
                        "q_gap": 0.0,
                        "ess": 0.0,
                        "ok": 0,
                    }
                    if len(valid) >= 2:
                        dr = (
                            (6 - game.current_trick)
                            if game.current_trick <= T_FULL
                            else D_SHORT
                        )
                        res = teacher.search(
                            game, seat, list(forced_public), det_rng, d_rollout=dr
                        )
                        row["searched"] = 1
                        row["ess"] = float(res["ess"])
                        if res["ok"]:
                            pi = res["pi"]
                            search_arg = max(valid, key=lambda a: pi[a - 1])
                            rq = res["root_q"]
                            row["ok"] = 1
                            row["deviated"] = int(search_arg != pol_arg)
                            row["q_gap"] = RETURN_SCALE * (
                                rq.get(search_arg, 0.0) - rq.get(pol_arg, 0.0)
                            )
                    rows.append(row)
                    a = pol_arg  # policy-greedy baseline trajectory
                else:
                    ag = agent if is_probe else field
                    a, _, _ = ag.act(
                        player.get_state_dict(),
                        valid,
                        player.position,
                        deterministic=True,
                    )
                if not _is_private(valid):
                    forced_public.append((player.position, a))
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for p in game.players:
                        ctrl = agent if p.position == seat else field
                        ctrl.observe(
                            p.get_last_trick_state_dict(), player_id=p.position
                        )

    return float(game.players[seat - 1].get_score())


# ---------------------------------------------------------------------------
# Offline analysis
# ---------------------------------------------------------------------------
def _curve(rows, key, ascending, total_q, n_play, budgets):
    """For trigger 'search the nodes with the most-uncertain `key` first',
    report (trigger_rate, ev_fraction, threshold) at each budget. ascending=True
    means low key = uncertain (margin/top1p); False means high key = uncertain
    (entropy)."""
    unforced = [r for r in rows if r["n_valid"] >= 2]
    vals = np.array([r[key] for r in unforced])
    qs = np.array([r["q_gap"] for r in unforced])
    order = np.argsort(vals)
    if not ascending:
        order = order[::-1]
    qs_sorted = qs[order]
    vals_sorted = vals[order]
    out = []
    for b in budgets:
        k = max(1, int(round(b * n_play)))  # k nodes searched (of ALL play nodes)
        k = min(k, len(unforced))
        ev = qs_sorted[:k].sum() / total_q if total_q else 0.0
        thr = vals_sorted[k - 1] if k <= len(vals_sorted) else vals_sorted[-1]
        out.append((k / n_play, ev, thr))
    return out


def analyze(rows, iters_play, elapsed=None):
    play = rows
    n_play = len(play)
    forced = [r for r in play if r["n_valid"] < 2]
    unforced = [r for r in play if r["n_valid"] >= 2]
    searched = [r for r in unforced if r["searched"]]
    dev = [r for r in searched if r["deviated"]]
    total_q = sum(r["q_gap"] for r in play)
    n_deals = len({r["deal"] for r in play})

    print()
    print("=" * 74)
    print(f"SELECTIVE-SEARCH TRIGGER DIAGNOSTIC  (iters_play={iters_play})")
    print("=" * 74)
    print(
        f"  deals {n_deals} | PLAY decisions {n_play} ({n_play / max(n_deals, 1):.1f}/deal)"
        + (f" | {elapsed:.0f}s" if elapsed else "")
    )
    print(
        f"  forced (1 legal, free skip): {len(forced)} "
        f"({100 * len(forced) / max(n_play, 1):.1f}%)  ->  "
        f"searchable nodes: {len(unforced)} ({100 * len(unforced) / max(n_play, 1):.1f}%)"
    )
    print(
        f"  deviations: {len(dev)} "
        f"({100 * len(dev) / max(len(searched), 1):.1f}% of searched)  |  "
        f"total q_gap EV: {total_q / max(n_deals, 1):+.4f} pts/deal "
        f"(search self-assessed)"
    )
    if total_q <= 0:
        print(
            "  total q_gap <= 0: no positive search EV to concentrate at this budget."
        )
        return

    budgets = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

    def _print_curve(title, key, ascending, unit):
        print(f"\n  {title}")
        print(f"    {'trigger%':>9}  {'EV kept':>8}  {'threshold':>12}")
        for tr, ev, thr in _curve(rows, key, ascending, total_q, n_play, budgets):
            print(f"    {100 * tr:8.1f}%  {100 * ev:7.1f}%  {thr:12.3f} {unit}")

    _print_curve(
        "Trigger: search lowest-MARGIN nodes first (search iff margin < threshold)",
        "margin",
        True,
        "(p1-p2)",
    )
    _print_curve(
        "Trigger: search highest-ENTROPY nodes first (search iff entropy > threshold)",
        "entropy",
        False,
        "nats",
    )

    # By trick: where does the EV live?
    print("\n  EV by trick index (q_gap pts/deal, deviation rate):")
    print(f"    {'trick':>5}  {'nodes':>6}  {'q_gap/deal':>11}  {'dev%':>6}")
    by_trick = {}
    for r in play:
        by_trick.setdefault(r["trick"], []).append(r)
    for t in sorted(by_trick):
        rs = by_trick[t]
        sr = [r for r in rs if r["searched"]]
        qd = sum(r["q_gap"] for r in rs) / max(n_deals, 1)
        dv = 100 * sum(r["deviated"] for r in sr) / max(len(sr), 1)
        print(f"    {t:5d}  {len(rs):6d}  {qd:+11.4f}  {dv:6.1f}")

    # Headline: smallest margin-trigger budget keeping >= 80% / 90% EV.
    print("\n  Headline (margin trigger):")
    curve = _curve(rows, "margin", True, total_q, n_play, np.linspace(0.01, 1.0, 100))
    for target in (0.80, 0.90, 0.95):
        hit = next((c for c in curve if c[1] >= target), curve[-1])
        tr, ev, thr = hit
        # crude latency model: searched nodes cost a search, the rest are instant
        print(
            f"    keep >={100 * target:.0f}% EV  ->  search {100 * tr:.1f}% of PLAY "
            f"decisions (margin < {thr:.3f}); ~{tr:.2f}x the flat-search latency"
        )


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append(
                {
                    "deal": int(r["deal"]),
                    "trick": int(r["trick"]),
                    "n_valid": int(r["n_valid"]),
                    "margin": float(r["margin"]),
                    "entropy": float(r["entropy"]),
                    "top1p": float(r["top1p"]),
                    "searched": int(r["searched"]),
                    "deviated": int(r["deviated"]),
                    "q_gap": float(r["q_gap"]),
                    "ess": float(r["ess"]),
                    "ok": int(r["ok"]),
                }
            )
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="final_pfsp_swish_ppo.pt")
    ap.add_argument("--deals", type=int, default=150)
    ap.add_argument("--iters-play", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/search_trigger_rows.csv")
    ap.add_argument(
        "--analyze-only",
        default=None,
        help="path to a saved rows CSV; analyze it without re-searching",
    )
    args = ap.parse_args()

    if args.analyze_only:
        rows = _read_csv(args.analyze_only)
        analyze(rows, args.iters_play)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model} ...  play-head search iters: {args.iters_play}")
    agent = _load(args.model)
    field = _load(args.model)
    cfg = ISMCTSConfig()
    cfg.iters = dict(cfg.iters, play=args.iters_play)
    teacher = ISMCTSTeacher(agent, cfg)
    det_rng = random.Random(args.seed + 1)

    rows: list[dict] = []
    t0 = time.time()
    for d in range(args.deals):
        mode = get_partner_selection_mode(d)
        seat = (d % 5) + 1
        deal_seed = args.seed * 1_000_003 + d
        _run_deal(deal_seed, mode, seat, agent, field, teacher, det_rng, rows, d)
        if (d + 1) % 10 == 0:
            _write_csv(args.out, rows)  # checkpoint so a long run is inspectable
            tq = sum(r["q_gap"] for r in rows) / (d + 1)
            print(
                f"  {d + 1}/{args.deals} deals ({time.time() - t0:.0f}s)  "
                f"{len(rows)} nodes  running q_gap {tq:+.4f} pts/deal",
                flush=True,
            )

    _write_csv(args.out, rows)
    print(f"\nrows written to {args.out}")
    analyze(rows, args.iters_play, elapsed=time.time() - t0)


if __name__ == "__main__":
    main()
