#!/usr/bin/env python3
"""Paired read of several checkpoints' policies on the SAME frozen decision
states (Arm_D_Review §9.8 primary; the E7 frozen-trajectory design).

A driver checkpoint plays each listed seed deterministically through the
/analyze path (``server.services.analyze.simulate_game``); its trace fixes
every action. Each probe checkpoint then replays that identical trajectory
through the same inference path (``run_inference_step``, all five seats
observing after each completed trick), so it rebuilds its OWN recurrent
memory over the same prefix, and reports its policy at the listed nodes.
The probes' choices never steer the game. Differences between probes are
therefore differences between networks on identical states, free of the
deal and trajectory variance a fresh-deal scan carries.

The node list is a scan file with ``seed`` and ``stepIndex`` per line (the
over-trump scan, ``scan_called_suit_overtrumps``). ``read`` writes each
probe's legal-action probabilities per node; ``overtrump`` summarises the
called-suit over-trump decision: the mean per-node difference in
over-trump probability between two probes with a one-sided 95% bootstrap
CI (resampled by seed), argmax over-trump rates with McNemar's test, and
the clear-blunder and schmear subsets. Replaying the driver as a probe
reproduces the scan's own probabilities, the built-in check.

Usage (from repo root):

    uv run python -m sheepshead.research.frozen_node_policy_read read \
        --scan runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl \
        --driver runs/202609_recall_rc/pi/iter3/distill_epoch10.pt \
        --probe base=runs/202609_recall_rc/pi/iter3/distill_epoch10.pt \
        --probe arm=runs/202609_recall_rc/pi/iter3_ablations/onesided_head/distill_epoch10.pt \
        --out runs/202609_recall_rc/pi/iter3_ablations/onesided_head/frozen_read.json
    uv run python -m sheepshead.research.frozen_node_policy_read overtrump \
        --read runs/202609_recall_rc/pi/iter3_ablations/onesided_head/frozen_read.json \
        --base base --arm arm
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from server.api.schemas import AnalyzeSimulateRequest
from server.runtime.seating import ANALYZE_SEAT_NAMES
from server.services.analysis_common import find_actor, run_inference_step, set_seed
from server.services.analyze import simulate_game
from sheepshead.agent.observation import last_trick_observation_for
from sheepshead.game import ACTIONS, CARD_POINTS, TRUMP_POWER, Game
from sheepshead.research.scan_defender_trump_leads import load_scan_agent

# Over-trumping with these is break-even (the trump carries 10-11 points
# itself, Arm_D_Review §9.3), so the clear-blunder subset excludes them.
POINT_TRUMPS = ("AD", "10D")


def driver_actions(agent: Any, seed: int, partner_mode: int) -> list[int]:
    """The driver's deterministic /analyze trace for ``seed`` as action ids."""
    resp = simulate_game(
        AnalyzeSimulateRequest(seed=seed, partnerMode=partner_mode, deterministic=True),
        agent=agent,
    )
    return [ACTIONS.index(ad.action) + 1 for ad in resp.trace]


def replay_probe(
    agent: Any, seed: int, partner_mode: int, actions: list[int], steps: set[int]
) -> dict[int, dict[str, float]]:
    """Replay ``actions`` with ``agent`` building its own memory; return its
    legal-action probabilities (card/action name -> p) at ``steps``."""
    set_seed(seed)
    agent.reset_recurrent_state()
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    device = torch.device("cpu")
    out: dict[int, dict[str, float]] = {}
    for step, action_id in enumerate(actions):
        actor = find_actor(game)
        if actor is None:
            raise RuntimeError(f"seed {seed}: no actor at step {step}")
        inference = run_inference_step(
            agent, actor, actor.position, list(ANALYZE_SEAT_NAMES), device
        )
        if action_id not in inference.valid_actions:
            raise RuntimeError(f"seed {seed} step {step}: trace action is illegal")
        if step in steps:
            probs = inference.action_probs.squeeze().cpu()
            out[step] = {
                ACTIONS[a - 1]: float(probs[a - 1]) for a in inference.valid_actions
            }
        actor.act(action_id)
        if game.was_trick_just_completed:
            for seat in game.players:
                agent.observe(
                    last_trick_observation_for(seat, agent), player_id=seat.position
                )
    missing = steps - out.keys()
    if missing:
        raise RuntimeError(f"seed {seed}: steps {sorted(missing)} never reached")
    return out


def cmd_read(args: argparse.Namespace) -> None:
    nodes = [json.loads(line) for line in open(args.scan)]
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for n in nodes:
        by_seed[int(n["seed"])].append(n)
    driver = load_scan_agent(args.driver)
    probes = dict(p.split("=", 1) for p in args.probe)
    agents = {name: load_scan_agent(path) for name, path in probes.items()}
    rows: list[dict] = []
    for i, seed in enumerate(sorted(by_seed)):
        actions = driver_actions(driver, seed, args.partner_mode)
        steps = {int(n["stepIndex"]) for n in by_seed[seed]}
        reads = {
            name: replay_probe(agent, seed, args.partner_mode, actions, steps)
            for name, agent in agents.items()
        }
        for n in by_seed[seed]:
            step = int(n["stepIndex"])
            rows.append({**n, "probs": {name: r[step] for name, r in reads.items()}})
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(by_seed)}] seeds", flush=True)
    with open(args.out, "w") as f:
        json.dump({"driver": args.driver, "probes": probes, "nodes": rows}, f)
    print(f"{len(rows)} nodes from {len(by_seed)} seeds -> {args.out}")


def _over_cards(node: dict, probs: dict[str, float]) -> list[str]:
    winner = node["winner"]
    return [
        a[5:]
        for a in probs
        if a.startswith("PLAY ")
        and a[5:] in TRUMP_POWER
        and TRUMP_POWER[a[5:]] > TRUMP_POWER[winner]
    ]


def _p_over(node: dict, probs: dict[str, float], clear: bool) -> float:
    cards = _over_cards(node, probs)
    if clear:
        cards = [c for c in cards if c not in POINT_TRUMPS]
    return sum(probs[f"PLAY {c}"] for c in cards)


def _argmax_over(node: dict, probs: dict[str, float], clear: bool) -> bool:
    top = max(probs, key=lambda a: probs[a])[5:]
    return top in _over_cards(node, probs) and not (clear and top in POINT_TRUMPS)


def _bootstrap_upper(d: np.ndarray, seeds: np.ndarray, reps: int, rng) -> float:
    """One-sided 95% upper bound of mean(d), resampling whole seeds."""
    uniq = np.unique(seeds)
    groups = [d[seeds == s] for s in uniq]
    means = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        means[r] = np.concatenate([groups[k] for k in pick]).mean()
    return float(np.quantile(means, 0.95))


def _mcnemar_one_sided(b_only: int, a_only: int) -> float:
    """Exact one-sided p that the arm over-trumps LESS: among discordant
    pairs, P(X <= arm-only) under Binomial(n, 1/2)."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(a_only + 1)) / 2**n


def cmd_overtrump(args: argparse.Namespace) -> None:
    data = json.load(open(args.read))
    nodes = data["nodes"]
    rng = np.random.default_rng(args.seed)
    subsets = {
        "all decisions": lambda n: True,
        "clear blunder (has a fail; not AD/10D)": lambda n: n["max_fail_pts"] >= 0,
        "holding a schmear (10/A fail)": lambda n: n["max_fail_pts"] >= 10,
    }
    # Check: a probe that IS the driver reproduces the scan's own p(over).
    for name, path in data["probes"].items():
        if path == data["driver"]:
            err = max(
                abs(_p_over(n, n["probs"][name], False) - n["p_over"]) for n in nodes
            )
            print(f"check: probe {name} = driver; max |p(over) - scan| {err:.2e}")
    for label, keep in subsets.items():
        sel = [n for n in nodes if keep(n)]
        clear = label.startswith("clear")
        pa = np.array([_p_over(n, n["probs"][args.base], clear) for n in sel])
        pb = np.array([_p_over(n, n["probs"][args.arm], clear) for n in sel])
        d = pb - pa
        seeds = np.array([int(n["seed"]) for n in sel])
        upper = _bootstrap_upper(d, seeds, args.reps, rng)
        ma = np.array([_argmax_over(n, n["probs"][args.base], clear) for n in sel])
        mb = np.array([_argmax_over(n, n["probs"][args.arm], clear) for n in sel])
        b_only = int((ma & ~mb).sum())
        a_only = int((mb & ~ma).sum())
        print(f"\n== {label} (n {len(sel)}, {len(np.unique(seeds))} seeds)")
        print(
            f"   p(over-trump): {args.base} {pa.mean():.4f}  {args.arm} {pb.mean():.4f}  "
            f"diff {d.mean():+.4f} (one-sided 95% upper {upper:+.4f}) -> "
            f"{'PASS' if upper < 0 else 'not significant'}"
        )
        print(
            f"   argmax over-trump: {args.base} {100 * ma.mean():.1f}%  {args.arm} "
            f"{100 * mb.mean():.1f}%  (stopped {b_only}, started {a_only}; McNemar "
            f"one-sided p {_mcnemar_one_sided(b_only, a_only):.3f})"
        )
    fails = [n for n in nodes if n["max_fail_pts"] >= 0]
    pts = np.array(
        [
            max(CARD_POINTS.get(c, 0) for c in n["hand"] if c not in TRUMP_POWER)
            for n in fails
        ]
    )
    print(
        f"\n(fail-holding nodes {len(fails)}; mean best fail points {pts.mean():.1f})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="replay the scan's nodes through each probe")
    r.add_argument("--scan", required=True, help="jsonl with seed and stepIndex")
    r.add_argument("--driver", required=True, help="checkpoint whose trace is frozen")
    r.add_argument(
        "--probe", action="append", required=True, help="name=checkpoint (repeat)"
    )
    r.add_argument("--partner-mode", type=int, default=1, help="1 called ace, 0 JD")
    r.add_argument("--out", required=True)
    o = sub.add_parser("overtrump", help="paired over-trump summary of a read")
    o.add_argument("--read", required=True)
    o.add_argument("--base", required=True, help="probe name of the reference")
    o.add_argument("--arm", required=True, help="probe name of the candidate")
    o.add_argument("--reps", type=int, default=10000, help="bootstrap resamples")
    o.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "read":
        cmd_read(args)
    else:
        cmd_overtrump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
