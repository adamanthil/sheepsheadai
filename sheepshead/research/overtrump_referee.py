#!/usr/bin/env python3
"""Hindsight referee for the called-suit over-trump decision
(Arm_D_Review §9.10): an estimator outside the search family.

The search committee that labels the corpus shares its structural errors
across budgets (determinized strategy fusion, oracle-leaf optimism, theta_k
in the other seats), so 256-vs-2048 iterations cannot see them (§9.5,
CE_Teacher §22.6). This referee values each legal card at a scan node by
CONTINUING THE ACTUAL DEAL: the game is replayed to the node along the
driver's frozen trajectory (``frozen_node_policy_read``), the state and every
seat's recurrent memory are snapshotted, each legal card is played, and a
continuation policy plays all five seats to the end of the hand. The node
seat's realized hand score is the card's value. Per node this is one draw of
the player's posterior over hidden cards, so it is noisy; averaged over the
scan's nodes it is unbiased for "at nodes like these, which card scores
more", and it carries none of the committee's errors.

Two estimators per card: a deterministic continuation (one playout, the
argmax policy in every seat) and the mean of ``--samples`` stochastic
playouts with common random numbers across the cards of a node (the same
torch seed per sample index), which pairs the branches.

``run`` writes a JSON with, per node, the value of every legal card under
both estimators, the continuation policy's own probabilities at the node,
and (when ``--budget-rows`` is given) the 256-iteration search Q of
``budget_rows generate`` for the same node. ``summary`` reports the referee
verdict over-trump minus the policy's alternative (mean, SE, bootstrap by
seed), the subsets of §9.8, the agreement with the search verdict, and the
referee's cost of the continuation policy's own choice.

Usage (from repo root):

    uv run python -m sheepshead.research.overtrump_referee run \\
        --scan runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl \\
        --driver runs/202609_recall_rc/pi/iter3_ablations/class_baseline/distill_epoch10.pt \\
        --continuation runs/202609_recall_rc/pi/iter4/bidding/final.pt \\
        --budget-rows runs/202609_recall_rc/overtrump_probe/budget_corpus_256.pt \\
        --samples 8 --out runs/202609_recall_rc/overtrump_probe/referee/referee_theta4.json
    uv run python -m sheepshead.research.overtrump_referee summary \\
        --read runs/202609_recall_rc/overtrump_probe/referee/referee_theta4.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from sheepshead.agent.observation import last_trick_observation_for
from sheepshead.agent.traced_game import find_actor, run_inference_step, set_seed
from sheepshead.game import ACTIONS, DECK, TRUMP_POWER, Game
from sheepshead.research.frozen_node_policy_read import (
    POINT_TRUMPS,
    driver_actions,
)
from sheepshead.research.scan_defender_trump_leads import load_scan_agent
from sheepshead.rewards import RETURN_SCALE

DEVICE = torch.device("cpu")


def _play_id(card: str) -> int:
    return ACTIONS.index(f"PLAY {card}") + 1


def _snapshot(game: Game, agent: Any) -> tuple[Game, dict[int, torch.Tensor]]:
    mem = {k: v.clone() for k, v in agent._player_memories.items()}
    return copy.deepcopy(game), mem


def _restore(snap: tuple[Game, dict[int, torch.Tensor]], agent: Any) -> Game:
    game, mem = snap
    agent._player_memories = {k: v.clone() for k, v in mem.items()}
    return copy.deepcopy(game)


def _step(game: Game, agent: Any, action_id: int, actor: Any) -> None:
    actor.act(action_id)
    if game.was_trick_just_completed:
        for seat in game.players:
            agent.observe(
                last_trick_observation_for(seat, agent), player_id=seat.position
            )


def _continue(game: Game, agent: Any, deterministic: bool) -> None:
    """Play the hand out with ``agent`` in every seat."""
    while not game.is_done():
        actor = find_actor(game)
        if actor is None:
            raise RuntimeError("no actor before the hand is done")
        inf = run_inference_step(agent, actor, DEVICE)
        probs = inf.action_probs.squeeze(0)
        if deterministic:
            a = int(torch.argmax(probs).item()) + 1
        else:
            a = int(torch.distributions.Categorical(probs).sample().item()) + 1
        _step(game, agent, a, actor)


def _value(game: Game, seat: int) -> float:
    return float(game.players[seat - 1].get_score())


def referee_seed(
    agent: Any,
    seed: int,
    partner_mode: int,
    actions: list[int],
    nodes: list[dict],
    samples: int,
) -> list[dict]:
    """Replay ``actions`` to each node of ``seed`` with ``agent`` building its
    own memory; at the node value every legal card by continuation."""
    steps = {int(n["stepIndex"]): n for n in nodes}
    set_seed(seed)
    agent.reset_recurrent_state()
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    out: list[dict] = []
    for step, action_id in enumerate(actions):
        actor = find_actor(game)
        if actor is None:
            raise RuntimeError(f"seed {seed}: no actor at step {step}")
        inf = run_inference_step(agent, actor, DEVICE)
        if action_id not in inf.valid_actions:
            raise RuntimeError(f"seed {seed} step {step}: trace action is illegal")
        if step in steps:
            node = steps[step]
            if actor.position != int(node["seat"]):
                raise RuntimeError(f"seed {seed} step {step}: seat mismatch")
            probs = inf.action_probs.squeeze(0).cpu()
            legal = [a for a in inf.valid_actions if ACTIONS[a - 1].startswith("PLAY ")]
            snap = _snapshot(game, agent)
            det: dict[str, float] = {}
            sto: dict[str, list[float]] = {}
            for a in legal:
                card = ACTIONS[a - 1][5:]
                g = _restore(snap, agent)
                act = g.players[actor.position - 1]
                _step(g, agent, a, act)
                _continue(g, agent, deterministic=True)
                det[card] = _value(g, actor.position)
                vals: list[float] = []
                for k in range(samples):
                    g = _restore(snap, agent)
                    act = g.players[actor.position - 1]
                    _step(g, agent, a, act)
                    torch.manual_seed(seed * 1000 + step * 10 + k)
                    _continue(g, agent, deterministic=False)
                    vals.append(_value(g, actor.position))
                sto[card] = vals
            # Restore the trace's state before continuing the prefix.
            game = _restore(snap, agent)
            actor = game.players[actor.position - 1]
            out.append(
                {
                    **node,
                    "legal": [ACTIONS[a - 1][5:] for a in legal],
                    "policy": {ACTIONS[a - 1][5:]: float(probs[a - 1]) for a in legal},
                    "v_det": det,
                    "v_sto": sto,
                }
            )
        _step(game, agent, action_id, actor)
    if len(out) != len(nodes):
        raise RuntimeError(f"seed {seed}: reached {len(out)} of {len(nodes)} nodes")
    return out


def search_q_by_node(path: str) -> dict[tuple[int, int, int], dict[str, float]]:
    """(seed, seat, trick) -> card -> search Q (score units) from a
    ``budget_rows generate`` file (a corpus shard: five episodes per game in
    the ``games`` order, targets flagged ``has_search_target``)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    games = [int(g["game"]) for g in d["games"]]
    out: dict[tuple[int, int, int], dict[str, float]] = {}
    for i, ep in enumerate(d["episodes"]):
        seed = games[i // 5]
        for e in ep:
            if not (isinstance(e, dict) and e.get("has_search_target")):
                continue
            valid = sorted(e["valid_actions"])
            q = e["search_q"]
            key = (seed, int(e["player_id"]), int(e["state"]["current_trick"]))
            out[key] = {
                ACTIONS[a - 1][5:]: RETURN_SCALE * float(v)
                for a, v in zip(valid, q)
                if ACTIONS[a - 1].startswith("PLAY ")
            }
    return out


def cmd_run(args: argparse.Namespace) -> None:
    nodes = [json.loads(line) for line in open(args.scan)]
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for n in nodes:
        by_seed[int(n["seed"])].append(n)
    driver = load_scan_agent(args.driver)
    cont = load_scan_agent(args.continuation)
    sq = search_q_by_node(args.budget_rows) if args.budget_rows else {}
    rows: list[dict] = []
    t0 = time.time()
    for i, seed in enumerate(sorted(by_seed)):
        actions = driver_actions(driver, seed, args.partner_mode)
        for r in referee_seed(
            cont, seed, args.partner_mode, actions, by_seed[seed], args.samples
        ):
            key = (seed, int(r["seat"]), int(r["trick"]))
            r["search_q"] = sq.get(key)
            rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(by_seed)}] seeds {time.time() - t0:.0f}s", flush=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "scan": args.scan,
                "driver": args.driver,
                "continuation": args.continuation,
                "budget_rows": args.budget_rows,
                "samples": args.samples,
                "nodes": rows,
            },
            f,
        )
    print(f"{len(rows)} nodes from {len(by_seed)} seeds -> {args.out}")


# ----------------------------------------------------------------------- #
# Summary
# ----------------------------------------------------------------------- #
def _over_cards(node: dict) -> list[str]:
    w = node["winner"]
    return [
        c for c in node["legal"] if c in TRUMP_POWER and TRUMP_POWER[c] > TRUMP_POWER[w]
    ]


def _top(d: dict[str, float], cards: list[str]) -> str:
    return max(cards, key=lambda c: d[c])


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(
        x
    ) > 1 else 0.0


def _boot_ci(d: np.ndarray, seeds: np.ndarray, reps: int, rng) -> tuple[float, float]:
    uniq = np.unique(seeds)
    groups = [d[seeds == s] for s in uniq]
    means = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        means[r] = np.concatenate([groups[k] for k in pick]).mean()
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def cmd_summary(args: argparse.Namespace) -> None:
    data = json.load(open(args.read))
    nodes = data["nodes"]
    rng = np.random.default_rng(args.seed)
    print(
        f"{len(nodes)} nodes; continuation {data['continuation']}; "
        f"{data['samples']} stochastic samples per card"
    )

    def verdicts(est: str) -> dict[str, np.ndarray]:
        """Per node: referee over-trump minus alternative, where the
        over-trump is the policy's (driver's) top over-trump card and the
        alternative its top non-over-trump card (the §9.5 definition);
        also the referee's cost of the continuation policy's argmax."""
        d_pol, d_best, cost, over_best, seeds, clear, schmear, n_after, trick = (
            [] for _ in range(9)
        )
        for n in nodes:
            v = (
                n["v_det"]
                if est == "det"
                else {c: float(np.mean(x)) for c, x in n["v_sto"].items()}
            )
            over = _over_cards(n)
            other = [c for c in n["legal"] if c not in over]
            if not over or not other:
                continue
            pol = n["policy"]
            o = _top(pol, over)
            a = _top(pol, other)
            d_pol.append(v[o] - v[a])
            d_best.append(max(v[c] for c in over) - max(v[c] for c in other))
            choice = _top(pol, n["legal"])
            cost.append(v[choice] - max(v.values()))
            over_best.append(_top(v, n["legal"]) in over)
            seeds.append(int(n["seed"]))
            clear.append(n["max_fail_pts"] >= 0 and o not in POINT_TRUMPS)
            schmear.append(n["max_fail_pts"] >= 10)
            n_after.append(int(n["n_after"]))
            trick.append(int(n["trick"]))
        return {
            k: np.array(v)
            for k, v in dict(
                d_pol=d_pol,
                d_best=d_best,
                cost=cost,
                over_best=over_best,
                seeds=seeds,
                clear=clear,
                schmear=schmear,
                n_after=n_after,
                trick=trick,
            ).items()
        }

    for est in ("det", "sto"):
        r = verdicts(est)
        label = (
            "deterministic continuation"
            if est == "det"
            else f"stochastic mean of {data['samples']}"
        )
        print(f"\n== referee: {label} (score per hand, node seat)")
        for name, sel in (
            ("all decisions", np.ones(len(r["seeds"]), bool)),
            ("clear blunder (has a fail; over-trump not AD/10D)", r["clear"]),
            ("holding a schmear (10/A fail)", r["schmear"]),
            ("0 players after", r["n_after"] == 0),
            ("1 player after", r["n_after"] == 1),
            ("2 players after", r["n_after"] == 2),
            ("trick 0", r["trick"] == 0),
            ("tricks 1-2", (r["trick"] >= 1) & (r["trick"] <= 2)),
            ("tricks 3-4", r["trick"] >= 3),
        ):
            d = r["d_pol"][sel]
            if len(d) < 5:
                continue
            m, se = _mean_se(d)
            lo, hi = _boot_ci(d, r["seeds"][sel], args.reps, rng)
            mb, _ = _mean_se(r["d_best"][sel])
            print(
                f"   {name:48s} n={len(d):3d}  over − alt {m:+.3f} ± {se:.3f} "
                f"[90% {lo:+.3f}, {hi:+.3f}]  best-over − best-alt {mb:+.3f}  "
                f"referee prefers an over-trump {100 * r['over_best'][sel].mean():.0f}%"
            )
        m, se = _mean_se(r["cost"])
        print(
            f"   continuation policy's argmax: referee regret vs its best card "
            f"{m:+.3f} ± {se:.3f} (0 = always picks the referee's best)"
        )

    # Agreement with the search label on the joined nodes.
    joined = [n for n in nodes if n.get("search_q")]
    print(
        f"\n== search (256-iteration committee at the same nodes): {len(joined)} joined"
    )
    if joined:
        ds, dr, dr_sto = [], [], []
        for n in joined:
            over = _over_cards(n)
            other = [c for c in n["legal"] if c not in over]
            if not over or not other:
                continue
            q = n["search_q"]
            if not all(c in q for c in n["legal"]):
                continue
            o, a = _top(n["policy"], over), _top(n["policy"], other)
            ds.append(q[o] - q[a])
            dr.append(n["v_det"][o] - n["v_det"][a])
            dr_sto.append(float(np.mean(n["v_sto"][o]) - np.mean(n["v_sto"][a])))
        s, rd, rs = np.array(ds), np.array(dr), np.array(dr_sto)
        for name, r in (("deterministic", rd), ("stochastic", rs)):
            print(
                f"   search over − alt {s.mean():+.3f} ± {s.std(ddof=1) / len(s) ** 0.5:.3f};  "
                f"referee ({name}) {r.mean():+.3f} ± {r.std(ddof=1) / len(r) ** 0.5:.3f};  "
                f"corr {np.corrcoef(s, r)[0, 1]:+.2f}"
            )
            endorsed = s > 0
            if endorsed.sum() >= 5:
                e = r[endorsed]
                print(
                    f"     where search endorses the over-trump (n={int(endorsed.sum())}): "
                    f"search {s[endorsed].mean():+.3f}, referee {e.mean():+.3f} ± {e.std(ddof=1) / len(e) ** 0.5:.3f}; "
                    f"referee agrees (over better) {100 * (e > 0).mean():.0f}%"
                )
            rejected = s <= 0
            if rejected.sum() >= 5:
                e = r[rejected]
                print(
                    f"     where search rejects it (n={int(rejected.sum())}): search {s[rejected].mean():+.3f}, "
                    f"referee {e.mean():+.3f} ± {e.std(ddof=1) / len(e) ** 0.5:.3f}; referee agrees (alt better) {100 * (e < 0).mean():.0f}%"
                )
        print(f"   (deck order check: {DECK[0]} is card id 1)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="value every legal card at the scan's nodes")
    r.add_argument("--scan", required=True, help="jsonl with seed and stepIndex")
    r.add_argument("--driver", required=True, help="checkpoint whose trace is frozen")
    r.add_argument("--continuation", required=True, help="checkpoint playing all seats")
    r.add_argument("--budget-rows", default=None, help="budget_rows generate output")
    r.add_argument(
        "--samples", type=int, default=8, help="stochastic playouts per card"
    )
    r.add_argument("--partner-mode", type=int, default=1, help="1 called ace, 0 JD")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary", help="referee verdicts and search agreement")
    s.add_argument("--read", required=True)
    s.add_argument("--reps", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
