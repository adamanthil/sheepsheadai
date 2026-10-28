#!/usr/bin/env python3
"""Re-search exact policy-iteration corpus nodes at a higher committee budget
(Convention_Optimality addendum A1; also used by ``called_suit_lead_cells``).

``search`` rebuilds each corpus game from its per-game seed exactly as
``distill_corpus`` dealt it, replays every seat's recorded actions up to the
target node (verifying the hand and legal set), then runs the committee at
the requested budget(s) and replicate counts. Output is one JSON line per
node and the run resumes: nodes already in ``--out`` are skipped. The corpus
labels (256-1024 iterations x 3 replicates) can then be compared against an
offline-grade budget on the same nodes.

``jd-nodes`` / ``jd-report`` are the A1 question: JD-mode t0 defender leads
holding exactly one trump. The report compares trump lead minus the policy's
preferred fail at the corpus budget and at the re-search budget, by fail aces
x fail void and by picker seat.

Usage (from repo root):

    uv run python -m sheepshead.research.research_corpus_nodes jd-nodes \
        --corpus-dir runs/202609_recall_rc/pi/iter2/targeted \
        --out runs/202609_recall_rc/t0_trump_confirm/jd1_nodes.json
    uv run python -m sheepshead.research.research_corpus_nodes search \
        --nodes runs/202609_recall_rc/t0_trump_confirm/jd1_nodes.json \
        --corpus-dir runs/202609_recall_rc/pi/iter2/targeted \
        --ckpt runs/202609_recall_rc/pi/iter1/bidding/final.pt \
        --out runs/202609_recall_rc/t0_trump_confirm/search_4096.jsonl \
        --budgets 4096 --replicates 6 --workers 4
    uv run python -m sheepshead.research.research_corpus_nodes jd-report \
        --corpus-dir runs/202609_recall_rc/pi/iter2/targeted \
        --results runs/202609_recall_rc/t0_trump_confirm/search_4096.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import torch

from sheepshead.game import ACTIONS, TRUMP_POWER
from sheepshead.rewards import RETURN_SCALE
from sheepshead.search.ismcts import (
    ISMCTSConfig,
    ISMCTSTeacher,
)
from sheepshead.training import distill_corpus as dc

CORPUS_SEED = 20260904  # the corpus's --seed (iteration 2 of 202609_recall_rc)

# Per worker process: one teacher per budget, and the search args.
_TEACHERS: dict[int, ISMCTSTeacher] = {}
_ARGS: dict = {}


def corpus_shards(corpus_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(corpus_dir, "corpus_*.pt")))


def node_key(n: dict) -> tuple[int, int, int]:
    return (int(n["shard"]), int(n["ep"]), int(n["ev"]))


def corpus_labels(
    corpus_dir: str, keys: list[tuple[int, int, int]]
) -> dict[tuple[int, int, int], dict]:
    """The corpus's own label per node: legal cards, act-time prior, search
    Q and its variance."""
    shards = corpus_shards(corpus_dir)
    by_shard: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for k in keys:
        by_shard[k[0]].append(k)
    out = {}
    for s, ks in by_shard.items():
        d = torch.load(shards[s], weights_only=False, map_location="cpu")
        for k in ks:
            ev = d["episodes"][k[1]][k[2]]
            valid = sorted(ev["valid_actions"])
            out[k] = {
                "cards": [ACTIONS[a - 1][5:] for a in valid],
                "prior": np.array(ev["anchor_probs"]),
                "q": np.array(ev["search_q"]),
                "qv": np.array(ev["search_q_var"]),
                "state": ev["state"],
            }
        del d
    return out


def load_results(path: str, budget: str) -> tuple[list[dict], int]:
    """Search results with a usable ``budget`` entry, and the error count."""
    rows, errors = [], 0
    for line in open(path):
        s = json.loads(line)
        if s.get("error") or not s.get("search") or not s["search"].get(budget):
            errors += 1
            continue
        rows.append(s)
    return rows, errors


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def build_tasks(corpus_dir: str, nodes: list[dict]) -> list[dict]:
    """Per node: its game meta, every seat's recorded actions, the target
    seat, its action count before the node, its hand and legal set."""
    shards = corpus_shards(corpus_dir)
    by_shard: dict[int, list[dict]] = defaultdict(list)
    for n in nodes:
        by_shard[n["shard"]].append(n)
    tasks = []
    for s, ns in sorted(by_shard.items()):
        d = torch.load(shards[s], weights_only=False, map_location="cpu")
        for n in ns:
            g = n["ep"] // 5
            first = g * 5
            meta = d["games"][g]
            seats = {}
            for pos in range(1, 6):
                acts = [
                    e for e in d["episodes"][first + pos - 1] if e["kind"] == "action"
                ]
                if not all(e["player_id"] == pos for e in acts):
                    raise ValueError(f"shard {s} game {g}: episode {pos} mixes seats")
                seats[pos] = [int(e["action"]) for e in acts]
            ep = d["episodes"][n["ep"]]
            target = ep[n["ev"]]
            tasks.append(
                {
                    **n,
                    "game": int(meta["game"]),
                    "mode": meta["mode"],
                    "seat": int(target["player_id"]),
                    "k": sum(1 for e in ep[: n["ev"]] if e["kind"] == "action"),
                    "seat_actions": seats,
                    "hand": [int(x) for x in target["state"]["hand_ids"]],
                    "valid": sorted(int(a) for a in target["valid_actions"]),
                }
            )
        del d
    return tasks


def _worker_init(init_args: dict) -> None:
    dc.worker_init(init_args)
    agent = dc._W["agent"]
    for b in init_args["budgets"]:
        _TEACHERS[b] = ISMCTSTeacher(
            agent,
            ISMCTSConfig(iters={h: b for h in ("pick", "partner", "bury", "play")}),
        )
    _ARGS.update(init_args)


def _run_task(t: dict) -> dict:
    from sheepshead.agent.observation import last_trick_observation_for, observation_for
    from sheepshead.game import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game
    from sheepshead.search.ismcts import is_private_action

    args = dc._W["args"]
    agent = dc._W["agent"]
    base_seed = int(args["seed"])
    game_idx = t["game"]
    mode = PARTNER_BY_CALLED_ACE if t["mode"] == "called" else PARTNER_BY_JD
    # The corpus's per-game streams (play_corpus_game): the same draws in
    # the same order, so the deal is the corpus's.
    game_rng = random.Random((base_seed << 20) ^ game_idx)
    torch.manual_seed((base_seed ^ (game_idx * 0x9E3779B1)) & 0x7FFFFFFF)
    _ = random.Random(game_rng.getrandbits(64))
    game = Game(partner_selection_mode=mode, seed=game_rng.randint(0, 2**31 - 1))
    agent.reset_recurrent_state()
    forced_public: list[tuple[int, int]] = []
    seat_actions = {int(k): v for k, v in t["seat_actions"].items()}
    used: dict[int, int] = defaultdict(int)
    out = {k: v for k, v in t.items() if k != "seat_actions"}
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                pos = player.position
                state = observation_for(player, agent)
                agent.act(state, valid, pos)
                if pos == t["seat"] and used[pos] == t["k"]:
                    if [int(x) for x in state["hand_ids"]] != t["hand"] or sorted(
                        valid
                    ) != t["valid"]:
                        out["error"] = "reconstruction mismatch"
                        return out
                    res = {}
                    for b, reps in zip(_ARGS["budgets"], _ARGS["replicates_by_budget"]):
                        args["replicates"] = reps
                        rng = random.Random((base_seed * 1000003) ^ (game_idx * 97) ^ b)
                        t0 = time.time()
                        _tgt, _info, summ, _ = dc._search_node(
                            game,
                            player,
                            valid,
                            forced_public,
                            rng,
                            None,
                            teacher=_TEACHERS[b],
                        )
                        if summ is None:
                            res[str(b)] = None
                            continue
                        res[str(b)] = {
                            "actions": [int(a) for a in summ.actions],
                            "q": [float(x) for x in summ.q_mean],
                            "qvar": [float(x) for x in summ.q_var],
                            "n": [float(x) for x in summ.n_mean],
                            "s2c": [float(x) for x in summ.q_s2_centered],
                            "nobs": [int(x) for x in summ.q_n_obs],
                            "sec": time.time() - t0,
                        }
                    out["search"] = res
                    return out
                action = seat_actions[pos][used[pos]]
                used[pos] += 1
                if not is_private_action(action):
                    forced_public.append((pos, action))
                player.act(action)
                if game.was_trick_just_completed and not game.is_done():
                    for seat in game.players:
                        agent.observe(
                            last_trick_observation_for(seat, agent),
                            player_id=seat.position,
                        )
                valid = player.get_valid_action_ids()
    out["error"] = "node not reached"
    return out


def search(args) -> None:
    with open(args.nodes) as f:
        nodes = json.load(f)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            done.add(node_key(json.loads(line)))
    nodes = [n for n in nodes if node_key(n) not in done]
    if args.limit:
        nodes = nodes[: args.limit]
    print(f"{len(nodes)} nodes to search ({len(done)} already done)", flush=True)
    tasks = build_tasks(args.corpus_dir, nodes)
    budgets = [int(x) for x in args.budgets]
    init_args = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "torch_threads": 1,
        "iters": 256,
        "iters_schedule": "",
        "replicates": 3,
        "d_rollout": 1,
        "shrink_nu": 4.0,
        "shrink_s2_global": 6.95e-4,
        "routed_encoder": args.routed_encoder or None,
        "budgets": budgets,
        "replicates_by_budget": [int(x) for x in args.replicates],
    }
    t0 = time.time()
    with (
        open(args.out, "a") as f,
        get_context("spawn").Pool(
            args.workers, initializer=_worker_init, initargs=(init_args,)
        ) as pool,
    ):
        for i, r in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                print(
                    f"[{i}/{len(tasks)}] {el / 60:.1f} min, eta {el / i * (len(tasks) - i) / 60:.0f} min",
                    flush=True,
                )


# --------------------------------------------------------------------------- #
# A1: JD singleton-trump t0 defender leads
# --------------------------------------------------------------------------- #
def _hand_features(cards: list[str]) -> dict:
    fails = [c for c in cards if c not in TRUMP_POWER]
    suits = {c[-1] for c in fails}
    return {"aces": sum(c[:-1] == "A" for c in fails), "void": 3 - len(suits)}


def jd_nodes(corpus_dir: str) -> list[dict]:
    """Searched JD t0 defender leads whose legal set holds exactly one trump."""
    nodes = []
    for si, path in enumerate(corpus_shards(corpus_dir)):
        d = torch.load(path, weights_only=False, map_location="cpu")
        for ei, ep in enumerate(d["episodes"]):
            for vi, ev in enumerate(ep):
                if (
                    str(ev.get("node_class")) != "std|t0-defender-lead"
                    or ev.get("search_q") is None
                ):
                    continue
                if int(ev["state"]["partner_mode"]) != 0:
                    continue
                cards = [ACTIONS[a - 1][5:] for a in sorted(ev["valid_actions"])]
                istr = np.array([c in TRUMP_POWER for c in cards])
                if istr.sum() != 1:
                    continue
                p = np.array(ev["anchor_probs"])
                nodes.append(
                    {
                        "shard": si,
                        "ep": ei,
                        "ev": vi,
                        **_hand_features(cards),
                        "arg": bool(istr[p.argmax()]),
                        "prel": int(ev["state"]["picker_rel"]),
                    }
                )
        del d
    return nodes


def jd_report(corpus_dir: str, results: str, budget: str) -> None:
    res, errors = load_results(results, budget)
    labels = corpus_labels(corpus_dir, [node_key(s) for s in res])
    rows = []
    for s in res:
        lab = labels[node_key(s)]
        h = s["search"][budget]
        if [ACTIONS[a - 1][5:] for a in h["actions"]] != lab["cards"]:
            raise ValueError(
                f"node {node_key(s)}: re-search legal set differs from the corpus's"
            )
        t = np.array([c in TRUMP_POWER for c in lab["cards"]])
        p, ql = lab["prior"], lab["q"]
        bt = int(np.argmax(t))
        bf = int(np.argmax(np.where(~t, p, -1)))  # the policy's preferred fail
        qh = np.array(h["q"])
        bfh = int(np.argmax(np.where(~t, qh, -9)))
        rows.append(
            {
                "aces": s["aces"],
                "void": s["void"],
                "arg": s["arg"],
                "prel": s["prel"],
                "lo": RETURN_SCALE * (ql[bt] - ql[bf]),
                "hi": RETURN_SCALE * (qh[bt] - qh[bf]),
                "hi_best": RETURN_SCALE * (qh[bt] - qh[bfh]),
                "hi_argmax_trump": bool(t[int(np.argmax(qh))]),
            }
        )
    print(
        f"{len(rows)} nodes read, {errors} errors; trump lead minus the policy's preferred fail (score)"
    )

    def show(label: str, x: list[dict]) -> None:
        if len(x) < 3:
            print(f"  {label:30s} n {len(x)}")
            return
        lo, hi, hb = (np.array([r[k] for r in x]) for k in ("lo", "hi", "hi_best"))

        def se(a: np.ndarray) -> float:
            return float(a.std(ddof=1) / np.sqrt(len(a)))

        print(
            f"  {label:30s} n {len(x):3d}  corpus {lo.mean():+.3f}±{se(lo):.3f}  {budget} {hi.mean():+.3f}±{se(hi):.3f}  "
            f"shift {np.mean(hi - lo):+.3f}±{se(hi - lo):.3f}  vs {budget}'s best fail {hb.mean():+.3f}  "
            f"{budget} top card trump {100 * np.mean([r['hi_argmax_trump'] for r in x]):.0f}%"
        )

    show("ALL", rows)
    show("policy argmax = trump", [r for r in rows if r["arg"]])
    for a in (2, 1, 0):
        for v in (0, 1):
            cell = [r for r in rows if r["aces"] == a and r["void"] == v]
            show(f"aces {a}, void {v}", cell)
            show("   ... & policy argmax trump", [r for r in cell if r["arg"]])
    for p in (2, 3, 4, 5):
        show(
            f"aces>=1 no void, picker_rel {p}",
            [r for r in rows if r["aces"] >= 1 and r["void"] == 0 and r["prel"] == p],
        )
    print(
        "cell counts (aces, void):",
        dict(collections.Counter((r["aces"], r["void"]) for r in rows)),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "search", help="re-search corpus nodes at higher budgets (resumable)"
    )
    s.add_argument(
        "--nodes",
        required=True,
        help='JSON list of {"shard", "ep", "ev", ...extra kept}',
    )
    s.add_argument(
        "--corpus-dir", required=True, help="the targeted corpus the nodes index"
    )
    s.add_argument(
        "--ckpt", required=True, help="the checkpoint the corpus was generated with"
    )
    s.add_argument("--out", required=True)
    s.add_argument("--budgets", nargs="+", default=["1024", "4096"])
    s.add_argument("--replicates", nargs="+", default=["3", "6"], help="one per budget")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=CORPUS_SEED, help="the corpus's --seed")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--routed-encoder", default="mps")
    n = sub.add_parser(
        "jd-nodes", help="A1 node list: JD t0 defender leads with one trump"
    )
    n.add_argument("--corpus-dir", required=True)
    n.add_argument("--out", required=True)
    r = sub.add_parser(
        "jd-report", help="A1: corpus vs re-search verdicts by aces x void"
    )
    r.add_argument("--corpus-dir", required=True)
    r.add_argument("--results", required=True, help="search --out JSONL")
    r.add_argument("--budget", default="4096")
    args = ap.parse_args(argv)
    if args.cmd == "search":
        if len(args.budgets) != len(args.replicates):
            ap.error("--replicates needs one count per budget")
        search(args)
    elif args.cmd == "jd-nodes":
        nodes = jd_nodes(args.corpus_dir)
        with open(args.out, "w") as f:
            json.dump(nodes, f)
        print(f"{len(nodes)} nodes -> {args.out}")
    else:
        jd_report(args.corpus_dir, args.results, args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
