"""Search-bias calibration for the committee labels (CE_Teacher_Design §22.6
addendum; notebooks/Arm_D_Review_202610.md §4.2).

The centered noise model measures the committee's RANDOM error from the
replicates' disagreement. Error the replicates share — strategy fusion,
the oracle leaves' full-information optimism, theta_k standing in for the
other seats, the finite budget's own bias — is invisible to it, and at a
quiet late-trick node the honest posterior passes that error on at full
confidence (gamma 0.8-0.9 under class pooling). The planned remedy is an
explicit allowance sigma_n,eff^2 = sigma_n^2 + b^2; this script measures
b^2 per telemetry class against a much higher budget.

Fresh self-play games with the corpus's acting agent and committee acting,
exactly as ``distill_corpus`` plays them; at a sampled play node the
PRODUCTION committee (the corpus budget for that node) and a HIGH-budget
committee (``--high-iters``, optionally a deeper rollout) search the same
state from the same history. Per node it records both committees' pooled
centered Q over the actions both labeled, the per-replicate centered
variances and replicate counts (the production noise model's inputs), and
plays on with the production committee's choice.

The read (``--report``), per class and pooled: with d_a = ybar_lo,a -
ybar_hi,a over labels and the class-pooled centered variances s2_lo, s2_hi
over the replicate counts,

    b^2 = E[d^2] - E[s2_lo / n_lo] - E[s2_hi / n_hi]

is the production label's systematic error relative to the high budget
(itself a lower bound on the error relative to the truth), in the units the
noise model uses. Also reported: the fraction of nodes whose argmax moves,
and b^2 as a multiple of the production label noise.

Usage:
    python -m sheepshead.analysis.search_bias_calibration \\
        --ckpt runs/.../checkpoint_8000000.pt --out-dir runs/.../search_bias \\
        --games 400 --workers 8 --iters 256 --iters-schedule t0-lead:1024,t1-lead:512 \\
        --high-iters 4096 --routed-encoder mps
    python -m sheepshead.analysis.search_bias_calibration --report runs/.../search_bias
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from sheepshead.game import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

NODES_FILE = "nodes.jsonl"
DEFAULT_P_TRICK = "0:0.15,1:0.15,2:0.3,3:0.6,4:0.6,5:0.6"


def parse_p_trick(spec: str) -> dict[int, float]:
    out = {}
    for item in spec.split(","):
        t, p = item.split(":")
        out[int(t)] = float(p)
    return out


def _centered_labels(summary) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """(labeled action ids, centered pooled Q, per-replicate centered
    variance, replicate count) over the actions the committee visited."""
    n_obs = np.asarray(summary.q_n_obs)
    lab = n_obs > 0
    acts = [a for a, ok in zip(summary.actions, lab) if ok]
    q = np.asarray(summary.q_mean, dtype=np.float64)[lab]
    q = q - q.mean()
    s2c = np.asarray(summary.q_s2_centered, dtype=np.float64)[lab]
    return acts, q, s2c, n_obs[lab]


def worker_init(init_args: dict) -> None:
    from sheepshead.search.ismcts import (
        ISMCTSConfig,
        ISMCTSTeacher,
    )
    from sheepshead.training import distill_corpus as dc

    dc.worker_init(init_args)
    agent = dc._W["agent"]
    high = int(init_args["high_iters"])
    cfg = ISMCTSConfig(iters={h: high for h in ("pick", "partner", "bury", "play")})
    dc._W["high_teacher"] = ISMCTSTeacher(agent, cfg)
    dc._W["p_trick"] = parse_p_trick(init_args["p_trick"])
    dc._W["high_d_rollout"] = init_args.get("high_d_rollout")


def play_game(task: tuple) -> dict:
    """One self-play game; returns the sampled nodes' paired searches."""
    import torch

    from sheepshead.agent.observation import last_trick_observation_for, observation_for
    from sheepshead.search.ismcts import (
        infer_head,
        is_private_action,
    )
    from sheepshead.training import distill_corpus as dc

    game_idx, mode = task
    W = dc._W
    init_args = W["args"]
    agent = W["agent"]
    base_seed = int(init_args["seed"])
    game_rng = random.Random((base_seed << 20) ^ game_idx)
    torch.manual_seed((base_seed ^ (game_idx * 0x9E3779B1)) & 0x7FFFFFFF)
    det_rng = random.Random(game_rng.getrandbits(64))
    game = Game(partner_selection_mode=mode, seed=game_rng.randint(0, 2**31 - 1))
    agent.reset_recurrent_state()
    forced_public: list[tuple[int, int]] = []
    nodes: list[dict] = []
    high_d = W.get("high_d_rollout")

    while not game.is_done():
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            while valid_actions:
                state = observation_for(player, agent)
                action, _lp, _v = agent.act(state, valid_actions, player.position)
                anchor = None
                probs = agent.last_action_probs
                if probs is not None:
                    acts = sorted(valid_actions)
                    a = np.clip(
                        np.array([float(probs[i - 1]) for i in acts], dtype=np.float64),
                        1e-12,
                        None,
                    )
                    anchor = a / a.sum()
                head = infer_head(valid_actions)
                if len(valid_actions) >= 2 and head == "play" and not game.is_leaster:
                    if game.alone_called:
                        is_lead = all(c == "" for c in game.history[game.current_trick])
                    else:
                        is_lead, _cs = dc.lead_features(game, player)
                    trick = int(game.current_trick)
                    cls = dc.node_class(game, player, head)
                    p = W["p_trick"].get(trick, 0.0)
                    if det_rng.random() < p:
                        t0 = time.time()
                        target, info, s_lo, _ = dc._search_node(
                            game,
                            player,
                            valid_actions,
                            forced_public,
                            det_rng,
                            anchor,
                            teacher=dc.teacher_for_node(trick, is_lead),
                        )
                        t1 = time.time()
                        saved = None
                        if high_d is not None:
                            saved = init_args["d_rollout"]
                            init_args["d_rollout"] = int(high_d)
                        try:
                            _t, _i, s_hi, _ = dc._search_node(
                                game,
                                player,
                                valid_actions,
                                forced_public,
                                det_rng,
                                anchor,
                                teacher=W["high_teacher"],
                            )
                        finally:
                            if saved is not None:
                                init_args["d_rollout"] = saved
                        t2 = time.time()
                        if s_lo is not None and s_hi is not None:
                            a_lo, q_lo, s2_lo, n_lo = _centered_labels(s_lo)
                            a_hi, q_hi, s2_hi, n_hi = _centered_labels(s_hi)
                            nodes.append(
                                {
                                    "game": game_idx,
                                    "class": cls,
                                    "trick": trick,
                                    "is_lead": bool(is_lead),
                                    "n_valid": len(valid_actions),
                                    "iters_lo": int(
                                        dc.budget_for_node(trick, is_lead)
                                        or init_args["iters"]
                                    ),
                                    "iters_hi": int(init_args["high_iters"]),
                                    "actions_lo": a_lo,
                                    "q_lo": q_lo.tolist(),
                                    "s2c_lo": [float(x) for x in s2_lo],
                                    "nobs_lo": [int(x) for x in n_lo],
                                    "actions_hi": a_hi,
                                    "q_hi": q_hi.tolist(),
                                    "s2c_hi": [float(x) for x in s2_hi],
                                    "nobs_hi": [int(x) for x in n_hi],
                                    "w_lo": float(info["w"]) if info else None,
                                    "sec_lo": t1 - t0,
                                    "sec_hi": t2 - t1,
                                }
                            )
                        # Committee acting as the corpus does: play the
                        # production target's argmax on an override.
                        if (
                            target is not None
                            and info is not None
                            and info["w"] > 0.0
                            and anchor is not None
                        ):
                            acts = sorted(valid_actions)
                            action = acts[int(np.argmax(target))]
                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)
                if game.was_trick_just_completed and not game.is_done():
                    for seat in game.players:
                        agent.observe(
                            last_trick_observation_for(seat, agent),
                            player_id=seat.position,
                        )
                valid_actions = player.get_valid_action_ids()
    return {"game": game_idx, "nodes": nodes}


def report(out_dir: str, min_nodes: int = 30) -> dict:
    """Per-class b^2 from the paired searches (module docstring)."""
    rows = [json.loads(line) for line in open(os.path.join(out_dir, NODES_FILE))]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)

    def pooled_s2(recs, key_s2, key_n):
        num = den = 0.0
        for r in recs:
            for s2, n in zip(r[key_s2], r[key_n]):
                if n >= 2 and np.isfinite(s2):
                    num += (n - 1) * s2
                    den += n - 1
        return num / den if den > 0 else float("nan")

    def read(recs) -> dict:
        s2_lo = pooled_s2(recs, "s2c_lo", "nobs_lo")
        s2_hi = pooled_s2(recs, "s2c_hi", "nobs_hi")
        d2, v_lo, v_hi, moved, labels = [], [], [], 0, 0
        for r in recs:
            common = [a for a in r["actions_lo"] if a in r["actions_hi"]]
            if len(common) < 2:
                continue
            il = [r["actions_lo"].index(a) for a in common]
            ih = [r["actions_hi"].index(a) for a in common]
            ql = np.array(r["q_lo"])[il]
            qh = np.array(r["q_hi"])[ih]
            ql = ql - ql.mean()
            qh = qh - qh.mean()
            d2.extend(((ql - qh) ** 2).tolist())
            v_lo.extend((s2_lo / np.array(r["nobs_lo"])[il]).tolist())
            v_hi.extend((s2_hi / np.array(r["nobs_hi"])[ih]).tolist())
            labels += len(common)
            moved += int(common[int(np.argmax(ql))] != common[int(np.argmax(qh))])
        if not d2:
            return {"nodes": len(recs), "labels": 0}
        d2m, vl, vh = float(np.mean(d2)), float(np.mean(v_lo)), float(np.mean(v_hi))
        b2 = d2m - vl - vh
        return {
            "nodes": len(recs),
            "labels": labels,
            "mean_d2": d2m,
            "noise_lo": vl,
            "noise_hi": vh,
            "b2": b2,
            "b2_over_noise_lo": b2 / vl if vl > 0 else float("nan"),
            "argmax_moved": moved / len(recs),
            "sec_lo": float(np.mean([r["sec_lo"] for r in recs])),
            "sec_hi": float(np.mean([r["sec_hi"] for r in recs])),
        }

    out = {"all": read(rows), "by_class": {}, "by_trick": {}}
    for cls, recs in sorted(by_class.items()):
        if len(recs) >= min_nodes:
            out["by_class"][cls] = read(recs)
    by_trick: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_trick[r["trick"]].append(r)
    for t, recs in sorted(by_trick.items()):
        out["by_trick"][str(t)] = read(recs)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def print_report(rep: dict) -> None:
    def line(name, r):
        if r.get("labels", 0) == 0:
            print(f"  {name:28s} nodes {r['nodes']:5d}  (no paired labels)")
            return
        print(
            f"  {name:28s} nodes {r['nodes']:5d} labels {r['labels']:6d} "
            f"E[d^2] {r['mean_d2']:.2e} noise lo/hi {r['noise_lo']:.2e}/{r['noise_hi']:.2e} "
            f"b2 {r['b2']:+.2e} = {r['b2_over_noise_lo']:+.2f} x noise_lo; "
            f"argmax moved {100 * r['argmax_moved']:.1f}%  "
            f"sec lo/hi {r['sec_lo']:.1f}/{r['sec_hi']:.1f}"
        )

    print("[search-bias] all rows:")
    line("all", rep["all"])
    print("[search-bias] by trick:")
    for t, r in rep["by_trick"].items():
        line(f"trick {t}", r)
    print("[search-bias] by class:")
    for cls, r in rep["by_class"].items():
        line(cls, r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--report", metavar="OUT_DIR", help="only summarize an existing run"
    )
    ap.add_argument("--ckpt")
    ap.add_argument("--out-dir")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--iters", type=int, default=256)
    ap.add_argument("--iters-schedule", default="t0-lead:1024,t1-lead:512")
    ap.add_argument("--high-iters", type=int, default=4096)
    ap.add_argument(
        "--high-d-rollout",
        type=int,
        default=None,
        help="rollout depth for the high-budget committee (default: --d-rollout)",
    )
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--d-rollout", type=int, default=1)
    ap.add_argument("--shrink-nu", type=float, default=4.0)
    ap.add_argument("--shrink-s2-global", type=float, default=6.95e-4)
    ap.add_argument(
        "--p-trick",
        default=DEFAULT_P_TRICK,
        help="sampling probability of a multi-action play node by trick",
    )
    ap.add_argument("--routed-encoder", nargs="?", const="mps", default=None)
    args = ap.parse_args()

    if args.report:
        print_report(report(args.report))
        return 0
    if not (args.ckpt and args.out_dir):
        ap.error("--ckpt and --out-dir are required for a run")
    os.makedirs(args.out_dir, exist_ok=True)
    init_args = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "collect_oracle": False,
        "iters": args.iters,
        "iters_schedule": args.iters_schedule,
        "replicates": args.replicates,
        "d_rollout": args.d_rollout,
        "shrink_nu": args.shrink_nu,
        "shrink_s2_global": args.shrink_s2_global,
        "p_base": 1.0,
        "boost_lead": 1.0,
        "boost_cs": 1.0,
        "p_min": 0.0,
        "p_max": 1.0,
        "routed_encoder": args.routed_encoder,
        "high_iters": args.high_iters,
        "high_d_rollout": args.high_d_rollout,
        "p_trick": args.p_trick,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    tasks = [
        (g, PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD)
        for g in range(args.games)
    ]
    ctx = get_context("spawn")
    t0 = time.time()
    n_nodes = 0
    with (
        open(os.path.join(args.out_dir, NODES_FILE), "a") as out,
        ctx.Pool(args.workers, initializer=worker_init, initargs=(init_args,)) as pool,
    ):
        for k, res in enumerate(pool.imap_unordered(play_game, tasks), 1):
            for node in res["nodes"]:
                out.write(json.dumps(node) + "\n")
            out.flush()
            n_nodes += len(res["nodes"])
            if k % 10 == 0 or k == len(tasks):
                el = time.time() - t0
                print(
                    f"[{k}/{len(tasks)} games, {n_nodes} nodes, {k / el:.3f} g/s, "
                    f"ETA {(len(tasks) - k) / max(k / el, 1e-9) / 60:.0f} min]",
                    flush=True,
                )
    print_report(report(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
