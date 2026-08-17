#!/usr/bin/env python3
"""Search help/harm matrix over play-node classes (E9, teacher-coverage design).

The original teacher plan restricted search to selected defender-lead nodes
because PUCT visit-count targets at deploy budgets showed harm at ties
(Q-inversion; Search_Readout / Deploy_Search studies, pre-pi_gumbel, weaker
critic). E7/E8 established that search is REQUIRED at nodes whose edge only
exists under improved continuations. This instrument reassesses, under the
adopted pi_gumbel readout and the current critic, WHERE search provides
genuine signal advantage over the policy, at WHAT budget, and at WHAT
rollout depth — per node class, not just defender leads.

Method: a driver checkpoint replays seeds greedily (E7/E8 convention).
Play decisions are sampled into cells (trick x role x lead/follow) up to a
per-cell quota. At each sampled node every config in the grid

    iters in {128, 384, 1024} x d_rollout in {2, terminal}

runs the ISMCTS teacher, plus one REFERENCE search (4096 iters, terminal
rollouts) whose root Q ordering serves as ground truth (the offline-grade
instrument validated in the targeted-search and E6 studies). Per node:

    a_policy = driver argmax          (what PG deploys)
    a_cfg    = pi_gumbel argmax of the config's search
    a_ref    = pi_gumbel argmax of the reference search
    headroom = Qref(a_ref) - Qref(a_policy)   (available improvement, >= 0-ish)
    uplift   = Qref(a_cfg) - Qref(a_policy)   (what the config captures)
    harm     = uplift < -eps                  (config flips to a WORSE action)

Per (cell, config): mean headroom, mean uplift, capture ratio, harm rate,
ESS-skip rate. The teacher-coverage rule reads off the matrix: include a
cell wherever headroom is materially > 0 and some config captures it with a
bounded harm rate; choose the cheapest such config per cell.

Leaf evaluation follows ``ISMCTSConfig.leaf_evaluator`` — "oracle" by default
since 2026-08-10 (privileged oracle critic on the observer's full-information
stream), silently falling back to the limited critic when the driver
checkpoint carries no oracle head (see Search_Teacher_Design notebook).

Reuse mode (--reuse-ref PRIOR.json): freeze the prior run's nodes and its
4096/term reference, and rerun only --configs arms (default: the d=2 arms,
whose leaf evaluator is the thing that changed). Node identity comes from
the deterministic greedy replay — each kept node is matched against the
prior rows in replay order by (seed, cell, policyAction, nLegal) — and the
original per-node RNG seeds are reproduced from the prior node index, so
world pools are identical to the prior run and the arms differ only through
leaf values: a tight leaf-evaluator A/B on shared ground truth.

Usage (from repo root):

    uv run python -m sheepshead.analysis.search_help_matrix \\
        --driver runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_7000000.pt \\
        --quota 8 --num-seeds 400 \\
        --out runs/convention_optimality_202607/search_help_matrix_e9.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from sheepshead import ACTION_LOOKUP, PARTNER_BY_CALLED_ACE, Game
from sheepshead.agent.ppo import load_agent
from sheepshead.analysis.fail_lead_logit_probe import _masked_logits
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher, _is_private_action
from sheepshead.training.pfsp_runtime import play_cell

DEVICE = torch.device("cpu")
BASE_RNG_SEED = 20260811
TERMINAL_DEPTH = 99
HARM_EPS = 0.01  # Q-units; flips smaller than this count as neutral

GRID = [
    (128, 2),
    (128, TERMINAL_DEPTH),
    (384, 2),
    (384, TERMINAL_DEPTH),
    (1024, 2),
    (1024, TERMINAL_DEPTH),
]
REFERENCE = (4096, TERMINAL_DEPTH)


def _teacher(agent, iters: int) -> ISMCTSTeacher:
    cfg = ISMCTSConfig(
        iters={"pick": iters, "partner": iters, "bury": iters, "play": iters}
    )
    return ISMCTSTeacher(agent, cfg)


# Cell taxonomy shared with the gated teacher (single source of truth): the
# instrument that measured the map and the trainer gate that consumes it must
# classify nodes identically.
_classify = play_cell


def _gumbel_argmax(res: dict) -> int | None:
    gum = res.get("pi_gumbel")
    if gum is None:
        return None
    return int(max(res["valid"], key=lambda a: float(gum[a - 1])))


class _QuotaFilled(Exception):
    """All --cells quotas met; stop replaying seeds."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=400)
    ap.add_argument("--quota", type=int, default=8, help="nodes per cell")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--reuse-ref",
        default=None,
        help="prior matrix JSON: freeze its nodes + reference, rerun only --configs",
    )
    ap.add_argument(
        "--configs",
        default=None,
        help="comma list of arm keys to run (reuse-mode default: d=2 arms; "
        "fresh-mode default: full grid; the reference always runs in fresh mode)",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="fresh mode: comma list of cells to sample (others skipped); "
        "replay stops once all listed cells hit quota",
    )
    ap.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="search-seed replicates per arm per node (certification mode)",
    )
    ap.add_argument(
        "--ref-replicates",
        type=int,
        default=1,
        help="reference replicates per node; root Q is averaged across them",
    )
    args = ap.parse_args()

    prior_pending: dict[int, list[dict]] = {}
    run_keys: set[str] | None = None
    cell_filter = set(args.cells.split(",")) if args.cells else None
    replicate_mode = args.replicates > 1 or args.ref_replicates > 1
    # (cfg_i, iters, depth) per key; cfg_i seeds the per-node RNG. GRID +
    # reference keep their historical indices; extra --configs arms (e.g.
    # "1024/3", "2048/2") get stable indices after them, in sorted order.
    arms: dict[str, tuple[int, int, int]] = {
        f"{i}/{'term' if d == TERMINAL_DEPTH else d}": (cfg_i, i, d)
        for cfg_i, (i, d) in enumerate(GRID + [REFERENCE])
    }
    if args.reuse_ref:
        prior = json.loads(Path(args.reuse_ref).read_text())
        for i, r in enumerate(prior["rows"]):
            r["_node_id"] = i
            prior_pending.setdefault(r["seed"], []).append(r)
    if args.reuse_ref or args.configs:
        run_keys = set((args.configs or "128/2,384/2,1024/2").split(","))
        for j, key in enumerate(sorted(run_keys - set(arms))):
            it_s, d_s = key.split("/")
            depth = TERMINAL_DEPTH if d_s == "term" else int(d_s)
            arms[key] = (len(GRID) + 1 + j, int(it_s), depth)

    driver = load_agent(args.driver)
    # League lineage trains UNDISCOUNTED (train_league_ppo --gamma default
    # 1.0), but checkpoints predating gamma persistence reload with the
    # constructor's historical 0.99, which the ISMCTS teacher would then
    # apply to leaf/terminal values. Pin the search discount to the
    # training objective. (Fixed-length hands make the old 0.99 discount
    # near-uniform per node, so earlier matrices' action orderings stand.)
    driver.gamma = 1.0
    teachers = {
        it: _teacher(driver, it)
        for it in sorted(
            {
                a[1]
                for k, a in arms.items()
                if run_keys is None or k in run_keys or k == f"{REFERENCE[0]}/term"
            }
        )
    }

    cells: dict[str, int] = {}
    rows: list[dict] = []

    try:
        _replay_seeds(
            args,
            driver,
            teachers,
            arms,
            run_keys,
            cell_filter,
            replicate_mode,
            prior_pending,
            cells,
            rows,
        )
    except _QuotaFilled:
        print(f"  all --cells quotas met after {len(rows)} nodes", flush=True)

    unmatched = sum(len(v) for v in prior_pending.values())
    if unmatched:
        print(f"WARNING: {unmatched} prior rows never matched during replay")

    return _summarize(args, arms, cells, rows)


def _replay_seeds(
    args,
    driver,
    teachers,
    arms,
    run_keys,
    cell_filter,
    replicate_mode,
    prior_pending,
    cells,
    rows,
):
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        if args.reuse_ref and not prior_pending.get(seed):
            continue  # games are independent (memory reset per seed)
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
        driver.reset_recurrent_state()
        forced_public: list[tuple[int, int]] = []
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    state = player.get_state_dict()
                    pos = player.position
                    valid_sorted = sorted(valid)
                    is_play = ACTION_LOOKUP.get(valid_sorted[0], "").startswith("PLAY ")

                    logits = _masked_logits(driver, pos, state, valid_sorted).squeeze(0)
                    aid = int(torch.argmax(logits).item()) + 1
                    if aid not in valid:
                        aid = valid_sorted[0]

                    sample = False
                    prior_row = None
                    if (
                        is_play
                        and not game.is_leaster
                        and not game.alone_called
                        and len(valid_sorted) >= 2
                    ):
                        cell = _classify(game, player)
                        if args.reuse_ref:
                            pend = prior_pending.get(seed) or []
                            if (
                                pend
                                and pend[0]["cell"] == cell
                                and pend[0]["policyAction"] == aid
                                and pend[0]["nLegal"] == len(valid_sorted)
                            ):
                                prior_row = pend.pop(0)
                                sample = True
                        elif (cell_filter is None or cell in cell_filter) and cells.get(
                            cell, 0
                        ) < args.quota:
                            sample = True

                    if sample:
                        if prior_row is None:
                            cells[cell] = cells.get(cell, 0) + 1
                            node_id = len(rows)
                        else:
                            cells[cell] = cells.get(cell, 0) + 1
                            node_id = prior_row["_node_id"]
                        node_game = copy.deepcopy(game)
                        row = {
                            "seed": seed,
                            "cell": cell,
                            "nLegal": len(valid_sorted),
                            "policyAction": aid,
                            "policyCard": ACTION_LOOKUP[aid][5:],
                            # Class-level analysis (called-suit fail vs other
                            # fail vs trump) needs the called card at the node.
                            "calledCard": game.called_card or "",
                            "configs": {},
                        }
                        ref_key = f"{REFERENCE[0]}/term"
                        for key, (cfg_i, iters, depth) in sorted(
                            arms.items(), key=lambda kv: kv[1][0]
                        ):
                            is_ref = key == ref_key
                            if run_keys is not None and key not in run_keys:
                                # The reference is ground truth: it always runs
                                # in fresh mode, never in reuse mode (frozen).
                                if not (is_ref and not args.reuse_ref):
                                    continue
                            n_rep = args.ref_replicates if is_ref else args.replicates
                            reps = []
                            for rep in range(n_rep):
                                rng = random.Random(
                                    # Historical scheme for rep 0 keeps
                                    # single-replicate runs comparable to the
                                    # earlier matrices; replicates >0 get a
                                    # disjoint, deterministic stream.
                                    BASE_RNG_SEED + node_id * 100 + cfg_i
                                    if rep == 0
                                    else BASE_RNG_SEED
                                    + 10_000_019
                                    + node_id * 1009
                                    + cfg_i * 31
                                    + rep
                                )
                                t0 = time.perf_counter()
                                res = teachers[iters].search(
                                    node_game,
                                    pos,
                                    list(forced_public),
                                    rng,
                                    d_rollout=depth,
                                )
                                reps.append(
                                    {
                                        "ok": bool(res["ok"]),
                                        "ess": float(res["ess"]),
                                        "sec": round(time.perf_counter() - t0, 3),
                                        "gumbelArgmax": _gumbel_argmax(res),
                                        "rootQ": {
                                            str(a): res["root_q"][a]
                                            for a in res["valid"]
                                        }
                                        if is_ref
                                        else None,
                                    }
                                )
                            if is_ref:
                                qs = [e["rootQ"] for e in reps if e["ok"]]
                                avg = (
                                    {a: sum(q[a] for q in qs) / len(qs) for a in qs[0]}
                                    if qs
                                    else None
                                )
                                row["configs"][key] = {
                                    "ok": bool(qs),
                                    "rootQ": avg,
                                    # Replicate mode judges by averaged root Q
                                    # (sharper ground truth); single-replicate
                                    # keeps the historical pi_gumbel argmax.
                                    "gumbelArgmax": (
                                        int(max(avg, key=avg.get))
                                        if replicate_mode and avg
                                        else reps[0]["gumbelArgmax"]
                                    ),
                                    "repArgmax": [e["gumbelArgmax"] for e in reps],
                                    # Per-replicate root Q: single-search
                                    # trust / Q-margin gate calibration needs
                                    # the un-averaged posteriors.
                                    "reps": reps,
                                }
                            elif replicate_mode:
                                row["configs"][key] = {"reps": reps}
                            else:
                                row["configs"][key] = reps[0]
                        if prior_row is not None:
                            row["configs"][ref_key] = prior_row["configs"][ref_key]
                        ref = row["configs"][ref_key]
                        if ref["ok"] and ref["gumbelArgmax"] is not None:
                            q = {int(a): v for a, v in ref["rootQ"].items()}
                            row["headroom"] = q[ref["gumbelArgmax"]] - q[aid]
                            for key, c in row["configs"].items():
                                if key == ref_key:
                                    continue
                                for e in c.get("reps", [c]):
                                    if e["ok"] and e["gumbelArgmax"] in q:
                                        e["uplift"] = q[e["gumbelArgmax"]] - q[aid]
                                        e["agreeRef"] = (
                                            e["gumbelArgmax"] == ref["gumbelArgmax"]
                                        )
                            rows.append(row)
                            if len(rows) % 10 == 0:
                                print(
                                    f"  [{len(rows)}] cells filled: "
                                    f"{sum(1 for v in cells.values() if v >= args.quota)}"
                                    f"/{len(cells)}",
                                    flush=True,
                                )
                            if cell_filter is not None and all(
                                cells.get(c, 0) >= args.quota for c in cell_filter
                            ):
                                raise _QuotaFilled
                        else:
                            cells[cell] -= 1  # reference unusable; return quota slot

                    if not _is_private_action(aid):
                        forced_public.append((pos, aid))
                    player.act(aid)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat_p in game.players:
                            driver.observe(
                                seat_p.get_last_trick_state_dict(),
                                player_id=seat_p.position,
                            )
                    valid = player.get_valid_action_ids()


def _summarize(args, arms, cells, rows) -> int:
    # ---------------- summary matrix ----------------
    def cell_of(r):
        return r["cell"]

    print(f"\nNodes judged: {len(rows)}  (quota {args.quota}/cell, {len(cells)} cells)")
    keys = [
        k
        for k, (cfg_i, _, _) in sorted(arms.items(), key=lambda kv: kv[1][0])
        if k != f"{REFERENCE[0]}/term"
    ]
    summary = {}
    for cell in sorted(set(map(cell_of, rows))):
        sub = [r for r in rows if r["cell"] == cell]
        head = float(np.mean([r["headroom"] for r in sub]))
        line = {"n": len(sub), "headroom": head, "configs": {}}
        parts = []
        for key in keys:
            cs = [
                e
                for r in sub
                if (c := r["configs"].get(key))
                for e in c.get("reps", [c])
                if e.get("uplift") is not None
            ]
            if not cs:
                continue
            up = float(np.mean([c["uplift"] for c in cs]))
            harm = float(np.mean([c["uplift"] < -HARM_EPS for c in cs]))
            agree = float(np.mean([c["agreeRef"] for c in cs]))
            line["configs"][key] = {
                "uplift": up,
                "harmRate": harm,
                "agreeRef": agree,
                "n": len(cs),
            }
            parts.append(f"{key}: up {up:+.3f} harm {harm:.0%} agr {agree:.0%}")
        summary[cell] = line
        print(f"{cell:>20} n={len(sub):3d} headroom {head:+.3f} | " + " | ".join(parts))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "meta": {
                        "driver": args.driver,
                        "grid": [f"{i}/{d}" for i, d in GRID],
                        "reference": f"{REFERENCE[0]}/{REFERENCE[1]}",
                        "quota": args.quota,
                        "numSeeds": args.num_seeds,
                        "harmEps": HARM_EPS,
                        "leafEvaluator": ISMCTSConfig().leaf_evaluator,
                        "reuseRef": args.reuse_ref,
                        "configsRun": sorted(args.configs.split(","))
                        if args.configs
                        else None,
                        "cells": sorted(args.cells.split(",")) if args.cells else None,
                        "replicates": args.replicates,
                        "refReplicates": args.ref_replicates,
                    },
                    "summary": summary,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"Wrote -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
