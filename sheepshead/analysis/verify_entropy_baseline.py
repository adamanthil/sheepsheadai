#!/usr/bin/env python3
"""Validate the CE-target entropy-injection mechanism and the baseline fix.

Attempt-11 observation (pre-registered expectation #4 failing): play/partner
Hn rising above target with the v2 controller pinned at the -0.05 clamp.
Candidate mechanisms (the root-level Jensen story was FALSIFIED by the
first smoke: at the root every determinized world presents the same
info-state, so the pooled root prior equals the expert's policy there —
gap ~1e-7 when expert = student):

  (1) TILT SOFTENING at material rows — when the policy is sharp but the
      committee Q-gap is modest, the pi_gumbel tilt redistributes mass
      and the target is SOFTER than the policy (smoke: KL(t_cur||pi)
      ~0.35, H(target) > H(pi) at material rows even with expert =
      student). Four CE epochs on ~10%% of rows generalize the pull.
  (2) FROZEN-EXPERT ANCHOR under drift — in the live trainer p_raw is
      the gen-start expert's prior while pi moves, so w = 0 rows acquire
      a nonzero CE pull toward SEED-entropy levels as the gen ages.
      Requires --teacher-ckpt != --ckpt to observe.

The baseline swap (``base_prior`` = student's label-time policy in
``build_ce_search_target``) eliminates (2) exactly (abstention becomes
zero-gradient by construction) and converts (1) from an absolute
re-anchoring into a pure evidence tilt off the student's own
distribution; whether the remaining tilt still injects is precisely
what the material-row entropy deltas measure.

At emission-eligible play nodes (standard game, >= 2 legal, stochastic
self-play of the given checkpoint — the training rollout distribution),
this instrument records four distributions over the legal set:

  pi      — the student policy at the info-state (the CE gradient's anchor)
  p_raw   — the pooled expert prior (current baseline)
  t_cur   — production target, default baseline
  t_fix   — production target, base_prior = pi

and reports, split by materiality:

  MECHANISM — mean normalized entropy of each; Jensen gap
      H(p_raw) - H(pi) (predicted > 0); CE pull KL(t_cur || pi) at
      w = 0 rows (predicted > 0 — the design said ~ 0) vs
      KL(t_fix || pi) (exactly 0 by construction).
  FIX SAFETY — at material rows: per-class mass push under t_cur vs
      t_fix (sign agreement + correlation) and argmax agreement, i.e.
      the teaching signal must survive the baseline swap.

Usage:
  uv run python -m sheepshead.analysis.verify_entropy_baseline \
      --ckpt runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt \
      --quota 120 --workers 2 \
      --out runs/ce_teacher_prelaunch/verify_entropy_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, Game

_W = {}  # per-worker state


def _worker_init(ckpt, iters, node_prob, teacher_ckpt=None):
    import torch

    torch.set_num_threads(1)
    from sheepshead.agent.ppo import load_agent
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
    from sheepshead.training.config import SearchConfig

    _W["agent"] = load_agent(ckpt)
    _W["teacher"] = ISMCTSTeacher(
        load_agent(teacher_ckpt or ckpt),
        ISMCTSConfig(iters={h: iters for h in ("pick", "partner", "bury", "play")}),
    )
    _W["search_cfg"] = SearchConfig()
    _W["node_prob"] = node_prob


def _entropy_norm(p) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    p = p / p.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / np.log(len(p)) if len(p) > 1 else 0.0


def _kl(p, q) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _node_row(game, player, valid, forced_public, deal_seed, node_idx, pi):
    from sheepshead.training.pfsp_runtime import build_ce_search_target

    teacher, cfg = _W["teacher"], _W["search_cfg"]
    rngs = [
        random.Random(hash((deal_seed, node_idx, rep)) & 0x7FFFFFFF)
        for rep in range(cfg.teacher_replicates)
    ]
    replicates = teacher.search_committee(
        game,
        player.position,
        list(forced_public),
        rngs,
        d_rollout=cfg.teacher_d_rollout,
    )
    kwargs = dict(
        shrink_nu=cfg.shrink_nu,
        shrink_s2_global=cfg.shrink_s2_global,
        gumbel_c_visit=teacher.config.gumbel_c_visit,
        gumbel_c_scale=teacher.config.gumbel_c_scale,
    )
    built = build_ce_search_target(replicates, valid, **kwargs)
    if built is None:
        return None
    t_cur, info = built
    built_fix = build_ce_search_target(replicates, valid, base_prior=pi, **kwargs)
    t_fix = built_fix[0]
    acts = sorted(valid)
    usable = [
        r
        for r in replicates
        if r["ok"] and r.get("root_q") is not None and r.get("root_prior") is not None
    ]
    p_raw = np.array(
        [np.mean([r["root_prior"][a] for r in usable]) for a in acts],
        dtype=np.float64,
    )
    p_raw = p_raw / p_raw.sum()
    return {
        "deal_seed": deal_seed,
        "node": node_idx,
        "trick": int(game.current_trick),
        "n_valid": len(acts),
        "w": info["w"],
        "H_pi": _entropy_norm(pi),
        "H_praw": _entropy_norm(p_raw),
        "H_tcur": _entropy_norm(t_cur),
        "H_tfix": _entropy_norm(t_fix),
        "kl_tcur_pi": _kl(t_cur, pi),
        "kl_tfix_pi": _kl(t_fix, pi),
        "kl_praw_pi": _kl(p_raw, pi),
        "argmax_agree": bool(int(np.argmax(t_cur)) == int(np.argmax(t_fix))),
        "push_cur": [float(t_cur[i] - p_raw[i]) for i in range(len(acts))],
        "push_fix": [float(t_fix[i] - pi[i]) for i in range(len(acts))],
        "cards": [ACTIONS[a - 1][5:] for a in acts],
    }


def _run_deal(deal_seed):
    """Stochastic self-play (training rollout distribution); committee at
    emission-eligible play nodes sampled at node_prob, cap 2 per deal."""
    from sheepshead.ismcts import is_private_action

    agent = _W["agent"]
    rng = random.Random(deal_seed ^ 0xA5A5A5)
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=deal_seed)
    agent.reset_recurrent_state()
    forced_public = []
    rows = []
    node_idx = 0
    t0 = time.time()
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                state = player.get_state_dict()
                eligible = (
                    game.play_started
                    and not game.is_leaster
                    and not game.alone_called
                    and len(valid) >= 2
                    and len(rows) < 2
                    and rng.random() < _W["node_prob"]
                )
                if eligible:
                    acts = sorted(valid)
                    probs, _ = agent.get_action_probs_with_logits(
                        state, valid, player_id=player.position
                    )
                    pi = np.array(
                        [float(probs[0][a - 1]) for a in acts], dtype=np.float64
                    )
                    pi = pi / pi.sum()
                    row = _node_row(
                        game, player, acts, forced_public, deal_seed, node_idx, pi
                    )
                    if row is not None:
                        rows.append(row)
                action, _, _ = agent.act(
                    state, valid, player.position, deterministic=False
                )
                node_idx += 1
                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(),
                            player_id=seat.position,
                        )
    return {"deal_seed": deal_seed, "rows": rows, "wall_s": time.time() - t0}


def summarize(rows) -> dict:
    def block(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "H_pi": float(np.mean([r["H_pi"] for r in rs])),
            "H_praw": float(np.mean([r["H_praw"] for r in rs])),
            "H_tcur": float(np.mean([r["H_tcur"] for r in rs])),
            "H_tfix": float(np.mean([r["H_tfix"] for r in rs])),
            "jensen_gap_mean": float(np.mean([r["H_praw"] - r["H_pi"] for r in rs])),
            "jensen_gap_frac_pos": float(
                np.mean([r["H_praw"] > r["H_pi"] for r in rs])
            ),
            "kl_tcur_pi_mean": float(np.mean([r["kl_tcur_pi"] for r in rs])),
            "kl_tfix_pi_mean": float(np.mean([r["kl_tfix_pi"] for r in rs])),
        }

    w0 = [r for r in rows if r["w"] == 0.0]
    mat = [r for r in rows if r["w"] > 0.0]
    out = {
        "all": block(rows),
        "abstain_w0": block(w0),
        "material": block(mat),
    }
    # Fix safety at material rows: does the teaching push survive?
    if mat:
        agree = float(np.mean([r["argmax_agree"] for r in mat]))
        sign_agree, corrs = [], []
        for r in mat:
            pc, pf = np.array(r["push_cur"]), np.array(r["push_fix"])
            big = np.abs(pc) > 0.02
            if big.any():
                sign_agree.append(float(np.mean(np.sign(pc[big]) == np.sign(pf[big]))))
            if len(pc) > 1 and pc.std() > 0 and pf.std() > 0:
                corrs.append(float(np.corrcoef(pc, pf)[0, 1]))
        out["fix_safety"] = {
            "argmax_agree_frac": agree,
            "push_sign_agree_mean": float(np.mean(sign_agree)) if sign_agree else None,
            "push_corr_mean": float(np.mean(corrs)) if corrs else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student policy checkpoint (pi)")
    ap.add_argument(
        "--teacher-ckpt",
        default=None,
        help="frozen expert checkpoint for the committee (defaults to --ckpt); "
        "decoupling the two measures the anchor gap under student drift",
    )
    ap.add_argument("--iters", type=int, default=1024)
    ap.add_argument("--quota", type=int, default=120)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--node-prob", type=float, default=0.15)
    ap.add_argument("--base-seed", type=int, default=900_000)
    ap.add_argument("--max-deals", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(args.ckpt, args.iters, args.node_prob, args.teacher_ckpt),
    ) as pool:
        seeds = [args.base_seed + i for i in range(args.max_deals)]
        for res in pool.imap_unordered(_run_deal, seeds, chunksize=1):
            for row in res["rows"]:
                rows.append(row)
                print(
                    f"[{len(rows)}/{args.quota}] deal {row['deal_seed']} t{row['trick']} "
                    f"w={row['w']:.2f} H(pi)={row['H_pi']:.3f} "
                    f"H(praw)={row['H_praw']:.3f} "
                    f"kl_cur={row['kl_tcur_pi']:.3f} kl_fix={row['kl_tfix_pi']:.3f}",
                    flush=True,
                )
            if len(rows) >= args.quota:
                pool.terminate()
                break

    summary = summarize(rows)
    report = {
        "instrument": "verify_entropy_baseline",
        "ckpt": args.ckpt,
        "teacher_ckpt": args.teacher_ckpt or args.ckpt,
        "iters": args.iters,
        "node_prob": args.node_prob,
        "summary": summary,
        "rows": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
