#!/usr/bin/env python3
"""Where does search say NOT to lead the called suit at trick 0?

The human convention is unconditional: a defender holding called-suit
fail leads it at the first opportunity. The instruments agree only on
average — EV-backed +0.257 at t0 (§12.15) with a known negative pocket
at picker+1, and the §13.3 ceiling committee raised t0 adherence to
56% while still deviating at ~44% of nodes. This instrument samples
FRESH defender-on-lead t0 nodes holding a called-suit fail, runs the
certified committee (lockstep R=3 @ 1024/1) on the 8M seed, and logs
per-node hand/position features alongside the production CE target
(``build_ce_search_target``) and the per-replicate pi_gumbel votes, so
the exception structure is readable in OBSERVABLE terms (hand shape,
seat relative to picker) rather than as an unexplained deviation rate.

Per-node categories (production-target read; PUSH_EPS tie band):
  SUPPORT — target argmax class 'called' or called-mass push > +eps
  AGAINST — material (w > 0), target argmax non-called AND push < −eps
  NEUTRAL — w = 0 or |push| <= eps (abstention: convention neither
            reinforced nor contradicted at this node)

Privileged fields (true partner seat) are recorded for mechanism
analysis only and flagged as such — a playable convention can condition
only on the hand, the picker's seat, and public history.

Usage:
  uv run python -m sheepshead.analysis.called_suit_exceptions \
      --ckpt runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt \
      --quota 250 --workers 3 \
      --out runs/called_suit_exceptions_202608/nodes.json
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

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, TRUMP, Game
from sheepshead.analysis.verify_shrinkage_cells import _lead_class
from sheepshead.game import get_card_points

PUSH_EPS = 0.02
QUEENS = {"QC", "QS", "QH", "QD"}
JACKS = {"JC", "JS", "JH", "JD"}

_W = {}  # per-worker state


def _eligible_t0_lead(game, player, valid) -> dict | None:
    """Classes per legal action at a defender t0 lead holding a
    called-suit fail (with at least one alternative), else None."""
    if game.is_leaster or game.alone_called or not game.play_started:
        return None
    if game.current_trick != 0 or game.cards_played != 0:
        return None
    if game.leader != player.position:
        return None
    if player.is_picker or player.is_partner or player.is_secret_partner:
        return None
    if len(valid) < 2:
        return None
    called = game.called_card
    classes = {}
    for a in valid:
        name = ACTIONS[a - 1]
        if not name.startswith("PLAY "):
            return None
        classes[a] = _lead_class(name[5:], called, False)
    kinds = set(classes.values())
    if "called" not in kinds or kinds == {"called"}:
        return None
    return classes


def _hand_features(game, player) -> dict:
    called_suit = game.called_card[-1]
    hand = sorted(player.hand)
    called_fails = [c for c in hand if c not in TRUMP and c[-1] == called_suit]
    fail_suits = {c[-1] for c in hand if c not in TRUMP}
    return {
        "hand": hand,
        "called_card": game.called_card,
        "seat_rel_picker": (player.position - game.picker) % 5,
        "called_len": len(called_fails),
        "called_ranks": sorted(c[:-1] for c in called_fails),
        "has_10_called": any(c[:-1] == "10" for c in called_fails),
        "has_K_called": any(c[:-1] == "K" for c in called_fails),
        "trump_count": sum(1 for c in hand if c in TRUMP),
        "n_queens": sum(1 for c in hand if c in QUEENS),
        "n_jacks": sum(1 for c in hand if c in JACKS),
        "hand_points": sum(get_card_points(c) for c in hand),
        "n_fail_suits": len(fail_suits),
        # privileged (mechanism analysis only — hidden from a defender):
        "partner_rel_picker_PRIV": (game.partner - game.picker) % 5,
        "partner_rel_leader_PRIV": (game.partner - player.position) % 5,
    }


def _worker_init(ckpt, iters):
    import torch

    torch.set_num_threads(1)
    from sheepshead.agent.ppo import load_agent
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
    from sheepshead.training.config import SearchConfig

    _W["agent"] = load_agent(ckpt)
    _W["teacher"] = ISMCTSTeacher(
        load_agent(ckpt),
        ISMCTSConfig(iters={h: iters for h in ("pick", "partner", "bury", "play")}),
    )
    _W["search_cfg"] = SearchConfig()


def _committee_row(game, player, valid, forced_public, deal_seed, classes):
    from sheepshead.training.pfsp_runtime import build_ce_search_target

    teacher, cfg = _W["teacher"], _W["search_cfg"]
    rngs = [
        random.Random(hash((deal_seed, "cse", rep)) & 0x7FFFFFFF)
        for rep in range(cfg.teacher_replicates)
    ]
    replicates = teacher.search_committee(
        game,
        player.position,
        list(forced_public),
        rngs,
        d_rollout=cfg.teacher_d_rollout,
    )
    built = build_ce_search_target(
        replicates,
        valid,
        shrink_nu=cfg.shrink_nu,
        shrink_s2_global=cfg.shrink_s2_global,
        gumbel_c_visit=teacher.config.gumbel_c_visit,
        gumbel_c_scale=teacher.config.gumbel_c_scale,
    )
    if built is None:
        return None
    target, info = built
    acts = sorted(valid)
    usable = [
        r
        for r in replicates
        if r["ok"] and r.get("root_q") is not None and r.get("root_prior") is not None
    ]
    prior = np.array(
        [np.mean([r["root_prior"][a] for r in usable]) for a in acts],
        dtype=np.float64,
    )
    prior = prior / prior.sum()

    # Per-replicate pi_gumbel argmax votes (ceiling_h2h semantics).
    votes = {}
    for r in usable:
        readout = r.get("pi_gumbel")
        if readout is None:
            continue
        choice = max(acts, key=lambda a: float(readout[a - 1]))
        votes[choice] = votes.get(choice, 0) + 1
    vote_winner = None
    if votes:
        top, count = max(votes.items(), key=lambda kv: kv[1])
        if count >= 2:
            vote_winner = top

    actions = []
    for i, a in enumerate(acts):
        card = ACTIONS[a - 1][5:]
        qs = [float(r["root_q"][a]) for r in usable if r["root_n"].get(a, 0.0) > 0.0]
        actions.append(
            {
                "card": card,
                "class": classes[a],
                "q_pooled": float(np.mean(qs)) if qs else None,
                "q_se": (
                    float(np.std(qs, ddof=1) / np.sqrt(len(qs)))
                    if len(qs) > 1
                    else None
                ),
                "visits_mean": float(
                    np.mean([r["root_n"].get(a, 0.0) for r in usable])
                ),
                "prior": float(prior[i]),
                "target": float(target[i]),
            }
        )
    mass = {}
    for name in ("trump", "called", "fat", "nopoint", "other"):
        idx = [i for i, a in enumerate(acts) if classes[a] == name]
        mass[name] = {
            "prior": float(sum(prior[i] for i in idx)),
            "target": float(sum(target[i] for i in idx)),
        }
    push = mass["called"]["target"] - mass["called"]["prior"]
    target_argmax = acts[int(np.argmax(target))]
    prior_argmax = acts[int(np.argmax(prior))]
    if info["w"] > 0.0 and classes[target_argmax] != "called" and push < -PUSH_EPS:
        category = "AGAINST"
    elif classes[target_argmax] == "called" or push > PUSH_EPS:
        category = "SUPPORT"
    else:
        category = "NEUTRAL"
    return {
        "deal_seed": deal_seed,
        "w": info["w"],
        "spread": info["spread"],
        "n_valid": len(acts),
        "actions": actions,
        "class_mass": mass,
        "called_push": push,
        "prior_argmax": ACTIONS[prior_argmax - 1][5:],
        "prior_argmax_class": classes[prior_argmax],
        "target_argmax": ACTIONS[target_argmax - 1][5:],
        "target_argmax_class": classes[target_argmax],
        "votes": {ACTIONS[a - 1][5:]: n for a, n in votes.items()},
        "vote_winner": ACTIONS[vote_winner - 1][5:] if vote_winner else None,
        "vote_winner_class": classes[vote_winner] if vote_winner else None,
        "category": category,
    }


def _run_deal(deal_seed):
    """Greedy self-play to the trick-0 lead; committee if eligible.
    Each deal yields at most one node, so no per-deal caps exist."""
    from sheepshead.ismcts import is_private_action

    agent = _W["agent"]
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=deal_seed)
    agent.reset_recurrent_state()
    forced_public = []
    t0 = time.time()
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                if game.play_started:
                    classes = _eligible_t0_lead(game, player, sorted(valid))
                    row = None
                    if classes is not None:
                        row = _committee_row(
                            game,
                            player,
                            sorted(valid),
                            forced_public,
                            deal_seed,
                            classes,
                        )
                        if row is not None:
                            row.update(_hand_features(game, player))
                            row["wall_s"] = time.time() - t0
                    # t0 lead reached (eligible or not): the deal is spent.
                    return row
                state = player.get_state_dict()
                action, _, _ = agent.act(
                    state, valid, player.position, deterministic=True
                )
                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)
                valid = player.get_valid_action_ids()
    return None


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _feature_means(rows, keys):
    return {k: (float(np.mean([r[k] for r in rows])) if rows else None) for k in keys}


def summarize(rows) -> dict:
    cats = {
        c: [r for r in rows if r["category"] == c]
        for c in ("SUPPORT", "AGAINST", "NEUTRAL")
    }
    num_keys = (
        "called_len",
        "trump_count",
        "n_queens",
        "n_jacks",
        "hand_points",
        "n_fail_suits",
        "w",
    )
    out = {"n": len(rows)}
    for c, rs in cats.items():
        out[c] = {
            "n": len(rs),
            "features": _feature_means(rs, num_keys),
            "frac_has_10_called": (
                float(np.mean([r["has_10_called"] for r in rs])) if rs else None
            ),
            "frac_singleton_called": (
                float(np.mean([r["called_len"] == 1 for r in rs])) if rs else None
            ),
            "seat_rel_picker_hist": {
                s: sum(1 for r in rs if r["seat_rel_picker"] == s) for s in (1, 2, 3, 4)
            },
        }
    against = cats["AGAINST"]
    out["against_winner_class"] = {
        c: sum(1 for r in against if r["target_argmax_class"] == c)
        for c in ("trump", "fat", "nopoint", "other")
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=1024)
    ap.add_argument("--quota", type=int, default=250)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=700_000)
    ap.add_argument("--max-deals", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(
        args.workers, initializer=_worker_init, initargs=(args.ckpt, args.iters)
    ) as pool:
        seeds = [args.base_seed + i for i in range(args.max_deals)]
        for row in pool.imap_unordered(_run_deal, seeds, chunksize=1):
            if row is None:
                continue
            rows.append(row)
            print(
                f"[{len(rows)}/{args.quota}] deal {row['deal_seed']} "
                f"{row['category']} target={row['target_argmax']} "
                f"({row['target_argmax_class']}) push={row['called_push']:+.3f} "
                f"w={row['w']:.2f} seat+{row['seat_rel_picker']} "
                f"({row['wall_s']:.0f}s)",
                flush=True,
            )
            if len(rows) >= args.quota:
                pool.terminate()
                break

    summary = summarize(rows)
    report = {
        "instrument": "called_suit_exceptions",
        "ckpt": args.ckpt,
        "iters": args.iters,
        "push_eps": PUSH_EPS,
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
