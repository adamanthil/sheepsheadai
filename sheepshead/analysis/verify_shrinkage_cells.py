#!/usr/bin/env python3
"""Fresh-draw shrinkage verification at the known-ground-truth cells
(CE_Teacher_Design §1.2 criteria (a)/(b); §10.1 flagged them uncoverable
from archives — the §12.15 EV studies recorded belief-MC deltas, not
committee Q tables, so this instrument generates LIVE committee draws).

Two cell families, sampled from greedy self-play of the 8M seed in
called-ace mode and fed through the PRODUCTION target builder
(``pfsp_runtime.build_ce_search_target``) at trainer defaults:

* **(a) EV-wash cells** — defender leads (tricks 0-2) whose legal fails
  include BOTH a fat (A/10) and a nopoint (7/8/9) card, called-suit fails
  excluded while the called suit is unled (the §12.15
  counterfactual_fat_leads eligibility). Ground truth: class-level EV
  wash (retired from teaching, §12.16). PASS = targets shrink to ~flat
  and any residual class push is small and directionally mixed — the
  teacher must not install a class preference here.
* **(b) called-suit cells** — defender t0 leads holding a called-suit
  fail plus an alternative, called suit unled (the deployable-priority
  convention; EV-backed +0.257 at t0, committee direction 153:7 in
  §12.17). PASS = a healthy material fraction whose tilt moves mass
  TOWARD the called-suit class near-unanimously — shrinkage calibrated
  strongly enough to silence (a) must not also silence this.

The two families bracket ``shrink_s2_global`` from both sides; §10.1's
split-committee tie-band analysis covers stability, this covers
behavior. Per-node cost is one lockstep R=3 committee at 1024/1
(~60 CPU-s), so default quotas run in well under an hour.

Usage:
  uv run python -m sheepshead.analysis.verify_shrinkage_cells \
      --ckpt runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt \
      --quota-wash 36 --quota-called 36 --workers 2 \
      --out runs/ce_teacher_prelaunch/verify_shrinkage_cells.json
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

FAT_RANKS = {"A", "10"}
NOPOINT_RANKS = {"7", "8", "9"}
PUSH_EPS = 0.02  # class-mass move below this is "no push" (tie-band scale)

_W = {}  # per-worker state


# --------------------------------------------------------------------------- #
# Cell classification (mirrors counterfactual_fat_leads / ceiling_h2h)
# --------------------------------------------------------------------------- #
def _lead_class(card: str, called: str | None, called_led: bool) -> str:
    """'trump' | 'called' | 'fat' | 'nopoint' | 'other' for a LEAD of card.
    Called-suit fails are their own class while the called suit is unled
    (they belong to convention (b), never to the fat/nopoint pools)."""
    if card in TRUMP:
        return "trump"
    if called and not called_led and card[-1] == called[-1]:
        return "called"
    rank = card[:-1]
    if rank in FAT_RANKS:
        return "fat"
    if rank in NOPOINT_RANKS:
        return "nopoint"
    return "other"  # K


def _classify_node(game, player, valid) -> dict | None:
    """Cell flags at a defender lead in a standard called-ace game, else
    None. ``classes`` maps each legal action id to its lead class."""
    if game.is_leaster or game.alone_called or not game.play_started:
        return None
    if game.cards_played != 0 or game.leader != player.position:
        return None
    if player.is_picker or player.is_partner or player.is_secret_partner:
        return None
    if len(valid) < 2:
        return None
    called = game.called_card
    called_led = bool(game.was_called_suit_played)
    classes = {}
    for a in valid:
        name = ACTIONS[a - 1]
        if not name.startswith("PLAY "):
            return None
        classes[a] = _lead_class(name[5:], called, called_led)
    trick = int(game.current_trick)
    kinds = set(classes.values())
    wash = trick <= 2 and "fat" in kinds and "nopoint" in kinds
    called_cell = trick == 0 and "called" in kinds and kinds != {"called"}
    if not wash and not called_cell:
        return None
    return {
        "trick": trick,
        "wash": wash,
        "called_cell": called_cell,
        "classes": {int(a): c for a, c in classes.items()},
    }


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
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


def _target_row(game, player, valid, forced_public, deal_seed, node_idx, cell):
    """One committee draw -> production target -> per-class mass ledger."""
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
    prior = prior / prior.sum()  # the w=0 target, exactly
    classes = cell["classes"]
    mass = {}
    for name in ("trump", "called", "fat", "nopoint", "other"):
        idx = [i for i, a in enumerate(acts) if classes[a] == name]
        mass[name] = {
            "prior": float(sum(prior[i] for i in idx)),
            "target": float(sum(target[i] for i in idx)),
        }
    return {
        "deal_seed": deal_seed,
        "node": node_idx,
        "trick": cell["trick"],
        "wash": cell["wash"],
        "called_cell": cell["called_cell"],
        "n_valid": len(acts),
        "w": info["w"],
        "spread": info["spread"],
        "class_mass": mass,
        "prior_argmax_class": classes[acts[int(np.argmax(prior))]],
        "target_argmax_class": classes[acts[int(np.argmax(target))]],
    }


def _run_deal(deal_seed):
    """Greedy self-play of one called-ace deal; committee at eligible cells
    (cap 2 per family per deal). Returns node rows."""
    from sheepshead.ismcts import is_private_action

    agent = _W["agent"]
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=deal_seed)
    agent.reset_recurrent_state()
    forced_public = []
    rows = []
    node_idx = 0
    seen = {"wash": 0, "called_cell": 0}
    t0 = time.time()
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                state = player.get_state_dict()
                action, _, _ = agent.act(
                    state, valid, player.position, deterministic=True
                )
                cell = _classify_node(game, player, sorted(valid))
                if cell is not None:
                    if seen["wash"] >= 2:
                        cell["wash"] = False
                    if seen["called_cell"] >= 2:
                        cell["called_cell"] = False
                if cell is not None and (cell["wash"] or cell["called_cell"]):
                    row = _target_row(
                        game,
                        player,
                        sorted(valid),
                        forced_public,
                        deal_seed,
                        node_idx,
                        cell,
                    )
                    if row is not None:
                        rows.append(row)
                        seen["wash"] += int(cell["wash"])
                        seen["called_cell"] += int(cell["called_cell"])
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


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _push(row, name):
    m = row["class_mass"][name]
    return m["target"] - m["prior"]


def _summarize(rows):
    wash = [r for r in rows if r["wash"]]
    called = [r for r in rows if r["called_cell"]]

    def base(rs):
        w = np.array([r["w"] for r in rs]) if rs else np.array([0.0])
        return {
            "n": len(rs),
            "w_mean": float(w.mean()),
            "w_median": float(np.median(w)),
            "frac_w0": float((w == 0.0).mean()),
            "frac_material": float((w > 0.0).mean()),
        }

    out = {"wash": base(wash), "called": base(called)}

    # (a): residual class push fat<->nopoint must be small AND mixed.
    d = [_push(r, "nopoint") - _push(r, "fat") for r in wash]
    out["wash"].update(
        {
            "push_np_minus_fat_mean": float(np.mean(d)) if d else 0.0,
            "push_abs_mean": float(np.mean(np.abs(d))) if d else 0.0,
            "n_push_nopoint": sum(1 for x in d if x > PUSH_EPS),
            "n_push_fat": sum(1 for x in d if x < -PUSH_EPS),
            "n_no_push": sum(1 for x in d if abs(x) <= PUSH_EPS),
        }
    )
    # (b): material tilts must move mass toward the called suit.
    material = [r for r in called if r["w"] > 0.0]
    p = [_push(r, "called") for r in material]
    out["called"].update(
        {
            "material_push_called_mean": float(np.mean(p)) if p else 0.0,
            "n_toward_called": sum(1 for x in p if x > PUSH_EPS),
            "n_away_called": sum(1 for x in p if x < -PUSH_EPS),
            "n_neutral": sum(1 for x in p if abs(x) <= PUSH_EPS),
            "argmax_installs": sum(
                1
                for r in material
                if r["prior_argmax_class"] != "called"
                and r["target_argmax_class"] == "called"
            ),
            "argmax_removals": sum(
                1
                for r in material
                if r["prior_argmax_class"] == "called"
                and r["target_argmax_class"] != "called"
            ),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=1024)
    ap.add_argument("--quota-wash", type=int, default=36)
    ap.add_argument("--quota-called", type=int, default=36)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=500_000)
    ap.add_argument("--max-deals", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    n_wash = n_called = 0
    ctx = get_context("spawn")
    with ctx.Pool(
        args.workers, initializer=_worker_init, initargs=(args.ckpt, args.iters)
    ) as pool:
        seeds = [args.base_seed + i for i in range(args.max_deals)]
        for res in pool.imap_unordered(_run_deal, seeds, chunksize=1):
            for row in res["rows"]:
                # Trim overshoot so the report matches the quotas.
                if row["wash"] and n_wash >= args.quota_wash:
                    row["wash"] = False
                if row["called_cell"] and n_called >= args.quota_called:
                    row["called_cell"] = False
                if not (row["wash"] or row["called_cell"]):
                    continue
                rows.append(row)
                n_wash += int(row["wash"])
                n_called += int(row["called_cell"])
            if res["rows"]:
                print(
                    f"[{n_wash}/{args.quota_wash} wash, "
                    f"{n_called}/{args.quota_called} called] "
                    f"deal {res['deal_seed']} +{len(res['rows'])} rows "
                    f"({res['wall_s']:.0f}s)",
                    flush=True,
                )
            if n_wash >= args.quota_wash and n_called >= args.quota_called:
                pool.terminate()
                break

    summary = _summarize(rows)
    report = {
        "instrument": "verify_shrinkage_cells",
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
