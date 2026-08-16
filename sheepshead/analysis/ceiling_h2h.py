#!/usr/bin/env python3
"""Ceiling h2h: committee-act search agent vs its own raw policy.

Measures the one-step search-improvement ceiling for the always-on search
teacher (Search_Teacher_Design §13.3): the hero plays the SAME weights as the
anchor field, but at every unforced PLAY decision (>= 2 legal actions —
class-blind: no confidence trigger, no convention special-casing, per the
operator's ceiling-measurement directive) it runs the E9-certified read-time
committee — R=3 replicates at the gate budget (1024 iters, d_rollout 1,
lockstep-batched), acting on a 2-of-3 top-action agreement and falling back
to the policy argmax otherwise (abstention on splits / ESS failures). If this
agent cannot beat the raw policy, always-on teaching has no EV signal to
distill at this budget; if it can, the edge is the per-generation ceiling.

Instrument matches h2h_duplicate (league_progress_eval): candidate seated in
all 5 seats per CRN deal against an all-anchor field, both partner modes,
per-deal mean score = the paired edge (anchor's own-field score is 0 by
symmetry). Deal seeds use the same generator schedule (seed 42).

Instrumentation (the run doubles as a behavioral study): every hero play-lead
node logs trick, convention-cell eligibility, the policy argmax, the
committee's per-replicate choices, resolution/abstention, and the acted card
— so all three conventions (defender t0 trump-lead, partner trump-lead,
defender called-suit lead) come out binned by trick for BOTH the search-acted
arm and the policy-argmax counterfactual at identical nodes.

Usage:
  uv run python -m sheepshead.analysis.ceiling_h2h \
      --ckpt <8M.pt> --deals-per-mode 250 --workers 7 \
      --out-json runs/.../ceiling_h2h.json --node-log runs/.../nodes.jsonl
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

from sheepshead import (
    ACTIONS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP,
    Game,
)

R_REPLICATES = 3
VOTES_NEEDED = 2
GATE_ITERS = 1024
GATE_D_ROLLOUT = 1
H2H_SEED = 42

_W = {}  # per-worker state


# --------------------------------------------------------------------------- #
# Node classification (calibrated-instrument definitions, generalized by trick)
# --------------------------------------------------------------------------- #
def _classify_lead_cells(game, player) -> dict | None:
    """At a lead decision (first card of a trick, standard game), return the
    convention-cell eligibility flags, else None."""
    if game.is_leaster or game.alone_called or not game.play_started:
        return None
    if game.cards_played != 0 or game.leader != player.position:
        return None
    is_partnerish = player.is_picker or player.is_partner or player.is_secret_partner
    has_trump = any(c in TRUMP for c in player.hand)
    has_fail = any(c not in TRUMP for c in player.hand)
    called = game.called_card
    cells = {
        "trick": int(game.current_trick),
        "def_lead": False,
        "partner_lead": False,
        "called_suit": False,
    }
    if not is_partnerish and has_trump and has_fail:
        cells["def_lead"] = True
    if player.is_secret_partner and not player.is_picker and has_trump and has_fail:
        cells["partner_lead"] = True
    if (
        not is_partnerish
        and bool(called)
        and not game.was_called_suit_played
        and any(c not in TRUMP and c[-1] == called[-1] for c in player.hand)
        and any(c in TRUMP or c[-1] != called[-1] for c in player.hand)
    ):
        cells["called_suit"] = True
    return cells


def _adherence(action_id: int, cells: dict, called_card) -> dict:
    """Convention adherence booleans for playing ``action_id`` at a node with
    ``cells`` eligibility."""
    name = ACTIONS[action_id - 1]
    card = name[5:] if name.startswith("PLAY ") else None
    out = {}
    if card is not None:
        if cells["def_lead"]:
            out["def_lead_no_trump"] = card not in TRUMP
        if cells["partner_lead"]:
            out["partner_trump"] = card in TRUMP
        if cells["called_suit"]:
            out["called_suit"] = card not in TRUMP and card[-1] == called_card[-1]
    return out


# --------------------------------------------------------------------------- #
# Committee-act hero
# --------------------------------------------------------------------------- #
def _committee_act(game, player, valid, forced_public, node_key):
    """E9 read-time committee at an unforced play node: R replicates in
    lockstep, per-replicate choice = pi_gumbel argmax (the adopted deploy
    readout), act on a >= 2-vote winner, abstain to the policy argmax."""
    teacher = _W["teacher"]
    rngs = [
        random.Random(hash((node_key, rep)) & 0x7FFFFFFF) for rep in range(R_REPLICATES)
    ]
    results = teacher.search_committee(
        game, player.position, list(forced_public), rngs, d_rollout=GATE_D_ROLLOUT
    )
    votes = {}
    n_ok = 0
    for res in results:
        if not res["ok"]:
            continue
        readout = res["pi_gumbel"] if res["pi_gumbel"] is not None else res["pi"]
        if readout is None:
            continue
        n_ok += 1
        choice = max(res["valid"], key=lambda a: float(readout[a - 1]))
        votes[choice] = votes.get(choice, 0) + 1
    winner = None
    if votes:
        top, count = max(votes.items(), key=lambda kv: kv[1])
        if count >= VOTES_NEEDED:
            winner = top
    return winner, votes, n_ok


def _play_hand_instrumented(mode, deal_seed, hero_seat, deal_idx):
    """One hand: hero (committee-act) at ``hero_seat``, anchor field
    elsewhere. Mirrors rigorous_eval.play_hand stepping exactly (deterministic
    acts, end-of-trick observe propagation). Returns (hero_score, node_rows)."""
    hero, anchor = _W["hero"], _W["anchor"]
    game = Game(partner_selection_mode=mode, seed=deal_seed)
    hero.reset_recurrent_state()
    anchor.reset_recurrent_state()
    forced_public = []
    node_rows = []
    node_idx = 0
    from sheepshead.ismcts import is_private_action

    while not game.is_done():
        for player in game.players:
            agent = hero if player.position == hero_seat else anchor
            valid = player.get_valid_action_ids()
            while valid:
                state = player.get_state_dict()
                action, _, _ = agent.act(
                    state, valid, player.position, deterministic=True
                )
                is_hero = player.position == hero_seat
                is_play = ACTIONS[action - 1].startswith("PLAY ")
                if is_hero and is_play and len(valid) >= 2:
                    policy_action = action
                    node_key = (mode, deal_seed, hero_seat, node_idx)
                    winner, votes, n_ok = _committee_act(
                        game, player, sorted(valid), forced_public, node_key
                    )
                    acted = winner if winner is not None else policy_action
                    cells = _classify_lead_cells(game, player)
                    row = {
                        "deal": deal_idx,
                        "mode": "called" if mode == PARTNER_BY_CALLED_ACE else "jd",
                        "seat": hero_seat,
                        "trick": int(game.current_trick),
                        "n_valid": len(valid),
                        "resolved": winner is not None,
                        "deviated": acted != policy_action,
                        "ok_replicates": n_ok,
                        "votes": {str(k): v for k, v in votes.items()},
                        "policy_action": policy_action,
                        "acted": acted,
                    }
                    if cells is not None:
                        called = game.called_card
                        row["cells"] = cells
                        row["adh_policy"] = _adherence(policy_action, cells, called)
                        row["adh_acted"] = _adherence(acted, cells, called)
                    node_rows.append(row)
                    node_idx += 1
                    action = acted
                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        (hero if seat.position == hero_seat else anchor).observe(
                            seat.get_last_trick_state_dict(),
                            player_id=seat.position,
                        )
    return float(game.players[hero_seat - 1].get_score()), node_rows


def _worker_init(ckpt, torch_threads, iters):
    import torch

    torch.set_num_threads(torch_threads)
    from sheepshead.agent.ppo import load_agent
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    _W["hero"] = load_agent(ckpt)
    _W["anchor"] = load_agent(ckpt)
    _W["teacher"] = ISMCTSTeacher(
        load_agent(ckpt),
        ISMCTSConfig(iters={h: iters for h in ("pick", "partner", "bury", "play")}),
    )


def _run_deal(task):
    """One CRN deal in one mode: hero in all 5 seats. Returns the per-deal
    mean hero score (the duplicate edge contribution) + node rows."""
    mode, deal_seed, deal_idx = task
    t0 = time.time()
    scores = []
    rows = []
    for hero_seat in range(1, 6):
        score, node_rows = _play_hand_instrumented(mode, deal_seed, hero_seat, deal_idx)
        scores.append(score)
        rows.extend(node_rows)
    return {
        "deal": deal_idx,
        "mode": "called" if mode == PARTNER_BY_CALLED_ACE else "jd",
        "deal_score": float(np.mean(scores)),
        "rows": rows,
        "wall_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _adherence_table(rows, which):
    """conventions x trick bins -> [eligible, adherent] from node rows;
    ``which`` is 'adh_policy' or 'adh_acted'."""
    conv_keys = {
        "def_lead_no_trump": "def_lead",
        "partner_trump": "partner_lead",
        "called_suit": "called_suit",
    }
    table = {c: {} for c in conv_keys}
    for row in rows:
        adh = row.get(which)
        if not adh:
            continue
        trick = row["cells"]["trick"]
        for conv in adh:
            bin_ = table[conv].setdefault(trick, [0, 0])
            bin_[0] += 1
            bin_[1] += int(adh[conv])
    return table


def _fmt_table(table):
    lines = []
    for conv, bins in table.items():
        parts = []
        total = [0, 0]
        for trick in sorted(bins):
            n, k = bins[trick]
            total[0] += n
            total[1] += k
            parts.append(f"t{trick} {100.0 * k / n:.1f}% ({k}/{n})")
        if total[0]:
            parts.append(f"ALL {100.0 * total[1] / total[0]:.1f}%")
        lines.append(f"  {conv:20s} " + "  ".join(parts))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deals-per-mode", type=int, default=250)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument(
        "--iters",
        type=int,
        default=GATE_ITERS,
        help="committee search budget per replicate (default: certified 1024; "
        "lower ONLY for smoke tests — sub-1024 budgets are not certified)",
    )
    ap.add_argument("--seed", type=int, default=H2H_SEED)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--node-log", required=True)
    args = ap.parse_args()

    seed_rng = random.Random(args.seed)
    deal_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(args.deals_per_mode)]
    # Interleave modes so partial results stay mode-balanced.
    tasks = []
    for d, s in enumerate(deal_seeds):
        tasks.append((PARTNER_BY_CALLED_ACE, s, d))
        tasks.append((PARTNER_BY_JD, s, d))

    mode_scores = {"called": [], "jd": []}
    all_rows = []
    done = 0
    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.node_log)), exist_ok=True)
    node_log = open(args.node_log, "w")

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.ckpt, args.torch_threads, args.iters),
    ) as pool:
        for res in pool.imap_unordered(_run_deal, tasks, chunksize=1):
            done += 1
            mode_scores[res["mode"]].append(res["deal_score"])
            all_rows.extend(res["rows"])
            for row in res["rows"]:
                node_log.write(json.dumps(row) + "\n")
            node_log.flush()
            if done % 5 == 0 or done == len(tasks):
                parts = []
                for m, xs in mode_scores.items():
                    if xs:
                        a = np.array(xs)
                        parts.append(
                            f"{m} {a.mean():+.4f}±{a.std(ddof=1) / max(1, np.sqrt(len(a))):.4f} (n={len(a)})"
                        )
                searched = len(all_rows)
                resolved = sum(r["resolved"] for r in all_rows)
                deviated = sum(r["deviated"] for r in all_rows)
                elapsed = (time.time() - t_start) / 3600
                print(
                    f"[{done}/{len(tasks)} deal-modes, {elapsed:.1f}h] "
                    + " | ".join(parts)
                    + f" | nodes {searched} resolved {resolved} "
                    f"({100.0 * resolved / max(1, searched):.0f}%) "
                    f"deviated {deviated} "
                    f"({100.0 * deviated / max(1, searched):.0f}%)",
                    flush=True,
                )
    node_log.close()

    called = np.array(mode_scores["called"])
    jd = np.array(mode_scores["jd"])
    modes = {}
    for name, arr in (("called", called), ("jd", jd)):
        modes[name] = {
            "edge": float(arr.mean()),
            "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
            "n_deals": int(len(arr)),
        }
    edge = 0.5 * (modes["called"]["edge"] + modes["jd"]["edge"])
    se = 0.5 * float(np.sqrt(modes["called"]["se"] ** 2 + modes["jd"]["se"] ** 2))
    pooled = np.concatenate([called, jd])

    out = {
        "instrument": "duplicate_bridge_committee_ceiling",
        "ckpt": args.ckpt,
        "committee": {
            "replicates": R_REPLICATES,
            "votes_needed": VOTES_NEEDED,
            "iters": args.iters,
            "d_rollout": GATE_D_ROLLOUT,
            "readout": "pi_gumbel_argmax",
        },
        "edge": edge,
        "se": se,
        "win_frac": float(((pooled > 0) + 0.5 * (pooled == 0)).mean()),
        "deviating_frac": float((pooled != 0).mean()),
        "n_deals": int(len(pooled)),
        "modes": modes,
        "nodes": {
            "searched": len(all_rows),
            "resolved": int(sum(r["resolved"] for r in all_rows)),
            "deviated": int(sum(r["deviated"] for r in all_rows)),
        },
        "adherence_acted_by_trick": _adherence_table(all_rows, "adh_acted"),
        "adherence_policy_by_trick": _adherence_table(all_rows, "adh_policy"),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nEDGE {edge:+.4f} ± {se:.4f}  (n_deals {len(pooled)})")
    print("adherence (search-ACTED arm) by trick:")
    print(_fmt_table(out["adherence_acted_by_trick"]))
    print("adherence (policy-argmax counterfactual) by trick:")
    print(_fmt_table(out["adherence_policy_by_trick"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
