#!/usr/bin/env python3
"""Play-head logit-diversity probe.

Drives greedy self-play with a single loaded checkpoint controlling all five
seats and, at every PLAY decision with >=2 legal cards, records the spread of
the policy's PRE-MIX logits over the legal cards:

  - logit_spread = max(logit) - min(logit)   over legal plays (raw separation)
  - margin       = p1 - p2                    over legal plays (post-softmax)
  - entropy      = H(policy over legal plays) in nats
  - top1p        = max prob over legal plays

If the user's claim is right ("identical logits for all valid plays"), spread
collapses to ~0, margin ~0, entropy ~= log(n_valid) (maximal). This separates
"play head is dead/uniform" from "play head is sharp but argmax happens to be
trump."  Reports the distribution + a per-trick breakdown.

Usage:
  PYTHONPATH=. .venv/bin/python validation/play_head_probe.py \
      -m runs/repro_league/checkpoints/pfsp_swish_checkpoint_4500000.pt \
      [--deals 300]
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from ppo import PPOAgent
from sheepshead import ACTIONS, Game
from training_utils import get_partner_selection_mode


def _load(model: str) -> PPOAgent:
    a = PPOAgent(len(ACTIONS))
    a.load(model, load_optimizers=False)
    return a


def _play_stats(agent, state, valid, pid):
    vlist = list(valid)
    probs_t, logits_t = agent.get_action_probs_with_logits(state, valid, player_id=pid)
    probs = probs_t[0].detach().cpu().numpy()
    logits = logits_t[0].detach().cpu().numpy()
    lv = np.array([logits[a - 1] for a in vlist], dtype=np.float64)
    pv = np.array([probs[a - 1] for a in vlist], dtype=np.float64)
    order = np.argsort(pv)[::-1]
    pol_arg = vlist[order[0]]
    p1 = float(pv[order[0]])
    p2 = float(pv[order[1]]) if len(pv) > 1 else 0.0
    s = pv.sum()
    pn = pv / s if s > 0 else pv
    nz = pn[pn > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    spread = float(lv.max() - lv.min())
    return pol_arg, spread, p1 - p2, entropy, p1


def run(model, deals, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = _load(model)

    rows = []  # (trick, n_valid, spread, margin, entropy, top1p)
    for d in range(deals):
        mode = get_partner_selection_mode(d)
        game = Game(partner_selection_mode=mode, seed=seed * 1_000_003 + d)
        agent.reset_recurrent_state()
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    is_play = game.play_started and all(
                        ACTIONS[a - 1].startswith("PLAY ") for a in valid
                    )
                    if is_play and len(valid) >= 2:
                        pol_arg, spread, margin, ent, top1p = _play_stats(
                            agent, player.get_state_dict(), valid, player.position
                        )
                        rows.append(
                            (game.current_trick, len(valid), spread, margin, ent, top1p)
                        )
                        a = pol_arg
                    else:
                        a, _, _ = agent.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=True,
                        )
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for p in game.players:
                            agent.observe(
                                p.get_last_trick_state_dict(), player_id=p.position
                            )
    return rows


def report(rows):
    if not rows:
        print("no PLAY decisions captured")
        return
    arr = np.array(rows, dtype=np.float64)
    trick, nval, spread, margin, ent, top1p = (arr[:, i] for i in range(6))
    n = len(rows)

    def pct(x, ps=(0, 10, 25, 50, 75, 90, 100)):
        return "  ".join(f"p{p}={np.percentile(x, p):.3f}" for p in ps)

    print("=" * 74)
    print(f"PLAY-HEAD LOGIT DIVERSITY  ({n} multi-legal PLAY decisions)")
    print("=" * 74)
    print(f"  logit spread (max-min over legal plays):  {pct(spread)}")
    print(f"  margin p1-p2 (post-softmax):              {pct(margin)}")
    print(f"  entropy (nats):                           {pct(ent)}")
    print(f"  top1 prob:                                {pct(top1p)}")
    # near-uniform fraction: spread < 0.05 logits ~ indistinguishable cards
    for thr in (0.01, 0.05, 0.1, 0.5):
        frac = 100 * np.mean(spread < thr)
        print(f"  fraction with logit spread < {thr:<4}: {frac:5.1f}%")
    print()
    print("  by trick:  trick  nodes  med_spread  med_margin  med_entropy  med_top1p")
    for t in sorted(set(trick.astype(int))):
        m = trick.astype(int) == t
        print(
            f"           {t:5d}  {int(m.sum()):5d}  "
            f"{np.median(spread[m]):10.3f}  {np.median(margin[m]):10.3f}  "
            f"{np.median(ent[m]):11.3f}  {np.median(top1p[m]):9.3f}"
        )
    print()
    print("  by n_valid: nval  nodes  med_spread  med_entropy  log(nval)=max_ent")
    for nv in sorted(set(nval.astype(int))):
        m = nval.astype(int) == nv
        print(
            f"            {nv:4d}  {int(m.sum()):5d}  {np.median(spread[m]):10.3f}  "
            f"{np.median(ent[m]):11.3f}  {np.log(nv):17.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"Loading {args.model} ...")
    rows = run(args.model, args.deals, args.seed)
    report(rows)


if __name__ == "__main__":
    main()
