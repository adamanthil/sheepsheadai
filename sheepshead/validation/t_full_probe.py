#!/usr/bin/env python3
"""t_full critic-calibration probe (one-off, NOT committed).

Sets the ISMCTS rollout-depth-schedule cutoff `t_full` on evidence. The schedule
is d_rollout = (6 - trick) if trick <= t_full else d_short: roll (near) to
terminal in the early tricks where the critic is blind to the partial-obs leak,
then BOOTSTRAP the critic once it is trustworthy. So the question is: at trick t,
how well can the value head predict the ExIt terminal return?

Method (mirrors critic_probe.py but with the TERMINAL target the ExIt rollout
bootstraps toward, not a discounted/shaped return): freeze the encoder, play
games, cache every play decision's 256-d features + that seat's realized terminal
return, train a fresh DEEP value head (production value_trunk shape) on a
held-out-by-game split, and report R^2 per trick. t_full = the last trick whose
R^2 is still below the trust threshold (roll deep through the blind tricks,
bootstrap after).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from sheepshead.agent import ppo
from sheepshead.validation.critic_probe import encode_decide, r2, train_head
from sheepshead.agent.ppo import load_agent
from sheepshead import Game
from sheepshead.training.training_utils import (
    RETURN_SCALE,
    get_partner_selection_mode,
    set_all_seeds,
)

DEV = ppo.device


def collect_terminal(agent, n_games, seed):
    """Per play decision: (features, terminal return, trick, lead,
    defender, leaster, game id)."""
    set_all_seeds(seed)
    feats, targets, gid, trick, is_lead, is_def, leaster = [], [], [], [], [], [], []
    for g in range(n_games):
        game = Game(partner_selection_mode=get_partner_selection_mode(g))
        agent.reset_recurrent_state()
        caps = []  # (pos, trick, features, lead, defender)
        while not game.is_done():
            for player in game.players:
                while player.get_valid_action_ids():
                    a, fz, is_play, lead, dfn = encode_decide(agent, game, player)
                    if is_play:
                        caps.append(
                            (player.position, game.current_trick, fz, lead, dfn)
                        )
                    player.act(a)
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
        term = {p: game.players[p - 1].get_score() / RETURN_SCALE for p in range(1, 6)}
        is_l = game.is_leaster
        for pos, tk, fz, lead, dfn in caps:
            feats.append(fz)
            targets.append(term[pos])
            gid.append(g)
            trick.append(tk)
            is_lead.append(lead)
            is_def.append(dfn)
            leaster.append(is_l)
    return {
        "X": np.array(feats, dtype=np.float32),
        "y": np.array(targets, dtype=np.float32),
        "gid": np.array(gid),
        "trick": np.array(trick),
        "is_lead": np.array(is_lead),
        "is_def": np.array(is_def),
        "leaster": np.array(leaster),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="final_pfsp_swish_ppo.pt")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="R^2 trust threshold; t_full = last trick below it",
    )
    args = ap.parse_args()

    print(f"Loading {args.model} (device={DEV}) ...")
    agent = load_agent(args.model)

    print(
        f"Collecting frozen-encoder features over {args.games} games "
        f"(target = terminal get_score()/12) ..."
    )
    data = collect_terminal(agent, args.games, args.seed)
    n = len(data["y"])
    print(f"  {n} play decisions; {int(data['leaster'].sum())} from leasters.")

    # Held-out-by-game split.
    games = np.unique(data["gid"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    val_games = set(games[: max(1, len(games) // 5)].tolist())
    val_mask = np.array([g in val_games for g in data["gid"]])
    tr = ~val_mask

    act = torch.nn.SiLU  # swish
    # Production value head is the DEEP trunk (depth=2): LN->256->act->256->act->Lin.
    preds = train_head(
        2, data["X"][tr], data["y"][tr], data["X"][val_mask], args.epochs, args.lr, act
    )
    yval = data["y"][val_mask]
    tv = data["trick"][val_mask]
    lv = data["is_lead"][val_mask]
    dv = data["is_def"][val_mask]
    leasv = data["leaster"][val_mask]
    overall = r2(preds, yval)
    print(
        f"\nFresh DEEP head, terminal target. Overall val R^2 = {overall:+.3f} "
        f"(target std {yval.std():.3f})"
    )

    def by_trick(mask, label):
        print(f"\n--- R^2 by trick: {label} ---")
        curve = {}
        for t in range(6):
            m = mask & (tv == t)
            if m.sum() >= 30:
                val = r2(preds[m], yval[m])
                curve[t] = val
                print(
                    f"  trick {t} (n={int(m.sum()):6d}): R^2={val:+.3f}  "
                    f"pred std={preds[m].std():.3f}  bias={preds[m].mean() - yval[m].mean():+.3f}"
                )
        return curve

    normal = ~leasv
    curve_all = by_trick(normal, "non-leaster, all play decisions")
    by_trick(normal & lv, "non-leaster, lead decisions")
    by_trick(normal & lv & dv, "non-leaster, defender leads (the trick-0 leak subset)")
    if leasv.sum() >= 150:
        by_trick(leasv, "leaster play decisions")

    # Recommend t_full from the all-play-decisions curve.
    thr = args.threshold
    below = [t for t, v in curve_all.items() if v < thr]
    rec = max(below) if below else -1
    print(f"\n=== t_full recommendation (threshold R^2 >= {thr}) ===")
    print(
        "  per-trick R^2: "
        + ", ".join(f"t{t}={curve_all[t]:+.2f}" for t in sorted(curve_all))
    )
    if rec == -1:
        print("  critic already trustworthy at trick 0 -> t_full = 0 (bootstrap early)")
    elif rec >= 5:
        print(
            f"  critic never crosses {thr} -> roll to terminal (t_full = 5 / high d_short)"
        )
    else:
        print(
            f"  t_full = {rec}  (roll deep tricks 0..{rec}; bootstrap from trick {rec + 1})"
        )


if __name__ == "__main__":
    main()
