#!/usr/bin/env python3
"""Step 0: frozen-encoder critic probe.

Question: is the weak critic calibration in first-trick defender-lead states a
*head-capacity* problem (a deeper value trunk will fix it) or a *feature*
problem (the frozen encoder simply doesn't represent the distinction, so no
head can)?

Method: with the 30M encoder FROZEN, play full games and cache, at every play
decision, the 256-d encoder features plus that player's Monte-Carlo
return-to-go in the critic's reward units at gamma=0.99 (shaping-free, matching
the planned fine-tune target). Then train two fresh value heads on the frozen
features and a held-out (by game) split:

  * SHALLOW  = LayerNorm -> 256 -> act -> Linear(256,1)   (old critic_adapter shape)
  * DEEP     = LayerNorm -> 256 -> act -> 256 -> act -> Linear(256,1)  (new value_trunk)

We report val R^2 / dispersion overall and on the trick-0 defender-lead subset,
for SHALLOW vs DEEP (apples-to-apples: same frozen features, same target). The
loaded 30M critic's own prediction is shown as a reference (caveat: it was
trained on a different gamma/shaping target, so compare SHALLOW-vs-DEEP for the
clean signal).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn

import ppo
from ppo import PPOAgent
from sheepshead import ACTIONS, Game
from training_utils import RETURN_SCALE, TRICK_POINT_RATIO, get_partner_selection_mode

DEV = ppo.device


def encode_decide(agent, game, player):
    """One encode (advance memory like act), sample an action, and return
    (action_id, features, is_play_decision, is_lead, is_defender)."""
    state = player.get_state_dict()
    valid = player.get_valid_action_ids()
    mem_in = agent.get_recurrent_memory(player.position, device=DEV)
    enc = agent.encoder.encode_batch([state], memory_in=mem_in.unsqueeze(0), device=DEV)
    agent.set_recurrent_memory(player.position, enc["memory_out"][0])
    mask = agent.get_action_mask(valid, agent.action_size).unsqueeze(0).to(DEV)
    hand_ids = torch.as_tensor(state["hand_ids"], dtype=torch.long, device=DEV).view(
        1, -1
    )
    with torch.no_grad():
        probs = agent.actor(enc, mask, hand_ids, agent.encoder.card)
        a = torch.distributions.Categorical(probs).sample().item() + 1

    is_play = any(ACTIONS[v - 1].startswith("PLAY ") for v in valid)
    is_lead = is_play and game.cards_played == 0 and game.leader == player.position
    is_defender = not (
        player.is_picker or player.is_partner or player.is_secret_partner
    )
    feats = enc["features"][0].detach().cpu().numpy() if is_play else None
    return a, feats, is_play, is_lead, is_defender


def player_play_returns(game, pos, gamma):
    """Discounted return-to-go for `pos` at each of its 6 play decisions (trick 0..5)."""
    picker, partner = game.picker, game.partner
    player_pick_team = pos in (picker, partner)
    r = [0.0] * 6
    for t in range(6):
        winner = game.trick_winners[t]
        winner_pick_team = winner in (picker, partner)
        sign = 1.0 if (winner_pick_team == player_pick_team) else -1.0
        r[t] = (game.trick_points[t] / TRICK_POINT_RATIO) * sign
    r[5] += game.players[pos - 1].get_score() / RETURN_SCALE
    g, acc = [0.0] * 6, 0.0
    for t in range(5, -1, -1):
        acc = r[t] + gamma * acc
        g[t] = acc
    return g


def collect(agent, n_games, gamma, seed):
    feats, targets, gid, trick, is_lead, is_def = [], [], [], [], [], []
    for g in range(n_games):
        game = Game(partner_selection_mode=get_partner_selection_mode(g))
        agent.reset_recurrent_state()
        # per-game capture: (pos, trick, features, is_lead, is_defender)
        caps = []
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
        if game.is_leaster:
            continue
        returns_by_pos = {p: player_play_returns(game, p, gamma) for p in range(1, 6)}
        for pos, tk, fz, lead, dfn in caps:
            feats.append(fz)
            targets.append(returns_by_pos[pos][tk])
            gid.append(g)
            trick.append(tk)
            is_lead.append(lead)
            is_def.append(dfn)
    return {
        "X": np.array(feats, dtype=np.float32),
        "y": np.array(targets, dtype=np.float32),
        "gid": np.array(gid),
        "trick": np.array(trick),
        "is_lead": np.array(is_lead),
        "is_def": np.array(is_def),
    }


class Head(nn.Module):
    def __init__(self, depth, act):
        super().__init__()
        layers = [nn.LayerNorm(256), nn.Linear(256, 256), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(256, 256), act()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(-1)


def r2(pred, y):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_head(depth, Xtr, ytr, Xval, epochs, lr, act):
    model = Head(depth, act).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xtr_t = torch.as_tensor(Xtr, device=DEV)
    ytr_t = torch.as_tensor(ytr, device=DEV)
    n = len(Xtr_t)
    bs = 4096
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.as_tensor(Xval, device=DEV)).cpu().numpy()
    return pred


def report_subset(name, mask, yval, preds):
    y = yval[mask]
    if len(y) < 5:
        print(f"\n{name}: too few states ({len(y)})")
        return
    print(f"\n{name} (n={len(y)})  target std={y.std():.4f}")
    for label, p in preds.items():
        pm = p[mask]
        print(
            f"  {label:<18} R^2={r2(pm, y):+.3f}  pred std={pm.std():.4f}  "
            f"bias={pm.mean() - y.mean():+.4f}"
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.model} (device={DEV}) ...")
    agent = PPOAgent(len(ACTIONS))
    agent.load(args.model, load_optimizers=False)
    act = nn.SiLU

    print(f"Collecting features from {args.games} games (gamma={args.gamma}) ...")
    data = collect(agent, args.games, args.gamma, args.seed)
    X, y = data["X"], data["y"]
    print(
        f"  {len(y)} play-decision samples; "
        f"{int(((data['trick'] == 0) & data['is_lead'] & data['is_def']).sum())} trick-0 defender leads."
    )

    # Split by game to avoid leakage.
    val_mask = data["gid"] % 5 == 0
    tr = ~val_mask
    Xtr, ytr, Xval, yval = X[tr], y[tr], X[val_mask], y[val_mask]
    print(f"  train {len(ytr)} / val {len(yval)}")

    # Reference: the loaded 30M critic's own value on val features (different
    # training target -> rough reference only).
    with torch.no_grad():
        fv = torch.as_tensor(Xval, device=DEV)
        orig = (
            agent.critic.value_head(agent.critic.critic_adapter(fv))
            .squeeze(-1)
            .cpu()
            .numpy()
        )

    print("Training fresh heads on frozen features ...")
    shallow = train_head(1, Xtr, ytr, Xval, args.epochs, args.lr, act)
    deep = train_head(2, Xtr, ytr, Xval, args.epochs, args.lr, act)

    preds = {"30M critic (ref)": orig, "fresh SHALLOW": shallow, "fresh DEEP": deep}

    print("\n" + "=" * 72)
    print("FROZEN-ENCODER PROBE RESULTS (held-out by game)")
    print("=" * 72)
    print(
        f"  Target: gamma={args.gamma} shaping-free return; val target std={yval.std():.4f}"
    )
    report_subset("ALL play decisions", np.ones(len(yval), bool), yval, preds)
    report_subset(
        "Lead decisions (cards_played==0)", data["is_lead"][val_mask], yval, preds
    )
    report_subset(
        "Trick-0 DEFENDER leads (the leak states)",
        (data["trick"][val_mask] == 0)
        & data["is_lead"][val_mask]
        & data["is_def"][val_mask],
        yval,
        preds,
    )

    # By-trick R^2 curve (deep head) -- tests the partial-observability hypothesis:
    # if predictability rises as cards are revealed, low trick-0 R^2 is a hidden-
    # information ceiling, not a fixable encoder/head deficiency.
    print("\n--- Fresh DEEP head: R^2 by trick (lead decisions only) ---")
    tv = data["trick"][val_mask]
    lv = data["is_lead"][val_mask]
    for t in range(6):
        m = (tv == t) & lv
        if m.sum() >= 20:
            print(
                f"  trick {t} leads (n={int(m.sum()):5d}): R^2={r2(deep[m], yval[m]):+.3f}  "
                f"target std={yval[m].std():.3f}  pred std={deep[m].std():.3f}"
            )
    print("--- Fresh DEEP head: R^2 by trick (all play decisions) ---")
    for t in range(6):
        m = tv == t
        if m.sum() >= 20:
            print(
                f"  trick {t} (n={int(m.sum()):5d}): R^2={r2(deep[m], yval[m]):+.3f}  "
                f"target std={yval[m].std():.3f}"
            )

    print("\n" + "=" * 72)
    print("READING IT")
    print("=" * 72)
    print("  Compare fresh SHALLOW vs fresh DEEP on the trick-0 subset:")
    print(
        "   - DEEP >> SHALLOW  -> head capacity was limiting; the value_trunk change helps."
    )
    print(
        "   - DEEP ~ SHALLOW, both decent -> features fine, head just needed retraining."
    )
    print("   - both low (and < ALL-decisions R^2) -> encoder lacks the distinction;")
    print("     a deeper head won't help -> encoder changes needed.")


if __name__ == "__main__":
    main()
