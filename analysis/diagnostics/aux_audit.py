"""Aux-head prediction-quality audit across checkpoints.

Plays N fixed-seed self-play games per checkpoint (CRN: identical deal
seeds for every checkpoint, alternating called-ace/JD), collects
per-action-state aux predictions (same extraction path as
server/services/analyze.py) plus the same ground-truth labels the
trainer stores, and reports per-head accuracy metrics.

Heads audited:
  win           P(final score > 0)             -> AUC, acc@0.5, base rate
  return        expected final score           -> corr, MAE (score units)
  partner       P(this player is secret ptnr)  -> AUC, acc@0.5, base rate
  points        per-seat known points (0-120)  -> MAE (points)
  seen_trump    per-card seen mask             -> per-bit acc, F1
  unseen_higher any unseen trump beats hand    -> AUC, acc@0.5, base rate
"""

import csv
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/Volumes/Nargothrond/dev/sheepsheadai")

from sheepshead.agent.ppo import load_agent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game
from sheepshead.training.training_utils import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
)

DEVICE = torch.device("cpu")
N_GAMES = 200
BASE_SEED = 20260708


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (ties handled by average rank)."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    all_scores = np.concatenate([pos, neg])
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def audit_checkpoint(ckpt_path: str) -> dict:
    agent = load_agent(ckpt_path)
    if not getattr(agent.critic, "has_aux_heads", False):
        raise SystemExit(f"{ckpt_path}: no aux heads")

    win_p, win_y = [], []
    ret_p, ret_y = [], []
    par_p, par_y = [], []
    pts_err = []
    stm_p, stm_y = [], []
    uns_p, uns_y = [], []

    for g in range(N_GAMES):
        mode = PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD
        torch.manual_seed(BASE_SEED + g)
        random.seed(BASE_SEED + g)
        game = Game(partner_selection_mode=mode, seed=BASE_SEED + g)
        agent.reset_recurrent_state()

        per_state = []  # (player_pos, win_prob, ret_pred)
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()
                while valid_actions:
                    state = player.get_state_dict()
                    memory_in = agent.get_recurrent_memory(
                        player.position, device=DEVICE
                    )
                    with torch.no_grad():
                        encoder_out = agent.encoder.encode_batch(
                            [state], memory_in=memory_in.unsqueeze(0), device=DEVICE
                        )
                        agent.set_recurrent_memory(
                            player.position, encoder_out["memory_out"][0]
                        )
                        mask_t = (
                            agent.get_action_mask(valid_actions, agent.action_size)
                            .unsqueeze(0)
                            .to(DEVICE)
                        )
                        hand_ids_t = torch.as_tensor(
                            state["hand_ids"], dtype=torch.long, device=DEVICE
                        ).view(1, -1)
                        action_probs, _ = agent.actor.forward_with_logits(
                            encoder_out, mask_t, hand_ids_t, agent.encoder.card
                        )
                        win_prob, exp_ret, secret_prob, points_vec = (
                            agent.critic.aux_predictions(encoder_out)
                        )
                        aux_feat = agent.critic._aux_features_single(encoder_out)
                        stm_logits = agent.critic.seen_trump_mask_logits(
                            aux_feat, agent.encoder.card
                        ).squeeze(0)
                        uns_logit = agent.critic.unseen_trump_higher_than_hand_logits(
                            aux_feat
                        ).squeeze(0)

                    # labels available now (same as trainer's per-action labels)
                    par_p.append(float(secret_prob))
                    par_y.append(1.0 if player.is_secret_partner else 0.0)
                    pts_label = np.asarray(
                        compute_known_points_rel(player), dtype=float
                    )
                    pts_pred = np.asarray(points_vec, dtype=float).reshape(-1)
                    pts_err.append(np.abs(pts_pred - pts_label).mean())
                    stm_label = np.asarray(compute_seen_trump_mask(player), dtype=float)
                    stm_prob = torch.sigmoid(stm_logits).cpu().numpy().reshape(-1)
                    stm_p.append(stm_prob)
                    stm_y.append(stm_label)
                    uns_p.append(float(torch.sigmoid(uns_logit)))
                    uns_y.append(
                        1.0
                        if compute_any_unseen_trump_higher_than_hand(player)
                        else 0.0
                    )
                    per_state.append((player.position, float(win_prob), float(exp_ret)))

                    action = (
                        int(torch.multinomial(action_probs.squeeze(0), 1).item()) + 1
                    )
                    if action not in valid_actions:  # numerical guard
                        action = random.choice(list(valid_actions))
                    player.act(action)

                    # Mirror the trainer's post-trick observation frames so
                    # the recurrent memory stream matches training exactly.
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                    valid_actions = player.get_valid_action_ids()

        scores = [p.get_score() for p in game.players]
        for pos, wp, rp in per_state:
            win_p.append(wp)
            win_y.append(1.0 if scores[pos - 1] > 0 else 0.0)
            ret_p.append(rp)
            ret_y.append(float(scores[pos - 1]))

    win_p, win_y = np.array(win_p), np.array(win_y)
    ret_p, ret_y = np.array(ret_p), np.array(ret_y)
    par_p, par_y = np.array(par_p), np.array(par_y)
    stm_p, stm_y = np.concatenate(stm_p), np.concatenate(stm_y)
    uns_p, uns_y = np.array(uns_p), np.array(uns_y)

    return {
        "n_states": len(win_p),
        "win_auc": auc(win_y, win_p),
        "win_acc": float(((win_p > 0.5) == (win_y > 0.5)).mean()),
        "win_base": float(win_y.mean()),
        "ret_corr": float(np.corrcoef(ret_p, ret_y)[0, 1]),
        "ret_mae": float(np.abs(ret_p - ret_y).mean()),
        "partner_auc": auc(par_y, par_p),
        "partner_acc": float(((par_p > 0.5) == (par_y > 0.5)).mean()),
        "partner_base": float(par_y.mean()),
        "points_mae": float(np.mean(pts_err)),
        "seen_trump_acc": float(((stm_p > 0.5) == (stm_y > 0.5)).mean()),
        "seen_trump_f1": float(
            2
            * ((stm_p > 0.5) & (stm_y > 0.5)).sum()
            / max(1, ((stm_p > 0.5).sum() + (stm_y > 0.5).sum()))
        ),
        "unseen_auc": auc(uns_y, uns_p),
        "unseen_acc": float(((uns_p > 0.5) == (uns_y > 0.5)).mean()),
        "unseen_base": float(uns_y.mean()),
    }


def main():
    ckpts = []
    for seed in (42, 1042, 2042):
        for ep in (100000, 200000):
            ckpts.append(
                (
                    f"full_s{seed}",
                    ep,
                    f"runs/ablate_full_s{seed}/full_checkpoint_{ep}.pt",
                )
            )
            ckpts.append(
                (
                    f"shared_s{seed}",
                    ep,
                    f"runs/ablate_perceiver-shared_s{seed}/perceiver-shared_checkpoint_{ep}.pt",
                )
            )
        ckpts.append(
            (
                f"full_s{seed}",
                300000,
                f"runs/ablate_full400_s{seed}/full_checkpoint_300000.pt",
            )
        )

    out_path = sys.argv[1] if len(sys.argv) > 1 else "aux_audit.csv"
    fields = None
    with open(out_path, "w", newline="") as f:
        writer = None
        for name, ep, path in ckpts:
            row = {"model": name, "episodes": ep}
            row.update(audit_checkpoint(path))
            if writer is None:
                fields = list(row)
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(
                f"{name}@{ep}: partner_auc={row['partner_auc']:.3f} "
                f"win_auc={row['win_auc']:.3f} ret_corr={row['ret_corr']:.3f} "
                f"points_mae={row['points_mae']:.1f} "
                f"seen_trump_acc={row['seen_trump_acc']:.3f} "
                f"unseen_acc={row['unseen_acc']:.3f} (n={row['n_states']})",
                flush=True,
            )


if __name__ == "__main__":
    main()
