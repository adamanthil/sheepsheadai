"""Aux-head prediction-quality audit across checkpoints.

Plays N fixed-seed self-play games per checkpoint (CRN: identical deal
seeds for every checkpoint, alternating called-ace/JD), collects
per-action-state aux predictions (same extraction path as
server/services/analyze.py) plus the same ground-truth labels the
trainer stores, and reports per-head accuracy metrics.

Heads audited (and the scale that makes each actionable):

  win           P(final score > 0)            -> AUC, Brier, base rate
  return        expected final score          -> corr, MAE (score units)
  partner       P(this player is secret ptnr) -> AUC. NOTE: this head
                predicts the player's OWN partner status, which is
                deterministic from their own hand (called card / JD in
                hand) — expect AUC ~1 early; kept for continuity only.
  points        per-seat known points (0-120) -> MAE, P(err<=5),
                P(err<=10), early/late-trick MAE. Point tracking needs
                single-digit precision to change decisions.
  seen_trump    per-card seen mask            -> per-bit Brier +
                positive-class F1, early/late splits. Raw per-bit
                accuracy saturates (most bits are easy 0s early).
  unseen_higher any unseen trump beats hand   -> AUC, Brier, plus
                trick-0/1 splits — the window where the answer is
                uncertain and decisions (leads, bury) actually hinge
                on it.

Trick buckets: t01 = prediction made during tricks 0-1 (hard, actionable),
t45 = tricks 4-5 (mostly bookkeeping). A head can be "accurate" overall
while useless at t01; the stratified numbers are the ones that bear on
whether aux quality changed in a way play can exploit.

Usage:
  PYTHONPATH=. python -m sheepshead.analysis.diagnostics.aux_audit \
    --ckpt selfplay400k=runs/.../warmstart_perceiver-shared-v2_400k.pt \
    --ckpt league2M=runs/.../pfsp_perceiver-shared-v2_checkpoint_2000000.pt \
    --out aux_ladder.csv
"""

import argparse
import csv
import random
import re

import numpy as np
import torch


from sheepshead.agent.ppo import load_agent
from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game
from sheepshead.training.training_utils import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
)

DEVICE = torch.device("cpu")
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


def brier(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2)) if len(labels) else float("nan")


def pos_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    pred = probs > 0.5
    truth = labels > 0.5
    denom = pred.sum() + truth.sum()
    return float(2 * (pred & truth).sum() / denom) if denom else float("nan")


def _bucket(tricks: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return (tricks >= lo) & (tricks <= hi)


def audit_checkpoint(ckpt_path: str, n_games: int) -> dict:
    agent = load_agent(ckpt_path)
    if not getattr(agent.critic, "has_aux_heads", False):
        raise SystemExit(f"{ckpt_path}: no aux heads")

    win_p, win_y = [], []
    ret_p, ret_y = [], []
    par_p, par_y = [], []
    pts_err, pts_trick = [], []  # pooled per-seat absolute errors
    stm_p, stm_y, stm_trick = [], [], []  # pooled per-bit
    uns_p, uns_y, uns_trick = [], [], []

    for g in range(n_games):
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
                    trick = int(game.current_trick)
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
                    err = np.abs(pts_pred - pts_label)
                    pts_err.extend(err.tolist())
                    pts_trick.extend([trick] * len(err))
                    stm_label = np.asarray(compute_seen_trump_mask(player), dtype=float)
                    stm_prob = torch.sigmoid(stm_logits).cpu().numpy().reshape(-1)
                    stm_p.extend(stm_prob.tolist())
                    stm_y.extend(stm_label.tolist())
                    stm_trick.extend([trick] * len(stm_label))
                    uns_p.append(float(torch.sigmoid(uns_logit)))
                    uns_y.append(
                        1.0
                        if compute_any_unseen_trump_higher_than_hand(player)
                        else 0.0
                    )
                    uns_trick.append(trick)
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
    pts_err, pts_trick = np.array(pts_err), np.array(pts_trick)
    stm_p, stm_y, stm_trick = np.array(stm_p), np.array(stm_y), np.array(stm_trick)
    uns_p, uns_y, uns_trick = np.array(uns_p), np.array(uns_y), np.array(uns_trick)

    p_t01, p_t45 = _bucket(pts_trick, 0, 1), _bucket(pts_trick, 4, 5)
    s_t01, s_t45 = _bucket(stm_trick, 0, 1), _bucket(stm_trick, 4, 5)
    u_t01 = _bucket(uns_trick, 0, 1)

    return {
        "n_states": len(win_p),
        "win_auc": auc(win_y, win_p),
        "win_brier": brier(win_y, win_p),
        "win_base": float(win_y.mean()),
        "ret_corr": float(np.corrcoef(ret_p, ret_y)[0, 1]),
        "ret_mae": float(np.abs(ret_p - ret_y).mean()),
        "partner_auc": auc(par_y, par_p),
        "points_mae": float(pts_err.mean()),
        "points_p_le5": float((pts_err <= 5).mean()),
        "points_p_le10": float((pts_err <= 10).mean()),
        "points_mae_t01": float(pts_err[p_t01].mean()),
        "points_mae_t45": float(pts_err[p_t45].mean()),
        "seen_brier": brier(stm_y, stm_p),
        "seen_pos_f1": pos_f1(stm_y, stm_p),
        "seen_brier_t01": brier(stm_y[s_t01], stm_p[s_t01]),
        "seen_pos_f1_t01": pos_f1(stm_y[s_t01], stm_p[s_t01]),
        "seen_pos_f1_t45": pos_f1(stm_y[s_t45], stm_p[s_t45]),
        "unseen_auc": auc(uns_y, uns_p),
        "unseen_brier": brier(uns_y, uns_p),
        "unseen_base": float(uns_y.mean()),
        "unseen_auc_t01": auc(uns_y[u_t01], uns_p[u_t01]),
        "unseen_brier_t01": brier(uns_y[u_t01], uns_p[u_t01]),
    }


def _episodes_from_path(path: str) -> int:
    m = re.search(r"checkpoint_(\d+)", path)
    return int(m.group(1)) if m else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aux-head prediction-quality audit")
    ap.add_argument(
        "--ckpt",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="checkpoint to audit (repeatable); rows keep the given order",
    )
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="aux_audit.csv")
    args = ap.parse_args(argv)

    ckpts = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--ckpt needs LABEL=PATH, got {spec!r}")
        ckpts.append((label, path))

    with open(args.out, "w", newline="") as f:
        writer = None
        for label, path in ckpts:
            row = {"model": label, "episodes": _episodes_from_path(path)}
            row.update(audit_checkpoint(path, args.games))
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(
                f"{label}: points_mae_t01={row['points_mae_t01']:.2f} "
                f"points_p_le5={row['points_p_le5']:.3f} "
                f"seen_pos_f1_t01={row['seen_pos_f1_t01']:.3f} "
                f"unseen_auc_t01={row['unseen_auc_t01']:.3f} "
                f"unseen_brier_t01={row['unseen_brier_t01']:.3f} "
                f"ret_corr={row['ret_corr']:.3f} (n={row['n_states']})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
