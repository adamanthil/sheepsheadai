#!/usr/bin/env python3
"""Critic-calibration diagnostic for first-trick defender-lead states.

Goal: decide whether the value function (the critic) is the bottleneck that
prevents PPO from cleaning up the first-trick defender trump-lead leak, *before*
committing to a retrain.

The critic predicts the discounted return in the exact units it was trained on:
    G = sum_t gamma^t * r_t  +  gamma^5 * (final_score / RETURN_SCALE)
    r_t = (trick_points_t / TRICK_POINT_RATIO) * (+1 if a defender won trick t
           else -1)              # seat-1 lead states are always defender states
This script reconstructs that return from full Monte-Carlo rollouts and compares
it to the critic's V(s).

Two analyses
------------
(A) CALIBRATION over all first-trick defender-lead states: is V(s) an accurate,
    low-variance predictor of the realized on-policy return? Reports bias, RMSE,
    R^2, rank correlation, and how much advantage variance the baseline removes.
    * High R^2  -> critic is a good baseline; the leak is NOT mainly a critic
      problem (it's small-gap-vs-noise + exploration) -> search/ExIt is the lever.
    * Low  R^2  -> critic is noisy here -> advantages are noisy -> improving the
      critic should directly help, so a retrain with critic changes is justified.

(B) LEAK SIGNAL for trump-pref states: paired forced rollouts (best trump vs best
    fail) in reward units. Reports the action-value gap, the per-sample noise, the
    implied PPO advantages relative to V(s), and how many on-policy samples would
    be needed to resolve the gap -- i.e. whether the signal is learnable at all
    from sampling, regardless of the critic.
"""

from __future__ import annotations

import os
import sys

# Repo-root imports work regardless of invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import copy
import random

import numpy as np
import torch

import ppo
from ppo import PPOAgent
from sheepshead import ACTION_IDS, ACTIONS, TRUMP, Game
from training_utils import RETURN_SCALE, TRICK_POINT_RATIO, get_partner_selection_mode

DEV = ppo.device
GAMMA = 0.95  # must match PPOAgent.gamma


def snapshot_memory(agent):
    return {pid: t.detach().clone() for pid, t in agent._player_memories.items()}


def restore_memory(agent, snap):
    agent._player_memories = {pid: t.detach().clone() for pid, t in snap.items()}


def policy_and_value(agent, state, valid, pid):
    """Single encode (memory M0->M1, like act) returning play-action probs and V(s)."""
    mem_in = agent.get_recurrent_memory(pid, device=DEV)
    enc = agent.encoder.encode_batch([state], memory_in=mem_in.unsqueeze(0), device=DEV)
    agent.set_recurrent_memory(pid, enc["memory_out"][0])
    mask = agent.get_action_mask(valid, agent.action_size).unsqueeze(0).to(DEV)
    hand_ids = torch.as_tensor(state["hand_ids"], dtype=torch.long, device=DEV).view(
        1, -1
    )
    with torch.no_grad():
        probs, _ = agent.actor.forward_with_logits(
            enc, mask, hand_ids, agent.encoder.card
        )
        value = agent.critic(enc)
    return probs[0].detach().cpu().numpy(), float(value.item())


def advance_to_trick0(game, agent):
    guard = 0
    while not game.play_started and guard < 200:
        guard += 1
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                state = player.get_state_dict()
                a, _, _ = agent.act(state, valid, player.position, deterministic=False)
                player.act(a)
                if game.play_started:
                    break
                valid = player.get_valid_action_ids()
            if game.play_started:
                break
    return game.play_started and not game.is_leaster


def play_out(game, agent):
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                state = player.get_state_dict()
                a, _, _ = agent.act(state, valid, player.position, deterministic=False)
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )


def seat_return(game, seat_pos):
    """Discounted return for `seat_pos` (a defender) in the critic's reward units."""
    picker, partner = game.picker, game.partner
    g = 0.0
    for t in range(6):
        pts = game.trick_points[t]
        winner = game.trick_winners[t]
        winner_picker_team = winner == picker or winner == partner
        sign = -1.0 if winner_picker_team else 1.0  # seat is a defender
        r = (pts / TRICK_POINT_RATIO) * sign
        if t == 5:
            r += game.players[seat_pos - 1].get_score() / RETURN_SCALE
        g += (GAMMA**t) * r
    return g


def play_card_distribution(probs, valid):
    cards = {}
    for a in valid:
        name = ACTIONS[a - 1]
        if name.startswith("PLAY "):
            cards[name[5:]] = float(probs[a - 1])
    return cards


def best_in_class(card_probs, want_trump):
    best, bp = None, -1.0
    for c, p in card_probs.items():
        if (c in TRUMP) == want_trump and p > bp:
            best, bp = c, p
    return best


def rollout_returns(cand, agent, first_card, R, seat=1):
    """Force seat's first-trick lead to `first_card`, then play out; return list of returns."""
    aid = ACTION_IDS[f"PLAY {first_card}"]
    out = []
    for _ in range(R):
        g = copy.deepcopy(cand["game"])
        restore_memory(agent, cand["mem"])
        g.players[seat - 1].act(aid)
        play_out(g, agent)
        out.append(seat_return(g, seat))
    return np.array(out)


def rollout_onpolicy(cand, agent, R, rng, seat=1):
    """Sample seat's first lead from the decision probs (faithful on-policy), play out."""
    cards = list(cand["card_probs"])
    w = np.array([cand["card_probs"][c] for c in cards])
    w = w / w.sum()
    out = []
    for _ in range(R):
        card = rng.choices(cards, weights=w)[0]
        out.append(rollout_returns(cand, agent, card, 1, seat=seat)[0])
    return np.array(out)


def collect(agent, max_games, cap_cal, cap_trump, seed):
    rng = random.Random(seed)
    cal, trump_pref = [], []
    for g in range(max_games):
        if len(cal) >= cap_cal and len(trump_pref) >= cap_trump:
            break
        game = Game(partner_selection_mode=get_partner_selection_mode(g))
        agent.reset_recurrent_state()
        if not advance_to_trick0(game, agent):
            continue
        leader = game.players[0]
        if leader.is_picker or leader.is_partner or leader.is_secret_partner:
            continue
        valid = leader.get_valid_action_ids()
        mem_m0 = snapshot_memory(agent)  # memory BEFORE the decision forward
        state = leader.get_state_dict()
        probs, value = policy_and_value(agent, state, valid, pid=1)
        card_probs = play_card_distribution(probs, valid)
        trump_card = best_in_class(card_probs, True)
        fail_card = best_in_class(card_probs, False)
        if trump_card is None or fail_card is None:
            continue
        top_card = max(card_probs, key=lambda c: card_probs[c])
        cand = {
            # snapshot at M0: rollouts re-run the decision forward exactly once,
            # so on-policy rollouts mirror real play with no double-encode.
            "game": copy.deepcopy(game),
            "mem": mem_m0,
            "value": value,
            "card_probs": card_probs,
            "trump_card": trump_card,
            "fail_card": fail_card,
            "p_trump": float(sum(p for c, p in card_probs.items() if c in TRUMP)),
            "n_trump": sum(1 for c in leader.hand if c in TRUMP),
            "argmax_is_trump": top_card in TRUMP,
        }
        if len(cal) < cap_cal:
            cal.append(cand)
        if cand["argmax_is_trump"] and len(trump_pref) < cap_trump:
            trump_pref.append(cand)
    return cal, trump_pref


def calibration(cal, agent, R, rng):
    V = np.array([c["value"] for c in cal])
    G_mean = np.zeros(len(cal))
    within_std = np.zeros(len(cal))
    for i, c in enumerate(cal):
        rs = rollout_onpolicy(c, agent, R, rng)
        G_mean[i] = rs.mean()
        within_std[i] = rs.std(ddof=1)

    err = V - G_mean
    bias = err.mean()
    rmse = np.sqrt((err**2).mean())
    mae = np.abs(err).mean()
    pear = np.corrcoef(V, G_mean)[0, 1]
    sp = np.corrcoef(np.argsort(np.argsort(V)), np.argsort(np.argsort(G_mean)))[0, 1]
    ss_res = np.sum((G_mean - V) ** 2)
    ss_tot = np.sum((G_mean - G_mean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    var_G = G_mean.var()
    var_adv = (G_mean - V).var()
    var_reduction = 1 - var_adv / var_G if var_G > 0 else float("nan")

    print("\n" + "=" * 72)
    print(f"(A) CRITIC CALIBRATION over first-trick defender leads (n={len(cal)})")
    print("=" * 72)
    print(f"  Return units: discounted (trick rewards + final_score/12), gamma={GAMMA}")
    print(
        f"  Realized return  mean {G_mean.mean():+.4f}  std(across states) {G_mean.std():.4f}"
    )
    print(f"  Critic V(s)      mean {V.mean():+.4f}  std(across states) {V.std():.4f}")
    print(f"  Bias  E[V - G]          : {bias:+.4f}")
    print(f"  RMSE                    : {rmse:.4f}")
    print(f"  MAE                     : {mae:.4f}")
    print(f"  Pearson  r(V, G)        : {pear:+.3f}")
    print(f"  Spearman rho            : {sp:+.3f}")
    print(f"  R^2 (V predicts G)      : {r2:+.3f}")
    print(
        f"  Baseline variance cut   : {var_reduction * 100:+.1f}%  "
        f"(var advantage {var_adv:.4f} vs var return {var_G:.4f})"
    )
    print(
        f"  Mean within-state noise : {within_std.mean():.4f}  "
        f"(per-rollout return std; irreducible sampling noise)"
    )
    return {
        "rmse": rmse,
        "r2": r2,
        "within_std": within_std.mean(),
        "return_std": G_mean.std(),
    }


def leak_signal(trump_pref, agent, R):
    if not trump_pref:
        print("\n(B) No trump-pref states collected for leak-signal analysis.")
        return
    gaps, pooled_std, adv_trump, adv_fail = [], [], [], []
    for c in trump_pref:
        gt = rollout_returns(c, agent, c["trump_card"], R)
        gf = rollout_returns(c, agent, c["fail_card"], R)
        gaps.append(gt.mean() - gf.mean())
        pooled_std.append(np.sqrt((gt.var(ddof=1) + gf.var(ddof=1)) / 2))
        adv_trump.append(gt.mean() - c["value"])
        adv_fail.append(gf.mean() - c["value"])
    gaps = np.array(gaps)
    pooled_std = np.array(pooled_std)
    adv_trump = np.array(adv_trump)
    adv_fail = np.array(adv_fail)
    mean_gap = gaps.mean()
    mean_std = pooled_std.mean()
    n_needed = (2 * mean_std / abs(mean_gap)) ** 2 if mean_gap != 0 else float("inf")

    print("\n" + "=" * 72)
    print(f"(B) LEAK SIGNAL on trump-pref first-trick states (n={len(trump_pref)})")
    print("=" * 72)
    print(f"  Action-value gap  E[G_trump - G_fail] : {mean_gap:+.4f} reward units")
    print("    (negative => fail lead is better, matching the counterfactual)")
    print(f"  Per-sample return noise (pooled std)  : {mean_std:.4f}")
    print(f"  Signal-to-noise per rollout           : {abs(mean_gap) / mean_std:.3f}")
    print(f"  On-policy samples to resolve gap (2σ) : ~{n_needed:.0f} per action")
    print(f"  Mean advantage of TRUMP (chosen) a-V  : {adv_trump.mean():+.4f}")
    print(f"  Mean advantage of FAIL  (alt)   a-V   : {adv_fail.mean():+.4f}")
    print(
        f"  Mean P(trump) the policy assigns      : {np.mean([c['p_trump'] for c in trump_pref]):.3f}"
    )
    print(
        f"  Mean P(best fail) approx              : "
        f"{np.mean([max((p for cc, p in c['card_probs'].items() if cc not in TRUMP), default=0) for c in trump_pref]):.3f}"
    )


def interpret(stats):
    print("\n" + "=" * 72)
    print("READING THE RESULT")
    print("=" * 72)
    if stats is None:
        return
    r2, rmse, wn, rstd = (
        stats["r2"],
        stats["rmse"],
        stats["within_std"],
        stats["return_std"],
    )
    print(f"  Critic explains R^2={r2:+.2f} of return variance across these states.")
    if r2 >= 0.5:
        print("  -> Critic is a DECENT baseline here. The first-trick leak is then")
        print("     primarily small-gap-vs-noise + exploration, not critic error.")
        print("     Highest-leverage fix = search/ExIt targets (evaluate the fail lead")
        print("     WITHOUT needing to sample it on-policy). gamma->0.99 still helps.")
    elif r2 >= 0.2:
        print("  -> Critic is a WEAK baseline. Both levers matter: a better/decoupled")
        print("     critic (deeper value trunk, higher value coeff, gamma->0.99) AND")
        print("     search-based targets.")
    else:
        print("  -> Critic is NOT predictive in these states (advantages are mostly")
        print(
            "     noise). A retrain with critic improvements is well justified before"
        )
        print(
            "     anything else; PPO currently cannot learn fine action distinctions here."
        )
    print(
        f"  RMSE {rmse:.3f} vs cross-state return std {rstd:.3f}; "
        f"per-rollout noise {wn:.3f}."
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("--max-games", type=int, default=6000)
    ap.add_argument(
        "--cap-cal",
        type=int,
        default=250,
        help="Number of first-trick defender-lead states for calibration.",
    )
    ap.add_argument(
        "--cap-trump",
        type=int,
        default=40,
        help="Number of trump-pref states for the leak-signal analysis.",
    )
    ap.add_argument("--rollouts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Loading {args.model} (device={DEV}) ...")
    agent = PPOAgent(len(ACTIONS))
    agent.load(args.model, load_optimizers=False)

    print(
        f"Scanning up to {args.max_games} games for first-trick defender leads "
        f"(need {args.cap_cal} calibration + {args.cap_trump} trump-pref) ..."
    )
    cal, trump_pref = collect(
        agent, args.max_games, args.cap_cal, args.cap_trump, args.seed
    )
    print(
        f"  Collected {len(cal)} calibration states, {len(trump_pref)} trump-pref states."
    )
    print(f"Running {args.rollouts} rollouts/state ...")

    stats = calibration(cal, agent, args.rollouts, rng)
    leak_signal(trump_pref, agent, args.rollouts)
    interpret(stats)


if __name__ == "__main__":
    main()
