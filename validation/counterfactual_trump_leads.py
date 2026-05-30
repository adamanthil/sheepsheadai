#!/usr/bin/env python3
"""Counterfactual EV of defender trump leads on the FIRST trick.

Question: when the agent (as a defender, on lead, trick 0, with a fail option)
prefers to lead trump, does forcing that trump lead actually win more points
than forcing its best *fail* lead from the same deal?

Method
------
Trick 0 is always led by seat 1 (the engine sets leader=1 after the bury), so we
reach every trick-0 lead by playing only the pick/call/bury phase. At each such
state where seat 1 is a *defender* (not picker/partner/secret-partner) and holds
both a trump and a fail option, we snapshot the full game + per-seat recurrent
memory. We then run paired Monte-Carlo rollouts from that identical state:

  * branch TRUMP: force seat 1 to lead its best (argmax) trump
  * branch FAIL : force seat 1 to lead its best (argmax) fail

Each branch is rolled out R times (all 5 seats driven by the same policy,
sampled) to completion. The deal is fixed, so the only difference between
branches is the first card; the only stochasticity is the policy's own sampling
over the continuation. We report the defending team's final card points, the
leader's game score (true RL payoff, incl. schneider / double-on-bump), and
defender win rate, trump minus fail.

We separately aggregate:
  * TRUMP-PREF states  -> the agent's argmax lead is a trump (the behavior in
    question). Positive delta => the trump lead is genuinely +EV.
  * FAIL-PREF states   -> control / method check; we expect fail >= trump here.
"""

from __future__ import annotations

import argparse
import copy
import random

import numpy as np
import torch

import ppo
from ppo import PPOAgent
from sheepshead import (
    ACTION_IDS,
    ACTIONS,
    TRUMP,
    Game,
    get_card_suit,
)
from training_utils import get_partner_selection_mode

DEV = ppo.device


def snapshot_memory(agent):
    return {pid: t.detach().clone() for pid, t in agent._player_memories.items()}


def restore_memory(agent, snap):
    agent._player_memories = {pid: t.detach().clone() for pid, t in snap.items()}


def advance_to_trick0(game, agent):
    """Drive pick/call/bury with the sampled policy until play starts.

    Returns True iff we reached a normal (non-leaster) trick-0 lead.
    """
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
    """Continue a mid-game state to completion (sampled), with end-of-trick
    memory propagation exactly like play.py."""
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


def best_in_class(probs, valid, want_trump):
    """Return (card, prob_mass_of_class) for the argmax PLAY action in the class."""
    best_card, best_p = None, -1.0
    mass = 0.0
    for a in valid:
        name = ACTIONS[a - 1]
        if not name.startswith("PLAY "):
            continue
        card = name[5:]
        is_trump = card in TRUMP
        if is_trump != want_trump:
            continue
        p = probs[a - 1]
        mass += p
        if p > best_p:
            best_p, best_card = p, card
    return best_card, mass


def rollout_branch(cand, agent, forced_card, R):
    aid = ACTION_IDS[f"PLAY {forced_card}"]
    leader_pos = cand["leader_pos"]
    pts, scores, wins = [], [], []
    for _ in range(R):
        g = copy.deepcopy(cand["game"])
        restore_memory(agent, cand["mem"])
        g.players[leader_pos - 1].act(aid)  # forced first-trick lead
        play_out(g, agent)
        dp = g.get_final_defender_points()
        pts.append(dp)
        scores.append(g.players[leader_pos - 1].get_score())
        wins.append(1 if dp >= 60 else 0)
    return np.mean(pts), np.mean(scores), np.mean(wins)


def collect(agent, max_games, target_trump, control_per_trump, seed):
    rng = random.Random(seed)
    trump_pref, fail_pref = [], []
    scanned = 0
    defender_leads = 0
    for g in range(max_games):
        if len(trump_pref) >= target_trump:
            break
        partner_mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=partner_mode)
        agent.reset_recurrent_state()
        if not advance_to_trick0(game, agent):
            continue
        scanned += 1
        leader = game.players[0]  # seat 1 always leads trick 0
        if leader.is_picker or leader.is_partner or leader.is_secret_partner:
            continue
        valid = leader.get_valid_action_ids()
        state = leader.get_state_dict()
        # This forward also advances seat-1 memory to its post-decision state,
        # which is exactly what both branches should start from.
        probs_t, _ = agent.get_action_probs_with_logits(state, valid, player_id=1)
        probs = probs_t[0].detach().cpu().numpy()

        trump_card, p_trump = best_in_class(probs, valid, want_trump=True)
        fail_card, _ = best_in_class(probs, valid, want_trump=False)
        if trump_card is None or fail_card is None:
            continue  # need both classes available
        defender_leads += 1

        # argmax over all valid PLAY actions
        play_actions = [a for a in valid if ACTIONS[a - 1].startswith("PLAY ")]
        top = max(play_actions, key=lambda a: probs[a - 1])
        argmax_is_trump = ACTIONS[top - 1][5:] in TRUMP

        hand = list(leader.hand)
        cand = {
            "game": copy.deepcopy(game),
            "mem": snapshot_memory(agent),
            "leader_pos": 1,
            "p_trump": float(p_trump),
            "trump_card": trump_card,
            "fail_card": fail_card,
            "n_trump": sum(1 for c in hand if c in TRUMP),
            "n_fail": sum(1 for c in hand if c not in TRUMP),
            "partner_mode": partner_mode,
            "hand": hand,
        }
        if argmax_is_trump:
            trump_pref.append(cand)
        elif len(fail_pref) < target_trump * control_per_trump:
            # keep a random control subset
            if rng.random() < 0.5:
                fail_pref.append(cand)
    return trump_pref, fail_pref, scanned, defender_leads


def evaluate(group, agent, R):
    rows = []
    for cand in group:
        tp, ts, tw = rollout_branch(cand, agent, cand["trump_card"], R)
        fp, fs, fw = rollout_branch(cand, agent, cand["fail_card"], R)
        rows.append(
            {
                "d_points": tp - fp,
                "d_score": ts - fs,
                "d_win": tw - fw,
                "trump_points": tp,
                "fail_points": fp,
                "trump_score": ts,
                "fail_score": fs,
                "trump_win": tw,
                "fail_win": fw,
                "n_trump": cand["n_trump"],
                "p_trump": cand["p_trump"],
                "trump_card": cand["trump_card"],
                "fail_card": cand["fail_card"],
                "hand": cand["hand"],
            }
        )
    return rows


def summarize(label, rows):
    n = len(rows)
    print("\n" + "=" * 72)
    print(f"{label}  (n = {n} states)")
    print("=" * 72)
    if not n:
        print("  (no states)")
        return
    for key, name, unit in [
        ("d_points", "Defender team points (trump - fail)", "pts"),
        ("d_score", "Leader game score (trump - fail)", "score"),
        ("d_win", "Defender win rate (trump - fail)", ""),
    ]:
        vals = np.array([r[key] for r in rows])
        mean = vals.mean()
        stderr = vals.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        frac_pos = np.mean(vals > 0)
        scale = 100 if key == "d_win" else 1
        print(
            f"  {name:<40} {mean*scale:+7.2f} {unit} "
            f"(SE {stderr*scale:5.2f}, trump better in {frac_pos*100:.0f}% of states)"
        )
    tp = np.mean([r["trump_points"] for r in rows])
    fp = np.mean([r["fail_points"] for r in rows])
    print(f"  Absolute EV: trump-lead {tp:.1f} pts vs fail-lead {fp:.1f} pts "
          f"(defenders need 60 to win)")

    # By trump-richness of the leading hand.
    print("  --- by trump count in hand ---")
    for nt in sorted({r["n_trump"] for r in rows}):
        sub = [r for r in rows if r["n_trump"] == nt]
        dp = np.mean([r["d_points"] for r in sub])
        ds = np.mean([r["d_score"] for r in sub])
        print(f"    {nt} trump (n={len(sub):3d}): Δpoints {dp:+6.2f}, Δscore {ds:+6.3f}")


def print_examples(rows, label, n=6):
    print(f"\n--- {label}: example states (largest |Δscore|) ---")
    for r in sorted(rows, key=lambda r: -abs(r["d_score"]))[:n]:
        print(
            f"  hand {' '.join(r['hand'])} | trump={r['trump_card']} vs fail={r['fail_card']} "
            f"| Δpts {r['d_points']:+5.1f} Δscore {r['d_score']:+5.2f} "
            f"(trump {r['trump_points']:.0f}pts/{r['trump_score']:+.2f} vs "
            f"fail {r['fail_points']:.0f}pts/{r['fail_score']:+.2f})"
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("--max-games", type=int, default=15000,
                    help="Scan cap (only pick/bury played to find trick-0 leads).")
    ap.add_argument("--target-trump", type=int, default=120,
                    help="Stop scanning after this many trump-pref states found.")
    ap.add_argument("--control-per-trump", type=float, default=1.0,
                    help="Collect up to target*this many fail-pref control states.")
    ap.add_argument("--rollouts", type=int, default=50,
                    help="Monte-Carlo rollouts per branch per state.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.model} (device={DEV}) ...")
    agent = PPOAgent(len(ACTIONS), activation="swish")
    agent.load(args.model, load_optimizers=False)

    print(f"Scanning up to {args.max_games} games for trick-0 defender leads ...")
    trump_pref, fail_pref, scanned, dleads = collect(
        agent, args.max_games, args.target_trump, args.control_per_trump, args.seed
    )
    print(f"  Reached {scanned} non-leaster trick-0 leads; "
          f"{dleads} were defender leads with both classes available.")
    print(f"  Collected {len(trump_pref)} TRUMP-pref and {len(fail_pref)} FAIL-pref states.")
    print(f"Running {args.rollouts} rollouts/branch/state ...")

    trump_rows = evaluate(trump_pref, agent, args.rollouts)
    fail_rows = evaluate(fail_pref, agent, args.rollouts)

    summarize("TRUMP-PREF first-trick leads (the behavior in question)", trump_rows)
    print_examples(trump_rows, "TRUMP-PREF")
    summarize("FAIL-PREF first-trick leads (control / method check)", fail_rows)

    print("\nInterpretation: Δ is (trump lead) - (fail lead). For TRUMP-PREF states, "
          "Δ>0 means the agent's first-trick trump lead really does win more; "
          "Δ<0 means it is a mistake. FAIL-PREF Δ should be <=0 if the method is sound.")


if __name__ == "__main__":
    main()
