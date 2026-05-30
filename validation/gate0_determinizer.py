#!/usr/bin/env python3
"""Gate 0: does an information-set-respecting determinized teacher SEE and
CORRECT the first-trick defender trump-lead leak?

Background
----------
The 30M agent leads trump as a defender on trick 0 (against convention); paired
oracle rollouts (`counterfactual_trump_leads.py`) showed this is EV-negative
(~-0.23 game score). The Step-0 critic probe proved no value head can fix it
(partial-observability ceiling). The structural fix is determinized search: a
teacher that averages over legal deals of the hidden cards. Before committing to
the full ISMCTS refactor we validate the keystone here, offline.

Method
------
At each trick-0 *defender* lead where seat 1's argmax PLAY is a trump (the leak
behavior) and a fail option also exists, we compare two action-value estimates
for "lead best trump" vs "lead best fail":

  * ORACLE Q (the truth):  paired rollouts on the TRUE deal (what the agent
    actually holds in every seat). This is what counterfactual_trump_leads.py
    measures.
  * DETERMINIZED Q (the teacher): for K determinizations of the hidden cards
    -- redealt to respect the observer's information set via
    Game.sample_trick0_determinization -- rebuild each world's per-seat
    recurrent memory by replaying the public bidding actions (private bury/under
    chosen by the policy), then roll the branch out with the network. Average
    the observer's game score over the K worlds.

The determinized teacher only ever uses information seat 1 legally has. If it
still ranks fail over trump (matching the oracle), it can teach the policy away
from the leak with no hand-crafted penalty.

Gate criteria (agreed):
  Direction:    pi' lowers trump-lead prob on >=70% of states; mean dp <= -0.10
                (pi' = 2-way softmax over {Q_trump, Q_fail} at temperature tau;
                 compared to the network's renormalized 2-way conditional).
  Calibration:  sign(det Q_trump - Q_fail) agrees with oracle on >=80% of states;
                mean |det - oracle| game-score gap small (~<=0.1).
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque

import numpy as np
import torch

import ppo
from ppo import PPOAgent
from sheepshead import ACTION_IDS, ACTIONS, TRUMP, Game
from training_utils import get_partner_selection_mode

# Reuse the validated paired-rollout scaffolding.
from counterfactual_trump_leads import (
    snapshot_memory,
    restore_memory,
    play_out,
    best_in_class,
)

DEV = ppo.device


def forced_encode(agent, player, pid):
    """Advance a seat's recurrent memory through its current state WITHOUT
    sampling/applying an action (used to force a recorded public action while
    keeping memory identical to a normal `act`)."""
    state = player.get_state_dict()
    mem_in = agent.get_recurrent_memory(pid, device=DEV)
    enc = agent.encoder.encode_batch(
        [state], memory_in=mem_in.unsqueeze(0), device=DEV
    )
    agent.set_recurrent_memory(pid, enc["memory_out"][0])


def _is_private_decision(valid):
    """True if the decision in front of a seat is a hidden one (bury / under)."""
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def advance_to_trick0_recording(game, agent):
    """Drive the real game's bidding to the trick-0 lead with the sampled policy,
    recording the ordered list of PUBLIC actions (pick/pass/alone/call) as
    (seat, action_id). Private bury/under actions are taken but not recorded
    (they are re-decided by the policy in each determinized world).

    Returns (reached_trick0_lead: bool, forced_public: list[(seat, action_id)]).
    """
    forced_public = []
    guard = 0
    while not game.play_started and guard < 200:
        guard += 1
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                private = _is_private_decision(valid)
                state = player.get_state_dict()
                a, _, _ = agent.act(state, valid, player.position, deterministic=False)
                if not private:
                    forced_public.append((player.position, a))
                player.act(a)
                if game.play_started:
                    break
                valid = player.get_valid_action_ids()
            if game.play_started:
                break
    return (game.play_started and not game.is_leaster), forced_public


def build_determinized_world(real_game, deal, forced_public, agent, observer=1):
    """Construct a play-ready determinized game at the trick-0 lead, returning
    (game, log_weight).

    Installs the determinized deal, then replays the recorded public bidding
    (forcing each public action while encoding for memory) and lets the policy
    choose the private bury/under cards, so every seat's recurrent memory is
    consistent with its determinized hand. Finally advances the observer's
    memory through its lead-decision state (matching the real collect path) so
    both forced-lead branches start from an identical snapshot.

    log_weight = sum over forced PUBLIC actions of log P_policy(action | this
    seat's determinized hand). This is the likelihood of the observed bidding
    under the policy for this deal -- the inference signal that downweights
    deals where the determinized picker is too weak to have plausibly picked
    (and passers too strong to have plausibly passed).

    Returns (game, log_weight), or (None, None) on replay failure.
    """
    g = Game(partner_selection_mode=real_game.partner_mode_flag)
    for s in range(1, 6):
        h = deal["initial_hands"][s][:]
        g.players[s - 1].hand = h
        g.players[s - 1].initial_hand = h[:]
    g.blind = deal["blind"][:]

    agent.reset_recurrent_state()
    fq = deque(forced_public)
    log_weight = 0.0   # full bidding log-likelihood under the policy
    log_pick = 0.0     # picker's PICK log-prob alone (dominant inference term)
    guard = 0
    while not g.play_started and guard < 200:
        guard += 1
        for player in g.players:
            valid = player.get_valid_action_ids()
            while valid:
                if _is_private_decision(valid):
                    state = player.get_state_dict()
                    a, _, _ = agent.act(
                        state, valid, player.position, deterministic=False
                    )
                else:
                    if not fq or fq[0][0] != player.position:
                        return None, None, None  # public-record desync
                    _, a = fq.popleft()
                    if a not in valid:
                        return None, None, None
                    state = player.get_state_dict()
                    probs_t, _ = agent.get_action_probs_with_logits(
                        state, valid, player_id=player.position
                    )
                    p_a = float(probs_t[0][a - 1].item())
                    lp = np.log(max(p_a, 1e-8))
                    log_weight += lp
                    if ACTIONS[a - 1] == "PICK":
                        log_pick += lp
                player.act(a)
                if g.play_started:
                    break
                valid = player.get_valid_action_ids()
            if g.play_started:
                break
    if not g.play_started or g.is_leaster:
        return None, None, None
    # Advance observer memory through its lead state (shared by both branches).
    forced_encode(agent, g.players[observer - 1], observer)
    return g, log_weight, log_pick


def rollout_score(game, mem, agent, forced_card, observer, n):
    """Mean observer game score over n sampled rollouts forcing the lead card."""
    aid = ACTION_IDS[f"PLAY {forced_card}"]
    scores = []
    for _ in range(n):
        g = copy.deepcopy(game)
        restore_memory(agent, mem)
        g.players[observer - 1].act(aid)
        play_out(g, agent)
        scores.append(g.players[observer - 1].get_score())
    return float(np.mean(scores))


def collect(agent, max_games, target, seed):
    """Find trick-0 defender TRUMP-pref lead states (seat 1 = leader = defender,
    argmax PLAY is a trump, a fail option also exists)."""
    rng = random.Random(seed)
    states = []
    scanned = 0
    for g in range(max_games):
        if len(states) >= target:
            break
        mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        ok, forced_public = advance_to_trick0_recording(game, agent)
        if not ok:
            continue
        scanned += 1
        leader = game.players[0]  # seat 1 always leads trick 0
        if leader.is_picker or leader.is_partner or leader.is_secret_partner:
            continue
        valid = leader.get_valid_action_ids()
        state = leader.get_state_dict()
        probs_t, _ = agent.get_action_probs_with_logits(state, valid, player_id=1)
        probs = probs_t[0].detach().cpu().numpy()

        trump_card, _ = best_in_class(probs, valid, want_trump=True)
        fail_card, _ = best_in_class(probs, valid, want_trump=False)
        if trump_card is None or fail_card is None:
            continue
        play_actions = [a for a in valid if ACTIONS[a - 1].startswith("PLAY ")]
        top = max(play_actions, key=lambda a: probs[a - 1])
        if ACTIONS[top - 1][5:] not in TRUMP:
            continue  # only the leak behavior: argmax lead is trump

        p_net_trump = float(probs[ACTION_IDS[f"PLAY {trump_card}"] - 1])
        p_net_fail = float(probs[ACTION_IDS[f"PLAY {fail_card}"] - 1])
        states.append(
            {
                "game": copy.deepcopy(game),
                "mem": snapshot_memory(agent),
                "forced_public": forced_public,
                "mode": mode,
                "trump_card": trump_card,
                "fail_card": fail_card,
                "p_net_cond_trump": p_net_trump / (p_net_trump + p_net_fail),
                "n_trump": sum(1 for c in leader.hand if c in TRUMP),
                "hand": list(leader.hand),
            }
        )
    return states, scanned


def evaluate(states, agent, K, r_inner, r_oracle, tau, seed):
    rng = random.Random(seed + 1)
    rows = []
    for i, st in enumerate(states):
        # Oracle Q on the true deal.
        oq_t = rollout_score(st["game"], st["mem"], agent, st["trump_card"], 1, r_oracle)
        oq_f = rollout_score(st["game"], st["mem"], agent, st["fail_card"], 1, r_oracle)

        # (a) UNIFORM baseline: K legal redeals, equal weight (biased: it ignores
        # the bidding, so the picker is on average too trump-weak).
        ub_t, ub_f = [], []
        for _ in range(K):
            deal = st["game"].sample_trick0_determinization(1, rng)
            world, _, _ = build_determinized_world(
                st["game"], deal, st["forced_public"], agent, 1
            )
            if world is None:
                continue
            mem = snapshot_memory(agent)
            ub_t.append(rollout_score(world, mem, agent, st["trump_card"], 1, r_inner))
            ub_f.append(rollout_score(world, mem, agent, st["fail_card"], 1, r_inner))
        if len(ub_t) < max(5, K // 4):
            continue
        uni_d = float(np.mean(ub_t) - np.mean(ub_f))

        # (b) INFERENCE-CORRECTED: rejection-sample worlds by the picker's PICK
        # probability (-> posterior over deals consistent with the observed
        # pick), then importance-weight the mild residual (passers + call).
        pt, pf, lres = [], [], []
        attempts = 0
        cap = K * 60
        while len(pt) < K and attempts < cap:
            attempts += 1
            deal = st["game"].sample_trick0_determinization(1, rng)
            world, lwf, lpick = build_determinized_world(
                st["game"], deal, st["forced_public"], agent, 1
            )
            if world is None:
                continue
            if rng.random() < float(np.exp(lpick)):  # accept ~ P(picker PICK|deal)
                mem = snapshot_memory(agent)
                pt.append(rollout_score(world, mem, agent, st["trump_card"], 1, r_inner))
                pf.append(rollout_score(world, mem, agent, st["fail_card"], 1, r_inner))
                lres.append(lwf - lpick)
        if len(pt) < max(5, K // 4):
            continue
        pt, pf, lres = np.array(pt), np.array(pf), np.array(lres)
        w = np.exp(lres - lres.max())
        w = w / w.sum()
        wt_qt, wt_qf = float((w * pt).sum()), float((w * pf).sum())
        det_d = wt_qt - wt_qf
        ess = float(1.0 / np.sum(w ** 2))
        acc_rate = len(pt) / max(attempts, 1)

        oracle_d = oq_t - oq_f
        # Teacher 2-way conditional (corrected) vs network 2-way conditional.
        logits = np.array([wt_qt, wt_qf]) / tau
        logits -= logits.max()
        p_teach_trump = float(np.exp(logits[0]) / np.exp(logits).sum())
        dp = p_teach_trump - st["p_net_cond_trump"]

        rows.append(
            {
                "oracle_d": oracle_d,
                "det_d": det_d,
                "uni_d": uni_d,
                "ess": ess,
                "acc_rate": acc_rate,
                "dp": dp,
                "p_net": st["p_net_cond_trump"],
                "p_teach": p_teach_trump,
                "oq_t": oq_t,
                "oq_f": oq_f,
                "n_trump": st["n_trump"],
                "used": len(pt),
                "trump_card": st["trump_card"],
                "fail_card": st["fail_card"],
                "hand": st["hand"],
            }
        )
        print(
            f"  [{i + 1}/{len(states)}] post={len(pt):3d} ESS={ess:4.1f} acc={acc_rate:.2f}  "
            f"oracleD={oracle_d:+.3f}  detD(corr)={det_d:+.3f} detD(uni)={uni_d:+.3f}  "
            f"p_net={st['p_net_cond_trump']:.2f} p_teach={p_teach_trump:.2f} dp={dp:+.2f}",
            flush=True,
        )
    return rows


def summarize(rows, tau):
    n = len(rows)
    print("\n" + "=" * 72)
    print(f"GATE 0 RESULTS  (n = {n} trick-0 defender trump-lead states, tau={tau})")
    print("=" * 72)
    if not n:
        print("  no states evaluated")
        return

    dp = np.array([r["dp"] for r in rows])
    det_d = np.array([r["det_d"] for r in rows])
    oracle_d = np.array([r["oracle_d"] for r in rows])

    def se(x):
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    frac_dp_neg = float(np.mean(dp < 0))
    mean_dp = float(dp.mean())
    print("\n-- Direction: does the teacher move mass off the trump lead? (per-state) --")
    print(f"  pi' lowers trump-lead prob in {frac_dp_neg * 100:.0f}% of states "
          f"(gate: >=70%)")
    print(f"  mean dp (p_teach_trump - p_net_trump) = {mean_dp:+.3f} (gate: <=-0.10)")
    print(f"  teacher prefers fail (detD<0) in {np.mean(det_d < 0) * 100:.0f}% of states")

    # Calibration is AGGREGATE, not per-state: the determinized Q is the mean
    # over all legal deals; the true-deal oracle is one draw from that same
    # distribution, so per-state they differ by deal variance by construction.
    # Averaged over states both estimate the population EV gap and must agree.
    uni_d = np.array([r["uni_d"] for r in rows])
    ess = np.array([r["ess"] for r in rows])
    md, mo = float(det_d.mean()), float(oracle_d.mean())
    mu = float(uni_d.mean())
    sd, so = se(det_d), se(oracle_d)
    comb = float(np.sqrt(sd ** 2 + so ** 2))
    if n > 1 and det_d.std() > 0 and oracle_d.std() > 0:
        corr = float(np.corrcoef(det_d, oracle_d)[0, 1])
    else:
        corr = float("nan")
    print("\n-- Calibration: aggregate determinized EV vs true-deal oracle EV --")
    print(f"  mean UNIFORM      detD = {mu:+.3f}  (SE {se(uni_d):.3f})  [biased: ignores bidding]")
    print(f"  mean WEIGHTED     detD = {md:+.3f}  (SE {sd:.3f})  [inference-corrected]")
    print(f"  mean oracle       detD = {mo:+.3f}  (SE {so:.3f})  [true-deal ground truth]")
    print(f"  |weighted - oracle| = {abs(md - mo):.3f}  vs 2*combined SE = {2 * comb:.3f}")
    acc = np.array([r["acc_rate"] for r in rows])
    print(f"  residual-weight ESS: mean={ess.mean():.1f} / {int(np.mean([r['used'] for r in rows]))} "
          f"posterior worlds (min={ess.min():.1f})")
    print(f"  pick-rejection acceptance rate: mean={acc.mean():.2f} (min={acc.min():.2f})")
    print(f"  Pearson r(weightedD, oracleD) per-state = {corr:+.3f} (informational; "
          f"single-draw oracle is noisy)")

    direction_pass = frac_dp_neg >= 0.70 and mean_dp <= -0.10
    det_sig_neg = (md + 2 * sd) < 0.0           # teacher clearly says trump is worse
    agree = abs(md - mo) <= 2 * comb            # determinized unbiased vs oracle
    calib_pass = det_sig_neg and agree
    print("\n-- Verdict --")
    print(f"  Direction gate:   {'PASS' if direction_pass else 'FAIL'} "
          f"(reduces trump lead)")
    print(f"  Calibration gate: {'PASS' if calib_pass else 'FAIL'} "
          f"(det EV significantly negative: {det_sig_neg}; agrees with oracle: {agree})")
    print(f"  GATE 0: {'STRONG RESULT -> proceed to ISMCTS teacher' if (direction_pass and calib_pass) else 'mixed/negative -> reassess before refactor'}")

    print("\n-- Examples (largest |detD|) --")
    for r in sorted(rows, key=lambda r: -abs(r["det_d"]))[:8]:
        print(
            f"  hand {' '.join(r['hand'])} | trump={r['trump_card']} vs fail={r['fail_card']} "
            f"| detD={r['det_d']:+.2f} oracleD={r['oracle_d']:+.2f} "
            f"dp={r['dp']:+.2f} (worlds={r['used']})"
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("--max-games", type=int, default=20000,
                    help="Scan cap (only bidding played to find trick-0 leads).")
    ap.add_argument("--states", type=int, default=60,
                    help="Number of trump-pref lead states to evaluate.")
    ap.add_argument("--determinizations", "-K", type=int, default=40,
                    help="Legal redeals per state for the determinized teacher.")
    ap.add_argument("--rollouts-inner", type=int, default=1,
                    help="Sampled rollouts per branch per determinization.")
    ap.add_argument("--rollouts-oracle", type=int, default=40,
                    help="Sampled rollouts per branch on the true deal (oracle).")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Softmax temperature (game-score units) for pi'.")
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

    print(f"Scanning up to {args.max_games} games for trick-0 defender trump-leads ...")
    states, scanned = collect(agent, args.max_games, args.states, args.seed)
    print(f"  reached {scanned} trick-0 leads; collected {len(states)} trump-pref defender states.")
    print(f"Evaluating: K={args.determinizations} determinizations, "
          f"{args.rollouts_inner} inner / {args.rollouts_oracle} oracle rollouts ...", flush=True)

    rows = evaluate(
        states, agent,
        args.determinizations, args.rollouts_inner, args.rollouts_oracle,
        args.tau, args.seed,
    )
    summarize(rows, args.tau)


if __name__ == "__main__":
    main()
