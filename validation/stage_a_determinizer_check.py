#!/usr/bin/env python3
"""Stage A validation: does the GENERALIZED inference-weighted determinizer
(`Game.sample_determinization`) produce legal mid-game redeals whose
inference-corrected EV tracks the paired true-deal oracle at tricks 1-3?

This is the broader-coverage analogue of Gate 0 (which validated the trick-0-only
sampler). One-off analysis script -- NOT slated for commit.

For each DEFENDER lead decision at a target trick (the leader is a defender with
both a trump and a fail option in hand) we compare two action-value estimates for
"lead best trump" vs "lead best fail":

  * ORACLE Q (truth): paired rollouts on the TRUE deal from the node.
  * DETERMINIZED Q (teacher): redeal the hidden cards with
    Game.sample_determinization (honouring forced plays, per-seat counts,
    play-revealed voids, called-ace placement), rebuild every seat's recurrent
    memory by FORCED REPLAY of the entire recorded public sequence (bidding +
    forced determinized bury/under + all plays), inference-weight the worlds
    (rejection on the picker's PICK probability + residual importance weighting
    over passers/call/plays), then roll out each world with the network.

Legality of every sampled redeal is asserted directly (partition / counts /
voids / called-ace / bury) and, most stringently, by requiring the forced replay
to reproduce the real game's public history exactly and land on the same node.
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque

import numpy as np
import torch

import ppo
from ppo import load_agent
from sheepshead import (
    ACTION_IDS,
    ACTIONS,
    DECK,
    TRUMP,
    UNDER_TOKEN,
    Game,
    get_card_suit,
    get_callable_cards,
)
from training_utils import get_partner_selection_mode

from counterfactual_trump_leads import (
    snapshot_memory,
    restore_memory,
    play_out,
    best_in_class,
)

DEV = ppo.device


def _is_private_decision(valid):
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def forced_encode(agent, player, pid):
    """Advance a seat's recurrent memory through its current state WITHOUT
    sampling/applying an action (matches a normal act's memory update)."""
    state = player.get_state_dict()
    mem_in = agent.get_recurrent_memory(pid, device=DEV)
    enc = agent.encoder.encode_batch([state], memory_in=mem_in.unsqueeze(0), device=DEV)
    agent.set_recurrent_memory(pid, enc["memory_out"][0])


def _qualifies(player):
    """A defender lead with both a trump and a fail option in hand."""
    g = player.game
    if player.is_picker or player.is_partner or player.is_secret_partner:
        return False
    has_trump = any(c in TRUMP for c in player.hand)
    has_fail = any(c not in TRUMP for c in player.hand)
    return has_trump and has_fail


def drive_record(game, agent, target_tricks):
    """Play the whole game with the sampled policy, recording the ordered list of
    PUBLIC actions (seat, action_id). Stop at the FIRST defender lead at a target
    trick, leaving the game + memory exactly at that node (leader's lead state
    already encoded into its memory, matching counterfactual collect).

    Returns (node, forced_public) or (None, None). `node` carries observer seat,
    trick, trump/fail cards, and the network's 2-way conditional trump prob.
    """
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                # Detect a target lead node BEFORE acting.
                if (
                    game.play_started
                    and not game.is_leaster
                    and game.cards_played == 0
                    and game.current_trick in target_tricks
                    and game.leader == player.position
                    and _qualifies(player)
                ):
                    state = player.get_state_dict()
                    probs_t, _ = agent.get_action_probs_with_logits(
                        state, valid, player_id=player.position
                    )
                    probs = probs_t[0].detach().cpu().numpy()
                    trump_card, _ = best_in_class(probs, valid, want_trump=True)
                    fail_card, _ = best_in_class(probs, valid, want_trump=False)
                    if trump_card is None or fail_card is None:
                        # Shouldn't happen given _qualifies, but be safe.
                        a, _, _ = agent.act(
                            state, valid, player.position, deterministic=False
                        )
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        continue
                    pt = float(probs[ACTION_IDS[f"PLAY {trump_card}"] - 1])
                    pf = float(probs[ACTION_IDS[f"PLAY {fail_card}"] - 1])
                    node = {
                        "observer": player.position,
                        "trick": game.current_trick,
                        "trump_card": trump_card,
                        "fail_card": fail_card,
                        "p_net_cond_trump": pt / (pt + pf) if (pt + pf) > 0 else 0.5,
                        "hand": list(player.hand),
                        "n_trump": sum(1 for c in player.hand if c in TRUMP),
                    }
                    return node, forced_public

                private = _is_private_decision(valid)
                state = player.get_state_dict()
                a, _, _ = agent.act(state, valid, player.position, deterministic=False)
                if not private:
                    forced_public.append((player.position, a))
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )
    return None, None


_FAIL = {}


def _fail(reason):
    _FAIL[reason] = _FAIL.get(reason, 0) + 1
    return None, None, None


def build_world(real_game, deal, forced_public, observer, target_trick, agent):
    """Rebuild a play-ready determinized world at the observer's target-trick lead
    by forced replay. Returns (game, log_weight, log_pick) or (None, None, None).

    Forces every recorded public action and the determinized bury/under, doing the
    same end-of-trick memory propagation as play_out, so every seat's recurrent
    memory matches its determinized hand. log_weight = sum of policy log-probs of
    all public forced actions (bidding + plays); log_pick = the picker's PICK
    log-prob alone (the dominant inference factor for rejection sampling).
    """
    g = Game(partner_selection_mode=real_game.partner_mode_flag)
    for s in range(1, 6):
        h = deal["initial_hands"][s][:]
        g.players[s - 1].hand = h
        g.players[s - 1].initial_hand = h[:]
    g.blind = deal["blind"][:]

    agent.reset_recurrent_state()
    pub = deque(forced_public)
    det_bury = deque(deal["bury"])
    det_under = deal["under_card"]
    log_weight = 0.0
    log_pick = 0.0
    guard = 0
    while True:
        guard += 1
        if guard > 5000:
            return _fail("guard")
        acted = False
        for player in g.players:
            valid = player.get_valid_action_ids()
            while valid:
                # Node reached: observer about to lead the target trick, no
                # public actions left to force. Checked at the top of the inner
                # loop (not just per-player) because the observer may complete
                # the prior trick and immediately lead the target trick within
                # the same continuation.
                if (
                    not pub
                    and g.play_started
                    and not g.is_leaster
                    and g.cards_played == 0
                    and g.current_trick == target_trick
                    and g.leader == observer
                    and player.position == observer
                ):
                    forced_encode(agent, player, observer)
                    if g.history != real_game.history:
                        return _fail("hist_mismatch")
                    return g, log_weight, log_pick

                if _is_private_decision(valid):
                    name_is_under = any(
                        ACTIONS[a - 1].startswith("UNDER ") for a in valid
                    )
                    if name_is_under:
                        if det_under is None:
                            return _fail("no_under")
                        aid = ACTION_IDS[f"UNDER {det_under}"]
                    else:
                        if not det_bury:
                            return _fail("no_bury")
                        aid = ACTION_IDS[f"BURY {det_bury.popleft()}"]
                    if aid not in valid:
                        return _fail("bad_private")
                    # Advance memory through this (forced, private) decision.
                    agent.get_action_probs_with_logits(
                        player.get_state_dict(), valid, player_id=player.position
                    )
                    player.act(aid)
                else:
                    if not pub or pub[0][0] != player.position:
                        return _fail("pub_desync")
                    _, aid = pub.popleft()
                    if aid not in valid:
                        return _fail("bad_public")
                    probs_t, _ = agent.get_action_probs_with_logits(
                        player.get_state_dict(), valid, player_id=player.position
                    )
                    # Inference weights the BIDDING likelihood only. Plays are
                    # honoured as hard void constraints in the determinizer (the
                    # dominant play-derived signal); importance-weighting the
                    # finer play-choice likelihood collapses ESS over a long
                    # history. Plays are still forced (to rebuild memory and
                    # reproduce the public record) but not weighted.
                    if not ACTIONS[aid - 1].startswith("PLAY "):
                        p_a = float(probs_t[0][aid - 1].item())
                        lp = np.log(max(p_a, 1e-8))
                        log_weight += lp
                        if ACTIONS[aid - 1] == "PICK":
                            log_pick += lp
                    player.act(aid)
                acted = True
                valid = player.get_valid_action_ids()
                if g.was_trick_just_completed:
                    for seat in g.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(), player_id=seat.position
                        )
        if not acted:
            return _fail("no_acted")


def rollout_score(game, mem, agent, forced_card, observer, n):
    aid = ACTION_IDS[f"PLAY {forced_card}"]
    scores = []
    for _ in range(n):
        g = copy.deepcopy(game)
        restore_memory(agent, mem)
        g.players[observer - 1].act(aid)
        play_out(g, agent)
        scores.append(g.players[observer - 1].get_score())
    return float(np.mean(scores))


def check_legality(real_game, deal, observer):
    """Assert a sampled redeal honours the observer's information set. Returns a
    list of violation strings (empty = legal)."""
    bad = []
    ih = deal["initial_hands"]
    blind, bury, under = deal["blind"], deal["bury"], deal["under_card"]
    picker = real_game.picker

    # Partition / counts.
    for s in range(1, 6):
        if len(ih[s]) != 6:
            bad.append(f"seat {s} dealt {len(ih[s])} != 6")
    allcards = [c for s in range(1, 6) for c in ih[s]] + list(blind)
    if sorted(allcards) != sorted(DECK):
        bad.append("initial_hands + blind != full deck (or duplicates)")

    # Observer's own dealt hand preserved.
    if sorted(ih[observer]) != sorted(real_game.players[observer - 1].initial_hand):
        bad.append("observer dealt hand altered")

    # Played cards must be dealt back to the seats that played them.
    played_by = {s: [] for s in range(1, 6)}
    for t in range(len(real_game.history)):
        for s in range(1, 6):
            c = real_game.history[t][s - 1]
            if c and c != UNDER_TOKEN:
                played_by[s].append(c)
    eight = ih[picker] + list(blind)
    for s in range(1, 6):
        pool = eight if s == picker else ih[s]
        for c in played_by[s]:
            if c not in pool:
                bad.append(f"seat {s} played {c} not in its dealt cards")

    # Picker bury / under live in the picker's pre-bury 8 and were never played.
    for c in list(bury) + ([under] if under else []):
        if c not in eight:
            bad.append(f"picker hidden card {c} not in pre-bury 8")
        if c in played_by[picker]:
            bad.append(f"picker buried/under {c} but also played it")

    # Voids: each seat's current hand has no card of a suit it is known void in.
    voids = real_game._play_revealed_voids()
    for s in range(1, 6):
        cur = set(eight if s == picker else ih[s])
        cur -= set(played_by[s])
        if s == picker:
            cur -= set(bury)
            if under:
                cur.discard(under)
        for c in cur:
            if get_card_suit(c) in voids[s]:
                bad.append(f"seat {s} holds {c} but is void in {get_card_suit(c)}")
        want = len(real_game.players[s - 1].hand)
        if s != observer and len(cur) != want:
            bad.append(f"seat {s} current size {len(cur)} != {want}")

    # Called-ace placement while partner still hidden.
    cc = real_game.called_card
    if (
        real_game.partner_mode_flag == 1
        and cc
        and not real_game.alone_called
        and not real_game.partner
        and cc not in real_game.players[observer - 1].initial_hand
        and cc not in [c for cards in played_by.values() for c in cards]
    ):
        holders = [s for s in range(1, 6) if cc in (eight if s == picker else ih[s])]
        if cc in eight:
            bad.append("called card placed in picker's hand")
        if cc in bury:
            bad.append("called card buried")
        non_picker_holders = [s for s in holders if s != picker]
        if len(non_picker_holders) != 1 or non_picker_holders[0] == observer:
            bad.append(f"called card holders {holders} (want one non-picker defender)")
        target = f"{cc} UNDER" if real_game.is_called_under else cc
        if target not in get_callable_cards(eight):
            bad.append("picker's 8 cannot legally justify the call")
    return bad


def collect(agent, max_games, target, trick, seed):
    rng_seed = seed
    states = []
    scanned = 0
    g = 0
    while len(states) < target and g < max_games:
        mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        node, forced_public = drive_record(game, agent, {trick})
        g += 1
        if node is None:
            continue
        scanned += 1
        states.append(
            {
                "game": copy.deepcopy(game),
                "mem": snapshot_memory(agent),
                "forced_public": forced_public,
                "node": node,
            }
        )
    return states, scanned, g


def evaluate(states, agent, K, r_inner, r_oracle, tau, trick, seed, legality_only):
    rng = random.Random(seed + 1)
    _FAIL.clear()
    rows = []
    legal_checks = 0
    legal_fail = 0
    replay_attempts = 0
    replay_fail = 0
    for i, st in enumerate(states):
        game = st["game"]
        obs = st["node"]["observer"]
        tc, fc = st["node"]["trump_card"], st["node"]["fail_card"]

        # Legality of raw redeals (independent of replay).
        for _ in range(8):
            try:
                d = game.sample_determinization(obs, rng)
            except RuntimeError:
                legal_fail += 1
                legal_checks += 1
                continue
            bad = check_legality(game, d, obs)
            legal_checks += 1
            if bad:
                legal_fail += 1
                if legal_fail <= 5:
                    print(f"  LEGALITY VIOLATION (state {i}): {bad[:3]}", flush=True)
        if legality_only:
            continue

        # Oracle Q on the true deal.
        oq_t = rollout_score(game, st["mem"], agent, tc, obs, r_oracle)
        oq_f = rollout_score(game, st["mem"], agent, fc, obs, r_oracle)
        oracle_d = oq_t - oq_f

        # Build K legal determinized worlds; self-normalized importance weighting
        # by the BIDDING likelihood (pick + passes + call) under the policy. No
        # rejection step (it zero-skips when redealt pickers look weak); plays
        # are honoured as hard void constraints, not soft weights.
        qt_l, qf_l, lw = [], [], []
        attempts = 0
        cap = K * 80
        while len(qt_l) < K and attempts < cap:
            attempts += 1
            replay_attempts += 1
            try:
                d = game.sample_determinization(obs, rng)
            except RuntimeError:
                continue
            world, lwf, lpick = build_world(
                game, d, st["forced_public"], obs, trick, agent
            )
            if world is None:
                replay_fail += 1
                continue
            mem = snapshot_memory(agent)
            qt_l.append(rollout_score(world, mem, agent, tc, obs, r_inner))
            qf_l.append(rollout_score(world, mem, agent, fc, obs, r_inner))
            lw.append(lwf)
        if len(qt_l) < max(5, K // 4):
            print(
                f"  [{i + 1}/{len(states)}] only {len(qt_l)} legal worlds; skipping",
                flush=True,
            )
            continue
        qt_a, qf_a, lw = np.array(qt_l), np.array(qf_l), np.array(lw)
        w = np.exp(lw - lw.max())
        w = w / w.sum()
        wt_qt, wt_qf = float((w * qt_a).sum()), float((w * qf_a).sum())
        det_d = wt_qt - wt_qf
        uni_d = float(qt_a.mean() - qf_a.mean())
        ess = float(1.0 / np.sum(w**2))
        acc_rate = len(qt_l) / max(attempts, 1)

        logits = np.array([wt_qt, wt_qf]) / tau
        logits -= logits.max()
        p_teach_trump = float(np.exp(logits[0]) / np.exp(logits).sum())
        dp = p_teach_trump - st["node"]["p_net_cond_trump"]

        rows.append(
            {
                "oracle_d": oracle_d,
                "det_d": det_d,
                "uni_d": uni_d,
                "ess": ess,
                "acc_rate": acc_rate,
                "dp": dp,
                "used": len(qt_l),
                "n_trump": st["node"]["n_trump"],
                "trump_card": tc,
                "fail_card": fc,
                "hand": st["node"]["hand"],
            }
        )
        print(
            f"  [{i + 1}/{len(states)}] post={len(qt_l):3d} ESS={ess:4.1f} acc={acc_rate:.2f}  "
            f"oracleD={oracle_d:+.3f} detD(corr)={det_d:+.3f} detD(uni)={uni_d:+.3f}  "
            f"dp={dp:+.2f}",
            flush=True,
        )
    return rows, (legal_checks, legal_fail, replay_attempts, replay_fail)


def summarize(rows, diag, tau, trick):
    legal_checks, legal_fail, replay_attempts, replay_fail = diag
    print("\n" + "=" * 72)
    print(f"STAGE A  (trick {trick} defender lead states, n={len(rows)}, tau={tau})")
    print("=" * 72)
    print(
        f"  Legality: {legal_checks - legal_fail}/{legal_checks} sampled redeals legal "
        f"({legal_fail} violations)"
    )
    if replay_attempts:
        print(
            f"  Forced-replay reproduced node: "
            f"{replay_attempts - replay_fail}/{replay_attempts} "
            f"({replay_fail} replay/desync failures)"
        )
        if _FAIL:
            print(f"  replay-failure reasons: {dict(_FAIL)}")
    if not rows:
        print("  no EV rows evaluated")
        return

    def se(x):
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    det_d = np.array([r["det_d"] for r in rows])
    uni_d = np.array([r["uni_d"] for r in rows])
    oracle_d = np.array([r["oracle_d"] for r in rows])
    ess = np.array([r["ess"] for r in rows])
    acc = np.array([r["acc_rate"] for r in rows])
    dp = np.array([r["dp"] for r in rows])

    md, mo, mu = float(det_d.mean()), float(oracle_d.mean()), float(uni_d.mean())
    sd, so = se(det_d), se(oracle_d)
    comb = float(np.sqrt(sd**2 + so**2))
    if len(rows) > 1 and det_d.std() > 0 and oracle_d.std() > 0:
        corr = float(np.corrcoef(det_d, oracle_d)[0, 1])
    else:
        corr = float("nan")

    print("\n-- Calibration: determinized EV gap vs true-deal oracle (trump - fail) --")
    print(
        f"  mean UNIFORM   detD = {mu:+.3f}  (SE {se(uni_d):.3f})  [ignores bidding+plays]"
    )
    print(f"  mean WEIGHTED  detD = {md:+.3f}  (SE {sd:.3f})  [inference-corrected]")
    print(f"  mean oracle    detD = {mo:+.3f}  (SE {so:.3f})  [true-deal ground truth]")
    print(
        f"  |weighted - oracle| = {abs(md - mo):.3f}  vs 2*combined SE = {2 * comb:.3f}"
    )
    print(f"  per-state Pearson r(weightedD, oracleD) = {corr:+.3f}")
    print(
        f"  inference ESS: mean={ess.mean():.1f}/{int(np.mean([r['used'] for r in rows]))} "
        f"(min={ess.min():.1f})   legal-world rate: mean={acc.mean():.2f} (min={acc.min():.2f})"
    )
    sign_agree = float(np.mean(np.sign(det_d) == np.sign(oracle_d)))
    print(f"  sign(detD)==sign(oracleD) on {sign_agree * 100:.0f}% of states")

    agree = abs(md - mo) <= 2 * comb
    print("\n-- Verdict --")
    print(f"  Legality: {'PASS' if legal_fail == 0 else 'FAIL'}")
    print(f"  Calibration (weighted tracks oracle): {'PASS' if agree else 'FAIL'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("--tricks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-games", type=int, default=30000)
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("-K", "--determinizations", type=int, default=40)
    ap.add_argument("--rollouts-inner", type=int, default=1)
    ap.add_argument("--rollouts-oracle", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--legality-only",
        action="store_true",
        help="Only run the legality assertions (fast, no rollouts).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.model} (device={DEV}) ...")
    agent = load_agent(args.model)

    for trick in args.tricks:
        print(f"\n##### TRICK {trick} #####")
        print(
            f"Scanning up to {args.max_games} games for trick-{trick} defender leads ..."
        )
        states, scanned, games = collect(
            agent, args.max_games, args.states, trick, args.seed
        )
        print(f"  collected {len(states)} states (scanned {games} games).")
        rows, diag = evaluate(
            states,
            agent,
            args.determinizations,
            args.rollouts_inner,
            args.rollouts_oracle,
            args.tau,
            trick,
            args.seed,
            args.legality_only,
        )
        summarize(rows, diag, args.tau, trick)


if __name__ == "__main__":
    main()
