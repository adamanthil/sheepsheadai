"""The greedy health probe: deterministic play of a checkpoint against
itself, read out as behavioral health and convention-adherence columns
(the trainer's greedy_health.csv).
"""

import random
from typing import Dict

import numpy as np
import torch

from sheepshead.agent.observation import (
    last_trick_observation_for,
    observation_for,
)
from sheepshead.game import (
    ACTIONS,
    TRUMP,
    Game,
    get_partner_selection_mode,
)
from sheepshead.rewards import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
)


def greedy_health_probe(agent, n_games: int = 200, seed: int = 0) -> Dict:
    """Deterministic-greedy self-play health metrics for the collapse guard.

    Stochastic training-time rates can mask a bidding collapse: a flattened
    policy still *samples* ~30% PICK while its argmax is PASS (the run-2
    failure mode went unseen for 586k episodes this way). This probe plays
    ``n_games`` of greedy self-play with ``agent`` in all five seats and
    reports the argmax-policy rates that collapse actually shows up in:
    PICK rate, ALONE rate (of partner decisions), leaster rate, and the
    trick-0 defender trump-lead rate (the original partial-obs leak).

    Side-effect free with respect to training: the agent's recurrent memories
    and the global ``random`` state are snapshotted and restored, and ``act``
    does not write into the training buffer.
    """
    rng_state = random.getstate()
    saved_mem = agent.snapshot_player_memories()
    random.seed(seed)
    picks = passes = 0
    alone = partner_decisions = 0
    leasters = 0
    t0_def_leads = 0  # trick-0 defender leads holding both trump and fail
    t0_def_trump = 0  # ...that led trump (greedy)
    ptn_leads = 0  # secret-partner leads, tricks 0-2, both classes legal
    ptn_trump = 0  # ...that led trump (the partner convention)
    cs_leads = 0  # defender leads, called-ace mode, called suit not yet led
    cs_adherent = 0  # ...that led a called-suit fail (convention C2)
    play_spreads = []  # legal-play logit spread (max-min) at multi-legal nodes
    # Early-trick defender leads (t0-t2): the search-teacher target nodes
    # (Search_Teacher_Design §12.7 relaunch monitoring spec). Composition of
    # the greedy FAIL lead by point class (policy bias = fat A/10; search
    # direction = no-point 7/8/9), plus top1-min logit gap at these nodes —
    # the conviction floor that separates benign top-compression from
    # erosion (§12.6 addendum 3).
    deflead_fail = 0  # greedy fail leads at t0-2 defender lead nodes
    deflead_fat = 0  # ...that led A/10
    deflead_nopoint = 0  # ...that led 7/8/9
    deflead_top1min = []  # top1 - min legal-play logit at those nodes
    # Seen-trump recall (Training_Program_Redesign §7.1, informational): at
    # every scored play node of every seat, the aux head's seen/known-trump
    # mask against the truth (compute_seen_trump_mask) — accuracy over all
    # 14 trumps, recall over the trumps the seat must REMEMBER
    # (seen_trump_recall_cards: seen, but neither in hand nor on the table,
    # which the observation never shows), the same recall by trick index
    # (forgetting reads as decay across tricks), and the false-seen rate on
    # trumps the seat has not seen (a head that says "seen" for everything
    # scores 100% recall and is caught here).
    seen_nodes = 0
    seen_correct = seen_total = 0
    recall_correct = recall_total = 0
    recall_by_trick = [[0, 0] for _ in range(6)]  # [correct, total] per trick
    # The must-remember trumps split by source: played in an earlier trick
    # (history; empty at trick 0 by construction) vs the picker's own bury
    # (or under-card), which only ~7% of deals contain a trump in, so its
    # entries are rare and its recall is a separate, harder read.
    recall_hist_by_trick = [[0, 0] for _ in range(6)]
    recall_bury_correct = recall_bury_total = 0
    false_seen = unseen_total = 0
    false_seen_by_trick = [[0, 0] for _ in range(6)]  # [false-seen, unseen]
    # The other three deterministic aux heads at the same nodes (§5.4):
    # secret partner (self label: holds the called card), known points per
    # relative seat (exact after rounding, and MAE), unseen-trump-higher.
    secret_correct = 0
    points_abs_err = 0.0
    points_exact = points_total = 0
    unseen_higher_correct = 0
    try:
        for g in range(n_games):
            game = Game(partner_selection_mode=get_partner_selection_mode(g))
            agent.reset_recurrent_state()
            while not game.is_done():
                for player in game.players:
                    valid = player.get_valid_action_ids()
                    while valid:
                        is_t0_def_lead = (
                            game.play_started
                            and not game.is_leaster
                            and game.cards_played == 0
                            and game.current_trick == 0
                            and game.leader == player.position
                            and not (
                                player.is_picker
                                or player.is_partner
                                or player.is_secret_partner
                            )
                            and any(c in TRUMP for c in player.hand)
                            and any(c not in TRUMP for c in player.hand)
                        )
                        is_partner_lead = (
                            game.play_started
                            and not game.is_leaster
                            and game.cards_played == 0
                            and game.current_trick <= 2
                            and game.leader == player.position
                            and player.is_secret_partner
                            and not player.is_picker
                            and any(c in TRUMP for c in player.hand)
                            and any(c not in TRUMP for c in player.hand)
                        )
                        # Convention C2 (called_suit_probe.py is the calibrated
                        # instrument; this is the cheap in-training canary, no
                        # under-call split): a true defender leading in
                        # called-ace mode before the called suit has been led,
                        # holding a called-suit fail AND an alternative.
                        called = game.called_card
                        is_called_suit_lead = (
                            game.play_started
                            and not game.is_leaster
                            and game.cards_played == 0
                            and game.leader == player.position
                            and bool(called)
                            and not game.was_called_suit_played
                            and not (
                                player.is_picker
                                or player.is_partner
                                or player.is_secret_partner
                            )
                            and any(
                                c not in TRUMP and c[-1] == called[-1]
                                for c in player.hand
                            )
                            and any(
                                c in TRUMP or c[-1] != called[-1] for c in player.hand
                            )
                        )
                        state = observation_for(player, agent)
                        is_play = game.play_started and all(
                            ACTIONS[x - 1].startswith("PLAY ") for x in valid
                        )
                        is_deflead = (
                            game.play_started
                            and not game.is_leaster
                            and game.cards_played == 0
                            and game.current_trick <= 2
                            and game.leader == player.position
                            and not (
                                player.is_picker
                                or player.is_partner
                                or player.is_secret_partner
                            )
                        )
                        if is_play:
                            # One forward yields the greedy action, the aux
                            # heads' read and the legal-play logit spread. Do
                            # NOT also call act(): that would advance recurrent
                            # memory a second time (the same forward + memory
                            # update as act; argmax over post-mix probs ==
                            # act(deterministic)). Single-legal nodes (the
                            # last trick) are scored for memory, not spread.
                            probs_t, logits_t, enc_out = (
                                agent.get_action_probs_logits_and_encoder_out(
                                    state, valid, player_id=player.position
                                )
                            )
                            a = int(torch.argmax(probs_t, dim=1).item()) + 1
                            aux = agent.aux_probe(enc_out)
                            if aux is not None:
                                truth = compute_seen_trump_mask(player)
                                pred = [
                                    int(x > 0.5) for x in aux["seen_trump"].tolist()
                                ]
                                secret_correct += int(
                                    (float(aux["secret"]) > 0.5)
                                    == bool(player.is_secret_partner)
                                )
                                unseen_higher_correct += int(
                                    (float(aux["unseen_higher"]) > 0.5)
                                    == bool(
                                        compute_any_unseen_trump_higher_than_hand(
                                            player
                                        )
                                    )
                                )
                                for p_pred, p_true in zip(
                                    aux["points"].tolist(),
                                    compute_known_points_rel(player),
                                ):
                                    err = abs(float(p_pred) - float(p_true))
                                    points_abs_err += err
                                    points_exact += int(err < 0.5)
                                    points_total += 1
                                must_recall = seen_trump_recall_cards(player)
                                trick_bin = min(int(game.current_trick), 5)
                                played = {
                                    c
                                    for tr in game.history[: int(game.current_trick)]
                                    for c in tr
                                }
                                seen_nodes += 1
                                for i, card in enumerate(TRUMP):
                                    ok = int(pred[i] == int(truth[i]))
                                    seen_total += 1
                                    seen_correct += ok
                                    if card in must_recall:
                                        recall_total += 1
                                        recall_correct += ok
                                        recall_by_trick[trick_bin][0] += ok
                                        recall_by_trick[trick_bin][1] += 1
                                        if card in played:
                                            recall_hist_by_trick[trick_bin][0] += ok
                                            recall_hist_by_trick[trick_bin][1] += 1
                                        else:
                                            recall_bury_correct += ok
                                            recall_bury_total += 1
                                    elif not truth[i]:
                                        unseen_total += 1
                                        false_seen += pred[i]
                                        false_seen_by_trick[trick_bin][0] += pred[i]
                                        false_seen_by_trick[trick_bin][1] += 1
                            if len(valid) >= 2:
                                lv = logits_t[0][[x - 1 for x in valid]]
                                play_spreads.append(float(lv.max() - lv.min()))
                                if is_deflead:
                                    deflead_top1min.append(float(lv.max() - lv.min()))
                                    gcard = ACTIONS[a - 1][5:]
                                    if gcard not in TRUMP:
                                        deflead_fail += 1
                                        rank = gcard[:-1]
                                        if rank in ("A", "10"):
                                            deflead_fat += 1
                                        elif rank in ("7", "8", "9"):
                                            deflead_nopoint += 1
                        else:
                            a, _, _ = agent.act(
                                state,
                                valid,
                                player.position,
                                deterministic=True,
                            )
                        name = ACTIONS[a - 1]
                        if name == "PICK":
                            picks += 1
                        elif name == "PASS":
                            passes += 1
                        elif name == "ALONE":
                            alone += 1
                            partner_decisions += 1
                        elif name == "JD PARTNER" or name.startswith("CALL "):
                            partner_decisions += 1
                        if is_t0_def_lead:
                            t0_def_leads += 1
                            if name.startswith("PLAY ") and name[5:] in TRUMP:
                                t0_def_trump += 1
                        if is_partner_lead:
                            ptn_leads += 1
                            if name.startswith("PLAY ") and name[5:] in TRUMP:
                                ptn_trump += 1
                        if is_called_suit_lead:
                            cs_leads += 1
                            card = name[5:] if name.startswith("PLAY ") else ""
                            if (
                                card
                                and called
                                and card not in TRUMP
                                and card[-1] == called[-1]
                            ):
                                cs_adherent += 1
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for seat in game.players:
                                agent.observe(
                                    last_trick_observation_for(seat, agent),
                                    player_id=seat.position,
                                )
            if game.is_leaster:
                leasters += 1
    finally:
        agent.restore_player_memories(saved_mem)
        random.setstate(rng_state)
    return {
        "games": n_games,
        "pick_rate": 100.0 * picks / max(picks + passes, 1),
        "alone_rate": 100.0 * alone / max(partner_decisions, 1),
        "leaster_rate": 100.0 * leasters / max(n_games, 1),
        "t0_trump_lead_rate": 100.0 * t0_def_trump / max(t0_def_leads, 1),
        "t0_def_leads": t0_def_leads,
        # The partner convention (B2's other half; its absence from this
        # probe hid the 2026-07 batch-arm convention collapse for 850k eps).
        "partner_trump_lead_rate": 100.0 * ptn_trump / max(ptn_leads, 1),
        "partner_leads": ptn_leads,
        # Convention C2: defender leads the called suit while it hasn't been
        # led (called-ace mode). Convention says 100% at eligible nodes.
        "called_suit_lead_rate": 100.0 * cs_adherent / max(cs_leads, 1),
        "called_leads": cs_leads,
        # Median legal-play logit spread (max-min). A healthy play head is well
        # separated (baseline ~6.5); a collapsed/uniform head reads ~0. This is
        # the cheap canary for the terminal-only play-head collapse.
        # Search-teacher target-node monitors (§12.7): greedy fail-lead point
        # composition at t0-2 defender leads (policy bias = fat; taught
        # direction = no-point) and the top1-min conviction floor there
        # (8M baseline ~9.7/9.5/6.9 nats t0/t1/t2; v7 damage read 5.2/3.9/2.8).
        "deflead_fail_leads": deflead_fail,
        "deflead_fat_lead_rate": 100.0 * deflead_fat / max(deflead_fail, 1),
        "deflead_nopoint_lead_rate": 100.0 * deflead_nopoint / max(deflead_fail, 1),
        "deflead_top1min_med": (
            float(np.median(deflead_top1min)) if deflead_top1min else 0.0
        ),
        "play_logit_spread_med": (
            float(np.median(play_spreads)) if play_spreads else 0.0
        ),
        "play_nodes": len(play_spreads),
        # Seen-trump memory at the scored play nodes of every seat
        # (informational, §7.1). `acc`: % of the 14 seen/known bits the aux
        # head gets right. `recall`: % of the trumps the seat must remember
        # (seen_trump_recall_cards) that it still reports seen; `recall_by_
        # trick` the same per trick index 0-5 (trick 0 is the picker's
        # bury/discarded-blind trumps alone — nothing has been played yet).
        # `false_seen`: % of the trumps the seat has NOT seen that it reports
        # seen, overall and per trick (read beside recall by trick: a head
        # that says "seen" for more and more as the deal goes on scores
        # rising recall AND rising false-seen; memory scores rising recall
        # at flat false-seen). `nodes` = play nodes scored (every play node
        # of every seat,
        # single-legal last-trick nodes included, when the critic has aux
        # heads, else 0); `recall_cards` = must-remember trump-node pairs
        # scored. All 0 with no nodes or no aux heads.
        "seen_trump_acc": 100.0 * seen_correct / max(seen_total, 1),
        "seen_trump_recall": 100.0 * recall_correct / max(recall_total, 1),
        "seen_trump_recall_by_trick": [
            100.0 * c / max(n, 1) for c, n in recall_by_trick
        ],
        # The same recall split by source: played-history entries per trick
        # (t0 has none, so the series starts at t1) and the picker's bury
        # entries pooled over all tricks (a 200-game probe holds ~12).
        "seen_trump_recall_hist_by_trick": [
            100.0 * c / max(n, 1) for c, n in recall_hist_by_trick[1:]
        ],
        "seen_trump_recall_bury": (
            100.0 * recall_bury_correct / max(recall_bury_total, 1)
        ),
        "seen_trump_bury_cards": recall_bury_total,
        "seen_trump_false_seen": 100.0 * false_seen / max(unseen_total, 1),
        "seen_trump_false_seen_by_trick": [
            100.0 * c / max(n, 1) for c, n in false_seen_by_trick
        ],
        "seen_trump_nodes": seen_nodes,
        "seen_trump_recall_cards": recall_total,
        # The other deterministic heads at the same nodes: % of nodes where
        # the secret-partner read matches the self label; % of per-seat
        # known-points reads exact after rounding (and their MAE, points);
        # % of nodes where the unseen-trump-higher read matches the truth.
        "aux_secret_acc": 100.0 * secret_correct / max(seen_nodes, 1),
        "aux_points_exact": 100.0 * points_exact / max(points_total, 1),
        "aux_points_mae": points_abs_err / max(points_total, 1),
        "aux_unseen_higher_acc": 100.0 * unseen_higher_correct / max(seen_nodes, 1),
    }


def seen_trump_recall_cards(player) -> set[str]:
    """The trumps ``player`` has seen (compute_seen_trump_mask's truth) that
    its observation does not show right now: neither in hand nor on the
    table in the current trick. Every one of these the network can only
    know from its own memory — trumps played in earlier tricks for every
    seat, plus the bury and the discarded blind for the picker
    (Training_Program_Redesign §3.2: play history is never re-shown)."""
    game = player.game
    idx = int(game.current_trick)
    on_table = set(game.history[idx]) if idx < len(game.history) else set()
    truth = compute_seen_trump_mask(player)
    return {
        card
        for card, seen in zip(TRUMP, truth)
        if seen and card not in player.hand and card not in on_table
    }
