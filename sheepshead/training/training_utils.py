#!/usr/bin/env python3
"""
Training utilities shared across training scripts.
"""

import csv
import os
import random
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch

from sheepshead import (
    Game,
    TRUMP,
    ACTIONS,
    ACTION_IDS,
    ACTION_LOOKUP,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
)

# Reward-shaping / auxiliary-target math lives in reward_shaping.py so that
# consumers who only need that pure math (e.g. the web server's /analyze
# service) don't have to import the rest of this trainer-internals module.
# Re-exported here, unchanged, as a permanent compatibility shim: this
# module's existing 20+ importers (trainers, analysis/validation scripts,
# tests) keep working without modification.
from sheepshead.training.reward_shaping import (  # noqa: F401
    LEASTER_FINAL_REWARD_BONUS,
    TRICK_POINT_RATIO,
    RETURN_SCALE,
    estimate_hand_strength_score,
    compute_known_points_rel,
    compute_seen_trump_mask,
    compute_any_unseen_trump_higher_than_hand,
    calculate_trick_reward,
    is_same_team_as_winner,
    apply_trick_rewards,
    apply_leaster_trick_rewards,
    update_intermediate_rewards_for_action,
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
)


def set_all_seeds(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` with the same value.

    Consolidates the copy-pasted ``random.seed(s); np.random.seed(s);
    torch.manual_seed(s)`` triple that recurred across the trainers and
    analysis/validation scripts. Order matters for exact reproducibility
    with existing runs: random, then numpy, then torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_partner_selection_mode(episode: int) -> int:
    return PARTNER_BY_CALLED_ACE if (episode % 2 == 0) else PARTNER_BY_JD


def play_paired_deal(deal_seed: int, mode, seat: int, probe, field) -> float:
    """One greedy deal; ``probe`` plays the probe seat, ``field`` the rest.
    Returns the probe seat's final score. ``probe`` and ``field`` may be the
    same agent instance (recurrent memory is keyed by player_id)."""
    game = Game(partner_selection_mode=mode, seed=deal_seed)
    probe.reset_recurrent_state()
    field.reset_recurrent_state()
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                ag = probe if player.position == seat else field
                a, _, _ = ag.act(
                    player.get_state_dict(), valid, player.position, deterministic=True
                )
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for p in game.players:
                        ctrl = probe if p.position == seat else field
                        ctrl.observe(
                            p.get_last_trick_state_dict(), player_id=p.position
                        )
    return float(game.players[seat - 1].get_score())


def paired_edge(
    challenger,
    incumbent,
    field,
    n_deals: int,
    seed: int = 0,
    log_every: int = 100,
) -> Dict:
    """Paired duplicate-deal edge of ``challenger`` over ``incumbent`` against
    the same greedy ``field``: mean (challenger − incumbent) score from the
    same seat on the same deal. Returns edge/SE/win_frac/n. A fixed ``seed``
    fixes the deal set, so a series of calls (e.g. the trainer's anchored
    probe) is itself paired across calls."""
    import time as _time

    deltas, wins = [], 0.0
    t0 = _time.time()
    for d in range(n_deals):
        mode = get_partner_selection_mode(d)
        seat = (d % 5) + 1
        deal_seed = seed * 1_000_003 + d
        s_inc = play_paired_deal(deal_seed, mode, seat, incumbent, field)
        s_chal = play_paired_deal(deal_seed, mode, seat, challenger, field)
        delta = s_chal - s_inc
        deltas.append(delta)
        wins += 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
        if log_every and (d + 1) % log_every == 0:
            arr = np.array(deltas)
            print(
                f"  gate {d + 1}/{n_deals} ({_time.time() - t0:.0f}s)  "
                f"edge {arr.mean():+.3f} +/- {arr.std(ddof=1) / np.sqrt(len(arr)):.3f}",
                flush=True,
            )
    arr = np.array(deltas)
    n = len(arr)
    return {
        "edge": float(arr.mean()),
        "se": float(arr.std(ddof=1) / np.sqrt(n)),
        "win_frac": float(wins / n),
        "n_deals": n,
        "deviating_frac": float(np.mean(arr != 0.0)),
    }


def analyze_strategic_decisions(agent, num_samples=100):
    """Analyze strategic decision quality instead of random opponent evaluation.

    Shared by all trainers (moved here from the self-play trainer so the PFSP
    trainers no longer cross-import it).
    """

    # Trump leading analysis
    trump_leads = {
        "picker_team": 0,
        "picker_total": 0,
        "defender_team": 0,
        "defender_total": 0,
    }

    # Bury quality analysis
    bury_quality = {"good_burys": 0, "bad_burys": 0, "total_burys": 0}

    # Pick decision correlation with hand strength
    pick_decisions = []
    hand_strengths = []

    for episode in range(num_samples):
        game = Game(partner_selection_mode=get_partner_selection_mode(episode))
        # Ensure recurrent state is fresh for analysis episode
        agent.reset_recurrent_state()

        # Analyze pick decisions
        initial_player = game.players[0]
        hand_strength = sum(
            3 if c[0] == "Q" else 2 if c[0] == "J" else 1 if c in TRUMP else 0
            for c in initial_player.hand
        )

        sdict = initial_player.get_state_dict()
        initial_actions = initial_player.get_valid_action_ids()
        with torch.no_grad():
            action_probs, _ = agent.get_action_probs_with_logits(
                sdict, initial_actions, player_id=initial_player.position
            )

        pick_prob = action_probs[0, ACTION_IDS["PICK"] - 1].item()
        pick_decisions.append(pick_prob)
        hand_strengths.append(hand_strength)

        # Play full game to analyze trump leading and bury decisions
        while not game.is_done():
            for player in game.players:
                actions = player.get_valid_action_ids()

                if actions:
                    sdict = player.get_state_dict()
                    with torch.no_grad():
                        action_probs, _ = agent.get_action_probs_with_logits(
                            sdict, actions, player_id=player.position
                        )
                    action = (
                        torch.distributions.Categorical(action_probs).sample().item()
                        + 1
                    )
                    action_name = ACTION_LOOKUP[action]

                    # Analyze trump leading
                    if (
                        "PLAY" in action_name
                        and game.play_started
                        and game.cards_played == 0
                    ):
                        card = action_name.split()[-1]
                        is_trump_lead = card in TRUMP
                        is_picker_team = (
                            player.is_picker
                            or player.is_partner
                            or player.is_secret_partner
                        )

                        if is_picker_team:
                            trump_leads["picker_total"] += 1
                            if is_trump_lead:
                                trump_leads["picker_team"] += 1
                        else:
                            trump_leads["defender_total"] += 1
                            if is_trump_lead:
                                trump_leads["defender_team"] += 1

                    # Analyze bury decisions
                    if "BURY" in action_name:
                        card = action_name.split()[-1]
                        bury_quality["total_burys"] += 1

                        # Good bury: fail
                        if card not in TRUMP:
                            bury_quality["good_burys"] += 1
                        else:
                            bury_quality["bad_burys"] += 1

                    player.act(action)

    # Calculate metrics
    pick_hand_correlation = (
        np.corrcoef(hand_strengths, pick_decisions)[0, 1]
        if len(hand_strengths) > 1
        else 0
    )

    picker_trump_rate = (
        trump_leads["picker_team"] / max(trump_leads["picker_total"], 1) * 100
    )
    defender_trump_rate = (
        trump_leads["defender_team"] / max(trump_leads["defender_total"], 1) * 100
    )

    bury_quality_rate = (
        bury_quality["good_burys"] / max(bury_quality["total_burys"], 1) * 100
    )

    return {
        "pick_hand_correlation": pick_hand_correlation,
        "picker_trump_rate": picker_trump_rate,
        "defender_trump_rate": defender_trump_rate,
        "bury_quality_rate": bury_quality_rate,
    }


def save_training_plot(training_data, save_path="training_progress.png"):
    """Enhanced training plots with strategic metrics."""
    episodes = training_data["episodes"]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # Score progression
    ax1.plot(episodes, training_data["picker_avg"], label="Picker Avg", alpha=0.8)
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Picker Score")
    ax1.set_title("Score Progression")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pick rate and correlation
    ax2.plot(
        episodes,
        training_data["called_pick_rate"],
        color="orange",
        alpha=0.8,
        label="Called Ace Pick Rate",
    )
    ax2.plot(
        episodes,
        training_data["jd_pick_rate"],
        color="purple",
        alpha=0.8,
        label="JD Pick Rate",
    )
    ax2_twin = ax2.twinx()
    if (
        "pick_hand_correlation" in training_data
        and len(training_data["pick_hand_correlation"]) > 0
    ):
        corr = training_data["pick_hand_correlation"]
        strat_eps = training_data.get("strategic_episodes", [])
        if strat_eps:
            # Align by trimming to common length from the tail
            n = min(len(corr), len(strat_eps))
            x = strat_eps[-n:]
            y = corr[-n:]
        else:
            # Fallback: align to tail of episodes
            x = episodes[-len(corr) :]
            y = corr
        ax2_twin.plot(
            x, y, color="green", alpha=0.8, label="Hand Correlation", marker="o"
        )
        ax2_twin.set_ylabel("Hand Strength Correlation")
        ax2_twin.legend(loc="upper right")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Pick Rate (%)")
    ax2.set_title("Pick Strategy Quality")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Trump leading rates
    if (
        "picker_trump_rate" in training_data
        and len(training_data["picker_trump_rate"]) > 0
    ):
        picker_trump = training_data["picker_trump_rate"]
        defender_trump = training_data.get("defender_trump_rate", [])
        strat_eps = training_data.get("strategic_episodes", [])
        if strat_eps:
            n = min(len(picker_trump), len(defender_trump), len(strat_eps))
            x = strat_eps[-n:]
            y1 = picker_trump[-n:]
            y2 = defender_trump[-n:]
        else:
            n = min(len(picker_trump), len(defender_trump))
            x = episodes[-n:]
            y1 = picker_trump[-n:]
            y2 = defender_trump[-n:]
        ax3.plot(x, y1, color="blue", alpha=0.8, label="Picker Team", marker="o")
        ax3.plot(x, y2, color="red", alpha=0.8, label="Defender Team", marker="o")
        ax3.axhline(
            y=60, color="blue", linestyle="--", alpha=0.5, label="Picker Target"
        )
        ax3.axhline(
            y=20, color="red", linestyle="--", alpha=0.5, label="Defender Target"
        )
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Trump Lead Rate (%)")
        ax3.set_title("Trump Leading Strategy")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Bury quality
    if (
        "bury_quality_rate" in training_data
        and len(training_data["bury_quality_rate"]) > 0
    ):
        bury = training_data["bury_quality_rate"]
        strat_eps = training_data.get("strategic_episodes", [])
        if strat_eps:
            n = min(len(bury), len(strat_eps))
            x = strat_eps[-n:]
            y = bury[-n:]
        else:
            x = episodes[-len(bury) :]
            y = bury
        ax4.plot(x, y, color="purple", alpha=0.8, marker="o")
        ax4.axhline(
            y=90, color="purple", linestyle="--", alpha=0.5, label="Good Target"
        )
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Good Bury Rate (%)")
        ax4.set_title("Bury Decision Quality")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Rates: Alone call and Leaster
    # When resuming training, historical CSVs may not contain these rate series.
    # Rather than require exact length matching with `episodes`, plot whatever
    # portion is available by aligning each series to the tail of the episodes
    # timeline. This keeps the figure renderable on resume while still showing
    # new data going forward.
    alone_rate = training_data.get("alone_rate", [])
    leaster_rate = training_data.get("leaster_rate", [])
    any_rate_plotted = False
    if alone_rate:
        ex = episodes[-len(alone_rate) :]
        ax5.plot(ex, alone_rate, color="brown", alpha=0.8, label="Alone Call Rate")
        any_rate_plotted = True
    if leaster_rate:
        ex = episodes[-len(leaster_rate) :]
        ax5.plot(ex, leaster_rate, color="gray", alpha=0.8, label="Leaster Rate")
        any_rate_plotted = True
    if any_rate_plotted:
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Rate (%)")
        ax5.set_title("Alone Call and Leaster Rates")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Team Point Difference
    if "team_point_diff" in training_data and len(training_data["team_point_diff"]) > 0:
        ax6.plot(episodes, training_data["team_point_diff"], color="red", alpha=0.8)
        ax6.axhline(
            y=0, color="black", linestyle="--", alpha=0.5, label="Perfect Balance"
        )
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Team Point Difference")
        ax6.set_title("Team Point Difference (Picker - Defender)")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


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
    play_spreads = []  # legal-play logit spread (max-min) at multi-legal nodes
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
                        state = player.get_state_dict()
                        is_play = game.play_started and all(
                            ACTIONS[x - 1].startswith("PLAY ") for x in valid
                        )
                        if is_play and len(valid) >= 2:
                            # One forward yields both the greedy action and the
                            # legal-play logit spread. Do NOT also call act():
                            # that would advance recurrent memory a second time.
                            # argmax over post-mix probs == act(deterministic).
                            probs_t, logits_t = agent.get_action_probs_with_logits(
                                state, valid, player_id=player.position
                            )
                            a = int(torch.argmax(probs_t, dim=1).item()) + 1
                            lv = logits_t[0][[x - 1 for x in valid]]
                            play_spreads.append(float(lv.max() - lv.min()))
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
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for seat in game.players:
                                agent.observe(
                                    seat.get_last_trick_state_dict(),
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
        # Median legal-play logit spread (max-min). A healthy play head is well
        # separated (baseline ~6.5); a collapsed/uniform head reads ~0. This is
        # the cheap canary for the terminal-only play-head collapse.
        "play_logit_spread_med": (
            float(np.median(play_spreads)) if play_spreads else 0.0
        ),
        "play_nodes": len(play_spreads),
    }


def append_csv_row(path: str, fieldnames: List[str], row: Dict) -> None:
    """Append one row to the CSV at ``path``, writing ``fieldnames`` as a
    header first if the file doesn't exist yet.

    Both PPO trainers periodically append a row to their own
    ``anchored_eval.csv`` (and train_league_ppo has a few sibling logs with
    the same idiom): check whether the file exists, open in append mode,
    write the header once, then write the row. The two trainers' anchored-
    eval schemas differ (different columns entirely), so this only factors
    out that shared mechanism -- callers still supply their own header list
    and row dict, so each trainer's exact byte-for-byte schema is unchanged.
    Uses ``csv.DictWriter`` regardless of whether the caller's original code
    used ``csv.writer`` with a plain list: given the same fieldnames order
    and matching dict keys, the emitted bytes are identical either way.
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
