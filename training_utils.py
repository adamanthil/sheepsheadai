#!/usr/bin/env python3
"""
Training utilities shared across training scripts.
"""

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
    get_card_suit,
    get_trick_points,
)


LEASTER_FINAL_REWARD_BONUS = 0.08
TRICK_POINT_RATIO = 360.0

# Episode-return scale shared across trainers: PPO trains the value head on
# final_score / RETURN_SCALE (~[-1, 1]), and ismcts.py bootstraps terminal
# scores by the same factor. Single source of truth (was duplicated inline).
RETURN_SCALE = 12.0


def estimate_hand_strength_score(cards: List[str]) -> int:
    """Return a simple strength score for a hand based on trump density.

    Scoring heuristic:
      - +3 per Queen
      - +2 per Jack
      - +1 per other trump
    """
    score = 0
    for card in cards:
        if card in TRUMP:
            if card.startswith("Q"):
                score += 3
            elif card.startswith("J"):
                score += 2
            else:
                score += 1
    return score


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


def compute_known_points_rel(player):
    """Return known point totals (0–120) for all seats from the acting player's perspective.

    Known points include:
      - Public trick points in game.points_taken (including blind/UNDER handling).
      - Bury card points, but only for the picker in non-leaster games once the bury is fixed.

    The result is a length-5 list ordered by relative seat:
      1 = self, 2 = left-hand opponent, ..., 5 = right-hand opponent.
    Values are raw point totals (0–120).
    """
    game = player.game
    base_points = list(getattr(game, "points_taken", [0, 0, 0, 0, 0]))

    # In standard games, the picker uniquely knows the bury card identities and their points.
    # We expose those points only to the picker, and only once the bury is fixed.
    if not game.is_leaster:
        if player.is_picker:
            base_points[player.position - 1] += get_trick_points(game.bury)

    rel_points = []
    for rel in range(1, 6):
        abs_seat = ((player.position + rel - 2) % 5) + 1
        rel_points.append(base_points[abs_seat - 1])
    return rel_points


def compute_seen_trump_mask(player) -> List[int]:
    """Return a length-len(TRUMP) 0/1 mask where 1 indicates the trump is seen/known.

    "Seen" is from the player's perspective:
      - Cards currently in hand
      - Cards already played to tricks (public history)
      - Blind / bury / under cards known only to the picker
    """

    def mark_seen(card: str, seen: set[str]) -> None:
        if card in TRUMP:
            seen.add(card)

    seen_trumps: set[str] = set()

    for card in player.hand:
        mark_seen(card, seen_trumps)

    if player.is_picker:
        for card in player.blind:
            mark_seen(card, seen_trumps)
        for card in player.bury:
            mark_seen(card, seen_trumps)
        if player.game.under_card:
            mark_seen(player.game.under_card, seen_trumps)

    for trick in player.game.history:
        for card in trick:
            if card:
                mark_seen(card, seen_trumps)

    return [1 if c in seen_trumps else 0 for c in TRUMP]


def compute_any_unseen_trump_higher_than_hand(player) -> int:
    """Return 1 if there exists an *unseen* trump higher than the player's best trump in hand.

    Interprets "highest card in my hand" as "highest trump in my hand" since trump-tracking
    is the intended subtask. If the player has no trump in hand, this reduces to:
      - 1 iff there exists any unseen trump.
    """
    seen_mask = compute_seen_trump_mask(player)

    best_trump_idx_in_hand = len(TRUMP)
    for card in player.hand:
        if card in TRUMP:
            best_trump_idx_in_hand = min(best_trump_idx_in_hand, TRUMP.index(card))

    highest_unseen_idx = len(TRUMP)
    for idx, seen in enumerate(seen_mask):
        if not seen:
            highest_unseen_idx = idx
            break

    return 1 if highest_unseen_idx < best_trump_idx_in_hand else 0


def calculate_trick_reward(trick_points: int) -> float:
    """Intermediate reward for trick points."""
    return trick_points / TRICK_POINT_RATIO


def is_same_team_as_winner(player, winner_pos: int, game) -> bool:
    if game.is_leaster:
        return False
    player_picker_team = (
        player.is_picker or player.is_partner or player.is_secret_partner
    )
    winner_player = game.players[winner_pos - 1]
    winner_picker_team = (
        winner_player.is_picker
        or winner_player.is_partner
        or winner_player.is_secret_partner
    )
    return player_picker_team == winner_picker_team


def apply_trick_rewards(
    trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float, game
) -> None:
    for transition in trick_transitions:
        player = transition["player"]
        reward_multiplier = (
            1.0 if is_same_team_as_winner(player, trick_winner_pos, game) else -1.0
        )
        transition["intermediate_reward"] += trick_reward * reward_multiplier


def apply_leaster_trick_rewards(
    trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float
) -> None:
    for transition in trick_transitions:
        player = transition["player"]
        if player.position == trick_winner_pos:
            transition["intermediate_reward"] -= trick_reward


def update_intermediate_rewards_for_action(
    game,
    player,
    action,
    transition,
    current_trick_transitions,
    pick_weight=1.0,  # Multiplier for pick head shaping
    partner_weight=1.0,  # Multiplier for partner head shaping
    bury_weight=1.0,  # Multiplier for bury head shaping
    play_weight=1.0,  # Multiplier for play head shaping
):
    """Apply shared intermediate reward shaping and trick tracking.
    Uses game engine state to detect leads and trick phase; no counter needed.
    """
    action_name = ACTIONS[action - 1]

    # Hand-conditioned PICK/PASS shaping (small human-like nudges)
    if action_name in ("PICK", "PASS"):
        score = estimate_hand_strength_score(player.hand)
        if score <= 4:
            pick_bonus, pass_bonus = -0.1, +0.1
        elif score <= 6:
            pick_bonus, pass_bonus = 0, 0
        elif score <= 7:
            pick_bonus, pass_bonus = +0.02, -0.02
        elif score >= 8:
            pick_bonus, pass_bonus = +0.15, -0.15
        transition["intermediate_reward"] += (
            pick_bonus if action_name == "PICK" else pass_bonus
        )
        transition["intermediate_reward"] *= pick_weight

    # ALONE shaping: discourage going alone with weak hands
    elif action_name == "ALONE":
        score = estimate_hand_strength_score(player.hand)
        if score <= 8:
            transition["intermediate_reward"] += -0.1
        transition["intermediate_reward"] *= partner_weight

    # Bury penalty: discourage burying trump if not required
    elif "BURY" in action_name:
        card = action_name[5:]
        # Derive allowed bury actions from this step's valid_actions
        valid_actions = transition.get("valid_actions", set())
        allowed_bury_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
        has_allowed_fail_bury = any(get_card_suit(c) != "T" for c in allowed_bury_cards)
        has_allowed_trump_bury = any(
            get_card_suit(c) == "T" for c in allowed_bury_cards
        )

        if card in TRUMP and has_allowed_fail_bury:
            transition["intermediate_reward"] += -0.1
        elif card not in TRUMP and has_allowed_trump_bury:
            # Small preference when both options exist
            transition["intermediate_reward"] += 0.02
        elif not has_allowed_fail_bury and card.startswith("Q"):
            # Even if we have to bury trump, we should not bury queens.
            transition["intermediate_reward"] += -0.1
        elif not has_allowed_fail_bury and card.startswith("J"):
            # Even if we have to bury trump, unlikely we should bury jacks.
            transition["intermediate_reward"] += -0.02
        transition["intermediate_reward"] *= bury_weight

    elif "PLAY" in action_name:
        is_lead = (game.cards_played == 0) and (game.leader == player.position)
        if is_lead:
            valid_actions = transition.get("valid_actions", set())
            allowed_play_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
            has_allowed_fail_play = any(
                get_card_suit(c) != "T" for c in allowed_play_cards
            )

            if game.called_card:
                has_called_suit = any(
                    get_card_suit(c) == get_card_suit(game.called_card)
                    for c in allowed_play_cards
                )
            else:
                has_called_suit = False

            card = action_name[5:]
            if (
                not game.is_leaster
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and has_allowed_fail_play
                and card in TRUMP
            ):
                # Discourage defenders from leading trump
                transition["intermediate_reward"] += -0.06
            elif (
                game.called_card
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and not game.was_called_suit_played
                and game.called_suit == get_card_suit(card)
            ):
                # Encourage defenders to lead called suit (early)
                transition["intermediate_reward"] += 0.1 - (0.02 * game.current_trick)
            elif (
                game.called_card
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and not game.was_called_suit_played
                and has_called_suit
                and game.called_suit != get_card_suit(card)
            ):
                # Discourage defenders from not leading called suit if they can
                transition["intermediate_reward"] += -0.01
            elif (
                not game.is_leaster
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and card not in TRUMP
            ):
                # Nudge defenders toward leading fail
                transition["intermediate_reward"] += 0.03
            elif (
                not game.is_leaster
                and (player.is_partner or player.is_secret_partner)
                and card in TRUMP
            ):
                # Encourage partners to lead trump
                transition["intermediate_reward"] += 0.08
            elif not game.is_leaster and player.is_picker and card in TRUMP:
                # Nudge picker toward leading trump
                transition["intermediate_reward"] += 0.03
        transition["intermediate_reward"] *= play_weight

        current_trick_transitions.append(transition)

    transition["head_shaping_reward"] = transition["intermediate_reward"]


def handle_trick_completion(game, current_trick_transitions):
    """If a trick has completed, apply trick-based rewards and reset tracking.

    Returns True if the trick just completed, else False.
    """
    if game.was_trick_just_completed:
        trick_points = game.trick_points[game.current_trick - 1]
        trick_winner = game.trick_winners[game.current_trick - 1]
        trick_reward = calculate_trick_reward(trick_points)

        if not game.is_leaster:
            apply_trick_rewards(
                current_trick_transitions, trick_winner, trick_reward, game
            )
        else:
            apply_leaster_trick_rewards(
                current_trick_transitions, trick_winner, trick_reward
            )

        # Reset tracking for next trick
        current_trick_transitions.clear()
        return True

    return False


def process_episode_rewards(episode_transitions, final_scores, is_leaster):
    """Compute per-action rewards for a single player's episode.

    The input must be a chronological list of action transitions for one player only.
    The final action in the list is considered the terminal step for assigning
    the episode-level final reward.
    """
    last_index = len(episode_transitions) - 1
    for i, transition in enumerate(episode_transitions):
        player = transition["player"]
        player_pos = player.position
        final_score = final_scores[player_pos - 1]
        is_episode_done = i == last_index

        if is_leaster:
            # Increase final reward for leasters to compensate for negative trick rewards.
            # Agent should dislike playing leasters most of the time (similar to human behavior)
            # but without this the bias is a bit too strong.
            leaster_reward = final_score / RETURN_SCALE + LEASTER_FINAL_REWARD_BONUS
            final_reward = leaster_reward if is_episode_done else 0.0
        else:
            final_reward = (final_score / RETURN_SCALE) if is_episode_done else 0.0

        total_reward = final_reward + transition["intermediate_reward"]
        yield {
            "transition": transition,
            "reward": total_reward,
        }


def process_terminal_rewards(episode_transitions, final_scores, is_leaster=False):
    """Terminal-only return for the ExIt/ISMCTS regime (mirrors the
    ``process_episode_rewards`` interface so the PFSP runtime can swap them).

    The full episode return ``final_score / RETURN_SCALE`` is placed on the
    player's LAST action; every other step gets 0 and is bridged by GAE/critic.
    No per-trick reward and no leaster bonus — ``get_score()`` already scores
    leasters correctly, so pass->leaster EV is win-likelihood driven with zero
    hand-tuning (``is_leaster`` is accepted for interface parity and ignored).
    See notebooks/ISMCTS_Teacher_Refactor_Plan.md §2.
    """
    last_index = len(episode_transitions) - 1
    for i, transition in enumerate(episode_transitions):
        player = transition["player"]
        final_score = final_scores[player.position - 1]
        reward = (final_score / RETURN_SCALE) if i == last_index else 0.0
        yield {
            "transition": transition,
            "reward": reward,
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
    saved_mem = {pid: t.detach().clone() for pid, t in agent._player_memories.items()}
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
        agent._player_memories = saved_mem
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
