# Improving point tracking in sheepshead AI

> **Note (2026-07-11):** file paths in this notebook predate the 2026-07 repo reorganization (core modules now live in `sheepshead/`, the hosted product under `app/`). Kept as-is for the historical record.
_Exported on 12/4/2025 at 10:11:26 CST from Cursor (2.1.47)_

---

**User**

I am training a neural network using PPO to play the trick taking card game sheepshead, hoping to achieve an expert human level of play.

In Sheepshead, the winners are the team that takes more than 60 points (there are a total of 120 points in the deck) with ties going to the "defending" team (non picking team). The exception to this is "leasters" where everyone passes and then the person who wins is the one who takes the fewest points but at least one trick.



Training overall is going well and the agent is exhibiting strong play. But it is not expert level yet. Based on the auxiliary critic heads that predict win probability, I don't think it is currently counting points accurately. An expert human player would count how many points each player has already taken to help inform the decision about which tricks to take. I am considering adding a point prediction head to the critic to help the shared trunk learn to attend to this information and track points.



1. Do you think this would yield beneficial results in actor behavior?

2. Given the domain, what is the most useful way to expose this prediction? We could output one prediction for points taken by the agent's team and another for the points taken by the opponents team (but what to do in case of leasters?) or perhaps a different output structure would work better. I'm open to suggestions well backed by the literature.

---

**Cursor**

I'll look up relevant auxiliary-task literature and then suggest concrete, domain-tailored prediction targets that should most help your PPO trunk and policy, including how to handle leasters.

## 1. Will a point‑prediction head actually help the policy?

**Yes, it is very likely to help, provided you design it carefully and weight it moderately.**

This is exactly the kind of auxiliary task that has been shown to improve actor performance in deep RL:

- **Auxiliary tasks → better representations.** Work like Jaderberg et al., *UNREAL: Reinforcement Learning with Unsupervised Auxiliary Tasks* and later multi‑task/IMPALA‑style agents show that training the shared trunk to solve extra, reward‑related prediction problems (e.g., reward prediction, pixel control) improves sample efficiency and final performance. Your “how many points have been taken?” task is highly aligned with the return.
- **Counting points is a key latent variable.** In Sheepshead, the optimal policy depends heavily on how close each side is to 61, who is forced to take which future tricks, etc. If the value head only predicts win/loss, the trunk can “get away” with a coarse notion of advantage. A supervised regression on point totals forces it to encode a much sharper notion of score and trick ownership.
- **It’s a deterministic, low‑noise target.** Unlike future returns, current point totals are (logically) noise‑free functions of the game history. That makes the auxiliary loss a very clean signal that stabilizes representation learning.

Caveats:

- **Don’t overweight the auxiliary loss.** If its coefficient is too large relative to policy+value loss, the trunk may overfit to “being a good scorekeeper” at the expense of return. Treat it as a regularizer (e.g., 0.1–0.5× the scale of the value loss, then tune).
- **Respect the information set.** Only predict quantities that are in principle knowable to a perfect memory player from public information. For Sheepshead, per‑trick cards and trick takers are public, so current point totals are fine.

Overall, this is a textbook, return‑aligned auxiliary task; I’d expect it to help the actor learn expert‑like score awareness, especially in tight endgames.

## 2. How to structure the prediction head (incl. leasters)

The most useful target is one that is:

- **General across contract types** (normal hands, leasters, and any variants you add later),
- **Simple to learn**, and
- **Expressive enough** that the policy can easily derive any team‑level quantity it needs.

### 2.1. Strong recommendation: predict per‑player cumulative points

Instead of only “my team vs their team”, I’d recommend:

- **Aux head output**: a length‑5 vector (for 5‑handed Sheepshead)  
  \[
  \hat{\mathbf{p}} = (\hat{p}_0, \hat{p}_1, \hat{p}_2, \hat{p}_3, \hat{p}_4)
  \]
  where \(\hat{p}_i\) is the predicted **current cumulative card‑points taken by seat \(i\)**.
- **Target**: the true current cumulative points per seat, as known to the environment from played tricks.
- **Loss**: something like Huber or MSE on each component.

Why this structure is good:

- **Works for both normal hands and leasters.** The semantics of “how many points has each player taken?” don’t change between modes. In normal hands, win conditions are about team aggregates; in leasters, they’re about being as low as possible while >0 — but in both, the primitive quantity is the per‑player total.
- **Policy can derive whatever it needs.** From this vector and role information (who is picker, partner, defenders, or “everyone for themselves”), the network can internally form:
  - My team total: \(\sum_{i \in \text{my side}} p_i\)
  - Opposing side total
  - Margin to 61, etc.
- **Avoids brittle, mode‑dependent heads.** You don’t need different head shapes or semantics when the contract is leasters vs normal. The *reward function* changes; the “score keeping” task does not.

You can help the trunk by also giving it, as part of the observation, a small role/contract encoding, e.g.:

- One‑hot or embedding for **contract type**: normal, leasters, etc.
- Per‑seat **role flags**: picker, partner, defender, or “solo” (for leasters).

Then a single shared trunk + per‑player points head is flexible enough to support all game types.

### 2.2. Should you instead predict team totals?

You mentioned:

> We could output one prediction for points taken by the agent's team and another for the points taken by the opponents team (but what to do in case of leasters?)

This can work, but it’s strictly **less general**:

- In normal hands, “my team vs opponents” is natural.
- In leasters, there *is* no stable team structure; you’d have to redefine the head or collapse all opponents into one “rest of table” bucket, which doesn’t match the payoff structure (you care about beating each individual, not the sum of their points).

If you really want a team‑view head, a clean approach is:

- Keep the **per‑player head** as the main auxiliary task.
- Optionally add a tiny linear layer on top of the trunk (or on top of the per‑player predictions) that predicts:
  - **My side total points**
  - **Opponent side total points**
  - **Margin to 61**, etc.

These are just linear combinations of per‑player totals given the role mask, so they don’t add much expressive power but can act as additional shaping if you want.

### 2.3. Practical details & variants

A few concrete suggestions:

- **Predict current points, not final points.**  
  You already have a value head for “probability I win / expected return”. Let the auxiliary head focus on **current known quantities** (points so far) rather than final score; that keeps its job narrow and low‑variance.
- **Normalize the targets.**  
  E.g. divide points by 120 so all outputs are in \([0,1]\). That generally makes optimization easier and loss‑weight tuning more stable.
- **Consider feeding point totals directly as features too.**  
  In real Sheepshead, everyone *knows* the exact score; it isn’t partially observed. There’s no unfairness in:
  - Maintaining exact per‑player point totals in the environment, and  
  - Including them explicitly in the observation vector.
  
  If you do this, the auxiliary head then acts more like a **consistency/attention regularizer** (forcing the trunk to use and reconstruct these features) rather than discovering them from scratch via memory over the full trick history. This is very common in card‑game RL work: key, hard‑to‑forget quantities (pot size, trick counts, etc.) are usually explicit features, not left to recurrent networks to rediscover every step.

- **Loss weighting:**  
  Start with something like:
  - Policy loss: 1.0
  - Value loss: c_v (e.g., 0.5)
  - Point‑prediction loss: λ (e.g., 0.1–0.3), then tune by looking at:
    - Does win‑rate improve?
    - Do endgame decisions that hinge on point margins improve?
    - Does the auxiliary loss converge reasonably without collapsing the other losses?

---

**In summary:** adding a critic auxiliary head that predicts **per‑player current point totals** is well‑justified by auxiliary‑task literature and very well aligned with what an expert Sheepshead player tracks. Structuring it as a per‑seat regression vector (plus contract/role features) makes it naturally handle both normal hands and leasters, and gives your PPO trunk exactly the score‑awareness needed to push play toward expert level.

---

**User**

Okay. I like the idea of predicting the known points taken by each player.

Regarding perfect information: one caveat here is that the picker buries 2 cards, and the point values of these cards is not known to any other player until the end of the game. I think this aux prediction should include the bury points but only for the picker, and otherwise only include points that an observer could count. Do you agree?

The get_state_dict output orders the cards played relative to the actor in trick_card_ids. We should use this same relative positioning for the point prediction head.

Propose detailed changes to add the per-player aux point prediction as described.

---

**Cursor**

I'm going to inspect your environment, encoder, and PPO code to see how state dicts and heads are currently structured, then I'll design and wire in a per-player aux point head that uses relative seating and treats bury points as picker-only knowledge. I'll keep edits tightly scoped to only what's needed for this feature.

```python
#!/usr/bin/env python3
"""
Training utilities shared across training scripts.
"""

from typing import List, Dict
import matplotlib.pyplot as plt

from sheepshead import (
    TRUMP,
    ACTIONS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    get_card_suit,
    get_trick_points,
)


LEASTER_FINAL_REWARD_BONUS = 0.05
TRICK_POINT_RATIO = 360.0
TOTAL_DECK_POINTS = 120.0


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


def estimate_hand_strength_category(cards: List[str]) -> str:
    """Categorize hand strength into 'weak' | 'medium' | 'strong'.

    Bins:
      - weak   ≤ 4
      - medium 5–7
      - strong ≥ 8
    """
    score = estimate_hand_strength_score(cards)
    if score <= 4:
        return "weak"
    if score <= 7:
        return "medium"
    return "strong"


def get_partner_selection_mode(episode: int) -> int:
    return PARTNER_BY_CALLED_ACE if (episode % 2 == 0) else PARTNER_BY_JD


def compute_known_points_rel(game, player):
    """Return normalized known point totals for all seats from the acting player's perspective.

    Known points include:
      - Public trick points in game.points_taken (including blind/UNDER handling).
      - Bury card points, but only for the picker in non-leaster games once the bury is fixed.

    The result is a length-5 list ordered by relative seat:
      1 = self, 2 = left-hand opponent, ..., 5 = right-hand opponent.
    Values are normalized by TOTAL_DECK_POINTS so they lie in [0, 1].
    """
    base_points = list(getattr(game, "points_taken", [0, 0, 0, 0, 0]))

    # In standard games, the picker uniquely knows the bury card identities and their points.
    # We expose those points only to the picker, and only once the bury is fixed.
    if not getattr(game, "is_leaster", False):
        if getattr(player, "is_picker", False) and len(getattr(game, "bury", [])) == 2:
            base_points[player.position - 1] += get_trick_points(game.bury)

    rel_points = []
    for rel in range(1, 6):
        abs_seat = ((player.position + rel - 2) % 5) + 1
        rel_points.append(base_points[abs_seat - 1] / TOTAL_DECK_POINTS)
    return rel_points


def calculate_trick_reward(trick_points: int) -> float:
    """Intermediate reward for trick points."""
    return trick_points / TRICK_POINT_RATIO


def is_same_team_as_winner(player, winner_pos: int, game) -> bool:
    if game.is_leaster:
        return False
    player_picker_team = (player.is_picker or player.is_partner or player.is_secret_partner)
    winner_player = game.players[winner_pos - 1]
    winner_picker_team = (winner_player.is_picker or winner_player.is_partner or winner_player.is_secret_partner)
    return player_picker_team == winner_picker_team


def apply_trick_rewards(trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float, game) -> None:
    for transition in trick_transitions:
        player = transition['player']
        reward_multiplier = 1.0 if is_same_team_as_winner(player, trick_winner_pos, game) else -1.0
        transition['intermediate_reward'] += trick_reward * reward_multiplier


def apply_leaster_trick_rewards(trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float) -> None:
    for transition in trick_transitions:
        player = transition['player']
        if player.position == trick_winner_pos:
            transition['intermediate_reward'] -= trick_reward


def update_intermediate_rewards_for_action(
    game,
    player,
    action,
    transition,
    current_trick_transitions,
    pick_weight=1.0, # Multiplier for pick head shaping
    partner_weight=1.0, # Multiplier for partner head shaping
    bury_weight=1.0, # Multiplier for bury head shaping
    play_weight=1.0, # Multiplier for play head shaping
):
    """Apply shared intermediate reward shaping and trick tracking.
    Uses game engine state to detect leads and trick phase; no counter needed.
    """
    action_name = ACTIONS[action - 1]

    # Hand-conditioned PICK/PASS shaping (small human-like nudges)
    if action_name in ("PICK", "PASS"):
        score = estimate_hand_strength_score(player.hand)
        if score <= 4:
            pick_bonus, pass_bonus = -0.15, +0.15
        elif score <= 6:
            pick_bonus, pass_bonus = 0, 0
        elif score <= 7:
            pick_bonus, pass_bonus = +0.02, -0.02
        elif score >= 8:
            pick_bonus, pass_bonus = +0.2, -0.2
        transition['intermediate_reward'] += pick_bonus if action_name == "PICK" else pass_bonus
        transition['intermediate_reward'] *= pick_weight

    # ALONE shaping: discourage going alone with weak hands
    elif action_name == "ALONE":
        score = estimate_hand_strength_score(player.hand)
        if score <= 8:
            transition['intermediate_reward'] += -0.1
        transition['intermediate_reward'] *= partner_weight

    # Bury penalty: discourage burying trump if not required
    elif "BURY" in action_name:
        card = action_name[5:]
        # Derive allowed bury actions from this step's valid_actions
        valid_actions = transition.get('valid_actions', set())
        allowed_bury_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
        has_allowed_fail_bury = any(get_card_suit(c) != "T" for c in allowed_bury_cards)
        has_allowed_trump_bury = any(get_card_suit(c) == "T" for c in allowed_bury_cards)

        if card in TRUMP and has_allowed_fail_bury:
            transition['intermediate_reward'] += -0.1
        elif card not in TRUMP and has_allowed_trump_bury:
            # Small preference when both options exist
            transition['intermediate_reward'] += 0.02
        elif not has_allowed_fail_bury and card.startswith("Q"):
            # Even if we have to bury trump, we should not bury queens.
            transition['intermediate_reward'] += -0.1
        elif not has_allowed_fail_bury and card.startswith("J"):
            # Even if we have to bury trump, unlikely we should bury jacks.
            transition['intermediate_reward'] += -0.02
        transition['intermediate_reward'] *= bury_weight

    elif "PLAY" in action_name:
        is_lead = (game.cards_played == 0) and (game.leader == player.position)
        if is_lead:
            valid_actions = transition.get('valid_actions', set())
            allowed_play_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
            has_allowed_fail_play = any(get_card_suit(c) != "T" for c in allowed_play_cards)
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
                transition['intermediate_reward'] += -0.06
            elif (
                game.called_card
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and not game.was_called_suit_played
                and game.called_suit == get_card_suit(card)
            ):
                # Encourage defenders to lead called suit (early)
                transition['intermediate_reward'] += 0.1 - (0.02 * game.current_trick)
            elif (
                not game.is_leaster
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and card not in TRUMP
            ):
                # Nudge defenders toward leading fail
                transition['intermediate_reward'] += 0.03
            elif (
                not game.is_leaster
                and (player.is_partner or player.is_secret_partner)
                and card in TRUMP
            ):
                # Encourage partners to lead trump
                transition['intermediate_reward'] += 0.1
            elif (
                not game.is_leaster
                and player.is_picker
                and card in TRUMP
            ):
                # Nudge picker toward leading trump
                transition['intermediate_reward'] += 0.03
        transition['intermediate_reward'] *= play_weight

        current_trick_transitions.append(transition)


def handle_trick_completion(game, current_trick_transitions):
    """If a trick has completed, apply trick-based rewards and reset tracking.

    Returns True if the trick just completed, else False.
    """
    if game.was_trick_just_completed:
        trick_points = game.trick_points[game.current_trick - 1]
        trick_winner = game.trick_winners[game.current_trick - 1]
        trick_reward = calculate_trick_reward(trick_points)

        if not game.is_leaster:
            apply_trick_rewards(current_trick_transitions, trick_winner, trick_reward, game)
        else:
            apply_leaster_trick_rewards(current_trick_transitions, trick_winner, trick_reward)

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
        player = transition['player']
        player_pos = player.position
        final_score = final_scores[player_pos - 1]
        is_episode_done = (i == last_index)

        if is_leaster:
            # Increase final reward for leasters to compensate for negative trick rewards.
            # Agent should dislike playing leasters most of the time (similar to human behavior)
            # but without this the bias is a bit too strong.
            leaster_reward = final_score / 12 + LEASTER_FINAL_REWARD_BONUS
            final_reward = leaster_reward if is_episode_done else 0.0
        else:
            final_reward = (final_score / 12) if is_episode_done else 0.0

        total_reward = final_reward + transition['intermediate_reward']
        yield {
            'transition': transition,
            'reward': total_reward,
        }


def save_training_plot(training_data, save_path='training_progress.png'):
    """Enhanced training plots with strategic metrics."""
    episodes = training_data['episodes']

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # Score progression
    ax1.plot(episodes, training_data['picker_avg'], label='Picker Avg', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Picker Score')
    ax1.set_title('Score Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pick rate and correlation
    ax2.plot(episodes, training_data['called_pick_rate'], color='orange', alpha=0.8, label='Called Ace Pick Rate')
    ax2.plot(episodes, training_data['jd_pick_rate'], color='purple', alpha=0.8, label='JD Pick Rate')
    ax2_twin = ax2.twinx()
    if 'pick_hand_correlation' in training_data and len(training_data['pick_hand_correlation']) > 0:
        corr = training_data['pick_hand_correlation']
        strat_eps = training_data.get('strategic_episodes', [])
        if strat_eps:
            # Align by trimming to common length from the tail
            n = min(len(corr), len(strat_eps))
            x = strat_eps[-n:]
            y = corr[-n:]
        else:
            # Fallback: align to tail of episodes
            x = episodes[-len(corr):]
            y = corr
        ax2_twin.plot(x, y, color='green', alpha=0.8, label='Hand Correlation', marker='o')
        ax2_twin.set_ylabel('Hand Strength Correlation')
        ax2_twin.legend(loc='upper right')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Pick Rate (%)')
    ax2.set_title('Pick Strategy Quality')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Trump leading rates
    if 'picker_trump_rate' in training_data and len(training_data['picker_trump_rate']) > 0:
        picker_trump = training_data['picker_trump_rate']
        defender_trump = training_data.get('defender_trump_rate', [])
        strat_eps = training_data.get('strategic_episodes', [])
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
        ax3.plot(x, y1, color='blue', alpha=0.8, label='Picker Team', marker='o')
        ax3.plot(x, y2, color='red', alpha=0.8, label='Defender Team', marker='o')
        ax3.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='Picker Target')
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Defender Target')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Trump Lead Rate (%)')
        ax3.set_title('Trump Leading Strategy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Bury quality
    if 'bury_quality_rate' in training_data and len(training_data['bury_quality_rate']) > 0:
        bury = training_data['bury_quality_rate']
        strat_eps = training_data.get('strategic_episodes', [])
        if strat_eps:
            n = min(len(bury), len(strat_eps))
            x = strat_eps[-n:]
            y = bury[-n:]
        else:
            x = episodes[-len(bury):]
            y = bury
        ax4.plot(x, y, color='purple', alpha=0.8, marker='o')
        ax4.axhline(y=90, color='purple', linestyle='--', alpha=0.5, label='Good Target')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Good Bury Rate (%)')
        ax4.set_title('Bury Decision Quality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Rates: Alone call and Leaster
    # When resuming training, historical CSVs may not contain these rate series.
    # Rather than require exact length matching with `episodes`, plot whatever
    # portion is available by aligning each series to the tail of the episodes
    # timeline. This keeps the figure renderable on resume while still showing
    # new data going forward.
    alone_rate = training_data.get('alone_rate', [])
    leaster_rate = training_data.get('leaster_rate', [])
    any_rate_plotted = False
    if alone_rate:
        ex = episodes[-len(alone_rate):]
        ax5.plot(ex, alone_rate, color='brown', alpha=0.8, label='Alone Call Rate')
        any_rate_plotted = True
    if leaster_rate:
        ex = episodes[-len(leaster_rate):]
        ax5.plot(ex, leaster_rate, color='gray', alpha=0.8, label='Leaster Rate')
        any_rate_plotted = True
    if any_rate_plotted:
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Rate (%)')
        ax5.set_title('Alone Call and Leaster Rates')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Team Point Difference
    if 'team_point_diff' in training_data and len(training_data['team_point_diff']) > 0:
        ax6.plot(episodes, training_data['team_point_diff'], color='red', alpha=0.8)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Team Point Difference')
        ax6.set_title('Team Point Difference (Picker - Defender)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()




```

I added a shared helper to compute per-seat known points (including bury only for the picker) and will now wire it into the PPO/PFSP training scripts and extend the critic with a per-player points head plus loss.

```python
#!/usr/bin/env python3
"""
Extended long-term PPO training for Sheepshead.
"""

import torch
import numpy as np
import random
import time
import os
from collections import deque
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from ppo import PPOAgent
from sheepshead import Game, ACTIONS, ACTION_LOOKUP, ACTION_IDS, TRUMP, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from training_utils import (
    process_episode_rewards,
    get_partner_selection_mode,
    save_training_plot,
    update_intermediate_rewards_for_action,
    handle_trick_completion,
    compute_known_points_rel,
)


def analyze_strategic_decisions(agent, num_samples=100):
    """Analyze strategic decision quality instead of random opponent evaluation."""

    # Trump leading analysis
    trump_leads = {'picker_team': 0, 'picker_total': 0, 'defender_team': 0, 'defender_total': 0}

    # Bury quality analysis
    bury_quality = {'good_burys': 0, 'bad_burys': 0, 'total_burys': 0}

    # Pick decision correlation with hand strength
    pick_decisions = []
    hand_strengths = []

    for episode in range(num_samples):
        game = Game(partner_selection_mode=get_partner_selection_mode(episode))
        # Ensure recurrent state is fresh for analysis episode
        agent.reset_recurrent_state()

        # Analyze pick decisions
        initial_player = game.players[0]
        hand_strength = sum(3 if c[0] == 'Q' else 2 if c[0] == 'J' else 1 if c in TRUMP else 0 for c in initial_player.hand)

        sdict = initial_player.get_state_dict()
        initial_actions = initial_player.get_valid_action_ids()
        with torch.no_grad():
            action_probs, _ = agent.get_action_probs_with_logits(sdict, initial_actions, player_id=initial_player.position)

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
                        action_probs, _ = agent.get_action_probs_with_logits(sdict, actions, player_id=player.position)
                    action = torch.distributions.Categorical(action_probs).sample().item() + 1
                    action_name = ACTION_LOOKUP[action]

                    # Analyze trump leading
                    if "PLAY" in action_name and game.play_started and game.cards_played == 0:
                        card = action_name.split()[-1]
                        is_trump_lead = card in TRUMP
                        is_picker_team = (player.is_picker or player.is_partner or player.is_secret_partner)

                        if is_picker_team:
                            trump_leads['picker_total'] += 1
                            if is_trump_lead:
                                trump_leads['picker_team'] += 1
                        else:
                            trump_leads['defender_total'] += 1
                            if is_trump_lead:
                                trump_leads['defender_team'] += 1

                    # Analyze bury decisions
                    if "BURY" in action_name:
                        card = action_name.split()[-1]
                        bury_quality['total_burys'] += 1

                        # Good bury: fail
                        if card not in TRUMP:
                            bury_quality['good_burys'] += 1
                        else:
                            bury_quality['bad_burys'] += 1

                    player.act(action)

    # Calculate metrics
    pick_hand_correlation = np.corrcoef(hand_strengths, pick_decisions)[0, 1] if len(hand_strengths) > 1 else 0

    picker_trump_rate = trump_leads['picker_team'] / max(trump_leads['picker_total'], 1) * 100
    defender_trump_rate = trump_leads['defender_team'] / max(trump_leads['defender_total'], 1) * 100

    bury_quality_rate = bury_quality['good_burys'] / max(bury_quality['total_burys'], 1) * 100

    return {
        'pick_hand_correlation': pick_hand_correlation,
        'picker_trump_rate': picker_trump_rate,
        'defender_trump_rate': defender_trump_rate,
        'bury_quality_rate': bury_quality_rate
    }

def train_ppo(num_episodes=300000, update_interval=2048, save_interval=5000,
                            strategic_eval_interval=10000, resume_model=None, activation='swish'):
    """
    PPO training with strategic evaluation metrics.
    """
    print("🚀 Starting PPO training...")
    print("="*60)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Activation function: {activation.upper()}")
    print("="*60)

    # Create agent with optimized hyperparameters
    agent = PPOAgent(len(ACTIONS),
                    lr_actor=1.0e-4,
                    lr_critic=1.0e-4,
                    activation=activation)

    # Resume from specified model or try to load best existing
    start_episode = 0
    if resume_model:
        try:
            agent.load(resume_model, load_optimizers=True)
            print(f"✅ Loaded {resume_model} for continuation")
            # Try to extract episode number from filename
            if 'checkpoint_' in resume_model:
                start_episode = int(resume_model.split('_')[-1].split('.')[0])
                print(f"📍 Resuming from episode {start_episode:,}")
        except Exception as e:
            print(f"❌ Could not load {resume_model}: {e}")
    else:
        print("🆕 Starting fresh training")

    picker_scores = deque(maxlen=3000)
    pick_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    pass_decisions = [deque(maxlen=3000), deque(maxlen=3000)]

    leaster_window = deque(maxlen=3000)          # 1 ⇒ leaster, 0 ⇒ regular game
    alone_call_window = deque(maxlen=3000)       # 1 ⇒ ALONE called (non-leaster games)
    called_ace_window = deque(maxlen=3000)       # 1 ⇒ partner mode = Called-Ace, else 0
    called_under_window = deque(maxlen=3000)     # 1 ⇒ called-under occurred that game
    called_10_window = deque(maxlen=3000)        # 1 ⇒ called-10s occurred that game
    team_point_differences = deque(maxlen=3000)
    best_team_difference = float('inf')  # Lower is better (smaller point difference)
    current_avg_picker_score = float('-inf')

    training_data = {
        'episodes': [],
        'recent_avg': [],
        'overall_avg': [],
        'picker_avg': [],
        'called_pick_rate': [],
        'jd_pick_rate': [],
        'alone_rate': [],
        'leaster_rate': [],
        'learning_rate': [],
        'time_elapsed': [],
        'pick_hand_correlation': [],
        'picker_trump_rate': [],
        'defender_trump_rate': [],
        'bury_quality_rate': [],
        'team_point_diff': [],
        'strategic_episodes': []
    }

    # Running picker baseline for reward shaping

    # Create checkpoint directory with activation function suffix
    checkpoint_dir = f'checkpoints_{activation}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    game_count = 0
    last_checkpoint_time = start_time

    print(f"\n🎮 Beginning training... (target: {num_episodes:,} episodes)")
    print("-" * 60)

    transitions_since_update = 0
    for episode in range(start_episode + 1, num_episodes + 1):
        partner_mode = get_partner_selection_mode(episode)
        game = Game(partner_selection_mode=partner_mode)
        # Reset recurrent hidden states in the actor at the start of each game
        agent.reset_recurrent_state()
        episode_scores = []
        episode_picks = 0
        episode_passes = 0

        # Store all transitions for this episode independently for each player
        episode_transitions = {player.position: [] for player in game.players}

        # Track PLAY transitions for the current trick
        current_trick_transitions = []

        # Play full game with self-play
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    state = player.get_state_dict()
                    action, log_prob, value = agent.act(state, valid_actions, player.position)

                    transition = {
                        'kind': 'action',
                        'player': player,
                        'state': state,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy(),
                        'intermediate_reward': 0.0,
                        'secret_partner_label': 1.0 if player.is_secret_partner else 0.0,
                        'points_label': compute_known_points_rel(game, player),
                    }

                    action_name = ACTIONS[action - 1]

                    # Track pick/pass decisions for statistics
                    if action_name == "PICK":
                        episode_picks += 1
                    elif action_name == "PASS":
                        episode_passes += 1

                    episode_transitions[player.position].append(transition)

                    # Apply shared intermediate rewards and track trick transitions
                    update_intermediate_rewards_for_action(
                        game,
                        player,
                        action,
                        transition,
                        current_trick_transitions,
                    )

                    player.act(action)

                    # Trick resolution and observation frames
                    trick_completed = handle_trick_completion(
                        game, current_trick_transitions
                    )
                    if trick_completed and not game.is_done():
                        # ------------------------------------------------
                        # Add post-trick observation frames for all seats
                        # (stored for training-time recurrent unroll)
                        # Also update the online recurrent hidden state
                        # ------------------------------------------------
                        for seat in game.players:
                            # Update online recurrent state
                            agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
                            # Store for training-time unroll
                            episode_transitions[seat.position].append({
                                'kind': 'observation',
                                'player': seat,
                                'state': seat.get_state_dict(),
                            })

                    valid_actions = player.get_valid_action_ids()

        final_scores = [player.get_score() for player in game.players]
        episode_scores = final_scores[:]

        # ---------------------------------------------
        # Compute rewards per player and store via episode API
        # ---------------------------------------------
        # Build per-player action lists
        actions_by_player = {pos: [t for t in episode_transitions[pos] if t['kind'] == 'action'] for pos in episode_transitions}

        # Compute rewards per player
        reward_maps_by_player = {}
        for pos, acts in actions_by_player.items():
            reward_map = {}
            for reward_data in process_episode_rewards(
                acts,
                final_scores,
                game.is_leaster
            ):
                tr = reward_data['transition']
                reward_map[id(tr)] = reward_data['reward']
            reward_maps_by_player[pos] = reward_map

        # Build annotated event streams and ingest per player
        for pos, seq in episode_transitions.items():
            events = []
            rmap = reward_maps_by_player.get(pos, {})
            for ev in seq:
                if ev.get('kind') == 'observation':
                    events.append({
                        'kind': 'observation',
                        'state': ev['state'],
                        'player_id': pos,
                    })
                else:
                    reward = rmap.get(id(ev), 0.0)
                    events.append({
                        'kind': 'action',
                        'state': ev['state'],
                        'action': ev['action'],
                        'log_prob': ev['log_prob'],
                        'value': ev['value'],
                        'valid_actions': ev['valid_actions'],
                        'reward': reward,
                        'player_id': pos,
                        'win_label': 1.0 if episode_scores[pos - 1] > 0 else 0.0,
                        'final_return_label': float(episode_scores[pos - 1]),
                        'secret_partner_label': ev.get('secret_partner_label', 0.0),
                        'points_label': ev.get('points_label', None),
                    })
            agent.store_episode_events(events)
            transitions_since_update += sum(1 for e in events if e['kind'] == 'action')

        # Track statistics
        picker_score = episode_scores[game.picker - 1] if game.picker else 0


        # Calculate team point difference (picker team points - defender team points)
        if game.picker and not game.is_leaster:
            picker_team_points = game.get_final_picker_points()
            defender_team_points = game.get_final_defender_points()
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0  # No team difference in leaster games

        # --------------------------------------------------
        # Append episode outcome to rolling windows
        # --------------------------------------------------
        is_leaster_ep = 1 if game.is_leaster else 0
        leaster_window.append(is_leaster_ep)

        is_called_ace_ep = 1 if partner_mode == PARTNER_BY_CALLED_ACE else 0
        called_ace_window.append(is_called_ace_ep)

        # Only meaningful for Called-Ace, non-leaster games
        if is_called_ace_ep and not is_leaster_ep:
            called_under_window.append(1 if game.is_called_under else 0)
            called_10_window.append(1 if (game.called_card and game.called_card.startswith("10")) else 0)
        elif is_called_ace_ep:
            called_under_window.append(0)
            called_10_window.append(0)

        picker_scores.append(picker_score)
        # ALONE tracking for games with a picker (exclude leaster)
        if not game.is_leaster:
            alone_call_window.append(1 if game.alone_called else 0)
        pick_decisions[get_partner_selection_mode(episode)].append(episode_picks)
        pass_decisions[get_partner_selection_mode(episode)].append(episode_passes)
        team_point_differences.append(team_point_diff)
        game_count += 1

        # Update model periodically by transition count (action transitions only)
        if transitions_since_update >= update_interval:
            print(f"🔄 Updating model after {transitions_since_update} transitions... (Episode {episode:,})")

            # Separate entropy decay schedules
            entropy_play_start, entropy_play_end = 0.05, 0.05
            entropy_pick_start, entropy_pick_end = 0.08, 0.05
            entropy_partner_start, entropy_partner_end = 0.05, 0.04
            entropy_bury_start, entropy_bury_end = 0.04, 0.03
            decay_fraction = min(episode / num_episodes, 1.0)
            agent.entropy_coeff_play = entropy_play_start + (entropy_play_end - entropy_play_start) * decay_fraction
            agent.entropy_coeff_pick = entropy_pick_start + (entropy_pick_end - entropy_pick_start) * decay_fraction
            agent.entropy_coeff_partner = entropy_partner_start + (entropy_partner_end - entropy_partner_start) * decay_fraction
            agent.entropy_coeff_bury = entropy_bury_start + (entropy_bury_end - entropy_bury_start) * decay_fraction

            update_stats = agent.update(epochs=4, batch_size=256)

            # Log advantage and value target statistics
            if update_stats:
                adv_stats = update_stats['advantage_stats']
                val_stats = update_stats['value_target_stats']
                num_transitions = update_stats['num_transitions']
                approx_kl = update_stats.get('approx_kl', None)
                early_stop = update_stats.get('early_stop', False)

                print(f"   Transitions: {num_transitions}")
                print(f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
                print(f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")
                if approx_kl is not None:
                    print(f"   PPO KL: {approx_kl:.4f}  Early stop: {early_stop}")
                if 'timing' in update_stats:
                    t = update_stats['timing']
                    print(
                        f"   Timing - build: {t['build_s']:.3f}s, forward: {t['forward_s']:.3f}s, "
                        f"backward: {t['backward_s']:.3f}s, step: {t['step_s']:.3f}s, total: {t['total_update_s']:.3f}s, "
                        f"opt_steps: {t['optimizer_steps']}"
                    )
                head_entropy = update_stats.get('head_entropy')
                if head_entropy:
                    print(
                        f"   Entropy - pick: {head_entropy.get('pick', 0.0):.3f}, "
                        f"partner: {head_entropy.get('partner', 0.0):.3f}, "
                        f"bury: {head_entropy.get('bury', 0.0):.3f}, "
                        f"play: {head_entropy.get('play', 0.0):.3f}"
                    )

            game_count = 0
            transitions_since_update = 0

                # Strategic evaluation at intervals
        if episode % strategic_eval_interval == 0:
            print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
            strategic_metrics = analyze_strategic_decisions(agent, num_samples=200)

            # Store strategic metrics
            training_data['strategic_episodes'].append(episode)
            training_data['pick_hand_correlation'].append(strategic_metrics['pick_hand_correlation'])
            training_data['picker_trump_rate'].append(strategic_metrics['picker_trump_rate'])
            training_data['defender_trump_rate'].append(strategic_metrics['defender_trump_rate'])
            training_data['bury_quality_rate'].append(strategic_metrics['bury_quality_rate'])

            print(f"   Pick-Hand Correlation: {strategic_metrics['pick_hand_correlation']:.3f}")
            print(f"   Picker Trump Rate: {strategic_metrics['picker_trump_rate']:.1f}%")
            print(f"   Defender Trump Rate: {strategic_metrics['defender_trump_rate']:.1f}%")
            print(f"   Bury Quality Rate: {strategic_metrics['bury_quality_rate']:.1f}%")

        # Progress reporting and data collection
        if episode % 1000 == 0:
            current_avg_picker_score = np.mean(picker_scores) if picker_scores else 0
            # Compute pick-rate over all individual decisions (weighting games by the number of decisions they contributed)
            total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
            total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
            total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
            total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
            current_called_pick_rate = (100 * total_called_picks / (total_called_picks + total_called_passes)) if (total_called_picks + total_called_passes) > 0 else 0
            current_jd_pick_rate = (100 * total_jd_picks / (total_jd_picks + total_jd_passes)) if (total_jd_picks + total_jd_passes) > 0 else 0
            current_team_diff = np.mean(team_point_differences) if team_point_differences else 0
            # --- Rolling-window rates ---
            current_leaster_rate = (sum(leaster_window) / len(leaster_window)) * 100 if leaster_window else 0
            current_alone_rate = (sum(alone_call_window) / len(alone_call_window)) * 100 if alone_call_window else 0

            ca_denominator = sum(called_ace_window) or 1  # avoid divide-by-zero
            current_called_under_rate = (sum(called_under_window) / ca_denominator) * 100
            current_called_10s_rate = (sum(called_10_window) / ca_denominator) * 100
            elapsed = time.time() - start_time

            # Collect data for plotting
            training_data['episodes'].append(episode)
            training_data['picker_avg'].append(current_avg_picker_score)
            training_data['called_pick_rate'].append(current_called_pick_rate)
            training_data['jd_pick_rate'].append(current_jd_pick_rate)
            training_data['learning_rate'].append(agent.actor_optimizer.param_groups[0]['lr'])
            training_data['time_elapsed'].append(elapsed)
            training_data['team_point_diff'].append(current_team_diff)
            training_data['alone_rate'].append(current_alone_rate)
            training_data['leaster_rate'].append(current_leaster_rate)

            # Strategic metrics are collected separately during strategic evaluation intervals
            # Don't try to collect them here as they're not always available

            # Calculate training speed
            games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

            print(f"📊 Episode {episode:,}/{num_episodes:,} ({episode/num_episodes*100:.1f}%)")
            print("   " + "-" * 40)
            print(f"   Picker avg: {current_avg_picker_score:+.3f}")
            print(f"   Team point diff: {current_team_diff:+.1f}")
            print(f"   Called Ace Pick rate: {current_called_pick_rate:.1f}%")
            print(f"   JD Pick rate: {current_jd_pick_rate:.1f}%")
            print("   " + "-" * 20)
            print(f"   Leaster Rate: {current_leaster_rate:.2f}%")
            print(f"   Alone Call Rate: {current_alone_rate:.2f}%")
            print(f"   Called Under Rate: {current_called_under_rate:.2f}%")
            print(f"   Called 10s Rate: {current_called_10s_rate:.2f}%")
            print("   " + "-" * 40)
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed/60:.1f} min")
            print("   " + "-" * 40)

            # Save best model based on team point difference (lower is better)
            # We want the absolute value to be as small as possible
            if current_team_diff < best_team_difference:
                best_team_difference = current_team_diff
                agent.save(f'best_{activation}_ppo.pt')
                print(f"   🏆 New best team point difference: {best_team_difference:.1f}! Model saved.")

        # Save regular checkpoints
        if episode % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/{activation}_checkpoint_{episode}.pt'
            agent.save(checkpoint_path)

            # Save enhanced training plot
            if len(training_data['episodes']) > 10:
                plot_path = f'{checkpoint_dir}/training_progress_{episode}.png'
                save_training_plot(training_data, plot_path)

            # Calculate time since last checkpoint
            checkpoint_time = time.time()
            time_since_last = checkpoint_time - last_checkpoint_time
            last_checkpoint_time = checkpoint_time

            print(f"💾 Checkpoint saved at episode {episode:,}")
            print(f"   Time for last {save_interval:,} episodes: {time_since_last/60:.1f} min")
            remaining_episodes = num_episodes - episode
            if remaining_episodes > 0:
                estimated_time = remaining_episodes * (time_since_last / save_interval) / 60
                print(f"   Estimated time remaining: {estimated_time:.1f} min")

    # Final update and save
    if agent.events:
        print("🔄 Final model update...")
        final_update_stats = agent.update()

        # Log final advantage and value target statistics
        if final_update_stats:
            adv_stats = final_update_stats['advantage_stats']
            val_stats = final_update_stats['value_target_stats']
            num_transitions = final_update_stats['num_transitions']

            print(f"   Final Transitions: {num_transitions}")
            print(f"   Final Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
            print(f"   Final Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")

    agent.save(f'final_{activation}_ppo.pt')

    # Save final enhanced training plot
    if len(training_data['episodes']) > 0:
        save_training_plot(training_data, f'final_{activation}_training.png')

    total_time = time.time() - start_time
    print("\n🎉 Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}")
    print(f"   Final team point difference: {np.mean(team_point_differences) if team_point_differences else 0:.1f}")
    print(f"   Best team point difference: {best_team_difference:.1f}")
    total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
    total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
    total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
    total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
    final_called_pick_rate = (100 * total_called_picks / (total_called_picks + total_called_passes)) if (total_called_picks + total_called_passes) > 0 else 0
    final_jd_pick_rate = (100 * total_jd_picks / (total_jd_picks + total_jd_passes)) if (total_jd_picks + total_jd_passes) > 0 else 0
    print(f"   Final called Ace Pick rate: {final_called_pick_rate:.1f}%")
    print(f"   Final JD Pick rate: {final_jd_pick_rate:.1f}%")
    print(f"   Training speed: {(num_episodes-start_episode)/(total_time/60):.1f} episodes/min")

def main():
    parser = ArgumentParser(description="PPO training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=100000,
                       help="Number of training episodes (default: 100,000)")
    parser.add_argument("--update-interval", type=int, default=4096,
                       help="Number of games between model updates")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Number of episodes between checkpoints")
    parser.add_argument("--strategic-eval-interval", type=int, default=10000,
                       help="Number of episodes between strategic evaluations")
    parser.add_argument("--resume", type=str, default=None,
                       help="Model file to resume from")
    parser.add_argument("--activation", type=str, default='swish', choices=['relu', 'swish'],
                       help="Activation function to use (default: swish)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend('Agg')

    train_ppo(
        args.episodes,
        args.update_interval,
        args.save_interval,
        args.strategic_eval_interval,
        args.resume,
        args.activation
    )

if __name__ == "__main__":
    main()
```

```python
#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) training for Sheepshead.

This script implements population-based training where a single agent trains against
a diverse population of opponents, ranked using OpenSkill ratings.
"""

import torch
import numpy as np
import random
import time
import os
import csv
import sys
from collections import deque
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass, field
from openskill.models import PlackettLuce

from ppo import PPOAgent
from pfsp import (
    PFSPPopulation,
    create_initial_population_from_checkpoints,
    profile_pop_agent_action,
    profile_trick_completion,
)
from training_utils import estimate_hand_strength_category, compute_known_points_rel
from sheepshead import (
    Game,
    ACTIONS,
    ACTION_IDS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    get_partner_mode_name,
)

from train_ppo import analyze_strategic_decisions
from training_utils import (
    process_episode_rewards,
    get_partner_selection_mode,
    save_training_plot,
    update_intermediate_rewards_for_action,
    handle_trick_completion,
)


@dataclass
class PFSPHyperparams:
    # Adaptive exploration for pick head (rate-based bump scheduling)
    low_pick_rate_threshold: float = 20.0  # percent
    high_pick_rate_threshold: float = 50.0  # percent
    pick_entropy_bump: float = 0.04        # added to base decayed pick entropy
    pick_entropy_bump_duration: int = 25000  # episodes

    # PASS-floor epsilon controller
    # Ensures minimum PASS probability on pick steps if picker average score is low.
    high_pick_rate_ceiling: float = 80.0  # Alter distribution to force PASS after this threshold
    pass_floor_eps_base: float = 0.0
    pass_floor_eps_target: float = 0.08
    pass_floor_eps_step_up: float = 0.02
    pass_floor_eps_step_down: float = 0.02
    pass_floor_eps_picker_avg_threshold: float = -0.75

    # PICK-floor epsilon controller
    # Ensures minimum PICK probability on pick steps if overall pick rate is low.
    low_pick_rate_floor: float = 8.0  # percent
    pick_floor_eps_base: float = 0.0
    pick_floor_eps_target: float = 0.05
    pick_floor_eps_step_up: float = 0.02
    pick_floor_eps_step_down: float = 0.02

    # Adaptive exploration for partner head (ALONE decision; bump scheduling)
    low_alone_rate_threshold: float = 2.5    # percent
    high_alone_rate_threshold: float = 30.0  # percent
    partner_entropy_bump: float = 0.04       # added to base decayed partner entropy
    partner_entropy_bump_duration: int = 25000  # episodes

    # Partner CALL mixture epsilon controller
    # Probability floor over CALL actions when picker average score is low.
    high_alone_rate_ceiling: float = 60.0  # Alter distribution to force partner calls after this threshold
    partner_call_eps_base: float = 0.0
    partner_call_eps_max_mid: float = 0.05   # when picker avg <= mid_picker_avg_threshold
    partner_call_eps_mid_picker_avg_threshold: float = -0.75
    partner_call_eps_max_high: float = 0.10  # when picker avg <= high_picker_avg_threshold
    partner_call_eps_high_picker_avg_threshold: float = -2
    partner_call_eps_step_up: float = 0.02
    partner_call_eps_step_down: float = 0.02

    # Adaptive exploration for bury head (bury decisions quality)
    # If bury_quality_rate drops below a threshold, temporarily bump bury entropy.
    low_bury_quality_threshold: float = 85.0  # percent
    bury_entropy_bump: float = 0.04          # added to base decayed bury entropy
    bury_entropy_bump_duration: int = 19000  # episodes

    # Entropy schedules (start -> end)
    entropy_pick_start: float = 0.05
    entropy_pick_end: float = 0.02
    entropy_partner_start: float = 0.05
    entropy_partner_end: float = 0.02
    entropy_bury_start: float = 0.04
    entropy_bury_end: float = 0.015
    entropy_play_start: float = 0.05
    entropy_play_end: float = 0.02

    # Shaped reward schedules (percent -> weight).
    shaping_schedule_pick: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    shaping_schedule_partner: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    shaping_schedule_bury: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    shaping_schedule_play: dict[int, float] = field(default_factory=lambda: {0: 1.0})


DEFAULT_HYPERPARAMS = PFSPHyperparams()


def interpolated_weight(schedule: dict, progress_pct: float) -> float:
    """Linear interpolation of schedule weights by percent progress.

    schedule: mapping of percent (0-100) to weight.
    progress_pct: current percent progress in [0, 100].
    """
    if not schedule:
        return 1.0
    # Normalize and sort points by percent
    points = sorted((float(k), float(v)) for k, v in schedule.items())
    # Clamp to endpoints
    if progress_pct <= points[0][0]:
        return points[0][1]
    if progress_pct >= points[-1][0]:
        return points[-1][1]
    # Find segment and interpolate
    for (k0, v0), (k1, v1) in zip(points, points[1:]):
        if k0 <= progress_pct <= k1:
            if k1 == k0:
                return v1
            t = (progress_pct - k0) / (k1 - k0)
            return v0 + t * (v1 - v0)
    # Fallback (should not hit due to clamps)
    return points[-1][1]


def play_population_game(training_agent: PPOAgent,
                        opponents: list,
                        partner_mode: int,
                        training_agent_position: int = 1,
                        shaping_weights: dict | None = None) -> tuple:
    """Play a single game with the training agent and population opponents.

    Returns:
        tuple: (game, episode_transitions, final_scores, training_agent_data)
    """
    game = Game(partner_selection_mode=partner_mode)
    weights = shaping_weights or {"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 1.0}

    # Reset recurrent states for all agents
    training_agent.reset_recurrent_state()
    for opponent in opponents:
        opponent.agent.reset_recurrent_state()

    # Create position-to-agent mapping; all five seats must be populated by training + 4 opponents
    agents = [None] * 5
    agents[training_agent_position - 1] = training_agent

    opponent_positions = [i for i in range(5) if i != training_agent_position - 1]
    for i, opponent in enumerate(opponents[:4]):
        agents[opponent_positions[i]] = opponent.agent

    # Store transitions only for the training agent
    episode_transitions = []
    current_trick_transitions = []

    # Map positions to population opponents for profile updates
    pos_to_pop_agent = {}
    opp_positions = [i for i in range(1, 6) if i != training_agent_position]
    for i, opp in enumerate(opponents[:len(opp_positions)]):
        pos_to_pop_agent[opp_positions[i]] = opp

    # Hand strength categories captured once at start
    hand_strength_by_pos = {p.position: estimate_hand_strength_category(p.hand) for p in game.players}

    while not game.is_done():
        for player in game.players:
            current_agent = agents[player.position - 1]
            valid_actions = player.get_valid_action_ids()

            while valid_actions:
                state = player.get_state_dict()

                # Get action from appropriate agent
                if current_agent == training_agent:
                    action, log_prob, value = current_agent.act(state, valid_actions, player.position)

                    # Store transition for training agent
                    transition = {
                        'kind': 'action',
                        'player': player,
                        'state': state,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy(),
                        'intermediate_reward': 0.0,
                        'secret_partner_label': 1.0 if player.is_secret_partner else 0.0,
                        'points_label': compute_known_points_rel(game, player),
                    }
                    episode_transitions.append(transition)

                    # Shared intermediate reward shaping and trick tracking (for training agent only)
                    update_intermediate_rewards_for_action(
                        game,
                        player,
                        action,
                        transition,
                        current_trick_transitions,
                        pick_weight=weights["pick"],
                        partner_weight=weights["partner"],
                        bury_weight=weights["bury"],
                        play_weight=weights["play"],
                    )

                else:
                    # Opponent action (stochastic for diversity)
                    action, _, _ = current_agent.act(state, valid_actions, player.position, deterministic=False)

                player.act(action)

                # Handle trick completion; PFSP-specific observation propagation
                trick_completed = handle_trick_completion(
                    game, current_trick_transitions
                )
                if trick_completed and not game.is_done():
                    # Emit observations for the completed trick using dedicated accessor
                    for seat in game.players:
                        seat_agent = agents[seat.position - 1]
                        if seat_agent == training_agent:
                            # Update training agent's recurrent hidden state and also store for unroll
                            training_agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
                            episode_transitions.append({
                                'kind': 'observation',
                                'player': seat,
                                'state': seat.get_state_dict(),
                            })
                        else:
                            seat_agent.observe(seat.get_last_trick_state_dict(), seat.position)
                    # Update trick-level EWMAs for population agents
                    pos_to_pop_agent_local = {}
                    for i, opp in enumerate(opponents[:len(opp_positions)]):
                        pos_to_pop_agent_local[opp_positions[i]] = opp
                    profile_trick_completion(game, pos_to_pop_agent_local)

                valid_actions = player.get_valid_action_ids()

                # --- Strategic profile updates for opponents only ---
                pop_agent = pos_to_pop_agent.get(player.position)
                if pop_agent:
                    profile_pop_agent_action(game, player, action, pop_agent, hand_strength_by_pos)

    final_scores = [player.get_score() for player in game.players]

    # Return training agent specific data
    training_agent_score = final_scores[training_agent_position - 1]
    was_picker = (game.picker == training_agent_position)

    training_agent_data = {
        'score': training_agent_score,
        'was_picker': was_picker,
        'position': training_agent_position
    }

    # Compute rewards for training agent actions
    reward_map = {}
    for reward_data in process_episode_rewards(
        [t for t in episode_transitions if t['kind'] == 'action'],
        final_scores,
        game.is_leaster
    ):
        reward_map[id(reward_data['transition'])] = reward_data['reward']

    # Build final episode event stream for storage
    episode_events = []
    for ev in episode_transitions:
        if ev['kind'] == 'observation':
            episode_events.append({
                'kind': 'observation',
                'state': ev['state'],
                'player_id': ev['player'].position,
            })
        else:
            seat_pos = ev['player'].position
            episode_events.append({
                'kind': 'action',
                'state': ev['state'],
                'action': ev['action'],
                'log_prob': ev['log_prob'],
                'value': ev['value'],
                'valid_actions': ev['valid_actions'],
                'reward': reward_map[id(ev)],
                'player_id': seat_pos,
                'win_label': 1.0 if final_scores[seat_pos - 1] > 0 else 0.0,
                'final_return_label': float(final_scores[seat_pos - 1]),
                'secret_partner_label': ev.get('secret_partner_label', 0.0),
                'points_label': ev.get('points_label', None),
            })

    return game, episode_events, final_scores, training_agent_data


def train_pfsp(num_episodes: int = 500000,
               update_interval: int = 2048,
               save_interval: int = 5000,
               strategic_eval_interval: int = 10000,
               population_add_interval: int = 5000,
               cross_eval_interval: int = 20000,
               resume_model: str = None,
               activation: str = 'swish',
               initial_checkpoints: list = None,
               hyperparams: PFSPHyperparams = DEFAULT_HYPERPARAMS):
    """
    PFSP training with population-based opponents.
    """
    print("🚀 Starting PFSP (Population-Based) Training...")
    print("="*80)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Population add interval: {population_add_interval}")
    print(f"  Cross-evaluation interval: {cross_eval_interval}")
    print(f"  Activation function: {activation.upper()}")
    print("  Opponent strategy: POPULATION-BASED (PFSP)")
    print("  Population management: OpenSkill ratings + diversity")
    print("="*80)

    # Create training agent
    training_agent = PPOAgent(len(ACTIONS),
                            lr_actor=1.5e-4,
                            lr_critic=1.5e-4,
                            activation=activation)

    # OpenSkill rating for the training agent
    rating_model = PlackettLuce()
    training_rating = rating_model.rating()
    training_agent_id = "training_agent"

    # Create population
    population = PFSPPopulation(
        max_population_jd=75,
        max_population_called_ace=75,
        population_dir="pfsp_population"
    )

    # Initialize population from checkpoints if provided
    if initial_checkpoints:
        create_initial_population_from_checkpoints(
            population,
            initial_checkpoints,
            activation=activation,
            max_agents_per_mode=10
        )

    # Load or create initial training agent
    start_episode = 0
    if resume_model:
        try:
            training_agent.load(resume_model, load_optimizers=True)
            print(f"✅ Loaded training agent from {resume_model}")
            if 'checkpoint_' in resume_model:
                start_episode = int(resume_model.split('_')[-1].split('.')[0])
                print(f"📍 Resuming from episode {start_episode:,}")
        except Exception as e:
            print(f"❌ Could not load {resume_model}: {e}")
    else:
        print("🆕 Starting fresh training agent")

    # Initialize tracking variables
    picker_scores = deque(maxlen=3000)
    pick_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    pass_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    leaster_window = deque(maxlen=3000)
    alone_call_window = deque(maxlen=3000)
    training_alone_window = deque(maxlen=3000)
    called_ace_window = deque(maxlen=3000)
    called_under_window = deque(maxlen=3000)
    called_10_window = deque(maxlen=3000)
    team_point_differences = deque(maxlen=3000)
    picker_window = deque(maxlen=3000)

    training_data = {
        'episodes': [],
        'picker_avg': [],
        'called_pick_rate': [],
        'jd_pick_rate': [],
        'learning_rate': [],
        'time_elapsed': [],
        'pick_hand_correlation': [],
        'picker_trump_rate': [],
        'defender_trump_rate': [],
        'bury_quality_rate': [],
        'team_point_diff': [],
        'alone_rate': [],
        'leaster_rate': [],
        'strategic_episodes': [],
        'population_stats': []
    }

    # Create checkpoint directory
    checkpoint_dir = f'pfsp_checkpoints_{activation}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # CSV log files for ongoing progress/metrics
    progress_csv = os.path.join(checkpoint_dir, 'pfsp_training_progress.csv')
    strategic_csv = os.path.join(checkpoint_dir, 'pfsp_strategic_metrics.csv')

    # If CSVs exist, preload them so training_data is continuous across resumes
    start_time_offset = 0.0
    if os.path.exists(progress_csv):
        with open(progress_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data['episodes'].append(int(row['episode']))
                training_data['picker_avg'].append(float(row['picker_avg']))
                training_data['called_pick_rate'].append(float(row['called_pick_rate']))
                training_data['jd_pick_rate'].append(float(row['jd_pick_rate']))
                training_data['learning_rate'].append(float(row['learning_rate']))
                training_data['time_elapsed'].append(float(row['time_elapsed']))
                training_data['team_point_diff'].append(float(row['team_point_diff']))
                # Optional historical series (may be missing in early CSVs)
                if 'alone_rate' in row and row['alone_rate'] != '':
                    training_data['alone_rate'].append(float(row['alone_rate']))
                if 'leaster_rate' in row and row['leaster_rate'] != '':
                    training_data['leaster_rate'].append(float(row['leaster_rate']))
        if training_data['time_elapsed']:
            start_time_offset = training_data['time_elapsed'][-1]

    if os.path.exists(strategic_csv):
        with open(strategic_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data['strategic_episodes'].append(int(row['episode']))
                training_data['pick_hand_correlation'].append(float(row['pick_hand_correlation']))
                training_data['picker_trump_rate'].append(float(row['picker_trump_rate']))
                training_data['defender_trump_rate'].append(float(row['defender_trump_rate']))
                training_data['bury_quality_rate'].append(float(row['bury_quality_rate']))

    start_time = time.time()
    game_count = 0
    # Track cumulative μ renormalisation shifts for logging
    cumulative_renorm = 0.0
    transitions_since_update = 0
    last_checkpoint_time = start_time

    bury_entropy_bump_until = 0
    partner_entropy_bump_until = 0
    pick_entropy_bump_until = 0

    # Partner CALL mixture epsilon controller state
    current_partner_call_eps = hyperparams.partner_call_eps_base
    training_agent.set_partner_call_epsilon(current_partner_call_eps)

    # PASS-floor epsilon controller state
    current_pass_floor_eps = hyperparams.pass_floor_eps_base
    training_agent.set_pass_floor_epsilon(current_pass_floor_eps)

    # PICK-floor epsilon controller state
    current_pick_floor_eps = hyperparams.pick_floor_eps_base
    training_agent.set_pick_floor_epsilon(current_pick_floor_eps)

    print(f"\n🎮 Beginning PFSP training... (target: {num_episodes:,} episodes)")
    print(population.get_population_summary())
    print("-" * 80)

    for episode in range(start_episode + 1, num_episodes + 1):
        # ------------------------------------------------------------------
        # Prevent numerical overflow in OpenSkill ratings by renormalising μ
        # ------------------------------------------------------------------
        MAX_ABS_MU = 350.0
        all_ratings = [training_rating] + [ag.rating for ag in population.jd_population] + [ag.rating for ag in population.called_ace_population]
        extreme_mu = max(abs(r.mu) for r in all_ratings) if all_ratings else 0
        if extreme_mu > MAX_ABS_MU:
            shift = extreme_mu - MAX_ABS_MU
            for r in all_ratings:
                r.mu -= np.sign(r.mu) * shift
            cumulative_renorm -= shift

        # Throttle log noise – print cumulative shift once every 2 000 episodes
        if episode % 2000 == 1:
            print(f"⚖️  Cumulative rating μ renorm Δ over last 2k eps: {cumulative_renorm:+.1f}  (|μ|max={extreme_mu:.1f})")
            cumulative_renorm = 0.0

        partner_mode = get_partner_selection_mode(episode)

        # Sample opponents from population based on current OpenSkill rating
        training_skill = training_rating.mu
        opponents = population.sample_opponents(
            partner_mode=partner_mode,
            n_opponents=4,
            training_agent_skill=training_skill,
            training_agent_id=training_agent_id,
            diversity_weight=0.35,
            curriculum_weight=0.4,
            exploitation_weight=0.1,
            uniform_mix=0.15,
        )

        # Require 4 opponents in population; exit script if not available
        if len(opponents) < 4:
            print(f"❌ Insufficient PFSP opponents for {get_partner_mode_name(partner_mode)} (need 4, got {len(opponents)}). Provide --initial-checkpoints or pre-populate pfsp_population.")
            sys.exit(1)

        # Randomly select training agent position (1-5)
        training_position = random.randint(1, 5)

        # Compute per-episode shaping weights from schedules
        progress_pct = min(100.0, max(0.0, (episode / num_episodes) * 100.0))
        shaping_weights = {
            "pick": interpolated_weight(hyperparams.shaping_schedule_pick, progress_pct),
            "partner": interpolated_weight(hyperparams.shaping_schedule_partner, progress_pct),
            "bury": interpolated_weight(hyperparams.shaping_schedule_bury, progress_pct),
            "play": interpolated_weight(hyperparams.shaping_schedule_play, progress_pct),
        }

        # Play game
        game, episode_events, final_scores, training_data_single = play_population_game(
            training_agent=training_agent,
            opponents=opponents,
            partner_mode=partner_mode,
            training_agent_position=training_position,
            shaping_weights=shaping_weights
        )

        training_agent.store_episode_events(episode_events)
        transitions_since_update += sum(1 for ev in episode_events if ev['kind'] == 'action')

        # Update statistics
        picker_score = training_data_single['score'] if training_data_single['was_picker'] else 0
        if training_data_single['was_picker']:
            picker_scores.append(picker_score)

        # Track how often the training agent is the picker (unconditional)
        picker_window.append(1 if training_data_single['was_picker'] else 0)

        # ---------- OpenSkill rating update (centralized in PFSPPopulation) ----------
        opponent_positions = [pos for pos in range(1, 6) if pos != training_position]
        position_to_agent = {}
        for idx, opp in enumerate(opponents[:len(opponent_positions)]):
            position_to_agent[opponent_positions[idx]] = opp

        training_rating = population.update_ratings_with_training(
            training_rating=training_rating,
            final_scores=final_scores,
            training_position=training_position,
            opponents_by_position=position_to_agent,
            picker_seat=game.picker,
            partner_seat=game.partner,
            is_leaster=game.is_leaster,
        )

        # Track team point differences
        if game.picker and not game.is_leaster:
            picker_team_points = game.get_final_picker_points()
            defender_team_points = game.get_final_defender_points()
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0
        team_point_differences.append(team_point_diff)

        # Final performance profiling per role for diversity/clustering
        for pos in opponent_positions:
            opp_agent = position_to_agent.get(pos)
            if opp_agent:
                if game.is_leaster:
                    opp_agent.update_strategic_profile_from_game({
                        'final_score': final_scores[pos - 1],
                        'role': 'leaster',
                    })
                else:
                    role_final = 'picker' if (game.picker == pos) else (
                        'partner' if getattr(game, 'is_partner_seat', lambda _pos: False)(pos) or getattr(game.players[pos-1], 'is_partner', False)
                        else 'defender'
                    )
                    opp_agent.update_strategic_profile_from_game({
                        'final_score': final_scores[pos - 1],
                        'role': role_final,
                    })

        # Track other statistics (similar to train_ppo.py)
        is_leaster_ep = 1 if game.is_leaster else 0
        leaster_window.append(is_leaster_ep)

        is_called_ace_ep = 1 if partner_mode == PARTNER_BY_CALLED_ACE else 0
        called_ace_window.append(is_called_ace_ep)

        if is_called_ace_ep and not is_leaster_ep:
            called_under_window.append(1 if game.is_called_under else 0)
            called_10_window.append(1 if (game.called_card and game.called_card.startswith("10")) else 0)
        elif is_called_ace_ep:
            called_under_window.append(0)
            called_10_window.append(0)

        # Track ALONE calls only for games with a picker (exclude leaster)
        if not game.is_leaster:
            # Global population ALONE indicator (any picker)
            alone_call_window.append(1 if game.alone_called else 0)
            # Training-agent-only ALONE indicator (only when training agent is picker)
            if training_data_single['was_picker']:
                training_alone_window.append(1 if game.alone_called else 0)

        # Count pick/pass decisions from transitions
        episode_picks = sum(
            1 for ev in episode_events
            if ev['kind'] == 'action' and ev['action'] == ACTION_IDS["PICK"]
        )
        episode_passes = sum(
            1 for ev in episode_events
            if ev['kind'] == 'action' and ev['action'] == ACTION_IDS["PASS"]
        )

        pick_decisions[partner_mode].append(episode_picks)
        pass_decisions[partner_mode].append(episode_passes)

        game_count += 1

        # Update model periodically
        if transitions_since_update >= update_interval:
            print(f"🔄 Updating model after {game_count} games... (Episode {episode:,})")

            # Entropy decay
            entropy_play_start, entropy_play_end = hyperparams.entropy_play_start, hyperparams.entropy_play_end
            entropy_pick_start, entropy_pick_end = hyperparams.entropy_pick_start, hyperparams.entropy_pick_end
            entropy_partner_start, entropy_partner_end = hyperparams.entropy_partner_start, hyperparams.entropy_partner_end
            entropy_bury_start, entropy_bury_end = hyperparams.entropy_bury_start, hyperparams.entropy_bury_end
            decay_fraction = min(episode / num_episodes, 1.0)
            training_agent.entropy_coeff_play = entropy_play_start + (entropy_play_end - entropy_play_start) * decay_fraction
            training_agent.entropy_coeff_pick = entropy_pick_start + (entropy_pick_end - entropy_pick_start) * decay_fraction
            training_agent.entropy_coeff_partner = entropy_partner_start + (entropy_partner_end - entropy_partner_start) * decay_fraction
            training_agent.entropy_coeff_bury = entropy_bury_start + (entropy_bury_end - entropy_bury_start) * decay_fraction


            # Apply temporary bumps to entropies
            if episode <= bury_entropy_bump_until:
                training_agent.entropy_coeff_bury += hyperparams.bury_entropy_bump
            if episode <= partner_entropy_bump_until:
                training_agent.entropy_coeff_partner += hyperparams.partner_entropy_bump
            if episode <= pick_entropy_bump_until:
                training_agent.entropy_coeff_pick += hyperparams.pick_entropy_bump

            # Update
            update_stats = training_agent.update(epochs=4, batch_size=256)

            if update_stats:
                adv_stats = update_stats['advantage_stats']
                val_stats = update_stats['value_target_stats']
                num_transitions = update_stats['num_transitions']
                approx_kl = update_stats.get('approx_kl', None)
                early_stop = update_stats.get('early_stop', False)
                head_entropy = update_stats.get('head_entropy', {})
                pick_pass_adv = update_stats.get('pick_pass_adv', {})

                print(f"   Transitions: {num_transitions}")
                print(f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
                print(f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")
                if approx_kl is not None:
                    print(f"   PPO KL: {approx_kl:.4f}  Early stop: {early_stop}")
                # Instrumentation logs: head entropy and PICK/PASS advantages
                if head_entropy:
                    print(
                        f"   Entropy - pick: {head_entropy.get('pick', 0.0):.3f}, "
                        f"partner: {head_entropy.get('partner', 0.0):.3f}, "
                        f"bury: {head_entropy.get('bury', 0.0):.3f}, "
                        f"play: {head_entropy.get('play', 0.0):.3f}"
                    )
                if pick_pass_adv:
                    print(
                        f"   Adv(PICK): {pick_pass_adv.get('pick_mean', 0.0):+.3f} (n={pick_pass_adv.get('pick_count', 0)}), "
                        f"Adv(PASS): {pick_pass_adv.get('pass_mean', 0.0):+.3f} (n={pick_pass_adv.get('pass_count', 0)})"
                    )
                if 'timing' in update_stats:
                    t = update_stats['timing']
                    print(
                        f"   Timing - build: {t['build_s']:.3f}s, forward: {t['forward_s']:.3f}s, "
                        f"backward: {t['backward_s']:.3f}s, step: {t['step_s']:.3f}s, total: {t['total_update_s']:.3f}s, "
                        f"opt_steps: {t['optimizer_steps']}"
                    )

            game_count = 0
            transitions_since_update = 0

        # Add agent to population periodically
        if episode % population_add_interval == 0:
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                # Snapshot training agent for population opponents
                agent_snapshot = copy.deepcopy(training_agent)
                # Disable CALL-uniform mixing for population agents
                agent_snapshot.set_partner_call_epsilon(hyperparams.partner_call_eps_base)
                # Disable PASS-floor mixing for population agents
                agent_snapshot.set_pass_floor_epsilon(hyperparams.pass_floor_eps_base)
                # Disable PICK-floor mixing for population agents
                agent_snapshot.set_pick_floor_epsilon(hyperparams.pick_floor_eps_base)

                agent_id = population.add_agent(
                    agent=agent_snapshot,
                    partner_mode=mode,
                    training_episodes=episode,
                    parent_id=None,
                    activation=activation,
                    # Seed snapshot rating with current training agent's rating (same μ/σ)
                    initial_rating=population.rating_model.rating(
                        mu=training_rating.mu,
                        sigma=training_rating.sigma,
                    )
                )
                print(f"👥 Added training agent snapshot to {get_partner_mode_name(mode)} population (ID: {agent_id})")

            # Log updated diversity stats after population change
            jd_div = population.get_diversity_stats(PARTNER_BY_JD)
            ca_div = population.get_diversity_stats(PARTNER_BY_CALLED_ACE)
            print("   Diversity (JD): avg=%.3f, spread=%.3f, clusters=%d, alone_rate_range=(%.2f, %.2f), pick_cv={weak: %.2f, med: %.2f, strong: %.2f}, coverage={early: %.2f, void: %.2f}" % (
                jd_div['avg_pairwise_diversity'], jd_div['diversity_spread'], jd_div['strategic_clusters'],
                jd_div['alone_rate_range'][0], jd_div['alone_rate_range'][1],
                jd_div['pick_rate_diversity']['weak'], jd_div['pick_rate_diversity']['medium'], jd_div['pick_rate_diversity']['strong'],
                jd_div['coverage']['early_leads'], jd_div['coverage']['void_events']))
            print("   Diversity (CA): avg=%.3f, spread=%.3f, clusters=%d, alone_rate_range=(%.2f, %.2f), pick_cv={weak: %.2f, med: %.2f, strong: %.2f}, coverage={early: %.2f, void: %.2f}" % (
                ca_div['avg_pairwise_diversity'], ca_div['diversity_spread'], ca_div['strategic_clusters'],
                ca_div['alone_rate_range'][0], ca_div['alone_rate_range'][1],
                ca_div['pick_rate_diversity']['weak'], ca_div['pick_rate_diversity']['medium'], ca_div['pick_rate_diversity']['strong'],
                ca_div['coverage']['early_leads'], ca_div['coverage']['void_events']))

        # Cross-evaluation periodically
        if episode % cross_eval_interval == 0:
            print(f"🏆 Running cross-evaluation tournaments... (Episode {episode:,})")
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                eval_stats = population.run_cross_evaluation(mode, num_games=40, max_agents=75)
                print(f"   {get_partner_mode_name(mode)}: {eval_stats['games_played']} games, "
                        f"avg skill: {eval_stats['avg_skill_after']:.1f}")
            # Rebuild clusters post-tournament to refresh anchors/sampling policies
            # Clustering is computed lazily when sampling; this call just forces recomputation for visibility
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                _ = population._cluster_population(mode)
            print(population.get_population_summary())
            print("-" * 80)

        # Strategic evaluation
        if episode % strategic_eval_interval == 0:
            print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
            strategic_metrics = analyze_strategic_decisions(training_agent, num_samples=200)

            training_data['strategic_episodes'].append(episode)
            training_data['pick_hand_correlation'].append(strategic_metrics['pick_hand_correlation'])
            training_data['picker_trump_rate'].append(strategic_metrics['picker_trump_rate'])
            training_data['defender_trump_rate'].append(strategic_metrics['defender_trump_rate'])
            training_data['bury_quality_rate'].append(strategic_metrics['bury_quality_rate'])

            # Append strategic metrics row to CSV (episode-keyed)
            write_header = (not os.path.exists(strategic_csv)) or (os.path.getsize(strategic_csv) == 0)
            with open(strategic_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode','pick_hand_correlation','picker_trump_rate','defender_trump_rate','bury_quality_rate'])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    'episode': episode,
                    'pick_hand_correlation': strategic_metrics['pick_hand_correlation'],
                    'picker_trump_rate': strategic_metrics['picker_trump_rate'],
                    'defender_trump_rate': strategic_metrics['defender_trump_rate'],
                    'bury_quality_rate': strategic_metrics['bury_quality_rate'],
                })

            print(f"   Pick-Hand Correlation: {strategic_metrics['pick_hand_correlation']:.3f}")
            print(f"   Picker Trump Rate: {strategic_metrics['picker_trump_rate']:.1f}%")
            print(f"   Defender Trump Rate: {strategic_metrics['defender_trump_rate']:.1f}%")
            print(f"   Bury Quality Rate: {strategic_metrics['bury_quality_rate']:.1f}%")

            # --- Check bury quality rate and schedule bury-head entropy bump if too low ---
            current_bury_quality_rate = strategic_metrics['bury_quality_rate']
            if (
                current_bury_quality_rate < hyperparams.low_bury_quality_threshold and episode > bury_entropy_bump_until
            ):
                bury_entropy_bump_until = episode + hyperparams.bury_entropy_bump_duration
                print(f"   ⚠️  Low bury-quality rate detected ({current_bury_quality_rate:.1f}%). Increasing bury entropy by {hyperparams.bury_entropy_bump:.3f} for the next {hyperparams.bury_entropy_bump_duration:,} episodes.")
            if (
                current_bury_quality_rate >= hyperparams.low_bury_quality_threshold and
                episode > bury_entropy_bump_until and
                bury_entropy_bump_until != 0
            ):
                # Reset any expired bump marker to reduce log noise later
                bury_entropy_bump_until = 0

        # Progress reporting
        if episode % 1000 == 0:
            current_avg_picker_score = np.mean(picker_scores) if picker_scores else 0

            # Calculate pick rates
            total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
            total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
            total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
            total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
            current_called_pick_rate = (100 * total_called_picks / (total_called_picks + total_called_passes)) if (total_called_picks + total_called_passes) > 0 else 0
            current_jd_pick_rate = (100 * total_jd_picks / (total_jd_picks + total_jd_passes)) if (total_jd_picks + total_jd_passes) > 0 else 0
            current_team_diff = np.mean(team_point_differences) if team_point_differences else 0

            # Rolling window rates
            current_leaster_rate = (sum(leaster_window) / len(leaster_window)) * 100 if leaster_window else 0
            current_alone_rate = (sum(alone_call_window) / len(alone_call_window)) * 100 if alone_call_window else 0
            current_training_alone_rate = (sum(training_alone_window) / len(training_alone_window)) * 100 if training_alone_window else 0
            ca_denominator = sum(called_ace_window) or 1
            current_called_under_rate = (sum(called_under_window) / ca_denominator) * 100
            current_called_10s_rate = (sum(called_10_window) / ca_denominator) * 100
            current_picker_frequency = (sum(picker_window) / len(picker_window)) * 100 if picker_window else 0
            elapsed = start_time_offset + (time.time() - start_time)

            # Collect data for plotting
            training_data['episodes'].append(episode)
            training_data['picker_avg'].append(current_avg_picker_score)
            training_data['called_pick_rate'].append(current_called_pick_rate)
            training_data['jd_pick_rate'].append(current_jd_pick_rate)
            training_data['learning_rate'].append(training_agent.actor_optimizer.param_groups[0]['lr'])
            training_data['time_elapsed'].append(elapsed)
            training_data['team_point_diff'].append(current_team_diff)
            training_data['alone_rate'].append(current_alone_rate)
            training_data['leaster_rate'].append(current_leaster_rate)

            # Append progress row to CSV (create with header if new)
            write_header = (not os.path.exists(progress_csv)) or (os.path.getsize(progress_csv) == 0)
            with open(progress_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode','picker_avg','called_pick_rate','jd_pick_rate','learning_rate','time_elapsed','team_point_diff','alone_rate','leaster_rate'])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    'episode': episode,
                    'picker_avg': current_avg_picker_score,
                    'called_pick_rate': current_called_pick_rate,
                    'jd_pick_rate': current_jd_pick_rate,
                    'learning_rate': training_agent.actor_optimizer.param_groups[0]['lr'],
                    'time_elapsed': elapsed,
                    'team_point_diff': current_team_diff,
                    'alone_rate': current_alone_rate,
                    'leaster_rate': current_leaster_rate,
                })

            # Population statistics
            jd_stats = population.get_population_stats(PARTNER_BY_JD)
            ca_stats = population.get_population_stats(PARTNER_BY_CALLED_ACE)
            training_data['population_stats'].append({
                'episode': episode,
                'jd_size': jd_stats['size'],
                'jd_avg_skill': jd_stats['avg_skill'],
                'ca_size': ca_stats['size'],
                'ca_avg_skill': ca_stats['avg_skill']
            })

            games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

            print(f"📊 Episode {episode:,}/{num_episodes:,} ({episode/num_episodes*100:.1f}%)")
            print("   " + "-" * 50)
            print(f"   Picker avg: {current_avg_picker_score:+.3f}")
            print(f"   Team point diff: {current_team_diff:+.1f}")
            print(f"   Called Ace Pick rate: {current_called_pick_rate:.1f}%")
            print(f"   JD Pick rate: {current_jd_pick_rate:.1f}%")
            print(f"   Picker Frequency: {current_picker_frequency:.2f}%")
            print("   " + "-" * 25)
            print(f"   Alone Call Rate (population): {current_alone_rate:.2f}%")
            print(f"   Alone Call Rate (training): {current_training_alone_rate:.2f}%")
            print(f"   Called Under Rate: {current_called_under_rate:.2f}%")
            print(f"   Called 10s Rate: {current_called_10s_rate:.2f}%")
            print(f"   Leaster Rate: {current_leaster_rate:.2f}%")
            print("   " + "-" * 25)
            print(f"   Population JD: {jd_stats['size']} agents (avg skill: {jd_stats['avg_skill']:.1f})")
            print(f"   Population CA: {ca_stats['size']} agents (avg skill: {ca_stats['avg_skill']:.1f})")
            print("   " + "-" * 50)
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed/60:.1f} min")
            print("   " + "-" * 50)

            # --- Adaptive partner-head entropy bump scheduling ---
            if (
                (current_training_alone_rate > hyperparams.high_alone_rate_threshold or
                 current_training_alone_rate < hyperparams.low_alone_rate_threshold) and
                episode > partner_entropy_bump_until
            ):
                partner_entropy_bump_until = episode + hyperparams.partner_entropy_bump_duration
                print(
                    f"   ⚠️  ALONE rate out of band ({current_training_alone_rate:.2f}%). "
                    f"Increasing partner entropy by {hyperparams.partner_entropy_bump:.3f} for the next "
                    f"{hyperparams.partner_entropy_bump_duration:,} episodes."
                )
            if (
                (hyperparams.low_alone_rate_threshold <= current_training_alone_rate <= hyperparams.high_alone_rate_threshold) and
                episode > partner_entropy_bump_until and
                partner_entropy_bump_until != 0
            ):
                partner_entropy_bump_until = 0

            # --- Partner CALL mixture epsilon controller ---
            # Gradually increase ε when ALONE rate is high and picker avg is poor.
            # Tiered caps: <= mid_picker_avg -> mid cap; <= high_picker_avg -> high cap. Otherwise, decay toward base.
            desired_partner_eps_max = hyperparams.partner_call_eps_base
            # Gate epsilon scheduling on training-agent-only ALONE rate
            if current_training_alone_rate > hyperparams.high_alone_rate_ceiling:
                if current_avg_picker_score <= hyperparams.partner_call_eps_high_picker_avg_threshold:
                    desired_partner_eps_max = hyperparams.partner_call_eps_max_high
                elif current_avg_picker_score <= hyperparams.partner_call_eps_mid_picker_avg_threshold:
                    desired_partner_eps_max = hyperparams.partner_call_eps_max_mid
                else:
                    desired_partner_eps_max = hyperparams.partner_call_eps_base
            else:
                desired_partner_eps_max = hyperparams.partner_call_eps_base

            desired_partner_eps = current_partner_call_eps
            if desired_partner_eps_max > current_partner_call_eps:
                desired_partner_eps = min(current_partner_call_eps + hyperparams.partner_call_eps_step_up, desired_partner_eps_max)
            elif desired_partner_eps_max < current_partner_call_eps:
                desired_partner_eps = max(current_partner_call_eps - hyperparams.partner_call_eps_step_down, desired_partner_eps_max)

            if abs(desired_partner_eps - current_partner_call_eps) > 1e-6:
                current_partner_call_eps = desired_partner_eps
                training_agent.set_partner_call_epsilon(current_partner_call_eps)
                print(f"   ⚠️  Partner CALL epsilon ε adjusted to: {current_partner_call_eps:.3f}")

            # --- Adaptive pick-head entropy bump scheduling ---
            overall_picks = total_called_picks + total_jd_picks
            overall_decisions = overall_picks + total_called_passes + total_jd_passes
            overall_pick_rate = (100 * overall_picks / overall_decisions) if overall_decisions > 0 else 0.0

            if (
                (overall_pick_rate > hyperparams.high_pick_rate_threshold or overall_pick_rate < hyperparams.low_pick_rate_threshold)
                and episode > pick_entropy_bump_until
            ):
                pick_entropy_bump_until = episode + hyperparams.pick_entropy_bump_duration
                print(
                    f"   ⚠️  Pick rate out of band ({overall_pick_rate:.2f}%). "
                    f"Increasing pick entropy by {hyperparams.pick_entropy_bump:.3f} for the next "
                    f"{hyperparams.pick_entropy_bump_duration:,} episodes."
                )
            if (
                (hyperparams.low_pick_rate_threshold <= overall_pick_rate <= hyperparams.high_pick_rate_threshold) and
                episode > pick_entropy_bump_until and
                pick_entropy_bump_until != 0
            ):
                pick_entropy_bump_until = 0

            # --- PASS-floor epsilon controller (activate when pick rate high and picker avg negative) ---
            desired_pass_eps_max = hyperparams.pass_floor_eps_base
            if (overall_pick_rate > hyperparams.high_pick_rate_ceiling) and (current_avg_picker_score < hyperparams.pass_floor_eps_picker_avg_threshold):
                desired_pass_eps_max = hyperparams.pass_floor_eps_target

            desired_pass_eps = current_pass_floor_eps
            if desired_pass_eps_max > current_pass_floor_eps:
                desired_pass_eps = min(current_pass_floor_eps + hyperparams.pass_floor_eps_step_up, desired_pass_eps_max)
            elif desired_pass_eps_max < current_pass_floor_eps:
                desired_pass_eps = max(current_pass_floor_eps - hyperparams.pass_floor_eps_step_down, desired_pass_eps_max)

            if abs(desired_pass_eps - current_pass_floor_eps) > 1e-6:
                current_pass_floor_eps = desired_pass_eps
                training_agent.set_pass_floor_epsilon(current_pass_floor_eps)
                print(f"   ⚠️  PASS floor epsilon ε_pass adjusted to: {current_pass_floor_eps:.3f}")

            # --- PICK-floor epsilon controller (activate when overall pick rate is very low) ---
            desired_pick_eps_max = hyperparams.pick_floor_eps_base
            if overall_pick_rate < hyperparams.low_pick_rate_floor:
                desired_pick_eps_max = hyperparams.pick_floor_eps_target

            desired_pick_eps = current_pick_floor_eps
            if desired_pick_eps_max > current_pick_floor_eps:
                desired_pick_eps = min(current_pick_floor_eps + hyperparams.pick_floor_eps_step_up, desired_pick_eps_max)
            elif desired_pick_eps_max < current_pick_floor_eps:
                desired_pick_eps = max(current_pick_floor_eps - hyperparams.pick_floor_eps_step_down, desired_pick_eps_max)

            if abs(desired_pick_eps - current_pick_floor_eps) > 1e-6:
                current_pick_floor_eps = desired_pick_eps
                training_agent.set_pick_floor_epsilon(current_pick_floor_eps)
                print(f"   ⚠️  PICK floor epsilon ε_pick adjusted to: {current_pick_floor_eps:.3f}")


        # Save checkpoints
        if episode % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/pfsp_{activation}_checkpoint_{episode}.pt'
            training_agent.save(checkpoint_path)

            # Save population state
            population.save_population_state()

            # Save training plot
            if len(training_data['episodes']) > 10:
                plot_path = f'{checkpoint_dir}/pfsp_training_progress_{episode}.png'
                save_training_plot(training_data, plot_path)  # Reuse plotting function

            checkpoint_time = time.time()
            time_since_last = checkpoint_time - last_checkpoint_time
            last_checkpoint_time = checkpoint_time

            print(f"💾 Checkpoint saved at episode {episode:,}")
            print(f"   Time for last {save_interval:,} episodes: {time_since_last/60:.1f} min")
            remaining_episodes = num_episodes - episode
            if remaining_episodes > 0:
                estimated_time = remaining_episodes * (time_since_last / save_interval) / 60
                print(f"   Estimated time remaining: {estimated_time:.1f} min")

    # Final update and save
    if training_agent.events:
        print("🔄 Final model update...")
        final_update_stats = training_agent.update()

        if final_update_stats:
            adv_stats = final_update_stats['advantage_stats']
            val_stats = final_update_stats['value_target_stats']
            num_transitions = final_update_stats['num_transitions']

            print(f"   Final Transitions: {num_transitions}")
            print(f"   Final Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
            print(f"   Final Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")

    training_agent.save(f'final_pfsp_{activation}_ppo.pt')
    population.save_population_state()

    # Save final training plot
    if len(training_data['episodes']) > 0:
        save_training_plot(training_data, f'final_pfsp_{activation}_training.png')

    total_time = time.time() - start_time
    print("\n🎉 PFSP Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}")
    print(f"   Final team point difference: {np.mean(team_point_differences) if team_point_differences else 0:.1f}")

    # Final population summary
    print("\n" + population.get_population_summary())


def main():
    parser = ArgumentParser(description="PFSP population-based training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=5000000,
                       help="Number of training episodes (default: 5,000,000)")
    parser.add_argument("--update-interval", type=int, default=2048,
                       help="Number of transitions between model updates")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Number of episodes between checkpoints")
    parser.add_argument("--strategic-eval-interval", type=int, default=10000,
                       help="Number of episodes between strategic evaluations")
    parser.add_argument("--population-add-interval", type=int, default=5000,
                       help="Number of episodes between adding agents to population")
    parser.add_argument("--cross-eval-interval", type=int, default=20000,
                       help="Number of episodes between cross-evaluation tournaments")
    parser.add_argument("--resume", type=str, default=None,
                       help="Model file to resume training from")
    parser.add_argument("--activation", type=str, default='swish', choices=['relu', 'swish'],
                       help="Activation function to use (default: swish)")
    parser.add_argument("--initial-checkpoints", nargs='+', default=None,
                       help="Checkpoint patterns to initialize population from")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend('Agg')

    train_pfsp(
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        save_interval=args.save_interval,
        strategic_eval_interval=args.strategic_eval_interval,
        population_add_interval=args.population_add_interval,
        cross_eval_interval=args.cross_eval_interval,
        resume_model=args.resume,
        activation=args.activation,
        initial_checkpoints=args.initial_checkpoints
    )


if __name__ == "__main__":
    main()
```

```python
*** Begin Patch
*** Update File: /Volumes/Nargothrond/dev/sheepsheadai/ppo.py
@@
 class RecurrentCriticNetwork(nn.Module):
     """Critic head using encoder features directly."""
@@
         self.critic_adapter = nn.Sequential(
             nn.LayerNorm(256),
             nn.Linear(256, 256),
             nn.ReLU() if activation == 'relu' else nn.SiLU()
         )
         self.value_head = nn.Linear(256, 1)
         # Auxiliary heads
         self.win_head = nn.Linear(256, 1)
         self.return_head = nn.Linear(256, 1)
         self.secret_partner_head = nn.Linear(256, 1)
+        # Per-player point prediction head (relative seating order, length-5 vector)
+        self.points_head = nn.Linear(256, 5)
@@
         # Auxiliary loss coefficients
         self.win_loss_coeff = 0.1
         self.return_loss_coeff = 0.1
         self.secret_loss_coeff = 0.07
+        self.points_loss_coeff = 0.1
@@
                 'state': ev['state'],
                 'action': int(ev['action']) - 1,
                 'reward': float(ev['reward']),
                 'value': float(ev['value']),
                 'log_prob': float(ev['log_prob']),
                 'done': (idx == last_action_idx),
                 'player_id': pid,
                 'win': float(ev.get('win_label', 0.0) or 0.0),
                 'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
                 'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
+                'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
             })
*** End Patch
```

I’ll now update the critic to add the points head, extend event storage and batching to carry per-player point labels, wire the new head through the vectorized forward pass, and add the auxiliary loss term into PPO’s update loop.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Dict
from sheepshead import ACTION_IDS, BURY_ACTIONS, CALL_ACTIONS, UNDER_ACTIONS, PLAY_ACTIONS, DECK_IDS, UNDER_TOKEN
from encoder import CardReasoningEncoder, CardEmbeddingConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network consuming encoder features, with separate heads for
    pick / partner-selection / bury / play phases. The four heads' logits are
    concatenated back into the full action space order so existing masking
    logic continues to work unchanged.
    """

    def __init__(self,
                 action_size,
                 action_groups,
                 activation='swish',
                 *,
                 d_card: int,
                 d_token: int,
                 map_cid_to_play_action_index: torch.Tensor,
                 map_cid_to_bury_action_index: torch.Tensor,
                 map_cid_to_under_action_index: torch.Tensor,
                 call_action_global_indices: torch.Tensor,
                 call_card_ids: torch.Tensor,
                 play_under_action_index: int):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'

        # Shared actor adapter to add nonlinearity and specialization before heads
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
        )

        # === Heads ===
        self.pick_head = nn.Linear(256, len(action_groups['pick']))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(256, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(256, 1)

        # Per-head temperatures (τ): logits will be divided by τ before softmax
        # τ > 1.0 softens (higher entropy), τ < 1.0 sharpens. Defaults to 1.0.
        self.temperature_pick = 1.0
        self.temperature_partner = 1.0
        self.temperature_bury = 1.0
        self.temperature_play = 1.0

        # Init (generic weights)
        self.apply(self._init_weights)

        # ---- Card-conditioned configuration (dims + mappings) ----
        self._d_card = int(d_card)
        self._d_token = int(d_token)
        # Pointer scorer (Bahdanau style)
        self.pointer_hidden = 64
        self.pointer_Wg = nn.Linear(256, self.pointer_hidden)
        self.pointer_Wt = nn.Linear(self._d_token, self.pointer_hidden)
        self.pointer_v = nn.Linear(self.pointer_hidden, 1, bias=False)
        # Two-tower (card CALL scoring)
        self.tw_latent = 64
        self.tw_Wg = nn.Linear(256, self.tw_latent)
        self.tw_We = nn.Linear(self._d_card, self.tw_latent)

        # Action index mappings
        self._map_cid_to_play_action_index = map_cid_to_play_action_index.clone().long()
        self._map_cid_to_bury_action_index = map_cid_to_bury_action_index.clone().long()
        self._map_cid_to_under_action_index = map_cid_to_under_action_index.clone().long()
        self._call_action_global_indices = call_action_global_indices.clone().long()
        self._call_card_ids = call_card_ids.clone().long()
        self._play_under_action_index = int(play_under_action_index)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------ internal helpers ------------------------
    def _score_cards_two_tower(self, feat: torch.Tensor, card_embedding: nn.Embedding) -> torch.Tensor:
        """Return per-card scores S for all 34 card ids using two-tower scorer.
        feat: (B, 256)
        Returns: (B, 34)
        """
        q = self.tw_Wg(feat)                      # (B, k)        <- tower 1 (actor features)
        table = card_embedding.weight             # (34, d_card)
        K = self.tw_We(table)                     # (34, k)       <- tower 2 (card embeddings)
        S = torch.matmul(q, K.t())    # (B, 34)       <- similarity scores
        return S

    def _score_hand_pointer(
        self,
        feat: torch.Tensor,
        hand_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score each hand slot using pointer scorer.
        feat: (B, 256); hand_tokens: (B, N, d_token)
        Returns: s_slots (B, N)
        """
        tok = hand_tokens
        B, N = tok.size(0), tok.size(1)
        g = self.pointer_Wg(feat).unsqueeze(1).expand(B, N, -1)  # (B, N, h)
        t = self.pointer_Wt(tok)                                   # (B, N, h)
        e = torch.tanh(g + t)                                      # (B, N, h)
        s = self.pointer_v(e).squeeze(-1)                          # (B, N)
        return s

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        hand_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        hand_tokens : Tensor (K, N, d_token)
        action_mask : BoolTensor (K, A) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # Adapt features
        feat = self.actor_adapter(actor_features)

        # PICK / PASS
        pick_logits = self.pick_head(feat) / max(self.temperature_pick, 1e-6)
        logits[:, self.action_groups['pick']] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(feat) / max(self.temperature_partner, 1e-6)
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(feat, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)] / max(self.temperature_partner, 1e-6)
        logits[:, self._call_action_global_indices.to(device)] = call_scores

        # Pointer scores over hand tokens (compute once, reuse for bury/under/play)
        slot_scores = self._score_hand_pointer(
            feat,
            hand_tokens,
        )  # (K, N)
        K_, N = slot_scores.size(0), slot_scores.size(1)
        cids = hand_ids.long()

        # BURY and UNDER scatter
        idx_bury = self._map_cid_to_bury_action_index.to(device)[cids]   # (K, N)
        idx_under = self._map_cid_to_under_action_index.to(device)[cids] # (K, N)
        for i in range(N):
            b_idx = idx_bury[:, i]
            u_idx = idx_under[:, i]
            valid_b = b_idx.ge(0)
            valid_u = u_idx.ge(0)
            if valid_b.any():
                logits.view(K_, -1)[valid_b, b_idx[valid_b]] = slot_scores[valid_b, i] / max(self.temperature_bury, 1e-6)
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[valid_u, i] / max(self.temperature_bury, 1e-6)

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[valid_p, i] / max(self.temperature_play, 1e-6)

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(feat).squeeze(-1) / max(self.temperature_play, 1e-6)
            logits[:, self._play_under_action_index] = play_under_logit

        # Apply action mask if provided
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def set_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Set per-head softmax temperatures.

        Parameters
        ----------
        pick, partner, bury, play : float | None
            If provided, sets the corresponding head's temperature τ. Values are
            clamped to a small positive minimum to avoid divide-by-zero.
        """
        eps = 1e-6
        if pick is not None:
            self.temperature_pick = max(float(pick), eps)
        if partner is not None:
            self.temperature_partner = max(float(partner), eps)
        if bury is not None:
            self.temperature_bury = max(float(bury), eps)
        if play is not None:
            self.temperature_play = max(float(play), eps)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features' and 'hand_tokens'
        action_mask : Bool Tensor (batch, action_size)
        hand_ids : Tensor (batch, 8)
        card_embedding : nn.Embedding

        Returns
        -------
        probs : Tensor (batch, action_size)
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_with_logits(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass that also returns pre-softmax logits.

        Returns
        -------
        probs : Tensor
        logits : Tensor
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs, logits



class RecurrentCriticNetwork(nn.Module):
    """Critic head using encoder features directly."""

    def __init__(self, activation='swish'):
        super().__init__()

        self.critic_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU()
        )
        self.value_head = nn.Linear(256, 1)
        # Auxiliary heads
        self.win_head = nn.Linear(256, 1)
        self.return_head = nn.Linear(256, 1)
        self.secret_partner_head = nn.Linear(256, 1)
        # Per-player point prediction head (relative seating order, length-5 vector)
        self.points_head = nn.Linear(256, 5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_out: Dict[str, torch.Tensor]):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        value : Tensor (batch, 1)
        """
        feat = self.critic_adapter(encoder_out['features'])
        value = self.value_head(feat)
        return value

    def aux_predictions(self, encoder_out: Dict[str, torch.Tensor]):
        """Return auxiliary predictions as scalars: (win_prob, expected_final_return, secret_partner_prob).

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        win_prob : float
            Sigmoid of win logits
        expected_final_return : float
            Linear head output for expected final return
        secret_partner_prob : float
            Sigmoid of secret partner logits
        """
        with torch.no_grad():
            aux_feat = self.critic_adapter(encoder_out['features'])
            win_logit = self.win_head(aux_feat).squeeze(-1)
            expected_return = self.return_head(aux_feat).squeeze(-1)
            secret_logit = self.secret_partner_head(aux_feat).squeeze(-1)
        win_prob_t = torch.sigmoid(win_logit)
        win_prob = float(win_prob_t.item())
        expected_final = float(expected_return.item())
        secret_prob = float(torch.sigmoid(secret_logit).item())
        return win_prob, expected_final, secret_prob


class PPOAgent:
    def __init__(self, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
        # Dict-based encoder → fixed 256-d features
        self.state_size = 256
        self.action_size = action_size

        # --------------------------------------------------
        # Action groups for multi–head policy
        # --------------------------------------------------
        pick_indices = [ACTION_IDS["PICK"] - 1, ACTION_IDS["PASS"] - 1]

        # Partner-selection head: ALONE, JD PARTNER, all partner CALL actions
        partner_selection_actions = ["ALONE", "JD PARTNER"] + CALL_ACTIONS
        partner_indices = sorted({ACTION_IDS[a] - 1 for a in partner_selection_actions})

        # Bury head: explicit bury actions and UNDER actions
        bury_indices = sorted({ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + UNDER_ACTIONS)})

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            'pick': sorted(pick_indices),
            'partner': sorted(partner_indices),
            'bury': sorted(bury_indices),
            'play': sorted(play_indices),
        }

        # Explicit indices for PICK and PASS actions (0-indexed)
        self.pick_action_index = ACTION_IDS["PICK"] - 1
        self.pass_action_index = ACTION_IDS["PASS"] - 1

        # Encoder with memory
        self.encoder = CardReasoningEncoder(card_config=CardEmbeddingConfig()).to(device)

        # Per-player memory tracking
        self._player_memories = {}

        # Build mappings before constructing actor
        (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            call_action_global_indices,
            call_card_ids,
            play_under_action_index,
        ) = self._build_action_index_mappings()

        self.actor = MultiHeadRecurrentActorNetwork(
            action_size,
            self.action_groups,
            activation=activation,
            d_card=self.encoder.d_card_dim,
            d_token=self.encoder.d_token_dim,
            map_cid_to_play_action_index=map_cid_to_play_action_index,
            map_cid_to_bury_action_index=map_cid_to_bury_action_index,
            map_cid_to_under_action_index=map_cid_to_under_action_index,
            call_action_global_indices=call_action_global_indices,
            call_card_ids=call_card_ids,
            play_under_action_index=play_under_action_index,
        ).to(device)

        self.critic = RecurrentCriticNetwork(activation=activation).to(device)

        # Optimizers (include encoder params with scaled LR for card embeddings)
        encoder_groups = self.encoder.param_groups(base_lr=lr_actor, card_lr_scale=0.2)
        actor_groups = [
            {'params': self.actor.parameters(), 'lr': lr_actor},
            *encoder_groups,
        ]
        self.actor_optimizer = optim.Adam(actor_groups)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.95
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_partner = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.3
        self.clip_epsilon_pick = 0.20
        self.clip_epsilon_partner = 0.25
        self.clip_epsilon_bury = 0.15
        self.clip_epsilon_play = 0.20
        self.value_clip_epsilon = 0.20

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.0

        # Auxiliary loss coefficients
        self.win_loss_coeff = 0.1
        self.return_loss_coeff = 0.1
        self.secret_loss_coeff = 0.07
        self.points_loss_coeff = 0.1

        # Storage for trajectory data
        self.reset_storage()

        # --------------------------------------------------
        # Partner-call mixture: epsilon floor over CALL actions
        # --------------------------------------------------
        # Subset of partner head actions eligible for CALL-uniform mixing
        self.partner_call_subindices = sorted(
            {ACTION_IDS["JD PARTNER"] - 1} | {ACTION_IDS[a] - 1 for a in CALL_ACTIONS}
        )
        # Blending coefficient ε (0 disables; typical 0.02–0.10)
        self.partner_call_epsilon = 0.0

        # PASS-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pass_floor_epsilon = 0.0
        # PICK-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pick_floor_epsilon = 0.0

    def get_recurrent_memory(self, player_id: int | None, device: torch.device | None = None) -> torch.Tensor:
        """Return the recurrent memory vector for a player, or a zero vector if unset.

        The returned tensor is (256,) on the requested device.
        """
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mem = self._player_memories.get(player_id) if player_id is not None else None
        if mem is None:
            return torch.zeros(256, device=target_device)
        return mem.to(target_device) if mem.device != target_device else mem

    def set_recurrent_memory(self, player_id: int | None, memory_vector: torch.Tensor) -> None:
        """Set the recurrent memory vector for a player."""
        self._player_memories[player_id] = memory_vector

    def _build_action_index_mappings(self):
        """Precompute global action index mappings for card-specific actions."""
        map_cid_to_play_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_bury_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_under_action_index = torch.full((34,), -1, dtype=torch.long)

        # PLAY actions
        play_under_index = None
        for a in PLAY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            if a == f"PLAY {UNDER_TOKEN}":
                play_under_index = a_idx
                continue
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_play_action_index[cid] = a_idx

        # BURY actions
        for a in BURY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_bury_action_index[cid] = a_idx

        # UNDER actions (choose card from hand to set as under)
        for a in UNDER_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_under_action_index[cid] = a_idx

        # CALL actions
        idx_call_global_list = []
        call_card_ids_list = []
        for a in CALL_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            parts = a.split()
            # a is either "CALL X" or "CALL X UNDER"
            card = parts[1]
            cid = DECK_IDS[card]
            idx_call_global_list.append(a_idx)
            call_card_ids_list.append(cid)

        idx_call_global = torch.tensor(sorted(idx_call_global_list), dtype=torch.long)
        # Align cid list to the sorted order of indices
        # Build mapping from index to cid then reorder
        idx_to_cid = {idx: cid for idx, cid in zip(idx_call_global_list, call_card_ids_list)}
        call_card_ids = torch.tensor([idx_to_cid[int(x)] for x in idx_call_global.tolist()], dtype=torch.long)

        return map_cid_to_play_action_index, map_cid_to_bury_action_index, map_cid_to_under_action_index, idx_call_global, call_card_ids, play_under_index

    def set_head_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Convenience proxy to set per-head temperatures on the actor."""
        self.actor.set_temperatures(pick=pick, partner=partner, bury=bury, play=play)

    def set_partner_call_epsilon(self, eps: float):
        """Set ε for uniform CALL mixture on partner steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.partner_call_epsilon = eps

    def set_pass_floor_epsilon(self, eps: float):
        """Set ε for PASS-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pass_floor_epsilon = eps

    def set_pick_floor_epsilon(self, eps: float):
        """Set ε for PICK-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pick_floor_epsilon = eps

    def _apply_epsilon_mixing(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply partner-call ε mixture and PASS-floor ε mixture to probability
        distributions in a batched way without in-place ops.

        Parameters
        ----------
        probs : Tensor
            Shape (B, A) probabilities over actions
        mask : Bool Tensor
            Shape (B, A) of valid actions
        """
        mixed = probs
        if self.partner_call_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            call_vec = torch.zeros(A, dtype=torch.bool, device=mixed.device)
            call_idx = torch.tensor(self.partner_call_subindices, dtype=torch.long, device=mixed.device)
            call_vec[call_idx] = True
            valid_call = mask & call_vec.view(1, A).expand(B, A)
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(
                has,
                valid_call.float() / count.clamp_min(1.0),
                torch.zeros_like(mixed),
            )
            eps = torch.where(
                has,
                torch.full_like(count, self.partner_call_epsilon),
                torch.zeros_like(count),
            )
            mixed = (1.0 - eps) * mixed + eps * ucall

        if self.pick_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pick = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand(B, A)
            valid_pick = mask[:, self.pick_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * mixed + self.pick_floor_epsilon * one_hot_pick,
                mixed,
            )

        if self.pass_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pass = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand(B, A)
            valid_pass = mask[:, self.pass_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * mixed + self.pass_floor_epsilon * one_hot_pass,
                mixed,
            )

        return mixed

    def reset_storage(self):
        # Ordered list of events: each is a dict with keys:
        # kind: 'action' | 'obs'
        # state: np.array state vector
        # mask: torch.bool mask or convertible
        # If kind == 'action': action (0-indexed), reward, value, log_prob, done
        # GAE writes back: 'advantage' and 'return'
        self.events = []

    def get_action_mask(self, valid_actions, action_size):
        """Convert valid actions set to boolean mask"""
        mask = torch.zeros(action_size, dtype=torch.bool)
        for action in valid_actions:
            mask[action - 1] = True  # Convert from 1-indexed to 0-indexed
        return mask

    def reset_recurrent_state(self):
        """Clear memory states (call at the start of every new game)."""
        self._player_memories = {}

    def get_action_probs_with_logits(self, state, valid_actions, player_id=None):
        """Return post-mixture action probabilities and pre-mix logits for a single dict state.

        Applies partner CALL-uniform mixture if enabled. Keeps PPO on-policy by
        exposing the same transformed distribution that sampling uses.
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        # Encode dict state with memory
        encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

        # Store updated memory
        if player_id is not None:
            self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

        action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
        hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

        with torch.no_grad():
            probs, logits = self.actor.forward_with_logits(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )
            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                probs = self._apply_epsilon_mixing(probs, action_mask_t)

        return probs, logits

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        with torch.no_grad():
            # Get or init memory for this player
            memory_in = self.get_recurrent_memory(player_id, device=device)

            # Encode with memory
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

            # Get action probabilities
            action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
            hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

            action_probs = self.actor(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )

            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                action_probs = self._apply_epsilon_mixing(action_probs, action_mask_t)

            # Get value
            value = self.critic(encoder_out)

        # Create distribution for consistent log probability calculation
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()

        # Get log probability from the distribution for consistency
        log_prob = dist.log_prob(action)

        return action.item() + 1, log_prob.item(), value.item()  # Convert back to 1-indexed

    # ------------------------------------------------------------------
    # Observation-only step (no action sampled / no transition stored)
    # ------------------------------------------------------------------
    def observe(self, state, player_id=None):
        """Propagate an observation through the encoder to update memory
        *without* sampling an action or storing any transition.  Useful for
        non-decision environment ticks such as the end of a trick.

        Parameters
        ----------
        state : dict
            Structured observation for the player.
        player_id : int | None
            Identifier to associate a persistent memory state (1-5 in game).
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        with torch.no_grad():
            # Encode with memory to update memory state
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

    # ------------------------------------------------------------------
    # Episode-level ingestion API
    # ------------------------------------------------------------------
    def store_episode_events(self, events: list, player_id_default: int | None = None) -> None:
        """Store a single player's episode as a chronological list of events.

        Each event must be one of:
          - Observation:
              {
                'kind': 'observation',
                'state': dict,
                'player_id': int (optional if player_id_default provided)
              }
          - Action:
              {
                'kind': 'action',
                'state': dict,
                'action': int (1-indexed),
                'log_prob': float,
                'value': float,
                'valid_actions': set[int],
                'reward': float,
                'player_id': int (optional if player_id_default provided),
                'win_label': float (optional),
                'final_return_label': float (optional),
              }

        """
        # Identify last action index to set done flag
        last_action_idx = -1
        for idx, ev in enumerate(events):
            if ev.get('kind') == 'action':
                last_action_idx = idx

        for idx, ev in enumerate(events):
            pid = ev.get('player_id', player_id_default)
            if pid is None:
                pid = 0
            if ev.get('kind') == 'action':
                mask = self.get_action_mask(ev['valid_actions'], self.action_size)
                self.events.append({
                    'kind': 'action',
                    'state': ev['state'],
                    'mask': mask,
                    'action': int(ev['action']) - 1,
                    'reward': float(ev['reward']),
                    'value': float(ev['value']),
                    'log_prob': float(ev['log_prob']),
                    'done': (idx == last_action_idx),
                    'player_id': pid,
                    'win': float(ev.get('win_label', 0.0) or 0.0),
                    'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
                    'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
                    'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
                })
            else:
                # Observation: mask = all ones by default
                mask = torch.ones(self.action_size, dtype=torch.bool)
                self.events.append({
                    'kind': 'observation',
                    'state': ev['state'],
                    'mask': mask,
                    'player_id': pid,
                })

    def compute_gae(self, next_value=0):
        """Compute GAE per player over action events; write results back into events."""
        # Group action indices by player
        actions_by_player: dict[int | None, list[int]] = {}
        for i, e in enumerate(self.events):
            if e['kind'] == 'action':
                pid = e.get('player_id', None)
                actions_by_player.setdefault(pid, []).append(i)

        all_advantages = []
        all_returns = []
        for pid, idxs in actions_by_player.items():
            if not idxs:
                continue
            rewards = np.array([self.events[i]['reward'] for i in idxs])
            # Bootstrap next value as 0.0 per player sequence
            values = np.array([self.events[i]['value'] for i in idxs] + [0.0])
            dones = np.array([self.events[i]['done'] for i in idxs] + [False])

            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values[:-1]

            for i, adv, ret in zip(idxs, advantages, returns):
                self.events[i]['advantage'] = float(adv)
                self.events[i]['return'] = float(ret)

            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_advantages:
            return np.array([]), np.array([])
        return np.concatenate(all_advantages), np.concatenate(all_returns)

    # ------------------------------------------------------------------
    # Internal helpers for PPO update
    # ------------------------------------------------------------------
    def _prepare_training_views(self):
        # Keep raw states (dicts) to encode inside the update for gradient flow
        states = [e['state'] for e in self.events]
        masks_t = [
            (e['mask'].to(device) if isinstance(e['mask'], torch.Tensor) else torch.as_tensor(e['mask'], dtype=torch.bool, device=device))
            for e in self.events
        ]
        kinds = [e['kind'] for e in self.events]
        pids = [e.get('player_id', None) for e in self.events]
        return states, masks_t, kinds, pids

    @staticmethod
    def _index_events_by_player(pids: list[int | None]) -> dict[int | None, list[int]]:
        events_by_player: dict[int | None, list[int]] = {}
        for idx, pid in enumerate(pids):
            events_by_player.setdefault(pid, []).append(idx)
        return events_by_player

    def _segments_from_events(self, events_by_player: dict[int | None, list[int]], kinds: list[str]):
        segments: list[tuple[int | None, int, int]] = []
        for pid, ev_idxs in events_by_player.items():
            action_ev_idxs = [i for i in ev_idxs if kinds[i] == 'action']
            if not action_ev_idxs:
                continue
            # Build done flags for this player's actions in order
            dones_pid = [self.events[i]['done'] for i in action_ev_idxs]
            start = ev_idxs[0]
            a_ptr = 0
            for i in ev_idxs:
                if kinds[i] == 'action':
                    if dones_pid[a_ptr]:
                        segments.append((pid, start, i))
                        start = i + 1
                    a_ptr += 1
            if start <= ev_idxs[-1]:
                segments.append((pid, start, ev_idxs[-1]))
        return segments

    @staticmethod
    def _pad_to_bt(lst, lengths, fill):
        T = max(lengths)
        out = []
        for i, tlen in enumerate(lengths):
            pad = T - tlen
            if pad > 0:
                pad_shape = list(lst[i].shape)
                pad_shape[0] = pad
                pad_tensor = torch.full(
                    pad_shape,
                    fill,
                    device=lst[i].device,
                    dtype=lst[i].dtype,
                )
                out.append(torch.cat([lst[i], pad_tensor], dim=0))
            else:
                out.append(lst[i])
        return torch.stack(out, dim=0), T

    def _build_minibatch_tensors(self, batch, states, masks_t, kinds, pids):
        lengths = []
        states_seqs = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        returns_list = []
        adv_list = []
        win_list_all = []
        final_ret_list_all = []
        secret_list_all = []
        points_list_all = []

        for pid, seg_start, seg_end in batch:
            # Restrict sequence strictly to this player's own events (actions + obs)
            # rather than a contiguous global slice. This ensures each agent learns
            # only from its direct experience without global context bleed-through.
            ev_range = [i for i in range(seg_start, seg_end + 1) if pids[i] == pid]
            lengths.append(len(ev_range))
            states_seqs.append([states[i] for i in ev_range])
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if (kinds[i] == 'action' and pids[i] == pid) else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, ret_bt, adv_bt = [], [], [], []
            win_bt, final_ret_bt, secret_bt = [], [], []
            points_bt = []
            for i in ev_range:
                if kinds[i] == 'action' and pids[i] == pid:
                    act_bt.append(torch.tensor(self.events[i]['action'], dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(self.events[i]['log_prob'], dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(self.events[i]['return'], dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(self.events[i]['advantage'], dtype=torch.float32, device=device))
                    win_lbl = self.events[i].get('win', 0.0) or 0.0
                    final_return_lbl = self.events[i].get('final_return', 0.0) or 0.0
                    secret_lbl = self.events[i].get('secret_partner', 0.0) or 0.0
                    win_bt.append(torch.tensor(float(win_lbl), dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(float(final_return_lbl), dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(float(secret_lbl), dtype=torch.float32, device=device))
                    pts_lbl = self.events[i].get('points_rel', [0.0] * 5)
                    points_bt.append(torch.tensor(pts_lbl, dtype=torch.float32, device=device))
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    win_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))
            win_list_all.append(torch.stack(win_bt, dim=0))
            final_ret_list_all.append(torch.stack(final_ret_bt, dim=0))
            secret_list_all.append(torch.stack(secret_bt, dim=0))
            points_list_all.append(torch.stack(points_bt, dim=0))

        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        win_bt, _ = self._pad_to_bt(win_list_all, lengths, 0.0)
        final_ret_bt, _ = self._pad_to_bt(final_ret_list_all, lengths, 0.0)
        secret_bt, _ = self._pad_to_bt(secret_list_all, lengths, 0.0)
        points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt, secret_bt, points_bt

    def _forward_vectorized(self, states_input, masks_bt, lengths_bt):
        """Vectorized forward pass for training with recurrent memory.

        Args:
            states_input: List of B sequences, each a list of observation dicts
            masks_bt: (B, T, action_size) action masks
            lengths_bt: (B,) sequence lengths

        Returns:
            logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
        """
        B = len(states_input)
        T = masks_bt.size(1)

        # Initialize memory to zeros for each segment
        memory_init = torch.zeros((B, 256), device=device)

        # Encode sequences with memory
        encoder_out_seq = self.encoder.encode_sequences(
            states_input,
            memory_in=memory_init,
            device=device
        )

        # Extract features (B, T, 256)
        features_bt = encoder_out_seq['features']

        # Build hand_ids_bt (B, T, N) from dict states
        N = 8  # Maximum hand size
        hand_ids_bt = torch.zeros((B, T, N), dtype=torch.long, device=device)
        for b, seq in enumerate(states_input):
            for t, s in enumerate(seq):
                if t >= T:
                    break
                arr = torch.as_tensor(s['hand_ids'], dtype=torch.long, device=device)
                if arr.dim() == 1:
                    arr = arr.view(-1)
                hand_ids_bt[b, t, :min(N, arr.numel())] = arr[:N]

        # Flatten time dimension to reuse single helper, then reshape back
        flat_feat = features_bt.reshape(B * T, -1)
        flat_hand = hand_ids_bt.reshape(B * T, N)
        flat_mask = masks_bt.view(B * T, -1)
        hand_tokens_bt = encoder_out_seq['hand_tokens']  # (B, T, N, d_token)
        flat_tokens = hand_tokens_bt.reshape(B * T, N, -1)

        logits_flat = self.actor._build_logits_from_features(
            actor_features=flat_feat,
            hand_ids=flat_hand,
            card_embedding=self.encoder.card,
            hand_tokens=flat_tokens,
            action_mask=flat_mask,
        )
        logits_bt = logits_flat.view(B, T, -1)

        # Critic values
        critic_feat_bt = self.critic.critic_adapter(features_bt)
        values_bt = self.critic.value_head(critic_feat_bt).squeeze(-1)

        # Auxiliary preds share critic_adapter features (gradients flow to encoder)
        aux_feat_bt = self.critic.critic_adapter(features_bt)
        win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
        ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
        secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
        points_pred_bt = self.critic.points_head(aux_feat_bt)

        return logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt

    @staticmethod
    def _flatten_action_steps(
        is_action_bt,
        logits_bt,
        values_bt,
        actions_bt,
        old_lp_bt,
        returns_bt,
        adv_bt,
        win_logits_bt,
        ret_pred_bt,
        win_bt,
        final_ret_bt,
        secret_logits_bt,
        secret_bt,
        masks_bt,
    ):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            logits_bt.view(-1, logits_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
            win_logits_bt.view(-1)[flat_mask],
            ret_pred_bt.view(-1)[flat_mask],
            win_bt.view(-1)[flat_mask],
            final_ret_bt.view(-1)[flat_mask],
            secret_logits_bt.view(-1)[flat_mask],
            secret_bt.view(-1)[flat_mask],
            masks_bt.view(-1, masks_bt.size(-1))[flat_mask],
        )

    @staticmethod
    def _entropy_from_probs(sub):
        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

    def _head_entropies(self, probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t):
        probs_pick = probs_flat[:, pick_idx_t]
        probs_partner = probs_flat[:, partner_idx_t]
        probs_bury = probs_flat[:, bury_idx_t]
        probs_play = probs_flat[:, play_idx_t]
        return (
            self._entropy_from_probs(probs_pick),
            self._entropy_from_probs(probs_partner),
            self._entropy_from_probs(probs_bury),
            self._entropy_from_probs(probs_play),
        )

    def _per_head_weights(self, actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, template):
        is_pick = torch.isin(actions_flat, pick_idx_t)
        is_partner = torch.isin(actions_flat, partner_idx_t)
        is_bury = torch.isin(actions_flat, bury_idx_t)
        is_play = ~(is_pick | is_partner | is_bury)

        count_pick = is_pick.sum().item()
        count_partner = is_partner.sum().item()
        count_bury = is_bury.sum().item()
        count_play = is_play.sum().item()
        total_count = float(count_pick + count_partner + count_bury + count_play)
        heads_present = int((count_pick > 0) + (count_partner > 0) + (count_bury > 0) + (count_play > 0))

        def per_head_weight(count: int) -> float:
            if heads_present == 0 or count == 0:
                return 0.0
            return total_count / (heads_present * float(count))

        w_pick = per_head_weight(count_pick)
        w_partner = per_head_weight(count_partner)
        w_bury = per_head_weight(count_bury)
        w_play = per_head_weight(count_play)

        head_weight = torch.ones_like(template)
        if w_pick > 0.0:
            head_weight[is_pick] = head_weight[is_pick] * w_pick
        else:
            head_weight[is_pick] = 0.0
        if w_partner > 0.0:
            head_weight[is_partner] = head_weight[is_partner] * w_partner
        else:
            head_weight[is_partner] = 0.0
        if w_bury > 0.0:
            head_weight[is_bury] = head_weight[is_bury] * w_bury
        else:
            head_weight[is_bury] = 0.0
        if w_play > 0.0:
            head_weight[is_play] = head_weight[is_play] * w_play
        else:
            head_weight[is_play] = 0.0

        return head_weight, is_pick, is_partner, is_bury

    def _actor_critic_losses(
        self,
        logits_flat,
        mask_flat,
        actions_flat,
        old_lp_flat,
        values_flat,
        returns_flat,
        adv_flat,
        pick_idx_t,
        partner_idx_t,
        bury_idx_t,
        play_idx_t,
    ):
        # Build probabilities fresh from logits to avoid in-place softmax conflicts
        probs_all = F.softmax(logits_flat, dim=-1)
        # Partner-call mixture on-the-fly
        if self.partner_call_epsilon > 0.0:
            A = probs_all.size(-1)
            call_mask = torch.zeros(A, dtype=torch.bool, device=probs_all.device)
            call_mask[self.partner_call_subindices] = True
            valid_call = call_mask.view(1, A).expand_as(mask_flat) & mask_flat
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(has, valid_call.float() / count.clamp_min(1.0), torch.zeros_like(probs_all))
            eps = torch.where(has, torch.full_like(count, self.partner_call_epsilon), torch.zeros_like(count))
            probs_all = (1.0 - eps) * probs_all + eps * ucall

        # PICK-floor mixing on-the-fly
        if self.pick_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pick = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand_as(probs_all)
            valid_pick = mask_flat[:, self.pick_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * probs_all + self.pick_floor_epsilon * one_hot_pick,
                probs_all,
            )

        # PASS-floor mixing on-the-fly
        if self.pass_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pass = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand_as(probs_all)
            valid_pass = mask_flat[:, self.pass_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * probs_all + self.pass_floor_epsilon * one_hot_pass,
                probs_all,
            )

        dist = torch.distributions.Categorical(probs_all.clamp(min=1e-12))
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = self._head_entropies(
            probs_all, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
        )
        entropy_term = (
            self.entropy_coeff_pick * pick_entropy
            + self.entropy_coeff_partner * partner_entropy
            + self.entropy_coeff_bury * bury_entropy
            + self.entropy_coeff_play * play_entropy
        )

        ratios = torch.exp(new_lp_flat - old_lp_flat)
        head_weight, is_pick, is_partner, is_bury = self._per_head_weights(
            actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, ratios
        )

        eps_flat = torch.full_like(ratios, self.clip_epsilon_play)
        eps_flat[is_pick] = self.clip_epsilon_pick
        eps_flat[is_partner] = self.clip_epsilon_partner
        eps_flat[is_bury] = self.clip_epsilon_bury

        surr1 = ratios * adv_flat
        clipped = torch.clamp(ratios, 1 - eps_flat, 1 + eps_flat) * adv_flat
        pg_loss_elements = -torch.min(surr1, clipped)
        policy_loss = (pg_loss_elements * head_weight).mean()

        returns_target = returns_flat.view(-1)
        values_old = values_flat.detach()
        v_clipped = values_old + torch.clamp(
            values_flat - values_old, -self.value_clip_epsilon, self.value_clip_epsilon
        )
        critic_loss_unclipped = F.mse_loss(values_flat, returns_target)
        critic_loss_clipped = F.mse_loss(v_clipped, returns_target)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

        actor_loss = policy_loss - entropy_term + self.kl_coef * approx_kl_t
        return actor_loss, critic_loss, approx_kl_t, (pick_entropy, partner_entropy, bury_entropy, play_entropy)

    def update(self, epochs=6, batch_size=256):
        """Update actor and critic networks using PPO with recurrent unrolling.
        Includes performance optimisations and per-update timing logs.
        """
        t_update_start = time.time()
        if len(self.events) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Store statistics before normalization
        raw_advantages = advantages.copy() if advantages.size else np.array([0.0])
        advantage_stats = {
            'mean': float(np.mean(raw_advantages)),
            'std': float(np.std(raw_advantages)),
            'min': float(np.min(raw_advantages)),
            'max': float(np.max(raw_advantages))
        }

        value_target_stats = {
            'mean': float(np.mean(returns)) if advantages.size else 0.0,
            'std': float(np.std(returns)) if advantages.size else 0.0,
            'min': float(np.min(returns)) if advantages.size else 0.0,
            'max': float(np.max(returns)) if advantages.size else 0.0,
        }

        # Normalize advantages and write back into events so the loss uses normalized values
        if advantages.size:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for e in self.events:
                if e.get('kind') == 'action':
                    e['advantage'] = float((e['advantage'] - adv_mean) / adv_std)

        # Build static views and segments
        t_build_start = time.time()
        states, masks_t, kinds, pids = self._prepare_training_views()
        events_by_player = self._index_events_by_player(pids)
        segments = self._segments_from_events(events_by_player, kinds)

        # Precompute static index tensors once
        pick_idx_tensor_static = torch.tensor(self.action_groups['pick'], device=device)
        partner_idx_tensor_static = torch.tensor(self.action_groups['partner'], device=device)
        bury_idx_tensor_static = torch.tensor(self.action_groups['bury'], device=device)
        play_idx_tensor_static = torch.tensor(self.action_groups['play'], device=device)

        t_build_end = time.time()

        # Training epochs – vectorized by batching segments
        forward_time = 0.0
        backward_time = 0.0
        step_time = 0.0
        optimizer_steps = 0
        early_stop_triggered = False
        last_approx_kl = 0.0

        # Instrumentation accumulators
        ent_pick_sum = 0.0
        ent_partner_sum = 0.0
        ent_bury_sum = 0.0
        ent_play_sum = 0.0
        ent_batches = 0
        pick_adv_sum = 0.0
        pick_adv_count = 0
        pass_adv_sum = 0.0
        pass_adv_count = 0

        for _ in range(epochs):
            if not segments:
                break
            perm = torch.randperm(len(segments))
            for mb_start in range(0, len(segments), batch_size):
                batch_idxs = perm[mb_start:mb_start + batch_size].tolist()
                batch = [segments[i] for i in batch_idxs]
                if len(batch) == 0:
                    continue

                (
                    states_bt,
                    masks_bt,
                    is_action_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    lengths_bt,
                    win_bt,
                    final_ret_bt,
                    secret_bt,
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds, pids)

                # Vectorized forward
                t_fwd = time.time()
                logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt,
                    logits_bt,
                    values_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    win_logits_bt,
                    ret_pred_bt,
                    win_bt,
                    final_ret_bt,
                    secret_logits_bt,
                    secret_bt,
                    masks_bt,
                )
                if flat is None:
                    continue
                (
                    logits_flat,
                    values_flat,
                    actions_flat,
                    old_lp_flat,
                    returns_flat,
                    adv_flat,
                    win_logits_flat,
                    ret_pred_flat,
                    win_labels_flat,
                    final_ret_labels_flat,
                    secret_logits_flat,
                    secret_labels_flat,
                    mask_flat,
                ) = flat

                # Record PICK/PASS advantages across minibatches
                with torch.no_grad():
                    pick_mask_specific = (actions_flat == self.pick_action_index)
                    if pick_mask_specific.any():
                        pick_adv_sum += adv_flat[pick_mask_specific].sum().item()
                        pick_adv_count += int(pick_mask_specific.sum().item())
                    pass_mask_specific = (actions_flat == self.pass_action_index)
                    if pass_mask_specific.any():
                        pass_adv_sum += adv_flat[pass_mask_specific].sum().item()
                        pass_adv_count += int(pass_mask_specific.sum().item())

                # Losses and metrics
                (
                    actor_loss,
                    critic_loss,
                    approx_kl_t,
                    (pick_entropy, partner_entropy, bury_entropy, play_entropy),
                ) = self._actor_critic_losses(
                    logits_flat,
                    mask_flat,
                    actions_flat,
                    old_lp_flat,
                    values_flat,
                    returns_flat,
                    adv_flat,
                    pick_idx_tensor_static,
                    partner_idx_tensor_static,
                    bury_idx_tensor_static,
                    play_idx_tensor_static,
                )
                last_approx_kl = float(approx_kl_t.item())

                # Entropy accumulation
                ent_pick_sum += pick_entropy.detach().item()
                ent_partner_sum += partner_entropy.detach().item()
                ent_bury_sum += bury_entropy.detach().item()
                ent_play_sum += play_entropy.detach().item()
                ent_batches += 1

                # Auxiliary losses
                bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
                return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)
                secret_loss = F.binary_cross_entropy_with_logits(secret_logits_flat, secret_labels_flat)

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = (
                    actor_loss
                    + self.value_loss_coeff * critic_loss
                    + self.win_loss_coeff * bce_loss
                    + self.return_loss_coeff * return_loss
                    + self.secret_loss_coeff * secret_loss
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                step_time += time.time() - t_step
                optimizer_steps += 1

                # Early stop further updates in this epoch if KL exceeds threshold
                if self.target_kl is not None and last_approx_kl > self.target_kl:
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        transitions = sum(1 for e in self.events if e['kind'] == 'action')

        # Clear storage
        self.reset_storage()

        # Return training statistics
        t_end = time.time()
        timing = {
            'build_s': t_build_end - t_build_start,
            'forward_s': forward_time,
            'backward_s': backward_time,
            'step_s': step_time,
            'total_update_s': t_end - t_update_start,
            'optimizer_steps': optimizer_steps,
        }
        return {
            'advantage_stats': advantage_stats,
            'value_target_stats': value_target_stats,
            'num_transitions': transitions,
            'approx_kl': last_approx_kl,
            'early_stop': early_stop_triggered,
            'timing': timing,
            'head_entropy': {
                'pick': (ent_pick_sum / ent_batches) if ent_batches > 0 else 0.0,
                'partner': (ent_partner_sum / ent_batches) if ent_batches > 0 else 0.0,
                'bury': (ent_bury_sum / ent_batches) if ent_batches > 0 else 0.0,
                'play': (ent_play_sum / ent_batches) if ent_batches > 0 else 0.0,
            },
            'pick_pass_adv': {
                'pick_mean': (pick_adv_sum / pick_adv_count) if pick_adv_count > 0 else 0.0,
                'pick_count': pick_adv_count,
                'pass_mean': (pass_adv_sum / pass_adv_count) if pass_adv_count > 0 else 0.0,
                'pass_count': pass_adv_count,
            },
        }

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath, load_optimizers: bool = True):
        """Load model parameters.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizers : bool
            If True, also loads optimizer states. For population/inference agents,
            pass False to skip optimizer loading entirely.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Core networks: accept missing/new critic aux heads for backward compatibility
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)

        # Optimizers: best-effort restore; fall back to fresh states if shapes changed
        if load_optimizers:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except ValueError as e:
                print(f"Warning: could not load optimizer state from {filepath}: {e}")

        # Reset memory states after loading
        self._player_memories = {}

```

Now I’ll finish threading the new points predictions and labels through the PPO update loop and then run a quick syntax check on the modified files.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Dict
from sheepshead import ACTION_IDS, BURY_ACTIONS, CALL_ACTIONS, UNDER_ACTIONS, PLAY_ACTIONS, DECK_IDS, UNDER_TOKEN
from encoder import CardReasoningEncoder, CardEmbeddingConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network consuming encoder features, with separate heads for
    pick / partner-selection / bury / play phases. The four heads' logits are
    concatenated back into the full action space order so existing masking
    logic continues to work unchanged.
    """

    def __init__(self,
                 action_size,
                 action_groups,
                 activation='swish',
                 *,
                 d_card: int,
                 d_token: int,
                 map_cid_to_play_action_index: torch.Tensor,
                 map_cid_to_bury_action_index: torch.Tensor,
                 map_cid_to_under_action_index: torch.Tensor,
                 call_action_global_indices: torch.Tensor,
                 call_card_ids: torch.Tensor,
                 play_under_action_index: int):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'

        # Shared actor adapter to add nonlinearity and specialization before heads
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
        )

        # === Heads ===
        self.pick_head = nn.Linear(256, len(action_groups['pick']))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(256, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(256, 1)

        # Per-head temperatures (τ): logits will be divided by τ before softmax
        # τ > 1.0 softens (higher entropy), τ < 1.0 sharpens. Defaults to 1.0.
        self.temperature_pick = 1.0
        self.temperature_partner = 1.0
        self.temperature_bury = 1.0
        self.temperature_play = 1.0

        # Init (generic weights)
        self.apply(self._init_weights)

        # ---- Card-conditioned configuration (dims + mappings) ----
        self._d_card = int(d_card)
        self._d_token = int(d_token)
        # Pointer scorer (Bahdanau style)
        self.pointer_hidden = 64
        self.pointer_Wg = nn.Linear(256, self.pointer_hidden)
        self.pointer_Wt = nn.Linear(self._d_token, self.pointer_hidden)
        self.pointer_v = nn.Linear(self.pointer_hidden, 1, bias=False)
        # Two-tower (card CALL scoring)
        self.tw_latent = 64
        self.tw_Wg = nn.Linear(256, self.tw_latent)
        self.tw_We = nn.Linear(self._d_card, self.tw_latent)

        # Action index mappings
        self._map_cid_to_play_action_index = map_cid_to_play_action_index.clone().long()
        self._map_cid_to_bury_action_index = map_cid_to_bury_action_index.clone().long()
        self._map_cid_to_under_action_index = map_cid_to_under_action_index.clone().long()
        self._call_action_global_indices = call_action_global_indices.clone().long()
        self._call_card_ids = call_card_ids.clone().long()
        self._play_under_action_index = int(play_under_action_index)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------ internal helpers ------------------------
    def _score_cards_two_tower(self, feat: torch.Tensor, card_embedding: nn.Embedding) -> torch.Tensor:
        """Return per-card scores S for all 34 card ids using two-tower scorer.
        feat: (B, 256)
        Returns: (B, 34)
        """
        q = self.tw_Wg(feat)                      # (B, k)        <- tower 1 (actor features)
        table = card_embedding.weight             # (34, d_card)
        K = self.tw_We(table)                     # (34, k)       <- tower 2 (card embeddings)
        S = torch.matmul(q, K.t())    # (B, 34)       <- similarity scores
        return S

    def _score_hand_pointer(
        self,
        feat: torch.Tensor,
        hand_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score each hand slot using pointer scorer.
        feat: (B, 256); hand_tokens: (B, N, d_token)
        Returns: s_slots (B, N)
        """
        tok = hand_tokens
        B, N = tok.size(0), tok.size(1)
        g = self.pointer_Wg(feat).unsqueeze(1).expand(B, N, -1)  # (B, N, h)
        t = self.pointer_Wt(tok)                                   # (B, N, h)
        e = torch.tanh(g + t)                                      # (B, N, h)
        s = self.pointer_v(e).squeeze(-1)                          # (B, N)
        return s

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        hand_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        hand_tokens : Tensor (K, N, d_token)
        action_mask : BoolTensor (K, A) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # Adapt features
        feat = self.actor_adapter(actor_features)

        # PICK / PASS
        pick_logits = self.pick_head(feat) / max(self.temperature_pick, 1e-6)
        logits[:, self.action_groups['pick']] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(feat) / max(self.temperature_partner, 1e-6)
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(feat, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)] / max(self.temperature_partner, 1e-6)
        logits[:, self._call_action_global_indices.to(device)] = call_scores

        # Pointer scores over hand tokens (compute once, reuse for bury/under/play)
        slot_scores = self._score_hand_pointer(
            feat,
            hand_tokens,
        )  # (K, N)
        K_, N = slot_scores.size(0), slot_scores.size(1)
        cids = hand_ids.long()

        # BURY and UNDER scatter
        idx_bury = self._map_cid_to_bury_action_index.to(device)[cids]   # (K, N)
        idx_under = self._map_cid_to_under_action_index.to(device)[cids] # (K, N)
        for i in range(N):
            b_idx = idx_bury[:, i]
            u_idx = idx_under[:, i]
            valid_b = b_idx.ge(0)
            valid_u = u_idx.ge(0)
            if valid_b.any():
                logits.view(K_, -1)[valid_b, b_idx[valid_b]] = slot_scores[valid_b, i] / max(self.temperature_bury, 1e-6)
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[valid_u, i] / max(self.temperature_bury, 1e-6)

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[valid_p, i] / max(self.temperature_play, 1e-6)

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(feat).squeeze(-1) / max(self.temperature_play, 1e-6)
            logits[:, self._play_under_action_index] = play_under_logit

        # Apply action mask if provided
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def set_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Set per-head softmax temperatures.

        Parameters
        ----------
        pick, partner, bury, play : float | None
            If provided, sets the corresponding head's temperature τ. Values are
            clamped to a small positive minimum to avoid divide-by-zero.
        """
        eps = 1e-6
        if pick is not None:
            self.temperature_pick = max(float(pick), eps)
        if partner is not None:
            self.temperature_partner = max(float(partner), eps)
        if bury is not None:
            self.temperature_bury = max(float(bury), eps)
        if play is not None:
            self.temperature_play = max(float(play), eps)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features' and 'hand_tokens'
        action_mask : Bool Tensor (batch, action_size)
        hand_ids : Tensor (batch, 8)
        card_embedding : nn.Embedding

        Returns
        -------
        probs : Tensor (batch, action_size)
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_with_logits(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass that also returns pre-softmax logits.

        Returns
        -------
        probs : Tensor
        logits : Tensor
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs, logits



class RecurrentCriticNetwork(nn.Module):
    """Critic head using encoder features directly."""

    def __init__(self, activation='swish'):
        super().__init__()

        self.critic_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU()
        )
        self.value_head = nn.Linear(256, 1)
        # Auxiliary heads
        self.win_head = nn.Linear(256, 1)
        self.return_head = nn.Linear(256, 1)
        self.secret_partner_head = nn.Linear(256, 1)
        # Per-player point prediction head (relative seating order, length-5 vector)
        self.points_head = nn.Linear(256, 5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_out: Dict[str, torch.Tensor]):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        value : Tensor (batch, 1)
        """
        feat = self.critic_adapter(encoder_out['features'])
        value = self.value_head(feat)
        return value

    def aux_predictions(self, encoder_out: Dict[str, torch.Tensor]):
        """Return auxiliary predictions as scalars: (win_prob, expected_final_return, secret_partner_prob).

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        win_prob : float
            Sigmoid of win logits
        expected_final_return : float
            Linear head output for expected final return
        secret_partner_prob : float
            Sigmoid of secret partner logits
        """
        with torch.no_grad():
            aux_feat = self.critic_adapter(encoder_out['features'])
            win_logit = self.win_head(aux_feat).squeeze(-1)
            expected_return = self.return_head(aux_feat).squeeze(-1)
            secret_logit = self.secret_partner_head(aux_feat).squeeze(-1)
        win_prob_t = torch.sigmoid(win_logit)
        win_prob = float(win_prob_t.item())
        expected_final = float(expected_return.item())
        secret_prob = float(torch.sigmoid(secret_logit).item())
        return win_prob, expected_final, secret_prob


class PPOAgent:
    def __init__(self, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
        # Dict-based encoder → fixed 256-d features
        self.state_size = 256
        self.action_size = action_size

        # --------------------------------------------------
        # Action groups for multi–head policy
        # --------------------------------------------------
        pick_indices = [ACTION_IDS["PICK"] - 1, ACTION_IDS["PASS"] - 1]

        # Partner-selection head: ALONE, JD PARTNER, all partner CALL actions
        partner_selection_actions = ["ALONE", "JD PARTNER"] + CALL_ACTIONS
        partner_indices = sorted({ACTION_IDS[a] - 1 for a in partner_selection_actions})

        # Bury head: explicit bury actions and UNDER actions
        bury_indices = sorted({ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + UNDER_ACTIONS)})

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            'pick': sorted(pick_indices),
            'partner': sorted(partner_indices),
            'bury': sorted(bury_indices),
            'play': sorted(play_indices),
        }

        # Explicit indices for PICK and PASS actions (0-indexed)
        self.pick_action_index = ACTION_IDS["PICK"] - 1
        self.pass_action_index = ACTION_IDS["PASS"] - 1

        # Encoder with memory
        self.encoder = CardReasoningEncoder(card_config=CardEmbeddingConfig()).to(device)

        # Per-player memory tracking
        self._player_memories = {}

        # Build mappings before constructing actor
        (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            call_action_global_indices,
            call_card_ids,
            play_under_action_index,
        ) = self._build_action_index_mappings()

        self.actor = MultiHeadRecurrentActorNetwork(
            action_size,
            self.action_groups,
            activation=activation,
            d_card=self.encoder.d_card_dim,
            d_token=self.encoder.d_token_dim,
            map_cid_to_play_action_index=map_cid_to_play_action_index,
            map_cid_to_bury_action_index=map_cid_to_bury_action_index,
            map_cid_to_under_action_index=map_cid_to_under_action_index,
            call_action_global_indices=call_action_global_indices,
            call_card_ids=call_card_ids,
            play_under_action_index=play_under_action_index,
        ).to(device)

        self.critic = RecurrentCriticNetwork(activation=activation).to(device)

        # Optimizers (include encoder params with scaled LR for card embeddings)
        encoder_groups = self.encoder.param_groups(base_lr=lr_actor, card_lr_scale=0.2)
        actor_groups = [
            {'params': self.actor.parameters(), 'lr': lr_actor},
            *encoder_groups,
        ]
        self.actor_optimizer = optim.Adam(actor_groups)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.95
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_partner = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.3
        self.clip_epsilon_pick = 0.20
        self.clip_epsilon_partner = 0.25
        self.clip_epsilon_bury = 0.15
        self.clip_epsilon_play = 0.20
        self.value_clip_epsilon = 0.20

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.0

        # Auxiliary loss coefficients
        self.win_loss_coeff = 0.1
        self.return_loss_coeff = 0.1
        self.secret_loss_coeff = 0.07
        self.points_loss_coeff = 0.1

        # Storage for trajectory data
        self.reset_storage()

        # --------------------------------------------------
        # Partner-call mixture: epsilon floor over CALL actions
        # --------------------------------------------------
        # Subset of partner head actions eligible for CALL-uniform mixing
        self.partner_call_subindices = sorted(
            {ACTION_IDS["JD PARTNER"] - 1} | {ACTION_IDS[a] - 1 for a in CALL_ACTIONS}
        )
        # Blending coefficient ε (0 disables; typical 0.02–0.10)
        self.partner_call_epsilon = 0.0

        # PASS-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pass_floor_epsilon = 0.0
        # PICK-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pick_floor_epsilon = 0.0

    def get_recurrent_memory(self, player_id: int | None, device: torch.device | None = None) -> torch.Tensor:
        """Return the recurrent memory vector for a player, or a zero vector if unset.

        The returned tensor is (256,) on the requested device.
        """
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mem = self._player_memories.get(player_id) if player_id is not None else None
        if mem is None:
            return torch.zeros(256, device=target_device)
        return mem.to(target_device) if mem.device != target_device else mem

    def set_recurrent_memory(self, player_id: int | None, memory_vector: torch.Tensor) -> None:
        """Set the recurrent memory vector for a player."""
        self._player_memories[player_id] = memory_vector

    def _build_action_index_mappings(self):
        """Precompute global action index mappings for card-specific actions."""
        map_cid_to_play_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_bury_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_under_action_index = torch.full((34,), -1, dtype=torch.long)

        # PLAY actions
        play_under_index = None
        for a in PLAY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            if a == f"PLAY {UNDER_TOKEN}":
                play_under_index = a_idx
                continue
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_play_action_index[cid] = a_idx

        # BURY actions
        for a in BURY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_bury_action_index[cid] = a_idx

        # UNDER actions (choose card from hand to set as under)
        for a in UNDER_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_under_action_index[cid] = a_idx

        # CALL actions
        idx_call_global_list = []
        call_card_ids_list = []
        for a in CALL_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            parts = a.split()
            # a is either "CALL X" or "CALL X UNDER"
            card = parts[1]
            cid = DECK_IDS[card]
            idx_call_global_list.append(a_idx)
            call_card_ids_list.append(cid)

        idx_call_global = torch.tensor(sorted(idx_call_global_list), dtype=torch.long)
        # Align cid list to the sorted order of indices
        # Build mapping from index to cid then reorder
        idx_to_cid = {idx: cid for idx, cid in zip(idx_call_global_list, call_card_ids_list)}
        call_card_ids = torch.tensor([idx_to_cid[int(x)] for x in idx_call_global.tolist()], dtype=torch.long)

        return map_cid_to_play_action_index, map_cid_to_bury_action_index, map_cid_to_under_action_index, idx_call_global, call_card_ids, play_under_index

    def set_head_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Convenience proxy to set per-head temperatures on the actor."""
        self.actor.set_temperatures(pick=pick, partner=partner, bury=bury, play=play)

    def set_partner_call_epsilon(self, eps: float):
        """Set ε for uniform CALL mixture on partner steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.partner_call_epsilon = eps

    def set_pass_floor_epsilon(self, eps: float):
        """Set ε for PASS-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pass_floor_epsilon = eps

    def set_pick_floor_epsilon(self, eps: float):
        """Set ε for PICK-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pick_floor_epsilon = eps

    def _apply_epsilon_mixing(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply partner-call ε mixture and PASS-floor ε mixture to probability
        distributions in a batched way without in-place ops.

        Parameters
        ----------
        probs : Tensor
            Shape (B, A) probabilities over actions
        mask : Bool Tensor
            Shape (B, A) of valid actions
        """
        mixed = probs
        if self.partner_call_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            call_vec = torch.zeros(A, dtype=torch.bool, device=mixed.device)
            call_idx = torch.tensor(self.partner_call_subindices, dtype=torch.long, device=mixed.device)
            call_vec[call_idx] = True
            valid_call = mask & call_vec.view(1, A).expand(B, A)
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(
                has,
                valid_call.float() / count.clamp_min(1.0),
                torch.zeros_like(mixed),
            )
            eps = torch.where(
                has,
                torch.full_like(count, self.partner_call_epsilon),
                torch.zeros_like(count),
            )
            mixed = (1.0 - eps) * mixed + eps * ucall

        if self.pick_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pick = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand(B, A)
            valid_pick = mask[:, self.pick_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * mixed + self.pick_floor_epsilon * one_hot_pick,
                mixed,
            )

        if self.pass_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pass = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand(B, A)
            valid_pass = mask[:, self.pass_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * mixed + self.pass_floor_epsilon * one_hot_pass,
                mixed,
            )

        return mixed

    def reset_storage(self):
        # Ordered list of events: each is a dict with keys:
        # kind: 'action' | 'obs'
        # state: np.array state vector
        # mask: torch.bool mask or convertible
        # If kind == 'action': action (0-indexed), reward, value, log_prob, done
        # GAE writes back: 'advantage' and 'return'
        self.events = []

    def get_action_mask(self, valid_actions, action_size):
        """Convert valid actions set to boolean mask"""
        mask = torch.zeros(action_size, dtype=torch.bool)
        for action in valid_actions:
            mask[action - 1] = True  # Convert from 1-indexed to 0-indexed
        return mask

    def reset_recurrent_state(self):
        """Clear memory states (call at the start of every new game)."""
        self._player_memories = {}

    def get_action_probs_with_logits(self, state, valid_actions, player_id=None):
        """Return post-mixture action probabilities and pre-mix logits for a single dict state.

        Applies partner CALL-uniform mixture if enabled. Keeps PPO on-policy by
        exposing the same transformed distribution that sampling uses.
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        # Encode dict state with memory
        encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

        # Store updated memory
        if player_id is not None:
            self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

        action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
        hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

        with torch.no_grad():
            probs, logits = self.actor.forward_with_logits(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )
            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                probs = self._apply_epsilon_mixing(probs, action_mask_t)

        return probs, logits

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        with torch.no_grad():
            # Get or init memory for this player
            memory_in = self.get_recurrent_memory(player_id, device=device)

            # Encode with memory
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

            # Get action probabilities
            action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
            hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

            action_probs = self.actor(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )

            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                action_probs = self._apply_epsilon_mixing(action_probs, action_mask_t)

            # Get value
            value = self.critic(encoder_out)

        # Create distribution for consistent log probability calculation
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()

        # Get log probability from the distribution for consistency
        log_prob = dist.log_prob(action)

        return action.item() + 1, log_prob.item(), value.item()  # Convert back to 1-indexed

    # ------------------------------------------------------------------
    # Observation-only step (no action sampled / no transition stored)
    # ------------------------------------------------------------------
    def observe(self, state, player_id=None):
        """Propagate an observation through the encoder to update memory
        *without* sampling an action or storing any transition.  Useful for
        non-decision environment ticks such as the end of a trick.

        Parameters
        ----------
        state : dict
            Structured observation for the player.
        player_id : int | None
            Identifier to associate a persistent memory state (1-5 in game).
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        with torch.no_grad():
            # Encode with memory to update memory state
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

    # ------------------------------------------------------------------
    # Episode-level ingestion API
    # ------------------------------------------------------------------
    def store_episode_events(self, events: list, player_id_default: int | None = None) -> None:
        """Store a single player's episode as a chronological list of events.

        Each event must be one of:
          - Observation:
              {
                'kind': 'observation',
                'state': dict,
                'player_id': int (optional if player_id_default provided)
              }
          - Action:
              {
                'kind': 'action',
                'state': dict,
                'action': int (1-indexed),
                'log_prob': float,
                'value': float,
                'valid_actions': set[int],
                'reward': float,
                'player_id': int (optional if player_id_default provided),
                'win_label': float (optional),
                'final_return_label': float (optional),
              }

        """
        # Identify last action index to set done flag
        last_action_idx = -1
        for idx, ev in enumerate(events):
            if ev.get('kind') == 'action':
                last_action_idx = idx

        for idx, ev in enumerate(events):
            pid = ev.get('player_id', player_id_default)
            if pid is None:
                pid = 0
            if ev.get('kind') == 'action':
                mask = self.get_action_mask(ev['valid_actions'], self.action_size)
                self.events.append({
                    'kind': 'action',
                    'state': ev['state'],
                    'mask': mask,
                    'action': int(ev['action']) - 1,
                    'reward': float(ev['reward']),
                    'value': float(ev['value']),
                    'log_prob': float(ev['log_prob']),
                    'done': (idx == last_action_idx),
                    'player_id': pid,
                    'win': float(ev.get('win_label', 0.0) or 0.0),
                    'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
                    'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
                    'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
                })
            else:
                # Observation: mask = all ones by default
                mask = torch.ones(self.action_size, dtype=torch.bool)
                self.events.append({
                    'kind': 'observation',
                    'state': ev['state'],
                    'mask': mask,
                    'player_id': pid,
                })

    def compute_gae(self, next_value=0):
        """Compute GAE per player over action events; write results back into events."""
        # Group action indices by player
        actions_by_player: dict[int | None, list[int]] = {}
        for i, e in enumerate(self.events):
            if e['kind'] == 'action':
                pid = e.get('player_id', None)
                actions_by_player.setdefault(pid, []).append(i)

        all_advantages = []
        all_returns = []
        for pid, idxs in actions_by_player.items():
            if not idxs:
                continue
            rewards = np.array([self.events[i]['reward'] for i in idxs])
            # Bootstrap next value as 0.0 per player sequence
            values = np.array([self.events[i]['value'] for i in idxs] + [0.0])
            dones = np.array([self.events[i]['done'] for i in idxs] + [False])

            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values[:-1]

            for i, adv, ret in zip(idxs, advantages, returns):
                self.events[i]['advantage'] = float(adv)
                self.events[i]['return'] = float(ret)

            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_advantages:
            return np.array([]), np.array([])
        return np.concatenate(all_advantages), np.concatenate(all_returns)

    # ------------------------------------------------------------------
    # Internal helpers for PPO update
    # ------------------------------------------------------------------
    def _prepare_training_views(self):
        # Keep raw states (dicts) to encode inside the update for gradient flow
        states = [e['state'] for e in self.events]
        masks_t = [
            (e['mask'].to(device) if isinstance(e['mask'], torch.Tensor) else torch.as_tensor(e['mask'], dtype=torch.bool, device=device))
            for e in self.events
        ]
        kinds = [e['kind'] for e in self.events]
        pids = [e.get('player_id', None) for e in self.events]
        return states, masks_t, kinds, pids

    @staticmethod
    def _index_events_by_player(pids: list[int | None]) -> dict[int | None, list[int]]:
        events_by_player: dict[int | None, list[int]] = {}
        for idx, pid in enumerate(pids):
            events_by_player.setdefault(pid, []).append(idx)
        return events_by_player

    def _segments_from_events(self, events_by_player: dict[int | None, list[int]], kinds: list[str]):
        segments: list[tuple[int | None, int, int]] = []
        for pid, ev_idxs in events_by_player.items():
            action_ev_idxs = [i for i in ev_idxs if kinds[i] == 'action']
            if not action_ev_idxs:
                continue
            # Build done flags for this player's actions in order
            dones_pid = [self.events[i]['done'] for i in action_ev_idxs]
            start = ev_idxs[0]
            a_ptr = 0
            for i in ev_idxs:
                if kinds[i] == 'action':
                    if dones_pid[a_ptr]:
                        segments.append((pid, start, i))
                        start = i + 1
                    a_ptr += 1
            if start <= ev_idxs[-1]:
                segments.append((pid, start, ev_idxs[-1]))
        return segments

    @staticmethod
    def _pad_to_bt(lst, lengths, fill):
        T = max(lengths)
        out = []
        for i, tlen in enumerate(lengths):
            pad = T - tlen
            if pad > 0:
                pad_shape = list(lst[i].shape)
                pad_shape[0] = pad
                pad_tensor = torch.full(
                    pad_shape,
                    fill,
                    device=lst[i].device,
                    dtype=lst[i].dtype,
                )
                out.append(torch.cat([lst[i], pad_tensor], dim=0))
            else:
                out.append(lst[i])
        return torch.stack(out, dim=0), T

    def _build_minibatch_tensors(self, batch, states, masks_t, kinds, pids):
        lengths = []
        states_seqs = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        returns_list = []
        adv_list = []
        win_list_all = []
        final_ret_list_all = []
        secret_list_all = []
        points_list_all = []

        for pid, seg_start, seg_end in batch:
            # Restrict sequence strictly to this player's own events (actions + obs)
            # rather than a contiguous global slice. This ensures each agent learns
            # only from its direct experience without global context bleed-through.
            ev_range = [i for i in range(seg_start, seg_end + 1) if pids[i] == pid]
            lengths.append(len(ev_range))
            states_seqs.append([states[i] for i in ev_range])
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if (kinds[i] == 'action' and pids[i] == pid) else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, ret_bt, adv_bt = [], [], [], []
            win_bt, final_ret_bt, secret_bt = [], [], []
            points_bt = []
            for i in ev_range:
                if kinds[i] == 'action' and pids[i] == pid:
                    act_bt.append(torch.tensor(self.events[i]['action'], dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(self.events[i]['log_prob'], dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(self.events[i]['return'], dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(self.events[i]['advantage'], dtype=torch.float32, device=device))
                    win_lbl = self.events[i].get('win', 0.0) or 0.0
                    final_return_lbl = self.events[i].get('final_return', 0.0) or 0.0
                    secret_lbl = self.events[i].get('secret_partner', 0.0) or 0.0
                    win_bt.append(torch.tensor(float(win_lbl), dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(float(final_return_lbl), dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(float(secret_lbl), dtype=torch.float32, device=device))
                    pts_lbl = self.events[i].get('points_rel', [0.0] * 5)
                    points_bt.append(torch.tensor(pts_lbl, dtype=torch.float32, device=device))
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    win_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))
            win_list_all.append(torch.stack(win_bt, dim=0))
            final_ret_list_all.append(torch.stack(final_ret_bt, dim=0))
            secret_list_all.append(torch.stack(secret_bt, dim=0))
            points_list_all.append(torch.stack(points_bt, dim=0))

        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        win_bt, _ = self._pad_to_bt(win_list_all, lengths, 0.0)
        final_ret_bt, _ = self._pad_to_bt(final_ret_list_all, lengths, 0.0)
        secret_bt, _ = self._pad_to_bt(secret_list_all, lengths, 0.0)
        points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt, secret_bt, points_bt

    def _forward_vectorized(self, states_input, masks_bt, lengths_bt):
        """Vectorized forward pass for training with recurrent memory.

        Args:
            states_input: List of B sequences, each a list of observation dicts
            masks_bt: (B, T, action_size) action masks
            lengths_bt: (B,) sequence lengths

        Returns:
            logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
        """
        B = len(states_input)
        T = masks_bt.size(1)

        # Initialize memory to zeros for each segment
        memory_init = torch.zeros((B, 256), device=device)

        # Encode sequences with memory
        encoder_out_seq = self.encoder.encode_sequences(
            states_input,
            memory_in=memory_init,
            device=device
        )

        # Extract features (B, T, 256)
        features_bt = encoder_out_seq['features']

        # Build hand_ids_bt (B, T, N) from dict states
        N = 8  # Maximum hand size
        hand_ids_bt = torch.zeros((B, T, N), dtype=torch.long, device=device)
        for b, seq in enumerate(states_input):
            for t, s in enumerate(seq):
                if t >= T:
                    break
                arr = torch.as_tensor(s['hand_ids'], dtype=torch.long, device=device)
                if arr.dim() == 1:
                    arr = arr.view(-1)
                hand_ids_bt[b, t, :min(N, arr.numel())] = arr[:N]

        # Flatten time dimension to reuse single helper, then reshape back
        flat_feat = features_bt.reshape(B * T, -1)
        flat_hand = hand_ids_bt.reshape(B * T, N)
        flat_mask = masks_bt.view(B * T, -1)
        hand_tokens_bt = encoder_out_seq['hand_tokens']  # (B, T, N, d_token)
        flat_tokens = hand_tokens_bt.reshape(B * T, N, -1)

        logits_flat = self.actor._build_logits_from_features(
            actor_features=flat_feat,
            hand_ids=flat_hand,
            card_embedding=self.encoder.card,
            hand_tokens=flat_tokens,
            action_mask=flat_mask,
        )
        logits_bt = logits_flat.view(B, T, -1)

        # Critic values
        critic_feat_bt = self.critic.critic_adapter(features_bt)
        values_bt = self.critic.value_head(critic_feat_bt).squeeze(-1)

        # Auxiliary preds share critic_adapter features (gradients flow to encoder)
        aux_feat_bt = self.critic.critic_adapter(features_bt)
        win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
        ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
        secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
        points_pred_bt = self.critic.points_head(aux_feat_bt)

        return logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt

    @staticmethod
    def _flatten_action_steps(
        is_action_bt,
        logits_bt,
        values_bt,
        actions_bt,
        old_lp_bt,
        returns_bt,
        adv_bt,
        win_logits_bt,
        ret_pred_bt,
        win_bt,
        final_ret_bt,
        secret_logits_bt,
        secret_bt,
        masks_bt,
    ):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            logits_bt.view(-1, logits_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
            win_logits_bt.view(-1)[flat_mask],
            ret_pred_bt.view(-1)[flat_mask],
            win_bt.view(-1)[flat_mask],
            final_ret_bt.view(-1)[flat_mask],
            secret_logits_bt.view(-1)[flat_mask],
            secret_bt.view(-1)[flat_mask],
            masks_bt.view(-1, masks_bt.size(-1))[flat_mask],
        )

    @staticmethod
    def _entropy_from_probs(sub):
        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

    def _head_entropies(self, probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t):
        probs_pick = probs_flat[:, pick_idx_t]
        probs_partner = probs_flat[:, partner_idx_t]
        probs_bury = probs_flat[:, bury_idx_t]
        probs_play = probs_flat[:, play_idx_t]
        return (
            self._entropy_from_probs(probs_pick),
            self._entropy_from_probs(probs_partner),
            self._entropy_from_probs(probs_bury),
            self._entropy_from_probs(probs_play),
        )

    def _per_head_weights(self, actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, template):
        is_pick = torch.isin(actions_flat, pick_idx_t)
        is_partner = torch.isin(actions_flat, partner_idx_t)
        is_bury = torch.isin(actions_flat, bury_idx_t)
        is_play = ~(is_pick | is_partner | is_bury)

        count_pick = is_pick.sum().item()
        count_partner = is_partner.sum().item()
        count_bury = is_bury.sum().item()
        count_play = is_play.sum().item()
        total_count = float(count_pick + count_partner + count_bury + count_play)
        heads_present = int((count_pick > 0) + (count_partner > 0) + (count_bury > 0) + (count_play > 0))

        def per_head_weight(count: int) -> float:
            if heads_present == 0 or count == 0:
                return 0.0
            return total_count / (heads_present * float(count))

        w_pick = per_head_weight(count_pick)
        w_partner = per_head_weight(count_partner)
        w_bury = per_head_weight(count_bury)
        w_play = per_head_weight(count_play)

        head_weight = torch.ones_like(template)
        if w_pick > 0.0:
            head_weight[is_pick] = head_weight[is_pick] * w_pick
        else:
            head_weight[is_pick] = 0.0
        if w_partner > 0.0:
            head_weight[is_partner] = head_weight[is_partner] * w_partner
        else:
            head_weight[is_partner] = 0.0
        if w_bury > 0.0:
            head_weight[is_bury] = head_weight[is_bury] * w_bury
        else:
            head_weight[is_bury] = 0.0
        if w_play > 0.0:
            head_weight[is_play] = head_weight[is_play] * w_play
        else:
            head_weight[is_play] = 0.0

        return head_weight, is_pick, is_partner, is_bury

    def _actor_critic_losses(
        self,
        logits_flat,
        mask_flat,
        actions_flat,
        old_lp_flat,
        values_flat,
        returns_flat,
        adv_flat,
        pick_idx_t,
        partner_idx_t,
        bury_idx_t,
        play_idx_t,
    ):
        # Build probabilities fresh from logits to avoid in-place softmax conflicts
        probs_all = F.softmax(logits_flat, dim=-1)
        # Partner-call mixture on-the-fly
        if self.partner_call_epsilon > 0.0:
            A = probs_all.size(-1)
            call_mask = torch.zeros(A, dtype=torch.bool, device=probs_all.device)
            call_mask[self.partner_call_subindices] = True
            valid_call = call_mask.view(1, A).expand_as(mask_flat) & mask_flat
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(has, valid_call.float() / count.clamp_min(1.0), torch.zeros_like(probs_all))
            eps = torch.where(has, torch.full_like(count, self.partner_call_epsilon), torch.zeros_like(count))
            probs_all = (1.0 - eps) * probs_all + eps * ucall

        # PICK-floor mixing on-the-fly
        if self.pick_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pick = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand_as(probs_all)
            valid_pick = mask_flat[:, self.pick_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * probs_all + self.pick_floor_epsilon * one_hot_pick,
                probs_all,
            )

        # PASS-floor mixing on-the-fly
        if self.pass_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pass = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand_as(probs_all)
            valid_pass = mask_flat[:, self.pass_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * probs_all + self.pass_floor_epsilon * one_hot_pass,
                probs_all,
            )

        dist = torch.distributions.Categorical(probs_all.clamp(min=1e-12))
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = self._head_entropies(
            probs_all, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
        )
        entropy_term = (
            self.entropy_coeff_pick * pick_entropy
            + self.entropy_coeff_partner * partner_entropy
            + self.entropy_coeff_bury * bury_entropy
            + self.entropy_coeff_play * play_entropy
        )

        ratios = torch.exp(new_lp_flat - old_lp_flat)
        head_weight, is_pick, is_partner, is_bury = self._per_head_weights(
            actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, ratios
        )

        eps_flat = torch.full_like(ratios, self.clip_epsilon_play)
        eps_flat[is_pick] = self.clip_epsilon_pick
        eps_flat[is_partner] = self.clip_epsilon_partner
        eps_flat[is_bury] = self.clip_epsilon_bury

        surr1 = ratios * adv_flat
        clipped = torch.clamp(ratios, 1 - eps_flat, 1 + eps_flat) * adv_flat
        pg_loss_elements = -torch.min(surr1, clipped)
        policy_loss = (pg_loss_elements * head_weight).mean()

        returns_target = returns_flat.view(-1)
        values_old = values_flat.detach()
        v_clipped = values_old + torch.clamp(
            values_flat - values_old, -self.value_clip_epsilon, self.value_clip_epsilon
        )
        critic_loss_unclipped = F.mse_loss(values_flat, returns_target)
        critic_loss_clipped = F.mse_loss(v_clipped, returns_target)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

        actor_loss = policy_loss - entropy_term + self.kl_coef * approx_kl_t
        return actor_loss, critic_loss, approx_kl_t, (pick_entropy, partner_entropy, bury_entropy, play_entropy)

    def update(self, epochs=6, batch_size=256):
        """Update actor and critic networks using PPO with recurrent unrolling.
        Includes performance optimisations and per-update timing logs.
        """
        t_update_start = time.time()
        if len(self.events) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Store statistics before normalization
        raw_advantages = advantages.copy() if advantages.size else np.array([0.0])
        advantage_stats = {
            'mean': float(np.mean(raw_advantages)),
            'std': float(np.std(raw_advantages)),
            'min': float(np.min(raw_advantages)),
            'max': float(np.max(raw_advantages))
        }

        value_target_stats = {
            'mean': float(np.mean(returns)) if advantages.size else 0.0,
            'std': float(np.std(returns)) if advantages.size else 0.0,
            'min': float(np.min(returns)) if advantages.size else 0.0,
            'max': float(np.max(returns)) if advantages.size else 0.0,
        }

        # Normalize advantages and write back into events so the loss uses normalized values
        if advantages.size:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for e in self.events:
                if e.get('kind') == 'action':
                    e['advantage'] = float((e['advantage'] - adv_mean) / adv_std)

        # Build static views and segments
        t_build_start = time.time()
        states, masks_t, kinds, pids = self._prepare_training_views()
        events_by_player = self._index_events_by_player(pids)
        segments = self._segments_from_events(events_by_player, kinds)

        # Precompute static index tensors once
        pick_idx_tensor_static = torch.tensor(self.action_groups['pick'], device=device)
        partner_idx_tensor_static = torch.tensor(self.action_groups['partner'], device=device)
        bury_idx_tensor_static = torch.tensor(self.action_groups['bury'], device=device)
        play_idx_tensor_static = torch.tensor(self.action_groups['play'], device=device)

        t_build_end = time.time()

        # Training epochs – vectorized by batching segments
        forward_time = 0.0
        backward_time = 0.0
        step_time = 0.0
        optimizer_steps = 0
        early_stop_triggered = False
        last_approx_kl = 0.0

        # Instrumentation accumulators
        ent_pick_sum = 0.0
        ent_partner_sum = 0.0
        ent_bury_sum = 0.0
        ent_play_sum = 0.0
        ent_batches = 0
        pick_adv_sum = 0.0
        pick_adv_count = 0
        pass_adv_sum = 0.0
        pass_adv_count = 0

        for _ in range(epochs):
            if not segments:
                break
            perm = torch.randperm(len(segments))
            for mb_start in range(0, len(segments), batch_size):
                batch_idxs = perm[mb_start:mb_start + batch_size].tolist()
                batch = [segments[i] for i in batch_idxs]
                if len(batch) == 0:
                    continue

                (
                    states_bt,
                    masks_bt,
                    is_action_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    lengths_bt,
                    win_bt,
                    final_ret_bt,
                    secret_bt,
                    points_bt,
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds, pids)

                # Vectorized forward
                t_fwd = time.time()
                logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt,
                    logits_bt,
                    values_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    win_logits_bt,
                    ret_pred_bt,
                    win_bt,
                    final_ret_bt,
                    secret_logits_bt,
                    secret_bt,
                    masks_bt,
                )
                if flat is None:
                    continue
                (
                    logits_flat,
                    values_flat,
                    actions_flat,
                    old_lp_flat,
                    returns_flat,
                    adv_flat,
                    win_logits_flat,
                    ret_pred_flat,
                    win_labels_flat,
                    final_ret_labels_flat,
                    secret_logits_flat,
                    secret_labels_flat,
                    mask_flat,
                ) = flat

                # Record PICK/PASS advantages across minibatches
                with torch.no_grad():
                    pick_mask_specific = (actions_flat == self.pick_action_index)
                    if pick_mask_specific.any():
                        pick_adv_sum += adv_flat[pick_mask_specific].sum().item()
                        pick_adv_count += int(pick_mask_specific.sum().item())
                    pass_mask_specific = (actions_flat == self.pass_action_index)
                    if pass_mask_specific.any():
                        pass_adv_sum += adv_flat[pass_mask_specific].sum().item()
                        pass_adv_count += int(pass_mask_specific.sum().item())

                # Losses and metrics
                (
                    actor_loss,
                    critic_loss,
                    approx_kl_t,
                    (pick_entropy, partner_entropy, bury_entropy, play_entropy),
                ) = self._actor_critic_losses(
                    logits_flat,
                    mask_flat,
                    actions_flat,
                    old_lp_flat,
                    values_flat,
                    returns_flat,
                    adv_flat,
                    pick_idx_tensor_static,
                    partner_idx_tensor_static,
                    bury_idx_tensor_static,
                    play_idx_tensor_static,
                )
                last_approx_kl = float(approx_kl_t.item())

                # Entropy accumulation
                ent_pick_sum += pick_entropy.detach().item()
                ent_partner_sum += partner_entropy.detach().item()
                ent_bury_sum += bury_entropy.detach().item()
                ent_play_sum += play_entropy.detach().item()
                ent_batches += 1

                # Auxiliary losses
                bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
                return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)
                secret_loss = F.binary_cross_entropy_with_logits(secret_logits_flat, secret_labels_flat)
                # Per-player points auxiliary loss (regression on normalized per-seat totals)
                # points_pred_flat and labels are (N, 5); smooth L1 stabilizes training.
                points_pred_flat = points_pred_bt.view(-1, points_pred_bt.size(-1))[is_action_bt.view(-1)]
                points_labels_flat = points_bt.view(-1, points_bt.size(-1))[is_action_bt.view(-1)]
                points_loss = F.smooth_l1_loss(points_pred_flat, points_labels_flat)

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = (
                    actor_loss
                    + self.value_loss_coeff * critic_loss
                    + self.win_loss_coeff * bce_loss
                    + self.return_loss_coeff * return_loss
                    + self.secret_loss_coeff * secret_loss
                    + self.points_loss_coeff * points_loss
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                step_time += time.time() - t_step
                optimizer_steps += 1

                # Early stop further updates in this epoch if KL exceeds threshold
                if self.target_kl is not None and last_approx_kl > self.target_kl:
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        transitions = sum(1 for e in self.events if e['kind'] == 'action')

        # Clear storage
        self.reset_storage()

        # Return training statistics
        t_end = time.time()
        timing = {
            'build_s': t_build_end - t_build_start,
            'forward_s': forward_time,
            'backward_s': backward_time,
            'step_s': step_time,
            'total_update_s': t_end - t_update_start,
            'optimizer_steps': optimizer_steps,
        }
        return {
            'advantage_stats': advantage_stats,
            'value_target_stats': value_target_stats,
            'num_transitions': transitions,
            'approx_kl': last_approx_kl,
            'early_stop': early_stop_triggered,
            'timing': timing,
            'head_entropy': {
                'pick': (ent_pick_sum / ent_batches) if ent_batches > 0 else 0.0,
                'partner': (ent_partner_sum / ent_batches) if ent_batches > 0 else 0.0,
                'bury': (ent_bury_sum / ent_batches) if ent_batches > 0 else 0.0,
                'play': (ent_play_sum / ent_batches) if ent_batches > 0 else 0.0,
            },
            'pick_pass_adv': {
                'pick_mean': (pick_adv_sum / pick_adv_count) if pick_adv_count > 0 else 0.0,
                'pick_count': pick_adv_count,
                'pass_mean': (pass_adv_sum / pass_adv_count) if pass_adv_count > 0 else 0.0,
                'pass_count': pass_adv_count,
            },
        }

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath, load_optimizers: bool = True):
        """Load model parameters.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizers : bool
            If True, also loads optimizer states. For population/inference agents,
            pass False to skip optimizer loading entirely.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Core networks: accept missing/new critic aux heads for backward compatibility
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)

        # Optimizers: best-effort restore; fall back to fresh states if shapes changed
        if load_optimizers:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except ValueError as e:
                print(f"Warning: could not load optimizer state from {filepath}: {e}")

        # Reset memory states after loading
        self._player_memories = {}

```

### 1. Semantics: counting bury points only for the picker

**Yes, I agree with your caveat and implemented it that way.**

- **What’s “known” to each seat**:
  - Every player can see who took which tricks and thus the public `points_taken` per seat.
  - Only the **picker** ever sees the bury cards; defenders never see their identities or point values mid‑hand.
- **Aux target semantics**:
  - For a given acting player, the target vector is **“points each seat is known to have” from this seat’s information set**:
    - Base is `game.points_taken` (which already includes blind points in leasters and UNDER points when used).
    - **If and only if** the acting player is the picker in a non‑leaster game and the bury is fixed (`len(game.bury) == 2`), we add `get_trick_points(game.bury)` to **their own** slot.
    - Defenders never get that bury added; for them, those points remain unknown and thus excluded.
- **Leasters**:
  - In leasters, there is no bury; the blind is added to the first trick and therefore already reflected in `points_taken`. The helper simply does **not** add any extra bury value in that mode.

This matches your intent: **picker sees bury points; everyone else only sees what a perfect memory observer could count from open play.**

---

### 2. Target structure and relative seating

The auxiliary head predicts **per‑player current points** in **relative seat order** matching `trick_card_ids`:

- **Length‑5 vector output**: index 0 = self, 1 = LHO, 2 = +2, 3 = +3, 4 = +4.
- **Targets are normalized** by 120 so each value lies in \([0, 1]\).

The shared helper in `training_utils.py` builds this label:

```64:87:training_utils.py
def compute_known_points_rel(game, player):
    """Return normalized known point totals for all seats from the acting player's perspective.

    Known points include:
      - Public trick points in game.points_taken (including blind/UNDER handling).
      - Bury card points, but only for the picker in non-leaster games once the bury is fixed.

    The result is a length-5 list ordered by relative seat:
      1 = self, 2 = left-hand opponent, ..., 5 = right-hand opponent.
    Values are normalized by TOTAL_DECK_POINTS so they lie in [0, 1].
    """
    base_points = list(getattr(game, "points_taken", [0, 0, 0, 0, 0]))

    # In standard games, the picker uniquely knows the bury card identities and their points.
    # We expose those points only to the picker, and only once the bury is fixed.
    if not getattr(game, "is_leaster", False):
        if getattr(player, "is_picker", False) and len(getattr(game, "bury", [])) == 2:
            base_points[player.position - 1] += get_trick_points(game.bury)

    rel_points = []
    for rel in range(1, 6):
        abs_seat = ((player.position + rel - 2) % 5) + 1
        rel_points.append(base_points[abs_seat - 1] / TOTAL_DECK_POINTS)
    return rel_points
```

This uses the same relative mapping as `get_state_dict` (`trick_card_ids`), so the **critic head’s index 0..4 lines up exactly with your existing relative seat convention**.

---

### 3. How the aux point prediction is wired into training

#### 3.1. Label generation in the training loops

Both PPO (`train_ppo.py`) and PFSP (`train_pfsp.py`) now attach a **per‑state, per‑player point label** at action time:

- **In `train_ppo.py`**, each transition records:

```219:230:train_ppo.py
transition = {
    'kind': 'action',
    'player': player,
    'state': state,
    'action': action,
    'log_prob': log_prob,
    'value': value,
    'valid_actions': valid_actions.copy(),
    'intermediate_reward': 0.0,
    'secret_partner_label': 1.0 if player.is_secret_partner else 0.0,
    'points_label': compute_known_points_rel(game, player),
}
```

And when building the stored events:

```309:320:train_ppo.py
events.append({
    'kind': 'action',
    'state': ev['state'],
    'action': ev['action'],
    'log_prob': ev['log_prob'],
    'value': ev['value'],
    'valid_actions': ev['valid_actions'],
    'reward': reward,
    'player_id': pos,
    'win_label': 1.0 if episode_scores[pos - 1] > 0 else 0.0,
    'final_return_label': float(episode_scores[pos - 1]),
    'secret_partner_label': ev.get('secret_partner_label', 0.0),
    'points_label': ev.get('points_label', None),
})
```

- **In `train_pfsp.py`**, the PFSP training agent’s transitions get the same label:

```191:207:train_pfsp.py
if current_agent == training_agent:
    action, log_prob, value = current_agent.act(state, valid_actions, player.position)

    # Store transition for training agent
    transition = {
        'kind': 'action',
        'player': player,
        'state': state,
        'action': action,
        'log_prob': log_prob,
        'value': value,
        'valid_actions': valid_actions.copy(),
        'intermediate_reward': 0.0,
        'secret_partner_label': 1.0 if player.is_secret_partner else 0.0,
        'points_label': compute_known_points_rel(game, player),
    }
```

And when converting to `episode_events`:

```290:303:train_pfsp.py
episode_events.append({
    'kind': 'action',
    'state': ev['state'],
    'action': ev['action'],
    'log_prob': ev['log_prob'],
    'value': ev['value'],
    'valid_actions': ev['valid_actions'],
    'reward': reward_map[id(ev)],
    'player_id': seat_pos,
    'win_label': 1.0 if final_scores[seat_pos - 1] > 0 else 0.0,
    'final_return_label': float(final_scores[seat_pos - 1]),
    'secret_partner_label': ev.get('secret_partner_label', 0.0),
    'points_label': ev.get('points_label', None),
})
```

#### 3.2. Storing these labels in PPO’s event buffer

`PPOAgent.store_episode_events` now pulls the label for every action event, defaulting to zeros if missing:

```792:810:ppo.py
if ev.get('kind') == 'action':
    mask = self.get_action_mask(ev['valid_actions'], self.action_size)
    self.events.append({
        'kind': 'action',
        'state': ev['state'],
        'mask': mask,
        'action': int(ev['action']) - 1,
        'reward': float(ev['reward']),
        'value': float(ev['value']),
        'log_prob': float(ev['log_prob']),
        'done': (idx == last_action_idx),
        'player_id': pid,
        'win': float(ev.get('win_label', 0.0) or 0.0),
        'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
        'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
        'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
    })
```

#### 3.3. Critic architecture: new per‑player points head

The critic now has a **5‑dimensional aux head** on top of its shared adapter:

```295:309:ppo.py
self.critic_adapter = nn.Sequential(
    nn.LayerNorm(256),
    nn.Linear(256, 256),
    nn.ReLU() if activation == 'relu' else nn.SiLU()
)
self.value_head = nn.Linear(256, 1)
# Auxiliary heads
self.win_head = nn.Linear(256, 1)
self.return_head = nn.Linear(256, 1)
self.secret_partner_head = nn.Linear(256, 1)
# Per-player point prediction head (relative seating order, length-5 vector)
self.points_head = nn.Linear(256, 5)
```

During the sequence‑wise vectorized forward pass, this head produces a `(B, T, 5)` tensor `points_pred_bt`:

```1057:1063:ppo.py
aux_feat_bt = self.critic.critic_adapter(features_bt)
win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
points_pred_bt = self.critic.points_head(aux_feat_bt)

return logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
```

#### 3.4. Batching and loss

In the minibatch builder, the per‑step labels are collected and padded into `(B, T, 5)`:

```920:996:ppo.py
points_list_all = []
...
points_bt = []
for i in ev_range:
    if kinds[i] == 'action' and pids[i] == pid:
        ...
        pts_lbl = self.events[i].get('points_rel', [0.0] * 5)
        points_bt.append(torch.tensor(pts_lbl, dtype=torch.float32, device=device))
    else:
        ...
        points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
...
points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
...
return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt, secret_bt, points_bt
```

In the PPO update, after computing the usual policy, value, win/return/secret losses, we add a **smooth‑L1 loss over the per‑player point predictions**, masking to action timesteps:

```1326:1437:ppo.py
(
    states_bt,
    masks_bt,
    is_action_bt,
    actions_bt,
    old_lp_bt,
    returns_bt,
    adv_bt,
    lengths_bt,
    win_bt,
    final_ret_bt,
    secret_bt,
    points_bt,
) = self._build_minibatch_tensors(...)

logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt = self._forward_vectorized(...)

...
bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)
secret_loss = F.binary_cross_entropy_with_logits(secret_logits_flat, secret_labels_flat)
# Per-player points auxiliary loss (regression on normalized per-seat totals)
# points_pred_flat and labels are (N, 5); smooth L1 stabilizes training.
points_pred_flat = points_pred_bt.view(-1, points_pred_bt.size(-1))[is_action_bt.view(-1)]
points_labels_flat = points_bt.view(-1, points_bt.size(-1))[is_action_bt.view(-1)]
points_loss = F.smooth_l1_loss(points_pred_flat, points_labels_flat)

total_loss = (
    actor_loss
    + self.value_loss_coeff * critic_loss
    + self.win_loss_coeff * bce_loss
    + self.return_loss_coeff * return_loss
    + self.secret_loss_coeff * secret_loss
    + self.points_loss_coeff * points_loss
)
```

You can tune the **strength of this auxiliary task** via:

```455:458:ppo.py
self.win_loss_coeff = 0.1
self.return_loss_coeff = 0.1
self.secret_loss_coeff = 0.07
self.points_loss_coeff = 0.1
```

If you want to be cautious initially, you can lower `points_loss_coeff` (e.g. `0.03–0.05`) and watch:

- Whether point prediction error drops reliably, and  
- Whether endgame decisions that depend on tight score margins improve.

---

### 4. Summary of behavior

- **What’s being predicted?** A 5‑dim vector giving each seat’s **known normalized points** from the acting seat’s perspective, in relative order (self, LHO, …), including bury only for picker post‑bury.
- **Where is it used?** As a critic aux head, sharing the same 256‑d trunk as your value and other aux heads; gradients flow back through the encoder and trunk.
- **Information‑set alignment:** Non‑picker seats never get targets that require secretly knowing bury points; only the picker sees those points as part of their label.

This should push the shared trunk to learn precise score‑tracking representations that your actor can exploit, without violating imperfect‑information constraints.

```python
def compute_known_points_rel(game, player):
    """Return normalized known point totals for all seats from the acting player's perspective.

    Known points include:
      - Public trick points in game.points_taken (including blind/UNDER handling).
      - Bury card points, but only for the picker in non-leaster games once the bury is fixed.

    The result is a length-5 list ordered by relative seat:
      1 = self, 2 = left-hand opponent, ..., 5 = right-hand opponent.
    Values are normalized by TOTAL_DECK_POINTS so they lie in [0, 1].
    """
    base_points = list(getattr(game, "points_taken", [0, 0, 0, 0, 0]))

    # In standard games, the picker uniquely knows the bury card identities and their points.
    # We expose those points only to the picker, and only once the bury is fixed.
    if not getattr(game, "is_leaster", False):
        if getattr(player, "is_picker", False) and len(getattr(game, "bury", [])) == 2:
            base_points[player.position - 1] += get_trick_points(game.bury)

    rel_points = []
    for rel in range(1, 6):
        abs_seat = ((player.position + rel - 2) % 5) + 1
        rel_points.append(base_points[abs_seat - 1] / TOTAL_DECK_POINTS)
    return rel_points
```

```python
transition = {
    'kind': 'action',
    'player': player,
    'state': state,
    'action': action,
    'log_prob': log_prob,
    'value': value,
    'valid_actions': valid_actions.copy(),
    'intermediate_reward': 0.0,
    'secret_partner_label': 1.0 if player.is_secret_partner else 0.0,
    'points_label': compute_known_points_rel(game, player),
}
```

```python
events.append({
    'kind': 'action',
    'state': ev['state'],
    'action': ev['action'],
    'log_prob': ev['log_prob'],
    'value': ev['value'],
    'valid_actions': ev['valid_actions'],
    'reward': reward,
    'player_id': pos,
    'win_label': 1.0 if episode_scores[pos - 1] > 0 else 0.0,
    'final_return_label': float(episode_scores[pos - 1]),
    'secret_partner_label': ev.get('secret_partner_label', 0.0),
    'points_label': ev.get('points_label', None),
})
```

```python
if current_agent == training_agent:
    action, log_prob, value = current_agent.act(state, valid_actions, player.position)

    # Store transition for training agent
    transition = {
        'kind': 'action',
        'player': player,
        'state': state,
        'action': action,
        'log_prob': log_prob,
        'value': value,
        'valid_actions': valid_actions.copy(),
        'intermediate_reward': 0.0,
        'secret_partner_label': 1.0 if player.is_secret_partner else 0.0,
        'points_label': compute_known_points_rel(game, player),
    }
```

```python
episode_events.append({
    'kind': 'action',
    'state': ev['state'],
    'action': ev['action'],
    'log_prob': ev['log_prob'],
    'value': ev['value'],
    'valid_actions': ev['valid_actions'],
    'reward': reward_map[id(ev)],
    'player_id': seat_pos,
    'win_label': 1.0 if final_scores[seat_pos - 1] > 0 else 0.0,
    'final_return_label': float(final_scores[seat_pos - 1]),
    'secret_partner_label': ev.get('secret_partner_label', 0.0),
    'points_label': ev.get('points_label', None),
})
```

```python
if ev.get('kind') == 'action':
    mask = self.get_action_mask(ev['valid_actions'], self.action_size)
    self.events.append({
        'kind': 'action',
        'state': ev['state'],
        'mask': mask,
        'action': int(ev['action']) - 1,
        'reward': float(ev['reward']),
        'value': float(ev['value']),
        'log_prob': float(ev['log_prob']),
        'done': (idx == last_action_idx),
        'player_id': pid,
        'win': float(ev.get('win_label', 0.0) or 0.0),
        'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
        'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
        'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
    })
```

```python
self.critic_adapter = nn.Sequential(
    nn.LayerNorm(256),
    nn.Linear(256, 256),
    nn.ReLU() if activation == 'relu' else nn.SiLU()
)
self.value_head = nn.Linear(256, 1)
# Auxiliary heads
self.win_head = nn.Linear(256, 1)
self.return_head = nn.Linear(256, 1)
self.secret_partner_head = nn.Linear(256, 1)
# Per-player point prediction head (relative seating order, length-5 vector)
self.points_head = nn.Linear(256, 5)
```

```python
aux_feat_bt = self.critic.critic_adapter(features_bt)
win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
points_pred_bt = self.critic.points_head(aux_feat_bt)

return logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
```

```python
points_list_all = []
...
points_bt = []
for i in ev_range:
    if kinds[i] == 'action' and pids[i] == pid:
        ...
        pts_lbl = self.events[i].get('points_rel', [0.0] * 5)
        points_bt.append(torch.tensor(pts_lbl, dtype=torch.float32, device=device))
    else:
        ...
        points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
...
points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
...
return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt, secret_bt, points_bt
```

```python
(
    states_bt,
    masks_bt,
    is_action_bt,
    actions_bt,
    old_lp_bt,
    returns_bt,
    adv_bt,
    lengths_bt,
    win_bt,
    final_ret_bt,
    secret_bt,
    points_bt,
) = self._build_minibatch_tensors(...)

logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt = self._forward_vectorized(...)

...
bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)
secret_loss = F.binary_cross_entropy_with_logits(secret_logits_flat, secret_labels_flat)
# Per-player points auxiliary loss (regression on normalized per-seat totals)
# points_pred_flat and labels are (N, 5); smooth L1 stabilizes training.
points_pred_flat = points_pred_bt.view(-1, points_pred_bt.size(-1))[is_action_bt.view(-1)]
points_labels_flat = points_bt.view(-1, points_bt.size(-1))[is_action_bt.view(-1)]
points_loss = F.smooth_l1_loss(points_pred_flat, points_labels_flat)

total_loss = (
    actor_loss
    + self.value_loss_coeff * critic_loss
    + self.win_loss_coeff * bce_loss
    + self.return_loss_coeff * return_loss
    + self.secret_loss_coeff * secret_loss
    + self.points_loss_coeff * points_loss
)
```

```python
self.win_loss_coeff = 0.1
self.return_loss_coeff = 0.1
self.secret_loss_coeff = 0.07
self.points_loss_coeff = 0.1
```

---

**User**

I see that the predicted value is being normalized [0, 1] and know that this is more optimal for the algorithm. However, I would like to be able to track the accuracy of this prediction in a more human readable way. Is it viable to remove this normalization, or should instead apply a scaling factor to the prediction to produce the real predicted score if I need it in a human readable form?

---

**Cursor**

Keeping the predictions normalized is still the better choice for learning—you get smaller, more numerically stable targets, consistent gradient scales, and you avoid having the aux loss dominate the value/actor updates as scores climb toward 120. (You can see the normalization happening here:)

```64:87:training_utils.py
def compute_known_points_rel(...):
    ...
    rel_points.append(base_points[abs_seat - 1] / TOTAL_DECK_POINTS)
```

For human-readable monitoring you don’t need to give up those benefits. Just rescale the model outputs or the tracked labels when you log them:

- Multiply the predicted vector (or the target) by `TOTAL_DECK_POINTS` (=120) whenever you want to display “real” points.
- Alternatively, compute your metrics (MAE, RMSE, confusion around the 61+ margin, etc.) in normalized space, and only convert to points when presenting the results.

If you later decide you want both, you can even track a metric in points by doing something like `mae_points = mae_norm * TOTAL_DECK_POINTS` without touching the training targets at all.

So: leave the normalization in place for training, and apply a simple scaling factor (×120) wherever you need human-friendly numbers.

```python
def compute_known_points_rel(...):
    ...
    rel_points.append(base_points[abs_seat - 1] / TOTAL_DECK_POINTS)
```

---

**User**

Now, add this point prediction to the React app output at web/app/analyze for each timestep.



All other aux predictions are displayed in the MetricsBar. I would like to do something similar here, but put the secretPartnerProb (currently in the metrics bar) and the new point predictions in a new area that is only visible when the ActionRow is expanded. Make it look beautiful. Use the real point values (scaled by multiplying the normalized head output by TOTAL_DECK_POINTS on the backend).

---

**Cursor**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Dict
from sheepshead import ACTION_IDS, BURY_ACTIONS, CALL_ACTIONS, UNDER_ACTIONS, PLAY_ACTIONS, DECK_IDS, UNDER_TOKEN
from encoder import CardReasoningEncoder, CardEmbeddingConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network consuming encoder features, with separate heads for
    pick / partner-selection / bury / play phases. The four heads' logits are
    concatenated back into the full action space order so existing masking
    logic continues to work unchanged.
    """

    def __init__(self,
                 action_size,
                 action_groups,
                 activation='swish',
                 *,
                 d_card: int,
                 d_token: int,
                 map_cid_to_play_action_index: torch.Tensor,
                 map_cid_to_bury_action_index: torch.Tensor,
                 map_cid_to_under_action_index: torch.Tensor,
                 call_action_global_indices: torch.Tensor,
                 call_card_ids: torch.Tensor,
                 play_under_action_index: int):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'

        # Shared actor adapter to add nonlinearity and specialization before heads
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
        )

        # === Heads ===
        self.pick_head = nn.Linear(256, len(action_groups['pick']))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(256, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(256, 1)

        # Per-head temperatures (τ): logits will be divided by τ before softmax
        # τ > 1.0 softens (higher entropy), τ < 1.0 sharpens. Defaults to 1.0.
        self.temperature_pick = 1.0
        self.temperature_partner = 1.0
        self.temperature_bury = 1.0
        self.temperature_play = 1.0

        # Init (generic weights)
        self.apply(self._init_weights)

        # ---- Card-conditioned configuration (dims + mappings) ----
        self._d_card = int(d_card)
        self._d_token = int(d_token)
        # Pointer scorer (Bahdanau style)
        self.pointer_hidden = 64
        self.pointer_Wg = nn.Linear(256, self.pointer_hidden)
        self.pointer_Wt = nn.Linear(self._d_token, self.pointer_hidden)
        self.pointer_v = nn.Linear(self.pointer_hidden, 1, bias=False)
        # Two-tower (card CALL scoring)
        self.tw_latent = 64
        self.tw_Wg = nn.Linear(256, self.tw_latent)
        self.tw_We = nn.Linear(self._d_card, self.tw_latent)

        # Action index mappings
        self._map_cid_to_play_action_index = map_cid_to_play_action_index.clone().long()
        self._map_cid_to_bury_action_index = map_cid_to_bury_action_index.clone().long()
        self._map_cid_to_under_action_index = map_cid_to_under_action_index.clone().long()
        self._call_action_global_indices = call_action_global_indices.clone().long()
        self._call_card_ids = call_card_ids.clone().long()
        self._play_under_action_index = int(play_under_action_index)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------ internal helpers ------------------------
    def _score_cards_two_tower(self, feat: torch.Tensor, card_embedding: nn.Embedding) -> torch.Tensor:
        """Return per-card scores S for all 34 card ids using two-tower scorer.
        feat: (B, 256)
        Returns: (B, 34)
        """
        q = self.tw_Wg(feat)                      # (B, k)        <- tower 1 (actor features)
        table = card_embedding.weight             # (34, d_card)
        K = self.tw_We(table)                     # (34, k)       <- tower 2 (card embeddings)
        S = torch.matmul(q, K.t())    # (B, 34)       <- similarity scores
        return S

    def _score_hand_pointer(
        self,
        feat: torch.Tensor,
        hand_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score each hand slot using pointer scorer.
        feat: (B, 256); hand_tokens: (B, N, d_token)
        Returns: s_slots (B, N)
        """
        tok = hand_tokens
        B, N = tok.size(0), tok.size(1)
        g = self.pointer_Wg(feat).unsqueeze(1).expand(B, N, -1)  # (B, N, h)
        t = self.pointer_Wt(tok)                                   # (B, N, h)
        e = torch.tanh(g + t)                                      # (B, N, h)
        s = self.pointer_v(e).squeeze(-1)                          # (B, N)
        return s

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        hand_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        hand_tokens : Tensor (K, N, d_token)
        action_mask : BoolTensor (K, A) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # Adapt features
        feat = self.actor_adapter(actor_features)

        # PICK / PASS
        pick_logits = self.pick_head(feat) / max(self.temperature_pick, 1e-6)
        logits[:, self.action_groups['pick']] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(feat) / max(self.temperature_partner, 1e-6)
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(feat, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)] / max(self.temperature_partner, 1e-6)
        logits[:, self._call_action_global_indices.to(device)] = call_scores

        # Pointer scores over hand tokens (compute once, reuse for bury/under/play)
        slot_scores = self._score_hand_pointer(
            feat,
            hand_tokens,
        )  # (K, N)
        K_, N = slot_scores.size(0), slot_scores.size(1)
        cids = hand_ids.long()

        # BURY and UNDER scatter
        idx_bury = self._map_cid_to_bury_action_index.to(device)[cids]   # (K, N)
        idx_under = self._map_cid_to_under_action_index.to(device)[cids] # (K, N)
        for i in range(N):
            b_idx = idx_bury[:, i]
            u_idx = idx_under[:, i]
            valid_b = b_idx.ge(0)
            valid_u = u_idx.ge(0)
            if valid_b.any():
                logits.view(K_, -1)[valid_b, b_idx[valid_b]] = slot_scores[valid_b, i] / max(self.temperature_bury, 1e-6)
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[valid_u, i] / max(self.temperature_bury, 1e-6)

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[valid_p, i] / max(self.temperature_play, 1e-6)

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(feat).squeeze(-1) / max(self.temperature_play, 1e-6)
            logits[:, self._play_under_action_index] = play_under_logit

        # Apply action mask if provided
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def set_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Set per-head softmax temperatures.

        Parameters
        ----------
        pick, partner, bury, play : float | None
            If provided, sets the corresponding head's temperature τ. Values are
            clamped to a small positive minimum to avoid divide-by-zero.
        """
        eps = 1e-6
        if pick is not None:
            self.temperature_pick = max(float(pick), eps)
        if partner is not None:
            self.temperature_partner = max(float(partner), eps)
        if bury is not None:
            self.temperature_bury = max(float(bury), eps)
        if play is not None:
            self.temperature_play = max(float(play), eps)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features' and 'hand_tokens'
        action_mask : Bool Tensor (batch, action_size)
        hand_ids : Tensor (batch, 8)
        card_embedding : nn.Embedding

        Returns
        -------
        probs : Tensor (batch, action_size)
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_with_logits(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass that also returns pre-softmax logits.

        Returns
        -------
        probs : Tensor
        logits : Tensor
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs, logits



class RecurrentCriticNetwork(nn.Module):
    """Critic head using encoder features directly."""

    def __init__(self, activation='swish'):
        super().__init__()

        self.critic_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU()
        )
        self.value_head = nn.Linear(256, 1)
        # Auxiliary heads
        self.win_head = nn.Linear(256, 1)
        self.return_head = nn.Linear(256, 1)
        self.secret_partner_head = nn.Linear(256, 1)
        # Per-player point prediction head (relative seating order, length-5 vector)
        self.points_head = nn.Linear(256, 5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_out: Dict[str, torch.Tensor]):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        value : Tensor (batch, 1)
        """
        feat = self.critic_adapter(encoder_out['features'])
        value = self.value_head(feat)
        return value

    def aux_predictions(self, encoder_out: Dict[str, torch.Tensor]):
        """Return auxiliary predictions as scalars and per-seat point estimates.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        win_prob : float
            Sigmoid of win logits
        expected_final_return : float
            Linear head output for expected final return
        secret_partner_prob : float
            Sigmoid of secret partner logits
        points_vector : list[float] | None
            Normalized per-seat point predictions in relative seat order.
        """
        with torch.no_grad():
            aux_feat = self.critic_adapter(encoder_out['features'])
            win_logit = self.win_head(aux_feat).squeeze(-1)
            expected_return = self.return_head(aux_feat).squeeze(-1)
            secret_logit = self.secret_partner_head(aux_feat).squeeze(-1)
            points_pred = torch.clamp(self.points_head(aux_feat), min=0.0, max=1.0)
        win_prob_t = torch.sigmoid(win_logit)
        win_prob = float(win_prob_t.item())
        expected_final = float(expected_return.item())
        secret_prob = float(torch.sigmoid(secret_logit).item())
        points_vector: list[float] | None = None
        if points_pred.dim() >= 2 and points_pred.size(0) > 0:
            points_vector = [float(v) for v in points_pred[0].tolist()]
        return win_prob, expected_final, secret_prob, points_vector


class PPOAgent:
    def __init__(self, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
        # Dict-based encoder → fixed 256-d features
        self.state_size = 256
        self.action_size = action_size

        # --------------------------------------------------
        # Action groups for multi–head policy
        # --------------------------------------------------
        pick_indices = [ACTION_IDS["PICK"] - 1, ACTION_IDS["PASS"] - 1]

        # Partner-selection head: ALONE, JD PARTNER, all partner CALL actions
        partner_selection_actions = ["ALONE", "JD PARTNER"] + CALL_ACTIONS
        partner_indices = sorted({ACTION_IDS[a] - 1 for a in partner_selection_actions})

        # Bury head: explicit bury actions and UNDER actions
        bury_indices = sorted({ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + UNDER_ACTIONS)})

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            'pick': sorted(pick_indices),
            'partner': sorted(partner_indices),
            'bury': sorted(bury_indices),
            'play': sorted(play_indices),
        }

        # Explicit indices for PICK and PASS actions (0-indexed)
        self.pick_action_index = ACTION_IDS["PICK"] - 1
        self.pass_action_index = ACTION_IDS["PASS"] - 1

        # Encoder with memory
        self.encoder = CardReasoningEncoder(card_config=CardEmbeddingConfig()).to(device)

        # Per-player memory tracking
        self._player_memories = {}

        # Build mappings before constructing actor
        (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            call_action_global_indices,
            call_card_ids,
            play_under_action_index,
        ) = self._build_action_index_mappings()

        self.actor = MultiHeadRecurrentActorNetwork(
            action_size,
            self.action_groups,
            activation=activation,
            d_card=self.encoder.d_card_dim,
            d_token=self.encoder.d_token_dim,
            map_cid_to_play_action_index=map_cid_to_play_action_index,
            map_cid_to_bury_action_index=map_cid_to_bury_action_index,
            map_cid_to_under_action_index=map_cid_to_under_action_index,
            call_action_global_indices=call_action_global_indices,
            call_card_ids=call_card_ids,
            play_under_action_index=play_under_action_index,
        ).to(device)

        self.critic = RecurrentCriticNetwork(activation=activation).to(device)

        # Optimizers (include encoder params with scaled LR for card embeddings)
        encoder_groups = self.encoder.param_groups(base_lr=lr_actor, card_lr_scale=0.2)
        actor_groups = [
            {'params': self.actor.parameters(), 'lr': lr_actor},
            *encoder_groups,
        ]
        self.actor_optimizer = optim.Adam(actor_groups)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.95
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_partner = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.3
        self.clip_epsilon_pick = 0.20
        self.clip_epsilon_partner = 0.25
        self.clip_epsilon_bury = 0.15
        self.clip_epsilon_play = 0.20
        self.value_clip_epsilon = 0.20

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.0

        # Auxiliary loss coefficients
        self.win_loss_coeff = 0.1
        self.return_loss_coeff = 0.1
        self.secret_loss_coeff = 0.07
        self.points_loss_coeff = 0.1

        # Storage for trajectory data
        self.reset_storage()

        # --------------------------------------------------
        # Partner-call mixture: epsilon floor over CALL actions
        # --------------------------------------------------
        # Subset of partner head actions eligible for CALL-uniform mixing
        self.partner_call_subindices = sorted(
            {ACTION_IDS["JD PARTNER"] - 1} | {ACTION_IDS[a] - 1 for a in CALL_ACTIONS}
        )
        # Blending coefficient ε (0 disables; typical 0.02–0.10)
        self.partner_call_epsilon = 0.0

        # PASS-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pass_floor_epsilon = 0.0
        # PICK-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pick_floor_epsilon = 0.0

    def get_recurrent_memory(self, player_id: int | None, device: torch.device | None = None) -> torch.Tensor:
        """Return the recurrent memory vector for a player, or a zero vector if unset.

        The returned tensor is (256,) on the requested device.
        """
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mem = self._player_memories.get(player_id) if player_id is not None else None
        if mem is None:
            return torch.zeros(256, device=target_device)
        return mem.to(target_device) if mem.device != target_device else mem

    def set_recurrent_memory(self, player_id: int | None, memory_vector: torch.Tensor) -> None:
        """Set the recurrent memory vector for a player."""
        self._player_memories[player_id] = memory_vector

    def _build_action_index_mappings(self):
        """Precompute global action index mappings for card-specific actions."""
        map_cid_to_play_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_bury_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_under_action_index = torch.full((34,), -1, dtype=torch.long)

        # PLAY actions
        play_under_index = None
        for a in PLAY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            if a == f"PLAY {UNDER_TOKEN}":
                play_under_index = a_idx
                continue
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_play_action_index[cid] = a_idx

        # BURY actions
        for a in BURY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_bury_action_index[cid] = a_idx

        # UNDER actions (choose card from hand to set as under)
        for a in UNDER_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_under_action_index[cid] = a_idx

        # CALL actions
        idx_call_global_list = []
        call_card_ids_list = []
        for a in CALL_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            parts = a.split()
            # a is either "CALL X" or "CALL X UNDER"
            card = parts[1]
            cid = DECK_IDS[card]
            idx_call_global_list.append(a_idx)
            call_card_ids_list.append(cid)

        idx_call_global = torch.tensor(sorted(idx_call_global_list), dtype=torch.long)
        # Align cid list to the sorted order of indices
        # Build mapping from index to cid then reorder
        idx_to_cid = {idx: cid for idx, cid in zip(idx_call_global_list, call_card_ids_list)}
        call_card_ids = torch.tensor([idx_to_cid[int(x)] for x in idx_call_global.tolist()], dtype=torch.long)

        return map_cid_to_play_action_index, map_cid_to_bury_action_index, map_cid_to_under_action_index, idx_call_global, call_card_ids, play_under_index

    def set_head_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Convenience proxy to set per-head temperatures on the actor."""
        self.actor.set_temperatures(pick=pick, partner=partner, bury=bury, play=play)

    def set_partner_call_epsilon(self, eps: float):
        """Set ε for uniform CALL mixture on partner steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.partner_call_epsilon = eps

    def set_pass_floor_epsilon(self, eps: float):
        """Set ε for PASS-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pass_floor_epsilon = eps

    def set_pick_floor_epsilon(self, eps: float):
        """Set ε for PICK-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pick_floor_epsilon = eps

    def _apply_epsilon_mixing(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply partner-call ε mixture and PASS-floor ε mixture to probability
        distributions in a batched way without in-place ops.

        Parameters
        ----------
        probs : Tensor
            Shape (B, A) probabilities over actions
        mask : Bool Tensor
            Shape (B, A) of valid actions
        """
        mixed = probs
        if self.partner_call_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            call_vec = torch.zeros(A, dtype=torch.bool, device=mixed.device)
            call_idx = torch.tensor(self.partner_call_subindices, dtype=torch.long, device=mixed.device)
            call_vec[call_idx] = True
            valid_call = mask & call_vec.view(1, A).expand(B, A)
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(
                has,
                valid_call.float() / count.clamp_min(1.0),
                torch.zeros_like(mixed),
            )
            eps = torch.where(
                has,
                torch.full_like(count, self.partner_call_epsilon),
                torch.zeros_like(count),
            )
            mixed = (1.0 - eps) * mixed + eps * ucall

        if self.pick_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pick = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand(B, A)
            valid_pick = mask[:, self.pick_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * mixed + self.pick_floor_epsilon * one_hot_pick,
                mixed,
            )

        if self.pass_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pass = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand(B, A)
            valid_pass = mask[:, self.pass_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * mixed + self.pass_floor_epsilon * one_hot_pass,
                mixed,
            )

        return mixed

    def reset_storage(self):
        # Ordered list of events: each is a dict with keys:
        # kind: 'action' | 'obs'
        # state: np.array state vector
        # mask: torch.bool mask or convertible
        # If kind == 'action': action (0-indexed), reward, value, log_prob, done
        # GAE writes back: 'advantage' and 'return'
        self.events = []

    def get_action_mask(self, valid_actions, action_size):
        """Convert valid actions set to boolean mask"""
        mask = torch.zeros(action_size, dtype=torch.bool)
        for action in valid_actions:
            mask[action - 1] = True  # Convert from 1-indexed to 0-indexed
        return mask

    def reset_recurrent_state(self):
        """Clear memory states (call at the start of every new game)."""
        self._player_memories = {}

    def get_action_probs_with_logits(self, state, valid_actions, player_id=None):
        """Return post-mixture action probabilities and pre-mix logits for a single dict state.

        Applies partner CALL-uniform mixture if enabled. Keeps PPO on-policy by
        exposing the same transformed distribution that sampling uses.
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        # Encode dict state with memory
        encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

        # Store updated memory
        if player_id is not None:
            self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

        action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
        hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

        with torch.no_grad():
            probs, logits = self.actor.forward_with_logits(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )
            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                probs = self._apply_epsilon_mixing(probs, action_mask_t)

        return probs, logits

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        with torch.no_grad():
            # Get or init memory for this player
            memory_in = self.get_recurrent_memory(player_id, device=device)

            # Encode with memory
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

            # Get action probabilities
            action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
            hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

            action_probs = self.actor(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )

            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                action_probs = self._apply_epsilon_mixing(action_probs, action_mask_t)

            # Get value
            value = self.critic(encoder_out)

        # Create distribution for consistent log probability calculation
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()

        # Get log probability from the distribution for consistency
        log_prob = dist.log_prob(action)

        return action.item() + 1, log_prob.item(), value.item()  # Convert back to 1-indexed

    # ------------------------------------------------------------------
    # Observation-only step (no action sampled / no transition stored)
    # ------------------------------------------------------------------
    def observe(self, state, player_id=None):
        """Propagate an observation through the encoder to update memory
        *without* sampling an action or storing any transition.  Useful for
        non-decision environment ticks such as the end of a trick.

        Parameters
        ----------
        state : dict
            Structured observation for the player.
        player_id : int | None
            Identifier to associate a persistent memory state (1-5 in game).
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        with torch.no_grad():
            # Encode with memory to update memory state
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

    # ------------------------------------------------------------------
    # Episode-level ingestion API
    # ------------------------------------------------------------------
    def store_episode_events(self, events: list, player_id_default: int | None = None) -> None:
        """Store a single player's episode as a chronological list of events.

        Each event must be one of:
          - Observation:
              {
                'kind': 'observation',
                'state': dict,
                'player_id': int (optional if player_id_default provided)
              }
          - Action:
              {
                'kind': 'action',
                'state': dict,
                'action': int (1-indexed),
                'log_prob': float,
                'value': float,
                'valid_actions': set[int],
                'reward': float,
                'player_id': int (optional if player_id_default provided),
                'win_label': float (optional),
                'final_return_label': float (optional),
              }

        """
        # Identify last action index to set done flag
        last_action_idx = -1
        for idx, ev in enumerate(events):
            if ev.get('kind') == 'action':
                last_action_idx = idx

        for idx, ev in enumerate(events):
            pid = ev.get('player_id', player_id_default)
            if pid is None:
                pid = 0
            if ev.get('kind') == 'action':
                mask = self.get_action_mask(ev['valid_actions'], self.action_size)
                self.events.append({
                    'kind': 'action',
                    'state': ev['state'],
                    'mask': mask,
                    'action': int(ev['action']) - 1,
                    'reward': float(ev['reward']),
                    'value': float(ev['value']),
                    'log_prob': float(ev['log_prob']),
                    'done': (idx == last_action_idx),
                    'player_id': pid,
                    'win': float(ev.get('win_label', 0.0) or 0.0),
                    'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
                    'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
                    'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
                })
            else:
                # Observation: mask = all ones by default
                mask = torch.ones(self.action_size, dtype=torch.bool)
                self.events.append({
                    'kind': 'observation',
                    'state': ev['state'],
                    'mask': mask,
                    'player_id': pid,
                })

    def compute_gae(self, next_value=0):
        """Compute GAE per player over action events; write results back into events."""
        # Group action indices by player
        actions_by_player: dict[int | None, list[int]] = {}
        for i, e in enumerate(self.events):
            if e['kind'] == 'action':
                pid = e.get('player_id', None)
                actions_by_player.setdefault(pid, []).append(i)

        all_advantages = []
        all_returns = []
        for pid, idxs in actions_by_player.items():
            if not idxs:
                continue
            rewards = np.array([self.events[i]['reward'] for i in idxs])
            # Bootstrap next value as 0.0 per player sequence
            values = np.array([self.events[i]['value'] for i in idxs] + [0.0])
            dones = np.array([self.events[i]['done'] for i in idxs] + [False])

            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values[:-1]

            for i, adv, ret in zip(idxs, advantages, returns):
                self.events[i]['advantage'] = float(adv)
                self.events[i]['return'] = float(ret)

            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_advantages:
            return np.array([]), np.array([])
        return np.concatenate(all_advantages), np.concatenate(all_returns)

    # ------------------------------------------------------------------
    # Internal helpers for PPO update
    # ------------------------------------------------------------------
    def _prepare_training_views(self):
        # Keep raw states (dicts) to encode inside the update for gradient flow
        states = [e['state'] for e in self.events]
        masks_t = [
            (e['mask'].to(device) if isinstance(e['mask'], torch.Tensor) else torch.as_tensor(e['mask'], dtype=torch.bool, device=device))
            for e in self.events
        ]
        kinds = [e['kind'] for e in self.events]
        pids = [e.get('player_id', None) for e in self.events]
        return states, masks_t, kinds, pids

    @staticmethod
    def _index_events_by_player(pids: list[int | None]) -> dict[int | None, list[int]]:
        events_by_player: dict[int | None, list[int]] = {}
        for idx, pid in enumerate(pids):
            events_by_player.setdefault(pid, []).append(idx)
        return events_by_player

    def _segments_from_events(self, events_by_player: dict[int | None, list[int]], kinds: list[str]):
        segments: list[tuple[int | None, int, int]] = []
        for pid, ev_idxs in events_by_player.items():
            action_ev_idxs = [i for i in ev_idxs if kinds[i] == 'action']
            if not action_ev_idxs:
                continue
            # Build done flags for this player's actions in order
            dones_pid = [self.events[i]['done'] for i in action_ev_idxs]
            start = ev_idxs[0]
            a_ptr = 0
            for i in ev_idxs:
                if kinds[i] == 'action':
                    if dones_pid[a_ptr]:
                        segments.append((pid, start, i))
                        start = i + 1
                    a_ptr += 1
            if start <= ev_idxs[-1]:
                segments.append((pid, start, ev_idxs[-1]))
        return segments

    @staticmethod
    def _pad_to_bt(lst, lengths, fill):
        T = max(lengths)
        out = []
        for i, tlen in enumerate(lengths):
            pad = T - tlen
            if pad > 0:
                pad_shape = list(lst[i].shape)
                pad_shape[0] = pad
                pad_tensor = torch.full(
                    pad_shape,
                    fill,
                    device=lst[i].device,
                    dtype=lst[i].dtype,
                )
                out.append(torch.cat([lst[i], pad_tensor], dim=0))
            else:
                out.append(lst[i])
        return torch.stack(out, dim=0), T

    def _build_minibatch_tensors(self, batch, states, masks_t, kinds, pids):
        lengths = []
        states_seqs = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        returns_list = []
        adv_list = []
        win_list_all = []
        final_ret_list_all = []
        secret_list_all = []
        points_list_all = []

        for pid, seg_start, seg_end in batch:
            # Restrict sequence strictly to this player's own events (actions + obs)
            # rather than a contiguous global slice. This ensures each agent learns
            # only from its direct experience without global context bleed-through.
            ev_range = [i for i in range(seg_start, seg_end + 1) if pids[i] == pid]
            lengths.append(len(ev_range))
            states_seqs.append([states[i] for i in ev_range])
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if (kinds[i] == 'action' and pids[i] == pid) else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, ret_bt, adv_bt = [], [], [], []
            win_bt, final_ret_bt, secret_bt = [], [], []
            points_bt = []
            for i in ev_range:
                if kinds[i] == 'action' and pids[i] == pid:
                    act_bt.append(torch.tensor(self.events[i]['action'], dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(self.events[i]['log_prob'], dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(self.events[i]['return'], dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(self.events[i]['advantage'], dtype=torch.float32, device=device))
                    win_lbl = self.events[i].get('win', 0.0) or 0.0
                    final_return_lbl = self.events[i].get('final_return', 0.0) or 0.0
                    secret_lbl = self.events[i].get('secret_partner', 0.0) or 0.0
                    win_bt.append(torch.tensor(float(win_lbl), dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(float(final_return_lbl), dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(float(secret_lbl), dtype=torch.float32, device=device))
                    pts_lbl = self.events[i].get('points_rel', [0.0] * 5)
                    points_bt.append(torch.tensor(pts_lbl, dtype=torch.float32, device=device))
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    win_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))
            win_list_all.append(torch.stack(win_bt, dim=0))
            final_ret_list_all.append(torch.stack(final_ret_bt, dim=0))
            secret_list_all.append(torch.stack(secret_bt, dim=0))
            points_list_all.append(torch.stack(points_bt, dim=0))

        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        win_bt, _ = self._pad_to_bt(win_list_all, lengths, 0.0)
        final_ret_bt, _ = self._pad_to_bt(final_ret_list_all, lengths, 0.0)
        secret_bt, _ = self._pad_to_bt(secret_list_all, lengths, 0.0)
        points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt, secret_bt, points_bt

    def _forward_vectorized(self, states_input, masks_bt, lengths_bt):
        """Vectorized forward pass for training with recurrent memory.

        Args:
            states_input: List of B sequences, each a list of observation dicts
            masks_bt: (B, T, action_size) action masks
            lengths_bt: (B,) sequence lengths

        Returns:
            logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
        """
        B = len(states_input)
        T = masks_bt.size(1)

        # Initialize memory to zeros for each segment
        memory_init = torch.zeros((B, 256), device=device)

        # Encode sequences with memory
        encoder_out_seq = self.encoder.encode_sequences(
            states_input,
            memory_in=memory_init,
            device=device
        )

        # Extract features (B, T, 256)
        features_bt = encoder_out_seq['features']

        # Build hand_ids_bt (B, T, N) from dict states
        N = 8  # Maximum hand size
        hand_ids_bt = torch.zeros((B, T, N), dtype=torch.long, device=device)
        for b, seq in enumerate(states_input):
            for t, s in enumerate(seq):
                if t >= T:
                    break
                arr = torch.as_tensor(s['hand_ids'], dtype=torch.long, device=device)
                if arr.dim() == 1:
                    arr = arr.view(-1)
                hand_ids_bt[b, t, :min(N, arr.numel())] = arr[:N]

        # Flatten time dimension to reuse single helper, then reshape back
        flat_feat = features_bt.reshape(B * T, -1)
        flat_hand = hand_ids_bt.reshape(B * T, N)
        flat_mask = masks_bt.view(B * T, -1)
        hand_tokens_bt = encoder_out_seq['hand_tokens']  # (B, T, N, d_token)
        flat_tokens = hand_tokens_bt.reshape(B * T, N, -1)

        logits_flat = self.actor._build_logits_from_features(
            actor_features=flat_feat,
            hand_ids=flat_hand,
            card_embedding=self.encoder.card,
            hand_tokens=flat_tokens,
            action_mask=flat_mask,
        )
        logits_bt = logits_flat.view(B, T, -1)

        # Critic values
        critic_feat_bt = self.critic.critic_adapter(features_bt)
        values_bt = self.critic.value_head(critic_feat_bt).squeeze(-1)

        # Auxiliary preds share critic_adapter features (gradients flow to encoder)
        aux_feat_bt = self.critic.critic_adapter(features_bt)
        win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
        ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
        secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
        points_pred_bt = self.critic.points_head(aux_feat_bt)

        return logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt

    @staticmethod
    def _flatten_action_steps(
        is_action_bt,
        logits_bt,
        values_bt,
        actions_bt,
        old_lp_bt,
        returns_bt,
        adv_bt,
        win_logits_bt,
        ret_pred_bt,
        win_bt,
        final_ret_bt,
        secret_logits_bt,
        secret_bt,
        masks_bt,
    ):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            logits_bt.view(-1, logits_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
            win_logits_bt.view(-1)[flat_mask],
            ret_pred_bt.view(-1)[flat_mask],
            win_bt.view(-1)[flat_mask],
            final_ret_bt.view(-1)[flat_mask],
            secret_logits_bt.view(-1)[flat_mask],
            secret_bt.view(-1)[flat_mask],
            masks_bt.view(-1, masks_bt.size(-1))[flat_mask],
        )

    @staticmethod
    def _entropy_from_probs(sub):
        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

    def _head_entropies(self, probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t):
        probs_pick = probs_flat[:, pick_idx_t]
        probs_partner = probs_flat[:, partner_idx_t]
        probs_bury = probs_flat[:, bury_idx_t]
        probs_play = probs_flat[:, play_idx_t]
        return (
            self._entropy_from_probs(probs_pick),
            self._entropy_from_probs(probs_partner),
            self._entropy_from_probs(probs_bury),
            self._entropy_from_probs(probs_play),
        )

    def _per_head_weights(self, actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, template):
        is_pick = torch.isin(actions_flat, pick_idx_t)
        is_partner = torch.isin(actions_flat, partner_idx_t)
        is_bury = torch.isin(actions_flat, bury_idx_t)
        is_play = ~(is_pick | is_partner | is_bury)

        count_pick = is_pick.sum().item()
        count_partner = is_partner.sum().item()
        count_bury = is_bury.sum().item()
        count_play = is_play.sum().item()
        total_count = float(count_pick + count_partner + count_bury + count_play)
        heads_present = int((count_pick > 0) + (count_partner > 0) + (count_bury > 0) + (count_play > 0))

        def per_head_weight(count: int) -> float:
            if heads_present == 0 or count == 0:
                return 0.0
            return total_count / (heads_present * float(count))

        w_pick = per_head_weight(count_pick)
        w_partner = per_head_weight(count_partner)
        w_bury = per_head_weight(count_bury)
        w_play = per_head_weight(count_play)

        head_weight = torch.ones_like(template)
        if w_pick > 0.0:
            head_weight[is_pick] = head_weight[is_pick] * w_pick
        else:
            head_weight[is_pick] = 0.0
        if w_partner > 0.0:
            head_weight[is_partner] = head_weight[is_partner] * w_partner
        else:
            head_weight[is_partner] = 0.0
        if w_bury > 0.0:
            head_weight[is_bury] = head_weight[is_bury] * w_bury
        else:
            head_weight[is_bury] = 0.0
        if w_play > 0.0:
            head_weight[is_play] = head_weight[is_play] * w_play
        else:
            head_weight[is_play] = 0.0

        return head_weight, is_pick, is_partner, is_bury

    def _actor_critic_losses(
        self,
        logits_flat,
        mask_flat,
        actions_flat,
        old_lp_flat,
        values_flat,
        returns_flat,
        adv_flat,
        pick_idx_t,
        partner_idx_t,
        bury_idx_t,
        play_idx_t,
    ):
        # Build probabilities fresh from logits to avoid in-place softmax conflicts
        probs_all = F.softmax(logits_flat, dim=-1)
        # Partner-call mixture on-the-fly
        if self.partner_call_epsilon > 0.0:
            A = probs_all.size(-1)
            call_mask = torch.zeros(A, dtype=torch.bool, device=probs_all.device)
            call_mask[self.partner_call_subindices] = True
            valid_call = call_mask.view(1, A).expand_as(mask_flat) & mask_flat
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(has, valid_call.float() / count.clamp_min(1.0), torch.zeros_like(probs_all))
            eps = torch.where(has, torch.full_like(count, self.partner_call_epsilon), torch.zeros_like(count))
            probs_all = (1.0 - eps) * probs_all + eps * ucall

        # PICK-floor mixing on-the-fly
        if self.pick_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pick = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand_as(probs_all)
            valid_pick = mask_flat[:, self.pick_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * probs_all + self.pick_floor_epsilon * one_hot_pick,
                probs_all,
            )

        # PASS-floor mixing on-the-fly
        if self.pass_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pass = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand_as(probs_all)
            valid_pass = mask_flat[:, self.pass_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * probs_all + self.pass_floor_epsilon * one_hot_pass,
                probs_all,
            )

        dist = torch.distributions.Categorical(probs_all.clamp(min=1e-12))
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = self._head_entropies(
            probs_all, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
        )
        entropy_term = (
            self.entropy_coeff_pick * pick_entropy
            + self.entropy_coeff_partner * partner_entropy
            + self.entropy_coeff_bury * bury_entropy
            + self.entropy_coeff_play * play_entropy
        )

        ratios = torch.exp(new_lp_flat - old_lp_flat)
        head_weight, is_pick, is_partner, is_bury = self._per_head_weights(
            actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, ratios
        )

        eps_flat = torch.full_like(ratios, self.clip_epsilon_play)
        eps_flat[is_pick] = self.clip_epsilon_pick
        eps_flat[is_partner] = self.clip_epsilon_partner
        eps_flat[is_bury] = self.clip_epsilon_bury

        surr1 = ratios * adv_flat
        clipped = torch.clamp(ratios, 1 - eps_flat, 1 + eps_flat) * adv_flat
        pg_loss_elements = -torch.min(surr1, clipped)
        policy_loss = (pg_loss_elements * head_weight).mean()

        returns_target = returns_flat.view(-1)
        values_old = values_flat.detach()
        v_clipped = values_old + torch.clamp(
            values_flat - values_old, -self.value_clip_epsilon, self.value_clip_epsilon
        )
        critic_loss_unclipped = F.mse_loss(values_flat, returns_target)
        critic_loss_clipped = F.mse_loss(v_clipped, returns_target)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

        actor_loss = policy_loss - entropy_term + self.kl_coef * approx_kl_t
        return actor_loss, critic_loss, approx_kl_t, (pick_entropy, partner_entropy, bury_entropy, play_entropy)

    def update(self, epochs=6, batch_size=256):
        """Update actor and critic networks using PPO with recurrent unrolling.
        Includes performance optimisations and per-update timing logs.
        """
        t_update_start = time.time()
        if len(self.events) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Store statistics before normalization
        raw_advantages = advantages.copy() if advantages.size else np.array([0.0])
        advantage_stats = {
            'mean': float(np.mean(raw_advantages)),
            'std': float(np.std(raw_advantages)),
            'min': float(np.min(raw_advantages)),
            'max': float(np.max(raw_advantages))
        }

        value_target_stats = {
            'mean': float(np.mean(returns)) if advantages.size else 0.0,
            'std': float(np.std(returns)) if advantages.size else 0.0,
            'min': float(np.min(returns)) if advantages.size else 0.0,
            'max': float(np.max(returns)) if advantages.size else 0.0,
        }

        # Normalize advantages and write back into events so the loss uses normalized values
        if advantages.size:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for e in self.events:
                if e.get('kind') == 'action':
                    e['advantage'] = float((e['advantage'] - adv_mean) / adv_std)

        # Build static views and segments
        t_build_start = time.time()
        states, masks_t, kinds, pids = self._prepare_training_views()
        events_by_player = self._index_events_by_player(pids)
        segments = self._segments_from_events(events_by_player, kinds)

        # Precompute static index tensors once
        pick_idx_tensor_static = torch.tensor(self.action_groups['pick'], device=device)
        partner_idx_tensor_static = torch.tensor(self.action_groups['partner'], device=device)
        bury_idx_tensor_static = torch.tensor(self.action_groups['bury'], device=device)
        play_idx_tensor_static = torch.tensor(self.action_groups['play'], device=device)

        t_build_end = time.time()

        # Training epochs – vectorized by batching segments
        forward_time = 0.0
        backward_time = 0.0
        step_time = 0.0
        optimizer_steps = 0
        early_stop_triggered = False
        last_approx_kl = 0.0

        # Instrumentation accumulators
        ent_pick_sum = 0.0
        ent_partner_sum = 0.0
        ent_bury_sum = 0.0
        ent_play_sum = 0.0
        ent_batches = 0
        pick_adv_sum = 0.0
        pick_adv_count = 0
        pass_adv_sum = 0.0
        pass_adv_count = 0

        for _ in range(epochs):
            if not segments:
                break
            perm = torch.randperm(len(segments))
            for mb_start in range(0, len(segments), batch_size):
                batch_idxs = perm[mb_start:mb_start + batch_size].tolist()
                batch = [segments[i] for i in batch_idxs]
                if len(batch) == 0:
                    continue

                (
                    states_bt,
                    masks_bt,
                    is_action_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    lengths_bt,
                    win_bt,
                    final_ret_bt,
                    secret_bt,
                    points_bt,
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds, pids)

                # Vectorized forward
                t_fwd = time.time()
                logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt,
                    logits_bt,
                    values_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    win_logits_bt,
                    ret_pred_bt,
                    win_bt,
                    final_ret_bt,
                    secret_logits_bt,
                    secret_bt,
                    masks_bt,
                )
                if flat is None:
                    continue
                (
                    logits_flat,
                    values_flat,
                    actions_flat,
                    old_lp_flat,
                    returns_flat,
                    adv_flat,
                    win_logits_flat,
                    ret_pred_flat,
                    win_labels_flat,
                    final_ret_labels_flat,
                    secret_logits_flat,
                    secret_labels_flat,
                    mask_flat,
                ) = flat

                # Record PICK/PASS advantages across minibatches
                with torch.no_grad():
                    pick_mask_specific = (actions_flat == self.pick_action_index)
                    if pick_mask_specific.any():
                        pick_adv_sum += adv_flat[pick_mask_specific].sum().item()
                        pick_adv_count += int(pick_mask_specific.sum().item())
                    pass_mask_specific = (actions_flat == self.pass_action_index)
                    if pass_mask_specific.any():
                        pass_adv_sum += adv_flat[pass_mask_specific].sum().item()
                        pass_adv_count += int(pass_mask_specific.sum().item())

                # Losses and metrics
                (
                    actor_loss,
                    critic_loss,
                    approx_kl_t,
                    (pick_entropy, partner_entropy, bury_entropy, play_entropy),
                ) = self._actor_critic_losses(
                    logits_flat,
                    mask_flat,
                    actions_flat,
                    old_lp_flat,
                    values_flat,
                    returns_flat,
                    adv_flat,
                    pick_idx_tensor_static,
                    partner_idx_tensor_static,
                    bury_idx_tensor_static,
                    play_idx_tensor_static,
                )
                last_approx_kl = float(approx_kl_t.item())

                # Entropy accumulation
                ent_pick_sum += pick_entropy.detach().item()
                ent_partner_sum += partner_entropy.detach().item()
                ent_bury_sum += bury_entropy.detach().item()
                ent_play_sum += play_entropy.detach().item()
                ent_batches += 1

                # Auxiliary losses
                bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
                return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)
                secret_loss = F.binary_cross_entropy_with_logits(secret_logits_flat, secret_labels_flat)
                # Per-player points auxiliary loss (regression on normalized per-seat totals)
                # points_pred_flat and labels are (N, 5); smooth L1 stabilizes training.
                points_pred_flat = points_pred_bt.view(-1, points_pred_bt.size(-1))[is_action_bt.view(-1)]
                points_labels_flat = points_bt.view(-1, points_bt.size(-1))[is_action_bt.view(-1)]
                points_loss = F.smooth_l1_loss(points_pred_flat, points_labels_flat)

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = (
                    actor_loss
                    + self.value_loss_coeff * critic_loss
                    + self.win_loss_coeff * bce_loss
                    + self.return_loss_coeff * return_loss
                    + self.secret_loss_coeff * secret_loss
                    + self.points_loss_coeff * points_loss
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                step_time += time.time() - t_step
                optimizer_steps += 1

                # Early stop further updates in this epoch if KL exceeds threshold
                if self.target_kl is not None and last_approx_kl > self.target_kl:
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        transitions = sum(1 for e in self.events if e['kind'] == 'action')

        # Clear storage
        self.reset_storage()

        # Return training statistics
        t_end = time.time()
        timing = {
            'build_s': t_build_end - t_build_start,
            'forward_s': forward_time,
            'backward_s': backward_time,
            'step_s': step_time,
            'total_update_s': t_end - t_update_start,
            'optimizer_steps': optimizer_steps,
        }
        return {
            'advantage_stats': advantage_stats,
            'value_target_stats': value_target_stats,
            'num_transitions': transitions,
            'approx_kl': last_approx_kl,
            'early_stop': early_stop_triggered,
            'timing': timing,
            'head_entropy': {
                'pick': (ent_pick_sum / ent_batches) if ent_batches > 0 else 0.0,
                'partner': (ent_partner_sum / ent_batches) if ent_batches > 0 else 0.0,
                'bury': (ent_bury_sum / ent_batches) if ent_batches > 0 else 0.0,
                'play': (ent_play_sum / ent_batches) if ent_batches > 0 else 0.0,
            },
            'pick_pass_adv': {
                'pick_mean': (pick_adv_sum / pick_adv_count) if pick_adv_count > 0 else 0.0,
                'pick_count': pick_adv_count,
                'pass_mean': (pass_adv_sum / pass_adv_count) if pass_adv_count > 0 else 0.0,
                'pass_count': pass_adv_count,
            },
        }

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath, load_optimizers: bool = True):
        """Load model parameters.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizers : bool
            If True, also loads optimizer states. For population/inference agents,
            pass False to skip optimizer loading entirely.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Core networks: accept missing/new critic aux heads for backward compatibility
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)

        # Optimizers: best-effort restore; fall back to fresh states if shapes changed
        if load_optimizers:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except ValueError as e:
                print(f"Warning: could not load optimizer state from {filepath}: {e}")

        # Reset memory states after loading
        self._player_memories = {}

```

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CreateTableRequest(BaseModel):
    name: str
    fillWithAI: bool = True
    rules: Dict[str, Any] = {}


class JoinTableRequest(BaseModel):
    display_name: str
    seat: Optional[int] = None


class UpdateTableRulesRequest(BaseModel):
    client_id: str
    rules: Dict[str, Any]


class StartGameRequest(BaseModel):
    client_id: str
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    client_id: str
    action_id: int


class SeatRequest(BaseModel):
    client_id: str
    seat: int


class RedealRequest(BaseModel):
    # Host-triggered; require client_id for authorization
    client_id: Optional[str] = None


class CloseTableRequest(BaseModel):
    client_id: str


class TablePublic(BaseModel):
    id: str
    name: str
    status: str
    rules: Dict[str, Any]
    fillWithAI: bool
    seats: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    runningBySeat: Dict[int, int] | Dict[str, int]
    seatOccupants: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    host: Optional[str]
    hostId: Optional[str]
    resultsHistory: List[Dict[str, Any]]
    initialSeatOrder: List[str]
    initialNames: Dict[str, str]


# Analyze AI Model schemas
class AnalyzeSimulateRequest(BaseModel):
    seed: Optional[int] = None
    partnerMode: int = 1  # 0 = JD, 1 = Called Ace
    deterministic: bool = False
    modelPath: Optional[str] = None
    maxSteps: int = 200


class AnalyzeProbability(BaseModel):
    actionId: int
    action: str
    prob: float
    logit: float


class AnalyzePointEstimate(BaseModel):
    seat: int
    seatName: str
    points: float
    relativePosition: int


class AnalyzeActionDetail(BaseModel):
    stepIndex: int
    seat: int
    seatName: str
    phase: str  # "pick" | "partner" | "bury" | "play"
    actionId: int
    action: str
    valueEstimate: float
    discountedReturn: Optional[float] = None
    stepReward: Optional[float] = None
    winProb: Optional[float] = None  # [0,1]
    expectedFinalReturn: Optional[float] = None  # unscaled, undiscounted
    secretPartnerProb: Optional[float] = None  # [0,1]
    pointEstimates: Optional[List[AnalyzePointEstimate]] = None
    validActionIds: List[int]
    probabilities: List[AnalyzeProbability]
    view: Dict[str, Any]
    state: Optional[List[float]] = None


class AnalyzeGameSummary(BaseModel):
    hands: Dict[str, List[str]]  # player name -> cards
    blind: List[str]
    picker: Optional[str] = None
    partner: Optional[str] = None
    bury: List[str]
    pickerPoints: int
    defenderPoints: int
    scores: List[int]  # indexed by seat-1


class AnalyzeSimulateResponse(BaseModel):
    meta: Dict[str, Any]
    actionLookup: Dict[int, str]
    players: List[str]
    summary: Optional[AnalyzeGameSummary] = None
    trace: List[AnalyzeActionDetail]
    final: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    id: str
    table_id: str
    type: str  # "player" | "system"
    author: Optional[str] = None  # player display name for player messages
    body: str
    timestamp: float


class ChatSendRequest(BaseModel):
    client_id: str
    message: str



```

```python
from __future__ import annotations

import os
import random
from typing import Dict, List
import numpy as np
import torch

from sheepshead import Game, Player, ACTION_LOOKUP, DECK
from server.api.schemas import (
    AnalyzeSimulateRequest,
    AnalyzeSimulateResponse,
    AnalyzeActionDetail,
    AnalyzeProbability,
    AnalyzeGameSummary
)
from server.services.ai_loader import load_agent
from training_utils import (
    update_intermediate_rewards_for_action,
    handle_trick_completion,
    process_episode_rewards,
    TOTAL_DECK_POINTS,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_player_state_for_analyze(player: Player) -> Dict[str, any]:
    """Build the per-seat state payload: dict state and a readable view (simplified for analyze)."""
    from server.main import build_player_state
    return build_player_state(player)


def infer_phase_from_action_id(action_id: int, action_groups: Dict[str, List[int]]) -> str:
    """Infer the game phase from action ID using PPOAgent action groups."""
    # Convert to 0-indexed
    zero_indexed = action_id - 1

    for phase, indices in action_groups.items():
        if zero_indexed in indices:
            return phase

    return "unknown"


def simulate_game(req: AnalyzeSimulateRequest) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace."""

    # Set seed if provided
    if req.seed is not None:
        set_seed(req.seed)

    # Load agent
    global_model_path = os.environ.get("SHEEPSHEAD_MODEL_PATH")
    model_path = req.modelPath or global_model_path

    agent = load_agent(model_path)
    if agent is None:
        raise ValueError("Failed to load AI model")

    # agent.set_head_temperatures(partner=3.0)

    # Reset recurrent state before simulation
    agent.reset_recurrent_state()

    # Initialize game with default double_on_the_bump=True (model doesn't consider this anyway)
    partner_mode = 1 if req.partnerMode == 1 else 0  # Convert to Game's expected format
    game = Game(
        partner_selection_mode=partner_mode,
    )

    # Player display names
    players = ['Dan', 'Kyle', 'Trevor', 'John', 'Andrew']

    # Capture initial hands for summary
    initial_hands = {}
    for player in game.players:
        initial_hands[players[player.position - 1]] = sorted(list(player.hand), key=lambda card: DECK.index(card))

    # Trace storage
    trace = []
    step_index = 0
    # For reward shaping and discounted return computation
    episode_transitions = []
    current_trick_transitions = []

    # Simulation loop
    while not game.is_done() and step_index < req.maxSteps:
        # Find the current actor seat
        actor_seat = None
        actor_player = None
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            if valid_actions:
                actor_seat = player.position
                actor_player = player
                break

        if actor_player is None:
            # No valid actions found, game should be done
            break

        # Get state and valid actions
        state = actor_player.get_state_dict()
        valid_actions = actor_player.get_valid_action_ids()

        # Get or init memory for this player
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        memory_in = agent.get_recurrent_memory(actor_player.position, device=device)

        # Encode dict state with memory
        encoder_out = agent.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

        # Store updated memory
        agent.set_recurrent_memory(actor_player.position, encoder_out['memory_out'][0])

        with torch.no_grad():
            # Build mask and hand ids for actor
            action_mask_t = agent.get_action_mask(valid_actions, agent.action_size).unsqueeze(0).to(device)
            hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

            # Use existing encoder_out for logits and probabilities
            action_probs, logits = agent.actor.forward_with_logits(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                agent.encoder.card,
            )

            value = agent.critic(encoder_out)

            # Auxiliary critic heads via accessor
            win_prob_val, expected_final_val, secret_partner_prob, point_vector = agent.critic.aux_predictions(encoder_out)

        point_estimates = []
        if point_vector:
            for rel_idx, rel_val in enumerate(point_vector, start=1):
                abs_seat = ((actor_seat + rel_idx - 2) % 5) + 1
                seat_label = players[abs_seat - 1] if 0 < abs_seat <= len(players) else f"Seat {abs_seat}"
                points_real = float(max(0.0, rel_val) * TOTAL_DECK_POINTS)
                point_estimates.append({
                    "seat": abs_seat,
                    "seatName": seat_label,
                    "points": points_real,
                    "relativePosition": rel_idx,
                })

        # Choose action
        if req.deterministic:
            action_id = torch.argmax(action_probs, dim=1).item() + 1  # Convert to 1-indexed
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_id = dist.sample().item() + 1  # Convert to 1-indexed

        # Build probabilities list (only for valid actions, sorted descending)
        probabilities = []
        action_probs_np = action_probs.squeeze().cpu().numpy()
        logits_np = logits.squeeze().cpu().numpy()
        for valid_action_id in valid_actions:
            zero_indexed = valid_action_id - 1
            prob = float(action_probs_np[zero_indexed])
            logit = float(logits_np[zero_indexed])
            probabilities.append(AnalyzeProbability(
                actionId=valid_action_id,
                action=ACTION_LOOKUP[valid_action_id],
                prob=prob,
                logit=logit
            ))

        # Sort probabilities by probability descending
        probabilities.sort(key=lambda x: x.prob, reverse=True)

        # Get player state/view
        player_state = build_player_state_for_analyze(actor_player)

        # Infer phase
        phase = infer_phase_from_action_id(action_id, agent.action_groups)

        # Create action detail
        action_detail = AnalyzeActionDetail(
            stepIndex=step_index,
            seat=actor_seat,
            seatName=players[actor_seat - 1],  # Convert to 0-indexed for array access
            phase=phase,
            actionId=action_id,
            action=ACTION_LOOKUP[action_id],
            valueEstimate=float(value.item()),
            discountedReturn=None,
            validActionIds=valid_actions,
            probabilities=probabilities,
            view=player_state["view"],
            # Provide encoded 256-d feature vector as a plain float list for schema compatibility
            # TODO: Update annotations to match new state and/or drop this in favor of state dict
            state=encoder_out['features'].squeeze(0).detach().cpu().tolist(),
            winProb=float(win_prob_val),
            expectedFinalReturn=expected_final_val,
            secretPartnerProb=float(secret_partner_prob) if secret_partner_prob is not None else None,
            pointEstimates=point_estimates or None
        )

        trace.append(action_detail)

        # Build training-like transition for reward shaping
        transition = {
            'player': actor_player,
            'state': state,
            'action': action_id,
            'log_prob': 0.0,
            'value': float(value.item()),
            'valid_actions': set(valid_actions),
            'intermediate_reward': 0.0,
        }

        # Apply shared intermediate shaping and trick tracking
        update_intermediate_rewards_for_action(
            game,
            actor_player,
            action_id,
            transition,
            current_trick_transitions,
        )

        episode_transitions.append(transition)

        # Apply action
        actor_player.act(action_id)

        # Trick completion rewards, if any
        trick_completed = handle_trick_completion(game, current_trick_transitions)
        if trick_completed:
            # Propagate an observation for the just-completed trick to all seats
            for seat in game.players:
                agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)

        step_index += 1

    # Get final state and build summary if game is done
    final_payload = None
    game_summary = None
    if game.is_done():
        # Use any player to get the final state
        final_state = build_player_state_for_analyze(game.players[0])
        final_payload = final_state["view"].get("final")

        # Build game summary
        picker_seat = game.picker if hasattr(game, 'picker') and game.picker else 0
        partner_seat = game.partner if hasattr(game, 'partner') and game.partner else 0

        # If partner not revealed yet, try to find the secret partner
        if partner_seat == 0 and picker_seat > 0 and not game.alone_called:
            for player in game.players:
                if player.is_secret_partner:
                    partner_seat = player.position
                    break

        # Get bury cards (from the picker or game)
        bury_cards = []
        if picker_seat > 0:
            picker_player = game.players[picker_seat - 1]
            bury_cards = list(picker_player.bury) if hasattr(picker_player, 'bury') else []
        elif hasattr(game, 'bury'):
            bury_cards = list(game.bury)

        # Get point totals
        picker_points = 0
        defender_points = 0
        if final_payload and final_payload.get("mode") == "standard":
            picker_points = final_payload.get("picker_score", 0)
            defender_points = final_payload.get("defender_score", 0)

        # Get final scores
        scores = [p.get_score() for p in game.players] if final_payload else [0] * 5

        game_summary = AnalyzeGameSummary(
            hands=initial_hands,
            blind=game.blind,
            picker=players[picker_seat - 1] if picker_seat > 0 else None,
            partner=players[partner_seat - 1] if partner_seat > 0 else None,
            bury=bury_cards,
            pickerPoints=picker_points,
            defenderPoints=defender_points,
            scores=scores
        )

    # Compute discounted returns per action if we have any transitions
    if episode_transitions:
        final_scores = [p.get_score() for p in game.players]

        # Group indices by player (to compute rewards per player sequence)
        idxs_by_player: Dict[int, List[int]] = {}
        for i, tr in enumerate(episode_transitions):
            idxs_by_player.setdefault(tr['player'].position, []).append(i)

        # Fill rewards aligned to original order by calling per-player
        rewards = [0.0] * len(episode_transitions)
        dones = [False] * len(episode_transitions)
        for idxs in idxs_by_player.values():
            acts = [episode_transitions[i] for i in idxs]
            # Per-player rewards (last item is terminal)
            for offset, reward_data in enumerate(
                process_episode_rewards(
                    acts,
                    final_scores,
                    game.is_leaster,
                )
            ):
                rewards[idxs[offset]] = float(reward_data['reward'])
            if idxs:
                dones[idxs[-1]] = True

        gamma = getattr(agent, 'gamma', 0.95)
        discounted_by_index: Dict[int, float] = {}
        for _, idxs in idxs_by_player.items():
            ret = 0.0
            for idx in reversed(idxs):
                if dones[idx]:
                    ret = 0.0
                ret = rewards[idx] + gamma * ret
                discounted_by_index[idx] = ret

        # Attach discounted returns and per-step rewards back to trace elements (aligned by order)
        for i, action_detail in enumerate(trace):
            if i in discounted_by_index:
                action_detail.discountedReturn = float(discounted_by_index[i])
            # Always provide the raw per-step reward value for this action
            if i < len(rewards):
                action_detail.stepReward = float(rewards[i])

    # Build response
    response = AnalyzeSimulateResponse(
        meta={
            "partnerMode": req.partnerMode,
            "deterministic": req.deterministic,
            "seed": req.seed,
            "modelPath": model_path or "auto-selected"
        },
        actionLookup={k: v for k, v in ACTION_LOOKUP.items()},
        players=players,
        summary=game_summary,
        trace=trace,
        final=final_payload
    )

    return response

```

```typescript
// TypeScript types that mirror the backend response schemas

export interface AnalyzeSimulateRequest {
  seed?: number;
  partnerMode?: number; // 0 = JD, 1 = Called Ace
  deterministic?: boolean;
  modelPath?: string;
  maxSteps?: number;
}

export interface AnalyzeProbability {
  actionId: number;
  action: string;
  prob: number;
  logit: number;
}

export interface AnalyzePointEstimate {
  seat: number;
  seatName: string;
  points: number;
  relativePosition: number;
}

export interface AnalyzeActionDetail {
  stepIndex: number;
  seat: number;
  seatName: string;
  phase: string; // "pick" | "partner" | "bury" | "play"
  actionId: number;
  action: string;
  valueEstimate: number;
  discountedReturn?: number;
  stepReward?: number;
  winProb?: number;
  expectedFinalReturn?: number;
  secretPartnerProb?: number;
  pointEstimates?: AnalyzePointEstimate[];
  validActionIds: number[];
  probabilities: AnalyzeProbability[];
  view: Record<string, any>;
  state?: number[];
}

export interface AnalyzeGameSummary {
  hands: Record<string, string[]>; // player name -> cards
  blind: string[];
  picker?: string;
  partner?: string;
  bury: string[];
  pickerPoints: number;
  defenderPoints: number;
  scores: number[]; // indexed by seat-1
}

export interface AnalyzeSimulateResponse {
  meta: {
    partnerMode: number;
    deterministic: boolean;
    seed?: number;
    modelPath?: string;
  };
  actionLookup: Record<number, string>;
  players: string[];
  summary?: AnalyzeGameSummary;
  trace: AnalyzeActionDetail[];
  final?: Record<string, any>;
}

```

```typescriptreact
import React, { useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionDetails from './ActionDetails';
import { CardText } from '../../lib/components';
import styles from './page.module.css';

interface ActionRowProps {
  action: AnalyzeActionDetail;
  picker?: string;
  partner?: string;
  normalizedValue: number; // 0-1 scale for color gradient (value)
  normalizedReward: number; // 0-1 scale for color gradient (reward)
  normalizedStepReward?: number; // 0-1 scale for color gradient (immediate step reward)
}

export default function ActionRow({ action, picker, partner, normalizedValue, normalizedReward, normalizedStepReward }: ActionRowProps) {
  const [expanded, setExpanded] = useState(false);

  const handleClick = () => {
    setExpanded(!expanded);
  };

  const formatValue = (value: number) => {
    return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
  };

  const getPlayerRole = () => {
    if (action.seatName === picker) return 'picker';
    if (action.seatName === partner) return 'partner';
    return 'other';
  };

  const getSeatBadgeClass = () => {
    const role = getPlayerRole();
    switch (role) {
      case 'picker':
        return `${styles.seatBadge} ${styles.seatBadgePicker}`;
      case 'partner':
        return `${styles.seatBadge} ${styles.seatBadgePartner}`;
      default:
        return `${styles.seatBadge} ${styles.seatBadgeOther}`;
    }
  };

  const getHueStyle = (normalized: number) => {
    const hue = normalized * 120; // 0 (red) .. 120 (green)
    return { '--value-hue': `${hue}deg` } as React.CSSProperties;
  };

  const formatSigned = (value: number, digits = 3) => {
    return value >= 0 ? `+${value.toFixed(digits)}` : value.toFixed(digits);
  };

  function MetricChip({
    label,
    value,
    colorStyle,
    tooltip,
    suffix = ''
  }: {
    label: React.ReactNode;
    value: string;
    colorStyle?: React.CSSProperties;
    tooltip?: string;
    suffix?: string;
  }) {
    return (
      <div className={styles.metricChip} title={tooltip} style={colorStyle}>
        <span className={styles.metricLabel}>{label}</span>
        <span className={styles.metricValue}>{value}{suffix}</span>
      </div>
    );
  }

  function MetricsBar({
    value,
    valueHue,
    discounted,
    discountedHue,
    step,
    stepHue,
    winPct,
    expectedFinal,
  }: {
    value: number;
    valueHue: number;
    discounted?: number;
    discountedHue?: number;
    step?: number;
    stepHue?: number;
    winPct?: number;
    expectedFinal?: number;
  }) {
    return (
      <div className={styles.metricsBar}>
        <MetricChip
          label={<>V</>}
          value={formatSigned(value)}
          colorStyle={getHueStyle(valueHue)}
          tooltip="Value estimate (critic)"
        />

        {typeof discounted === 'number' && (
          <MetricChip
            label={<>G<sub>t</sub></>}
            value={formatSigned(discounted)}
            colorStyle={getHueStyle(typeof discountedHue === 'number' ? discountedHue : 0.5)}
            tooltip="Discounted return"
          />
        )}

        {typeof step === 'number' && (
          <MetricChip
            label={<>r</>}
            value={formatSigned(step)}
            colorStyle={getHueStyle(typeof stepHue === 'number' ? stepHue : 0.5)}
            tooltip="Immediate step reward"
          />
        )}

        {typeof winPct === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Win probability">
            <span className={styles.metricLabel}>Win</span>
            <span className={styles.metricValue}>{`${Math.round(winPct)}%`}</span>
          </div>
        )}

        {typeof expectedFinal === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Expected final return (undiscounted)">
            <span className={styles.metricLabel}>E[Ret]</span>
            <span className={styles.metricValue}>{expectedFinal.toFixed(2)}</span>
          </div>
        )}

      </div>
    );
  }

  return (
    <div className={styles.actionRow}>
      <div
        className={styles.actionSummary}
        onClick={handleClick}
      >
        <div className={styles.actionBasic}>
          <div className={styles.stepIndex}>
            {action.stepIndex + 1}
          </div>

          <div className={getSeatBadgeClass()}>
            {action.seatName}
          </div>

          <div className={styles.actionInfo}>
            <div className={styles.actionText}>
              <CardText>{action.action}</CardText>
            </div>
            <div className={styles.phaseText}>
              {action.phase} Head
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <MetricsBar
            value={action.valueEstimate}
            valueHue={normalizedValue}
            discounted={typeof action.discountedReturn === 'number' ? action.discountedReturn : undefined}
            discountedHue={normalizedReward}
            step={typeof action.stepReward === 'number' ? action.stepReward : undefined}
            stepHue={typeof normalizedStepReward === 'number' ? normalizedStepReward : undefined}
            winPct={typeof action.winProb === 'number' ? (action.winProb * 100) : undefined}
            expectedFinal={typeof action.expectedFinalReturn === 'number' ? action.expectedFinalReturn : undefined}
          />

          <div className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}>▼</div>
        </div>
      </div>

      {expanded && (
        <ActionDetails action={action} />
      )}
    </div>
  );
}

```

```typescriptreact
import React, { useMemo, useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ProbabilityBar from './ProbabilityBar';
import { PlayingCard } from '../../lib/components';
import styles from './page.module.css';

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const [showStateVector, setShowStateVector] = useState(false);

  const { view, probabilities, state, secretPartnerProb, pointEstimates } = action;
  const maxProb = Math.max(...probabilities.map(p => p.prob));
  const sortedPointEstimates = useMemo(() => {
    if (!pointEstimates) return [];
    return [...pointEstimates].sort((a, b) => a.seat - b.seat);
  }, [pointEstimates]);
  const secretPct = typeof secretPartnerProb === 'number'
    ? Math.round(Math.max(0, Math.min(1, secretPartnerProb)) * 100)
    : 0;

  // --- Constants mirroring backend deck order and special tokens ---
  const TRUMP: string[] = useMemo(() => [
    'QC', 'QS', 'QH', 'QD', 'JC', 'JS', 'JH', 'JD', 'AD', '10D', 'KD', '9D', '8D', '7D'
  ], []);
  const FAIL: string[] = useMemo(() => [
    'AC', '10C', 'KC', '9C', '8C', '7C',
    'AS', '10S', 'KS', '9S', '8S', '7S',
    'AH', '10H', 'KH', '9H', '8H', '7H',
  ], []);
  const DECK: string[] = useMemo(() => [...TRUMP, ...FAIL], [TRUMP, FAIL]);
  const UNDER_TOKEN = 'UNDER';
  const STATE_SIZE = 292;

  type Row = { idx: number; value: number; micro: string; sub: string; cat: 'Header' | 'Private Card Indicators' | 'Current Trick Block' };

  const rows: Row[] = useMemo(() => {
    if (!state || state.length === 0) return [];

    const r: Row[] = [];

    // Header (0-15)
    const headerMicro: string[] = [
      'Partner selection mode (0 = JD, 1 = Called Ace)',
      'Player position (1-5)',
      'Last position to pass (5 ⇒ all passed → Leaster)',
      'Picker position (0 if not yet picked)',
      'Partner position (0 if unknown)',
      'Alone called flag (0/1)',
      'Called card one-hot — AC',
      'Called card one-hot — AS',
      'Called card one-hot — AH',
      'Called card one-hot — 10C',
      'Called card one-hot — 10S',
      'Called card one-hot — 10H',
      'Is called-under flag (0/1)',
      'Is leaster flag (0/1)',
      'Play started flag (0/1)',
      'Current trick index (0-5)'
    ];

    const headerSubs: string[] = [
      'Core header', 'Core header', 'Core header', 'Core header', 'Core header', 'Core header',
      'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot',
      'Flags', 'Flags', 'Flags', 'Current trick index'
    ];

    for (let i = 0; i <= 15; i++) {
      r.push({ idx: i, value: state[i] ?? 0, micro: headerMicro[i], sub: headerSubs[i], cat: 'Header' });
    }

    // Private zones (Hand, Blind, Bury)
    const glyph = (s: 'C'|'S'|'H'|'D'): string => (s === 'C' ? '♣' : s === 'S' ? '♠' : s === 'H' ? '♥' : '♦');
    const pushPrivate = (start: number, count: number, sub: string, labelPrefix: string) => {
      for (let i = 0; i < count; i++) {
        const gi = start + i;
        const cardLabel = DECK[i] ?? `Card ${i + 1}`;
        const suit = cardLabel.endsWith('C') ? 'C' : cardLabel.endsWith('S') ? 'S' : cardLabel.endsWith('H') ? 'H' : 'D';
        const withGlyph = `${cardLabel} ${glyph(suit as any)}`;
        r.push({ idx: gi, value: state[gi] ?? 0, micro: `${labelPrefix} — ${withGlyph}`, sub, cat: 'Private Card Indicators' });
      }
    };
    // 16-47 Hand, 48-79 Blind, 80-111 Bury
    pushPrivate(16, 32, 'Hand one-hot', 'Hand');
    pushPrivate(48, 32, 'Blind one-hot', 'Blind');
    pushPrivate(80, 32, 'Bury one-hot', 'Bury');

    // Current trick block (112+) — 5 seats × (34 card one-hot + 2 flags)
    const seatLabels = ['Current Trick — Self', 'Current Trick — Player to the left', 'Current Trick — Player 2 to the left', 'Current Trick — Player 2 to the right', 'Current Trick — Player to the right'];
    const startBase = 112;
    for (let rel = 0; rel < 5; rel++) {
      const seatStart = startBase + rel * 36;
      // 34 card one-hot entries
      for (let j = 0; j < 34; j++) {
        const gi = seatStart + j;
        let cardDesc: string;
        if (j === 0) cardDesc = 'Card one-hot — empty';
        else if (j === 33) cardDesc = `Card one-hot — ${UNDER_TOKEN}`;
        else {
          const label = DECK[j - 1];
          const s = label.endsWith('C') ? 'C' : label.endsWith('S') ? 'S' : label.endsWith('H') ? 'H' : 'D';
          cardDesc = `Card one-hot — ${label} ${glyph(s as any)}`;
        }
        r.push({ idx: gi, value: state[gi] ?? 0, micro: cardDesc, sub: seatLabels[rel], cat: 'Current Trick Block' });
      }
      // 2 role flags
      r.push({ idx: seatStart + 34, value: state[seatStart + 34] ?? 0, micro: 'Role flag — is_picker (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
      r.push({ idx: seatStart + 35, value: state[seatStart + 35] ?? 0, micro: 'Role flag — is_known_partner (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
    }

    return r.slice(0, STATE_SIZE);
  }, [state, DECK]);

  // Compute rowSpans for sub and cat columns by contiguous groups
  const { subRowSpan, catRowSpan, rowClasses, subStartIdx, catStartIdx, subZeroOnly } = useMemo(() => {
    const subRowSpanMap: Record<number, number> = {};
    const catRowSpanMap: Record<number, number> = {};
    const classes: Record<number, string> = {};
    const subStart: Record<number, boolean> = {};
    const catStart: Record<number, boolean> = {};
    const subZero: Record<number, boolean> = {};

    // Helper to assign background class by large category
    const catClass = (cat: Row['cat']): string => {
      if (cat === 'Header') return styles.rowHeader;
      if (cat === 'Private Card Indicators') return styles.rowPrivate;
      return styles.rowTrick;
    };

    // Category groups
    let i = 0;
    while (i < rows.length) {
      const cat = rows[i]?.cat;
      if (!cat) break;
      let j = i;
      while (j < rows.length && rows[j].cat === cat) j++;
      catRowSpanMap[i] = j - i;
      catStart[i] = true;
      for (let k = i; k < j; k++) classes[k] = catClass(cat);

      // Within this category, compute sub-groups
      let s = i;
      while (s < j) {
        const sub = rows[s].sub;
        let t = s;
        while (t < j && rows[t].sub === sub) t++;
        subRowSpanMap[s] = t - s;
        subStart[s] = true;
        // Determine if every row value within this sub-group is zero
        let allZero = true;
        for (let k = s; k < t; k++) {
          if (rows[k].value) { allZero = false; break; }
        }
        subZero[s] = allZero;
        s = t;
      }

      i = j;
    }

    return { subRowSpan: subRowSpanMap, catRowSpan: catRowSpanMap, rowClasses: classes, subStartIdx: subStart, catStartIdx: catStart, subZeroOnly: subZero };
  }, [rows, styles.rowHeader, styles.rowPrivate, styles.rowTrick]);

  // Extract relevant cards from view
  const hand = view.hand || [];
  const blind = view.blind || [];
  const bury = view.bury || [];
  const currentTrick = view.current_trick || [];

  return (
    <div className={styles.actionDetails}>
      {(typeof secretPartnerProb === 'number' || sortedPointEstimates.length > 0) && (
        <div className={styles.insightsGrid}>
          {typeof secretPartnerProb === 'number' && (
            <div className={`${styles.insightCard} ${styles.secretInsight}`}>
              <div className={styles.insightTitle}>Secret Partner Confidence</div>
              <div className={styles.secretGaugeWrapper}>
                <div
                  className={styles.secretGauge}
                  style={{ background: `conic-gradient(#f59e0b ${secretPct * 3.6}deg, #e2e8f0 0deg)` }}
                >
                  <div className={styles.secretGaugeValue}>{secretPct}%</div>
                </div>
                <p className={styles.secretSubtitle}>Probability this seat is the hidden ally.</p>
              </div>
            </div>
          )}

          {sortedPointEstimates.length > 0 && (
            <div className={`${styles.insightCard} ${styles.pointsInsight}`}>
              <div className={styles.insightTitle}>Point Forecast</div>
              <div className={styles.pointsList}>
                {sortedPointEstimates.map((pt) => {
                  const widthPct = Math.min(100, (pt.points / 120) * 100);
                  return (
                    <div key={pt.seat} className={styles.pointRow}>
                      <div className={styles.pointSeat}>
                        Seat {pt.seat}
                        <span>{pt.seatName}</span>
                      </div>
                      <div className={styles.pointBarWrapper}>
                        <div className={styles.pointBarTrack}>
                          <div className={styles.pointBarFill} style={{ width: `${widthPct}%` }} />
                        </div>
                        <div className={styles.pointValue}>{pt.points.toFixed(1)} pts</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      <div className={styles.detailsGrid}>
        {/* Game State */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Game State</div>

          {hand.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Hand
              </div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Blind
              </div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Bury
              </div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== '') && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Current Trick
              </div>
              <div className={styles.cardList}>
                {currentTrick.filter((card: string) => card !== '').map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
              {view.picker && <div>Picker: Seat {view.picker}</div>}
              {view.partner && <div>Partner: Seat {view.partner}</div>}
              {view.called_card && <div>Called Card: {view.called_card}</div>}
              {view.is_leaster && <div>Mode: Leaster</div>}
              {view.current_trick_index !== undefined && (
                <div>Trick: {view.current_trick_index + 1}/6</div>
              )}
            </div>
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar
                key={i}
                probability={prob}
                maxProb={maxProb}
              />
            ))}
          </div>
        </div>
      </div>

      {/* State Vector Section */}
      {state && (
        <div className={styles.stateVectorSection}>
          <button
            className={styles.toggleButton}
            onClick={() => setShowStateVector(!showStateVector)}
          >
            {showStateVector ? 'Hide' : 'Show'} State Vector
          </button>

          {showStateVector && (
            <>
              <div className={styles.stateVectorTableWrapper}>
                <div className={styles.stateVectorTableContainer}>
                  <table className={styles.stateVectorTable}>
                    <thead className={styles.stateVectorHead}>
                      <tr>
                        <th>Index</th>
                        <th>Value</th>
                        <th>Detail</th>
                        <th colSpan={2}>Category</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => {
                        const showSub = subRowSpan[i] !== undefined;
                        const showCat = catRowSpan[i] !== undefined;
                        const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                        // Find the index where this row's sub-group starts to apply dimming
                        let subAnchor = i;
                        while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                        const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                        return (
                          <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                            <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                            <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                            <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                            {showSub && (
                              <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>{row.sub}</td>
                            )}
                            {showCat && (
                              <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>{row.cat}</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <button
                className={styles.toggleButton}
                onClick={() => setShowStateVector(!showStateVector)}
              >
                ⌃ Collapse
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

```

```css
/* Analyze AI Model Page Styles */

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}

.header {
  text-align: center;
  margin-bottom: 3rem;
}

.title {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 1rem;
}

.subtitle {
  font-size: 1.1rem;
  color: #64748b;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

/* Controls Panel */
.controlsPanel {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
}

.controlsTitle {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.controlsGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.controlGroup {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.label {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.select, .input, .checkbox {
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.select:focus, .input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.checkboxWrapper {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  transition: all 0.2s ease;
}

.checkboxWrapper:hover {
  background: #f1f5f9;
}

.checkbox {
  margin: 0;
  padding: 0;
  width: 1.25rem;
  height: 1.25rem;
  cursor: pointer;
  accent-color: #3b82f6;
}

.checkboxLabel {
  margin: 0;
  cursor: pointer;
  user-select: none;
}

.simulateButton {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  justify-content: center;
  min-width: 200px;
}

.simulateButton:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 10px 25px -3px rgba(59, 130, 246, 0.4);
}

.simulateButton:disabled {
  background: #9ca3af;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.buttonRow {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

/* Error Styling */
.error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

/* Results Panel */
.resultsPanel {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.resultsHeader {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid #e2e8f0;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

.resultsTitle {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 0.75rem;
  font-size: 0.875rem;
  color: #64748b;
}

.metaItem {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
}

/* Action Row Styles */
.actionRow {
  border-bottom: 1px solid #f1f5f9;
  cursor: pointer;
  transition: all 0.15s ease;
  background: white;
}

.actionRow:hover {
  background: #f8fafc;
}

.actionRow:last-child {
  border-bottom: none;
}

.actionSummary {
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.actionBasic {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
}

.stepIndex {
  background: #e2e8f0;
  color: #475569;
  font-weight: 600;
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  min-width: 2rem;
  text-align: center;
}

.seatBadge {
  font-weight: 600;
  font-size: 0.875rem;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  min-width: 4rem;
  text-align: center;
}

.seatBadgePicker {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
}

.seatBadgePartner {
  background: linear-gradient(135deg, #6faffc 0%, #3b82f6 100%);
  color: white;
}

.seatBadgeOther {
  background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
  color: white;
}

.actionInfo {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.actionText {
  font-weight: 600;
  color: #1e293b;
  font-size: 0.875rem;
}

.phaseText {
  font-size: 0.75rem;
  color: #64748b;
  text-transform: capitalize;
}

.valueEstimate {
  color: hsl(var(--value-hue), 70%, 40%);
  background: hsl(var(--value-hue), 60%, 95%);
  font-weight: 600;
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  white-space: nowrap;
}

/* Metric strip for compact display of five values */
.metricsBar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: nowrap;
}

.metricChip {
  --chip-bg: hsl(var(--value-hue, 0), 60%, 97%);
  border: 1px solid color-mix(in oklab, hsl(var(--value-hue, 0), 85%, 50%) 30%, rgba(0,0,0,0.06));
  background: var(--chip-bg);
  border-radius: 8px;
  padding: 4px 6px;
  display: inline-flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px;
  min-width: 0;
}

/* Neutral chip (no hue tint) */
.metricChipNeutral {
  background: #f8fafc;
  border: 1px solid #e5e7eb;
}

.metricLabel {
  font-size: 0.65rem;
  letter-spacing: 0.02em;
  opacity: 0.7;
}

.metricValue {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  font-size: 0.75rem;
}

@media (max-width: 768px) {
  .metricsBar {
    flex-wrap: wrap;
  }
}

.expandIcon {
  color: #9ca3af;
  transition: transform 0.2s ease;
}

.expandIcon.expanded {
  transform: rotate(180deg);
}

/* Action Details */
.actionDetails {
  padding: 0 2rem 1.5rem 2rem;
  background: #f8fafc;
  border-top: 1px solid #f1f5f9;
}

.insightsGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.5rem;
  margin-top: 1.25rem;
}

.insightCard {
  background: white;
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  border: 1px solid #e2e8f0;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.insightTitle {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #475569;
}

.secretInsight {
  background: linear-gradient(145deg, #fff7ed 0%, #fffbeb 100%);
  border-color: #fed7aa;
}

.secretGaugeWrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.secretGauge {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  padding: 12px;
  background: conic-gradient(#f59e0b 0deg, #fde68a 0deg);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  box-shadow: inset 0 0 12px rgba(249, 115, 22, 0.35);
}

.secretGauge::after {
  content: '';
  position: absolute;
  inset: 8px;
  background: #fff7ed;
  border-radius: 50%;
}

.secretGaugeValue {
  position: relative;
  font-size: 1.5rem;
  font-weight: 700;
  color: #9a3412;
  font-variant-numeric: tabular-nums;
}

.secretSubtitle {
  margin: 0;
  font-size: 0.85rem;
  color: #92400e;
  text-align: center;
}

.pointsInsight {
  background: linear-gradient(145deg, #eff6ff 0%, #f8fafc 100%);
  border-color: #c7d2fe;
}

.pointsList {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.pointRow {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.pointSeat {
  min-width: 110px;
  font-weight: 600;
  color: #1e3a8a;
  font-size: 0.9rem;
}

.pointSeat span {
  display: block;
  font-size: 0.7rem;
  color: #6b7280;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.pointBarWrapper {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.pointBarTrack {
  flex: 1;
  height: 8px;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
}

.pointBarFill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #1e3a8a 100%);
  border-radius: 999px;
  transition: width 200ms ease;
}

.pointValue {
  min-width: 72px;
  font-weight: 700;
  color: #0f172a;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.detailsGrid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 1rem;
}

@media (max-width: 768px) {
  .detailsGrid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }

  .insightsGrid {
    grid-template-columns: 1fr;
  }
}

.detailSection {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.detailTitle {
  font-weight: 600;
  color: #374151;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.cardList {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.card {
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 500;
  color: #374151;
}

/* Probability Bars */
.probabilitiesList {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.probabilityItem {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.probabilityAction {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
  min-width: 80px;
}

.probabilityBarContainer {
  flex: 1;
  height: 8px;
  background: #f3f4f6;
  border-radius: 4px;
  overflow: hidden;
}

.probabilityBar {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.probabilityValue {
  font-weight: 600;
  font-size: 0.75rem;
  color: #6b7280;
  min-width: 40px;
  text-align: right;
}

/* Subtle inline logits */
.probabilityLogit {
  color: #9ca3af; /* gray-400 */
  font-size: 0.75rem;
  width: 60px;
  display: inline-block;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  margin-left: 10px;
  transition: color 120ms ease;
  font-variant-numeric: tabular-nums;
  border-left: 2px solid #e5e7eb;
}
.probabilityItem:hover .probabilityLogit {
  color: #6b7280; /* gray-500 on hover */
}

/* State Vector Section */
.stateVectorSection {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.toggleButton {
  background: none;
  border: 1px solid #d1d5db;
  color: #6b7280;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.toggleButton:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

/* State Vector Table */
.stateVectorTableWrapper {
  margin-top: 1rem;
  margin-bottom: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
  background: white;
}

.stateVectorTableContainer {
  /* max-height: 800px; */
  overflow: auto;
}

.stateVectorTable {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.8125rem;
}

.stateVectorHead th {
  position: sticky;
  top: 0;
  background: #f8fafc;
  z-index: 1;
  text-align: left;
  font-weight: 600;
  color: #475569;
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #e5e7eb;
  border-right: 1px solid #e5e7eb;
}
.stateVectorHead th:last-child { border-right: none; }

.stateVectorCell,
.stateVectorIndex,
.stateVectorValue,
.stateVectorMicroCat,
.stateVectorSubCat,
.stateVectorCategory {
  padding: 0.5rem 0.75rem;
  border: 1px solid #e5e7eb;
  vertical-align: top;
}
.stateVectorTable td:last-child { border-right: none; }

.stateVectorIndex {
  font-variant-numeric: tabular-nums;
  color: #64748b;
  background: #f9fafb;
  min-width: 64px;
}

.stateVectorValue {
  font-variant-numeric: tabular-nums;
  font-family: 'Monaco', 'Consolas', monospace;
  color: #111827;
  min-width: 72px;
}

.valueHighlighted {
  background: #fcf9e4;
  font-weight: 700;
}

.cellHighlighted {
  background: #fcf9e4;
}

.stateVectorMicroCat {
  color: #374151;
}

.stateVectorSubCat {
  color: #1f2937;
  font-weight: 600;
  vertical-align: middle;
}

.stateVectorCategory {
  color: #111827;
  font-weight: 700;
  vertical-align: middle;
}

.rowHeader {
  background: #f8fafc;
}

.rowPrivate {
  background: #fafaf9;
}

.rowTrick {
  background: #f9fafb;
}

.stateVectorFoot td {
  padding: 0.5rem 0.75rem;
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
  color: #64748b;
}

/* Group separators */
.subGroupStart td { border-top: 2px solid #cbd5e1; }
.catGroupStart td { border-top: 3px solid #94a3b8; }

.subGroupDim td {
  color: #9ca3af;
}

/* Loading State */
.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 3rem;
  color: #6b7280;
}

.spinner {
  border: 2px solid #f3f4f6;
  border-radius: 50%;
  border-top: 2px solid #3b82f6;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;
  margin-right: 0.75rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Game Summary */
.gameSummary {
  background: #f8fafc;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid #e2e8f0;
}

.summarySection {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.summaryTitle {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
}

.summarySubsection {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.summarySubtitle {
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: 0;
}

.handsGrid {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.handRow {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.playerName {
  font-weight: 600;
  color: #374151;
  min-width: 4rem;
  text-align: left;
}

.gameInfoGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
}

.gameInfoItem {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.infoLabel {
  font-weight: 500;
  color: #6b7280;
  font-size: 0.875rem;
}

.infoValue {
  font-weight: 600;
  color: #374151;
  font-size: 0.875rem;
}

.scoresGrid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.scoreItem {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.scorePlayer {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
}

.scoreValue {
  font-weight: 600;
  font-size: 0.875rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  background: #f3f4f6;
  color: #6b7280;
}

.positiveScore {
  background: #dcfce7;
  color: #166534;
}

.negativeScore {
  background: #fee2e2;
  color: #dc2626;
}

/* Dividers */
.phaseDivider {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 2rem;
  padding: 0.75rem 0;
}

.phaseDividerLine {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, #9ca3af 50%, transparent 100%);
}

.phaseDividerLabel {
  font-size: 0.75rem;
  font-weight: 500;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
  padding: 0.375rem 0.75rem;
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 16px;
}

.trickDivider {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.75rem 2rem;
  padding: 0.5rem 0;
}

.trickDividerLine {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
}

.trickDividerLabel {
  font-size: 0.75rem;
  font-weight: 500;
  color: #64748b;
  text-transform: capitalize;
  letter-spacing: 0.025em;
  white-space: nowrap;
  padding: 0.25rem 0.75rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
}

/* Empty State */
.emptyState {
  text-align: center;
  padding: 3rem;
  color: #6b7280;
}

.emptyState h3 {
  color: #374151;
  margin-bottom: 0.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }

  .title {
    font-size: 2rem;
  }

  .controlsGrid {
    grid-template-columns: 1fr;
  }

  .actionSummary {
    padding: 1rem;
    flex-direction: column;
    align-items: stretch;
    gap: 0.75rem;
  }

  .actionBasic {
    justify-content: space-between;
  }

  .actionDetails {
    padding: 1rem;
  }

  .phaseDivider {
    margin: 0.75rem 1rem;
    gap: 0.75rem;
  }

  .phaseDividerLabel {
    font-size: 0.625rem;
    padding: 0.25rem 0.5rem;
  }

  .trickDivider {
    margin: 0.5rem 1rem;
    gap: 0.5rem;
  }

  .trickDividerLabel {
    font-size: 0.625rem;
    padding: 0.25rem 0.5rem;
  }
}

```

Here’s what changed:

- **Critic & backend data**: the auxiliary head now returns normalized per-seat point predictions, clamped for stability; `server/services/analyze.py` scales them to real points (× `TOTAL_DECK_POINTS`) and rotates them into absolute seats before sending them in each `AnalyzeActionDetail`. The API schema and TS types gained a `pointEstimates` model so the data flows end-to-end.

- **Analyze view**: the compact metrics bar stays focused on value/return stats, while the expanded panel now unveils a styled “insights” area. It highlights secret partner probability with a radial gauge and lists the predicted point totals for all seats with animated bars and player names. New CSS covers the cards, gauges, and bars, so the section reads cleanly within the existing layout.

No automated tests were run (frontend/backend changes only). To verify, re-run an analyze simulation and expand an action row—you should see the new insights block with correctly scaled point totals and secret probability, while the summary metrics stay uncluttered.

---

**User**

I like this, but let's make some changes:

- The action probabilities and game state should be listed first in the expanded view.
- The additional predicitons should be in their own component, not just stuck in the ActionDetails component
- The secret partner prediction should be much smaller and secondary. A simple labeled percentage indicator is fine here. It's actually harder to read the large circle now. And it is too large.

---

**Cursor**

```typescriptreact
import React, { useMemo, useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ProbabilityBar from './ProbabilityBar';
import { PlayingCard } from '../../lib/components';
import styles from './page.module.css';

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const [showStateVector, setShowStateVector] = useState(false);

  const { view, probabilities, state } = action;
  const maxProb = Math.max(...probabilities.map(p => p.prob));

  // --- Constants mirroring backend deck order and special tokens ---
  const TRUMP: string[] = useMemo(() => [
    'QC', 'QS', 'QH', 'QD', 'JC', 'JS', 'JH', 'JD', 'AD', '10D', 'KD', '9D', '8D', '7D'
  ], []);
  const FAIL: string[] = useMemo(() => [
    'AC', '10C', 'KC', '9C', '8C', '7C',
    'AS', '10S', 'KS', '9S', '8S', '7S',
    'AH', '10H', 'KH', '9H', '8H', '7H',
  ], []);
  const DECK: string[] = useMemo(() => [...TRUMP, ...FAIL], [TRUMP, FAIL]);
  const UNDER_TOKEN = 'UNDER';
  const STATE_SIZE = 292;

  type Row = { idx: number; value: number; micro: string; sub: string; cat: 'Header' | 'Private Card Indicators' | 'Current Trick Block' };

  const rows: Row[] = useMemo(() => {
    if (!state || state.length === 0) return [];

    const r: Row[] = [];

    // Header (0-15)
    const headerMicro: string[] = [
      'Partner selection mode (0 = JD, 1 = Called Ace)',
      'Player position (1-5)',
      'Last position to pass (5 ⇒ all passed → Leaster)',
      'Picker position (0 if not yet picked)',
      'Partner position (0 if unknown)',
      'Alone called flag (0/1)',
      'Called card one-hot — AC',
      'Called card one-hot — AS',
      'Called card one-hot — AH',
      'Called card one-hot — 10C',
      'Called card one-hot — 10S',
      'Called card one-hot — 10H',
      'Is called-under flag (0/1)',
      'Is leaster flag (0/1)',
      'Play started flag (0/1)',
      'Current trick index (0-5)'
    ];

    const headerSubs: string[] = [
      'Core header', 'Core header', 'Core header', 'Core header', 'Core header', 'Core header',
      'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot',
      'Flags', 'Flags', 'Flags', 'Current trick index'
    ];

    for (let i = 0; i <= 15; i++) {
      r.push({ idx: i, value: state[i] ?? 0, micro: headerMicro[i], sub: headerSubs[i], cat: 'Header' });
    }

    // Private zones (Hand, Blind, Bury)
    const glyph = (s: 'C'|'S'|'H'|'D'): string => (s === 'C' ? '♣' : s === 'S' ? '♠' : s === 'H' ? '♥' : '♦');
    const pushPrivate = (start: number, count: number, sub: string, labelPrefix: string) => {
      for (let i = 0; i < count; i++) {
        const gi = start + i;
        const cardLabel = DECK[i] ?? `Card ${i + 1}`;
        const suit = cardLabel.endsWith('C') ? 'C' : cardLabel.endsWith('S') ? 'S' : cardLabel.endsWith('H') ? 'H' : 'D';
        const withGlyph = `${cardLabel} ${glyph(suit as any)}`;
        r.push({ idx: gi, value: state[gi] ?? 0, micro: `${labelPrefix} — ${withGlyph}`, sub, cat: 'Private Card Indicators' });
      }
    };
    // 16-47 Hand, 48-79 Blind, 80-111 Bury
    pushPrivate(16, 32, 'Hand one-hot', 'Hand');
    pushPrivate(48, 32, 'Blind one-hot', 'Blind');
    pushPrivate(80, 32, 'Bury one-hot', 'Bury');

    // Current trick block (112+) — 5 seats × (34 card one-hot + 2 flags)
    const seatLabels = ['Current Trick — Self', 'Current Trick — Player to the left', 'Current Trick — Player 2 to the left', 'Current Trick — Player 2 to the right', 'Current Trick — Player to the right'];
    const startBase = 112;
    for (let rel = 0; rel < 5; rel++) {
      const seatStart = startBase + rel * 36;
      // 34 card one-hot entries
      for (let j = 0; j < 34; j++) {
        const gi = seatStart + j;
        let cardDesc: string;
        if (j === 0) cardDesc = 'Card one-hot — empty';
        else if (j === 33) cardDesc = `Card one-hot — ${UNDER_TOKEN}`;
        else {
          const label = DECK[j - 1];
          const s = label.endsWith('C') ? 'C' : label.endsWith('S') ? 'S' : label.endsWith('H') ? 'H' : 'D';
          cardDesc = `Card one-hot — ${label} ${glyph(s as any)}`;
        }
        r.push({ idx: gi, value: state[gi] ?? 0, micro: cardDesc, sub: seatLabels[rel], cat: 'Current Trick Block' });
      }
      // 2 role flags
      r.push({ idx: seatStart + 34, value: state[seatStart + 34] ?? 0, micro: 'Role flag — is_picker (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
      r.push({ idx: seatStart + 35, value: state[seatStart + 35] ?? 0, micro: 'Role flag — is_known_partner (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
    }

    return r.slice(0, STATE_SIZE);
  }, [state, DECK]);

  // Compute rowSpans for sub and cat columns by contiguous groups
  const { subRowSpan, catRowSpan, rowClasses, subStartIdx, catStartIdx, subZeroOnly } = useMemo(() => {
    const subRowSpanMap: Record<number, number> = {};
    const catRowSpanMap: Record<number, number> = {};
    const classes: Record<number, string> = {};
    const subStart: Record<number, boolean> = {};
    const catStart: Record<number, boolean> = {};
    const subZero: Record<number, boolean> = {};

    // Helper to assign background class by large category
    const catClass = (cat: Row['cat']): string => {
      if (cat === 'Header') return styles.rowHeader;
      if (cat === 'Private Card Indicators') return styles.rowPrivate;
      return styles.rowTrick;
    };

    // Category groups
    let i = 0;
    while (i < rows.length) {
      const cat = rows[i]?.cat;
      if (!cat) break;
      let j = i;
      while (j < rows.length && rows[j].cat === cat) j++;
      catRowSpanMap[i] = j - i;
      catStart[i] = true;
      for (let k = i; k < j; k++) classes[k] = catClass(cat);

      // Within this category, compute sub-groups
      let s = i;
      while (s < j) {
        const sub = rows[s].sub;
        let t = s;
        while (t < j && rows[t].sub === sub) t++;
        subRowSpanMap[s] = t - s;
        subStart[s] = true;
        // Determine if every row value within this sub-group is zero
        let allZero = true;
        for (let k = s; k < t; k++) {
          if (rows[k].value) { allZero = false; break; }
        }
        subZero[s] = allZero;
        s = t;
      }

      i = j;
    }

    return { subRowSpan: subRowSpanMap, catRowSpan: catRowSpanMap, rowClasses: classes, subStartIdx: subStart, catStartIdx: catStart, subZeroOnly: subZero };
  }, [rows, styles.rowHeader, styles.rowPrivate, styles.rowTrick]);

  // Extract relevant cards from view
  const hand = view.hand || [];
  const blind = view.blind || [];
  const bury = view.bury || [];
  const currentTrick = view.current_trick || [];

  return (
    <div className={styles.actionDetails}>
      {(typeof secretPartnerProb === 'number' || sortedPointEstimates.length > 0) && (
        <div className={styles.insightsGrid}>
          {typeof secretPartnerProb === 'number' && (
            <div className={`${styles.insightCard} ${styles.secretInsight}`}>
              <div className={styles.insightTitle}>Secret Partner Confidence</div>
              <div className={styles.secretGaugeWrapper}>
                <div
                  className={styles.secretGauge}
                  style={{ background: `conic-gradient(#f59e0b ${secretPct * 3.6}deg, #e2e8f0 0deg)` }}
                >
                  <div className={styles.secretGaugeValue}>{secretPct}%</div>
                </div>
                <p className={styles.secretSubtitle}>Probability this seat is the hidden ally.</p>
              </div>
            </div>
          )}

          {sortedPointEstimates.length > 0 && (
            <div className={`${styles.insightCard} ${styles.pointsInsight}`}>
              <div className={styles.insightTitle}>Point Forecast</div>
              <div className={styles.pointsList}>
                {sortedPointEstimates.map((pt) => {
                  const widthPct = Math.min(100, (pt.points / 120) * 100);
                  return (
                    <div key={pt.seat} className={styles.pointRow}>
                      <div className={styles.pointSeat}>
                        Seat {pt.seat}
                        <span>{pt.seatName}</span>
                      </div>
                      <div className={styles.pointBarWrapper}>
                        <div className={styles.pointBarTrack}>
                          <div className={styles.pointBarFill} style={{ width: `${widthPct}%` }} />
                        </div>
                        <div className={styles.pointValue}>{pt.points.toFixed(1)} pts</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      <div className={styles.detailsGrid}>
        {/* Game State */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Game State</div>

          {hand.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Hand
              </div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Blind
              </div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Bury
              </div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== '') && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Current Trick
              </div>
              <div className={styles.cardList}>
                {currentTrick.filter((card: string) => card !== '').map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
              {view.picker && <div>Picker: Seat {view.picker}</div>}
              {view.partner && <div>Partner: Seat {view.partner}</div>}
              {view.called_card && <div>Called Card: {view.called_card}</div>}
              {view.is_leaster && <div>Mode: Leaster</div>}
              {view.current_trick_index !== undefined && (
                <div>Trick: {view.current_trick_index + 1}/6</div>
              )}
            </div>
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar
                key={i}
                probability={prob}
                maxProb={maxProb}
              />
            ))}
          </div>
        </div>
      </div>

      {/* State Vector Section */}
      {state && (
        <div className={styles.stateVectorSection}>
          <button
            className={styles.toggleButton}
            onClick={() => setShowStateVector(!showStateVector)}
          >
            {showStateVector ? 'Hide' : 'Show'} State Vector
          </button>

          {showStateVector && (
            <>
              <div className={styles.stateVectorTableWrapper}>
                <div className={styles.stateVectorTableContainer}>
                  <table className={styles.stateVectorTable}>
                    <thead className={styles.stateVectorHead}>
                      <tr>
                        <th>Index</th>
                        <th>Value</th>
                        <th>Detail</th>
                        <th colSpan={2}>Category</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => {
                        const showSub = subRowSpan[i] !== undefined;
                        const showCat = catRowSpan[i] !== undefined;
                        const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                        // Find the index where this row's sub-group starts to apply dimming
                        let subAnchor = i;
                        while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                        const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                        return (
                          <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                            <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                            <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                            <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                            {showSub && (
                              <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>{row.sub}</td>
                            )}
                            {showCat && (
                              <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>{row.cat}</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <button
                className={styles.toggleButton}
                onClick={() => setShowStateVector(!showStateVector)}
              >
                ⌃ Collapse
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

```

```typescriptreact
import React, { useMemo } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import styles from './page.module.css';

interface ActionInsightsProps {
  action: AnalyzeActionDetail;
}

export default function ActionInsights({ action }: ActionInsightsProps) {
  const { secretPartnerProb, pointEstimates } = action;

  const sortedPointEstimates = useMemo(() => {
    if (!pointEstimates) return [];
    return [...pointEstimates].sort((a, b) => a.seat - b.seat);
  }, [pointEstimates]);

  const showSecret = typeof secretPartnerProb === 'number';
  const showPoints = sortedPointEstimates.length > 0;

  if (!showSecret && !showPoints) {
    return null;
  }

  const secretPct = showSecret
    ? Math.round(Math.max(0, Math.min(1, secretPartnerProb as number)) * 100)
    : 0;

  return (
    <div className={styles.actionInsightsSection}>
      <div className={styles.insightsGrid}>
        {showPoints && (
          <div className={`${styles.insightCard} ${styles.pointsInsight}`}>
            <div className={styles.insightTitle}>Point Forecast</div>
            <div className={styles.pointsList}>
              {sortedPointEstimates.map((pt) => {
                const widthPct = Math.min(100, (pt.points / 120) * 100);
                return (
                  <div key={pt.seat} className={styles.pointRow}>
                    <div className={styles.pointSeat}>
                      Seat {pt.seat}
                      <span>{pt.seatName}</span>
                    </div>
                    <div className={styles.pointBarWrapper}>
                      <div className={styles.pointBarTrack}>
                        <div className={styles.pointBarFill} style={{ width: `${widthPct}%` }} />
                      </div>
                      <div className={styles.pointValue}>{pt.points.toFixed(1)} pts</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {showSecret && (
          <div className={`${styles.insightCard} ${styles.secretInsight}`}>
            <div className={styles.insightTitle}>Secret Partner</div>
            <div className={styles.secretMini}>
              <div className={styles.secretMiniValue}>{secretPct}%</div>
              <div className={styles.secretMiniLabel}>confidence</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

```

```typescriptreact
import React, { useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionDetails from './ActionDetails';
import ActionInsights from './ActionInsights';
import { CardText } from '../../lib/components';
import styles from './page.module.css';

interface ActionRowProps {
  action: AnalyzeActionDetail;
  picker?: string;
  partner?: string;
  normalizedValue: number; // 0-1 scale for color gradient (value)
  normalizedReward: number; // 0-1 scale for color gradient (reward)
  normalizedStepReward?: number; // 0-1 scale for color gradient (immediate step reward)
}

export default function ActionRow({ action, picker, partner, normalizedValue, normalizedReward, normalizedStepReward }: ActionRowProps) {
  const [expanded, setExpanded] = useState(false);

  const handleClick = () => {
    setExpanded(!expanded);
  };

  const formatValue = (value: number) => {
    return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
  };

  const getPlayerRole = () => {
    if (action.seatName === picker) return 'picker';
    if (action.seatName === partner) return 'partner';
    return 'other';
  };

  const getSeatBadgeClass = () => {
    const role = getPlayerRole();
    switch (role) {
      case 'picker':
        return `${styles.seatBadge} ${styles.seatBadgePicker}`;
      case 'partner':
        return `${styles.seatBadge} ${styles.seatBadgePartner}`;
      default:
        return `${styles.seatBadge} ${styles.seatBadgeOther}`;
    }
  };

  const getHueStyle = (normalized: number) => {
    const hue = normalized * 120; // 0 (red) .. 120 (green)
    return { '--value-hue': `${hue}deg` } as React.CSSProperties;
  };

  const formatSigned = (value: number, digits = 3) => {
    return value >= 0 ? `+${value.toFixed(digits)}` : value.toFixed(digits);
  };

  function MetricChip({
    label,
    value,
    colorStyle,
    tooltip,
    suffix = ''
  }: {
    label: React.ReactNode;
    value: string;
    colorStyle?: React.CSSProperties;
    tooltip?: string;
    suffix?: string;
  }) {
    return (
      <div className={styles.metricChip} title={tooltip} style={colorStyle}>
        <span className={styles.metricLabel}>{label}</span>
        <span className={styles.metricValue}>{value}{suffix}</span>
      </div>
    );
  }

  function MetricsBar({
    value,
    valueHue,
    discounted,
    discountedHue,
    step,
    stepHue,
    winPct,
    expectedFinal,
  }: {
    value: number;
    valueHue: number;
    discounted?: number;
    discountedHue?: number;
    step?: number;
    stepHue?: number;
    winPct?: number;
    expectedFinal?: number;
  }) {
    return (
      <div className={styles.metricsBar}>
        <MetricChip
          label={<>V</>}
          value={formatSigned(value)}
          colorStyle={getHueStyle(valueHue)}
          tooltip="Value estimate (critic)"
        />

        {typeof discounted === 'number' && (
          <MetricChip
            label={<>G<sub>t</sub></>}
            value={formatSigned(discounted)}
            colorStyle={getHueStyle(typeof discountedHue === 'number' ? discountedHue : 0.5)}
            tooltip="Discounted return"
          />
        )}

        {typeof step === 'number' && (
          <MetricChip
            label={<>r</>}
            value={formatSigned(step)}
            colorStyle={getHueStyle(typeof stepHue === 'number' ? stepHue : 0.5)}
            tooltip="Immediate step reward"
          />
        )}

        {typeof winPct === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Win probability">
            <span className={styles.metricLabel}>Win</span>
            <span className={styles.metricValue}>{`${Math.round(winPct)}%`}</span>
          </div>
        )}

        {typeof expectedFinal === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Expected final return (undiscounted)">
            <span className={styles.metricLabel}>E[Ret]</span>
            <span className={styles.metricValue}>{expectedFinal.toFixed(2)}</span>
          </div>
        )}

      </div>
    );
  }

  return (
    <div className={styles.actionRow}>
      <div
        className={styles.actionSummary}
        onClick={handleClick}
      >
        <div className={styles.actionBasic}>
          <div className={styles.stepIndex}>
            {action.stepIndex + 1}
          </div>

          <div className={getSeatBadgeClass()}>
            {action.seatName}
          </div>

          <div className={styles.actionInfo}>
            <div className={styles.actionText}>
              <CardText>{action.action}</CardText>
            </div>
            <div className={styles.phaseText}>
              {action.phase} Head
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <MetricsBar
            value={action.valueEstimate}
            valueHue={normalizedValue}
            discounted={typeof action.discountedReturn === 'number' ? action.discountedReturn : undefined}
            discountedHue={normalizedReward}
            step={typeof action.stepReward === 'number' ? action.stepReward : undefined}
            stepHue={typeof normalizedStepReward === 'number' ? normalizedStepReward : undefined}
            winPct={typeof action.winProb === 'number' ? (action.winProb * 100) : undefined}
            expectedFinal={typeof action.expectedFinalReturn === 'number' ? action.expectedFinalReturn : undefined}
          />

          <div className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}>▼</div>
        </div>
      </div>

      {expanded && (
        <>
          <ActionDetails action={action} />
          <ActionInsights action={action} />
        </>
      )}
    </div>
  );
}

```

```css
/* Analyze AI Model Page Styles */

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}

.header {
  text-align: center;
  margin-bottom: 3rem;
}

.title {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 1rem;
}

.subtitle {
  font-size: 1.1rem;
  color: #64748b;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

/* Controls Panel */
.controlsPanel {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
}

.controlsTitle {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.controlsGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.controlGroup {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.label {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.select, .input, .checkbox {
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.select:focus, .input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.checkboxWrapper {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  transition: all 0.2s ease;
}

.checkboxWrapper:hover {
  background: #f1f5f9;
}

.checkbox {
  margin: 0;
  padding: 0;
  width: 1.25rem;
  height: 1.25rem;
  cursor: pointer;
  accent-color: #3b82f6;
}

.checkboxLabel {
  margin: 0;
  cursor: pointer;
  user-select: none;
}

.simulateButton {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  justify-content: center;
  min-width: 200px;
}

.simulateButton:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 10px 25px -3px rgba(59, 130, 246, 0.4);
}

.simulateButton:disabled {
  background: #9ca3af;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.buttonRow {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

/* Error Styling */
.error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

/* Results Panel */
.resultsPanel {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.resultsHeader {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid #e2e8f0;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

.resultsTitle {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 0.75rem;
  font-size: 0.875rem;
  color: #64748b;
}

.metaItem {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
}

/* Action Row Styles */
.actionRow {
  border-bottom: 1px solid #f1f5f9;
  cursor: pointer;
  transition: all 0.15s ease;
  background: white;
}

.actionRow:hover {
  background: #f8fafc;
}

.actionRow:last-child {
  border-bottom: none;
}

.actionSummary {
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.actionBasic {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
}

.stepIndex {
  background: #e2e8f0;
  color: #475569;
  font-weight: 600;
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  min-width: 2rem;
  text-align: center;
}

.seatBadge {
  font-weight: 600;
  font-size: 0.875rem;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  min-width: 4rem;
  text-align: center;
}

.seatBadgePicker {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
}

.seatBadgePartner {
  background: linear-gradient(135deg, #6faffc 0%, #3b82f6 100%);
  color: white;
}

.seatBadgeOther {
  background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
  color: white;
}

.actionInfo {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.actionText {
  font-weight: 600;
  color: #1e293b;
  font-size: 0.875rem;
}

.phaseText {
  font-size: 0.75rem;
  color: #64748b;
  text-transform: capitalize;
}

.valueEstimate {
  color: hsl(var(--value-hue), 70%, 40%);
  background: hsl(var(--value-hue), 60%, 95%);
  font-weight: 600;
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  white-space: nowrap;
}

/* Metric strip for compact display of five values */
.metricsBar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: nowrap;
}

.metricChip {
  --chip-bg: hsl(var(--value-hue, 0), 60%, 97%);
  border: 1px solid color-mix(in oklab, hsl(var(--value-hue, 0), 85%, 50%) 30%, rgba(0,0,0,0.06));
  background: var(--chip-bg);
  border-radius: 8px;
  padding: 4px 6px;
  display: inline-flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px;
  min-width: 0;
}

/* Neutral chip (no hue tint) */
.metricChipNeutral {
  background: #f8fafc;
  border: 1px solid #e5e7eb;
}

.metricLabel {
  font-size: 0.65rem;
  letter-spacing: 0.02em;
  opacity: 0.7;
}

.metricValue {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  font-size: 0.75rem;
}

@media (max-width: 768px) {
  .metricsBar {
    flex-wrap: wrap;
  }
}

.expandIcon {
  color: #9ca3af;
  transition: transform 0.2s ease;
}

.expandIcon.expanded {
  transform: rotate(180deg);
}

/* Action Details */
.actionDetails {
  padding: 0 2rem 1.5rem 2rem;
  background: #f8fafc;
  border-top: 1px solid #f1f5f9;
}

.actionInsightsSection {
  padding: 0 2rem 1.5rem 2rem;
  background: #f8fafc;
  border-top: 1px solid #eef2ff;
}

.insightsGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.5rem;
  margin-top: 1.25rem;
}

.insightCard {
  background: white;
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  border: 1px solid #e2e8f0;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.insightTitle {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #475569;
}

.secretInsight {
  background: linear-gradient(145deg, #fff7ed 0%, #fffbeb 100%);
  border-color: #fed7aa;
}

.secretMini {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
  align-items: flex-start;
}

.secretMiniValue {
  font-size: 1.5rem;
  font-weight: 700;
  color: #9a3412;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}

.secretMiniLabel {
  font-size: 0.75rem;
  color: #b45309;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.pointsInsight {
  background: linear-gradient(145deg, #eff6ff 0%, #f8fafc 100%);
  border-color: #c7d2fe;
}

.pointsList {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.pointRow {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.pointSeat {
  min-width: 110px;
  font-weight: 600;
  color: #1e3a8a;
  font-size: 0.9rem;
}

.pointSeat span {
  display: block;
  font-size: 0.7rem;
  color: #6b7280;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.pointBarWrapper {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.pointBarTrack {
  flex: 1;
  height: 8px;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
}

.pointBarFill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #1e3a8a 100%);
  border-radius: 999px;
  transition: width 200ms ease;
}

.pointValue {
  min-width: 72px;
  font-weight: 700;
  color: #0f172a;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.detailsGrid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 1rem;
}

@media (max-width: 768px) {
  .detailsGrid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }

  .actionInsightsSection {
    padding: 0 1rem 1.25rem 1rem;
  }

  .insightsGrid {
    grid-template-columns: 1fr;
  }
}

.detailSection {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.detailTitle {
  font-weight: 600;
  color: #374151;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.cardList {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.card {
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 500;
  color: #374151;
}

/* Probability Bars */
.probabilitiesList {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.probabilityItem {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.probabilityAction {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
  min-width: 80px;
}

.probabilityBarContainer {
  flex: 1;
  height: 8px;
  background: #f3f4f6;
  border-radius: 4px;
  overflow: hidden;
}

.probabilityBar {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.probabilityValue {
  font-weight: 600;
  font-size: 0.75rem;
  color: #6b7280;
  min-width: 40px;
  text-align: right;
}

/* Subtle inline logits */
.probabilityLogit {
  color: #9ca3af; /* gray-400 */
  font-size: 0.75rem;
  width: 60px;
  display: inline-block;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  margin-left: 10px;
  transition: color 120ms ease;
  font-variant-numeric: tabular-nums;
  border-left: 2px solid #e5e7eb;
}
.probabilityItem:hover .probabilityLogit {
  color: #6b7280; /* gray-500 on hover */
}

/* State Vector Section */
.stateVectorSection {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.toggleButton {
  background: none;
  border: 1px solid #d1d5db;
  color: #6b7280;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.toggleButton:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

/* State Vector Table */
.stateVectorTableWrapper {
  margin-top: 1rem;
  margin-bottom: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
  background: white;
}

.stateVectorTableContainer {
  /* max-height: 800px; */
  overflow: auto;
}

.stateVectorTable {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.8125rem;
}

.stateVectorHead th {
  position: sticky;
  top: 0;
  background: #f8fafc;
  z-index: 1;
  text-align: left;
  font-weight: 600;
  color: #475569;
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #e5e7eb;
  border-right: 1px solid #e5e7eb;
}
.stateVectorHead th:last-child { border-right: none; }

.stateVectorCell,
.stateVectorIndex,
.stateVectorValue,
.stateVectorMicroCat,
.stateVectorSubCat,
.stateVectorCategory {
  padding: 0.5rem 0.75rem;
  border: 1px solid #e5e7eb;
  vertical-align: top;
}
.stateVectorTable td:last-child { border-right: none; }

.stateVectorIndex {
  font-variant-numeric: tabular-nums;
  color: #64748b;
  background: #f9fafb;
  min-width: 64px;
}

.stateVectorValue {
  font-variant-numeric: tabular-nums;
  font-family: 'Monaco', 'Consolas', monospace;
  color: #111827;
  min-width: 72px;
}

.valueHighlighted {
  background: #fcf9e4;
  font-weight: 700;
}

.cellHighlighted {
  background: #fcf9e4;
}

.stateVectorMicroCat {
  color: #374151;
}

.stateVectorSubCat {
  color: #1f2937;
  font-weight: 600;
  vertical-align: middle;
}

.stateVectorCategory {
  color: #111827;
  font-weight: 700;
  vertical-align: middle;
}

.rowHeader {
  background: #f8fafc;
}

.rowPrivate {
  background: #fafaf9;
}

.rowTrick {
  background: #f9fafb;
}

.stateVectorFoot td {
  padding: 0.5rem 0.75rem;
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
  color: #64748b;
}

/* Group separators */
.subGroupStart td { border-top: 2px solid #cbd5e1; }
.catGroupStart td { border-top: 3px solid #94a3b8; }

.subGroupDim td {
  color: #9ca3af;
}

/* Loading State */
.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 3rem;
  color: #6b7280;
}

.spinner {
  border: 2px solid #f3f4f6;
  border-radius: 50%;
  border-top: 2px solid #3b82f6;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;
  margin-right: 0.75rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Game Summary */
.gameSummary {
  background: #f8fafc;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid #e2e8f0;
}

.summarySection {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.summaryTitle {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
}

.summarySubsection {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.summarySubtitle {
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: 0;
}

.handsGrid {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.handRow {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.playerName {
  font-weight: 600;
  color: #374151;
  min-width: 4rem;
  text-align: left;
}

.gameInfoGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
}

.gameInfoItem {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.infoLabel {
  font-weight: 500;
  color: #6b7280;
  font-size: 0.875rem;
}

.infoValue {
  font-weight: 600;
  color: #374151;
  font-size: 0.875rem;
}

.scoresGrid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.scoreItem {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.scorePlayer {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
}

.scoreValue {
  font-weight: 600;
  font-size: 0.875rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  background: #f3f4f6;
  color: #6b7280;
}

.positiveScore {
  background: #dcfce7;
  color: #166534;
}

.negativeScore {
  background: #fee2e2;
  color: #dc2626;
}

/* Dividers */
.phaseDivider {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 2rem;
  padding: 0.75rem 0;
}

.phaseDividerLine {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, #9ca3af 50%, transparent 100%);
}

.phaseDividerLabel {
  font-size: 0.75rem;
  font-weight: 500;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
  padding: 0.375rem 0.75rem;
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 16px;
}

.trickDivider {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.75rem 2rem;
  padding: 0.5rem 0;
}

.trickDividerLine {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
}

.trickDividerLabel {
  font-size: 0.75rem;
  font-weight: 500;
  color: #64748b;
  text-transform: capitalize;
  letter-spacing: 0.025em;
  white-space: nowrap;
  padding: 0.25rem 0.75rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
}

/* Empty State */
.emptyState {
  text-align: center;
  padding: 3rem;
  color: #6b7280;
}

.emptyState h3 {
  color: #374151;
  margin-bottom: 0.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }

  .title {
    font-size: 2rem;
  }

  .controlsGrid {
    grid-template-columns: 1fr;
  }

  .actionSummary {
    padding: 1rem;
    flex-direction: column;
    align-items: stretch;
    gap: 0.75rem;
  }

  .actionBasic {
    justify-content: space-between;
  }

  .actionDetails {
    padding: 1rem;
  }

  .phaseDivider {
    margin: 0.75rem 1rem;
    gap: 0.75rem;
  }

  .phaseDividerLabel {
    font-size: 0.625rem;
    padding: 0.25rem 0.5rem;
  }

  .trickDivider {
    margin: 0.5rem 1rem;
    gap: 0.5rem;
  }

  .trickDividerLabel {
    font-size: 0.625rem;
    padding: 0.25rem 0.5rem;
  }
}

```

```typescriptreact
import React, { useMemo, useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ProbabilityBar from './ProbabilityBar';
import { PlayingCard } from '../../lib/components';
import styles from './page.module.css';

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const [showStateVector, setShowStateVector] = useState(false);

  const { view, probabilities, state } = action;
  const maxProb = Math.max(...probabilities.map(p => p.prob));

  // --- Constants mirroring backend deck order and special tokens ---
  const TRUMP: string[] = useMemo(() => [
    'QC', 'QS', 'QH', 'QD', 'JC', 'JS', 'JH', 'JD', 'AD', '10D', 'KD', '9D', '8D', '7D'
  ], []);
  const FAIL: string[] = useMemo(() => [
    'AC', '10C', 'KC', '9C', '8C', '7C',
    'AS', '10S', 'KS', '9S', '8S', '7S',
    'AH', '10H', 'KH', '9H', '8H', '7H',
  ], []);
  const DECK: string[] = useMemo(() => [...TRUMP, ...FAIL], [TRUMP, FAIL]);
  const UNDER_TOKEN = 'UNDER';
  const STATE_SIZE = 292;

  type Row = { idx: number; value: number; micro: string; sub: string; cat: 'Header' | 'Private Card Indicators' | 'Current Trick Block' };

  const rows: Row[] = useMemo(() => {
    if (!state || state.length === 0) return [];

    const r: Row[] = [];

    // Header (0-15)
    const headerMicro: string[] = [
      'Partner selection mode (0 = JD, 1 = Called Ace)',
      'Player position (1-5)',
      'Last position to pass (5 ⇒ all passed → Leaster)',
      'Picker position (0 if not yet picked)',
      'Partner position (0 if unknown)',
      'Alone called flag (0/1)',
      'Called card one-hot — AC',
      'Called card one-hot — AS',
      'Called card one-hot — AH',
      'Called card one-hot — 10C',
      'Called card one-hot — 10S',
      'Called card one-hot — 10H',
      'Is called-under flag (0/1)',
      'Is leaster flag (0/1)',
      'Play started flag (0/1)',
      'Current trick index (0-5)'
    ];

    const headerSubs: string[] = [
      'Core header', 'Core header', 'Core header', 'Core header', 'Core header', 'Core header',
      'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot',
      'Flags', 'Flags', 'Flags', 'Current trick index'
    ];

    for (let i = 0; i <= 15; i++) {
      r.push({ idx: i, value: state[i] ?? 0, micro: headerMicro[i], sub: headerSubs[i], cat: 'Header' });
    }

    // Private zones (Hand, Blind, Bury)
    const glyph = (s: 'C'|'S'|'H'|'D'): string => (s === 'C' ? '♣' : s === 'S' ? '♠' : s === 'H' ? '♥' : '♦');
    const pushPrivate = (start: number, count: number, sub: string, labelPrefix: string) => {
      for (let i = 0; i < count; i++) {
        const gi = start + i;
        const cardLabel = DECK[i] ?? `Card ${i + 1}`;
        const suit = cardLabel.endsWith('C') ? 'C' : cardLabel.endsWith('S') ? 'S' : cardLabel.endsWith('H') ? 'H' : 'D';
        const withGlyph = `${cardLabel} ${glyph(suit as any)}`;
        r.push({ idx: gi, value: state[gi] ?? 0, micro: `${labelPrefix} — ${withGlyph}`, sub, cat: 'Private Card Indicators' });
      }
    };
    // 16-47 Hand, 48-79 Blind, 80-111 Bury
    pushPrivate(16, 32, 'Hand one-hot', 'Hand');
    pushPrivate(48, 32, 'Blind one-hot', 'Blind');
    pushPrivate(80, 32, 'Bury one-hot', 'Bury');

    // Current trick block (112+) — 5 seats × (34 card one-hot + 2 flags)
    const seatLabels = ['Current Trick — Self', 'Current Trick — Player to the left', 'Current Trick — Player 2 to the left', 'Current Trick — Player 2 to the right', 'Current Trick — Player to the right'];
    const startBase = 112;
    for (let rel = 0; rel < 5; rel++) {
      const seatStart = startBase + rel * 36;
      // 34 card one-hot entries
      for (let j = 0; j < 34; j++) {
        const gi = seatStart + j;
        let cardDesc: string;
        if (j === 0) cardDesc = 'Card one-hot — empty';
        else if (j === 33) cardDesc = `Card one-hot — ${UNDER_TOKEN}`;
        else {
          const label = DECK[j - 1];
          const s = label.endsWith('C') ? 'C' : label.endsWith('S') ? 'S' : label.endsWith('H') ? 'H' : 'D';
          cardDesc = `Card one-hot — ${label} ${glyph(s as any)}`;
        }
        r.push({ idx: gi, value: state[gi] ?? 0, micro: cardDesc, sub: seatLabels[rel], cat: 'Current Trick Block' });
      }
      // 2 role flags
      r.push({ idx: seatStart + 34, value: state[seatStart + 34] ?? 0, micro: 'Role flag — is_picker (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
      r.push({ idx: seatStart + 35, value: state[seatStart + 35] ?? 0, micro: 'Role flag — is_known_partner (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
    }

    return r.slice(0, STATE_SIZE);
  }, [state, DECK]);

  // Compute rowSpans for sub and cat columns by contiguous groups
  const { subRowSpan, catRowSpan, rowClasses, subStartIdx, catStartIdx, subZeroOnly } = useMemo(() => {
    const subRowSpanMap: Record<number, number> = {};
    const catRowSpanMap: Record<number, number> = {};
    const classes: Record<number, string> = {};
    const subStart: Record<number, boolean> = {};
    const catStart: Record<number, boolean> = {};
    const subZero: Record<number, boolean> = {};

    // Helper to assign background class by large category
    const catClass = (cat: Row['cat']): string => {
      if (cat === 'Header') return styles.rowHeader;
      if (cat === 'Private Card Indicators') return styles.rowPrivate;
      return styles.rowTrick;
    };

    // Category groups
    let i = 0;
    while (i < rows.length) {
      const cat = rows[i]?.cat;
      if (!cat) break;
      let j = i;
      while (j < rows.length && rows[j].cat === cat) j++;
      catRowSpanMap[i] = j - i;
      catStart[i] = true;
      for (let k = i; k < j; k++) classes[k] = catClass(cat);

      // Within this category, compute sub-groups
      let s = i;
      while (s < j) {
        const sub = rows[s].sub;
        let t = s;
        while (t < j && rows[t].sub === sub) t++;
        subRowSpanMap[s] = t - s;
        subStart[s] = true;
        // Determine if every row value within this sub-group is zero
        let allZero = true;
        for (let k = s; k < t; k++) {
          if (rows[k].value) { allZero = false; break; }
        }
        subZero[s] = allZero;
        s = t;
      }

      i = j;
    }

    return { subRowSpan: subRowSpanMap, catRowSpan: catRowSpanMap, rowClasses: classes, subStartIdx: subStart, catStartIdx: catStart, subZeroOnly: subZero };
  }, [rows, styles.rowHeader, styles.rowPrivate, styles.rowTrick]);

  // Extract relevant cards from view
  const hand = view.hand || [];
  const blind = view.blind || [];
  const bury = view.bury || [];
  const currentTrick = view.current_trick || [];

  return (
    <div className={styles.actionDetails}>
      <div className={styles.detailsGrid}>
        {/* Game State */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Game State</div>

          {hand.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Hand
              </div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Blind
              </div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Bury
              </div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== '') && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Current Trick
              </div>
              <div className={styles.cardList}>
                {currentTrick.filter((card: string) => card !== '').map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
              {view.picker && <div>Picker: Seat {view.picker}</div>}
              {view.partner && <div>Partner: Seat {view.partner}</div>}
              {view.called_card && <div>Called Card: {view.called_card}</div>}
              {view.is_leaster && <div>Mode: Leaster</div>}
              {view.current_trick_index !== undefined && (
                <div>Trick: {view.current_trick_index + 1}/6</div>
              )}
            </div>
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar
                key={i}
                probability={prob}
                maxProb={maxProb}
              />
            ))}
          </div>
        </div>
      </div>

      {/* State Vector Section */}
      {state && (
        <div className={styles.stateVectorSection}>
          <button
            className={styles.toggleButton}
            onClick={() => setShowStateVector(!showStateVector)}
          >
            {showStateVector ? 'Hide' : 'Show'} State Vector
          </button>

          {showStateVector && (
            <>
              <div className={styles.stateVectorTableWrapper}>
                <div className={styles.stateVectorTableContainer}>
                  <table className={styles.stateVectorTable}>
                    <thead className={styles.stateVectorHead}>
                      <tr>
                        <th>Index</th>
                        <th>Value</th>
                        <th>Detail</th>
                        <th colSpan={2}>Category</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => {
                        const showSub = subRowSpan[i] !== undefined;
                        const showCat = catRowSpan[i] !== undefined;
                        const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                        // Find the index where this row's sub-group starts to apply dimming
                        let subAnchor = i;
                        while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                        const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                        return (
                          <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                            <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                            <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                            <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                            {showSub && (
                              <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>{row.sub}</td>
                            )}
                            {showCat && (
                              <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>{row.cat}</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <button
                className={styles.toggleButton}
                onClick={() => setShowStateVector(!showStateVector)}
              >
                ⌃ Collapse
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

```

Updates made per your feedback:

- **ActionDetails** now strictly shows the game state and action probabilities first when an action row expands; predictive insights were removed from this component entirely.
- **New `ActionInsights` component** renders the supplementary predictions (secret-partner confidence + per-seat point forecasts) beneath the main details. It’s imported and rendered from `ActionRow`, so expanded rows now show details first, then insights.
- **Secret partner UI** switched to a compact labeled percentage chip, keeping it secondary while still readable.
- **Point predictions** remain in a separate card with elegant bars, reusing the styling previously created but now scoped to the dedicated component. Supporting CSS classes were added/adjusted, and unused ones (like the large gauge) were replaced.

All lint checks pass. Let me know if you’d like further styling tweaks or additional metrics surfaced.

---

**User**

Split out the "Show State Vector" from the ActionDetails component and make that its own component. That button and table should be below the ActionInsights component.

---

**Cursor**

```typescriptreact
import React from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ProbabilityBar from './ProbabilityBar';
import { PlayingCard } from '../../lib/components';
import styles from './page.module.css';

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const { view, probabilities } = action;
  const maxProb = Math.max(...probabilities.map((p) => p.prob));

  const hand = view.hand || [];
  const blind = view.blind || [];
  const bury = view.bury || [];
  const currentTrick = view.current_trick || [];

  return (
    <div className={styles.actionDetails}>
      <div className={styles.detailsGrid}>
        {/* Game State */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Game State</div>

          {hand.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Hand
              </div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Blind
              </div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Bury
              </div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== '') && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Current Trick
              </div>
              <div className={styles.cardList}>
                {currentTrick.filter((card: string) => card !== '').map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
              {view.picker && <div>Picker: Seat {view.picker}</div>}
              {view.partner && <div>Partner: Seat {view.partner}</div>}
              {view.called_card && <div>Called Card: {view.called_card}</div>}
              {view.is_leaster && <div>Mode: Leaster</div>}
              {view.current_trick_index !== undefined && (
                <div>Trick: {view.current_trick_index + 1}/6</div>
              )}
            </div>
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar
                key={i}
                probability={prob}
                maxProb={maxProb}
              />
            ))}
          </div>
        </div>
      </div>

      {/* State Vector Section */}
      {state && (
        <div className={styles.stateVectorSection}>
          <button
            className={styles.toggleButton}
            onClick={() => setShowStateVector(!showStateVector)}
          >
            {showStateVector ? 'Hide' : 'Show'} State Vector
          </button>

          {showStateVector && (
            <>
              <div className={styles.stateVectorTableWrapper}>
                <div className={styles.stateVectorTableContainer}>
                  <table className={styles.stateVectorTable}>
                    <thead className={styles.stateVectorHead}>
                      <tr>
                        <th>Index</th>
                        <th>Value</th>
                        <th>Detail</th>
                        <th colSpan={2}>Category</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => {
                        const showSub = subRowSpan[i] !== undefined;
                        const showCat = catRowSpan[i] !== undefined;
                        const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                        // Find the index where this row's sub-group starts to apply dimming
                        let subAnchor = i;
                        while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                        const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                        return (
                          <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                            <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                            <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                            <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                            {showSub && (
                              <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>{row.sub}</td>
                            )}
                            {showCat && (
                              <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>{row.cat}</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <button
                className={styles.toggleButton}
                onClick={() => setShowStateVector(!showStateVector)}
              >
                ⌃ Collapse
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

```

```typescriptreact
import React, { useMemo, useState } from 'react';
import styles from './page.module.css';

const TRUMP = [
  'QC', 'QS', 'QH', 'QD', 'JC', 'JS', 'JH', 'JD', 'AD', '10D', 'KD', '9D', '8D', '7D'
] as const;
const FAIL = [
  'AC', '10C', 'KC', '9C', '8C', '7C',
  'AS', '10S', 'KS', '9S', '8S', '7S',
  'AH', '10H', 'KH', '9H', '8H', '7H',
] as const;
const DECK = [...TRUMP, ...FAIL];
const UNDER_TOKEN = 'UNDER';
const STATE_SIZE = 292;

type Row = {
  idx: number;
  value: number;
  micro: string;
  sub: string;
  cat: 'Header' | 'Private Card Indicators' | 'Current Trick Block';
};

interface ActionStateVectorProps {
  state?: number[];
}

export default function ActionStateVector({ state }: ActionStateVectorProps) {
  const [expanded, setExpanded] = useState(false);

  const rows: Row[] = useMemo(() => {
    if (!state || state.length === 0) return [];

    const r: Row[] = [];

    const headerMicro: string[] = [
      'Partner selection mode (0 = JD, 1 = Called Ace)',
      'Player position (1-5)',
      'Last position to pass (5 ⇒ all passed → Leaster)',
      'Picker position (0 if not yet picked)',
      'Partner position (0 if unknown)',
      'Alone called flag (0/1)',
      'Called card one-hot — AC',
      'Called card one-hot — AS',
      'Called card one-hot — AH',
      'Called card one-hot — 10C',
      'Called card one-hot — 10S',
      'Called card one-hot — 10H',
      'Is called-under flag (0/1)',
      'Is leaster flag (0/1)',
      'Play started flag (0/1)',
      'Current trick index (0-5)'
    ];

    const headerSubs: string[] = [
      'Core header', 'Core header', 'Core header', 'Core header', 'Core header', 'Core header',
      'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot',
      'Flags', 'Flags', 'Flags', 'Current trick index'
    ];

    for (let i = 0; i <= 15; i++) {
      r.push({ idx: i, value: state[i] ?? 0, micro: headerMicro[i], sub: headerSubs[i], cat: 'Header' });
    }

    const glyph = (s: 'C' | 'S' | 'H' | 'D'): string => (s === 'C' ? '♣' : s === 'S' ? '♠' : s === 'H' ? '♥' : '♦');

    const pushPrivate = (start: number, count: number, sub: string, labelPrefix: string) => {
      for (let i = 0; i < count; i++) {
        const gi = start + i;
        const cardLabel = DECK[i] ?? `Card ${i + 1}`;
        const suit = cardLabel.endsWith('C') ? 'C' : cardLabel.endsWith('S') ? 'S' : cardLabel.endsWith('H') ? 'H' : 'D';
        const withGlyph = `${cardLabel} ${glyph(suit as any)}`;
        r.push({ idx: gi, value: state[gi] ?? 0, micro: `${labelPrefix} — ${withGlyph}`, sub, cat: 'Private Card Indicators' });
      }
    };

    pushPrivate(16, 32, 'Hand one-hot', 'Hand');
    pushPrivate(48, 32, 'Blind one-hot', 'Blind');
    pushPrivate(80, 32, 'Bury one-hot', 'Bury');

    const seatLabels = [
      'Current Trick — Self',
      'Current Trick — Player to the left',
      'Current Trick — Player 2 to the left',
      'Current Trick — Player 2 to the right',
      'Current Trick — Player to the right'
    ];
    const startBase = 112;

    for (let rel = 0; rel < 5; rel++) {
      const seatStart = startBase + rel * 36;
      for (let j = 0; j < 34; j++) {
        const gi = seatStart + j;
        let cardDesc: string;
        if (j === 0) cardDesc = 'Card one-hot — empty';
        else if (j === 33) cardDesc = `Card one-hot — ${UNDER_TOKEN}`;
        else {
          const label = DECK[j - 1];
          const s = label.endsWith('C') ? 'C' : label.endsWith('S') ? 'S' : label.endsWith('H') ? 'H' : 'D';
          cardDesc = `Card one-hot — ${label} ${glyph(s as any)}`;
        }
        r.push({ idx: gi, value: state[gi] ?? 0, micro: cardDesc, sub: seatLabels[rel], cat: 'Current Trick Block' });
      }
      r.push({ idx: seatStart + 34, value: state[seatStart + 34] ?? 0, micro: 'Role flag — is_picker (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
      r.push({ idx: seatStart + 35, value: state[seatStart + 35] ?? 0, micro: 'Role flag — is_known_partner (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
    }

    return r.slice(0, STATE_SIZE);
  }, [state]);

  const { subRowSpan, catRowSpan, rowClasses, subStartIdx, catStartIdx, subZeroOnly } = useMemo(() => {
    const subRowSpanMap: Record<number, number> = {};
    const catRowSpanMap: Record<number, number> = {};
    const classes: Record<number, string> = {};
    const subStart: Record<number, boolean> = {};
    const catStart: Record<number, boolean> = {};
    const subZero: Record<number, boolean> = {};

    const catClass = (cat: Row['cat']): string => {
      if (cat === 'Header') return styles.rowHeader;
      if (cat === 'Private Card Indicators') return styles.rowPrivate;
      return styles.rowTrick;
    };

    let i = 0;
    while (i < rows.length) {
      const cat = rows[i]?.cat;
      if (!cat) break;
      let j = i;
      while (j < rows.length && rows[j].cat === cat) j++;
      catRowSpanMap[i] = j - i;
      catStart[i] = true;
      for (let k = i; k < j; k++) classes[k] = catClass(cat);

      let s = i;
      while (s < j) {
        const sub = rows[s].sub;
        let t = s;
        while (t < j && rows[t].sub === sub) t++;
        subRowSpanMap[s] = t - s;
        subStart[s] = true;
        let allZero = true;
        for (let k = s; k < t; k++) {
          if (rows[k].value) { allZero = false; break; }
        }
        subZero[s] = allZero;
        s = t;
      }

      i = j;
    }

    return { subRowSpan: subRowSpanMap, catRowSpan: catRowSpanMap, rowClasses: classes, subStartIdx: subStart, catStartIdx: catStart, subZeroOnly: subZero };
  }, [rows]);

  if (!state || state.length === 0) {
    return null;
  }

  return (
    <div className={styles.stateVectorSection}>
      <button className={styles.toggleButton} onClick={() => setExpanded(!expanded)}>
        {expanded ? 'Hide' : 'Show'} State Vector
      </button>

      {expanded && (
        <>
          <div className={styles.stateVectorTableWrapper}>
            <div className={styles.stateVectorTableContainer}>
              <table className={styles.stateVectorTable}>
                <thead className={styles.stateVectorHead}>
                  <tr>
                    <th>Index</th>
                    <th>Value</th>
                    <th>Detail</th>
                    <th colSpan={2}>Category</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => {
                    const showSub = subRowSpan[i] !== undefined;
                    const showCat = catRowSpan[i] !== undefined;
                    const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                    let subAnchor = i;
                    while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                    const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                    return (
                      <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                        <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                        <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                        <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                        {showSub && (
                          <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>
                            {row.sub}
                          </td>
                        )}
                        {showCat && (
                          <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>
                            {row.cat}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <button className={styles.toggleButton} onClick={() => setExpanded(!expanded)}>
            ⌃ Collapse
          </button>
        </>
      )}
    </div>
  );
}

```

```typescriptreact
import React, { useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionDetails from './ActionDetails';
import ActionInsights from './ActionInsights';
import ActionStateVector from './ActionStateVector';
import { CardText } from '../../lib/components';
import styles from './page.module.css';

interface ActionRowProps {
  action: AnalyzeActionDetail;
  picker?: string;
  partner?: string;
  normalizedValue: number; // 0-1 scale for color gradient (value)
  normalizedReward: number; // 0-1 scale for color gradient (reward)
  normalizedStepReward?: number; // 0-1 scale for color gradient (immediate step reward)
}

export default function ActionRow({ action, picker, partner, normalizedValue, normalizedReward, normalizedStepReward }: ActionRowProps) {
  const [expanded, setExpanded] = useState(false);

  const handleClick = () => {
    setExpanded(!expanded);
  };

  const formatValue = (value: number) => {
    return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
  };

  const getPlayerRole = () => {
    if (action.seatName === picker) return 'picker';
    if (action.seatName === partner) return 'partner';
    return 'other';
  };

  const getSeatBadgeClass = () => {
    const role = getPlayerRole();
    switch (role) {
      case 'picker':
        return `${styles.seatBadge} ${styles.seatBadgePicker}`;
      case 'partner':
        return `${styles.seatBadge} ${styles.seatBadgePartner}`;
      default:
        return `${styles.seatBadge} ${styles.seatBadgeOther}`;
    }
  };

  const getHueStyle = (normalized: number) => {
    const hue = normalized * 120; // 0 (red) .. 120 (green)
    return { '--value-hue': `${hue}deg` } as React.CSSProperties;
  };

  const formatSigned = (value: number, digits = 3) => {
    return value >= 0 ? `+${value.toFixed(digits)}` : value.toFixed(digits);
  };

  function MetricChip({
    label,
    value,
    colorStyle,
    tooltip,
    suffix = ''
  }: {
    label: React.ReactNode;
    value: string;
    colorStyle?: React.CSSProperties;
    tooltip?: string;
    suffix?: string;
  }) {
    return (
      <div className={styles.metricChip} title={tooltip} style={colorStyle}>
        <span className={styles.metricLabel}>{label}</span>
        <span className={styles.metricValue}>{value}{suffix}</span>
      </div>
    );
  }

  function MetricsBar({
    value,
    valueHue,
    discounted,
    discountedHue,
    step,
    stepHue,
    winPct,
    expectedFinal,
  }: {
    value: number;
    valueHue: number;
    discounted?: number;
    discountedHue?: number;
    step?: number;
    stepHue?: number;
    winPct?: number;
    expectedFinal?: number;
  }) {
    return (
      <div className={styles.metricsBar}>
        <MetricChip
          label={<>V</>}
          value={formatSigned(value)}
          colorStyle={getHueStyle(valueHue)}
          tooltip="Value estimate (critic)"
        />

        {typeof discounted === 'number' && (
          <MetricChip
            label={<>G<sub>t</sub></>}
            value={formatSigned(discounted)}
            colorStyle={getHueStyle(typeof discountedHue === 'number' ? discountedHue : 0.5)}
            tooltip="Discounted return"
          />
        )}

        {typeof step === 'number' && (
          <MetricChip
            label={<>r</>}
            value={formatSigned(step)}
            colorStyle={getHueStyle(typeof stepHue === 'number' ? stepHue : 0.5)}
            tooltip="Immediate step reward"
          />
        )}

        {typeof winPct === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Win probability">
            <span className={styles.metricLabel}>Win</span>
            <span className={styles.metricValue}>{`${Math.round(winPct)}%`}</span>
          </div>
        )}

        {typeof expectedFinal === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Expected final return (undiscounted)">
            <span className={styles.metricLabel}>E[Ret]</span>
            <span className={styles.metricValue}>{expectedFinal.toFixed(2)}</span>
          </div>
        )}

      </div>
    );
  }

  return (
    <div className={styles.actionRow}>
      <div
        className={styles.actionSummary}
        onClick={handleClick}
      >
        <div className={styles.actionBasic}>
          <div className={styles.stepIndex}>
            {action.stepIndex + 1}
          </div>

          <div className={getSeatBadgeClass()}>
            {action.seatName}
          </div>

          <div className={styles.actionInfo}>
            <div className={styles.actionText}>
              <CardText>{action.action}</CardText>
            </div>
            <div className={styles.phaseText}>
              {action.phase} Head
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <MetricsBar
            value={action.valueEstimate}
            valueHue={normalizedValue}
            discounted={typeof action.discountedReturn === 'number' ? action.discountedReturn : undefined}
            discountedHue={normalizedReward}
            step={typeof action.stepReward === 'number' ? action.stepReward : undefined}
            stepHue={typeof normalizedStepReward === 'number' ? normalizedStepReward : undefined}
            winPct={typeof action.winProb === 'number' ? (action.winProb * 100) : undefined}
            expectedFinal={typeof action.expectedFinalReturn === 'number' ? action.expectedFinalReturn : undefined}
          />

          <div className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}>▼</div>
        </div>
      </div>

      {expanded && (
        <>
          <ActionDetails action={action} />
          <ActionInsights action={action} />
          <ActionStateVector action={action} />
        </>
      )}
    </div>
  );
}

```

```typescriptreact
import React, { useMemo, useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import styles from './page.module.css';

const TRUMP = [
  'QC', 'QS', 'QH', 'QD', 'JC', 'JS', 'JH', 'JD', 'AD', '10D', 'KD', '9D', '8D', '7D'
] as const;
const FAIL = [
  'AC', '10C', 'KC', '9C', '8C', '7C',
  'AS', '10S', 'KS', '9S', '8S', '7S',
  'AH', '10H', 'KH', '9H', '8H', '7H',
] as const;
const DECK = [...TRUMP, ...FAIL];
const UNDER_TOKEN = 'UNDER';
const STATE_SIZE = 292;

type Row = {
  idx: number;
  value: number;
  micro: string;
  sub: string;
  cat: 'Header' | 'Private Card Indicators' | 'Current Trick Block';
};

interface ActionStateVectorProps {
  action: AnalyzeActionDetail;
}

export default function ActionStateVector({ action }: ActionStateVectorProps) {
  const [expanded, setExpanded] = useState(false);
  const state = action.state;

  const rows: Row[] = useMemo(() => {
    if (!state || state.length === 0) return [];

    const r: Row[] = [];

    const headerMicro: string[] = [
      'Partner selection mode (0 = JD, 1 = Called Ace)',
      'Player position (1-5)',
      'Last position to pass (5 ⇒ all passed → Leaster)',
      'Picker position (0 if not yet picked)',
      'Partner position (0 if unknown)',
      'Alone called flag (0/1)',
      'Called card one-hot — AC',
      'Called card one-hot — AS',
      'Called card one-hot — AH',
      'Called card one-hot — 10C',
      'Called card one-hot — 10S',
      'Called card one-hot — 10H',
      'Is called-under flag (0/1)',
      'Is leaster flag (0/1)',
      'Play started flag (0/1)',
      'Current trick index (0-5)'
    ];

    const headerSubs: string[] = [
      'Core header', 'Core header', 'Core header', 'Core header', 'Core header', 'Core header',
      'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot', 'Called card one-hot',
      'Flags', 'Flags', 'Flags', 'Current trick index'
    ];

    for (let i = 0; i <= 15; i++) {
      r.push({ idx: i, value: state[i] ?? 0, micro: headerMicro[i], sub: headerSubs[i], cat: 'Header' });
    }

    const glyph = (s: 'C' | 'S' | 'H' | 'D'): string => (s === 'C' ? '♣' : s === 'S' ? '♠' : s === 'H' ? '♥' : '♦');

    const pushPrivate = (start: number, count: number, sub: string, labelPrefix: string) => {
      for (let i = 0; i < count; i++) {
        const gi = start + i;
        const cardLabel = DECK[i] ?? `Card ${i + 1}`;
        const suit = cardLabel.endsWith('C') ? 'C' : cardLabel.endsWith('S') ? 'S' : cardLabel.endsWith('H') ? 'H' : 'D';
        const withGlyph = `${cardLabel} ${glyph(suit as any)}`;
        r.push({ idx: gi, value: state[gi] ?? 0, micro: `${labelPrefix} — ${withGlyph}`, sub, cat: 'Private Card Indicators' });
      }
    };

    pushPrivate(16, 32, 'Hand one-hot', 'Hand');
    pushPrivate(48, 32, 'Blind one-hot', 'Blind');
    pushPrivate(80, 32, 'Bury one-hot', 'Bury');

    const seatLabels = [
      'Current Trick — Self',
      'Current Trick — Player to the left',
      'Current Trick — Player 2 to the left',
      'Current Trick — Player 2 to the right',
      'Current Trick — Player to the right'
    ];
    const startBase = 112;

    for (let rel = 0; rel < 5; rel++) {
      const seatStart = startBase + rel * 36;
      for (let j = 0; j < 34; j++) {
        const gi = seatStart + j;
        let cardDesc: string;
        if (j === 0) cardDesc = 'Card one-hot — empty';
        else if (j === 33) cardDesc = `Card one-hot — ${UNDER_TOKEN}`;
        else {
          const label = DECK[j - 1];
          const s = label.endsWith('C') ? 'C' : label.endsWith('S') ? 'S' : label.endsWith('H') ? 'H' : 'D';
          cardDesc = `Card one-hot — ${label} ${glyph(s as any)}`;
        }
        r.push({ idx: gi, value: state[gi] ?? 0, micro: cardDesc, sub: seatLabels[rel], cat: 'Current Trick Block' });
      }
      r.push({ idx: seatStart + 34, value: state[seatStart + 34] ?? 0, micro: 'Role flag — is_picker (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
      r.push({ idx: seatStart + 35, value: state[seatStart + 35] ?? 0, micro: 'Role flag — is_known_partner (0/1)', sub: seatLabels[rel], cat: 'Current Trick Block' });
    }

    return r.slice(0, STATE_SIZE);
  }, [state]);

  const { subRowSpan, catRowSpan, rowClasses, subStartIdx, catStartIdx, subZeroOnly } = useMemo(() => {
    const subRowSpanMap: Record<number, number> = {};
    const catRowSpanMap: Record<number, number> = {};
    const classes: Record<number, string> = {};
    const subStart: Record<number, boolean> = {};
    const catStart: Record<number, boolean> = {};
    const subZero: Record<number, boolean> = {};

    const catClass = (cat: Row['cat']): string => {
      if (cat === 'Header') return styles.rowHeader;
      if (cat === 'Private Card Indicators') return styles.rowPrivate;
      return styles.rowTrick;
    };

    let i = 0;
    while (i < rows.length) {
      const cat = rows[i]?.cat;
      if (!cat) break;
      let j = i;
      while (j < rows.length && rows[j].cat === cat) j++;
      catRowSpanMap[i] = j - i;
      catStart[i] = true;
      for (let k = i; k < j; k++) classes[k] = catClass(cat);

      let s = i;
      while (s < j) {
        const sub = rows[s].sub;
        let t = s;
        while (t < j && rows[t].sub === sub) t++;
        subRowSpanMap[s] = t - s;
        subStart[s] = true;
        let allZero = true;
        for (let k = s; k < t; k++) {
          if (rows[k].value) { allZero = false; break; }
        }
        subZero[s] = allZero;
        s = t;
      }

      i = j;
    }

    return { subRowSpan: subRowSpanMap, catRowSpan: catRowSpanMap, rowClasses: classes, subStartIdx: subStart, catStartIdx: catStart, subZeroOnly: subZero };
  }, [rows]);

  if (!state || state.length === 0) {
    return null;
  }

  return (
    <div className={styles.stateVectorSection}>
      <button className={styles.toggleButton} onClick={() => setExpanded(!expanded)}>
        {expanded ? 'Hide' : 'Show'} State Vector
      </button>

      {expanded && (
        <>
          <div className={styles.stateVectorTableWrapper}>
            <div className={styles.stateVectorTableContainer}>
              <table className={styles.stateVectorTable}>
                <thead className={styles.stateVectorHead}>
                  <tr>
                    <th>Index</th>
                    <th>Value</th>
                    <th>Detail</th>
                    <th colSpan={2}>Category</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => {
                    const showSub = subRowSpan[i] !== undefined;
                    const showCat = catRowSpan[i] !== undefined;
                    const groupClass = `${rowClasses[i]} ${subStartIdx[i] ? styles.subGroupStart : ''} ${catStartIdx[i] ? styles.catGroupStart : ''}`.trim();
                    let subAnchor = i;
                    while (subAnchor > 0 && rows[subAnchor - 1].sub === row.sub && rows[subAnchor - 1].cat === row.cat) subAnchor--;
                    const dimClass = subZeroOnly[subAnchor] ? styles.subGroupDim : '';
                    return (
                      <tr key={row.idx} className={`${groupClass} ${dimClass}`.trim()}>
                        <td className={`${styles.stateVectorIndex} ${row.value ? styles.cellHighlighted : ''}`}>{row.idx}</td>
                        <td className={`${styles.stateVectorValue} ${row.value ? styles.valueHighlighted : ''}`}>{row.value}</td>
                        <td className={`${styles.stateVectorMicroCat} ${row.value ? styles.cellHighlighted : ''}`}>{row.micro}</td>
                        {showSub && (
                          <td className={styles.stateVectorSubCat} rowSpan={subRowSpan[i]}>
                            {row.sub}
                          </td>
                        )}
                        {showCat && (
                          <td className={styles.stateVectorCategory} rowSpan={catRowSpan[i]}>
                            {row.cat}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <button className={styles.toggleButton} onClick={() => setExpanded(!expanded)}>
            ⌃ Collapse
          </button>
        </>
      )}
    </div>
  );
}


```

```typescriptreact
import React from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ProbabilityBar from './ProbabilityBar';
import { PlayingCard } from '../../lib/components';
import styles from './page.module.css';

interface ActionDetailsProps {
  action: AnalyzeActionDetail;
}

export default function ActionDetails({ action }: ActionDetailsProps) {
  const { view, probabilities } = action;
  const maxProb = Math.max(...probabilities.map((p) => p.prob));

  const hand = view.hand || [];
  const blind = view.blind || [];
  const bury = view.bury || [];
  const currentTrick = view.current_trick || [];

  return (
    <div className={styles.actionDetails}>
      <div className={styles.detailsGrid}>
        {/* Game State */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Game State</div>

          {hand.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Hand
              </div>
              <div className={styles.cardList}>
                {hand.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {blind.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Blind
              </div>
              <div className={styles.cardList}>
                {blind.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {bury.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Bury
              </div>
              <div className={styles.cardList}>
                {bury.map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {currentTrick.some((card: string) => card !== '') && (
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>
                Current Trick
              </div>
              <div className={styles.cardList}>
                {currentTrick.filter((card: string) => card !== '').map((card: string, i: number) => (
                  <PlayingCard key={i} label={card} small />
                ))}
              </div>
            </div>
          )}

          {/* Additional game context */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
              {view.picker && <div>Picker: Seat {view.picker}</div>}
              {view.partner && <div>Partner: Seat {view.partner}</div>}
              {view.called_card && <div>Called Card: {view.called_card}</div>}
              {view.is_leaster && <div>Mode: Leaster</div>}
              {view.current_trick_index !== undefined && (
                <div>Trick: {view.current_trick_index + 1}/6</div>
              )}
            </div>
          </div>
        </div>

        {/* Action Probabilities */}
        <div className={styles.detailSection}>
          <div className={styles.detailTitle}>Action Probabilities</div>
          <div className={styles.probabilitiesList}>
            {probabilities.map((prob, i) => (
              <ProbabilityBar
                key={i}
                probability={prob}
                maxProb={maxProb}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

```

Changes delivered:

- **ActionDetails** now strictly covers the game state + action probabilities; all state-vector logic was removed to keep this component focused.
- **New `ActionStateVector` component** encapsulates the “Show/Hide State Vector” button and table using the previous logic and styling. It renders only when an action supplies state data.
- **ActionRow** now renders components in order: `ActionDetails`, `ActionInsights`, then `ActionStateVector`, so the insights remain between the core info and the detailed state vector.
- **Styling** already supported the moved elements, so only the layout changed; no new styles were required beyond those we already had.

Lint checks pass. In the UI, expanding an action row now shows (1) game state and probabilities, (2) the compact insights card, and (3) the collapsible state vector, matching your requested ordering and structure.

---

**User**

Why is the if statement at ppo.py line 364 needed?

---

**Cursor**

That guard is just a safety check before converting the tensor to a Python list. `points_pred = self.points_head(aux_feat)` has shape `(batch_size, 5)`, but `aux_predictions` is called from analyze/inspect code where we pass a single observation but could (by mistake) pass an empty batch. If we blindly index `points_pred[0]` when `batch_size == 0`, PyTorch would raise an IndexError. The `if points_pred.dim() >= 2 and points_pred.size(0) > 0` ensures we only read the first row when there is at least one sample in this batch. If somehow `points_pred` came back as a 1‑D tensor (e.g., older checkpoints that lack the new head, or strict=False loads) the condition also prevents treating it as `(B,5)` and crashing. So it’s just defensive programming—skip emitting point estimates unless we have a valid (non-empty, 2‑D) batch to pull them from.

---

**User**

I think the only time this would realistically happen is if we load an old checkpoint, correct? Otherwise it should not be required.

---

**Cursor**

Right—under normal training/inference (where `points_head` exists and you always call `aux_predictions` with at least one observation), `points_pred` will always be a 2‑D tensor with batch size ≥ 1, so the guard is redundant.

The situations it protects against are:

1. **Empty batch**: if `aux_predictions` were ever called with an empty `encoder_out` (e.g., accidentally slicing `[:0]`). Not typical, but the check avoids an IndexError from `points_pred[0]`.

2. **Older checkpoints**: if you load a model saved before the points head existed, `self.points_head` won’t be in the checkpoint and PyTorch will keep its randomly-initialized weights, but it will still produce a tensor. So “missing head” isn’t a concern; the only compatibility issue would be a completely missing module, which strict=False already handles. However, that check also prevents the code from trying to interpret a 1‑D tensor (in case something odd happens during load) as `(B,5)`.

So yes—under realistic usage today it isn’t strictly needed, but it’s harmless defensive code that keeps the analyze endpoint from crashing if we ever pass an empty batch or load something with unexpected shapes.

---

**User**

Let's drop this guard. I'd rather error if we can't get a clean prediction and understand why.

---

**Cursor**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Dict
from sheepshead import ACTION_IDS, BURY_ACTIONS, CALL_ACTIONS, UNDER_ACTIONS, PLAY_ACTIONS, DECK_IDS, UNDER_TOKEN
from encoder import CardReasoningEncoder, CardEmbeddingConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network consuming encoder features, with separate heads for
    pick / partner-selection / bury / play phases. The four heads' logits are
    concatenated back into the full action space order so existing masking
    logic continues to work unchanged.
    """

    def __init__(self,
                 action_size,
                 action_groups,
                 activation='swish',
                 *,
                 d_card: int,
                 d_token: int,
                 map_cid_to_play_action_index: torch.Tensor,
                 map_cid_to_bury_action_index: torch.Tensor,
                 map_cid_to_under_action_index: torch.Tensor,
                 call_action_global_indices: torch.Tensor,
                 call_card_ids: torch.Tensor,
                 play_under_action_index: int):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'

        # Shared actor adapter to add nonlinearity and specialization before heads
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
        )

        # === Heads ===
        self.pick_head = nn.Linear(256, len(action_groups['pick']))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(256, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(256, 1)

        # Per-head temperatures (τ): logits will be divided by τ before softmax
        # τ > 1.0 softens (higher entropy), τ < 1.0 sharpens. Defaults to 1.0.
        self.temperature_pick = 1.0
        self.temperature_partner = 1.0
        self.temperature_bury = 1.0
        self.temperature_play = 1.0

        # Init (generic weights)
        self.apply(self._init_weights)

        # ---- Card-conditioned configuration (dims + mappings) ----
        self._d_card = int(d_card)
        self._d_token = int(d_token)
        # Pointer scorer (Bahdanau style)
        self.pointer_hidden = 64
        self.pointer_Wg = nn.Linear(256, self.pointer_hidden)
        self.pointer_Wt = nn.Linear(self._d_token, self.pointer_hidden)
        self.pointer_v = nn.Linear(self.pointer_hidden, 1, bias=False)
        # Two-tower (card CALL scoring)
        self.tw_latent = 64
        self.tw_Wg = nn.Linear(256, self.tw_latent)
        self.tw_We = nn.Linear(self._d_card, self.tw_latent)

        # Action index mappings
        self._map_cid_to_play_action_index = map_cid_to_play_action_index.clone().long()
        self._map_cid_to_bury_action_index = map_cid_to_bury_action_index.clone().long()
        self._map_cid_to_under_action_index = map_cid_to_under_action_index.clone().long()
        self._call_action_global_indices = call_action_global_indices.clone().long()
        self._call_card_ids = call_card_ids.clone().long()
        self._play_under_action_index = int(play_under_action_index)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------ internal helpers ------------------------
    def _score_cards_two_tower(self, feat: torch.Tensor, card_embedding: nn.Embedding) -> torch.Tensor:
        """Return per-card scores S for all 34 card ids using two-tower scorer.
        feat: (B, 256)
        Returns: (B, 34)
        """
        q = self.tw_Wg(feat)                      # (B, k)        <- tower 1 (actor features)
        table = card_embedding.weight             # (34, d_card)
        K = self.tw_We(table)                     # (34, k)       <- tower 2 (card embeddings)
        S = torch.matmul(q, K.t())    # (B, 34)       <- similarity scores
        return S

    def _score_hand_pointer(
        self,
        feat: torch.Tensor,
        hand_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score each hand slot using pointer scorer.
        feat: (B, 256); hand_tokens: (B, N, d_token)
        Returns: s_slots (B, N)
        """
        tok = hand_tokens
        B, N = tok.size(0), tok.size(1)
        g = self.pointer_Wg(feat).unsqueeze(1).expand(B, N, -1)  # (B, N, h)
        t = self.pointer_Wt(tok)                                   # (B, N, h)
        e = torch.tanh(g + t)                                      # (B, N, h)
        s = self.pointer_v(e).squeeze(-1)                          # (B, N)
        return s

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        hand_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        hand_tokens : Tensor (K, N, d_token)
        action_mask : BoolTensor (K, A) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # Adapt features
        feat = self.actor_adapter(actor_features)

        # PICK / PASS
        pick_logits = self.pick_head(feat) / max(self.temperature_pick, 1e-6)
        logits[:, self.action_groups['pick']] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(feat) / max(self.temperature_partner, 1e-6)
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(feat, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)] / max(self.temperature_partner, 1e-6)
        logits[:, self._call_action_global_indices.to(device)] = call_scores

        # Pointer scores over hand tokens (compute once, reuse for bury/under/play)
        slot_scores = self._score_hand_pointer(
            feat,
            hand_tokens,
        )  # (K, N)
        K_, N = slot_scores.size(0), slot_scores.size(1)
        cids = hand_ids.long()

        # BURY and UNDER scatter
        idx_bury = self._map_cid_to_bury_action_index.to(device)[cids]   # (K, N)
        idx_under = self._map_cid_to_under_action_index.to(device)[cids] # (K, N)
        for i in range(N):
            b_idx = idx_bury[:, i]
            u_idx = idx_under[:, i]
            valid_b = b_idx.ge(0)
            valid_u = u_idx.ge(0)
            if valid_b.any():
                logits.view(K_, -1)[valid_b, b_idx[valid_b]] = slot_scores[valid_b, i] / max(self.temperature_bury, 1e-6)
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[valid_u, i] / max(self.temperature_bury, 1e-6)

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[valid_p, i] / max(self.temperature_play, 1e-6)

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(feat).squeeze(-1) / max(self.temperature_play, 1e-6)
            logits[:, self._play_under_action_index] = play_under_logit

        # Apply action mask if provided
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def set_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Set per-head softmax temperatures.

        Parameters
        ----------
        pick, partner, bury, play : float | None
            If provided, sets the corresponding head's temperature τ. Values are
            clamped to a small positive minimum to avoid divide-by-zero.
        """
        eps = 1e-6
        if pick is not None:
            self.temperature_pick = max(float(pick), eps)
        if partner is not None:
            self.temperature_partner = max(float(partner), eps)
        if bury is not None:
            self.temperature_bury = max(float(bury), eps)
        if play is not None:
            self.temperature_play = max(float(play), eps)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features' and 'hand_tokens'
        action_mask : Bool Tensor (batch, action_size)
        hand_ids : Tensor (batch, 8)
        card_embedding : nn.Embedding

        Returns
        -------
        probs : Tensor (batch, action_size)
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_with_logits(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass that also returns pre-softmax logits.

        Returns
        -------
        probs : Tensor
        logits : Tensor
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out['features'],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out['hand_tokens'],
            action_mask=action_mask,
        )
        probs = F.softmax(logits, dim=-1)
        return probs, logits



class RecurrentCriticNetwork(nn.Module):
    """Critic head using encoder features directly."""

    def __init__(self, activation='swish'):
        super().__init__()

        self.critic_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU()
        )
        self.value_head = nn.Linear(256, 1)
        # Auxiliary heads
        self.win_head = nn.Linear(256, 1)
        self.return_head = nn.Linear(256, 1)
        self.secret_partner_head = nn.Linear(256, 1)
        # Per-player point prediction head (relative seating order, length-5 vector)
        self.points_head = nn.Linear(256, 5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_out: Dict[str, torch.Tensor]):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        value : Tensor (batch, 1)
        """
        feat = self.critic_adapter(encoder_out['features'])
        value = self.value_head(feat)
        return value

    def aux_predictions(self, encoder_out: Dict[str, torch.Tensor]):
        """Return auxiliary predictions as scalars and per-seat point estimates.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        win_prob : float
            Sigmoid of win logits
        expected_final_return : float
            Linear head output for expected final return
        secret_partner_prob : float
            Sigmoid of secret partner logits
        points_vector : list[float] | None
            Normalized per-seat point predictions in relative seat order.
        """
        with torch.no_grad():
            aux_feat = self.critic_adapter(encoder_out['features'])
            win_logit = self.win_head(aux_feat).squeeze(-1)
            expected_return = self.return_head(aux_feat).squeeze(-1)
            secret_logit = self.secret_partner_head(aux_feat).squeeze(-1)
            points_pred = torch.clamp(self.points_head(aux_feat), min=0.0, max=1.0)
        win_prob_t = torch.sigmoid(win_logit)
        win_prob = float(win_prob_t.item())
        expected_final = float(expected_return.item())
        secret_prob = float(torch.sigmoid(secret_logit).item())
        points_vector = [float(v) for v in points_pred[0].tolist()]
        return win_prob, expected_final, secret_prob, points_vector


class PPOAgent:
    def __init__(self, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
        # Dict-based encoder → fixed 256-d features
        self.state_size = 256
        self.action_size = action_size

        # --------------------------------------------------
        # Action groups for multi–head policy
        # --------------------------------------------------
        pick_indices = [ACTION_IDS["PICK"] - 1, ACTION_IDS["PASS"] - 1]

        # Partner-selection head: ALONE, JD PARTNER, all partner CALL actions
        partner_selection_actions = ["ALONE", "JD PARTNER"] + CALL_ACTIONS
        partner_indices = sorted({ACTION_IDS[a] - 1 for a in partner_selection_actions})

        # Bury head: explicit bury actions and UNDER actions
        bury_indices = sorted({ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + UNDER_ACTIONS)})

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            'pick': sorted(pick_indices),
            'partner': sorted(partner_indices),
            'bury': sorted(bury_indices),
            'play': sorted(play_indices),
        }

        # Explicit indices for PICK and PASS actions (0-indexed)
        self.pick_action_index = ACTION_IDS["PICK"] - 1
        self.pass_action_index = ACTION_IDS["PASS"] - 1

        # Encoder with memory
        self.encoder = CardReasoningEncoder(card_config=CardEmbeddingConfig()).to(device)

        # Per-player memory tracking
        self._player_memories = {}

        # Build mappings before constructing actor
        (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            call_action_global_indices,
            call_card_ids,
            play_under_action_index,
        ) = self._build_action_index_mappings()

        self.actor = MultiHeadRecurrentActorNetwork(
            action_size,
            self.action_groups,
            activation=activation,
            d_card=self.encoder.d_card_dim,
            d_token=self.encoder.d_token_dim,
            map_cid_to_play_action_index=map_cid_to_play_action_index,
            map_cid_to_bury_action_index=map_cid_to_bury_action_index,
            map_cid_to_under_action_index=map_cid_to_under_action_index,
            call_action_global_indices=call_action_global_indices,
            call_card_ids=call_card_ids,
            play_under_action_index=play_under_action_index,
        ).to(device)

        self.critic = RecurrentCriticNetwork(activation=activation).to(device)

        # Optimizers (include encoder params with scaled LR for card embeddings)
        encoder_groups = self.encoder.param_groups(base_lr=lr_actor, card_lr_scale=0.2)
        actor_groups = [
            {'params': self.actor.parameters(), 'lr': lr_actor},
            *encoder_groups,
        ]
        self.actor_optimizer = optim.Adam(actor_groups)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.95
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_partner = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.3
        self.clip_epsilon_pick = 0.20
        self.clip_epsilon_partner = 0.25
        self.clip_epsilon_bury = 0.15
        self.clip_epsilon_play = 0.20
        self.value_clip_epsilon = 0.20

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.0

        # Auxiliary loss coefficients
        self.win_loss_coeff = 0.1
        self.return_loss_coeff = 0.1
        self.secret_loss_coeff = 0.07
        self.points_loss_coeff = 0.1

        # Storage for trajectory data
        self.reset_storage()

        # --------------------------------------------------
        # Partner-call mixture: epsilon floor over CALL actions
        # --------------------------------------------------
        # Subset of partner head actions eligible for CALL-uniform mixing
        self.partner_call_subindices = sorted(
            {ACTION_IDS["JD PARTNER"] - 1} | {ACTION_IDS[a] - 1 for a in CALL_ACTIONS}
        )
        # Blending coefficient ε (0 disables; typical 0.02–0.10)
        self.partner_call_epsilon = 0.0

        # PASS-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pass_floor_epsilon = 0.0
        # PICK-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pick_floor_epsilon = 0.0

    def get_recurrent_memory(self, player_id: int | None, device: torch.device | None = None) -> torch.Tensor:
        """Return the recurrent memory vector for a player, or a zero vector if unset.

        The returned tensor is (256,) on the requested device.
        """
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mem = self._player_memories.get(player_id) if player_id is not None else None
        if mem is None:
            return torch.zeros(256, device=target_device)
        return mem.to(target_device) if mem.device != target_device else mem

    def set_recurrent_memory(self, player_id: int | None, memory_vector: torch.Tensor) -> None:
        """Set the recurrent memory vector for a player."""
        self._player_memories[player_id] = memory_vector

    def _build_action_index_mappings(self):
        """Precompute global action index mappings for card-specific actions."""
        map_cid_to_play_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_bury_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_under_action_index = torch.full((34,), -1, dtype=torch.long)

        # PLAY actions
        play_under_index = None
        for a in PLAY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            if a == f"PLAY {UNDER_TOKEN}":
                play_under_index = a_idx
                continue
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_play_action_index[cid] = a_idx

        # BURY actions
        for a in BURY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_bury_action_index[cid] = a_idx

        # UNDER actions (choose card from hand to set as under)
        for a in UNDER_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_under_action_index[cid] = a_idx

        # CALL actions
        idx_call_global_list = []
        call_card_ids_list = []
        for a in CALL_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            parts = a.split()
            # a is either "CALL X" or "CALL X UNDER"
            card = parts[1]
            cid = DECK_IDS[card]
            idx_call_global_list.append(a_idx)
            call_card_ids_list.append(cid)

        idx_call_global = torch.tensor(sorted(idx_call_global_list), dtype=torch.long)
        # Align cid list to the sorted order of indices
        # Build mapping from index to cid then reorder
        idx_to_cid = {idx: cid for idx, cid in zip(idx_call_global_list, call_card_ids_list)}
        call_card_ids = torch.tensor([idx_to_cid[int(x)] for x in idx_call_global.tolist()], dtype=torch.long)

        return map_cid_to_play_action_index, map_cid_to_bury_action_index, map_cid_to_under_action_index, idx_call_global, call_card_ids, play_under_index

    def set_head_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Convenience proxy to set per-head temperatures on the actor."""
        self.actor.set_temperatures(pick=pick, partner=partner, bury=bury, play=play)

    def set_partner_call_epsilon(self, eps: float):
        """Set ε for uniform CALL mixture on partner steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.partner_call_epsilon = eps

    def set_pass_floor_epsilon(self, eps: float):
        """Set ε for PASS-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pass_floor_epsilon = eps

    def set_pick_floor_epsilon(self, eps: float):
        """Set ε for PICK-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pick_floor_epsilon = eps

    def _apply_epsilon_mixing(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply partner-call ε mixture and PASS-floor ε mixture to probability
        distributions in a batched way without in-place ops.

        Parameters
        ----------
        probs : Tensor
            Shape (B, A) probabilities over actions
        mask : Bool Tensor
            Shape (B, A) of valid actions
        """
        mixed = probs
        if self.partner_call_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            call_vec = torch.zeros(A, dtype=torch.bool, device=mixed.device)
            call_idx = torch.tensor(self.partner_call_subindices, dtype=torch.long, device=mixed.device)
            call_vec[call_idx] = True
            valid_call = mask & call_vec.view(1, A).expand(B, A)
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(
                has,
                valid_call.float() / count.clamp_min(1.0),
                torch.zeros_like(mixed),
            )
            eps = torch.where(
                has,
                torch.full_like(count, self.partner_call_epsilon),
                torch.zeros_like(count),
            )
            mixed = (1.0 - eps) * mixed + eps * ucall

        if self.pick_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pick = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand(B, A)
            valid_pick = mask[:, self.pick_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * mixed + self.pick_floor_epsilon * one_hot_pick,
                mixed,
            )

        if self.pass_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pass = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand(B, A)
            valid_pass = mask[:, self.pass_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * mixed + self.pass_floor_epsilon * one_hot_pass,
                mixed,
            )

        return mixed

    def reset_storage(self):
        # Ordered list of events: each is a dict with keys:
        # kind: 'action' | 'obs'
        # state: np.array state vector
        # mask: torch.bool mask or convertible
        # If kind == 'action': action (0-indexed), reward, value, log_prob, done
        # GAE writes back: 'advantage' and 'return'
        self.events = []

    def get_action_mask(self, valid_actions, action_size):
        """Convert valid actions set to boolean mask"""
        mask = torch.zeros(action_size, dtype=torch.bool)
        for action in valid_actions:
            mask[action - 1] = True  # Convert from 1-indexed to 0-indexed
        return mask

    def reset_recurrent_state(self):
        """Clear memory states (call at the start of every new game)."""
        self._player_memories = {}

    def get_action_probs_with_logits(self, state, valid_actions, player_id=None):
        """Return post-mixture action probabilities and pre-mix logits for a single dict state.

        Applies partner CALL-uniform mixture if enabled. Keeps PPO on-policy by
        exposing the same transformed distribution that sampling uses.
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        # Encode dict state with memory
        encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

        # Store updated memory
        if player_id is not None:
            self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

        action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
        hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

        with torch.no_grad():
            probs, logits = self.actor.forward_with_logits(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )
            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                probs = self._apply_epsilon_mixing(probs, action_mask_t)

        return probs, logits

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        with torch.no_grad():
            # Get or init memory for this player
            memory_in = self.get_recurrent_memory(player_id, device=device)

            # Encode with memory
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

            # Get action probabilities
            action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)
            hand_ids_t = torch.as_tensor(state['hand_ids'], dtype=torch.long, device=device).view(1, -1)

            action_probs = self.actor(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )

            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                action_probs = self._apply_epsilon_mixing(action_probs, action_mask_t)

            # Get value
            value = self.critic(encoder_out)

        # Create distribution for consistent log probability calculation
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()

        # Get log probability from the distribution for consistency
        log_prob = dist.log_prob(action)

        return action.item() + 1, log_prob.item(), value.item()  # Convert back to 1-indexed

    # ------------------------------------------------------------------
    # Observation-only step (no action sampled / no transition stored)
    # ------------------------------------------------------------------
    def observe(self, state, player_id=None):
        """Propagate an observation through the encoder to update memory
        *without* sampling an action or storing any transition.  Useful for
        non-decision environment ticks such as the end of a trick.

        Parameters
        ----------
        state : dict
            Structured observation for the player.
        player_id : int | None
            Identifier to associate a persistent memory state (1-5 in game).
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        with torch.no_grad():
            # Encode with memory to update memory state
            encoder_out = self.encoder.encode_batch([state], memory_in=memory_in.unsqueeze(0), device=device)

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out['memory_out'][0])

    # ------------------------------------------------------------------
    # Episode-level ingestion API
    # ------------------------------------------------------------------
    def store_episode_events(self, events: list, player_id_default: int | None = None) -> None:
        """Store a single player's episode as a chronological list of events.

        Each event must be one of:
          - Observation:
              {
                'kind': 'observation',
                'state': dict,
                'player_id': int (optional if player_id_default provided)
              }
          - Action:
              {
                'kind': 'action',
                'state': dict,
                'action': int (1-indexed),
                'log_prob': float,
                'value': float,
                'valid_actions': set[int],
                'reward': float,
                'player_id': int (optional if player_id_default provided),
                'win_label': float (optional),
                'final_return_label': float (optional),
              }

        """
        # Identify last action index to set done flag
        last_action_idx = -1
        for idx, ev in enumerate(events):
            if ev.get('kind') == 'action':
                last_action_idx = idx

        for idx, ev in enumerate(events):
            pid = ev.get('player_id', player_id_default)
            if pid is None:
                pid = 0
            if ev.get('kind') == 'action':
                mask = self.get_action_mask(ev['valid_actions'], self.action_size)
                self.events.append({
                    'kind': 'action',
                    'state': ev['state'],
                    'mask': mask,
                    'action': int(ev['action']) - 1,
                    'reward': float(ev['reward']),
                    'value': float(ev['value']),
                    'log_prob': float(ev['log_prob']),
                    'done': (idx == last_action_idx),
                    'player_id': pid,
                    'win': float(ev.get('win_label', 0.0) or 0.0),
                    'final_return': float(ev.get('final_return_label', 0.0) or 0.0),
                    'secret_partner': float(ev.get('secret_partner_label', 0.0) or 0.0),
                    'points_rel': [float(x) for x in (ev.get('points_label') or [0.0] * 5)],
                })
            else:
                # Observation: mask = all ones by default
                mask = torch.ones(self.action_size, dtype=torch.bool)
                self.events.append({
                    'kind': 'observation',
                    'state': ev['state'],
                    'mask': mask,
                    'player_id': pid,
                })

    def compute_gae(self, next_value=0):
        """Compute GAE per player over action events; write results back into events."""
        # Group action indices by player
        actions_by_player: dict[int | None, list[int]] = {}
        for i, e in enumerate(self.events):
            if e['kind'] == 'action':
                pid = e.get('player_id', None)
                actions_by_player.setdefault(pid, []).append(i)

        all_advantages = []
        all_returns = []
        for pid, idxs in actions_by_player.items():
            if not idxs:
                continue
            rewards = np.array([self.events[i]['reward'] for i in idxs])
            # Bootstrap next value as 0.0 per player sequence
            values = np.array([self.events[i]['value'] for i in idxs] + [0.0])
            dones = np.array([self.events[i]['done'] for i in idxs] + [False])

            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values[:-1]

            for i, adv, ret in zip(idxs, advantages, returns):
                self.events[i]['advantage'] = float(adv)
                self.events[i]['return'] = float(ret)

            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_advantages:
            return np.array([]), np.array([])
        return np.concatenate(all_advantages), np.concatenate(all_returns)

    # ------------------------------------------------------------------
    # Internal helpers for PPO update
    # ------------------------------------------------------------------
    def _prepare_training_views(self):
        # Keep raw states (dicts) to encode inside the update for gradient flow
        states = [e['state'] for e in self.events]
        masks_t = [
            (e['mask'].to(device) if isinstance(e['mask'], torch.Tensor) else torch.as_tensor(e['mask'], dtype=torch.bool, device=device))
            for e in self.events
        ]
        kinds = [e['kind'] for e in self.events]
        pids = [e.get('player_id', None) for e in self.events]
        return states, masks_t, kinds, pids

    @staticmethod
    def _index_events_by_player(pids: list[int | None]) -> dict[int | None, list[int]]:
        events_by_player: dict[int | None, list[int]] = {}
        for idx, pid in enumerate(pids):
            events_by_player.setdefault(pid, []).append(idx)
        return events_by_player

    def _segments_from_events(self, events_by_player: dict[int | None, list[int]], kinds: list[str]):
        segments: list[tuple[int | None, int, int]] = []
        for pid, ev_idxs in events_by_player.items():
            action_ev_idxs = [i for i in ev_idxs if kinds[i] == 'action']
            if not action_ev_idxs:
                continue
            # Build done flags for this player's actions in order
            dones_pid = [self.events[i]['done'] for i in action_ev_idxs]
            start = ev_idxs[0]
            a_ptr = 0
            for i in ev_idxs:
                if kinds[i] == 'action':
                    if dones_pid[a_ptr]:
                        segments.append((pid, start, i))
                        start = i + 1
                    a_ptr += 1
            if start <= ev_idxs[-1]:
                segments.append((pid, start, ev_idxs[-1]))
        return segments

    @staticmethod
    def _pad_to_bt(lst, lengths, fill):
        T = max(lengths)
        out = []
        for i, tlen in enumerate(lengths):
            pad = T - tlen
            if pad > 0:
                pad_shape = list(lst[i].shape)
                pad_shape[0] = pad
                pad_tensor = torch.full(
                    pad_shape,
                    fill,
                    device=lst[i].device,
                    dtype=lst[i].dtype,
                )
                out.append(torch.cat([lst[i], pad_tensor], dim=0))
            else:
                out.append(lst[i])
        return torch.stack(out, dim=0), T

    def _build_minibatch_tensors(self, batch, states, masks_t, kinds, pids):
        lengths = []
        states_seqs = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        returns_list = []
        adv_list = []
        win_list_all = []
        final_ret_list_all = []
        secret_list_all = []
        points_list_all = []

        for pid, seg_start, seg_end in batch:
            # Restrict sequence strictly to this player's own events (actions + obs)
            # rather than a contiguous global slice. This ensures each agent learns
            # only from its direct experience without global context bleed-through.
            ev_range = [i for i in range(seg_start, seg_end + 1) if pids[i] == pid]
            lengths.append(len(ev_range))
            states_seqs.append([states[i] for i in ev_range])
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if (kinds[i] == 'action' and pids[i] == pid) else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, ret_bt, adv_bt = [], [], [], []
            win_bt, final_ret_bt, secret_bt = [], [], []
            points_bt = []
            for i in ev_range:
                if kinds[i] == 'action' and pids[i] == pid:
                    act_bt.append(torch.tensor(self.events[i]['action'], dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(self.events[i]['log_prob'], dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(self.events[i]['return'], dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(self.events[i]['advantage'], dtype=torch.float32, device=device))
                    win_lbl = self.events[i].get('win', 0.0) or 0.0
                    final_return_lbl = self.events[i].get('final_return', 0.0) or 0.0
                    secret_lbl = self.events[i].get('secret_partner', 0.0) or 0.0
                    win_bt.append(torch.tensor(float(win_lbl), dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(float(final_return_lbl), dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(float(secret_lbl), dtype=torch.float32, device=device))
                    pts_lbl = self.events[i].get('points_rel', [0.0] * 5)
                    points_bt.append(torch.tensor(pts_lbl, dtype=torch.float32, device=device))
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    win_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    secret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))
            win_list_all.append(torch.stack(win_bt, dim=0))
            final_ret_list_all.append(torch.stack(final_ret_bt, dim=0))
            secret_list_all.append(torch.stack(secret_bt, dim=0))
            points_list_all.append(torch.stack(points_bt, dim=0))

        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        win_bt, _ = self._pad_to_bt(win_list_all, lengths, 0.0)
        final_ret_bt, _ = self._pad_to_bt(final_ret_list_all, lengths, 0.0)
        secret_bt, _ = self._pad_to_bt(secret_list_all, lengths, 0.0)
        points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt, secret_bt, points_bt

    def _forward_vectorized(self, states_input, masks_bt, lengths_bt):
        """Vectorized forward pass for training with recurrent memory.

        Args:
            states_input: List of B sequences, each a list of observation dicts
            masks_bt: (B, T, action_size) action masks
            lengths_bt: (B,) sequence lengths

        Returns:
            logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
        """
        B = len(states_input)
        T = masks_bt.size(1)

        # Initialize memory to zeros for each segment
        memory_init = torch.zeros((B, 256), device=device)

        # Encode sequences with memory
        encoder_out_seq = self.encoder.encode_sequences(
            states_input,
            memory_in=memory_init,
            device=device
        )

        # Extract features (B, T, 256)
        features_bt = encoder_out_seq['features']

        # Build hand_ids_bt (B, T, N) from dict states
        N = 8  # Maximum hand size
        hand_ids_bt = torch.zeros((B, T, N), dtype=torch.long, device=device)
        for b, seq in enumerate(states_input):
            for t, s in enumerate(seq):
                if t >= T:
                    break
                arr = torch.as_tensor(s['hand_ids'], dtype=torch.long, device=device)
                if arr.dim() == 1:
                    arr = arr.view(-1)
                hand_ids_bt[b, t, :min(N, arr.numel())] = arr[:N]

        # Flatten time dimension to reuse single helper, then reshape back
        flat_feat = features_bt.reshape(B * T, -1)
        flat_hand = hand_ids_bt.reshape(B * T, N)
        flat_mask = masks_bt.view(B * T, -1)
        hand_tokens_bt = encoder_out_seq['hand_tokens']  # (B, T, N, d_token)
        flat_tokens = hand_tokens_bt.reshape(B * T, N, -1)

        logits_flat = self.actor._build_logits_from_features(
            actor_features=flat_feat,
            hand_ids=flat_hand,
            card_embedding=self.encoder.card,
            hand_tokens=flat_tokens,
            action_mask=flat_mask,
        )
        logits_bt = logits_flat.view(B, T, -1)

        # Critic values
        critic_feat_bt = self.critic.critic_adapter(features_bt)
        values_bt = self.critic.value_head(critic_feat_bt).squeeze(-1)

        # Auxiliary preds share critic_adapter features (gradients flow to encoder)
        aux_feat_bt = self.critic.critic_adapter(features_bt)
        win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
        ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
        secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
        points_pred_bt = self.critic.points_head(aux_feat_bt)

        return logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt

    @staticmethod
    def _flatten_action_steps(
        is_action_bt,
        logits_bt,
        values_bt,
        actions_bt,
        old_lp_bt,
        returns_bt,
        adv_bt,
        win_logits_bt,
        ret_pred_bt,
        win_bt,
        final_ret_bt,
        secret_logits_bt,
        secret_bt,
        masks_bt,
    ):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            logits_bt.view(-1, logits_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
            win_logits_bt.view(-1)[flat_mask],
            ret_pred_bt.view(-1)[flat_mask],
            win_bt.view(-1)[flat_mask],
            final_ret_bt.view(-1)[flat_mask],
            secret_logits_bt.view(-1)[flat_mask],
            secret_bt.view(-1)[flat_mask],
            masks_bt.view(-1, masks_bt.size(-1))[flat_mask],
        )

    @staticmethod
    def _entropy_from_probs(sub):
        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

    def _head_entropies(self, probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t):
        probs_pick = probs_flat[:, pick_idx_t]
        probs_partner = probs_flat[:, partner_idx_t]
        probs_bury = probs_flat[:, bury_idx_t]
        probs_play = probs_flat[:, play_idx_t]
        return (
            self._entropy_from_probs(probs_pick),
            self._entropy_from_probs(probs_partner),
            self._entropy_from_probs(probs_bury),
            self._entropy_from_probs(probs_play),
        )

    def _per_head_weights(self, actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, template):
        is_pick = torch.isin(actions_flat, pick_idx_t)
        is_partner = torch.isin(actions_flat, partner_idx_t)
        is_bury = torch.isin(actions_flat, bury_idx_t)
        is_play = ~(is_pick | is_partner | is_bury)

        count_pick = is_pick.sum().item()
        count_partner = is_partner.sum().item()
        count_bury = is_bury.sum().item()
        count_play = is_play.sum().item()
        total_count = float(count_pick + count_partner + count_bury + count_play)
        heads_present = int((count_pick > 0) + (count_partner > 0) + (count_bury > 0) + (count_play > 0))

        def per_head_weight(count: int) -> float:
            if heads_present == 0 or count == 0:
                return 0.0
            return total_count / (heads_present * float(count))

        w_pick = per_head_weight(count_pick)
        w_partner = per_head_weight(count_partner)
        w_bury = per_head_weight(count_bury)
        w_play = per_head_weight(count_play)

        head_weight = torch.ones_like(template)
        if w_pick > 0.0:
            head_weight[is_pick] = head_weight[is_pick] * w_pick
        else:
            head_weight[is_pick] = 0.0
        if w_partner > 0.0:
            head_weight[is_partner] = head_weight[is_partner] * w_partner
        else:
            head_weight[is_partner] = 0.0
        if w_bury > 0.0:
            head_weight[is_bury] = head_weight[is_bury] * w_bury
        else:
            head_weight[is_bury] = 0.0
        if w_play > 0.0:
            head_weight[is_play] = head_weight[is_play] * w_play
        else:
            head_weight[is_play] = 0.0

        return head_weight, is_pick, is_partner, is_bury

    def _actor_critic_losses(
        self,
        logits_flat,
        mask_flat,
        actions_flat,
        old_lp_flat,
        values_flat,
        returns_flat,
        adv_flat,
        pick_idx_t,
        partner_idx_t,
        bury_idx_t,
        play_idx_t,
    ):
        # Build probabilities fresh from logits to avoid in-place softmax conflicts
        probs_all = F.softmax(logits_flat, dim=-1)
        # Partner-call mixture on-the-fly
        if self.partner_call_epsilon > 0.0:
            A = probs_all.size(-1)
            call_mask = torch.zeros(A, dtype=torch.bool, device=probs_all.device)
            call_mask[self.partner_call_subindices] = True
            valid_call = call_mask.view(1, A).expand_as(mask_flat) & mask_flat
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(has, valid_call.float() / count.clamp_min(1.0), torch.zeros_like(probs_all))
            eps = torch.where(has, torch.full_like(count, self.partner_call_epsilon), torch.zeros_like(count))
            probs_all = (1.0 - eps) * probs_all + eps * ucall

        # PICK-floor mixing on-the-fly
        if self.pick_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pick = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand_as(probs_all)
            valid_pick = mask_flat[:, self.pick_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * probs_all + self.pick_floor_epsilon * one_hot_pick,
                probs_all,
            )

        # PASS-floor mixing on-the-fly
        if self.pass_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pass = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand_as(probs_all)
            valid_pass = mask_flat[:, self.pass_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * probs_all + self.pass_floor_epsilon * one_hot_pass,
                probs_all,
            )

        dist = torch.distributions.Categorical(probs_all.clamp(min=1e-12))
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = self._head_entropies(
            probs_all, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
        )
        entropy_term = (
            self.entropy_coeff_pick * pick_entropy
            + self.entropy_coeff_partner * partner_entropy
            + self.entropy_coeff_bury * bury_entropy
            + self.entropy_coeff_play * play_entropy
        )

        ratios = torch.exp(new_lp_flat - old_lp_flat)
        head_weight, is_pick, is_partner, is_bury = self._per_head_weights(
            actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, ratios
        )

        eps_flat = torch.full_like(ratios, self.clip_epsilon_play)
        eps_flat[is_pick] = self.clip_epsilon_pick
        eps_flat[is_partner] = self.clip_epsilon_partner
        eps_flat[is_bury] = self.clip_epsilon_bury

        surr1 = ratios * adv_flat
        clipped = torch.clamp(ratios, 1 - eps_flat, 1 + eps_flat) * adv_flat
        pg_loss_elements = -torch.min(surr1, clipped)
        policy_loss = (pg_loss_elements * head_weight).mean()

        returns_target = returns_flat.view(-1)
        values_old = values_flat.detach()
        v_clipped = values_old + torch.clamp(
            values_flat - values_old, -self.value_clip_epsilon, self.value_clip_epsilon
        )
        critic_loss_unclipped = F.mse_loss(values_flat, returns_target)
        critic_loss_clipped = F.mse_loss(v_clipped, returns_target)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

        actor_loss = policy_loss - entropy_term + self.kl_coef * approx_kl_t
        return actor_loss, critic_loss, approx_kl_t, (pick_entropy, partner_entropy, bury_entropy, play_entropy)

    def update(self, epochs=6, batch_size=256):
        """Update actor and critic networks using PPO with recurrent unrolling.
        Includes performance optimisations and per-update timing logs.
        """
        t_update_start = time.time()
        if len(self.events) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Store statistics before normalization
        raw_advantages = advantages.copy() if advantages.size else np.array([0.0])
        advantage_stats = {
            'mean': float(np.mean(raw_advantages)),
            'std': float(np.std(raw_advantages)),
            'min': float(np.min(raw_advantages)),
            'max': float(np.max(raw_advantages))
        }

        value_target_stats = {
            'mean': float(np.mean(returns)) if advantages.size else 0.0,
            'std': float(np.std(returns)) if advantages.size else 0.0,
            'min': float(np.min(returns)) if advantages.size else 0.0,
            'max': float(np.max(returns)) if advantages.size else 0.0,
        }

        # Normalize advantages and write back into events so the loss uses normalized values
        if advantages.size:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for e in self.events:
                if e.get('kind') == 'action':
                    e['advantage'] = float((e['advantage'] - adv_mean) / adv_std)

        # Build static views and segments
        t_build_start = time.time()
        states, masks_t, kinds, pids = self._prepare_training_views()
        events_by_player = self._index_events_by_player(pids)
        segments = self._segments_from_events(events_by_player, kinds)

        # Precompute static index tensors once
        pick_idx_tensor_static = torch.tensor(self.action_groups['pick'], device=device)
        partner_idx_tensor_static = torch.tensor(self.action_groups['partner'], device=device)
        bury_idx_tensor_static = torch.tensor(self.action_groups['bury'], device=device)
        play_idx_tensor_static = torch.tensor(self.action_groups['play'], device=device)

        t_build_end = time.time()

        # Training epochs – vectorized by batching segments
        forward_time = 0.0
        backward_time = 0.0
        step_time = 0.0
        optimizer_steps = 0
        early_stop_triggered = False
        last_approx_kl = 0.0

        # Instrumentation accumulators
        ent_pick_sum = 0.0
        ent_partner_sum = 0.0
        ent_bury_sum = 0.0
        ent_play_sum = 0.0
        ent_batches = 0
        pick_adv_sum = 0.0
        pick_adv_count = 0
        pass_adv_sum = 0.0
        pass_adv_count = 0

        for _ in range(epochs):
            if not segments:
                break
            perm = torch.randperm(len(segments))
            for mb_start in range(0, len(segments), batch_size):
                batch_idxs = perm[mb_start:mb_start + batch_size].tolist()
                batch = [segments[i] for i in batch_idxs]
                if len(batch) == 0:
                    continue

                (
                    states_bt,
                    masks_bt,
                    is_action_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    lengths_bt,
                    win_bt,
                    final_ret_bt,
                    secret_bt,
                    points_bt,
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds, pids)

                # Vectorized forward
                t_fwd = time.time()
                logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt,
                    logits_bt,
                    values_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    win_logits_bt,
                    ret_pred_bt,
                    win_bt,
                    final_ret_bt,
                    secret_logits_bt,
                    secret_bt,
                    masks_bt,
                )
                if flat is None:
                    continue
                (
                    logits_flat,
                    values_flat,
                    actions_flat,
                    old_lp_flat,
                    returns_flat,
                    adv_flat,
                    win_logits_flat,
                    ret_pred_flat,
                    win_labels_flat,
                    final_ret_labels_flat,
                    secret_logits_flat,
                    secret_labels_flat,
                    mask_flat,
                ) = flat

                # Record PICK/PASS advantages across minibatches
                with torch.no_grad():
                    pick_mask_specific = (actions_flat == self.pick_action_index)
                    if pick_mask_specific.any():
                        pick_adv_sum += adv_flat[pick_mask_specific].sum().item()
                        pick_adv_count += int(pick_mask_specific.sum().item())
                    pass_mask_specific = (actions_flat == self.pass_action_index)
                    if pass_mask_specific.any():
                        pass_adv_sum += adv_flat[pass_mask_specific].sum().item()
                        pass_adv_count += int(pass_mask_specific.sum().item())

                # Losses and metrics
                (
                    actor_loss,
                    critic_loss,
                    approx_kl_t,
                    (pick_entropy, partner_entropy, bury_entropy, play_entropy),
                ) = self._actor_critic_losses(
                    logits_flat,
                    mask_flat,
                    actions_flat,
                    old_lp_flat,
                    values_flat,
                    returns_flat,
                    adv_flat,
                    pick_idx_tensor_static,
                    partner_idx_tensor_static,
                    bury_idx_tensor_static,
                    play_idx_tensor_static,
                )
                last_approx_kl = float(approx_kl_t.item())

                # Entropy accumulation
                ent_pick_sum += pick_entropy.detach().item()
                ent_partner_sum += partner_entropy.detach().item()
                ent_bury_sum += bury_entropy.detach().item()
                ent_play_sum += play_entropy.detach().item()
                ent_batches += 1

                # Auxiliary losses
                bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
                return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)
                secret_loss = F.binary_cross_entropy_with_logits(secret_logits_flat, secret_labels_flat)
                # Per-player points auxiliary loss (regression on normalized per-seat totals)
                # points_pred_flat and labels are (N, 5); smooth L1 stabilizes training.
                points_pred_flat = points_pred_bt.view(-1, points_pred_bt.size(-1))[is_action_bt.view(-1)]
                points_labels_flat = points_bt.view(-1, points_bt.size(-1))[is_action_bt.view(-1)]
                points_loss = F.smooth_l1_loss(points_pred_flat, points_labels_flat)

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = (
                    actor_loss
                    + self.value_loss_coeff * critic_loss
                    + self.win_loss_coeff * bce_loss
                    + self.return_loss_coeff * return_loss
                    + self.secret_loss_coeff * secret_loss
                    + self.points_loss_coeff * points_loss
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                step_time += time.time() - t_step
                optimizer_steps += 1

                # Early stop further updates in this epoch if KL exceeds threshold
                if self.target_kl is not None and last_approx_kl > self.target_kl:
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        transitions = sum(1 for e in self.events if e['kind'] == 'action')

        # Clear storage
        self.reset_storage()

        # Return training statistics
        t_end = time.time()
        timing = {
            'build_s': t_build_end - t_build_start,
            'forward_s': forward_time,
            'backward_s': backward_time,
            'step_s': step_time,
            'total_update_s': t_end - t_update_start,
            'optimizer_steps': optimizer_steps,
        }
        return {
            'advantage_stats': advantage_stats,
            'value_target_stats': value_target_stats,
            'num_transitions': transitions,
            'approx_kl': last_approx_kl,
            'early_stop': early_stop_triggered,
            'timing': timing,
            'head_entropy': {
                'pick': (ent_pick_sum / ent_batches) if ent_batches > 0 else 0.0,
                'partner': (ent_partner_sum / ent_batches) if ent_batches > 0 else 0.0,
                'bury': (ent_bury_sum / ent_batches) if ent_batches > 0 else 0.0,
                'play': (ent_play_sum / ent_batches) if ent_batches > 0 else 0.0,
            },
            'pick_pass_adv': {
                'pick_mean': (pick_adv_sum / pick_adv_count) if pick_adv_count > 0 else 0.0,
                'pick_count': pick_adv_count,
                'pass_mean': (pass_adv_sum / pass_adv_count) if pass_adv_count > 0 else 0.0,
                'pass_count': pass_adv_count,
            },
        }

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath, load_optimizers: bool = True):
        """Load model parameters.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizers : bool
            If True, also loads optimizer states. For population/inference agents,
            pass False to skip optimizer loading entirely.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Core networks: accept missing/new critic aux heads for backward compatibility
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)

        # Optimizers: best-effort restore; fall back to fresh states if shapes changed
        if load_optimizers:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except ValueError as e:
                print(f"Warning: could not load optimizer state from {filepath}: {e}")

        # Reset memory states after loading
        self._player_memories = {}

```

Removed the defensive guard so `aux_predictions` now always converts `points_pred[0]` into a Python list; any unexpected tensor shape or empty batch will now raise immediately, helping surface bugs in inference or checkpoint loading faster. Lint is clean—no other files needed changes.

