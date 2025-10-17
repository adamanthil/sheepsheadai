#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) Population Management System for Sheepshead.

This module implements a population-based training system inspired by AlphaStar and OpenAI Five,
using OpenSkill ratings for opponent selection and maintaining diverse agent populations.
"""

import argparse
import logging
import glob
import numpy as np
import copy
import random
import os
import json
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from openskill.models import PlackettLuce
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from ppo import PPOAgent
from sheepshead import (
    Game,
    PARTNER_BY_JD,
    PARTNER_BY_CALLED_ACE,
    TRUMP,
    ACTION_LOOKUP,
    UNDER_TOKEN,
    get_card_points,
    get_partner_mode_name,
    filter_by_suit,
    ACTIONS,
)
from training_utils import estimate_hand_strength_category

# EWMA smoothing factors
EMA_ALPHA_TRICK = 0.001   # frequent events (per trick)
EMA_ALPHA_PLAY = 0.001    # per-play events
EMA_ALPHA_RARE = 0.01     # rare events (e.g., bury)


def standard_role_for_lead(player) -> str:
    """Map a player's current status to the role bucket used by strategic lead counters.

    Partner seats (including secret partners) are treated as 'picker' for lead counters
    to merge picker+partner leads into a single bucket.
    """
    try:
        is_partner = bool(getattr(player, 'is_partner', False)) or bool(getattr(player, 'is_secret_partner', False))
    except Exception:
        is_partner = False
    return 'picker' if (getattr(player, 'is_picker', False) or is_partner) else 'defender'


def detailed_role_for_metrics(player) -> str:
    """Return detailed role label among {'picker','partner','defender'} for metrics."""
    if getattr(player, 'is_picker', False):
        return 'picker'
    if bool(getattr(player, 'is_partner', False)) or bool(getattr(player, 'is_secret_partner', False)):
        return 'partner'
    return 'defender'


def profile_pop_agent_action(game: Game,
                             player,
                             action: int,
                             pop_agent: 'PopulationAgent',
                             hand_strength_by_pos: dict[int, str]) -> None:
    """Apply strategic profiling updates for a population agent based on an action.

    Centralizes pick/pass, bury, alone/call, lead/void profiling to keep training and
    evaluation loops consistent.
    """
    action_name = ACTION_LOOKUP[action]

    # Pick / Pass profiling
    if action_name in ("PICK", "PASS"):
        pop_agent.update_strategic_profile_from_game({
            'pick_decision': (action_name == "PICK"),
            'hand_strength_category': hand_strength_by_pos.get(player.position, 'medium'),
            'position': player.position,
        })

    # Bury profiling
    if action_name.startswith("BURY "):
        card = action_name.split()[-1]
        pop_agent.update_strategic_profile_from_game({
            'bury_decision': card,
            'card_points': get_card_points(card),
        })

    # ALONE vs partner call profiling
    if action_name == "ALONE":
        pop_agent.update_strategic_profile_from_game({'alone_call': True})
    elif action_name.startswith("CALL "):
        pop_agent.update_strategic_profile_from_game({'alone_call': False, 'called_card': action_name.split()[1]})

    # Lead/void profiling
    if action_name.startswith("PLAY "):
        card = action_name.split()[-1]
        is_lead = (game.cards_played == 0)
        is_early = is_lead and (game.current_trick <= 1)
        if is_lead:
            pop_agent.update_strategic_profile_from_game({
                'trump_lead': (card in TRUMP),
                'role': standard_role_for_lead(player),
                'is_early_lead': is_early,
            })

        # Void logic: only when following and can't follow suit
        if (not is_lead) and game.current_suit:
            can_follow = len(filter_by_suit(player.hand, game.current_suit)) > 0
            if not can_follow:
                played_trump = (card in TRUMP or card == UNDER_TOKEN)
                # Schmear defined as throwing A/K/10 off-suit when void
                pts = get_card_points(card)
                is_schmear_card = (not played_trump) and (pts >= 4)
                pop_agent.update_strategic_profile_from_game({
                    'void_event': True,
                    'void_played_trump': played_trump,
                    'schmeared_points': 0 if played_trump else get_card_points(card),
                    'void_schmear': is_schmear_card,
                })

        # Early-trump play rate (first two tricks, any seat, any position)
        is_early_trick_play = game.current_trick <= 1
        if is_early_trick_play:
            pop_agent.update_strategic_profile_from_game({
                'early_trump_play': bool(card in TRUMP)
            })

def profile_trick_completion(game: Game, pos_to_agent: dict[int, 'PopulationAgent']) -> None:
    """Update trick-level EWMAs for population agents after a trick completes.

    - trick_win_rate_by_role: each agent gets a sample 1/0 based on win for their role
    - lead_win_rate_by_role: only leader gets a 1/0 sample for whether lead won
    """
    if game.current_trick <= 0:
        return
    last_idx = game.current_trick - 1
    # Guard index bounds
    if last_idx < 0:
        return

    winner_raw = game.trick_winners[last_idx]
    winner_pos = int(winner_raw)

    leader_raw = game.leaders[last_idx]
    leader_pos = int(leader_raw)

    # Skip leaster since roles differ
    if getattr(game, 'is_leaster', False):
        return

    # Trick win rate by role (update all seats present in mapping)
    for seat in game.players:
        pop_agent = pos_to_agent.get(seat.position)
        if not pop_agent:
            continue
        role = detailed_role_for_metrics(seat)
        sample = 1.0 if seat.position == winner_pos else 0.0
        if role == 'picker':
            pop_agent.strategic_profile.trick_win_rate_picker = (
                (1.0 - EMA_ALPHA_TRICK) * pop_agent.strategic_profile.trick_win_rate_picker + EMA_ALPHA_TRICK * sample
            )
        elif role == 'partner':
            pop_agent.strategic_profile.trick_win_rate_partner = (
                (1.0 - EMA_ALPHA_TRICK) * pop_agent.strategic_profile.trick_win_rate_partner + EMA_ALPHA_TRICK * sample
            )
        else:
            pop_agent.strategic_profile.trick_win_rate_defender = (
                (1.0 - EMA_ALPHA_TRICK) * pop_agent.strategic_profile.trick_win_rate_defender + EMA_ALPHA_TRICK * sample
            )

    # Lead win rate by role (only for leader if present in mapping)
    if leader_pos is not None:
        leader_agent = pos_to_agent.get(leader_pos)
        leader_seat = None
        for seat in game.players:
            if seat.position == leader_pos:
                leader_seat = seat
                break
        if leader_agent and leader_seat:
            role = detailed_role_for_metrics(leader_seat)
            sample = 1.0 if leader_pos == winner_pos else 0.0
            if role == 'picker':
                leader_agent.strategic_profile.lead_win_rate_picker = (
                    (1.0 - EMA_ALPHA_TRICK) * leader_agent.strategic_profile.lead_win_rate_picker + EMA_ALPHA_TRICK * sample
                )
            elif role == 'partner':
                leader_agent.strategic_profile.lead_win_rate_partner = (
                    (1.0 - EMA_ALPHA_TRICK) * leader_agent.strategic_profile.lead_win_rate_partner + EMA_ALPHA_TRICK * sample
                )
            else:
                leader_agent.strategic_profile.lead_win_rate_defender = (
                    (1.0 - EMA_ALPHA_TRICK) * leader_agent.strategic_profile.lead_win_rate_defender + EMA_ALPHA_TRICK * sample
                )

@dataclass
class StrategicProfile:
    """Tracks behavioral patterns and strategic tendencies of an agent."""

    # Pick decision patterns (counters)
    pick_true_by_hand_strength: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pick_total_by_hand_strength: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


    # Lead-trump counters (overall and early-trick)
    lead_counts_picker: int = 0
    lead_trump_counts_picker: int = 0
    lead_counts_defender: int = 0
    lead_trump_counts_defender: int = 0
    early_lead_counts_picker: int = 0
    early_lead_trump_counts_picker: int = 0
    early_lead_counts_defender: int = 0
    early_lead_trump_counts_defender: int = 0

    # Void behavior counters (unable to follow suit)
    void_events: int = 0
    void_trump_events: int = 0
    void_fail_events: int = 0
    schmeared_points_sum: int = 0

    # Bury decision aggregates
    bury_points_sum: int = 0
    bury_count: int = 0

    alone_true_count: int = 0
    alone_total_count: int = 0

    # Performance aggregates in different scenarios
    performance_sum_picker: float = 0.0
    performance_count_picker: int = 0
    performance_sum_partner: float = 0.0
    performance_count_partner: int = 0
    performance_sum_defender: float = 0.0
    performance_count_defender: int = 0
    performance_sum_leaster: float = 0.0
    performance_count_leaster: int = 0

    # --- EWMA strategic rates ---
    # Trick win rate by role
    trick_win_rate_picker: float = 0.5
    trick_win_rate_partner: float = 0.5
    trick_win_rate_defender: float = 0.5
    # Lead win rate by role
    lead_win_rate_picker: float = 0.5
    lead_win_rate_partner: float = 0.5
    lead_win_rate_defender: float = 0.5
    # Early trump play rate (first two tricks)
    early_trump_play_rate: float = 0.5
    # Bury high points rate (>=10)
    bury_high_points_rate: float = 0.5
    # Schmear rate when void (only A,K,10 off-suit)
    schmear_rate_when_void: float = 0.5
    # Pick-hand correlation (persisted EWMA of external measurement)
    pick_hand_correlation: float = 0.0
    # Internal EWMAs for online pick-hand correlation (X: 0,1,2; Y: 0/1)
    _pick_corr_ewma_x: float = 0.0
    _pick_corr_ewma_y: float = 0.0
    _pick_corr_ewma_x2: float = 0.0
    _pick_corr_ewma_y2: float = 0.0
    _pick_corr_ewma_xy: float = 0.0

    def get_pick_rate_by_hand_strength(self, strength_category: str) -> float:
        """Get pick rate for a specific hand strength category."""
        picked = self.pick_true_by_hand_strength.get(strength_category, 0)
        total = self.pick_total_by_hand_strength.get(strength_category, 0)
        if total <= 0:
            return 0.5
        return picked / max(1, total)

    def get_trump_lead_rate(self, role: str) -> float:
        """Get trump leading rate for picker or defender role."""
        if role == 'picker':
            total = self.lead_counts_picker
            trump = self.lead_trump_counts_picker
        elif role == 'defender':
            total = self.lead_counts_defender
            trump = self.lead_trump_counts_defender
        else:
            return 0.0

        return (trump / max(1, total)) if total > 0 else 0.0

    def get_early_trump_lead_rate(self, role: str) -> float:
        if role == 'picker':
            total = self.early_lead_counts_picker
            trump = self.early_lead_trump_counts_picker
        elif role == 'defender':
            total = self.early_lead_counts_defender
            trump = self.early_lead_trump_counts_defender
        else:
            return 0.0
        return (trump / max(1, total)) if total > 0 else 0.0

    def get_bury_quality_score(self) -> float:
        """Calculate bury decision quality (lower point cards are better)."""
        if self.bury_count <= 0:
            return 0.5
        avg_points = self.bury_points_sum / max(1, self.bury_count)
        return max(0.0, 1.0 - (avg_points / 11.0))

    def get_void_trump_rate(self) -> float:
        return (self.void_trump_events / self.void_events) if self.void_events > 0 else 0.0

    def get_avg_schmeared_points(self) -> float:
        return (self.schmeared_points_sum / self.void_fail_events) if self.void_fail_events > 0 else 0.0

    def get_risk_tolerance(self) -> float:
        """Risk proxy based on ALONE usage only (0 = conservative, 1 = aggressive)."""
        if self.alone_total_count <= 0:
            return 0.5
        return self.alone_true_count / max(1, self.alone_total_count)

    def get_role_performance(self, role: str) -> float:
        """Get average performance in a specific role."""
        if role == 'picker':
            return (self.performance_sum_picker / max(1, self.performance_count_picker)) if self.performance_count_picker > 0 else 0.0
        elif role == 'partner':
            return (self.performance_sum_partner / max(1, self.performance_count_partner)) if self.performance_count_partner > 0 else 0.0
        elif role == 'defender':
            return (self.performance_sum_defender / max(1, self.performance_count_defender)) if self.performance_count_defender > 0 else 0.0
        elif role == 'leaster':
            return (self.performance_sum_leaster / max(1, self.performance_count_leaster)) if self.performance_count_leaster > 0 else 0.0
        else:
            return 0.0

    def get_alone_rate(self) -> float:
        return (self.alone_true_count / max(1, self.alone_total_count)) if self.alone_total_count > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pick_true_by_hand_strength': dict(self.pick_true_by_hand_strength),
            'pick_total_by_hand_strength': dict(self.pick_total_by_hand_strength),
            'lead_counts_picker': self.lead_counts_picker,
            'lead_trump_counts_picker': self.lead_trump_counts_picker,
            'lead_counts_defender': self.lead_counts_defender,
            'lead_trump_counts_defender': self.lead_trump_counts_defender,
            'early_lead_counts_picker': self.early_lead_counts_picker,
            'early_lead_trump_counts_picker': self.early_lead_trump_counts_picker,
            'early_lead_counts_defender': self.early_lead_counts_defender,
            'early_lead_trump_counts_defender': self.early_lead_trump_counts_defender,
            'void_events': self.void_events,
            'void_trump_events': self.void_trump_events,
            'void_fail_events': self.void_fail_events,
            'schmeared_points_sum': self.schmeared_points_sum,
            'bury_points_sum': self.bury_points_sum,
            'bury_count': self.bury_count,
            'alone_true_count': self.alone_true_count,
            'alone_total_count': self.alone_total_count,
            'performance_sum_picker': self.performance_sum_picker,
            'performance_count_picker': self.performance_count_picker,
            'performance_sum_partner': self.performance_sum_partner,
            'performance_count_partner': self.performance_count_partner,
            'performance_sum_defender': self.performance_sum_defender,
            'performance_count_defender': self.performance_count_defender,
            'performance_sum_leaster': self.performance_sum_leaster,
            'performance_count_leaster': self.performance_count_leaster,
            'trick_win_rate_picker': self.trick_win_rate_picker,
            'trick_win_rate_partner': self.trick_win_rate_partner,
            'trick_win_rate_defender': self.trick_win_rate_defender,
            'lead_win_rate_picker': self.lead_win_rate_picker,
            'lead_win_rate_partner': self.lead_win_rate_partner,
            'lead_win_rate_defender': self.lead_win_rate_defender,
            'early_trump_play_rate': self.early_trump_play_rate,
            'bury_high_points_rate': self.bury_high_points_rate,
            'schmear_rate_when_void': self.schmear_rate_when_void,
            'pick_hand_correlation': self.pick_hand_correlation,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategicProfile':
        """Create from dictionary."""
        profile = cls()

        # Pick counters
        profile.pick_true_by_hand_strength = defaultdict(int, data.get('pick_true_by_hand_strength', {}))
        profile.pick_total_by_hand_strength = defaultdict(int, data.get('pick_total_by_hand_strength', {}))

        # Lead/void counters
        profile.lead_counts_picker = data.get('lead_counts_picker', 0)
        profile.lead_trump_counts_picker = data.get('lead_trump_counts_picker', 0)
        profile.lead_counts_defender = data.get('lead_counts_defender', 0)
        profile.lead_trump_counts_defender = data.get('lead_trump_counts_defender', 0)
        profile.early_lead_counts_picker = data.get('early_lead_counts_picker', 0)
        profile.early_lead_trump_counts_picker = data.get('early_lead_trump_counts_picker', 0)
        profile.early_lead_counts_defender = data.get('early_lead_counts_defender', 0)
        profile.early_lead_trump_counts_defender = data.get('early_lead_trump_counts_defender', 0)
        profile.void_events = data.get('void_events', 0)
        profile.void_trump_events = data.get('void_trump_events', 0)
        profile.void_fail_events = data.get('void_fail_events', 0)
        profile.schmeared_points_sum = data.get('schmeared_points_sum', 0)

        # Bury aggregates
        profile.bury_points_sum = data.get('bury_points_sum', 0)
        profile.bury_count = data.get('bury_count', 0)

        # Risk/call aggregates
        profile.alone_true_count = data.get('alone_true_count', 0)
        profile.alone_total_count = data.get('alone_total_count', 0)

        # Performance aggregates
        profile.performance_sum_picker = float(data.get('performance_sum_picker', 0.0))
        profile.performance_count_picker = int(data.get('performance_count_picker', 0))
        profile.performance_sum_partner = float(data.get('performance_sum_partner', 0.0))
        profile.performance_count_partner = int(data.get('performance_count_partner', 0))
        profile.performance_sum_defender = float(data.get('performance_sum_defender', 0.0))
        profile.performance_count_defender = int(data.get('performance_count_defender', 0))
        profile.performance_sum_leaster = float(data.get('performance_sum_leaster', 0.0))
        profile.performance_count_leaster = int(data.get('performance_count_leaster', 0))

        # EWMA strategic rates
        profile.trick_win_rate_picker = float(data.get('trick_win_rate_picker', 0.5))
        profile.trick_win_rate_partner = float(data.get('trick_win_rate_partner', 0.5))
        profile.trick_win_rate_defender = float(data.get('trick_win_rate_defender', 0.5))
        profile.lead_win_rate_picker = float(data.get('lead_win_rate_picker', 0.5))
        profile.lead_win_rate_partner = float(data.get('lead_win_rate_partner', 0.5))
        profile.lead_win_rate_defender = float(data.get('lead_win_rate_defender', 0.5))
        profile.early_trump_play_rate = float(data.get('early_trump_play_rate', 0.5))
        profile.bury_high_points_rate = float(data.get('bury_high_points_rate', 0.5))
        profile.schmear_rate_when_void = float(data.get('schmear_rate_when_void', 0.5))
        profile.pick_hand_correlation = float(data.get('pick_hand_correlation', 0.0))

        return profile


@dataclass
class AgentMetadata:
    """Metadata for population agents."""
    agent_id: str
    creation_time: float
    parent_id: Optional[str]
    training_episodes: int
    partner_mode: int  # PARTNER_BY_JD or PARTNER_BY_CALLED_ACE
    activation: str

    # Performance metrics
    games_played: int = 0
    total_score: float = 0.0
    picker_games: int = 0
    picker_score: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class PopulationAgent:
    """Wrapper for agents in the population with ratings and metadata."""

    def __init__(self, agent: PPOAgent, metadata: AgentMetadata, rating=None, strategic_profile=None):
        self.agent = agent
        self.metadata = metadata
        # Default OpenSkill rating using PlackettLuce model
        if rating is None:
            model = PlackettLuce()
            self.rating = model.rating()
        else:
            self.rating = rating

        # Strategic behavior tracking
        self.strategic_profile = strategic_profile or StrategicProfile()

        # Track recent performance for diversity assessment
        self.recent_games = deque(maxlen=100)

        # Decayed win-rate vs training agent for exploitation sampling
        # Exponentially-smoothed estimate in [0,1], initialised to neutral 0.5
        self.exploitation_win_rate_ema: float = 0.5
        self.exploitation_samples: int = 0

    def update_rating(self, new_rating):
        """Update the agent's rating."""
        self.rating = new_rating

    def add_game_result(self, score: float, was_picker: bool = False):
        """Add a game result for tracking performance."""
        self.metadata.games_played += 1
        self.metadata.total_score += score
        self.recent_games.append(score)

        if was_picker:
            self.metadata.picker_games += 1
            self.metadata.picker_score += score

    def get_average_score(self) -> float:
        """Get average score across all games."""
        if self.metadata.games_played == 0:
            return 0.0
        return self.metadata.total_score / self.metadata.games_played

    def get_picker_average(self) -> float:
        """Get average score when picking."""
        if self.metadata.picker_games == 0:
            return 0.0
        return self.metadata.picker_score / self.metadata.picker_games

    def get_skill_estimate(self) -> float:
        """Get current skill estimate (rating μ)."""
        return self.rating.mu

    def get_skill_uncertainty(self) -> float:
        """Get skill uncertainty (rating σ)."""
        return self.rating.sigma

    def calculate_strategic_diversity(self, other: 'PopulationAgent') -> float:
        """Calculate strategic diversity score compared to another agent (0-1, higher = more diverse)."""
        if not hasattr(other, 'strategic_profile'):
            return 0.5  # Default moderate diversity

        diversity_components = []

        # Pick behavior diversity (across different hand strengths)
        pick_diversities = []
        hand_strengths = ['weak', 'medium', 'strong']
        for strength in hand_strengths:
            rate1 = self.strategic_profile.get_pick_rate_by_hand_strength(strength)
            rate2 = other.strategic_profile.get_pick_rate_by_hand_strength(strength)
            pick_diversities.append(abs(rate1 - rate2))

        if pick_diversities:
            diversity_components.append(np.mean(pick_diversities))

        # Lead-trump behavior diversity (overall and early-trick)
        lead_divs = []
        for role in ['picker', 'defender']:
            rate1 = self.strategic_profile.get_trump_lead_rate(role)
            rate2 = other.strategic_profile.get_trump_lead_rate(role)
            early1 = self.strategic_profile.get_early_trump_lead_rate(role)
            early2 = other.strategic_profile.get_early_trump_lead_rate(role)
            lead_divs.extend([abs(rate1 - rate2), abs(early1 - early2)])
        diversity_components.append(np.mean(lead_divs) if lead_divs else 0.0)

        # Void behavior diversity (trump vs slough and schmear points)
        void_rate_diff = abs(self.strategic_profile.get_void_trump_rate() - other.strategic_profile.get_void_trump_rate())
        schmear_diff = abs(self.strategic_profile.get_avg_schmeared_points() - other.strategic_profile.get_avg_schmeared_points()) / 11.0
        diversity_components.append(np.mean([void_rate_diff, schmear_diff]))

        # Risk tolerance diversity (focus on ALONE usage)
        risk1 = self.strategic_profile.get_alone_rate()
        risk2 = other.strategic_profile.get_alone_rate() if hasattr(other, 'strategic_profile') else 0.0
        diversity_components.append(abs(risk1 - risk2))

        # Bury quality diversity
        bury1 = self.strategic_profile.get_bury_quality_score()
        bury2 = other.strategic_profile.get_bury_quality_score()
        diversity_components.append(abs(bury1 - bury2))

        # Role performance diversity
        role_diversities = []
        for role in ['picker', 'partner', 'defender', 'leaster']:
            perf1 = self.strategic_profile.get_role_performance(role)
            perf2 = other.strategic_profile.get_role_performance(role)
            role_diversities.append(abs(perf1 - perf2))

        if role_diversities:
            diversity_components.append(np.mean(role_diversities))

        # Calculate overall diversity (weighted average)
        if not diversity_components:
            return 0.5

        return min(1.0, np.mean(diversity_components))

    def get_strategic_signature(self) -> np.ndarray:
        """Get a compact numerical signature representing this agent's strategy."""
        signature = []

        # Pick rates by hand strength
        for strength in ['weak', 'medium', 'strong']:
            signature.append(self.strategic_profile.get_pick_rate_by_hand_strength(strength))

        # Trump leading rates (overall and early)
        signature.append(self.strategic_profile.get_trump_lead_rate('picker'))
        signature.append(self.strategic_profile.get_trump_lead_rate('defender'))
        signature.append(self.strategic_profile.get_early_trump_lead_rate('picker'))
        signature.append(self.strategic_profile.get_early_trump_lead_rate('defender'))

        # Void behavior
        signature.append(self.strategic_profile.get_void_trump_rate())
        signature.append(min(1.0, self.strategic_profile.get_avg_schmeared_points() / 11.0))

        # ALONE usage rate
        signature.append(self.strategic_profile.get_alone_rate())

        # Bury quality
        signature.append(self.strategic_profile.get_bury_quality_score())

        # Role performances (optional)
        for role in ['picker', 'partner', 'defender', 'leaster']:
            signature.append(self.strategic_profile.get_role_performance(role))

        # EWMA strategic rates (augment clustering features)
        sp = self.strategic_profile
        signature.append(sp.trick_win_rate_picker)
        signature.append(sp.trick_win_rate_partner)
        signature.append(sp.trick_win_rate_defender)
        signature.append(sp.lead_win_rate_picker)
        signature.append(sp.lead_win_rate_partner)
        signature.append(sp.lead_win_rate_defender)
        signature.append(sp.early_trump_play_rate)
        signature.append(sp.bury_high_points_rate)
        signature.append(sp.schmear_rate_when_void)
        signature.append(sp.pick_hand_correlation)

        return np.array(signature, dtype=np.float32)

    def update_strategic_profile_from_game(self, game_data: Dict):
        """Update strategic profile based on game observations."""
        if 'pick_decision' in game_data:
            hand_strength = game_data.get('hand_strength_category', 'medium')
            picked = game_data['pick_decision']
            self.strategic_profile.pick_total_by_hand_strength[hand_strength] += 1
            if picked:
                self.strategic_profile.pick_true_by_hand_strength[hand_strength] += 1
            # Online pick-hand correlation via EWMAs: X in {0,1,2}, Y in {0,1}
            x_map = {'weak': 0.0, 'medium': 1.0, 'strong': 2.0}
            x = float(x_map.get(hand_strength, 1.0))
            y = 1.0 if picked else 0.0
            a = EMA_ALPHA_PLAY
            sp = self.strategic_profile
            sp._pick_corr_ewma_x = (1.0 - a) * sp._pick_corr_ewma_x + a * x
            sp._pick_corr_ewma_y = (1.0 - a) * sp._pick_corr_ewma_y + a * y
            sp._pick_corr_ewma_x2 = (1.0 - a) * sp._pick_corr_ewma_x2 + a * (x * x)
            sp._pick_corr_ewma_y2 = (1.0 - a) * sp._pick_corr_ewma_y2 + a * (y * y)
            sp._pick_corr_ewma_xy = (1.0 - a) * sp._pick_corr_ewma_xy + a * (x * y)
            # Compute correlation safely
            var_x = max(1e-6, sp._pick_corr_ewma_x2 - sp._pick_corr_ewma_x * sp._pick_corr_ewma_x)
            var_y = max(1e-6, sp._pick_corr_ewma_y2 - sp._pick_corr_ewma_y * sp._pick_corr_ewma_y)
            cov_xy = sp._pick_corr_ewma_xy - sp._pick_corr_ewma_x * sp._pick_corr_ewma_y
            sp.pick_hand_correlation = float(np.clip(cov_xy / (np.sqrt(var_x) * np.sqrt(var_y)), -1.0, 1.0))

        if 'trump_lead' in game_data:
            role = game_data.get('role', 'defender')
            led_trump = bool(game_data['trump_lead'])
            is_early = bool(game_data.get('is_early_lead', False))

            if role == 'picker':
                self.strategic_profile.lead_counts_picker += 1
                if led_trump:
                    self.strategic_profile.lead_trump_counts_picker += 1
                if is_early:
                    self.strategic_profile.early_lead_counts_picker += 1
                    if led_trump:
                        self.strategic_profile.early_lead_trump_counts_picker += 1
            elif role in ['defender']:
                self.strategic_profile.lead_counts_defender += 1
                if led_trump:
                    self.strategic_profile.lead_trump_counts_defender += 1
                if is_early:
                    self.strategic_profile.early_lead_counts_defender += 1
                    if led_trump:
                        self.strategic_profile.early_lead_trump_counts_defender += 1

        if 'early_trump_play' in game_data:
            s = 1.0 if bool(game_data['early_trump_play']) else 0.0
            sp = self.strategic_profile
            sp.early_trump_play_rate = (1.0 - EMA_ALPHA_PLAY) * sp.early_trump_play_rate + EMA_ALPHA_PLAY * s

        if game_data.get('void_event', False):
            self.strategic_profile.void_events += 1
            if game_data.get('void_played_trump', False):
                self.strategic_profile.void_trump_events += 1
            else:
                self.strategic_profile.void_fail_events += 1
                self.strategic_profile.schmeared_points_sum += int(game_data.get('schmeared_points', 0))
                # EWMA schmear rate when void
                void_schmear = bool(game_data.get('void_schmear', False))
                s = 1.0 if void_schmear else 0.0
                sp = self.strategic_profile
                sp.schmear_rate_when_void = (1.0 - EMA_ALPHA_PLAY) * sp.schmear_rate_when_void + EMA_ALPHA_PLAY * s

        if 'bury_decision' in game_data:
            points = game_data.get('card_points', 0)
            try:
                pts = int(points)
                self.strategic_profile.bury_points_sum += pts
                self.strategic_profile.bury_count += 1
                # EWMA bury high points rate
                sp = self.strategic_profile
                sp.bury_high_points_rate = (1.0 - EMA_ALPHA_RARE) * sp.bury_high_points_rate + EMA_ALPHA_RARE * (1.0 if pts >= 10 else 0.0)
            except (TypeError, ValueError) as err:
                logging.warning("Invalid bury card points in strategic profile update", extra={
                    "error": str(err),
                    "points": points,
                    "game_data_keys": list(game_data.keys())
                })

        if 'alone_call' in game_data:
            self.strategic_profile.alone_total_count += 1
            if game_data['alone_call']:
                self.strategic_profile.alone_true_count += 1

        # Performance tracking
        if 'final_score' in game_data and 'role' in game_data:
            score = game_data['final_score']
            role = game_data['role']
            try:
                score_f = float(score)
            except (TypeError, ValueError) as err:
                logging.warning("Invalid final_score in strategic profile update", extra={
                    "error": str(err),
                    "score": score,
                    "role": role
                })
                score_f = 0.0
            if role == 'picker':
                self.strategic_profile.performance_sum_picker += score_f
                self.strategic_profile.performance_count_picker += 1
            elif role == 'partner':
                self.strategic_profile.performance_sum_partner += score_f
                self.strategic_profile.performance_count_partner += 1
            elif role == 'defender':
                self.strategic_profile.performance_sum_defender += score_f
                self.strategic_profile.performance_count_defender += 1
            elif role == 'leaster':
                self.strategic_profile.performance_sum_leaster += score_f
                self.strategic_profile.performance_count_leaster += 1

    def record_vs_training_outcome(self, result: float, alpha: float = 0.05) -> None:
        """Update exponentially-smoothed win-rate vs the training agent.

        Args:
            result: 1.0 if this agent ranked ahead of the training agent, 0.0 if behind, 0.5 for ties
            alpha: smoothing factor for EMA (higher reacts faster)
        """
        # Clamp to [0,1] and update EMA
        if result < 0.0:
            result = 0.0
        elif result > 1.0:
            result = 1.0
        self.exploitation_win_rate_ema = (1.0 - alpha) * self.exploitation_win_rate_ema + alpha * float(result)
        self.exploitation_samples += 1

    def get_exploitation_score_against(self) -> float:
        """Return decayed win-rate vs training agent for exploitation sampling."""
        return float(self.exploitation_win_rate_ema)


class PFSPPopulation:
    """Manages the population of agents for PFSP training."""

    def __init__(self,
                 max_population_jd: int = 75,
                 max_population_called_ace: int = 75,
                 population_dir: str = "pfsp_population",
                 anchor_quota_per_cluster: int = 2):

        self.max_population_jd = max_population_jd
        self.max_population_called_ace = max_population_called_ace
        self.population_dir = Path(population_dir)
        self.rating_model = PlackettLuce()

        # Anchor quota per cluster (style-preserving anchors)
        self.anchor_quota_per_cluster = anchor_quota_per_cluster

        # Separate populations for different partner modes
        self.jd_population: List[PopulationAgent] = []
        self.called_ace_population: List[PopulationAgent] = []

        # Cluster sampling counters (ephemeral, not persisted)
        self._cluster_sampling_counts = {
            PARTNER_BY_JD: defaultdict(int),
            PARTNER_BY_CALLED_ACE: defaultdict(int),
        }
        self._cluster_sampling_totals = {
            PARTNER_BY_JD: 0,
            PARTNER_BY_CALLED_ACE: 0,
        }

        # Selection category sampling counters (ephemeral, not persisted)
        # Keys per mode: 'hardest', 'rarest', 'anchor', 'pressure', 'weighted'
        self._selection_sampling_counts = {
            PARTNER_BY_JD: defaultdict(int),
            PARTNER_BY_CALLED_ACE: defaultdict(int),
        }
        self._selection_sampling_totals = {
            PARTNER_BY_JD: 0,
            PARTNER_BY_CALLED_ACE: 0,
        }

        # Ensure population directory exists
        self.population_dir.mkdir(exist_ok=True)
        (self.population_dir / "jd_agents").mkdir(exist_ok=True)
        (self.population_dir / "called_ace_agents").mkdir(exist_ok=True)

        # Load existing population if available
        self._load_population()

    def _get_population(self, partner_mode: int) -> List[PopulationAgent]:
        """Get the appropriate population for the partner mode."""
        if partner_mode == PARTNER_BY_JD:
            return self.jd_population
        elif partner_mode == PARTNER_BY_CALLED_ACE:
            return self.called_ace_population
        else:
            raise ValueError(f"Unknown partner mode: {partner_mode}")

    def _get_max_population(self, partner_mode: int) -> int:
        """Get max population size for the partner mode."""
        if partner_mode == PARTNER_BY_JD:
            return self.max_population_jd
        elif partner_mode == PARTNER_BY_CALLED_ACE:
            return self.max_population_called_ace
        else:
            raise ValueError(f"Unknown partner mode: {partner_mode}")

    def _get_subdir(self, partner_mode: int) -> Path:
        """Get subdirectory for the partner mode."""
        if partner_mode == PARTNER_BY_JD:
            return self.population_dir / "jd_agents"
        elif partner_mode == PARTNER_BY_CALLED_ACE:
            return self.population_dir / "called_ace_agents"
        else:
            raise ValueError(f"Unknown partner mode: {partner_mode}")

    def _get_hof_anchors(self, partner_mode: int) -> List[PopulationAgent]:
        """Select Hall-of-Fame anchors per cluster (top by skill), capped by population size.

        Rationale: clusters approximate strategic styles; anchoring by cluster preserves style coverage.
        """
        population = self._get_population(partner_mode)
        max_size = self._get_max_population(partner_mode)
        if not population or self.anchor_quota_per_cluster <= 0:
            return []

        _, clusters = self._cluster_population(partner_mode)
        anchors: List[PopulationAgent] = []
        for label, agents in clusters.items():
            # For noise cluster (-1), still select top-K by skill
            top_k = sorted(agents, key=lambda a: a.get_skill_estimate(), reverse=True)[:self.anchor_quota_per_cluster]
            anchors.extend(top_k)

        if len(anchors) > max_size:
            anchors = sorted(anchors, key=lambda a: a.get_skill_estimate(), reverse=True)[:max_size]

        return anchors

    def _select_rarest_cluster(self, clusters: dict[int, list[PopulationAgent]]) -> Optional[int]:
        if not clusters:
            return None
        return min(clusters.keys(), key=lambda label_key: len(clusters[label_key]))

    def _select_hardest_cluster(self,
                                clusters: dict[int, list[PopulationAgent]],
                                training_agent_id: Optional[str]) -> Optional[int]:
        if not clusters:
            return None
        if not training_agent_id:
            # If no training agent context, we cannot score hardness; return None
            return None
        cluster_difficulty: dict[int, float] = {}
        for label, members in clusters.items():
            if members:
                cluster_difficulty[label] = float(np.mean([m.get_exploitation_score_against() for m in members]))
            else:
                cluster_difficulty[label] = 0.0
        return max(clusters.keys(), key=lambda label_key: cluster_difficulty.get(label_key, 0.0))

    def _pick_one_from_cluster(self,
                               label: int,
                               clusters: dict[int, list[PopulationAgent]],
                               remaining_population: list[PopulationAgent],
                               partner_mode: int,
                               selected_agents: list[PopulationAgent]) -> bool:
        agents_in_cluster = [a for a in remaining_population if a in clusters.get(label, [])]
        if not agents_in_cluster:
            return False
        choice = max(agents_in_cluster, key=lambda a: a.get_skill_estimate())
        selected_agents.append(choice)
        remaining_population.remove(choice)
        self._cluster_sampling_counts[partner_mode][label] += 1
        self._cluster_sampling_totals[partner_mode] += 1
        return True

    def add_agent(self,
                  agent: PPOAgent,
                  partner_mode: int,
                  training_episodes: int,
                  parent_id: Optional[str] = None,
                  activation: str = 'swish',
                  save: bool = True,
                  prune: bool = True,
                  initial_rating=None) -> str:
        """Add a new agent to the population."""

        # Generate unique ID
        agent_id = f"{partner_mode}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Create metadata
        metadata = AgentMetadata(
            agent_id=agent_id,
            creation_time=time.time(),
            parent_id=parent_id,
            training_episodes=training_episodes,
            partner_mode=partner_mode,
            activation=activation
        )

        # Create population agent
        pop_agent = PopulationAgent(agent, metadata, rating=initial_rating)

        # Add to appropriate population
        population = self._get_population(partner_mode)
        population.append(pop_agent)

        print(f"✅ Added agent {agent_id} to {get_partner_mode_name(partner_mode)} population")

        # Manage population size (optional for bulk operations)
        if prune:
            self._manage_population_size(partner_mode)

        # Save the new agent (optional for bulk operations)
        if save:
            self._save_agent(pop_agent)

        print(f"   Population size: {len(population)}/{self._get_max_population(partner_mode)}")

        return agent_id

    def _select_pruning_plan(self, partner_mode: int) -> tuple[list['PopulationAgent'], list['PopulationAgent']]:
        """Compute agents to keep and to remove without side effects.

        Returns:
            (agents_to_keep, agents_to_remove)
        """
        population = self._get_population(partner_mode)
        max_size = self._get_max_population(partner_mode)

        anchors = self._get_hof_anchors(partner_mode)
        anchor_set = set(anchors)

        if len(population) <= max_size:
            return population.copy(), []

        labels, clusters = self._cluster_population(partner_mode)
        mandatory_ids = set()
        for label, members in clusters.items():
            if len(members) <= 1:
                mandatory_ids.update(id(m) for m in members)

        # Phase 1 pruning by simple priority while preserving anchors/singletons
        working_population = population.copy()
        if len(working_population) > max_size:
            candidates = [a for a in working_population if a not in anchor_set and id(a) not in mandatory_ids]
            # Blend skill (μ) with exploitation EMA (keep high exploitation agents longer)
            # Lower blended score ⇒ removed earlier
            exploitation_weight = 5.0
            def removal_heuristic(agent: PopulationAgent) -> float:
                return float(agent.get_skill_estimate()) + exploitation_weight * float(getattr(agent, 'exploitation_win_rate_ema', 0.5))
            candidates.sort(key=removal_heuristic)
            for agent in candidates:
                if len(working_population) <= max_size:
                    break
                working_population.remove(agent)

        if len(working_population) <= max_size:
            agents_to_keep = working_population
            agents_to_remove = [a for a in population if a not in agents_to_keep]
            return agents_to_keep, agents_to_remove

        # Phase 2 diversity-aware selection among non-anchors
        non_anchor_population = [a for a in working_population if a not in anchor_set]
        slots_for_non_anchors = max(0, max_size - len(anchor_set))
        agents_to_keep_non_anchor = self._select_diverse_population_subset(non_anchor_population, slots_for_non_anchors)
        agents_to_keep = anchors + agents_to_keep_non_anchor
        agents_to_remove = [agent for agent in population if agent not in agents_to_keep]
        return agents_to_keep, agents_to_remove

    def _manage_population_size(self, partner_mode: int):
        """Remove agents if population exceeds max size, preserving diversity and strength.

        Selection (who to keep/remove) is computed first without side effects,
        then file deletions are applied.
        """
        population = self._get_population(partner_mode)
        agents_to_keep, agents_to_remove_list = self._select_pruning_plan(partner_mode)

        # Mutate the existing list in-place to avoid stale references
        population[:] = agents_to_keep

        # Remove files for deleted agents
        for removed_agent in agents_to_remove_list:
            self._delete_agent_files(removed_agent)
            print(f"🗑️  Removed agent {removed_agent.metadata.agent_id} "
                  f"(skill: {removed_agent.get_skill_estimate():.1f}, "
                  f"diversity preserved)")

    def _select_diverse_population_subset(self, population: List[PopulationAgent], target_size: int) -> List[PopulationAgent]:
        """Select a diverse subset of agents using strategic diversity metrics."""
        if len(population) <= target_size:
            return population.copy()

        # Step 1: Always keep the strongest agents (top 20%)
        population_by_skill = sorted(population, key=lambda x: x.get_skill_estimate(), reverse=True)
        essential_agents = population_by_skill[:max(1, target_size // 5)]
        remaining_candidates = [agent for agent in population if agent not in essential_agents]
        slots_remaining = target_size - len(essential_agents)

        if slots_remaining <= 0:
            return essential_agents

        # Step 2: Use greedy diversity selection for remaining slots
        selected_agents = essential_agents.copy()

        while len(selected_agents) < target_size and remaining_candidates:
            best_candidate = None
            best_score = -1

            for candidate in remaining_candidates:
                # Calculate diversity score compared to already selected agents
                diversity_scores = []
                for selected_agent in selected_agents:
                    diversity_score = candidate.calculate_strategic_diversity(selected_agent)
                    diversity_scores.append(diversity_score)

                avg_diversity = np.mean(diversity_scores) if diversity_scores else 1.0

                # Combine diversity with skill and recency
                skill_score = candidate.get_skill_estimate() / 50.0  # Normalize to 0-1
                recency_score = 1.0 / (1.0 + (time.time() - candidate.metadata.creation_time) / (24 * 3600 * 30))  # Decay over 30 days
                uncertainty_bonus = 0.0

                # Weighted combination favoring diversity
                total_score = (
                    0.5 * avg_diversity +      # 50% diversity
                    0.25 * skill_score +       # 25% skill
                    0.15 * recency_score +     # 15% recency
                    0.1 * uncertainty_bonus    # 10% learning potential
                )

                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate

            if best_candidate:
                selected_agents.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break  # No more candidates

        # If we still need more agents, add the most recent ones
        while len(selected_agents) < target_size and remaining_candidates:
            most_recent = max(remaining_candidates, key=lambda x: x.metadata.creation_time)
            selected_agents.append(most_recent)
            remaining_candidates.remove(most_recent)

        return selected_agents

    def _delete_agent_files(self, pop_agent: PopulationAgent):
        """Delete agent files from disk."""
        subdir = self._get_subdir(pop_agent.metadata.partner_mode)
        agent_file = subdir / f"{pop_agent.metadata.agent_id}.pt"
        metadata_file = subdir / f"{pop_agent.metadata.agent_id}_metadata.json"

        for file_path in [agent_file, metadata_file]:
            if file_path.exists():
                file_path.unlink()

    def sample_opponents(self,
                        partner_mode: int,
                        n_opponents: int = 4,
                        training_agent_skill: float = 25.0,
                        training_agent_id: str = None,
                        skill_weight: float = 0.4,
                        diversity_weight: float = 0.3,
                        exploitation_weight: float = 0.2,
                        curriculum_weight: float = 0.1,
                        selected_opponents: List[PopulationAgent] = None,
                        uniform_mix: float = 0.15,
                        include_cluster_anchor: bool = True,
                        # Reserved slot controls
                        max_reserved_slots: int = 1,
                        p_rarest: float = 0.3,
                        p_hardest: float = 0.6,
                        p_anchor: float = 0.3,
                        min_anchor_percentile: float = 0.4,
                        sigma_ceiling: float = 12.0,
                        include_pressure_slot: bool = True) -> List[PopulationAgent]:
        """Sample opponents using multi-objective selection (skill + diversity + exploitation + curriculum),
        with optional inclusion of a cluster-based anchor to prevent forgetting.
        """
        population = self._get_population(partner_mode)

        if len(population) == 0:
            return []

        if len(population) <= n_opponents:
            return population.copy()

        # Track already selected opponents to ensure diversity
        already_selected = selected_opponents or []

        selected_agents = []
        remaining_population = population.copy()

        # Cluster-aware selection metadata
        labels, clusters = self._cluster_population(partner_mode)
        # Stable mapping from agent instance to its cluster label for telemetry during sampling
        agent_to_label = {id(agent): int(label) for agent, label in zip(population, labels)}
        rarest_label = self._select_rarest_cluster(clusters)
        hardest_label = self._select_hardest_cluster(clusters, training_agent_id)

        # Competence floor for reserved/anchor picks (avoid forcing weak agents)
        mu_values = [a.get_skill_estimate() for a in population]
        mu_floor = float(np.percentile(mu_values, min(99.0, max(0.0, min_anchor_percentile * 100.0)))) if mu_values else -1e9
        def is_competent(agent: PopulationAgent) -> bool:
            try:
                return (agent.get_skill_estimate() >= mu_floor) and (agent.get_skill_uncertainty() <= sigma_ceiling)
            except Exception:
                return False

        # At most one reserved pick via probabilistic gating among {hardest, rarest, anchor}
        reserved_used = 0
        if max_reserved_slots > 0 and len(selected_agents) < n_opponents:
            choices = []
            probs = []
            if hardest_label is not None:
                choices.append('hardest')
                probs.append(max(0.0, p_hardest))
            if rarest_label is not None:
                choices.append('rarest')
                probs.append(max(0.0, p_rarest))
            if include_cluster_anchor:
                choices.append('anchor')
                probs.append(max(0.0, p_anchor))
            if choices and sum(probs) > 0:
                probs = np.array(probs, dtype=np.float64)
                probs = probs / probs.sum()
                pick_order = list(np.random.choice(choices, size=len(choices), replace=False, p=probs))
            else:
                pick_order = []

            for kind in pick_order:
                if reserved_used >= max_reserved_slots or len(selected_agents) >= n_opponents:
                    break
                if kind == 'hardest' and hardest_label is not None:
                    cluster_agents = [a for a in remaining_population if a in clusters.get(hardest_label, [])]
                    if cluster_agents:
                        candidate = max(cluster_agents, key=lambda a: a.get_skill_estimate())
                        if is_competent(candidate):
                            selected_agents.append(candidate)
                            remaining_population.remove(candidate)
                            self._cluster_sampling_counts[partner_mode][int(hardest_label)] += 1
                            self._cluster_sampling_totals[partner_mode] += 1
                            self._selection_sampling_counts[partner_mode]['hardest'] += 1
                            self._selection_sampling_totals[partner_mode] += 1
                            reserved_used += 1
                elif kind == 'rarest' and rarest_label is not None:
                    cluster_agents = [a for a in remaining_population if a in clusters.get(rarest_label, [])]
                    if cluster_agents:
                        candidate = max(cluster_agents, key=lambda a: a.get_skill_estimate())
                        if is_competent(candidate):
                            selected_agents.append(candidate)
                            remaining_population.remove(candidate)
                            self._cluster_sampling_counts[partner_mode][int(rarest_label)] += 1
                            self._cluster_sampling_totals[partner_mode] += 1
                            self._selection_sampling_counts[partner_mode]['rarest'] += 1
                            self._selection_sampling_totals[partner_mode] += 1
                            reserved_used += 1
                elif kind == 'anchor':
                    anchors = [a for a in self._get_hof_anchors(partner_mode) if a not in already_selected]
                    if anchors:
                        anchors_sorted = sorted(anchors, key=lambda a: abs(a.get_skill_estimate() - training_agent_skill), reverse=True)
                        anchor_choice = next((a for a in anchors_sorted if a in remaining_population and a not in selected_agents and is_competent(a)), None)
                        if anchor_choice is not None:
                            selected_agents.append(anchor_choice)
                            remaining_population.remove(anchor_choice)
                            # Update cluster counters if known
                            lb = agent_to_label.get(id(anchor_choice))
                            if lb is not None:
                                self._cluster_sampling_counts[partner_mode][int(lb)] += 1
                                self._cluster_sampling_totals[partner_mode] += 1
                            self._selection_sampling_counts[partner_mode]['anchor'] += 1
                            self._selection_sampling_totals[partner_mode] += 1
                            reserved_used += 1

        # Dedicated pressure slot (hard opponent)
        if include_pressure_slot and len(selected_agents) < n_opponents and remaining_population:
            # Prefer agents that recently beat the training agent
            def threat_key(a: PopulationAgent) -> float:
                val = float(getattr(a, 'exploitation_win_rate_ema', 0.5))
                n = int(getattr(a, 'exploitation_samples', 0))
                return val * (1.0 - np.exp(-max(0, n) / 5.0))
            candidate = max(remaining_population, key=threat_key)
            selected_agents.append(candidate)
            remaining_population.remove(candidate)
            lb = agent_to_label.get(id(candidate))
            if lb is not None:
                self._cluster_sampling_counts[partner_mode][int(lb)] += 1
                self._cluster_sampling_totals[partner_mode] += 1
            self._selection_sampling_counts[partner_mode]['pressure'] += 1
            self._selection_sampling_totals[partner_mode] += 1

        for _ in range(n_opponents - len(selected_agents)):
            if not remaining_population:
                break

            weights = []
            for pop_agent in remaining_population:
                weight_components = {}

                # 1. Skill-based weight (prefer similar skill levels) factoring in uncertainty
                agent_mu = pop_agent.get_skill_estimate()
                skill_diff = abs(agent_mu - training_agent_skill)
                # Higher uncertainty reduces the weight
                weight_components['skill'] = np.exp(-(skill_diff) / 10.0)

                # 2. Diversity weight (prefer agents different from already selected)
                diversity_scores = []

                # Compare with already selected agents
                for selected_agent in (already_selected + selected_agents):
                    if hasattr(selected_agent, 'strategic_profile'):
                        diversity_score = pop_agent.calculate_strategic_diversity(selected_agent)
                        diversity_scores.append(diversity_score)

                # High diversity = good (want different strategies)
                if diversity_scores:
                    weight_components['diversity'] = np.mean(diversity_scores)
                else:
                    weight_components['diversity'] = 1.0  # Max diversity if no comparison

                # 3. Exploitation weight (prefer agents that recently beat training agent)
                if training_agent_id:
                    exploitation_score = pop_agent.get_exploitation_score_against()
                    weight_components['exploitation'] = exploitation_score
                else:
                    weight_components['exploitation'] = 0.0

                # 4. Curriculum weight (include some weaker agents for stable learning)
                skill_gap = training_agent_skill - pop_agent.get_skill_estimate()
                if skill_gap > 0:  # Agent is weaker than training agent
                    # Prefer moderately weaker agents (not too weak)
                    curriculum_score = np.exp(-abs(skill_gap - 5.0) / 5.0)  # Peak at 5 skill points weaker
                else:
                    curriculum_score = 0.1  # Small bonus for stronger agents
                weight_components['curriculum'] = curriculum_score

                # Combine weights
                total_weight = (
                    skill_weight * weight_components['skill'] +
                    diversity_weight * weight_components['diversity'] +
                    exploitation_weight * weight_components['exploitation'] +
                    curriculum_weight * weight_components['curriculum']
                )

                weights.append(max(0.01, total_weight))  # Ensure minimum weight

            # Sample one opponent
            if not weights:
                break

            # Normalise weights and blend with a small uniform mixture for exploration breadth
            weights = np.array(weights, dtype=np.float64)
            total = weights.sum()
            if total <= 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / total
            if uniform_mix > 0:
                uniform = np.full_like(weights, 1.0 / len(weights))
                weights = (1 - uniform_mix) * weights + uniform_mix * uniform
                weights = weights / weights.sum()

            selected_idx = np.random.choice(len(remaining_population), p=weights)
            selected_agent = remaining_population.pop(selected_idx)
            selected_agents.append(selected_agent)
            # Update sampling counters by cluster label when available
            label = agent_to_label.get(id(selected_agent))
            if label is not None:
                self._cluster_sampling_counts[partner_mode][label] += 1
                self._cluster_sampling_totals[partner_mode] += 1
            else:
                logging.warning("Cluster label missing for selected agent during sampling", extra={
                    "partner_mode": get_partner_mode_name(partner_mode),
                    "population_size": len(population)
                })
            # Selection category share for weighted sampling
            self._selection_sampling_counts[partner_mode]['weighted'] += 1
            self._selection_sampling_totals[partner_mode] += 1

        return selected_agents

    def get_population_stats(self, partner_mode: int) -> Dict:
        """Get statistics about the population."""
        population = self._get_population(partner_mode)

        if not population:
            return {
                'size': 0,
                'avg_skill': 0.0,
                'skill_range': (0.0, 0.0),
                'avg_games': 0,
                'oldest_agent_days': 0.0
            }

        skills = [agent.get_skill_estimate() for agent in population]
        games_played = [agent.metadata.games_played for agent in population]
        ages = [(time.time() - agent.metadata.creation_time) / (24 * 3600) for agent in population]

        return {
            'size': len(population),
            'avg_skill': np.mean(skills),
            'skill_range': (np.min(skills), np.max(skills)),
            'avg_games': int(np.mean(games_played)) if games_played else 0,
            'oldest_agent_days': max(ages) if ages else 0.0
        }

    def renormalize_all_mus(self, training_rating, max_abs_mu: float = 350.0) -> tuple[float, float]:
        """Clamp all rating μ magnitudes to a ceiling to avoid numerical overflow.

        Returns (extreme_mu_before, shift_applied).
        """
        all_ratings = [training_rating] + [ag.rating for ag in self.jd_population] + [ag.rating for ag in self.called_ace_population]
        extreme_mu = max(abs(r.mu) for r in all_ratings) if all_ratings else 0.0
        shift = 0.0
        if extreme_mu > max_abs_mu:
            shift = float(extreme_mu - max_abs_mu)
            for r in all_ratings:
                # Reduce magnitude while preserving sign
                r.mu -= np.sign(r.mu) * shift
        return float(extreme_mu), float(shift)

    @staticmethod
    def compute_ranks_from_scores(scores: List[float]) -> List[int]:
        """Compute OpenSkill ranks from score list (higher score ⇒ better rank).

        Ties receive the same rank index as implemented in existing logic.
        """
        score_rank_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranks = [0] * len(scores)
        current_rank = 0
        last_score = None
        for i, (idx, sc) in enumerate(score_rank_pairs):
            if last_score is None or sc != last_score:
                current_rank = i
            ranks[idx] = current_rank
            last_score = sc
        return ranks

    def update_ratings_with_training(self,
                                     training_rating,
                                     final_scores: List[float],
                                     training_position: int,
                                     opponents_by_position: Dict[int, PopulationAgent],
                                     picker_seat: Optional[int]) -> object:
        """Update ratings for a single 5-seat game that includes the training agent.

        - Uses this population's rating model
        - Updates population opponents' ratings and game stats
        - Records decayed exploitation results vs the training agent
        - Returns the updated training agent rating object
        """
        if not final_scores or len(final_scores) != 5:
            return training_rating

        positions_order = [1, 2, 3, 4, 5]
        # Build teams list: training agent is a single-member team; each population opponent as its own team
        teams = []
        for pos in positions_order:
            if pos == training_position:
                teams.append([training_rating])
            else:
                opp = opponents_by_position.get(pos)
                # In rare cases a seat may be unassigned; skip rating update gracefully
                teams.append([opp.rating] if opp else [PlackettLuce().rating()])

        ranks = PFSPPopulation.compute_ranks_from_scores(final_scores)

        try:
            new_teams = self.rating_model.rate(teams, ranks=ranks)
        except ValueError as err:
            logging.warning("Failed to update ratings (train match)", extra={
                "error": str(err),
                "teams": len(teams),
                "ranks_len": len(ranks)
            })
            return training_rating

        # Apply updated ratings and exploitation results
        training_rank = ranks[training_position - 1]
        for idx, pos in enumerate(positions_order):
            if pos == training_position:
                training_rating = new_teams[idx][0]
            else:
                opp_agent = opponents_by_position.get(pos)
                if not opp_agent:
                    continue
                opp_agent.update_rating(new_teams[idx][0])
                was_picker = (picker_seat == pos)
                opp_agent.add_game_result(final_scores[pos - 1], was_picker)

                # Rank-based exploitation outcome vs training agent
                opp_rank = ranks[pos - 1]
                if opp_rank < training_rank:
                    result_vs_training = 1.0
                elif opp_rank > training_rank:
                    result_vs_training = 0.0
                else:
                    result_vs_training = 0.5
                opp_agent.record_vs_training_outcome(result_vs_training)

        return training_rating

    def update_ratings(self,
                      game_results: List[Tuple[PopulationAgent, int, float, int]],
                      partner_mode: int):
        """Update agent ratings based on game results.

        Args:
            game_results: List of (agent, position, score) tuples
            partner_mode: Partner mode used in the game
        """
        if len(game_results) < 2:
            return

        # Convert to format expected by OpenSkill
        # Create teams (each player is their own team)
        teams = [[agent.rating] for agent, _, _, _ in game_results]

        # Convert scores to OpenSkill ranks (lower rank index = better performance)
        scores = [result[2] for result in game_results]
        ranks = PFSPPopulation.compute_ranks_from_scores(scores)

        # Update ratings using OpenSkill
        try:
            new_teams = self.rating_model.rate(teams, ranks=ranks)

            # Update agent ratings
            for i, (agent, position, score, picker_seat) in enumerate(game_results):
                agent.update_rating(new_teams[i][0])
                # Also update performance tracking
                was_picker = (position == picker_seat)
                agent.add_game_result(score, was_picker)

        except ValueError as err:
            logging.warning("Failed to update ratings due to invalid inputs", extra={
                "error": str(err),
                "teams": len(teams),
                "ranks_len": len(ranks)
            })

    def run_cross_evaluation(self, partner_mode: int, num_games: int, max_agents: int) -> Dict:
        """Run cross-evaluation tournament to update ratings."""
        population = self._get_population(partner_mode)

        if len(population) < 2:
            print(f"Skipping cross-evaluation for {get_partner_mode_name(partner_mode)} - insufficient agents")
            return {}

        # Limit to most recent/strongest agents to keep evaluation manageable
        if len(population) > max_agents:
            # Sort by skill and recency, take top agents
            population = sorted(
                population,
                key=lambda x: x.get_skill_estimate() + (x.metadata.creation_time - time.time()) / (24 * 3600 * 30),
                reverse=True
            )[:max_agents]

        print(f"🏆 Running cross-evaluation for {get_partner_mode_name(partner_mode)} population")
        print(f"   Evaluating {len(population)} agents with {num_games} games")

        total_games = 0
        game_results = []

        # Run round-robin style evaluation
        for game_idx in range(num_games):
            # Shuffle agents for this round
            agents_shuffled = population.copy()
            random.shuffle(agents_shuffled)

            # Play games with random 5-agent groups
            for start_idx in range(0, len(agents_shuffled) - 4, 5):
                game_agents = agents_shuffled[start_idx:start_idx + 5]

                # Create and play game
                game = Game(partner_selection_mode=partner_mode)

                # Reset agent states
                for agent in game_agents:
                    agent.agent.reset_recurrent_state()

                # Play game with population agents
                agent_positions = list(range(1, 6))  # Positions 1-5
                final_scores, picker_seat = self._play_evaluation_game(game, game_agents, agent_positions)

                # Store results
                round_results = []
                for i, agent in enumerate(game_agents):
                    round_results.append((agent, agent_positions[i], final_scores[i], picker_seat))

                game_results.extend(round_results)
                total_games += 1

        # Update ratings based on all games
        if game_results:
            # Group results by game and update ratings
            games_played = total_games
            for game_idx in range(games_played):
                start_idx = game_idx * 5
                end_idx = start_idx + 5
                if end_idx <= len(game_results):
                    game_group = game_results[start_idx:end_idx]
                    self.update_ratings(game_group, partner_mode)

        print(f"   ✅ Completed {total_games} evaluation games")

        # Return summary statistics
        return {
            'games_played': total_games,
            'agents_evaluated': len(population),
            'avg_skill_after': np.mean([a.get_skill_estimate() for a in population]),
            'skill_spread': np.std([a.get_skill_estimate() for a in population])
        }

    def _play_evaluation_game(self, game: Game, agents: List[PopulationAgent], positions: List[int]) -> Tuple[List[float], int]:
        """Play a single evaluation game and return final scores."""
        # Create position to agent mapping
        pos_to_agent = {pos: agent for pos, agent in zip(positions, agents)}

        # Hand strength categories captured once at start for pick profiling
        hand_strength_by_pos = {p.position: estimate_hand_strength_category(p.hand) for p in game.players}

        while not game.is_done():
            for player in game.players:
                agent = pos_to_agent[player.position]
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    state = player.get_state_dict()
                    action, _, _ = agent.agent.act(state, valid_actions, player.position, deterministic=True)
                    player.act(action)
                    valid_actions = player.get_valid_action_ids()

                    # Propagate observations to all agents at the end of the trick
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            seat_agent = pos_to_agent[seat.position]
                            seat_agent.agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                        # Update trick-level EWMAs for population agents
                        profile_trick_completion(game, pos_to_agent)

                    # Centralized profiling for population agents
                    pop_agent = pos_to_agent[player.position]
                    profile_pop_agent_action(game, player, action, pop_agent, hand_strength_by_pos)

        # Final per-role performance profiling for strategic metrics
        if game.is_leaster:
            for player in game.players:
                pop_agent = pos_to_agent[player.position]
                pop_agent.update_strategic_profile_from_game({
                    'final_score': player.get_score(),
                    'role': 'leaster',
                })
        else:
            for player in game.players:
                pop_agent = pos_to_agent[player.position]
                role = 'picker' if player.is_picker else ('partner' if player.is_partner else 'defender')
                pop_agent.update_strategic_profile_from_game({
                    'final_score': player.get_score(),
                    'role': role,
                })

        # Return final scores and picker seat
        return [player.get_score() for player in game.players], game.picker

    def _warm_profile_agents(self, partner_mode: int, min_games_per_agent: int = 10) -> None:
        """Play a small number of deterministic evaluation games to seed strategic profiles
        and calibrate ratings with coverage for every agent.

        Ensures each agent participates in at least `min_games_per_agent` games.
        """
        population = self._get_population(partner_mode)

        # Create a stable copy to iterate over
        agents_all = population.copy()

        # Track per-agent game counts
        played_counts: dict[str, int] = {a.metadata.agent_id: 0 for a in agents_all}

        # Helper to run one game for a group of 5 PopulationAgents
        def run_game_for_group(group: List[PopulationAgent]):
            game = Game(partner_selection_mode=partner_mode)
            positions = [1, 2, 3, 4, 5]
            final_scores, picker_seat = self._play_evaluation_game(game, group, positions)
            game_group: List[Tuple[PopulationAgent, int, float, int]] = []
            for idx, agent in enumerate(group):
                game_group.append((agent, positions[idx], final_scores[idx], picker_seat))
                played_counts[agent.metadata.agent_id] += 1
            self.update_ratings(game_group, partner_mode)

        # Scheduling strategy: sliding windows with staggered offset to cover all agents
        offset = 0
        max_cycles = max(1, (min_games_per_agent * len(agents_all) + 4) // 5)
        cycles = 0
        while True:
            if all(played_counts[aid] >= min_games_per_agent for aid in played_counts):
                break
            # Stagger offset so that tail agents are covered across cycles
            start = offset % 5
            i = start
            progressed = False
            while i <= len(agents_all) - 5:
                group = agents_all[i:i+5]
                # Only run if any agent in group still needs games
                if any(played_counts[a.metadata.agent_id] < min_games_per_agent for a in group):
                    run_game_for_group(group)
                    progressed = True
                i += 5

            # Handle tail (<5) by wrapping with earliest agents when needed
            tail_size = (len(agents_all) - start) % 5
            if tail_size and len(agents_all) >= 5:
                tail = agents_all[-tail_size:]
                wrap = agents_all[:5 - tail_size]
                group = tail + wrap
                if any(played_counts[a.metadata.agent_id] < min_games_per_agent for a in group):
                    run_game_for_group(group)
                    progressed = True

            offset += 1
            cycles += 1
            if not progressed and cycles > max_cycles:
                # Safety exit to avoid infinite loops in degenerate cases
                break

    def _save_agent(self, pop_agent: PopulationAgent):
        """Save agent to disk."""
        subdir = self._get_subdir(pop_agent.metadata.partner_mode)

        # Save model weights
        model_path = subdir / f"{pop_agent.metadata.agent_id}.pt"
        pop_agent.agent.save(str(model_path))

        # Save metadata
        metadata_path = subdir / f"{pop_agent.metadata.agent_id}_metadata.json"
        metadata_dict = pop_agent.metadata.to_dict()
        metadata_dict['rating_mu'] = float(pop_agent.rating.mu)
        metadata_dict['rating_sigma'] = float(pop_agent.rating.sigma)

        # Save strategic profile
        if hasattr(pop_agent, 'strategic_profile'):
            metadata_dict['strategic_profile'] = pop_agent.strategic_profile.to_dict()

        # Persist exploitation stats
        metadata_dict['exploitation_win_rate_ema'] = float(getattr(pop_agent, 'exploitation_win_rate_ema', 0.5))
        metadata_dict['exploitation_samples'] = int(getattr(pop_agent, 'exploitation_samples', 0))

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def _load_population(self):
        """Load existing population from disk."""
        for partner_mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
            subdir = self._get_subdir(partner_mode)

            if not subdir.exists():
                continue

            population = self._get_population(partner_mode)

            # Find all metadata files
            metadata_files = list(subdir.glob("*_metadata.json"))

            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata_dict = json.load(f)

                    # Extract rating if available
                    rating_mu = metadata_dict.pop('rating_mu', 25.0)
                    rating_sigma = metadata_dict.pop('rating_sigma', 25.0/3)
                    # Create rating using PlackettLuce model
                    rating = PlackettLuce().rating(mu=rating_mu, sigma=rating_sigma)

                    # Extract persisted non-AgentMetadata fields to restore later
                    persisted_exploitation_ema = metadata_dict.get('exploitation_win_rate_ema', None)
                    persisted_exploitation_n = metadata_dict.get('exploitation_samples', None)

                    # Extract strategic profile if available
                    strategic_profile_data = metadata_dict.pop('strategic_profile', None)
                    strategic_profile = None
                    if strategic_profile_data:
                        try:
                            strategic_profile = StrategicProfile.from_dict(strategic_profile_data)
                        except (TypeError, ValueError, KeyError) as err:
                            logging.warning("Failed to load strategic profile", extra={
                                "error": str(err),
                                "agent_id": metadata_dict.get('agent_id', 'unknown')
                            })

                    # Load metadata
                    # Remove non-AgentMetadata keys that we persist alongside metadata
                    metadata_dict.pop('exploitation_win_rate_ema', None)
                    metadata_dict.pop('exploitation_samples', None)
                    # Remove deprecated synergy keys if present
                    metadata_dict.pop('coop_synergy_ema', None)
                    metadata_dict.pop('coop_samples', None)
                    # Remove legacy strategic mirrors at top level if present
                    metadata_dict.pop('pick_hand_correlation', None)
                    metadata_dict.pop('trump_lead_rate', None)
                    metadata_dict.pop('bury_quality_rate', None)

                    metadata = AgentMetadata.from_dict(metadata_dict)

                    # Load agent model
                    model_path = subdir / f"{metadata.agent_id}.pt"
                    if not model_path.exists():
                        logging.warning("Model file missing for agent", extra={"agent_id": metadata.agent_id, "path": str(model_path)})
                        continue

                    # Create agent with appropriate parameters
                    agent = PPOAgent(len(ACTIONS), activation=metadata.activation)
                    agent.load(str(model_path), load_optimizers=False)

                    # Create population agent
                    pop_agent = PopulationAgent(agent, metadata, rating, strategic_profile)
                    # Restore exploitation stats if present
                    exp_ema = persisted_exploitation_ema
                    exp_n = persisted_exploitation_n
                    if exp_ema is not None:
                        try:
                            pop_agent.exploitation_win_rate_ema = float(exp_ema)
                        except (TypeError, ValueError):
                            pass
                    if exp_n is not None:
                        try:
                            pop_agent.exploitation_samples = int(exp_n)
                        except (TypeError, ValueError):
                            pass
                    population.append(pop_agent)

                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as err:
                    logging.error("Failed to load agent metadata", extra={
                        "error": str(err),
                        "metadata_file": str(metadata_file)
                    })
                    continue

            if population:
                print(f"📂 Loaded {len(population)} agents for {get_partner_mode_name(partner_mode)} population")

    def save_population_state(self):
        """Save current population state to disk."""
        for partner_mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
            population = self._get_population(partner_mode)
            for pop_agent in population:
                self._save_agent(pop_agent)

    def get_strongest_agent(self, partner_mode: int) -> Optional[PopulationAgent]:
        """Get the strongest agent in the population."""
        population = self._get_population(partner_mode)
        if not population:
            return None

        return max(population, key=lambda x: x.get_skill_estimate())

    def _compute_scaled_signatures(self, population: List[PopulationAgent]) -> np.ndarray:
        """Build and standardize strategic signatures for clustering/stats.

        - Uses StandardScaler to zero-mean/unit-variance features across the population
        - Clips to [-3, 3] to reduce outlier influence
        """
        signatures = np.array([agent.get_strategic_signature() for agent in population], dtype=np.float32)
        if signatures.size == 0:
            return signatures
        # Replace any nan/inf with zeros before scaling
        signatures = np.nan_to_num(signatures, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(signatures)
        scaled = np.clip(scaled, -3.0, 3.0)
        return scaled.astype(np.float32)

    def _cluster_population(self, partner_mode: int) -> tuple[np.ndarray, dict[int, list[PopulationAgent]]]:
        population = self._get_population(partner_mode)
        n = len(population)
        if n == 0:
            return np.array([]), {}

        # Fallback for tiny populations: skip clustering to avoid errors like
        # "n_samples=1 while HDBSCAN requires more than one sample" or
        # "min_samples (k) must be at most the number of samples in X (n)".
        if n < 3:
            labels = np.full((n,), -1, dtype=int)
            clusters: dict[int, list[PopulationAgent]] = defaultdict(list)
            for agent in population:
                clusters[-1].append(agent)
            return labels, clusters

        signatures = self._compute_scaled_signatures(population)
        # Ensure parameters respect sample count to prevent small-n errors
        min_param = max(2, min(3, signatures.shape[0]))
        clustering = HDBSCAN(min_cluster_size=min_param, min_samples=min_param, cluster_selection_epsilon=0.0).fit(signatures)
        labels = clustering.labels_
        clusters: dict[int, list[PopulationAgent]] = defaultdict(list)
        for agent, label in zip(population, labels):
            clusters[int(label)].append(agent)
        return labels, clusters

    def get_diversity_stats(self, partner_mode: int) -> Dict:
        """Get diversity statistics for the population."""
        population = self._get_population(partner_mode)

        if len(population) < 2:
            return {
                'avg_pairwise_diversity': 0.0,
                'diversity_spread': 0.0,
                'strategic_clusters': 0,
                'alone_rate_range': (0.0, 0.0),
                'pick_rate_diversity': {'weak': 0.0, 'medium': 0.0, 'strong': 0.0},
                'coverage': {'early_leads': 0.0, 'void_events': 0.0}
            }

        # Calculate pairwise diversity scores
        diversity_scores = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diversity = population[i].calculate_strategic_diversity(population[j])
                diversity_scores.append(diversity)

        # ALONE usage range (risk proxy)
        alone_rates = [ agent.strategic_profile.get_alone_rate() for agent in population ]

        # Pick rate diversity per bin (coefficient of variation)
        def cv(values: list[float]) -> float:
            m = np.mean(values)
            return float(np.std(values) / (m + 1e-8)) if values else 0.0
        pick_weak = [agent.strategic_profile.get_pick_rate_by_hand_strength('weak') for agent in population]
        pick_med = [agent.strategic_profile.get_pick_rate_by_hand_strength('medium') for agent in population]
        pick_str = [agent.strategic_profile.get_pick_rate_by_hand_strength('strong') for agent in population]
        pick_rate_diversity = {
            'weak': cv(pick_weak),
            'medium': cv(pick_med),
            'strong': cv(pick_str),
        }

        # Coverage: fraction of agents with sufficient early leads and void events
        EARLY_LEAD_MIN = 10
        VOID_MIN = 10
        early_lead_counts = [ (a.strategic_profile.early_lead_counts_picker + a.strategic_profile.early_lead_counts_defender) for a in population ]
        void_counts = [ a.strategic_profile.void_events for a in population ]
        coverage = {
            'early_leads': float(np.mean([1.0 if c >= EARLY_LEAD_MIN else 0.0 for c in early_lead_counts])),
            'void_events': float(np.mean([1.0 if c >= VOID_MIN else 0.0 for c in void_counts])),
        }

        # Estimate strategic clusters using standardized/clipped strategic signatures
        signatures = self._compute_scaled_signatures(population)
        min_param = max(2, min(3, signatures.shape[0]))
        clustering = HDBSCAN(min_cluster_size=min_param, min_samples=min_param, cluster_selection_epsilon=0.0).fit(signatures)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Cluster composition
        label_counts = defaultdict(int)
        for lb in labels:
            label_counts[int(lb)] += 1
        noise_share = float(label_counts.get(-1, 0)) / float(len(population)) if population else 0.0
        largest_cluster_size = max((cnt for lbl,cnt in label_counts.items() if lbl != -1), default=0)

        return {
            'avg_pairwise_diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
            'diversity_spread': np.std(diversity_scores) if diversity_scores else 0.0,
            'strategic_clusters': n_clusters,
            'alone_rate_range': (float(np.min(alone_rates)) if alone_rates else 0.0,
                                 float(np.max(alone_rates)) if alone_rates else 0.0),
            'pick_rate_diversity': pick_rate_diversity,
            'coverage': coverage,
            'cluster_label_counts': dict(label_counts),
            'noise_share': noise_share,
            'largest_cluster_size': int(largest_cluster_size),
            # EWMA population aggregates
            'ewma': {
                'trick_win_picker': float(np.mean([a.strategic_profile.trick_win_rate_picker for a in population])),
                'trick_win_partner': float(np.mean([a.strategic_profile.trick_win_rate_partner for a in population])),
                'trick_win_defender': float(np.mean([a.strategic_profile.trick_win_rate_defender for a in population])),
                'lead_win_picker': float(np.mean([a.strategic_profile.lead_win_rate_picker for a in population])),
                'lead_win_partner': float(np.mean([a.strategic_profile.lead_win_rate_partner for a in population])),
                'lead_win_defender': float(np.mean([a.strategic_profile.lead_win_rate_defender for a in population])),
                'early_trump_play': float(np.mean([a.strategic_profile.early_trump_play_rate for a in population])),
                'bury_high_points': float(np.mean([a.strategic_profile.bury_high_points_rate for a in population])),
                'schmear_when_void': float(np.mean([a.strategic_profile.schmear_rate_when_void for a in population])),
                'pick_hand_corr_mean': float(np.mean([a.strategic_profile.pick_hand_correlation for a in population])),
                'pick_hand_corr_std': float(np.std([a.strategic_profile.pick_hand_correlation for a in population])),
            },
            # Uncertainty (sigma) aggregates
            'sigma_avg': float(np.mean([a.rating.sigma for a in population])),
            'sigma_p90': float(np.percentile([a.rating.sigma for a in population], 90)),
        }

    def get_population_summary(self) -> str:
        """Get an enhanced summary string of the current population state."""
        jd_stats = self.get_population_stats(PARTNER_BY_JD)
        ca_stats = self.get_population_stats(PARTNER_BY_CALLED_ACE)
        jd_diversity = self.get_diversity_stats(PARTNER_BY_JD)
        ca_diversity = self.get_diversity_stats(PARTNER_BY_CALLED_ACE)

        # Cluster sampling share (last run counters)
        def cluster_sampling_str(mode: int) -> str:
            total = self._cluster_sampling_totals[mode]
            if total == 0:
                return ""
            parts = []
            for label, cnt in sorted(self._cluster_sampling_counts[mode].items(), key=lambda kv: kv[0]):
                share = (cnt / total) * 100.0
                parts.append(f"{label}:{share:.1f}%")
            return "  🎯 Cluster sampling: " + ", ".join(parts) + "\n"

        # Selection category sampling share (pressure/anchor/rarest/hardest/weighted)
        def selection_sampling_str(mode: int) -> str:
            total = self._selection_sampling_totals[mode]
            if total == 0:
                return ""
            parts = []
            for key in sorted(self._selection_sampling_counts[mode].keys()):
                cnt = self._selection_sampling_counts[mode][key]
                share = (cnt / total) * 100.0
                parts.append(f"{key}:{share:.1f}%")
            return "  🧭 Selection sampling: " + ", ".join(parts) + "\n"

        summary = "🏟️  PFSP Population Summary\n"
        summary += "=" * 65 + "\n"

        summary += "Jack-of-Diamonds Population:\n"
        summary += f"  📊 Size: {jd_stats['size']}/{self.max_population_jd}\n"
        summary += f"  🎯 Avg Skill: {jd_stats['avg_skill']:.1f} ± {(jd_stats['skill_range'][1] - jd_stats['skill_range'][0])/2:.1f}\n"
        summary += f"  🎮 Avg Games: {jd_stats['avg_games']}\n"
        summary += f"  🕐 Oldest Agent: {jd_stats['oldest_agent_days']:.1f} days\n"
        summary += f"  🎪 Strategic Diversity: {jd_diversity['avg_pairwise_diversity']:.3f}\n"
        summary += f"  🎭 Strategic Clusters: {jd_diversity['strategic_clusters']}\n"
        summary += f"  🎲 Alone Rate Range: {jd_diversity['alone_rate_range'][0]:.2f} - {jd_diversity['alone_rate_range'][1]:.2f}\n\n"
        # JD EWMA and cluster/uncertainty details
        jd_e = jd_diversity.get('ewma', {})
        summary += (
            f"  📈 EWMA trick-win (P/Pa/D): {jd_e.get('trick_win_picker',0):.3f} / {jd_e.get('trick_win_partner',0):.3f} / {jd_e.get('trick_win_defender',0):.3f}\n"
            f"  📈 EWMA lead-win (P/Pa/D): {jd_e.get('lead_win_picker',0):.3f} / {jd_e.get('lead_win_partner',0):.3f} / {jd_e.get('lead_win_defender',0):.3f}\n"
            f"  ⚡ Early trump: {jd_e.get('early_trump_play',0):.3f}  🪦 Bury≥10: {jd_e.get('bury_high_points',0):.3f}  💠 Schmear@void: {jd_e.get('schmear_when_void',0):.3f}\n"
            f"  🤝 Pick-hand corr: μ={jd_e.get('pick_hand_corr_mean',0):+.3f}, σ={jd_e.get('pick_hand_corr_std',0):.3f}\n"
            f"  🧩 Noise share: {jd_diversity.get('noise_share',0)*100:.1f}%  Largest cluster: {jd_diversity.get('largest_cluster_size',0)}\n"
            f"  σ (avg/p90): {jd_diversity.get('sigma_avg',0):.2f} / {jd_diversity.get('sigma_p90',0):.2f}\n\n"
        )
        # JD: Selection sampling shares
        sel_jd = selection_sampling_str(PARTNER_BY_JD)
        if sel_jd:
            summary += sel_jd

        summary += "Called-Ace Population:\n"
        summary += f"  📊 Size: {ca_stats['size']}/{self.max_population_called_ace}\n"
        summary += f"  🎯 Avg Skill: {ca_stats['avg_skill']:.1f} ± {(ca_stats['skill_range'][1] - ca_stats['skill_range'][0])/2:.1f}\n"
        summary += f"  🎮 Avg Games: {ca_stats['avg_games']}\n"
        summary += f"  🕐 Oldest Agent: {ca_stats['oldest_agent_days']:.1f} days\n"
        summary += f"  🎪 Strategic Diversity: {ca_diversity['avg_pairwise_diversity']:.3f}\n"
        summary += f"  🎭 Strategic Clusters: {ca_diversity['strategic_clusters']}\n"
        summary += f"  🎲 Alone Rate Range: {ca_diversity['alone_rate_range'][0]:.2f} - {ca_diversity['alone_rate_range'][1]:.2f}\n\n"
        # CA EWMA and cluster/uncertainty details
        ca_e = ca_diversity.get('ewma', {})
        summary += (
            f"  📈 EWMA trick-win (P/Pa/D): {ca_e.get('trick_win_picker',0):.3f} / {ca_e.get('trick_win_partner',0):.3f} / {ca_e.get('trick_win_defender',0):.3f}\n"
            f"  📈 EWMA lead-win (P/Pa/D): {ca_e.get('lead_win_picker',0):.3f} / {ca_e.get('lead_win_partner',0):.3f} / {ca_e.get('lead_win_defender',0):.3f}\n"
            f"  ⚡ Early trump: {ca_e.get('early_trump_play',0):.3f}  🪦 Bury≥10: {ca_e.get('bury_high_points',0):.3f}  💠 Schmear@void: {ca_e.get('schmear_when_void',0):.3f}\n"
            f"  🤝 Pick-hand corr: μ={ca_e.get('pick_hand_corr_mean',0):+.3f}, σ={ca_e.get('pick_hand_corr_std',0):.3f}\n"
            f"  🧩 Noise share: {ca_diversity.get('noise_share',0)*100:.1f}%  Largest cluster: {ca_diversity.get('largest_cluster_size',0)}\n"
            f"  σ (avg/p90): {ca_diversity.get('sigma_avg',0):.2f} / {ca_diversity.get('sigma_p90',0):.2f}\n"
        )
        # CA: Selection sampling shares
        sel_ca = selection_sampling_str(PARTNER_BY_CALLED_ACE)
        if sel_ca:
            summary += sel_ca
        # Append cluster sampling shares if available
        share_jd = cluster_sampling_str(PARTNER_BY_JD)
        if share_jd:
            summary += share_jd
        share_ca = cluster_sampling_str(PARTNER_BY_CALLED_ACE)
        if share_ca:
            summary += share_ca

        return summary

    def repopulate_from_checkpoints(self,
                                    checkpoint_patterns: List[str],
                                    activation: str = 'swish',
                                    clear_existing: bool = True,
                                    profile_games_per_agent: int = 10,
                                    max_checkpoints: Optional[int] = None) -> Dict[str, int]:
        """Repopulate populations from checkpoints using warm profiling + cluster anchoring.

        Steps:
        1) Discover checkpoints and load lightweight agents
        2) If clear_existing, wipe previous population files
        3) Add agents to both modes (bulk: no save/prune)
        4) Warm profile to seed strategic metrics and ratings for every agent
        5) Prune once per mode and persist population
        """
        def clear_population_and_files(partner_mode: int):
            population = self._get_population(partner_mode)
            population.clear()
            subdir = self._get_subdir(partner_mode)
            for f in subdir.glob("*.pt"):
                try:
                    f.unlink()
                except OSError:
                    pass
            for f in subdir.glob("*_metadata.json"):
                try:
                    f.unlink()
                except OSError:
                    pass

        # Discover checkpoints
        all_paths: List[str] = []
        for pattern in checkpoint_patterns:
            all_paths.extend(glob.glob(pattern))
        all_paths = [p for p in all_paths if os.path.isfile(p)]

        # Sort by modification time (newest first) and cap intake if requested
        all_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        selected_paths = all_paths[:max_checkpoints] if (max_checkpoints is not None and max_checkpoints > 0) else all_paths

        if not selected_paths:
            print("❌ No checkpoint files found for repopulation")
            return {"jd_added": 0, "called_ace_added": 0}

        # Optionally clear existing population and on-disk files
        if clear_existing:
            clear_population_and_files(PARTNER_BY_JD)
            clear_population_and_files(PARTNER_BY_CALLED_ACE)

        counts = {"jd_added": 0, "called_ace_added": 0}
        for path in selected_paths:
            try:
                agent = PPOAgent(len(ACTIONS), activation=activation)
                agent.load(path, load_optimizers=False)
            except (OSError, RuntimeError, ValueError) as err:
                logging.error("Failed to load checkpoint", extra={"error": str(err), "path": path})
                continue

            # Extract training_episodes from filename if possible
            try:
                filename = os.path.basename(path)
                if 'checkpoint_' in filename:
                    episode_str = filename.split('checkpoint_')[1].split('.')[0]
                    training_episodes = int(episode_str)
                else:
                    training_episodes = 0
            except (ValueError, IndexError):
                training_episodes = 0

            # Add to both partner modes (bulk add: no save/prune yet)
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                try:
                    self.add_agent(
                        agent=copy.deepcopy(agent),
                        partner_mode=mode,
                        training_episodes=training_episodes,
                        parent_id=None,
                        activation=activation,
                        save=False,
                        prune=False
                    )
                    if mode == PARTNER_BY_JD:
                        counts["jd_added"] += 1
                    else:
                        counts["called_ace_added"] += 1
                except (ValueError, OSError) as err:
                    logging.error("Failed to add agent to population", extra={
                        "error": str(err),
                        "mode": get_partner_mode_name(mode),
                        "checkpoint": os.path.basename(path)
                    })
                    continue

        # Warm profiling pass to seed strategic profiles and calibrate ratings
        try:
            self._warm_profile_agents(PARTNER_BY_JD, min_games_per_agent=profile_games_per_agent)
            self._warm_profile_agents(PARTNER_BY_CALLED_ACE, min_games_per_agent=profile_games_per_agent)
        except (RuntimeError, ValueError) as err:
            logging.warning("Warm profiling failed", extra={"error": str(err)})

        # Prune once per mode using cluster-aware logic (post-profile)
        self._manage_population_size(PARTNER_BY_JD)
        self._manage_population_size(PARTNER_BY_CALLED_ACE)

        # Persist all agents once at the end for efficiency
        self.save_population_state()

        print("🎯 Repopulation complete:")
        print(f"   JD agents added: {counts['jd_added']}")
        print(f"   Called-Ace agents added: {counts['called_ace_added']}")
        return counts


def create_initial_population_from_checkpoints(population: PFSPPopulation,
                                             checkpoint_patterns: List[str],
                                             activation: str = 'swish',
                                             max_agents_per_mode: int = 10) -> None:
    """Create initial population from existing training checkpoints.

    Args:
        population: PFSP population instance
        checkpoint_patterns: List of checkpoint file patterns/paths
        activation: Activation function used in the checkpoints
        max_agents_per_mode: Maximum agents to add per partner mode
    """
    import glob

    print("🚀 Creating initial population from checkpoints...")

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))

    if not all_checkpoints:
        print("❌ No checkpoint files found matching the patterns")
        return

    # Sort checkpoints by modification time (newest first)
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Take a diverse sample of checkpoints
    selected_checkpoints = all_checkpoints[:max_agents_per_mode * 2]  # *2 because we'll create both modes

    print(f"📊 Found {len(all_checkpoints)} checkpoints, using {len(selected_checkpoints)} best ones")

    agents_added = {PARTNER_BY_JD: 0, PARTNER_BY_CALLED_ACE: 0}

    for checkpoint_path in selected_checkpoints:
        # Extract episode number from filename if possible
        try:
            filename = os.path.basename(checkpoint_path)
            if 'checkpoint_' in filename:
                episode_str = filename.split('checkpoint_')[1].split('.')[0]
                training_episodes = int(episode_str)
            else:
                training_episodes = 0
        except (ValueError, IndexError):
            training_episodes = 0

        # Create agents for both partner modes
        for partner_mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
            if agents_added[partner_mode] >= max_agents_per_mode:
                continue

            try:
                # Load checkpoint
                agent = PPOAgent(len(ACTIONS), activation=activation)
                agent.load(checkpoint_path, load_optimizers=False)

                # Add to population
                population.add_agent(
                    agent=agent,
                    partner_mode=partner_mode,
                    training_episodes=training_episodes,
                    parent_id=None,
                    activation=activation
                )

                agents_added[partner_mode] += 1
                mode_name = get_partner_mode_name(partner_mode)
                print(f"   ✅ Added {os.path.basename(checkpoint_path)} as {mode_name} agent")

            except (OSError, RuntimeError, ValueError) as err:
                # Emit a clear console message and a diagnostic log entry, then continue.
                try:
                    basename = os.path.basename(checkpoint_path)
                except Exception:
                    basename = str(checkpoint_path)
                print(f"   ❌ Skipped {basename} as {get_partner_mode_name(partner_mode)} agent: {err}")
                logging.warning(
                    f"Failed to load checkpoint '{checkpoint_path}' for mode '{get_partner_mode_name(partner_mode)}': {err}"
                )
                continue

    print("🎉 Successfully created initial population:")
    print(f"   Jack-of-Diamonds agents: {agents_added[PARTNER_BY_JD]}")
    print(f"   Called-Ace agents: {agents_added[PARTNER_BY_CALLED_ACE]}")


def _pfsp_cli_main():

    parser = argparse.ArgumentParser(description="PFSP population utilities")
    subparsers = parser.add_subparsers(dest="command")

    # Repopulate command
    rep = subparsers.add_parser("repopulate", help="Repopulate PFSP populations from checkpoints using initial tournament + cluster anchors")
    rep.add_argument("--patterns", nargs="+", required=True,
                     help="Glob patterns to checkpoint files, e.g. 'pfsp_checkpoints_swish/*.pt'")
    rep.add_argument("--activation", type=str, default="swish",
                     help="Activation used when instantiating agents (default: swish)")
    rep.add_argument("--no-clear", action="store_true",
                     help="Do not clear existing population/files before repopulating")
    rep.add_argument("--population-dir", type=str, default="pfsp_population",
                     help="Population directory (default: pfsp_population)")
    rep.add_argument("--jd-max", type=int, default=75,
                     help="Max JD population size (default: 75)")
    rep.add_argument("--called-max", type=int, default=75,
                     help="Max Called-Ace population size (default: 75)")
    rep.add_argument("--anchor-quota-per-cluster", type=int, default=2,
                     help="Anchors per cluster to retain (default: 2)")
    rep.add_argument("--profile-games-per-agent", type=int, default=20,
                     help="Minimum deterministic profiling games per agent to seed strategic metrics (default: 20)")
    rep.add_argument("--max-checkpoints", type=int, default=None,
                     help="Cap the number of checkpoints to load (newest first). Default: no cap")

    args = parser.parse_args()

    if args.command == "repopulate":
        pop = PFSPPopulation(
            max_population_jd=args.jd_max,
            max_population_called_ace=args.called_max,
            population_dir=args.population_dir,
            anchor_quota_per_cluster=args.anchor_quota_per_cluster,
        )

        _ = pop.repopulate_from_checkpoints(
            checkpoint_patterns=args.patterns,
            activation=args.activation,
            clear_existing=(not args.no_clear),
            profile_games_per_agent=args.profile_games_per_agent,
            max_checkpoints=args.max_checkpoints,
        )

        print(pop.get_population_summary())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_pfsp_cli_main())
