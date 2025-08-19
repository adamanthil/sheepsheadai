#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) Population Management System for Sheepshead.

This module implements a population-based training system inspired by AlphaStar and OpenAI Five,
using OpenSkill ratings for opponent selection and maintaining diverse agent populations.
"""

import numpy as np
import random
import os
import json
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from openskill.models import PlackettLuce

from ppo import PPOAgent
from sheepshead import (
    Game,
    PARTNER_BY_JD,
    PARTNER_BY_CALLED_ACE,
    TRUMP,
    ACTION_LOOKUP,
    UNDER_TOKEN,
    get_card_points,
    filter_by_suit,
)
from training_utils import estimate_hand_strength_category




@dataclass
class StrategicProfile:
    """Tracks behavioral patterns and strategic tendencies of an agent."""

    # Pick decision patterns
    pick_decisions_by_hand_strength: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))
    pick_rate_by_position: Dict[int, List[bool]] = field(default_factory=lambda: defaultdict(list))

    # Trump leading patterns
    trump_leads_as_picker: List[bool] = field(default_factory=list)
    trump_leads_as_defender: List[bool] = field(default_factory=list)

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

    # Bury decision patterns
    bury_decisions: List[Tuple[str, int]] = field(default_factory=list)  # (card, points_value)

    # Risk tolerance indicators
    alone_calls: List[bool] = field(default_factory=list)
    risky_plays: List[bool] = field(default_factory=list)

    # Partner selection patterns (for called-ace mode)
    called_cards: List[str] = field(default_factory=list)
    under_calls: List[bool] = field(default_factory=list)

    # Performance in different scenarios
    performance_as_picker: List[float] = field(default_factory=list)
    performance_as_partner: List[float] = field(default_factory=list)
    performance_as_defender: List[float] = field(default_factory=list)
    performance_in_leaster: List[float] = field(default_factory=list)

    def get_pick_rate_by_hand_strength(self, strength_category: str) -> float:
        """Get pick rate for a specific hand strength category."""
        decisions = self.pick_decisions_by_hand_strength.get(strength_category, [])
        if not decisions:
            return 0.5  # Default neutral rate
        return sum(decisions) / len(decisions)

    def get_trump_lead_rate(self, role: str) -> float:
        """Get trump leading rate for picker or defender role."""
        if role == 'picker':
            total = self.lead_counts_picker
            trump = self.lead_trump_counts_picker
            legacy = self.trump_leads_as_picker
        elif role == 'defender':
            total = self.lead_counts_defender
            trump = self.lead_trump_counts_defender
            legacy = self.trump_leads_as_defender
        else:
            return 0.0

        if total > 0:
            return trump / max(1, total)
        if legacy:
            return sum(legacy) / len(legacy)
        return 0.0

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
        if not self.bury_decisions:
            return 0.5  # Neutral score

        # Good bury: average point value of buried cards should be low
        avg_points = sum(points for _, points in self.bury_decisions) / len(self.bury_decisions)
        # Normalize to 0-1 scale (assuming max card value is ~11 points)
        return max(0.0, 1.0 - (avg_points / 11.0))

    def get_void_trump_rate(self) -> float:
        return (self.void_trump_events / self.void_events) if self.void_events > 0 else 0.0

    def get_avg_schmeared_points(self) -> float:
        return (self.schmeared_points_sum / self.void_fail_events) if self.void_fail_events > 0 else 0.0

    def get_risk_tolerance(self) -> float:
        """Calculate risk tolerance score (0 = conservative, 1 = aggressive)."""
        risk_indicators = []

        if self.alone_calls:
            risk_indicators.append(sum(self.alone_calls) / len(self.alone_calls))

        if self.risky_plays:
            risk_indicators.append(sum(self.risky_plays) / len(self.risky_plays))

        if self.under_calls:
            risk_indicators.append(sum(self.under_calls) / len(self.under_calls))

        if not risk_indicators:
            return 0.5  # Neutral

        return sum(risk_indicators) / len(risk_indicators)

    def get_role_performance(self, role: str) -> float:
        """Get average performance in a specific role."""
        if role == 'picker':
            scores = self.performance_as_picker
        elif role == 'partner':
            scores = self.performance_as_partner
        elif role == 'defender':
            scores = self.performance_as_defender
        elif role == 'leaster':
            scores = self.performance_in_leaster
        else:
            return 0.0

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pick_decisions_by_hand_strength': dict(self.pick_decisions_by_hand_strength),
            'pick_rate_by_position': dict(self.pick_rate_by_position),
            'trump_leads_as_picker': self.trump_leads_as_picker[-50:],  # Keep recent data
            'trump_leads_as_defender': self.trump_leads_as_defender[-50:],
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
            'bury_decisions': self.bury_decisions[-20:],
            'alone_calls': self.alone_calls[-30:],
            'risky_plays': self.risky_plays[-50:],
            'called_cards': self.called_cards[-20:],
            'under_calls': self.under_calls[-20:],
            'performance_as_picker': self.performance_as_picker[-30:],
            'performance_as_partner': self.performance_as_partner[-30:],
            'performance_as_defender': self.performance_as_defender[-30:],
            'performance_in_leaster': self.performance_in_leaster[-20:]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategicProfile':
        """Create from dictionary."""
        profile = cls()
        profile.pick_decisions_by_hand_strength = defaultdict(list, data.get('pick_decisions_by_hand_strength', {}))
        profile.pick_rate_by_position = defaultdict(list, data.get('pick_rate_by_position', {}))
        profile.trump_leads_as_picker = data.get('trump_leads_as_picker', [])
        profile.trump_leads_as_defender = data.get('trump_leads_as_defender', [])
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
        profile.bury_decisions = [tuple(item) if isinstance(item, list) else item
                                for item in data.get('bury_decisions', [])]
        profile.alone_calls = data.get('alone_calls', [])
        profile.risky_plays = data.get('risky_plays', [])
        profile.called_cards = data.get('called_cards', [])
        profile.under_calls = data.get('under_calls', [])
        profile.performance_as_picker = data.get('performance_as_picker', [])
        profile.performance_as_partner = data.get('performance_as_partner', [])
        profile.performance_as_defender = data.get('performance_as_defender', [])
        profile.performance_in_leaster = data.get('performance_in_leaster', [])
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

    # Strategic metrics (when available)
    pick_hand_correlation: Optional[float] = None
    trump_lead_rate: Optional[float] = None
    bury_quality_rate: Optional[float] = None

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
        self.recent_decisions = {
            'pick_decisions': deque(maxlen=50),
            'trump_leads': deque(maxlen=50),
            'bury_choices': deque(maxlen=20)
        }

        # Track recent opponents beaten for exploitation sampling
        self.recent_victories = deque(maxlen=50)  # IDs of agents this agent recently beat

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
        risk1 = np.mean(self.strategic_profile.alone_calls) if self.strategic_profile.alone_calls else 0.0
        risk2 = np.mean(other.strategic_profile.alone_calls) if other.strategic_profile.alone_calls else 0.0
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
        alone_rate = np.mean(self.strategic_profile.alone_calls) if self.strategic_profile.alone_calls else 0.0
        signature.append(alone_rate)

        # Bury quality
        signature.append(self.strategic_profile.get_bury_quality_score())

        # Role performances (optional)
        for role in ['picker', 'partner', 'defender', 'leaster']:
            signature.append(self.strategic_profile.get_role_performance(role))

        return np.array(signature, dtype=np.float32)

    def update_strategic_profile_from_game(self, game_data: Dict):
        """Update strategic profile based on game observations."""
        if 'pick_decision' in game_data:
            hand_strength = game_data.get('hand_strength_category', 'medium')
            picked = game_data['pick_decision']
            position = game_data.get('position', 1)

            self.strategic_profile.pick_decisions_by_hand_strength[hand_strength].append(picked)
            self.strategic_profile.pick_rate_by_position[position].append(picked)

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
                self.strategic_profile.trump_leads_as_picker.append(led_trump)
            elif role in ['defender']:
                self.strategic_profile.lead_counts_defender += 1
                if led_trump:
                    self.strategic_profile.lead_trump_counts_defender += 1
                if is_early:
                    self.strategic_profile.early_lead_counts_defender += 1
                    if led_trump:
                        self.strategic_profile.early_lead_trump_counts_defender += 1
                self.strategic_profile.trump_leads_as_defender.append(led_trump)

        if game_data.get('void_event', False):
            self.strategic_profile.void_events += 1
            if game_data.get('void_played_trump', False):
                self.strategic_profile.void_trump_events += 1
            else:
                self.strategic_profile.void_fail_events += 1
                self.strategic_profile.schmeared_points_sum += int(game_data.get('schmeared_points', 0))

        if 'bury_decision' in game_data:
            card = game_data['bury_decision']
            points = game_data.get('card_points', 0)
            self.strategic_profile.bury_decisions.append((card, points))

        if 'alone_call' in game_data:
            self.strategic_profile.alone_calls.append(game_data['alone_call'])

        if 'under_call' in game_data:
            self.strategic_profile.under_calls.append(game_data['under_call'])

        if 'called_card' in game_data:
            self.strategic_profile.called_cards.append(game_data['called_card'])

        if 'risky_play' in game_data:
            self.strategic_profile.risky_plays.append(game_data['risky_play'])

        # Performance tracking
        if 'final_score' in game_data and 'role' in game_data:
            score = game_data['final_score']
            role = game_data['role']

            if role == 'picker':
                self.strategic_profile.performance_as_picker.append(score)
            elif role == 'partner':
                self.strategic_profile.performance_as_partner.append(score)
            elif role == 'defender':
                self.strategic_profile.performance_as_defender.append(score)
            elif role == 'leaster':
                self.strategic_profile.performance_in_leaster.append(score)

    def add_victory_against(self, opponent_id: str):
        """Record that this agent beat another agent."""
        self.recent_victories.append(opponent_id)

    def get_exploitation_score_against(self, training_agent_id: str) -> float:
        """Get exploitation score - how often this agent beats the training agent."""
        if not self.recent_victories:
            return 0.0

        victories_against_training = sum(1 for opp_id in self.recent_victories
                                       if opp_id == training_agent_id)
        return victories_against_training / len(self.recent_victories)


class PFSPPopulation:
    """Manages the population of agents for PFSP training."""

    def __init__(self,
                 max_population_jd: int = 25,
                 max_population_called_ace: int = 25,
                 population_dir: str = "pfsp_population"):

        self.max_population_jd = max_population_jd
        self.max_population_called_ace = max_population_called_ace
        self.population_dir = Path(population_dir)
        self.rating_model = PlackettLuce()

        # Separate populations for different partner modes
        self.jd_population: List[PopulationAgent] = []
        self.called_ace_population: List[PopulationAgent] = []

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

    def add_agent(self,
                  agent: PPOAgent,
                  partner_mode: int,
                  training_episodes: int,
                  parent_id: Optional[str] = None,
                  activation: str = 'swish') -> str:
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
        pop_agent = PopulationAgent(agent, metadata)

        # Add to appropriate population
        population = self._get_population(partner_mode)
        population.append(pop_agent)

        # Manage population size
        self._manage_population_size(partner_mode)

        # Save the new agent
        self._save_agent(pop_agent)

        print(f"✅ Added agent {agent_id} to {self._get_partner_mode_name(partner_mode)} population")
        print(f"   Population size: {len(population)}/{self._get_max_population(partner_mode)}")

        return agent_id

    def _get_partner_mode_name(self, partner_mode: int) -> str:
        """Get human-readable name for partner mode."""
        return "Jack-of-Diamonds" if partner_mode == PARTNER_BY_JD else "Called-Ace"

    def _manage_population_size(self, partner_mode: int):
        """Remove agents if population exceeds max size, preserving diversity and strength."""
        population = self._get_population(partner_mode)
        max_size = self._get_max_population(partner_mode)

        # First, prune agents with very low uncertainty (σ) – they are fully explored
        LOW_SIGMA_THRESHOLD = 2.5
        if len(population) > max_size:
            low_sigma_agents = [agent for agent in population if agent.get_skill_uncertainty() < LOW_SIGMA_THRESHOLD]
            # Remove oldest among low-sigma first
            low_sigma_agents.sort(key=lambda a: a.metadata.creation_time)
            for agent in low_sigma_agents:
                if len(population) <= max_size:
                    break
                population.remove(agent)
                self._delete_agent_files(agent)
                print(f"🗑️  Pruned low-σ agent {agent.metadata.agent_id} (μ={agent.get_skill_estimate():.1f}, σ={agent.get_skill_uncertainty():.1f})")

        if len(population) <= max_size:
            return

        # Use sophisticated diversity-aware removal strategy
        agents_to_keep = self._select_diverse_population_subset(population, max_size)
        agents_to_remove_list = [agent for agent in population if agent not in agents_to_keep]

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
                uncertainty_bonus = candidate.get_skill_uncertainty() / 10.0  # Prefer uncertain agents (more potential)

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
        agent_file = subdir / f"{pop_agent.metadata.agent_id}.pth"
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
                        uniform_mix: float = 0.15) -> List[PopulationAgent]:
        """Sample opponents using multi-objective selection (skill + diversity + exploitation + curriculum)."""
        population = self._get_population(partner_mode)

        if len(population) == 0:
            return []

        if len(population) <= n_opponents:
            return population.copy()

        # Track already selected opponents to ensure diversity
        already_selected = selected_opponents or []

        selected_agents = []
        remaining_population = population.copy()

        for _ in range(n_opponents):
            if not remaining_population:
                break

            weights = []
            for pop_agent in remaining_population:
                weight_components = {}

                # 1. Skill-based weight (prefer similar skill levels) factoring in uncertainty
                agent_mu = pop_agent.get_skill_estimate()
                agent_sigma = pop_agent.get_skill_uncertainty()
                skill_diff = abs(agent_mu - training_agent_skill)
                # Higher uncertainty reduces the weight
                weight_components['skill'] = np.exp(-(skill_diff + agent_sigma) / 10.0)

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
                    exploitation_score = pop_agent.get_exploitation_score_against(training_agent_id)
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

        # Convert scores to rankings (lower rank = better performance)
        scores = [result[2] for result in game_results]

        # Sort by score (descending) and assign ranks
        score_rank_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranks = [0] * len(scores)
        current_rank = 0
        last_score = None

        for i, (original_idx, score) in enumerate(score_rank_pairs):
            if last_score is None or score != last_score:
                current_rank = i
            ranks[original_idx] = current_rank
            last_score = score

        # Update ratings using OpenSkill
        try:
            new_teams = self.rating_model.rate(teams, ranks=ranks)

            # Update agent ratings
            for i, (agent, position, score, picker_seat) in enumerate(game_results):
                agent.update_rating(new_teams[i][0])
                # Also update performance tracking
                was_picker = (position == picker_seat)
                agent.add_game_result(score, was_picker)

        except Exception as e:
            print(f"Warning: Failed to update ratings: {e}")

    def run_cross_evaluation(self, partner_mode: int, num_games: int, max_agents: int) -> Dict:
        """Run cross-evaluation tournament to update ratings."""
        population = self._get_population(partner_mode)

        if len(population) < 2:
            print(f"Skipping cross-evaluation for {self._get_partner_mode_name(partner_mode)} - insufficient agents")
            return {}

        # Limit to most recent/strongest agents to keep evaluation manageable
        if len(population) > max_agents:
            # Sort by skill and recency, take top agents
            population = sorted(
                population,
                key=lambda x: x.get_skill_estimate() + (x.metadata.creation_time - time.time()) / (24 * 3600 * 30),
                reverse=True
            )[:max_agents]

        print(f"🏆 Running cross-evaluation for {self._get_partner_mode_name(partner_mode)} population")
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
                    state = player.get_state_vector()
                    action, _, _ = agent.agent.act(state, valid_actions, player.position, deterministic=True)
                    player.act(action)
                    valid_actions = player.get_valid_action_ids()

                    # Propagate observations to all agents at the end of the trick
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            seat_agent = pos_to_agent[seat.position]
                            # Completed trick observation
                            seat_agent.agent.observe(
                                seat.get_last_trick_state_vector(),
                                player_id=seat.position,
                            )

                    # Profile strategic events
                    action_name = ACTION_LOOKUP[action]
                    role = 'picker' if (player.is_picker or player.is_partner) else 'defender'
                    pop_agent = pos_to_agent[player.position]

                    if action_name in ("PICK", "PASS"):
                        pop_agent.update_strategic_profile_from_game({
                            'pick_decision': (action_name == "PICK"),
                            'hand_strength_category': hand_strength_by_pos[player.position],
                            'position': player.position,
                        })

                    if action_name.startswith("BURY "):
                        card = action_name.split()[-1]
                        pop_agent.update_strategic_profile_from_game({
                            'bury_decision': card,
                            'card_points': get_card_points(card),
                        })

                    if action_name == "ALONE":
                        pop_agent.update_strategic_profile_from_game({'alone_call': True})
                    elif action_name.startswith("CALL "):
                        pop_agent.update_strategic_profile_from_game({'alone_call': False, 'called_card': action_name.split()[1]})

                    if action_name.startswith("PLAY "):
                        card = action_name.split()[-1]
                        is_lead = (game.cards_played == 0)
                        is_early = is_lead and (game.current_trick <= 1)
                        if is_lead:
                            pop_agent.update_strategic_profile_from_game({
                                'trump_lead': (card in TRUMP or card == UNDER_TOKEN),
                                'role': role,
                                'is_early_lead': is_early,
                            })

                        if not is_lead and game.current_suit:
                            can_follow = len(filter_by_suit(player.hand, game.current_suit)) > 0
                            if not can_follow:
                                played_trump = (card in TRUMP or card == UNDER_TOKEN)
                                pop_agent.update_strategic_profile_from_game({
                                    'void_event': True,
                                    'void_played_trump': played_trump,
                                    'schmeared_points': 0 if played_trump else get_card_points(card),
                                })

        # Return final scores and picker seat
        return [player.get_score() for player in game.players], game.picker

    def _save_agent(self, pop_agent: PopulationAgent):
        """Save agent to disk."""
        subdir = self._get_subdir(pop_agent.metadata.partner_mode)

        # Save model weights
        model_path = subdir / f"{pop_agent.metadata.agent_id}.pth"
        pop_agent.agent.save(str(model_path))

        # Save metadata
        metadata_path = subdir / f"{pop_agent.metadata.agent_id}_metadata.json"
        metadata_dict = pop_agent.metadata.to_dict()
        metadata_dict['rating_mu'] = float(pop_agent.rating.mu)
        metadata_dict['rating_sigma'] = float(pop_agent.rating.sigma)

        # Save strategic profile
        if hasattr(pop_agent, 'strategic_profile'):
            metadata_dict['strategic_profile'] = pop_agent.strategic_profile.to_dict()

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

                    # Extract strategic profile if available
                    strategic_profile_data = metadata_dict.pop('strategic_profile', None)
                    strategic_profile = None
                    if strategic_profile_data:
                        try:
                            strategic_profile = StrategicProfile.from_dict(strategic_profile_data)
                        except Exception as e:
                            print(f"Warning: Failed to load strategic profile for {metadata_dict.get('agent_id', 'unknown')}: {e}")

                    # Load metadata
                    metadata = AgentMetadata.from_dict(metadata_dict)

                    # Load agent model
                    model_path = subdir / f"{metadata.agent_id}.pth"
                    if not model_path.exists():
                        print(f"Warning: Model file missing for {metadata.agent_id}")
                        continue

                    # Create agent with appropriate parameters
                    from sheepshead import STATE_SIZE, ACTIONS
                    agent = PPOAgent(STATE_SIZE, len(ACTIONS), activation=metadata.activation)
                    agent.load(str(model_path))

                    # Create population agent
                    pop_agent = PopulationAgent(agent, metadata, rating, strategic_profile)
                    population.append(pop_agent)

                except Exception as e:
                    print(f"Warning: Failed to load agent from {metadata_file}: {e}")
                    continue

            if population:
                print(f"📂 Loaded {len(population)} agents for {self._get_partner_mode_name(partner_mode)} population")

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
        alone_rates = [ (np.mean(agent.strategic_profile.alone_calls) if agent.strategic_profile.alone_calls else 0.0) for agent in population ]

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

        # Estimate strategic clusters using strategic signatures
        signatures = np.array([agent.get_strategic_signature() for agent in population])
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(signatures)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        except ImportError:
            # Fallback: simple heuristic based on diversity scores
            n_clusters = max(1, len(population) // 3)  # Rough estimate

        return {
            'avg_pairwise_diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
            'diversity_spread': np.std(diversity_scores) if diversity_scores else 0.0,
            'strategic_clusters': n_clusters,
            'alone_rate_range': (float(np.min(alone_rates)) if alone_rates else 0.0,
                                 float(np.max(alone_rates)) if alone_rates else 0.0),
            'pick_rate_diversity': pick_rate_diversity,
            'coverage': coverage,
        }

    def get_population_summary(self) -> str:
        """Get an enhanced summary string of the current population state."""
        jd_stats = self.get_population_stats(PARTNER_BY_JD)
        ca_stats = self.get_population_stats(PARTNER_BY_CALLED_ACE)
        jd_diversity = self.get_diversity_stats(PARTNER_BY_JD)
        ca_diversity = self.get_diversity_stats(PARTNER_BY_CALLED_ACE)

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

        summary += "Called-Ace Population:\n"
        summary += f"  📊 Size: {ca_stats['size']}/{self.max_population_called_ace}\n"
        summary += f"  🎯 Avg Skill: {ca_stats['avg_skill']:.1f} ± {(ca_stats['skill_range'][1] - ca_stats['skill_range'][0])/2:.1f}\n"
        summary += f"  🎮 Avg Games: {ca_stats['avg_games']}\n"
        summary += f"  🕐 Oldest Agent: {ca_stats['oldest_agent_days']:.1f} days\n"
        summary += f"  🎪 Strategic Diversity: {ca_diversity['avg_pairwise_diversity']:.3f}\n"
        summary += f"  🎭 Strategic Clusters: {ca_diversity['strategic_clusters']}\n"
        summary += f"  🎲 Alone Rate Range: {ca_diversity['alone_rate_range'][0]:.2f} - {ca_diversity['alone_rate_range'][1]:.2f}\n"

        return summary


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
    from sheepshead import STATE_SIZE, ACTIONS
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
                agent = PPOAgent(STATE_SIZE, len(ACTIONS), activation=activation)
                agent.load(checkpoint_path)

                # Add to population
                population.add_agent(
                    agent=agent,
                    partner_mode=partner_mode,
                    training_episodes=training_episodes,
                    parent_id=None,
                    activation=activation
                )

                agents_added[partner_mode] += 1
                mode_name = population._get_partner_mode_name(partner_mode)
                print(f"   ✅ Added {os.path.basename(checkpoint_path)} as {mode_name} agent")

            except Exception as e:
                print(f"   ❌ Failed to load {checkpoint_path}: {e}")
                continue

    print("🎉 Successfully created initial population:")
    print(f"   Jack-of-Diamonds agents: {agents_added[PARTNER_BY_JD]}")
    print(f"   Called-Ace agents: {agents_added[PARTNER_BY_CALLED_ACE]}")