#!/usr/bin/env python3
"""
Rigorous strength comparison for Sheepshead RL agents.

Why this exists (vs. a round-robin tournament)
----------------------------------------------
A Sheepshead hand is *exactly zero-sum* across the 5 seats (picker 2x, partner
1x, three defenders -1x per multiplier; leaster +4/-1x4; alone 4x/-1x4 -- every
hand sums to 0; see Player.get_score in sheepshead.py). Consequences:

  * "Average score per hand" has no absolute meaning -- it is defined only
    relative to whoever fills the other seats. Measuring each snapshot against a
    rotating pool of all the *other* snapshots gives a number that shifts every
    time the pool changes and is mechanically pinned near 0.
  * Teammate composition (dynamic 2-v-3 teams) is a large, uncontrolled
    confounder when every seat is a different model.
  * Policies act deterministically, so the unit of independent randomness is the
    *deal*, not the *game*. Counting rotations as samples inflates N.

This tool instead runs a controlled experiment:

  Anchored gauntlet ("hero in a reference field")
  -----------------------------------------------
  A candidate C occupies one seat; a FIXED frozen reference field (a panel of
  one or more anchor models) fills the other four. Because the field is held
  constant across all candidates, the resulting points are directly comparable,
  reproducible, and in true score-per-hand units.

  Common random numbers + duplicate-bridge replay
  ------------------------------------------------
  For each deal we seat C in ALL FIVE seats in turn (the reference fills the
  rest), so C plays all five hands of that deal and positional / pick-eligibility
  luck is averaged out *within* the deal. The per-deal mean score m(C, d) is the
  independent observation. Every candidate is evaluated on the IDENTICAL set of
  deals and the IDENTICAL reference seat-assignments, so candidate-vs-candidate
  comparisons are *paired*: the deal luck (which dominates total variance)
  cancels deal-by-deal.

  Honest uncertainty
  -------------------
  Confidence intervals come from a block bootstrap that resamples *deals* (the
  independent unit), not games -- correctly accounting for the 5 correlated
  seatings that share a deal, and robust to the heavy-tailed score distribution
  (multipliers 1/2/3, doubled on the bump). Pairwise comparisons use the paired
  per-deal differences.

Metrics
-------
  * PRIMARY: mean game *score* (zero-sum payoff) per hand vs the reference field.
    Consistently signed across all hand types (higher always better, including
    leasters), so it is a valid cross-hand strength measure.
  * TIEBREAKER: a direction-corrected card-points margin, since raw card points
    are NOT comparable across hand types (in a leaster you want fewer points and
    the 120 are split 5 ways). Keyed to the agent's role on the hand:
        picking team -> picker_points - 60
        defending    -> defender_points - 60
        leaster      -> -points_taken           (fewer is better)

  Secondary (optional) round-robin cross-check fits a Massey-style linear rating
  to the antisymmetric head-to-head advantage matrix to catch strategic
  non-transitivity that an anchored gauntlet alone could miss.

Example
-------
  uv run python analysis/rigorous_eval.py \
    --input-dir checkpoints_swish \
    --episode-divisor 1000000 \
    --anchors final_pfsp_swish_ppo.pt \
    --deals 1000 \
    --partner-mode called \
    --out-csv rigorous_results.csv \
    --out-plot rigorous_strength.png
"""

from __future__ import annotations

import sys

# Repo-root imports work regardless of invocation directory.

import argparse
import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead import (
    Game,
    PARTNER_BY_JD,
    PARTNER_BY_CALLED_ACE,
)

NUM_SEATS = 5


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    """A loaded snapshot: the network plus identity/metadata."""

    model_id: str
    filepath: Path
    episodes: Optional[int]
    agent: PPOAgent


class ModelRegistry:
    """Loads each distinct .pt file exactly once.

    A single PPOAgent instance can safely fill several seats in one game because
    its recurrent memory is keyed by player_id (ppo.PPOAgent._player_memories),
    so distinct seats never share state. A convention-wrapped entrant (``wrap``,
    design E4) shares the raw instance for the same reason: the wrapper only
    filters valid actions per call and holds no per-seat state itself.
    """

    def __init__(self) -> None:
        self._by_path: Dict[Path, Model] = {}
        self._wrapped: Dict[tuple, Model] = {}

    def get(
        self, path: Path, episodes: Optional[int] = None, wrap: Optional[str] = None
    ) -> Model:
        path = path.resolve()
        if path not in self._by_path:
            # Arch-aware: builds whatever architecture the checkpoint records
            # (legacy checkpoints without the key are the full architecture).
            agent = load_agent(str(path))
            self._by_path[path] = Model(
                model_id=path.stem, filepath=path, episodes=episodes, agent=agent
            )
        if wrap is None:
            return self._by_path[path]
        key = (path, wrap)
        if key not in self._wrapped:
            from sheepshead.agent.convention_wrapper import wrap_agent

            self._wrapped[key] = Model(
                model_id=f"{path.stem}@{wrap}",
                filepath=path,
                episodes=episodes,
                agent=wrap_agent(self._by_path[path].agent, wrap),
            )
        return self._wrapped[key]


def extract_episodes_from_name(path: Path) -> Optional[int]:
    """Extract an episode count from a snapshot filename, or None."""
    name = path.name
    m = re.search(r"checkpoint_(\d+)", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)(?:\.pt)$", name)
    if m2:
        return int(m2.group(1))
    return None


def discover_candidates(
    input_dir: Path, episode_divisor: int
) -> List[Tuple[Path, Optional[int]]]:
    """Find candidate snapshots, optionally filtered to episode markers.

    episode_divisor <= 1 keeps every .pt file (episodes may be None).
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    out: List[Tuple[Path, Optional[int]]] = []
    for p in sorted(input_dir.rglob("*.pt")):
        if not p.is_file():
            continue
        eps = extract_episodes_from_name(p)
        if episode_divisor > 1:
            if eps is None or eps % episode_divisor != 0:
                continue
        out.append((p, eps))
    out.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0, str(t[0])))
    return out


# --------------------------------------------------------------------------- #
# Passive decision probe (opt-in telemetry; never alters play)
# --------------------------------------------------------------------------- #
@dataclass
class DecisionProbe:
    """Observe selected hero decisions during evaluation, for free telemetry
    piggybacked on the gauntlet's fixed deal panel (e.g. the defender
    trump-lead leak trend in league_progress_eval.py).

    wants(game, player, valid_actions) is a cheap predicate deciding whether
    this node is recorded; record(...) additionally receives the chosen action
    and the policy's action-probability vector at the node. The probe is
    strictly passive: it consumes no RNG and leaves recurrent state untouched,
    so per-deal scores are bit-identical with or without it.
    """

    wants: Callable[[Game, object, List[int]], bool]
    record: Callable[[Game, object, List[int], int, np.ndarray], None]


def _probe_action_probs(agent: PPOAgent, state, valid_actions, player_id) -> np.ndarray:
    """Policy distribution at a decision node, without perturbing play.

    get_action_probs_with_logits advances the recurrent memory, and act()
    afterwards encodes the same state again -- without a snapshot/restore the
    hand would proceed from a double-encoded memory (probe-vs-eval discrepancy
    found 2026-06-10: 73% vs 7% greedy trump-lead rate on identical weights;
    see validation/exit_validation.py). Snapshot, probe, restore.
    """
    saved_mem = {pid: t.detach().clone() for pid, t in agent._player_memories.items()}
    probs, _ = agent.get_action_probs_with_logits(
        state, valid_actions, player_id=player_id
    )
    agent._player_memories = saved_mem
    return probs[0].detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Game driver
# --------------------------------------------------------------------------- #
@dataclass
class HandResult:
    """Per-seat outcome of one played hand."""

    scores: List[float]  # zero-sum payoff per seat (index 0 == seat 1)
    points_margin: List[float]  # direction-corrected card-points margin per seat
    picker: int
    partner: int
    is_leaster: bool


def _role_points_margin(game: Game, seat: int) -> float:
    """Direction-corrected card-points margin for `seat` (1-indexed).

    Higher is always better, regardless of hand type. Not comparable in
    magnitude across hand types -- used only as a tiebreaker / low-variance
    secondary signal, never blended into the primary mean.
    """
    if game.is_leaster:
        # Fewer points taken is better in a leaster.
        return -float(game.points_taken[seat - 1])
    picking_team = {game.picker, game.partner}
    if seat in picking_team:
        return float(game.get_final_picker_points()) - 60.0
    return float(game.get_final_defender_points()) - 60.0


def play_hand(
    seat_to_model: Dict[int, Model],
    partner_mode: int,
    deal_seed: int,
    probe: Optional[DecisionProbe] = None,
    probe_seat: Optional[int] = None,
) -> HandResult:
    """Play one full hand with a fixed seat -> model assignment.

    The same Model may appear in multiple seats; recurrent state stays separate
    because act/observe are keyed by the seat's player_id.

    An optional DecisionProbe observes `probe_seat`'s decisions (see the probe
    docstring for the passivity guarantee).
    """
    game = Game(partner_selection_mode=partner_mode, seed=deal_seed)

    # Reset recurrent state once per distinct underlying agent instance.
    for agent in {id(m.agent): m.agent for m in seat_to_model.values()}.values():
        agent.reset_recurrent_state()

    while not game.is_done():
        for player in game.players:
            model = seat_to_model[player.position]
            valid_actions = player.get_valid_action_ids()
            while valid_actions:
                state = player.get_state_dict()
                probs = None
                if (
                    probe is not None
                    and player.position == probe_seat
                    and probe.wants(game, player, valid_actions)
                ):
                    probs = _probe_action_probs(
                        model.agent, state, valid_actions, player.position
                    )
                action, _, _ = model.agent.act(
                    state, valid_actions, player.position, deterministic=True
                )
                if probs is not None:
                    probe.record(game, player, valid_actions, action, probs)
                player.act(action)
                valid_actions = player.get_valid_action_ids()

                # Propagate end-of-trick observation to every seat's memory.
                if game.was_trick_just_completed:
                    for seat in game.players:
                        seat_to_model[seat.position].agent.observe(
                            seat.get_last_trick_state_dict(),
                            player_id=seat.position,
                        )

    scores = [float(p.get_score()) for p in game.players]
    margins = [_role_points_margin(game, p.position) for p in game.players]
    return HandResult(
        scores=scores,
        points_margin=margins,
        picker=int(game.picker),
        partner=int(game.partner),
        is_leaster=bool(getattr(game, "is_leaster", False)),
    )


# --------------------------------------------------------------------------- #
# Hero-in-a-field evaluation (shared by gauntlet and round-robin)
# --------------------------------------------------------------------------- #
# A field assignment maps (deal_index, hero_seat) -> {seat: Model} for the four
# non-hero seats. Held identical across heroes so comparisons stay paired.
FieldFn = Callable[[int, int], Dict[int, Model]]


@dataclass
class HeroEval:
    """Result of evaluating one hero against one field over a set of deals."""

    # per-deal mean (over the 5 hero seatings) score and points margin
    deal_score: np.ndarray  # shape (n_deals,)
    deal_margin: np.ndarray  # shape (n_deals,)
    # raw per (deal, hero_seat) scores, for diagnostics
    raw_score: np.ndarray  # shape (n_deals, 5)
    # role tallies across all deal x seat games
    role_counts: Dict[str, int] = field(default_factory=dict)


def evaluate_hero_in_field(
    hero: Model,
    field_fn: FieldFn,
    deal_seeds: Sequence[int],
    partner_mode: int,
    probe: Optional[DecisionProbe] = None,
) -> HeroEval:
    """Seat `hero` in all 5 seats per deal; the field fills the rest."""
    n = len(deal_seeds)
    raw_score = np.zeros((n, NUM_SEATS), dtype=np.float64)
    raw_margin = np.zeros((n, NUM_SEATS), dtype=np.float64)
    role_counts = {"picker": 0, "partner": 0, "defender": 0, "leaster": 0}

    for d, seed in enumerate(deal_seeds):
        for k in range(1, NUM_SEATS + 1):  # hero seat
            seat_to_model = dict(field_fn(d, k))
            seat_to_model[k] = hero
            res = play_hand(
                seat_to_model, partner_mode, seed, probe=probe, probe_seat=k
            )
            raw_score[d, k - 1] = res.scores[k - 1]
            raw_margin[d, k - 1] = res.points_margin[k - 1]
            if res.is_leaster:
                role_counts["leaster"] += 1
            elif k == res.picker:
                role_counts["picker"] += 1
            elif k == res.partner:
                role_counts["partner"] += 1
            else:
                role_counts["defender"] += 1

    return HeroEval(
        deal_score=raw_score.mean(axis=1),
        deal_margin=raw_margin.mean(axis=1),
        raw_score=raw_score,
        role_counts=role_counts,
    )


def make_panel_field_fn(panel: List[Model], n_deals: int, rng_seed: int) -> FieldFn:
    """Fixed reference-field assignment reused across all candidates.

    For each (deal, hero_seat) we pre-draw which panel model fills each of the
    four other seats. The draw is seeded and independent of the candidate, so
    every candidate faces an identical field -> comparisons remain paired.

    With >= 4 panel models the four seats are drawn WITHOUT replacement (a
    4-model panel therefore appears in full, in random seat order, in every
    game): same expectation, lower field-composition variance, and no games
    against multiple copies of the panel's weakest member. Smaller panels fall
    back to i.i.d. draws with replacement.
    """
    rng = random.Random(rng_seed)
    # assignment[d][k] = {seat: panel_index}
    assignment: List[List[Dict[int, int]]] = []
    for _ in range(n_deals):
        per_seat: List[Dict[int, int]] = [dict() for _ in range(NUM_SEATS + 1)]
        for k in range(1, NUM_SEATS + 1):
            field_seats = [s for s in range(1, NUM_SEATS + 1) if s != k]
            if len(panel) >= len(field_seats):
                draws = rng.sample(range(len(panel)), len(field_seats))
            else:
                draws = [rng.randrange(len(panel)) for _ in field_seats]
            per_seat[k] = dict(zip(field_seats, draws))
        assignment.append(per_seat)

    def field_fn(d: int, k: int) -> Dict[int, Model]:
        return {seat: panel[idx] for seat, idx in assignment[d][k].items()}

    return field_fn


def make_homogeneous_field_fn(opponent: Model) -> FieldFn:
    """All non-hero seats are the same opponent (used for round-robin)."""

    def field_fn(d: int, k: int) -> Dict[int, Model]:
        return {seat: opponent for seat in range(1, NUM_SEATS + 1) if seat != k}

    return field_fn


# --------------------------------------------------------------------------- #
# Statistics: block bootstrap over deals
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    mean: float
    lo: float  # CI lower
    hi: float  # CI upper
    se: float


def _bootstrap_deal_indices(
    n_deals: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n_deals) array of resampled deal indices (with replacement)."""
    return rng.integers(0, n_deals, size=(n_boot, n_deals))


def bootstrap_mean(
    per_deal: np.ndarray,
    boot_idx: np.ndarray,
    ci: float = 0.95,
) -> Estimate:
    """Bootstrap CI for the mean of a per-deal statistic."""
    point = float(per_deal.mean())
    boots = per_deal[boot_idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return Estimate(mean=point, lo=float(lo), hi=float(hi), se=float(boots.std(ddof=1)))


@dataclass
class PairwiseResult:
    a_id: str
    b_id: str
    diff: float  # mean(a) - mean(b), paired over deals
    lo: float
    hi: float
    p_value: float  # two-sided bootstrap p that diff == 0


def paired_diff(
    per_deal_a: np.ndarray,
    per_deal_b: np.ndarray,
    boot_idx: np.ndarray,
    a_id: str,
    b_id: str,
    ci: float = 0.95,
) -> PairwiseResult:
    """Paired (common-deal) bootstrap comparison of two heroes."""
    d = per_deal_a - per_deal_b
    point = float(d.mean())
    boots = d[boot_idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    # Two-sided bootstrap p-value (mass on the wrong side of 0, doubled).
    frac_le = float(np.mean(boots <= 0.0))
    frac_ge = float(np.mean(boots >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return PairwiseResult(
        a_id=a_id, b_id=b_id, diff=point, lo=float(lo), hi=float(hi), p_value=p
    )


def required_deals_for_mde(
    per_deal_diff_std: float, mde: float, ci: float = 0.95, power: float = 0.8
) -> int:
    """Rough paired-design sample size to detect `mde` score-per-hand.

    Uses a normal approximation on the per-deal paired difference. Returned as a
    planning aid; the reported CIs themselves are bootstrap (no normality
    assumed).
    """
    from math import ceil

    # z for two-sided alpha and one-sided power
    z_alpha = _inv_norm(1.0 - (1.0 - ci) / 2.0)
    z_power = _inv_norm(power)
    if mde <= 0:
        return 0
    n = ((z_alpha + z_power) * per_deal_diff_std / mde) ** 2
    return int(ceil(n))


def _inv_norm(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = (-2 * np.log(1 - p)) ** 0.5
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


# --------------------------------------------------------------------------- #
# Round-robin (secondary non-transitivity cross-check)
# --------------------------------------------------------------------------- #
def massey_ratings(adv: np.ndarray) -> np.ndarray:
    """Least-squares strengths s s.t. adv[i,j] ~= s[i]-s[j] (mean-centered).

    `adv` is the antisymmetric head-to-head advantage matrix. Returns a transitive
    rating; large residuals indicate non-transitivity (strategy cycles).
    """
    k = adv.shape[0]
    # Normal equations for min sum_{i,j} (s_i - s_j - adv_ij)^2 with sum s = 0.
    M = k * np.eye(k) - np.ones((k, k))
    rhs = adv.sum(axis=1)  # since sum_j (s_i - s_j) = k*s_i - sum s = k*s_i
    # Solve M s = rhs with the gauge sum(s)=0 (M is rank k-1).
    s, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return s - s.mean()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@dataclass
class CandidateReport:
    model: Model
    score: Estimate
    margin: Estimate
    role_counts: Dict[str, int]
    deal_score: np.ndarray  # retained for pairwise tests


def run_gauntlet(
    candidates: List[Model],
    panel: List[Model],
    deal_seeds: Sequence[int],
    partner_mode: int,
    boot_idx: np.ndarray,
) -> List[CandidateReport]:
    field_fn = make_panel_field_fn(panel, len(deal_seeds), rng_seed=20260619)
    reports: List[CandidateReport] = []
    for i, cand in enumerate(candidates, start=1):
        ev = evaluate_hero_in_field(cand, field_fn, deal_seeds, partner_mode)
        reports.append(
            CandidateReport(
                model=cand,
                score=bootstrap_mean(ev.deal_score, boot_idx),
                margin=bootstrap_mean(ev.deal_margin, boot_idx),
                role_counts=ev.role_counts,
                deal_score=ev.deal_score,
            )
        )
        s = reports[-1].score
        print(
            f"  [{i}/{len(candidates)}] {cand.model_id:<32} "
            f"score/hand = {s.mean:+.3f}  [{s.lo:+.3f}, {s.hi:+.3f}]"
        )
    return reports


def run_round_robin(
    candidates: List[Model],
    deal_seeds: Sequence[int],
    partner_mode: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Head-to-head advantage matrix via hero-in-homogeneous-opponent-field.

    adv[i,j] = mean_score(i among field of j) - mean_score(j among field of i),
    averaged over the same deals. Antisymmetric; diagonal 0.
    """
    k = len(candidates)
    # hero_in[i][j] = mean per-deal score of candidate i among a field of cand j.
    hero_in = np.zeros((k, k), dtype=np.float64)
    field_fns = [make_homogeneous_field_fn(c) for c in candidates]
    for i, hero in enumerate(candidates):
        for j in range(k):
            if i == j:
                continue
            ev = evaluate_hero_in_field(hero, field_fns[j], deal_seeds, partner_mode)
            hero_in[i, j] = float(ev.deal_score.mean())
        print(f"  round-robin: {hero.model_id} done ({i + 1}/{k})")
    adv = hero_in - hero_in.T
    return adv, hero_in


def write_csv(reports: List[CandidateReport], out_csv: Path) -> None:
    fieldnames = [
        "model_id",
        "filepath",
        "episodes",
        "score_per_hand",
        "score_ci_lo",
        "score_ci_hi",
        "score_se",
        "points_margin",
        "margin_ci_lo",
        "margin_ci_hi",
        "pick_rate",
        "partner_rate",
        "defender_rate",
        "leaster_rate",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in reports:
            total = max(1, sum(r.role_counts.values()))
            w.writerow(
                {
                    "model_id": r.model.model_id,
                    "filepath": str(r.model.filepath),
                    "episodes": r.model.episodes
                    if r.model.episodes is not None
                    else "",
                    "score_per_hand": f"{r.score.mean:.4f}",
                    "score_ci_lo": f"{r.score.lo:.4f}",
                    "score_ci_hi": f"{r.score.hi:.4f}",
                    "score_se": f"{r.score.se:.4f}",
                    "points_margin": f"{r.margin.mean:.4f}",
                    "margin_ci_lo": f"{r.margin.lo:.4f}",
                    "margin_ci_hi": f"{r.margin.hi:.4f}",
                    "pick_rate": f"{r.role_counts.get('picker', 0) / total:.4f}",
                    "partner_rate": f"{r.role_counts.get('partner', 0) / total:.4f}",
                    "defender_rate": f"{r.role_counts.get('defender', 0) / total:.4f}",
                    "leaster_rate": f"{r.role_counts.get('leaster', 0) / total:.4f}",
                }
            )
    print(f"Wrote CSV: {out_csv}")


def plot_strength(reports: List[CandidateReport], out_plot: Path) -> None:
    have_eps = [r for r in reports if r.model.episodes is not None]
    plt.figure(figsize=(10, 6))
    if len(have_eps) >= 2:
        have_eps.sort(key=lambda r: r.model.episodes)  # type: ignore[arg-type]
        x = np.array([r.model.episodes for r in have_eps], dtype=float)
        y = np.array([r.score.mean for r in have_eps])
        lo = np.array([r.score.lo for r in have_eps])
        hi = np.array([r.score.hi for r in have_eps])
        plt.plot(x, y, marker="o", color="C0", label="score/hand vs reference")
        plt.fill_between(x, lo, hi, color="C0", alpha=0.2, label="95% CI")
        plt.xlabel("Training Episodes")
    else:
        order = sorted(range(len(reports)), key=lambda i: reports[i].score.mean)
        x = np.arange(len(reports))
        y = np.array([reports[i].score.mean for i in order])
        err_lo = y - np.array([reports[i].score.lo for i in order])
        err_hi = np.array([reports[i].score.hi for i in order]) - y
        plt.errorbar(
            x,
            y,
            yerr=[err_lo, err_hi],
            fmt="o",
            capsize=4,
            color="C0",
            label="score/hand vs reference (95% CI)",
        )
        plt.xticks(
            x, [reports[i].model.model_id for i in order], rotation=45, ha="right"
        )
        plt.xlabel("Model")
    plt.axhline(0.0, color="gray", lw=1, ls=":")
    plt.ylabel("Mean game score per hand (vs reference field)")
    plt.title("Sheepshead agent strength (anchored, 95% bootstrap CI)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Wrote plot: {out_plot}")


def print_pairwise(reports: List[CandidateReport], boot_idx: np.ndarray) -> None:
    print("\nPaired pairwise comparisons (common deals; positive => row stronger):")
    ranked = sorted(reports, key=lambda r: r.score.mean, reverse=True)
    for ai in range(len(ranked)):
        for bi in range(ai + 1, len(ranked)):
            a, b = ranked[ai], ranked[bi]
            pr = paired_diff(
                a.deal_score, b.deal_score, boot_idx, a.model.model_id, b.model.model_id
            )
            sig = "" if pr.lo <= 0 <= pr.hi else "  *"
            print(
                f"  {a.model.model_id:<28} vs {b.model.model_id:<28} "
                f"d={pr.diff:+.3f} [{pr.lo:+.3f}, {pr.hi:+.3f}] p={pr.p_value:.3f}{sig}"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_partner_mode(mode_str: str) -> int:
    if mode_str in ("0", "jd"):
        return PARTNER_BY_JD
    if mode_str in ("1", "called"):
        return PARTNER_BY_CALLED_ACE
    raise ValueError(f"Unknown partner mode: {mode_str}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rigorous Sheepshead agent comparison")
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory of candidate .pt snapshots",
    )
    p.add_argument(
        "--candidates",
        type=str,
        nargs="*",
        default=None,
        help="Explicit candidate .pt paths (overrides --input-dir discovery)",
    )
    p.add_argument(
        "--anchors",
        type=str,
        nargs="*",
        default=None,
        help="Reference-field panel .pt paths. Default: strongest candidate.",
    )
    p.add_argument(
        "--episode-divisor",
        type=int,
        default=1,
        help="If >1, keep only candidates whose episode marker is divisible by it",
    )
    p.add_argument(
        "--deals",
        type=int,
        default=1000,
        help="Number of unique deals (the independent sample unit)",
    )
    p.add_argument(
        "--rr-deals",
        type=int,
        default=0,
        help="Deals for the round-robin cross-check (0 disables it)",
    )
    p.add_argument(
        "--partner-mode", type=str, default="called", choices=["jd", "called", "0", "1"]
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=5000, help="Bootstrap resamples")
    p.add_argument(
        "--mde",
        type=float,
        default=0.1,
        help="Effect size (score/hand) for the planning power calc",
    )
    p.add_argument("--out-csv", type=str, default="rigorous_results.csv")
    p.add_argument("--out-plot", type=str, default="rigorous_strength.png")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    partner_mode = resolve_partner_mode(args.partner_mode)
    registry = ModelRegistry()

    # --- Resolve candidates -------------------------------------------------
    from sheepshead.agent.convention_wrapper import parse_wrap_spec

    cand_specs: List[Tuple[Path, Optional[int], Optional[str]]] = []
    if args.candidates:
        for c in args.candidates:
            raw_path, wrap = parse_wrap_spec(c)
            pth = Path(raw_path)
            cand_specs.append((pth, extract_episodes_from_name(pth), wrap))
    elif args.input_dir:
        cand_specs = [
            (p, eps, None)
            for p, eps in discover_candidates(
                Path(args.input_dir).resolve(), args.episode_divisor
            )
        ]
    else:
        print("Provide --candidates or --input-dir.")
        return 1
    if not cand_specs:
        print("No candidates found.")
        return 1
    if len(cand_specs) == 1 and not args.anchors:
        print("A single candidate needs an explicit --anchors panel to score against.")
        return 1

    print(f"Loading {len(cand_specs)} candidates ...")
    candidates = [registry.get(p, eps, wrap) for p, eps, wrap in cand_specs]

    # --- Resolve anchor panel ----------------------------------------------
    if args.anchors:
        panel = []
        for a in args.anchors:
            raw_path, wrap = parse_wrap_spec(a)
            panel.append(registry.get(Path(raw_path), wrap=wrap))
        # Canonical order: the frozen field assignment is drawn from panel
        # indices, so CLI argument order must not change the experiment.
        # (model_id tiebreak only distinguishes wrapped arms of one file;
        # raw-only panels keep their historical ordering.)
        panel.sort(key=lambda m: (str(m.filepath), m.model_id))
        print(f"Reference panel: {[m.model_id for m in panel]}")
    else:
        strongest = max(
            candidates, key=lambda m: (m.episodes is not None, m.episodes or 0)
        )
        panel = [strongest]
        print(
            f"No --anchors given; using strongest candidate as panel: "
            f"{strongest.model_id}"
        )

    # --- Deals + bootstrap indices -----------------------------------------
    seed_rng = random.Random(args.seed)
    deal_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(args.deals)]
    boot_rng = np.random.default_rng(args.seed)
    boot_idx = _bootstrap_deal_indices(args.deals, args.n_boot, boot_rng)

    games = len(candidates) * args.deals * NUM_SEATS
    print(
        f"\nAnchored gauntlet: {len(candidates)} candidates x {args.deals} deals "
        f"x {NUM_SEATS} seatings = {games:,} games"
    )
    reports = run_gauntlet(candidates, panel, deal_seeds, partner_mode, boot_idx)

    # --- Report -------------------------------------------------------------
    reports_sorted = sorted(reports, key=lambda r: r.score.mean, reverse=True)
    print("\n=== Strength ranking (score/hand vs reference field) ===")
    for rank, r in enumerate(reports_sorted, start=1):
        print(
            f" {rank:>2}. {r.model.model_id:<32} "
            f"{r.score.mean:+.3f} [{r.score.lo:+.3f}, {r.score.hi:+.3f}]  "
            f"(margin tiebreak {r.margin.mean:+.2f}, pick {r.role_counts.get('picker', 0)})"
        )
    print_pairwise(reports, boot_idx)

    # Planning aid: typical per-deal paired-diff std across adjacent ranks.
    if len(reports_sorted) >= 2:
        diffs_std = np.std(
            reports_sorted[0].deal_score - reports_sorted[-1].deal_score, ddof=1
        )
        need = required_deals_for_mde(float(diffs_std), args.mde)
        print(
            f"\nPower: to resolve a {args.mde:+.2f} score/hand gap "
            f"(paired, 95%/80%) ~{need:,} deals (have {args.deals:,})."
        )

    # --- Optional round-robin cross-check ----------------------------------
    if args.rr_deals and len(candidates) >= 3:
        rr_seeds = deal_seeds[: min(args.rr_deals, args.deals)]
        print(
            f"\nRound-robin cross-check on {len(rr_seeds)} deals "
            f"({len(candidates)}x{len(candidates) - 1} matchups) ..."
        )
        adv, _ = run_round_robin(candidates, rr_seeds, partner_mode)
        s = massey_ratings(adv)
        resid = adv - (s[:, None] - s[None, :])
        nontrans = float(np.sqrt(np.mean(resid**2)))
        print("\nRound-robin Massey strength (relative, transitive fit):")
        for i in np.argsort(-s):
            print(f"  {candidates[i].model_id:<32} {s[i]:+.3f}")
        print(
            f"  non-transitivity (RMS residual): {nontrans:.3f} "
            f"(0 = perfectly transitive)"
        )

    write_csv(reports_sorted, Path(args.out_csv).resolve())
    plot_strength(reports, Path(args.out_plot).resolve())
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
