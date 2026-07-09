#!/usr/bin/env python3
"""
Stopping-rule and calibration decisions for the extended league run.

Pre-registered in notebooks/Extended_League_202607.md; consumed by
analysis/run_extended_league.py. Pure numpy on purpose: every decision the
orchestrator makes about *when training is over* lives here as a deterministic
function of recorded numbers, so the whole rule is unit-testable on synthetic
curves without loading torch or playing a single game.

The rule tightens the pre-registered recipe in
notebooks/Architecture_Ablation_202607.md ("Continuing league generations"):
the repro league learned ~0.015 score/hand per 1M episodes, so the original
1000-deal / 0.07-MDE version would false-stop on real-but-slow learning.
Tightenings: 4000-deal composite panels (MDE ~= 0.035), plus a paired
three-endpoint slope test that can see slow climbs the single-generation gain
test cannot.

A generation g is *flat* when none of three improvement signals fire:

  A. gain vs previous best   mean(panel_g - panel_best) >= MDE, bootstrap lo > 0
  B. head-to-head vs g-1     edge >= H2H_MIN and edge >= 2*se
  C. paired 3-endpoint slope mean((panel_g - panel_{g-2})/2) >= SLOPE_MIN, lo > 0

Stop candidate: two consecutive flat generations (the onehot-ff false-plateau
lesson) after a minimum-generations floor, confirmed by a fresh-deal panel
(seed 20260706) before the run actually ends.

Bootstrap math mirrors analysis/rigorous_eval.py (bootstrap over deals, the
independent unit; boot_idx arrays are interchangeable between the two modules)
but is reimplemented here to keep this module import-light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Pre-registered constants (change only via a new pre-registration)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StopRuleConfig:
    """Thresholds for the generation stopping rule.

    mde: minimum detectable/meaningful per-generation panel gain (score/hand).
        0.035 matches the ~4000-deal composite panel's resolution.
    h2h_min: minimum head-to-head edge vs the previous generation.
    slope_min: minimum per-generation slope over the last three endpoints.
        MDE/2: a climb that would take two generations to accumulate one MDE.
    ci: bootstrap confidence level for all interval checks.
    min_generations: no stop candidate before this many completed generations.
    max_generations: safety cap; forces the confirmation phase regardless.
    """

    mde: float = 0.035
    h2h_min: float = 0.05
    slope_min: float = 0.0175
    ci: float = 0.95
    min_generations: int = 4
    max_generations: int = 12


# --------------------------------------------------------------------------- #
# Bootstrap primitives (deal = independent unit; see rigorous_eval.py)
# --------------------------------------------------------------------------- #
def bootstrap_deal_indices(
    n_deals: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n_deals) array of resampled deal indices (with replacement)."""
    return rng.integers(0, n_deals, size=(n_boot, n_deals))


@dataclass
class IntervalStat:
    """A per-deal mean with its bootstrap interval."""

    mean: float
    lo: float
    hi: float
    se: float
    p_value: float  # two-sided bootstrap p that the mean == 0


def bootstrap_interval(
    per_deal: np.ndarray, boot_idx: np.ndarray, ci: float = 0.95
) -> IntervalStat:
    boots = per_deal[boot_idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    frac_le = float(np.mean(boots <= 0.0))
    frac_ge = float(np.mean(boots >= 0.0))
    return IntervalStat(
        mean=float(per_deal.mean()),
        lo=float(lo),
        hi=float(hi),
        se=float(boots.std(ddof=1)),
        p_value=min(1.0, 2.0 * min(frac_le, frac_ge)),
    )


# --------------------------------------------------------------------------- #
# The three improvement tests
# --------------------------------------------------------------------------- #
@dataclass
class GainTest:
    """A: paired panel gain of generation g over the previous best endpoint."""

    stat: IntervalStat
    improving: bool
    best_generation: int  # which earlier generation held the best


def gain_vs_best(
    per_deal_g: np.ndarray,
    per_deal_best: np.ndarray,
    best_generation: int,
    boot_idx: np.ndarray,
    cfg: StopRuleConfig,
) -> GainTest:
    stat = bootstrap_interval(per_deal_g - per_deal_best, boot_idx, cfg.ci)
    return GainTest(
        stat=stat,
        improving=bool(stat.mean >= cfg.mde and stat.lo > 0.0),
        best_generation=best_generation,
    )


@dataclass
class H2HTest:
    """B: CRN paired head-to-head edge of generation g over generation g-1."""

    edge: float
    se: float
    improving: bool


def h2h_test(edge: float, se: float, cfg: StopRuleConfig) -> H2HTest:
    return H2HTest(
        edge=float(edge),
        se=float(se),
        improving=bool(edge >= cfg.h2h_min and edge >= 2.0 * se),
    )


@dataclass
class SlopeTest:
    """C: paired per-generation slope over endpoints g-2, g-1, g.

    For three equally spaced points the OLS slope is (y3 - y1)/2 -- the middle
    point has zero leverage -- so the per-deal statistic (panel_g[i] -
    panel_{g-2}[i]) / 2 IS the slope estimator, and CRN pairing cancels the
    deal effect inside each term.
    """

    stat: IntervalStat
    climbing: bool
    # Statistically real climb that is nonetheless below slope_min: the rule
    # treats learning as concluded, but the report must surface it.
    small_but_significant: bool


def slope_test(
    per_deal_g: np.ndarray,
    per_deal_gm2: np.ndarray,
    boot_idx: np.ndarray,
    cfg: StopRuleConfig,
) -> SlopeTest:
    stat = bootstrap_interval((per_deal_g - per_deal_gm2) / 2.0, boot_idx, cfg.ci)
    significant = stat.lo > 0.0
    return SlopeTest(
        stat=stat,
        climbing=bool(stat.mean >= cfg.slope_min and significant),
        small_but_significant=bool(significant and stat.mean < cfg.slope_min),
    )


# --------------------------------------------------------------------------- #
# Per-generation verdict and the stop decision
# --------------------------------------------------------------------------- #
@dataclass
class FlatVerdict:
    generation: int
    gain: GainTest
    h2h: H2HTest
    slope: Optional[SlopeTest]  # None before generation 2
    flat: bool = field(init=False)

    def __post_init__(self) -> None:
        climbing = self.slope.climbing if self.slope is not None else False
        self.flat = not (self.gain.improving or self.h2h.improving or climbing)


def flat_verdict(
    generation: int,
    per_deal_by_gen: Dict[int, np.ndarray],
    endpoint_means: Dict[int, float],
    h2h_edge: float,
    h2h_se: float,
    boot_idx: np.ndarray,
    cfg: StopRuleConfig,
) -> FlatVerdict:
    """Assemble the flat/improving verdict for `generation`.

    per_deal_by_gen: deal-aligned per-deal endpoint vectors keyed by generation
        (generation 0 = the resume-checkpoint baseline). Must contain every
        generation <= `generation` that this verdict needs: g itself, the
        previous-best generation, and g-2 when g >= 2.
    endpoint_means: mean endpoint score per generation, used only to locate the
        previous best (kept explicit so callers can't accidentally rank on a
        different statistic than they stored).
    """
    prior_gens = [h for h in endpoint_means if h < generation]
    if not prior_gens:
        raise ValueError("flat_verdict needs at least one earlier endpoint")
    best_gen = max(prior_gens, key=lambda h: endpoint_means[h])

    gain = gain_vs_best(
        per_deal_by_gen[generation],
        per_deal_by_gen[best_gen],
        best_gen,
        boot_idx,
        cfg,
    )
    h2h = h2h_test(h2h_edge, h2h_se, cfg)
    slope: Optional[SlopeTest] = None
    if generation >= 2:
        slope = slope_test(
            per_deal_by_gen[generation],
            per_deal_by_gen[generation - 2],
            boot_idx,
            cfg,
        )
    return FlatVerdict(generation=generation, gain=gain, h2h=h2h, slope=slope)


@dataclass
class StopDecision:
    stop_candidate: bool
    forced_by_cap: bool
    flat_streak: int
    reason: str


def decide_stop(
    flat_history: Sequence[bool], generation: int, cfg: StopRuleConfig
) -> StopDecision:
    """Decide whether to enter the confirmation phase after `generation`.

    flat_history: flat flags for generations 1..generation, in order. A
    confirmation-contradiction reset is expressed by the caller rewriting the
    trailing flags to False (the streak restarts from the resumed generation).
    """
    if len(flat_history) != generation:
        raise ValueError(
            f"flat_history covers {len(flat_history)} generations, expected {generation}"
        )
    streak = 0
    for is_flat in reversed(flat_history):
        if not is_flat:
            break
        streak += 1

    if generation >= cfg.max_generations:
        return StopDecision(
            stop_candidate=True,
            forced_by_cap=True,
            flat_streak=streak,
            reason=f"max_generations cap ({cfg.max_generations}) reached",
        )
    if generation < cfg.min_generations:
        return StopDecision(
            stop_candidate=False,
            forced_by_cap=False,
            flat_streak=streak,
            reason=f"below min_generations floor ({cfg.min_generations})",
        )
    if streak >= 2:
        return StopDecision(
            stop_candidate=True,
            forced_by_cap=False,
            flat_streak=streak,
            reason="two consecutive flat generations",
        )
    return StopDecision(
        stop_candidate=False,
        forced_by_cap=False,
        flat_streak=streak,
        reason="improvement signal within the last two generations",
    )


@dataclass
class ConfirmationVerdict:
    """Fresh-deal (seed 20260706) check of the last two generations' gain."""

    stat: IntervalStat
    contradiction: bool  # fresh deals say the gain was real -> resume training


def confirmation_verdict(
    conf_per_deal_g: np.ndarray,
    conf_per_deal_gm2: np.ndarray,
    boot_idx: np.ndarray,
    cfg: StopRuleConfig,
) -> ConfirmationVerdict:
    stat = bootstrap_interval(conf_per_deal_g - conf_per_deal_gm2, boot_idx, cfg.ci)
    return ConfirmationVerdict(
        stat=stat,
        contradiction=bool(stat.mean >= cfg.mde and stat.lo > 0.0),
    )


# --------------------------------------------------------------------------- #
# Anchor-coefficient calibration
# --------------------------------------------------------------------------- #
@dataclass
class ProbeSummary:
    """Distilled telemetry of one ~15k-episode calibration probe."""

    coeff: float
    kl_last: float  # mean anchor_kl over the final 3 PPO updates
    kl_max: float  # max anchor_kl over the whole probe
    gate_violations: int  # greedy-health gate violations during the probe
    final_pick_rate: float  # last greedy-health probe pick rate, PERCENT (0-100)


@dataclass
class CalibrationChoice:
    coeff: float
    qualified: bool
    reason: str


def pick_anchor_coeff(
    probes: Sequence[ProbeSummary],
    baseline_pick_rate: float,
    *,
    kl_last_limit: float = 0.05,
    kl_max_limit: float = 0.10,
    pick_rate_tolerance: float = 10.0,
) -> CalibrationChoice:
    """Choose the smallest coefficient that anchors without freezing.

    Qualifying probe: anchor-KL bounded (settled below kl_last_limit nats with
    no excursion past kl_max_limit), zero greedy gate violations, and a final
    pick rate within pick_rate_tolerance PERCENTAGE POINTS of the resume
    checkpoint's baseline (rates are percent, 0-100, matching
    training_utils.greedy_health_probe). Smallest wins because the anchor caps
    bidding improvement -- we want the minimum sufficient restraint.

    If nothing qualifies, fall back to the LARGEST candidate (strongest
    anchoring is the safe failure mode for a warm start) and flag it.
    """
    if not probes:
        raise ValueError("pick_anchor_coeff needs at least one probe")
    qualifying = [
        p
        for p in probes
        if p.kl_last <= kl_last_limit
        and p.kl_max <= kl_max_limit
        and p.gate_violations == 0
        and abs(p.final_pick_rate - baseline_pick_rate) <= pick_rate_tolerance
    ]
    if qualifying:
        best = min(qualifying, key=lambda p: p.coeff)
        return CalibrationChoice(
            coeff=best.coeff,
            qualified=True,
            reason=f"smallest qualifying coefficient of {len(qualifying)}",
        )
    fallback = max(probes, key=lambda p: p.coeff)
    return CalibrationChoice(
        coeff=fallback.coeff,
        qualified=False,
        reason="fallback_largest: no probe met the KL/health criteria",
    )


# --------------------------------------------------------------------------- #
# Serialization helpers (state.json round-trip)
# --------------------------------------------------------------------------- #
def interval_to_dict(s: IntervalStat) -> Dict[str, float]:
    return {"mean": s.mean, "lo": s.lo, "hi": s.hi, "se": s.se, "p_value": s.p_value}


def verdict_to_dict(v: FlatVerdict) -> Dict[str, object]:
    out: Dict[str, object] = {
        "generation": v.generation,
        "flat": v.flat,
        "gain": {
            **interval_to_dict(v.gain.stat),
            "improving": v.gain.improving,
            "best_generation": v.gain.best_generation,
        },
        "h2h": {
            "edge": v.h2h.edge,
            "se": v.h2h.se,
            "improving": v.h2h.improving,
        },
        "slope": None,
    }
    if v.slope is not None:
        out["slope"] = {
            **interval_to_dict(v.slope.stat),
            "climbing": v.slope.climbing,
            "small_but_significant": v.slope.small_but_significant,
        }
    return out
