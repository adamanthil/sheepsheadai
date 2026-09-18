#!/usr/bin/env python3
"""Decision rules of the release-candidate program, as pure functions of
recorded numbers (Training_Program_Redesign §5; pre-registered 2026-09-02).

League phase — the marginal-value handoff rule. Continue policy gradient
while the duplicate h2h gain over the previous generation is at least
``min_gain`` with the CI lower bound above zero; the bar is search's
measured yield at equal compute (+0.026 per ~2.5-day iteration). Applied to
the July retention run it passes generations 1, 2 and 5 and fails 3, 4, 6,
7, 8 — exactly the generations that paid. The first failing generation
(after the floor) fires the single play-target entropy step; a second
failure hands off to search. A fresh-deal confirmation guards a noise miss
(at 2,000 deals/mode the SE is ~0.012, so a true +0.02 passes about half
the time).

The handoff checkpoint is the last generation wholly at a SETTLED entropy
target: E7 found the two deepest wrong-side lock-ins at the two transition
checkpoints, so the generation right after a step never hands off — the
pre-step boundary does.

Policy iteration — stop when the certified gain over theta_k is below
``se_multiple`` SEs for ``flat_iterations`` consecutive iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class HandoffRuleConfig:
    min_gain: float = 0.02
    ci_z: float = 2.0
    min_generations: int = 3
    max_generations: int = 8


def gain_improving(edge: float, se: float, cfg: HandoffRuleConfig) -> bool:
    return bool(edge >= cfg.min_gain and edge - cfg.ci_z * se > 0.0)


@dataclass
class GenerationVerdict:
    generation: int
    edge: float
    se: float
    improving: bool
    # The fresh-deal confirmation (edge, se) when the primary read failed;
    # None when the primary read passed and no confirmation was needed.
    confirm_edge: Optional[float] = None
    confirm_se: Optional[float] = None
    needs_confirmation: bool = False


def generation_verdict(
    generation: int,
    edge: float,
    se: float,
    cfg: HandoffRuleConfig,
    confirm: Optional[tuple[float, float]] = None,
) -> GenerationVerdict:
    """The primary read decides; a failing primary read is re-tried on the
    confirmation deals (``confirm`` = (edge, se) on the fresh seed) and the
    generation counts as improving if either read passes."""
    primary = gain_improving(edge, se, cfg)
    if primary:
        return GenerationVerdict(generation, edge, se, True)
    if confirm is None:
        return GenerationVerdict(generation, edge, se, False, needs_confirmation=True)
    c_edge, c_se = confirm
    return GenerationVerdict(
        generation,
        edge,
        se,
        gain_improving(c_edge, c_se, cfg),
        confirm_edge=c_edge,
        confirm_se=c_se,
    )


@dataclass(frozen=True)
class AuxReadinessConfig:
    """The aux-head readiness precondition of the handoff (§5.4, 09-18).

    The four DETERMINISTIC aux heads — seen-trump mask, unseen-trump-higher,
    known points, secret partner — are exact functions of what the seat has
    observed, so 100% is their true ceiling; and the league is the only
    phase that can build the trunk memory they read (policy iteration
    maintains the heads on 8,000 games at 3e-5, the bidding phase freezes
    the encoder). Before the handoff they must be essentially never wrong on
    the boundary battery (4 x 1,000 greedy games, every seat's play nodes).
    Percent units; the recall and false-seen bars apply to each of tricks
    1-5 (trick 0 is the picker's bury alone: too few cards, and erratic even
    with the bury in the observation). Known points is a regression head
    under a smooth-L1 loss, never integer-exact, so its bar is the mean
    absolute error in points at the v2 lineage's converged level (0.88 at
    league 7.7M; it read 5-9 points at 0.7-4.7M while seen-trump converged,
    which this bar catches); the exact-after-rounding rate is recorded
    beside it. Win and return predict outcomes with irreducible uncertainty
    and are not gated."""

    seen_trump_acc_min: float = 99.5
    seen_trump_false_seen_max: float = 0.5
    seen_trump_recall_min: float = 99.0
    unseen_higher_acc_min: float = 99.0
    points_mae_max: float = 1.0
    secret_acc_min: float = 99.5


def aux_readiness(reads: dict, cfg: AuxReadinessConfig) -> tuple[bool, list[str]]:
    """``reads`` = the boundary battery's means (the greedy probe's aux keys).
    Returns (ready, failures); every failing bar is named."""
    failures: list[str] = []

    def at_least(key: str, bar: float, label: str) -> None:
        v = float(reads.get(key, 0.0))
        if v < bar:
            failures.append(f"{label} {v:.2f} < {bar}")

    at_least("seen_trump_acc", cfg.seen_trump_acc_min, "seen-trump acc")
    fs = float(reads.get("seen_trump_false_seen", 100.0))
    if fs > cfg.seen_trump_false_seen_max:
        failures.append(
            f"seen-trump false-seen {fs:.2f} > {cfg.seen_trump_false_seen_max}"
        )
    recall_t = list(reads.get("seen_trump_recall_by_trick") or [])
    false_t = list(reads.get("seen_trump_false_seen_by_trick") or [])
    for t in range(1, 6):
        r = float(recall_t[t]) if t < len(recall_t) else 0.0
        if r < cfg.seen_trump_recall_min:
            failures.append(
                f"seen-trump recall t{t} {r:.2f} < {cfg.seen_trump_recall_min}"
            )
        f = float(false_t[t]) if t < len(false_t) else 100.0
        if f > cfg.seen_trump_false_seen_max:
            failures.append(
                f"seen-trump false-seen t{t} {f:.2f} > {cfg.seen_trump_false_seen_max}"
            )
    at_least("aux_unseen_higher_acc", cfg.unseen_higher_acc_min, "unseen-higher acc")
    mae = float(reads.get("aux_points_mae", 1e9))
    if mae > cfg.points_mae_max:
        failures.append(f"points mae {mae:.2f} > {cfg.points_mae_max}")
    at_least("aux_secret_acc", cfg.secret_acc_min, "secret-partner acc")
    return not failures, failures


@dataclass
class HandoffDecision:
    action: str  # "continue" | "entropy_step" | "handoff" | "review"
    reason: str


def decide_handoff(
    improving_history: Sequence[bool],
    generation: int,
    step_generation: Optional[int],
    cfg: HandoffRuleConfig,
    aux_ready: bool = True,
) -> HandoffDecision:
    """Decision after ``generation`` given the improving flags for
    generations 1..generation and the generation at whose boundary the
    entropy step fired (None if it has not). ``aux_ready`` (aux_readiness)
    is the handoff's precondition: a handoff the marginal-value rule would
    make is deferred (``continue``) while the deterministic aux heads are
    not ready, and becomes ``review`` at the generation cap."""
    base = _marginal_value_decision(improving_history, generation, step_generation, cfg)
    if base.action != "handoff" or aux_ready:
        return base
    if generation >= cfg.max_generations:
        return HandoffDecision(
            "review",
            f"max_generations cap ({cfg.max_generations}) reached with the "
            "deterministic aux heads not ready",
        )
    return HandoffDecision(
        "continue", f"handoff deferred, aux heads not ready ({base.reason})"
    )


def _marginal_value_decision(
    improving_history: Sequence[bool],
    generation: int,
    step_generation: Optional[int],
    cfg: HandoffRuleConfig,
) -> HandoffDecision:
    if len(improving_history) != generation:
        raise ValueError(
            f"improving_history covers {len(improving_history)} generations, "
            f"expected {generation}"
        )
    if generation >= cfg.max_generations:
        return HandoffDecision(
            "handoff", f"max_generations cap ({cfg.max_generations})"
        )
    if improving_history[-1]:
        return HandoffDecision("continue", "h2h gain clears the marginal-value bar")
    if generation < cfg.min_generations:
        return HandoffDecision(
            "continue", f"below min_generations floor ({cfg.min_generations})"
        )
    if step_generation is None:
        if generation < 2:
            # The controller attaches at the gen-1 boundary; there is no
            # settled target to step yet.
            return HandoffDecision("continue", "no entropy target to step before gen 2")
        return HandoffDecision(
            "entropy_step", "first failure of the marginal-value bar: play-target step"
        )
    return HandoffDecision(
        "handoff", "second failure of the marginal-value bar after the entropy step"
    )


def settled_generation(generation: int, step_generation: Optional[int]) -> int:
    """The generation whose boundary checkpoint hands off: ``generation``
    itself unless it is the one trained right after the entropy step, in
    which case the pre-step boundary (the last settled one)."""
    if step_generation is not None and generation == step_generation + 1:
        return step_generation
    return generation


@dataclass(frozen=True)
class IterationRuleConfig:
    se_multiple: float = 2.0
    flat_iterations: int = 2
    max_iterations: int = 5


def iteration_stop(
    gains: Sequence[tuple[float, float]], cfg: IterationRuleConfig
) -> tuple[bool, str]:
    """``gains`` = (edge, se) of each certified iteration vs its theta_k, in
    order. Stop after the cap, or once the last ``flat_iterations`` gains all
    sit below ``se_multiple`` SEs."""
    if len(gains) >= cfg.max_iterations:
        return True, f"max_iterations cap ({cfg.max_iterations})"
    if len(gains) >= cfg.flat_iterations and all(
        edge < cfg.se_multiple * se for edge, se in gains[-cfg.flat_iterations :]
    ):
        return True, (
            f"gain below {cfg.se_multiple} SE for {cfg.flat_iterations} "
            "consecutive iterations"
        )
    return False, "gain still resolvable"
