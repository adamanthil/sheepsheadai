#!/usr/bin/env python3
"""Target-entropy controller for the league trainer (adaptive entropy, Phase 2).

Replaces the clock-based entropy-coefficient schedule with feedback control
of the MEASURED policy entropy, two loops:

Inner loop (here, per update): each head's entropy coefficient becomes a
log-space integral controller holding the measured normalized entropy
(``stats["head_entropy_norm"]``, theta_old, H/ln n_legal) at a target —
SAC's automatic temperature adjustment (Haarnoja et al., arXiv:1812.05905
§5) in its discrete fraction-of-max form (Christodoulou, arXiv:1910.07207).
Targets initialize BUMPLESSLY: a head with no explicit target adopts the
first measured value, so switch-on changes nothing at t=0 (Astrom &
Wittenmark, *Adaptive Control*, 2nd ed. 1995, ch. 9).

Outer loop (``step_targets``, called by the orchestrator at generation
boundaries on a flat h2h verdict): anneal-head targets step geometrically
toward their floor, ``target <- floor + retain * (target - floor)``. Step
magnitude follows PBT's hyperparameter perturbation scale (~20-25%;
Jaderberg et al., arXiv:1711.09846). Per the 2026-07-28 backfill
(runs/league_retention_pg/entropy_backfill.json), only the PLAY head
anneals — it is the one head where the regularizer measurably binds
(H_norm 0.88 -> 0.75 over 1.8M eps); pick/partner/bury targets HOLD their
operating point (pick is near-deterministic per node with a thin soft
boundary band that holding protects; see SOFTBAND_HNORM telemetry). When
the next step would be smaller than ``min_step`` (checkpoint-noise scale),
the head is at floor and no longer steps — the orchestrator then lets flat
generations count toward stopping (targets-at-floor is the precondition
for "flat means converged", removing the converged-vs-entropy-limited
confound from the stop rule).

Floors are never zero: imperfect-information equilibria are genuinely
mixed (Sokota et al., arXiv:2206.05825; quantal-response equilibria), and
a deterministic pick threshold would leak hand information.

State (targets + coefficients) persists as a JSON sidecar next to the
checkpoints so crash-resume and generation handoffs are seamless; the
orchestrator edits the same file to step targets between generations.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from sheepshead.training.config import PFSPHyperparams

HEADS = ("pick", "partner", "bury", "play")

_HP = PFSPHyperparams()

# Coefficient bounds: floor at the legacy schedule's end values (its "small,
# not zero" intent), cap at 4x the legacy start (room to fight a collapse
# without runaway).
ALPHA_MIN = {
    "pick": _HP.entropy_pick_end,
    "partner": _HP.entropy_partner_end,
    "bury": _HP.entropy_bury_end,
    "play": _HP.entropy_play_end,
}
ALPHA_MAX = {
    "pick": 4.0 * _HP.entropy_pick_start,
    "partner": 4.0 * _HP.entropy_partner_start,
    "bury": 4.0 * _HP.entropy_bury_start,
    "play": 4.0 * _HP.entropy_play_start,
}


@dataclass(frozen=True)
class EntropyControllerConfig:
    """Inner-gain and outer-step constants (pre-registered 2026-07-28,
    notebooks/Learning_System_Redesign_202607.md Phase 2).

    eta: log-space integral gain — d(log alpha) = eta * (target - measured)
        per update. At eta=1 a sustained error the size of the backfill's
        organic per-generation play drift (0.057) moves alpha ~5.9%/update,
        settling in ~10-20 of a generation's ~61 updates; per-update
        measurement noise (SE ~0.002-0.004 at 16k rows) contributes ~0.4%
        jitter — two orders below the clamp.
    max_log_step: per-update |d log alpha| clamp (safety against transients).
    retain: outer-step gap retention — target <- floor + retain*(target-floor).
        1-retain = 0.25 of the gap (~0.12 first play step), ~2x the organic
        drift (distinguishable) and PBT-sized.
    min_step: when the would-be step (1-retain)*(target-floor) falls below
        this, the head is at floor (steps smaller than checkpoint-to-
        checkpoint probe noise ~±0.02 are not worth a generation).
    anneal_heads / floors: which targets descend and to where. Backfill
        verdict: play only; floor 0.28 = mixed-equilibrium reserve (~37% of
        the 1.8M operating point) — the least-data-grounded number here,
        approached gradually under h2h gates so a wrong floor is caught by
        the plateau ladder, not paid all at once.
    """

    eta: float = 1.0
    max_log_step: float = 0.1
    retain: float = 0.75
    min_step: float = 0.03
    anneal_heads: tuple = ("play",)
    floors: dict = field(default_factory=lambda: {"play": 0.28})


class EntropyTargetController:
    def __init__(
        self,
        config: EntropyControllerConfig | None = None,
        targets: dict | None = None,
        alphas: dict | None = None,
    ):
        self.config = config or EntropyControllerConfig()
        # None target = bumpless: adopt the first measured value.
        self.targets: dict = {h: None for h in HEADS}
        if targets:
            self.targets.update({h: targets[h] for h in targets if h in HEADS})
        self.alphas: dict = dict(alphas) if alphas else {}

    # ------------------------------------------------------------------ #
    # Inner loop
    # ------------------------------------------------------------------ #
    def attach(self, agent) -> None:
        """Adopt the agent's current coefficients as the controller state
        (bumpless in alpha), then take over."""
        for h in HEADS:
            if h not in self.alphas:
                self.alphas[h] = float(getattr(agent, f"entropy_coeff_{h}"))
        self.apply(agent)

    def apply(self, agent) -> None:
        for h in HEADS:
            if h in self.alphas:
                setattr(agent, f"entropy_coeff_{h}", self.alphas[h])

    def observe(self, head_entropy_norm: dict) -> dict:
        """One integral-control step from an update's theta_old measurement.
        Heads without a measurement this update are skipped. Returns the
        per-head log-alpha deltas actually applied (telemetry)."""
        deltas = {}
        for h in HEADS:
            measured = head_entropy_norm.get(h)
            if measured is None or h not in self.alphas:
                continue
            if self.targets[h] is None:
                self.targets[h] = float(measured)  # bumpless initialization
                deltas[h] = 0.0
                continue
            err = self.targets[h] - measured
            dlog = max(
                -self.config.max_log_step,
                min(self.config.max_log_step, self.config.eta * err),
            )
            self.alphas[h] = max(
                ALPHA_MIN[h],
                min(ALPHA_MAX[h], self.alphas[h] * math.exp(dlog)),
            )
            deltas[h] = dlog
        return deltas

    # ------------------------------------------------------------------ #
    # Outer loop (generation boundaries)
    # ------------------------------------------------------------------ #
    def head_at_floor(self, h: str) -> bool:
        """True when the head no longer steps: hold heads always (they have
        no ladder), anneal heads once the next step would be < min_step or
        their target is not yet initialized."""
        if h not in self.config.anneal_heads:
            return True
        t = self.targets.get(h)
        if t is None:
            return False  # not yet initialized: cannot be judged converged
        gap = t - self.config.floors[h]
        return (1.0 - self.config.retain) * gap < self.config.min_step

    def at_floor(self) -> bool:
        return all(self.head_at_floor(h) for h in self.config.anneal_heads)

    def step_targets(self) -> dict:
        """Geometric step of every anneal head not yet at floor. Returns
        {head: (old, new)} for the heads that moved (empty when at floor)."""
        moved = {}
        for h in self.config.anneal_heads:
            if self.head_at_floor(h) or self.targets.get(h) is None:
                continue
            old = self.targets[h]
            floor = self.config.floors[h]
            self.targets[h] = floor + self.config.retain * (old - floor)
            moved[h] = (old, self.targets[h])
        return moved

    # ------------------------------------------------------------------ #
    # Persistence (JSON sidecar; orchestrator edits the same file)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "targets": self.targets,
            "alphas": self.alphas,
            "config": {
                "eta": self.config.eta,
                "max_log_step": self.config.max_log_step,
                "retain": self.config.retain,
                "min_step": self.config.min_step,
                "anneal_heads": list(self.config.anneal_heads),
                "floors": self.config.floors,
            },
        }

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, d: dict) -> "EntropyTargetController":
        cfg = d.get("config", {})
        config = EntropyControllerConfig(
            eta=cfg.get("eta", 1.0),
            max_log_step=cfg.get("max_log_step", 0.1),
            retain=cfg.get("retain", 0.75),
            min_step=cfg.get("min_step", 0.03),
            anneal_heads=tuple(cfg.get("anneal_heads", ("play",))),
            floors=dict(cfg.get("floors", {"play": 0.28})),
        )
        return cls(config=config, targets=d.get("targets"), alphas=d.get("alphas"))

    @classmethod
    def load(cls, path: str) -> "EntropyTargetController":
        with open(path) as f:
            return cls.from_dict(json.load(f))


__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "EntropyControllerConfig",
    "EntropyTargetController",
    "HEADS",
]
