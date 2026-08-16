#!/usr/bin/env python3
"""Target-entropy controller for the league trainer (v2, signed).

Replaces the clock-based entropy-coefficient schedule with feedback control
of the MEASURED policy entropy, two loops:

Inner loop (here, per update): each head's entropy coefficient is an
integral controller holding the measured normalized entropy
(``stats["head_entropy_norm"]``, theta_old, H/ln n_legal) at a target —
SAC's automatic temperature adjustment (Haarnoja et al., arXiv:1812.05905
§5) in its discrete fraction-of-max form (Christodoulou, arXiv:1910.07207).
Targets initialize BUMPLESSLY: a head with no explicit target adopts the
first measured value, so switch-on changes nothing at t=0 (Astrom &
Wittenmark, *Adaptive Control*, 2nd ed. 1995, ch. 9).

v1 vs v2 — why the step is signed and linear. v1 stepped LOG-alpha
(``alpha <- alpha * exp(eta * err)``), which keeps alpha > 0 by
construction: the controller could only ever *reduce* the regularizer
toward its floor, and against a term that INJECTS entropy it saturated
there and lost authority (the §12.20 diagnosis in
notebooks/Search_Teacher_Design_202608.md; fix specified in
notebooks/CE_Teacher_Design_202608.md §4). v2 steps alpha in LINEAR
space, ``alpha <- clip(alpha + eta_lin * err, alpha_min, alpha_max)``,
with ``alpha_min < 0``. A negative coefficient is an entropy PENALTY —
active sharpening — so the loop retains authority in both directions.
The negative range is a BACKSTOP, not the operating point: the CE
teacher is approximately entropy-neutral (prior-preserving at ties), so
the expected trajectory hovers near the legacy positive values and alpha
sign flips are logged as telemetry precisely because they are the signal
that something is injecting entropy.

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
confound from the stop rule). This loop is UNCHANGED from v1.

Floors are never zero: imperfect-information equilibria are genuinely
mixed (Sokota et al., arXiv:2206.05825; quantal-response equilibria), and
a deterministic pick threshold would leak hand information.

State (targets + coefficients) persists as a JSON sidecar next to the
checkpoints so crash-resume and generation handoffs are seamless; the
orchestrator edits the same file to step targets between generations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

HEADS = ("pick", "partner", "bury", "play")


@dataclass(frozen=True)
class EntropyControllerConfig:
    """Inner-gain and outer-step constants (v2 per CE_Teacher_Design_202608.md
    §4; outer-loop values pre-registered 2026-07-28,
    notebooks/Learning_System_Redesign_202607.md Phase 2).

    eta_lin: LINEAR integral gain — d(alpha) = eta_lin * (target - measured)
        per update. Calibrated to reproduce the legacy log-space response at
        the reference operating point alpha ~= 0.15: v1's eta=1.0 gave
        d(log alpha) = err, i.e. d_alpha ~= 0.15 * err there, hence 0.15.
        A sustained error the size of the backfill's organic per-generation
        play drift (0.057) therefore moves alpha ~5.9%/update at that point,
        settling in ~10-20 of a generation's ~61 updates; per-update
        measurement noise (SE ~0.002-0.004 at 16k rows) contributes ~0.4%
        jitter — two orders below the clamp.
    max_step: per-update |d alpha| clamp (safety against transients).
        Calibrated the same way: v1's max_log_step=0.1 at alpha=0.15 allowed
        0.15*(exp(0.1)-1) ~= 0.0158 of absolute movement, hence 0.015.
    alpha_min / alpha_max: coefficient bounds, the SAME for every head
        (v1's per-head bounds derived from the legacy schedule are gone).
        alpha_min < 0 is the backstop that gives the loop authority against
        an entropy-injecting term — v1 could not go there and saturated at
        its floor instead (§12.20). alpha_max is tighter than the legacy 4x
        cap precisely because the negative range now exists.
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

    eta_lin: float = 0.15
    max_step: float = 0.015
    alpha_min: float = -0.05
    alpha_max: float = 0.25
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
        sign_flips: dict | None = None,
    ):
        self.config = config or EntropyControllerConfig()
        # None target = bumpless: adopt the first measured value.
        self.targets: dict = {h: None for h in HEADS}
        if targets:
            self.targets.update({h: targets[h] for h in targets if h in HEADS})
        self.alphas: dict = dict(alphas) if alphas else {}
        # Telemetry: how often each head's coefficient crossed zero, i.e.
        # switched between regularizing and actively sharpening.
        self.sign_flips: dict = {h: 0 for h in HEADS}
        if sign_flips:
            self.sign_flips.update(
                {h: int(sign_flips[h]) for h in sign_flips if h in HEADS}
            )

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
        per-head LINEAR alpha deltas actually applied (telemetry)."""
        cfg = self.config
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
            delta = max(-cfg.max_step, min(cfg.max_step, cfg.eta_lin * err))
            old = self.alphas[h]
            new = max(cfg.alpha_min, min(cfg.alpha_max, old + delta))
            if (old >= 0.0) != (new >= 0.0):
                self.sign_flips[h] = self.sign_flips.get(h, 0) + 1
            self.alphas[h] = new
            deltas[h] = new - old
        return deltas

    # ------------------------------------------------------------------ #
    # Outer loop (generation boundaries) — unchanged from v1
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
            "sign_flips": self.sign_flips,
            "config": {
                "eta_lin": self.config.eta_lin,
                "max_step": self.config.max_step,
                "alpha_min": self.config.alpha_min,
                "alpha_max": self.config.alpha_max,
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
        """Rebuild from a sidecar, V1-COMPATIBLE.

        A v1 sidecar carries the log-space gains ``eta`` / ``max_log_step``
        and no ``eta_lin`` / ``max_step`` / ``alpha_min`` / ``alpha_max``.
        Those gains are in different units and do not convert per-head, so
        they are IGNORED and the v2 defaults apply; everything that is
        state rather than gain — targets, alphas, and the outer-loop
        settings retain/min_step/anneal_heads/floors — carries over
        unchanged. A mid-run v1 -> v2 upgrade is therefore bumpless in
        alpha and continues the same target ladder. ``sign_flips`` is
        absent in v1 and starts at zero.
        """
        cfg = d.get("config", {})
        defaults = EntropyControllerConfig()
        config = EntropyControllerConfig(
            eta_lin=cfg.get("eta_lin", defaults.eta_lin),
            max_step=cfg.get("max_step", defaults.max_step),
            alpha_min=cfg.get("alpha_min", defaults.alpha_min),
            alpha_max=cfg.get("alpha_max", defaults.alpha_max),
            retain=cfg.get("retain", defaults.retain),
            min_step=cfg.get("min_step", defaults.min_step),
            anneal_heads=tuple(cfg.get("anneal_heads", defaults.anneal_heads)),
            floors=dict(cfg.get("floors", {"play": 0.28})),
        )
        return cls(
            config=config,
            targets=d.get("targets"),
            alphas=d.get("alphas"),
            sign_flips=d.get("sign_flips"),
        )

    @classmethod
    def load(cls, path: str) -> "EntropyTargetController":
        with open(path) as f:
            return cls.from_dict(json.load(f))


__all__ = [
    "EntropyControllerConfig",
    "EntropyTargetController",
    "HEADS",
]
