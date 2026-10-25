"""Deployment-time stochastic action selection: a per-head-group softmax
temperature with an optional logit band.

The trained policy's probabilities are a soft optimum (bidding heads) or a
search tilt (play head), so the deployed sampling distribution is chosen,
not inherited (Training_Program_Redesign §4.5). The operator's target is a
policy that picks genuinely tied actions at random, keeps some mass on
actions that are occasionally better than the argmax, and is otherwise
one-hot. Temperature scaling does exactly that on the logit scale: a gap
of δ nats between two actions becomes δ/T, so ties stay mixed at any T and
real gaps collapse as T falls. The band is the sharper cousin: actions
more than ``band`` nats below the best are dropped before sampling, so
the tail of "never" actions cannot be drawn at all while the near-tie
structure survives.

Head groups follow the trainer's entropy telemetry: ``pick`` (PICK/PASS),
``partner`` (ALONE, JD PARTNER, CALL …), ``bury`` (UNDER …, BURY …) and
``play`` (PLAY …). Every decision's legal set lies in one group, so the
temperature is a per-decision choice. T = 0 is the argmax.

Sampling uses inverse-CDF on the actions sorted by descending probability
with one uniform per decision from a reseedable generator: paired
evaluations (duplicate deals, temperature sweeps) that reseed per deal
then share the uniforms, and arms differing only in T are monotonically
coupled (a lower T can only move the draw toward the argmax for the same
uniform). ``PPOAgent.act(deterministic=True)`` consults the agent's
``deploy_sampling`` when one is set, so every evaluation harness and the
app play the deployed policy through one code path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from sheepshead.game import ACTIONS

HEAD_GROUPS = ("pick", "partner", "bury", "play")


def _head_of_action(name: str) -> str:
    if name in ("PICK", "PASS"):
        return "pick"
    if name.startswith(("ALONE", "JD PARTNER", "CALL ")):
        return "partner"
    if name.startswith(("UNDER ", "BURY ")):
        return "bury"
    return "play"


_HEAD_BY_ACTION_ID = {i + 1: _head_of_action(name) for i, name in enumerate(ACTIONS)}


def head_of(valid_actions: Sequence[int]) -> str:
    """Head group of a decision, from any of its (1-indexed) legal actions."""
    return _HEAD_BY_ACTION_ID[int(next(iter(valid_actions)))]


@dataclass
class DeploySampling:
    """Per-head-group temperatures (0 = argmax) and an optional logit band.

    Picklable (the generator is rebuilt from ``seed`` on first use and on
    ``reseed``), so it rides along into the h2h worker processes.
    """

    temperature: Mapping[str, float] = field(
        default_factory=lambda: {h: 0.0 for h in HEAD_GROUPS}
    )
    band: float | None = None  # nats below the best logit; None = no band
    seed: int = 0

    def __post_init__(self) -> None:
        unknown = set(self.temperature) - set(HEAD_GROUPS)
        if unknown:
            raise ValueError(f"unknown head groups {sorted(unknown)}")
        if any(t < 0.0 for t in self.temperature.values()):
            raise ValueError("temperatures must be >= 0")
        if self.band is not None and self.band < 0.0:
            raise ValueError("band must be >= 0")
        self._rng = random.Random(self.seed)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_rng", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._rng = random.Random(self.seed)

    def reseed(self, seed: int) -> None:
        """Restart the uniform stream (per deal, for common random numbers)."""
        self._rng = random.Random(seed)

    def label(self) -> str:
        temps = ",".join(
            f"{h}={self.temperature.get(h, 0.0):g}"
            for h in HEAD_GROUPS
            if self.temperature.get(h, 0.0) > 0.0
        )
        return (temps or "argmax") + (
            f" band={self.band:g}" if self.band is not None else ""
        )

    def distribution(
        self, log_probs: np.ndarray, valid_actions: Sequence[int]
    ) -> np.ndarray:
        """The deployed distribution over the full action vector (0-indexed),
        given the policy's masked log-probabilities and the legal set."""
        t = float(self.temperature.get(head_of(valid_actions), 0.0))
        legal = np.zeros(len(log_probs), dtype=bool)
        legal[[int(a) - 1 for a in valid_actions]] = True
        lp = np.where(legal, log_probs, -np.inf)
        best = lp.max()
        if self.band is not None:
            lp = np.where(lp >= best - self.band, lp, -np.inf)
        out = np.zeros(len(log_probs), dtype=np.float64)
        if t == 0.0:
            out[int(np.argmax(lp))] = 1.0
            return out
        z = (lp - best) / t
        w = np.exp(z, where=np.isfinite(z), out=np.zeros_like(z))
        return w / w.sum()

    def sample(self, log_probs: np.ndarray, valid_actions: Sequence[int]) -> int:
        """Draw a 0-indexed action by inverse CDF over descending probability
        (one uniform per decision; T = 0 returns the argmax without a draw)."""
        p = self.distribution(log_probs, valid_actions)
        if p.max() >= 1.0:
            return int(np.argmax(p))
        order = np.argsort(-p, kind="stable")
        cdf = np.cumsum(p[order])
        u = self._rng.random()
        k = int(np.searchsorted(cdf, u, side="right"))
        return int(order[min(k, len(order) - 1)])
