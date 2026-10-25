"""The deployed AI's sampling configuration, from settings.

Four per-head temperatures (pick, partner, bury, play; 0 = argmax) and an
optional logit band make up a ``DeploySampling``
(sheepshead/agent/sampling.py). The same five numbers are the AI player's
identity in the database (``ai_player``), so a tempered deployment is a
different player from the argmax one for stats and the leaderboard. They
are deploy configuration beside the model path and label
(SHEEPSHEAD_TEMPERATURE_* / SHEEPSHEAD_LOGIT_BAND) because they are tuned
per checkpoint (Training_Program_Redesign §4.5).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from sheepshead.agent.sampling import HEAD_GROUPS, DeploySampling


@dataclass(frozen=True)
class SamplingSpec:
    """The five identity numbers, in column order."""

    pick: float
    partner: float
    bury: float
    play: float
    band: float | None

    @classmethod
    def from_settings(cls, settings) -> SamplingSpec:
        spec = cls(
            pick=float(settings.sheepshead_temperature_pick),
            partner=float(settings.sheepshead_temperature_partner),
            bury=float(settings.sheepshead_temperature_bury),
            play=float(settings.sheepshead_temperature_play),
            band=(
                None
                if settings.sheepshead_logit_band is None
                else float(settings.sheepshead_logit_band)
            ),
        )
        if min(spec.pick, spec.partner, spec.bury, spec.play) < 0.0:
            raise ValueError("SHEEPSHEAD_TEMPERATURE_* must be >= 0")
        if spec.band is not None and spec.band < 0.0:
            raise ValueError("SHEEPSHEAD_LOGIT_BAND must be >= 0")
        return spec

    @property
    def is_argmax(self) -> bool:
        return self.band is None and not any(
            (self.pick, self.partner, self.bury, self.play)
        )

    def columns(self) -> tuple[float, float, float, float, float | None]:
        return (self.pick, self.partner, self.bury, self.play, self.band)

    def sampler(self) -> DeploySampling | None:
        """A fresh sampler with its own random seed (one per table agent, so
        two tables never share a uniform stream), or None for pure argmax
        (the agent's default path, no per-decision host round trip)."""
        if self.is_argmax:
            return None
        temps: dict[str, float] = dict(
            zip(HEAD_GROUPS, (self.pick, self.partner, self.bury, self.play))
        )
        return DeploySampling(
            temperature=temps, band=self.band, seed=secrets.randbits(62)
        )

    def label(self) -> str:
        temps: dict[str, float] = dict(
            zip(HEAD_GROUPS, (self.pick, self.partner, self.bury, self.play))
        )
        return DeploySampling(temperature=temps, band=self.band).label()
