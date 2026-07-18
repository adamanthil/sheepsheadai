"""Selective pick-entropy kick against the always-PASS collapse.

Shared by the self-play and league trainers (``--leaster-watchdog`` on
both, default off; enable uniformly across the arms of any comparison).
"""


class LeasterWatchdog:
    """Selective pick-entropy kick against the always-PASS collapse.

    Every from-scratch shaped-reward self-play run collapses to ~100%
    leasters within the first few thousand episodes: play skill is
    terrible at init, so picking has negative EV and every seat learns to
    PASS (see the leaster-rate scan in
    notebooks/Architecture_Ablation_202607.md; `analysis/leaster_scan.py`).
    The stage-1 league runs (July 2026) showed the same attractor can be
    re-entered from a well-trained policy once the bidding anchor is
    released, so the league trainer takes the watchdog as an option too.
    The scheduled entropy bonus cannot prevent the freeze because its
    gradient vanishes as the pick head approaches determinism — so this
    watchdog fires EARLY, at the 90% leaster crossing while pick entropy
    is still alive, and multiplies the pick head's scheduled entropy
    coefficient until the rolling rate falls back below 30% (hysteresis).
    Bidding rewards are untouched, and the kick is inert outside the
    pathological region — the pick head anneals normally once released.
    """

    ENGAGE_RATE = 0.90
    RELEASE_RATE = 0.30
    KICK = 10.0
    MIN_SAMPLES = 1000

    def __init__(self):
        self.engaged = False
        self.engaged_updates = 0

    def observe(self, leaster_window) -> str | None:
        """Advance state from the rolling 0/1 leaster window. Returns
        "engaged"/"released" on a transition, else None."""
        if len(leaster_window) < self.MIN_SAMPLES:
            return None
        rate = sum(leaster_window) / len(leaster_window)
        if not self.engaged and rate >= self.ENGAGE_RATE:
            self.engaged = True
            return "engaged"
        if self.engaged and rate < self.RELEASE_RATE:
            self.engaged = False
            return "released"
        return None

    def tick(self, agent, leaster_window) -> None:
        """Per-update step: advance state, log transitions, and while
        engaged multiply the agent's freshly scheduled pick entropy
        coefficient. Call after the entropy schedules have been applied
        (they overwrite the coefficient each update, so the kick is a
        flat ×KICK, not compounding)."""
        transition = self.observe(leaster_window)
        if transition is not None:
            rate = sum(leaster_window) / len(leaster_window)
            if transition == "engaged":
                print(
                    f"🚨 Leaster watchdog ENGAGED (rate {rate:.0%}): "
                    f"pick entropy coeff ×{self.KICK:g} "
                    f"until rate < {self.RELEASE_RATE:.0%}"
                )
            else:
                print(
                    f"✅ Leaster watchdog released (rate {rate:.0%}) "
                    f"after {self.engaged_updates} kicked updates"
                )
        if self.engaged:
            agent.entropy_coeff_pick *= self.KICK
            self.engaged_updates += 1
