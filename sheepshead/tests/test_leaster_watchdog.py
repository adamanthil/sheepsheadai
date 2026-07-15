#!/usr/bin/env python3
"""LeasterWatchdog invariants: min-sample gating, engage/release hysteresis,
re-engagement, and default-off wiring in the trainer."""

import inspect
from collections import deque

from sheepshead.training.train_selfplay_ppo import LeasterWatchdog, train_ppo


def _window(rate, n=3000):
    ones = int(round(n * rate))
    return deque([1] * ones + [0] * (n - ones), maxlen=n)


class TestLeasterWatchdog:
    def test_no_engage_below_min_samples(self):
        wd = LeasterWatchdog()
        tiny = deque([1] * (LeasterWatchdog.MIN_SAMPLES - 1))
        assert wd.observe(tiny) is None
        assert not wd.engaged

    def test_engage_release_hysteresis(self):
        wd = LeasterWatchdog()
        assert wd.observe(_window(0.95)) == "engaged"
        assert wd.engaged
        # Repeated high rate: no duplicate transition, stays engaged.
        assert wd.observe(_window(0.95)) is None
        # In the hysteresis band (30-90%): still engaged, no transition.
        assert wd.observe(_window(0.50)) is None
        assert wd.engaged
        # Below the release threshold: releases.
        assert wd.observe(_window(0.10)) == "released"
        assert not wd.engaged
        # Healthy rate never re-engages...
        assert wd.observe(_window(0.05)) is None
        # ...but a relapse does.
        assert wd.observe(_window(0.92)) == "engaged"

    def test_no_engage_at_healthy_rates(self):
        wd = LeasterWatchdog()
        for rate in (0.0, 0.05, 0.30, 0.89):
            assert wd.observe(_window(rate)) is None
            assert not wd.engaged

    def test_trainer_default_off(self):
        sig = inspect.signature(train_ppo)
        assert sig.parameters["leaster_watchdog"].default is False


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
