#!/usr/bin/env python3
"""LeasterWatchdog invariants: min-sample gating, engage/release hysteresis,
re-engagement, and default-off wiring in the trainer."""

import inspect
import os
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead.training.train_selfplay_ppo import LeasterWatchdog, train_ppo


def _window(rate, n=3000):
    ones = int(round(n * rate))
    return deque([1] * ones + [0] * (n - ones), maxlen=n)


class TestLeasterWatchdog(unittest.TestCase):
    def test_no_engage_below_min_samples(self):
        wd = LeasterWatchdog()
        tiny = deque([1] * (LeasterWatchdog.MIN_SAMPLES - 1))
        self.assertIsNone(wd.observe(tiny))
        self.assertFalse(wd.engaged)

    def test_engage_release_hysteresis(self):
        wd = LeasterWatchdog()
        self.assertEqual(wd.observe(_window(0.95)), "engaged")
        self.assertTrue(wd.engaged)
        # Repeated high rate: no duplicate transition, stays engaged.
        self.assertIsNone(wd.observe(_window(0.95)))
        # In the hysteresis band (30-90%): still engaged, no transition.
        self.assertIsNone(wd.observe(_window(0.50)))
        self.assertTrue(wd.engaged)
        # Below the release threshold: releases.
        self.assertEqual(wd.observe(_window(0.10)), "released")
        self.assertFalse(wd.engaged)
        # Healthy rate never re-engages...
        self.assertIsNone(wd.observe(_window(0.05)))
        # ...but a relapse does.
        self.assertEqual(wd.observe(_window(0.92)), "engaged")

    def test_no_engage_at_healthy_rates(self):
        wd = LeasterWatchdog()
        for rate in (0.0, 0.05, 0.30, 0.89):
            self.assertIsNone(wd.observe(_window(rate)))
            self.assertFalse(wd.engaged)

    def test_trainer_default_off(self):
        sig = inspect.signature(train_ppo)
        self.assertIs(sig.parameters["leaster_watchdog"].default, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
