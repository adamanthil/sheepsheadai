#!/usr/bin/env python3
"""Orchestrator per-generation health verdicts on synthetic telemetry CSVs.

Covers the 2026-07-18 amendment semantics: greedy-gate streaks warn but
never halt, only the leaster trend halts, and every verdict is one-shot
(recorded in state, never re-litigated on relaunch).
"""

import csv
import os
from types import SimpleNamespace

import pytest

from sheepshead.training.run_extended_league import NeedsReview, Orchestrator

GREEDY_FIELDS = [
    "episode",
    "pick_rate",
    "alone_rate",
    "leaster_rate",
    "t0_trump_lead_rate",
    "t0_def_leads",
    "play_logit_spread_med",
    "play_nodes",
    "games",
]

HEALTHY = {
    "pick_rate": 35.0,
    "alone_rate": 10.0,
    "leaster_rate": 2.0,
    "t0_trump_lead_rate": 1.0,
    "t0_def_leads": 90,
    "play_logit_spread_med": 1.1,
    "play_nodes": 3000,
    "games": 200,
}


def make_orch(tmp_path, monkeypatch, **arg_overrides):
    monkeypatch.chdir(tmp_path)
    defaults = dict(
        run_name="health_t",
        min_generations=4,
        max_generations=12,
        anchor_coeff=1.0,
        main_episodes=1_000_000,
        ignore_health_halt=False,
    )
    args = SimpleNamespace(**{**defaults, **arg_overrides})
    orch = Orchestrator(args)
    # Baseline normally measured by ensure_baseline_health; pin it so the
    # relative ALONE limit is deterministic (max(20, 25 + 5) = 30).
    orch.state["baseline_health"] = {"alone_rate": 25.0}
    os.makedirs(orch.ckpt_dir, exist_ok=True)
    return orch


def write_greedy(orch, rows):
    """rows: list of (episode, overrides) applied on top of HEALTHY."""
    with open(os.path.join(orch.ckpt_dir, "greedy_health.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GREEDY_FIELDS)
        w.writeheader()
        for episode, overrides in rows:
            w.writerow({**HEALTHY, "episode": episode, **overrides})


def write_progress(orch, leaster_rates, start_episode=25_000, step=25_000):
    path = os.path.join(orch.ckpt_dir, "league_training_progress.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "leaster_rate"])
        for i, rate in enumerate(leaster_rates):
            w.writerow([start_episode + i * step, rate])


class TestGateStreaksWarn:
    def test_streak_warns_but_does_not_halt(self, tmp_path, monkeypatch):
        orch = make_orch(tmp_path, monkeypatch)
        # Three consecutive trump-lead probes over the 8% gate (the exact
        # stage-1 gen-2 pattern) plus an isolated pick dip.
        write_greedy(
            orch,
            [
                (450_000, {"pick_rate": 14.0}),
                (500_000, {}),
                (600_000, {"t0_trump_lead_rate": 9.8}),
                (650_000, {"t0_trump_lead_rate": 14.1}),
                (700_000, {"t0_trump_lead_rate": 14.3}),
                (750_000, {}),
            ],
        )
        orch.health_checks(1)  # must not raise
        health = orch.state["generations"]["1"]["health"]
        assert health["halt"] is None
        assert health["warnings"] == ["trump_lead: 3 consecutive greedy-probe violations"]

    def test_short_streaks_and_other_generations_ignored(self, tmp_path, monkeypatch):
        orch = make_orch(tmp_path, monkeypatch)
        write_greedy(
            orch,
            [
                (500_000, {"pick_rate": 10.0}),
                (550_000, {"pick_rate": 10.0}),  # streak of 2: below threshold
                (600_000, {}),
                (1_200_000, {"pick_rate": 10.0}),  # gen 2, not gen 1
                (1_250_000, {"pick_rate": 10.0}),
                (1_300_000, {"pick_rate": 10.0}),
            ],
        )
        orch.health_checks(1)
        assert orch.state["generations"]["1"]["health"]["warnings"] == []


class TestLeasterHalt:
    RISING = [0.05] * 20 + [0.45] * 20

    def test_halts_once_then_relaunch_continues(self, tmp_path, monkeypatch):
        orch = make_orch(tmp_path, monkeypatch)
        write_progress(orch, self.RISING)
        with pytest.raises(NeedsReview):
            orch.health_checks(1)
        health = orch.state["generations"]["1"]["health"]
        assert health["halt"] is not None
        # Same data, fresh orchestrator (relaunch): verdict on record, no halt.
        relaunch = Orchestrator(orch.args)
        relaunch.health_checks(1)

    def test_ignore_flag_records_without_halt(self, tmp_path, monkeypatch):
        orch = make_orch(tmp_path, monkeypatch, ignore_health_halt=True)
        write_progress(orch, self.RISING)
        orch.health_checks(1)  # must not raise
        assert orch.state["generations"]["1"]["health"]["halt"] is not None

    def test_flat_leaster_rate_is_healthy(self, tmp_path, monkeypatch):
        orch = make_orch(tmp_path, monkeypatch)
        write_progress(orch, [0.35] * 40)  # high but not rising
        orch.health_checks(1)
        assert orch.state["generations"]["1"]["health"]["halt"] is None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
