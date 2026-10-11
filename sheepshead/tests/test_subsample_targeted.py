"""Matched-composition subsample of a targeted corpus (§20.12 matched-volume
control)."""

from __future__ import annotations

import random

from sheepshead.analysis.subsample_targeted import subsample_episodes


def row(node_class: str, dset: str, weight: float | None = None) -> dict:
    return {
        "kind": "action",
        "node_class": node_class,
        "distill_set": dset,
        "search_weight": weight,
    }


class TestSubsampleTargeted:
    def test_only_targeted_follow_rows_are_demoted(self):
        rng = random.Random(3)
        episodes = [
            [
                row("std|t0-defender-lead", "targeted", 1.5),
                row("std|t1-defender-follow", "targeted", 0.5),
                row("std|t3-defender-follow", "none"),
                row("bid|pick", "retention"),
                {"kind": "trick"},
            ]
            for _ in range(200)
        ]
        _, counts, _ = subsample_episodes(episodes, follow_keep=0.5, rng=rng)
        # leads and untargeted rows are untouched
        assert counts["lead_targeted_kept"] == 200
        assert all(ep[0]["distill_set"] == "targeted" for ep in episodes)
        assert all(ep[2]["distill_set"] == "none" for ep in episodes)
        assert all(ep[3]["distill_set"] == "retention" for ep in episodes)
        # targeted follows kept at roughly the requested rate
        kept = counts["follow_targeted_kept"]
        assert kept + counts["follow_targeted_dropped"] == 200
        assert 60 <= kept <= 140
        assert all(ep[1]["distill_set"] in ("targeted", "none") for ep in episodes)

    def test_target_weights_renormalized_to_mean_one(self):
        rng = random.Random(0)
        episodes = [
            [row("std|t0-defender-lead", "targeted", 1.5)],
            [row("std|t1-defender-follow", "targeted", 0.5)],
            [row("std|t1-defender-follow", "targeted", 0.5)],
            [row("std|t1-defender-follow", "targeted", 0.5)],
        ]
        subsample_episodes(episodes, follow_keep=0.0, rng=rng)
        remaining = [
            ev["search_weight"]
            for ep in episodes
            for ev in ep
            if ev["distill_set"] == "targeted"
        ]
        assert remaining == [1.0]

    def test_follow_keep_one_is_identity(self):
        rng = random.Random(0)
        episodes = [[row("std|t1-defender-follow", "targeted", 1.0)] for _ in range(50)]
        _, counts, _ = subsample_episodes(episodes, follow_keep=1.0, rng=rng)
        assert counts["follow_targeted_kept"] == 50
        assert all(ep[0]["distill_set"] == "targeted" for ep in episodes)
