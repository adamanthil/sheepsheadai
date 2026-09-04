"""Pooling schema-2 corpora (§20.13): game-index offsets and manifest sums."""

from __future__ import annotations

import json
import os

import torch

from sheepshead.analysis.merge_corpora import main, merge_manifests, offset_games


def shard(game_ids):
    return {
        "episodes": [
            [{"kind": "action", "distill_set": "none"}]
            for _ in game_ids
            for _ in range(5)
        ],
        "games": [{"game": g, "mode": "called"} for g in game_ids],
    }


def manifest(games, searched, ckpt="theta.pt"):
    return {
        "ckpt": ckpt,
        "row_schema": 2,
        "games": games,
        "kept_games": games,
        "episodes": 5 * games,
        "committee_acted_nodes": 0,
        "classes": {"std|t0-defender-lead": {"nodes": searched, "searched": searched}},
        "gap_percentiles": {"p50": 0.1},
        "shards": ["corpus_0000.pt"],
    }


def test_offset_games_shifts_only_the_game_index():
    out = offset_games(shard([0, 1]), 10)
    assert [g["game"] for g in out["games"]] == [10, 11]
    assert out["games"][0]["mode"] == "called"
    assert len(out["episodes"]) == 10


def test_merge_manifests_sums_counters_and_checks_agreement():
    merged = merge_manifests(
        [manifest(2, 7), manifest(3, 5)],
        ["a", "b"],
        ["corpus_0000.pt", "corpus_0001.pt"],
    )
    assert merged["games"] == 5 and merged["episodes"] == 25
    assert merged["classes"]["std|t0-defender-lead"]["searched"] == 12
    assert "gap_percentiles" not in merged
    assert merged["shards"] == ["corpus_0000.pt", "corpus_0001.pt"]


def test_merge_manifests_rejects_mismatched_checkpoints():
    import pytest

    with pytest.raises(SystemExit):
        merge_manifests(
            [manifest(1, 1), manifest(1, 1, ckpt="other.pt")], ["a", "b"], []
        )


def test_cli_renumbers_shards_and_offsets_games(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    for d, ids in ((a, [0, 1]), (b, [0, 1, 2])):
        d.mkdir()
        torch.save(shard(ids), d / "corpus_0000.pt")
        (d / "manifest.json").write_text(json.dumps(manifest(len(ids), len(ids))))
    assert (
        main(["--corpus-dir", str(a), "--corpus-dir", str(b), "--out-dir", str(out)])
        == 0
    )
    names = sorted(f for f in os.listdir(out) if f.endswith(".pt"))
    assert names == ["corpus_0000.pt", "corpus_0001.pt"]
    second = torch.load(out / "corpus_0001.pt", weights_only=False)
    assert [g["game"] for g in second["games"]] == [2, 3, 4]
    merged = json.loads((out / "manifest.json").read_text())
    assert merged["games"] == 5 and len(merged["merged_from"]) == 2
