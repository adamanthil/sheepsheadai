"""Corpus resume (distill_corpus --resume): a killed generator continues
in place, playing exactly the game indices its flushed shards lack, and
refuses to extend a corpus from another checkpoint or recipe."""

import json
import os
import sys

import pytest
import torch

import sheepshead.training.distill_corpus as dc

GAMES = 23
SHARD_GAMES = 4


class _Crash(Exception):
    pass


class _FakePool:
    """Completes tasks out of order (neighbours swapped across shard
    boundaries, as a worker pool does) and optionally dies after
    ``crash_after`` results."""

    def __init__(self, crash_after):
        self.crash_after = crash_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, tasks, chunksize=1):
        tasks = list(tasks)
        for i in range(1, len(tasks) - 1, 2):
            tasks[i], tasks[i + 1] = tasks[i + 1], tasks[i]
        for n, task in enumerate(tasks):
            if self.crash_after is not None and n == self.crash_after:
                raise _Crash
            yield fn(task)


class _FakeContext:
    def __init__(self, crash_after):
        self.crash_after = crash_after

    def Pool(self, processes, initializer, initargs):
        return _FakePool(self.crash_after)


def _fake_game(task):
    g = task[0]
    return {
        "game": g,
        "mode": task[1],
        "is_leaster": False,
        "alone_called": False,
        "committee_act": bool(task[2]),
        "committee_acted": 0,
        "episodes": [[{"kind": "action", "game": g}]],
        "counts": {
            "std|play": {
                "nodes": 1,
                "searched": 1,
                "resolved": 1,
                "unresolved": 0,
                "failed": 0,
            }
        },
        "gaps": [0.01],
        "telemetry": [{"game": g, "w": 0.5, "gap": 0.01}],
    }


def _run(monkeypatch, tmp_path, *extra, crash_after=None):
    ckpt = tmp_path / "theta.pt"
    if not ckpt.exists():
        ckpt.write_bytes(b"theta")
    out = tmp_path / "corpus"
    argv = [
        "distill_corpus",
        "--ckpt",
        str(ckpt),
        "--out-dir",
        str(out),
        "--games",
        str(GAMES),
        "--shard-games",
        str(SHARD_GAMES),
        "--node-telemetry",
        str(out / "nodes.jsonl"),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(dc, "get_context", lambda _: _FakeContext(crash_after))
    monkeypatch.setattr(dc, "play_corpus_game", _fake_game)
    return dc.main()


def _games_in_shards(out):
    games = []
    for f in sorted(os.listdir(out)):
        if f.startswith("corpus_") and f.endswith(".pt"):
            games += [g["game"] for g in torch.load(out / f)["games"]]
    return games


def test_resume_plays_exactly_the_unflushed_games(tmp_path, monkeypatch):
    out = tmp_path / "corpus"
    with pytest.raises(_Crash):
        _run(monkeypatch, tmp_path, "--resume", crash_after=11)
    manifest = json.loads((out / "manifest.json").read_text())
    flushed = {g for s in manifest["shards"] for g in s["games"]}
    # Out-of-order completion: the flushed set is not an index prefix.
    assert len(flushed) == 8 and flushed != set(range(8))

    assert _run(monkeypatch, tmp_path, "--resume") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    games = _games_in_shards(out)
    assert sorted(games) == list(range(GAMES))  # every index once
    assert manifest["kept_games"] == manifest["games"] == GAMES
    assert manifest["episodes"] == GAMES
    assert manifest["classes"]["std|play"]["searched"] == GAMES
    assert manifest["resumes"][0]["games_done"] == 8
    telemetry = [json.loads(line)["game"] for line in open(out / "nodes.jsonl")]
    assert sorted(telemetry) == list(range(GAMES))


def test_resume_reads_legacy_shards_without_game_lists(tmp_path, monkeypatch):
    out = tmp_path / "corpus"
    with pytest.raises(_Crash):
        _run(monkeypatch, tmp_path, crash_after=9)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = {g for s in manifest["shards"] for g in s.pop("games")}
    path.write_text(json.dumps(manifest))
    assert dc.flushed_games(str(out), manifest) == expected


def test_resume_without_a_manifest_starts_fresh(tmp_path, monkeypatch):
    assert _run(monkeypatch, tmp_path, "--resume") == 0
    assert sorted(_games_in_shards(tmp_path / "corpus")) == list(range(GAMES))


def test_resume_refuses_another_checkpoint_or_recipe(tmp_path, monkeypatch):
    with pytest.raises(_Crash):
        _run(monkeypatch, tmp_path, crash_after=9)
    with pytest.raises(SystemExit, match="p_base 0.1 -> 0.2"):
        _run(monkeypatch, tmp_path, "--resume", "--p-base", "0.2")
    # Operational knobs may change across a resume.
    with pytest.raises(_Crash):
        _run(monkeypatch, tmp_path, "--resume", "--workers", "3", crash_after=2)
    (tmp_path / "theta.pt").write_bytes(b"another theta")
    with pytest.raises(SystemExit, match="checkpoint"):
        _run(monkeypatch, tmp_path, "--resume")
