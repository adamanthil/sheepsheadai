"""fetch_models: the committed manifest lints clean; downloads verify size and
sha256 before landing; a bad cache or a tampered asset fails loudly;
managed-file verification ignores unmanaged paths; panels resolve to
models/<key>.pt in frozen order. Downloads use a file:// release base."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sheepshead.analysis.panels import panel_files
from sheepshead.devtools.fetch_models import (
    ArtifactError,
    fetch,
    lint,
    load_manifest,
    select,
    verify,
    verify_managed,
)

TAG = "artifact/test"
PAYLOAD = b"not really a checkpoint"


def _manifest(release_dir: Path) -> dict:
    return {
        "release_base": release_dir.as_uri(),
        "artifacts": {
            "toy": {
                "tag": TAG,
                "asset": "toy.pt",
                "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
                "bytes": len(PAYLOAD),
                "roles": ["deploy"],
                "arch": "full",
                "provenance": "test",
                "sampling": {
                    "pick": 0,
                    "partner": 0,
                    "bury": 0,
                    "play": 0.5,
                    "band": 1,
                },
            }
        },
        "panels": {},
    }


@pytest.fixture
def release(tmp_path: Path) -> tuple[dict, Path]:
    """A one-asset release served from disk, and an empty models dir."""
    asset = tmp_path / "release" / TAG / "toy.pt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(PAYLOAD)
    models = tmp_path / "models"
    models.mkdir()
    manifest = _manifest(tmp_path / "release")
    (models / "manifest.json").write_text(json.dumps(manifest))
    return manifest, models


def test_committed_manifest_lints_clean():
    assert lint(load_manifest()) == []


def test_fetch_downloads_once_and_verifies(release):
    manifest, models = release
    assert fetch("toy", manifest, models) is True
    assert (models / "toy.pt").read_bytes() == PAYLOAD
    assert fetch("toy", manifest, models) is False
    assert verify("toy", manifest, models) == models / "toy.pt"


def test_fetch_replaces_a_corrupted_cache(release):
    manifest, models = release
    (models / "toy.pt").write_bytes(b"corrupt")
    with pytest.raises(ArtifactError, match="hashes to"):
        verify("toy", manifest, models)
    assert fetch("toy", manifest, models) is True
    assert (models / "toy.pt").read_bytes() == PAYLOAD


def test_tampered_asset_never_lands(release, tmp_path):
    manifest, models = release
    (tmp_path / "release" / TAG / "toy.pt").write_bytes(b"tampered payload")
    with pytest.raises(ArtifactError, match="manifest pins"):
        fetch("toy", manifest, models)
    assert sorted(p.name for p in models.iterdir()) == ["manifest.json"]


def test_verify_managed_checks_only_manifest_files(release, tmp_path):
    _, models = release
    (models / "toy.pt").write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):
        verify_managed(models / "toy.pt", models)
    elsewhere = tmp_path / "toy.pt"
    elsewhere.write_bytes(b"anything")
    verify_managed(elsewhere, models)
    verify_managed(models / "unlisted.pt", models)


def test_select_by_key_and_role():
    manifest = load_manifest()
    assert select(manifest, None, None) == list(manifest["artifacts"])
    assert select(manifest, ["v2-release"], ["deploy"]) == [
        "v2-release",
        "recall-rc-theta6",
    ]
    with pytest.raises(ArtifactError):
        select(manifest, ["no-such-key"], None)
    with pytest.raises(ArtifactError):
        select(manifest, None, ["no-such-role"])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: m["artifacts"]["toy"].update(roles=["panel-a"]), "exactly one"),
        (lambda m: m["artifacts"]["toy"]["sampling"].pop("band"), "sampling"),
        (lambda m: m["artifacts"]["toy"].update(asset="other.pt"), "toy.pt"),
        (lambda m: m["artifacts"]["toy"].update(sha256="abc"), "64"),
        (lambda m: m["artifacts"]["toy"].update(arch="nope"), "not registered"),
        (lambda m: m["panels"].update({"PANEL-A": ["ghost"]}), "unknown key"),
    ],
)
def test_lint_rejects(release, mutate, message):
    manifest = copy.deepcopy(release[0])
    manifest["release_base"] = "https://example.invalid"
    assert lint(manifest) == []
    mutate(manifest)
    assert any(message in e for e in lint(manifest))


def test_panel_files_follow_manifest_order():
    manifest = load_manifest()
    for name, keys in manifest["panels"].items():
        assert panel_files(name) == [f"models/{k}.pt" for k in keys]
