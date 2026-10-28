"""Fetch and verify the model artifacts pinned in ``models/manifest.json``.

Every checkpoint the repository depends on — the deployed model and the
frozen evaluation panels — is a GitHub Release asset (tag namespace
``artifact/<name>``, immutable), pinned in the committed manifest by sha256
and size. This tool downloads the selected artifacts into ``models/``
(gitignored) and verifies them; a file that does not hash to the manifest
is never used. Stdlib only, so it runs before ``uv sync`` has anything to
offer. See notebooks/Model_Artifacts_Plan_202610.md.

    uv run python -m sheepshead.eval.fetch_models                # everything
    uv run python -m sheepshead.eval.fetch_models --role panel-a --role panel-b
    uv run python -m sheepshead.eval.fetch_models --key recall-rc-theta6
    uv run python -m sheepshead.eval.fetch_models --check        # verify only, no network
    uv run python -m sheepshead.eval.fetch_models --lint         # manifest shape (CI)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.json"

SAMPLING_KEYS = ("pick", "partner", "bury", "play", "band")
_CHUNK = 1 << 20


class ArtifactError(Exception):
    """A missing or corrupted artifact, or an unknown key."""


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _entry(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        return manifest["artifacts"][key]
    except KeyError:
        raise ArtifactError(f"unknown artifact key: {key}") from None


def verify(key: str, manifest: dict[str, Any], models_dir: Path = MODELS_DIR) -> Path:
    """Path of ``key``'s file if it exists and hashes to the manifest."""
    entry = _entry(manifest, key)
    path = models_dir / entry["asset"]
    if not path.exists():
        raise ArtifactError(
            f"{key}: {path} is missing; run "
            f"`uv run python -m sheepshead.eval.fetch_models --key {key}`"
        )
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ArtifactError(
            f"{key}: {path} hashes to {digest[:12]}, manifest pins "
            f"{entry['sha256'][:12]}; delete it and fetch again"
        )
    return path


def verify_managed(path: Path, models_dir: Path = MODELS_DIR) -> None:
    """Verify ``path`` if it is a manifest artifact under ``models_dir``.

    Paths outside the models directory (run checkpoints, scratch files) pass
    through unchecked: only manifest-managed files have a pinned hash.
    """
    path = path.resolve()
    if path.parent != models_dir.resolve():
        return
    manifest = load_manifest(models_dir / "manifest.json")
    for key, entry in manifest["artifacts"].items():
        if entry["asset"] == path.name:
            verify(key, manifest, models_dir)
            return


def fetch(key: str, manifest: dict[str, Any], models_dir: Path = MODELS_DIR) -> bool:
    """Ensure ``key`` is present and verified; True if it was downloaded.

    Downloads into a temp file in ``models_dir``, checks size and sha256
    while streaming, and renames into place only on a match, so a partial
    or tampered download never lands under the asset's name.
    """
    entry = _entry(manifest, key)
    path = models_dir / entry["asset"]
    if path.exists() and sha256_file(path) == entry["sha256"]:
        return False
    url = f"{manifest['release_base']}/{entry['tag']}/{entry['asset']}"
    models_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=models_dir, prefix=f".{entry['asset']}.")
    tmp = Path(tmp_name)
    try:
        h = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            while chunk := resp.read(_CHUNK):
                h.update(chunk)
                size += len(chunk)
                out.write(chunk)
        if size != entry["bytes"] or h.hexdigest() != entry["sha256"]:
            raise ArtifactError(
                f"{key}: download from {url} is {size} bytes / "
                f"{h.hexdigest()[:12]}, manifest pins {entry['bytes']} / "
                f"{entry['sha256'][:12]}"
            )
        tmp.chmod(0o644)  # mkstemp creates 0600
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def lint(manifest: dict[str, Any]) -> list[str]:
    """Shape errors in the manifest; empty when it is well formed.

    Assets must be named ``<key>.pt``: panel field order is canonicalised
    by path, so this keeps it equal to key order.
    """
    from sheepshead.agent.architectures.registry import ARCHITECTURES

    errors: list[str] = []
    if not str(manifest.get("release_base", "")).startswith("https://"):
        errors.append("release_base must be an https URL")
    artifacts: dict[str, Any] = manifest.get("artifacts", {})
    panels: dict[str, list[str]] = manifest.get("panels", {})
    deploy = [k for k, e in artifacts.items() if "deploy" in e.get("roles", [])]
    if len(deploy) != 1:
        errors.append(f"exactly one artifact must have role deploy, found {deploy}")
    known_roles = {"deploy"} | {name.lower() for name in panels}
    for key, e in artifacts.items():
        where = f"artifacts.{key}"
        if not str(e.get("tag", "")).startswith("artifact/"):
            errors.append(f"{where}.tag must start with artifact/")
        if e.get("asset") != f"{key}.pt":
            errors.append(f"{where}.asset must be {key}.pt")
        if not re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256", ""))):
            errors.append(f"{where}.sha256 must be 64 lowercase hex chars")
        size = e.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            errors.append(f"{where}.bytes must be a positive int")
        roles = e.get("roles")
        if not roles or not set(roles) <= known_roles:
            errors.append(f"{where}.roles must be a non-empty subset of {known_roles}")
        if e.get("arch") not in ARCHITECTURES:
            errors.append(f"{where}.arch {e.get('arch')!r} is not registered")
        if not e.get("provenance"):
            errors.append(f"{where}.provenance is required")
        sampling = e.get("sampling")
        if "deploy" in (roles or []):
            if not isinstance(sampling, dict) or set(sampling) != set(SAMPLING_KEYS):
                errors.append(f"{where}.sampling must have exactly {SAMPLING_KEYS}")
            elif not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
                for v in sampling.values()
            ):
                errors.append(f"{where}.sampling values must be non-negative numbers")
        elif sampling is not None:
            errors.append(f"{where}.sampling belongs on the deploy artifact only")
    for name, members in panels.items():
        for key in members:
            if key not in artifacts:
                errors.append(f"panels.{name} lists unknown key {key}")
            elif name.lower() not in artifacts[key].get("roles", []):
                errors.append(f"artifacts.{key}.roles is missing {name.lower()}")
        if len(set(members)) != len(members):
            errors.append(f"panels.{name} lists a key twice")
    for key, e in artifacts.items():
        for role in e.get("roles", []):
            if role != "deploy" and key not in panels.get(role.upper(), []):
                errors.append(
                    f"artifacts.{key} has role {role} but is not in that panel"
                )
    return errors


def select(
    manifest: dict[str, Any], keys: list[str] | None, roles: list[str] | None
) -> list[str]:
    """Keys named directly or by role; everything when neither is given."""
    if not keys and not roles:
        return list(manifest["artifacts"])
    for key in keys or []:
        _entry(manifest, key)
    chosen = list(keys or [])
    for role in roles or []:
        matched = [k for k, e in manifest["artifacts"].items() if role in e["roles"]]
        if not matched:
            raise ArtifactError(f"no artifact has role {role}")
        chosen += [k for k in matched if k not in chosen]
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key", action="append", help="artifact key (repeatable)")
    ap.add_argument("--role", action="append", help="artifact role (repeatable)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify only, no network")
    mode.add_argument("--lint", action="store_true", help="check the manifest's shape")
    args = ap.parse_args(argv)
    manifest = load_manifest()

    if args.lint:
        errors = lint(manifest)
        for e in errors:
            print(f"manifest: {e}", file=sys.stderr)
        if not errors:
            print(f"manifest OK: {len(manifest['artifacts'])} artifacts")
        return 1 if errors else 0

    try:
        for key in select(manifest, args.key, args.role):
            if args.check:
                verify(key, manifest)
                print(f"ok        {key}")
            else:
                print(f"{'fetched' if fetch(key, manifest) else 'cached':<9} {key}")
    except (ArtifactError, urllib.error.URLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
