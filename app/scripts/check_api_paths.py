#!/usr/bin/env python
"""Fail if app/deploy/api-paths.txt does not cover the api image's code.

The api image tag is a hash of the paths in api-paths.txt
(app/scripts/deploy_tags.sh), so a commit outside them does not restart
the api -- and with it every live game. That narrowing is only safe if the
list covers every sheepshead module the server can import. This script
computes that import closure from the source and checks it:

1. every ``sheepshead`` import in app/server, including function-local
   ones (an AST walk, not ``sys.modules``, so lazy imports count);
2. transitively, every ``sheepshead`` import of each module reached;
3. each file in the closure must equal, or sit under, a listed path
   (a conservative extra entry is fine: over-restart is a nuisance,
   under-restart is a bug);
4. each listed path must exist (a missing one would hash as empty).

Run from anywhere:  uv run python app/scripts/check_api_paths.py
See notebooks/CD_Deployment_Plan_202608.md section 4.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "app" / "server"
PATHS_FILE = ROOT / "app" / "deploy" / "api-paths.txt"
PACKAGE = "sheepshead"


def module_file(module: str) -> Path | None:
    """The repo file defining ``module``, or None if there is none."""
    base = ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def package_name(path: Path) -> str:
    """The dotted package a relative import in ``path`` is relative to."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1])


def imported_files(path: Path) -> set[Path]:
    """Files of the sheepshead modules ``path`` imports, plus the
    ``__init__.py`` of every enclosing package (importing a submodule
    runs its parents' initializers)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package_name(path).split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            modules.add(base)
            # `from pkg import name` imports pkg.name when it is a submodule.
            modules.update(f"{base}.{alias.name}" for alias in node.names)

    files: set[Path] = set()
    for module in modules:
        if module != PACKAGE and not module.startswith(PACKAGE + "."):
            continue
        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            found = module_file(".".join(parts[:i]))
            if found is not None:
                files.add(found)
    return files


def closure() -> set[Path]:
    """Every sheepshead file the server can import, transitively."""
    pending: set[Path] = set()
    for path in SERVER.rglob("*.py"):
        pending |= imported_files(path)
    reached: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        reached.add(path)
        pending |= imported_files(path) - reached
    return reached


def main() -> int:
    listed = [
        line.strip()
        for line in PATHS_FILE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    errors: list[str] = []

    missing = [p for p in listed if not (ROOT / p).exists()]
    for p in missing:
        errors.append(f"listed path does not exist: {p}")

    def covered(rel: Path) -> bool:
        return any(rel == Path(p) or Path(p) in rel.parents for p in listed)

    uncovered = sorted(
        rel
        for rel in (path.relative_to(ROOT) for path in closure())
        if not covered(rel)
    )
    if uncovered:
        errors.append("the api imports files outside api-paths.txt; add these lines:")
        errors.extend(f"  {rel.as_posix()}" for rel in uncovered)

    if errors:
        print(f"{PATHS_FILE.relative_to(ROOT)} is stale:", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print(f"api-paths.txt covers the api import closure ({len(listed)} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
