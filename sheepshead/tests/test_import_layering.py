#!/usr/bin/env python3
"""The import layering of the sheepshead package and the app, checked on
the source.

Each package is one layer and imports only from its own layer or below:

  0 core       game, rewards, scripted_agent, views
  1 agent      networks, PPO, observation, deploy sampling, traced games
  2 search     the ISMCTS teacher
  3 rl         RL building blocks: episode runtime, corpus plumbing, the
               search-advantage model, entropy controller, watchdog
  4 eval       maintained instruments and model artifacts
  5 training   trainers, league, drivers, the program orchestrator

``research`` and ``devtools`` are leaves: they may import any ranked layer
and nothing imports them. The app (app/) imports only core and agent, and
nothing in the package imports the app.

Every import counts, including function-local ones: a lazy import hides a
dependency from a casual reader, not from this test.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
APP = PACKAGE.parent / "app"
RANKS = {"core": 0, "agent": 1, "search": 2, "rl": 3, "eval": 4, "training": 5}
LEAVES = ("research", "devtools")
APP_MAX_RANK = RANKS["agent"]


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _layer(module: str) -> str:
    parts = module.split(".")
    if len(parts) > 2 or (len(parts) == 2 and (PACKAGE / parts[1]).is_dir()):
        return parts[1]
    return "core"


MODULES = {
    _module_name(p): p
    for p in PACKAGE.rglob("*.py")
    if "tests" not in p.relative_to(PACKAGE).parts
}


def _imports(path: Path) -> set[str]:
    """Every sheepshead module ``path`` imports, at any nesting depth.
    ``from pkg import name`` also counts ``pkg.name`` when that is a module."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.startswith("sheepshead")}
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("sheepshead")
        ):
            found.add(node.module)
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                if candidate in MODULES:
                    found.add(candidate)
    return found


GRAPH = {name: _imports(path) for name, path in MODULES.items()}


def test_every_package_has_a_layer():
    unranked = sorted({_layer(m) for m in MODULES} - set(RANKS) - set(LEAVES))
    assert unranked == [], f"rank these packages in RANKS or LEAVES: {unranked}"


def test_ranked_layers_import_only_downward():
    upward = sorted(
        f"{module} imports {dep}"
        for module, deps in GRAPH.items()
        if _layer(module) in RANKS
        for dep in deps
        if _layer(dep) not in RANKS or RANKS[_layer(dep)] > RANKS[_layer(module)]
    )
    assert upward == []


def test_leaves_import_no_other_leaf():
    crossing = sorted(
        f"{module} imports {dep}"
        for module, deps in GRAPH.items()
        if _layer(module) in LEAVES
        for dep in deps
        if _layer(dep) in LEAVES and _layer(dep) != _layer(module)
    )
    assert crossing == []


def test_app_imports_only_core_and_agent():
    too_high = sorted(
        f"{path.relative_to(APP)} imports {dep}"
        for path in APP.rglob("*.py")
        if "node_modules" not in path.parts
        for dep in _imports(path)
        if RANKS.get(_layer(dep), len(RANKS)) > APP_MAX_RANK
    )
    assert too_high == []


def test_package_never_imports_the_app():
    app_packages = {p.name for p in APP.iterdir() if (p / "__init__.py").exists()}
    found = sorted(
        f"{module} imports {name}"
        for module, path in MODULES.items()
        for node in ast.walk(ast.parse(path.read_text()))
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if name.split(".")[0] in app_packages
    )
    assert found == []
