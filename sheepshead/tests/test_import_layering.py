#!/usr/bin/env python3
"""The sheepshead package's import layering, checked on the source.

Layers, bottom to top:

  core       game, rewards, scripted_agent, agent.*
  search     ismcts (imports only core)
  eval       maintained measurement instruments and model artifacts
  training   library modules (configs, corpus formats, the episode
             runtime, ...) usable by eval, and the part ON TOP of the
             instruments: the PPO trainer and league roster
             (``TRAINING_STACK``) plus every training module that imports
             them or eval, directly or transitively (the orchestrators).
  research, devtools
             leaves: they may import any layer; nothing imports them.

Every import counts, including function-local ones: a lazy import hides a
dependency from a casual reader, not from this test.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
TRAINING_STACK = ("train_ppo", "league")


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _layer(module: str) -> str:
    parts = module.split(".")
    if len(parts) > 2 or (len(parts) == 2 and (PACKAGE / parts[1]).is_dir()):
        return "core" if parts[1] == "agent" else parts[1]
    return "core"


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


MODULES = {
    _module_name(p): p
    for p in PACKAGE.rglob("*.py")
    if "tests" not in p.relative_to(PACKAGE).parts
}
GRAPH = {name: _imports(path) for name, path in MODULES.items()}


def _training_on_top() -> set[str]:
    on_top = {f"sheepshead.training.{name}" for name in TRAINING_STACK}
    grew = True
    while grew:
        grew = False
        for module, deps in GRAPH.items():
            if _layer(module) != "training" or module in on_top:
                continue
            if any(d in on_top or _layer(d) == "eval" for d in deps):
                on_top.add(module)
                grew = True
    return on_top


def _violations(importers: tuple[str, ...], forbidden) -> list[str]:
    return sorted(
        f"{module} imports {dep}"
        for module, deps in GRAPH.items()
        if _layer(module) in importers
        for dep in deps
        if forbidden(dep)
    )


def test_core_imports_only_core():
    assert _violations(("core",), lambda d: _layer(d) != "core") == []


def test_instruments_stay_below_the_training_stack():
    on_top = _training_on_top()
    assert _violations(("eval",), lambda d: d in on_top) == []


def test_nothing_imports_the_leaves():
    importers = ("core", "search", "eval", "training")
    leaves = ("research", "devtools")
    assert _violations(importers, lambda d: _layer(d) in leaves) == []


def test_stack_closure_reaches_the_orchestrators():
    # Guards the closure itself: if it stopped propagating, the rule above
    # would pass vacuously.
    on_top = _training_on_top()
    for name in ("run_training_program", "policy_iteration", "exploitability_audit"):
        assert f"sheepshead.training.{name}" in on_top
    assert "sheepshead.training.pfsp_runtime" not in on_top
