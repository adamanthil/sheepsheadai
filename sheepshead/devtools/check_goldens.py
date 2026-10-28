#!/usr/bin/env python3
"""Bit-exact golden check of the working tree against a baseline commit.

The architecture and ISMCTS search fingerprints (capture_arch_goldens,
capture_search_goldens) depend on the BLAS/LAPACK the torch build links -- on
macOS the system Accelerate framework -- so an OS or torch update changes them
with no code change, and they cannot be committed. Instead this driver
captures them twice on the same machine, in the same Python environment: once
at a baseline commit (checked out into a temporary git worktree) and once at
the working tree. The two must be bit-identical; the only thing that differs
between them is the code.

Units and the trailer
---------------------
Each registered architecture is a unit, and so is ``search`` (the ISMCTS
fixture; it runs ``capture_search_goldens.ARCH``, so naming that architecture
covers it too). A commit that changes a unit's outputs on purpose says so in
the trailer block at the end of its message:

    Changes-Network-Outputs: perceiver-recall, full-recall (re-init pointer_V)

Names are comma-separated, parenthesized text is a free-form reason, and
``all`` names every unit. For a named unit the range is split around the
marked commit: baseline -> its parent and the commit -> working tree must each
be bit-identical, and only the step into the marked commit is exempt. When
several commits in the range name the same unit, everything from the first to
the last is exempt.

Environments
------------
Every source is first imported from its own checkout inside the current
environment (PYTHONPATH), so a dependency bump alone never fails the check.
A baseline that cannot run there (a breaking upgrade) is re-run in its own
locked environment (``uv run --frozen`` inside its worktree); the report says
which environment each capture used.

A source that has no capture script at the paths below -- it predates the
scripts, or their move into ``sheepshead.devtools`` -- contributes nothing:
its units are reported as absent, like an architecture registered later.

Baseline
--------
``--baseline REF`` (default: the upstream branch, else origin/main). A ref
that does not resolve -- e.g. the all-zero ``before`` of a branch's first
push -- falls back to the merge-base with origin/main; one that is not an
ancestor of HEAD (a force-push) is replaced by its merge-base with HEAD. The
check is skipped when no watched path changed since the baseline.

Usage:
    uv run python -m sheepshead.devtools.check_goldens
    uv run python -m sheepshead.devtools.check_goldens --baseline main
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import torch

from sheepshead.agent.architectures.registry import available_architectures
from sheepshead.devtools import capture_arch_goldens, capture_search_goldens

TRAILER = "Changes-Network-Outputs"
SEARCH_UNIT = "search"
ALL_UNITS = "all"
# The uncommitted side of the comparison when the working tree is dirty.
WORKING_TREE = "working tree"

ARCH_SCRIPT = "sheepshead/devtools/capture_arch_goldens.py"
SEARCH_SCRIPT = "sheepshead/devtools/capture_search_goldens.py"

# Run inside the source's interpreter: proves it imports, that the package
# really comes from the source checkout, and lists what is registered there.
_PROBE = (
    "import json, sheepshead\n"
    "from sheepshead.agent.architectures.registry import available_architectures\n"
    "print(json.dumps({'package': sheepshead.__file__,"
    " 'archs': available_architectures()}))\n"
)


class CaptureFailed(Exception):
    pass


@dataclass(frozen=True)
class MarkedCommit:
    sha: str
    parent: str  # first parent
    names: frozenset[str]


@dataclass
class Capture:
    label: str
    out_dir: str
    env: str
    units: set[str]  # the requested units that exist at this source


def is_watched(path: str) -> bool:
    """Whether a change to ``path`` can move a fingerprint: package code
    other than tests, instruments, research and dev tooling (no capture
    imports them), plus the capture scripts themselves."""
    if path in (ARCH_SCRIPT, SEARCH_SCRIPT):
        return True
    return path.startswith("sheepshead/") and not path.startswith(
        (
            "sheepshead/tests/",
            "sheepshead/eval/",
            "sheepshead/research/",
            "sheepshead/devtools/",
        )
    )


def parse_trailer(value: str) -> frozenset[str]:
    """``"a, b (why, briefly)"`` -> ``{"a", "b"}``. The search fixture rides
    along with the architecture it runs."""
    without_reasons = re.sub(r"\([^)]*\)", "", value)
    names = {name.strip() for name in without_reasons.split(",") if name.strip()}
    if capture_search_goldens.ARCH in names:
        names.add(SEARCH_UNIT)
    return frozenset(names)


def plan_segments(
    units: list[str], base: str, head: str, marked: list[MarkedCommit]
) -> dict[str, list[tuple[str, str]]]:
    """For each unit, the (before, after) source pairs that must be
    bit-identical. ``marked`` is oldest first; pairs whose ends are the same
    commit are dropped (trivially identical)."""
    plan = {}
    for unit in units:
        hits = [m for m in marked if unit in m.names or ALL_UNITS in m.names]
        if hits:
            segments = [(base, hits[0].parent), (hits[-1].sha, head)]
        else:
            segments = [(base, head)]
        plan[unit] = [(a, b) for a, b in segments if a != b]
    return plan


def exempt_span(unit: str, marked: list[MarkedCommit]) -> tuple[str, str] | None:
    hits = [m for m in marked if unit in m.names or ALL_UNITS in m.names]
    return (hits[0].parent, hits[-1].sha) if hits else None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_try(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_baseline(ref: str | None) -> tuple[str, list[str]]:
    notes = []
    if ref is None:
        ref = _git_try("rev-parse", "--abbrev-ref", "@{upstream}") or "origin/main"
    sha = _git_try("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if sha is None:
        sha = _git("merge-base", "HEAD", "origin/main")
        notes.append(f"{ref!r} does not resolve; using the merge-base with origin/main")
    elif _git_try("merge-base", "--is-ancestor", sha, "HEAD") is None:
        sha = _git("merge-base", sha, "HEAD")
        notes.append(f"{ref!r} is not an ancestor of HEAD; using the merge-base")
    return sha, notes


def changed_paths(base: str) -> list[str]:
    """Tracked changes from ``base`` to the working tree, plus untracked files."""
    tracked = _git("diff", "--name-only", base).splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    return [p for p in tracked + untracked if p]


def marked_commits(base: str) -> list[MarkedCommit]:
    log = _git(
        "log",
        "-z",
        "--reverse",
        "--topo-order",
        f"--format=%H%x1f%P%x1f%(trailers:key={TRAILER},valueonly,separator=%x2C)",
        f"{base}..HEAD",
    )
    marked = []
    for record in filter(None, log.split("\0")):
        sha, parents, value = record.strip("\n").split("\x1f")
        if value.strip():
            marked.append(MarkedCommit(sha, parents.split()[0], parse_trailer(value)))
    return marked


def _short(source: str) -> str:
    return source if source == WORKING_TREE else source[:9]


def _run(what: str, command: list[str], cwd: str, env: dict[str, str]) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        raise CaptureFailed(f"{what} exited {result.returncode}:\n{tail}")
    return result.stdout


def _probe(python: list[str], src: str, env: dict[str, str]) -> list[str]:
    """The architectures registered at ``src``, imported from ``src``."""
    lines = _run("import probe", [*python, "-c", _PROBE], src, env).splitlines()
    try:
        probe = json.loads(lines[-1])
    except (IndexError, ValueError) as e:
        raise CaptureFailed(f"import probe printed no JSON ({e})") from e
    package = os.path.realpath(probe["package"])
    if not package.startswith(os.path.realpath(src) + os.sep):
        raise CaptureFailed(f"imported sheepshead from {package}, not from {src}")
    return probe["archs"]


def _environments(src: str, allow_own: bool):
    """(label, python command, env) to try in order for one source."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    yield "current environment", [sys.executable], {**env, "PYTHONPATH": src}
    if allow_own:
        own = {k: v for k, v in env.items() if k != "VIRTUAL_ENV"}
        yield "own locked environment", ["uv", "run", "--frozen", "python"], own


def capture_source(
    src: str, label: str, units: set[str], out_dir: str, allow_own: bool
) -> Capture:
    # A source without a capture script (it predates the script, or the
    # script's move into devtools) has nothing to compare: its units count
    # as absent, exactly like an architecture registered after it.
    has_arch = os.path.exists(os.path.join(src, ARCH_SCRIPT))
    search = SEARCH_UNIT in units and os.path.exists(os.path.join(src, SEARCH_SCRIPT))
    missing = [
        p for p, ok in ((ARCH_SCRIPT, has_arch), (SEARCH_SCRIPT, search)) if not ok
    ]
    if missing:
        print(f"  {label} has no {', '.join(missing)}; those units count as absent")
    if not (has_arch or search):
        return Capture(label, out_dir, "nothing to capture", set())
    failures = []
    for env_label, python, env in _environments(src, allow_own):
        try:
            archs = sorted(units & set(_probe(python, src, env))) if has_arch else []
            what = f"{len(archs)} architectures" + (" + search" if search else "")
            print(f"  capturing {label}: {what} ({env_label})", flush=True)
            if archs:
                arch_args = [arg for arch in archs for arg in ("--arch", arch)]
                _run(
                    "capture_arch_goldens",
                    [
                        *python,
                        "-m",
                        "sheepshead.devtools.capture_arch_goldens",
                        "--fixture-dir",
                        os.path.join(out_dir, "arch"),
                        *arch_args,
                    ],
                    src,
                    env,
                )
            if search:
                _run(
                    "capture_search_goldens",
                    [
                        *python,
                        "-m",
                        "sheepshead.devtools.capture_search_goldens",
                        "--fixture-dir",
                        os.path.join(out_dir, "search"),
                    ],
                    src,
                    env,
                )
            present = set(archs) | ({SEARCH_UNIT} if search else set())
            return Capture(label, out_dir, env_label, present)
        except CaptureFailed as e:
            failures.append(f"[{env_label}] {e}")
            print(f"  {label} failed in the {env_label}", flush=True)
    raise CaptureFailed(f"{label} could not be captured:\n" + "\n".join(failures))


def compare_unit(unit: str, before: Capture, after: Capture) -> list[str]:
    if unit == SEARCH_UNIT:
        golden, current = (
            torch.load(
                capture_search_goldens.fixture_path(os.path.join(c.out_dir, "search")),
                weights_only=True,
            )
            for c in (before, after)
        )
        return capture_search_goldens.compare(golden, current)
    golden, current = (
        torch.load(
            capture_arch_goldens.fixture_path(unit, os.path.join(c.out_dir, "arch")),
            weights_only=True,
        )
        for c in (before, after)
    )
    return capture_arch_goldens.compare_arch(golden, current)


def _torch_version(capture: Capture) -> str:
    for sub in ("arch", "search"):
        path = os.path.join(capture.out_dir, sub, "manifest.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["torch"]
    return "?"


FAILURE_HINT = f"""
If these output changes are intended, say so in the message of the commit
that makes them, in the trailer block at the end, naming each affected
architecture ("search" for the ISMCTS fixture, "all" for everything):

    {TRAILER}: perceiver-recall, full-recall (why, briefly)

The check then exempts only that commit for those names and still compares
everything before and after it."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        help="commit to compare against (default: upstream branch, else origin/main)",
    )
    args = parser.parse_args()

    toplevel = _git("rev-parse", "--show-toplevel")
    os.chdir(toplevel)
    base, notes = resolve_baseline(args.baseline)
    head_sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    head = WORKING_TREE if dirty else head_sha
    for note in notes:
        print(f"note: {note}")
    print(f"baseline  {_git('log', '-1', '--format=%h %s', base)}")
    print(
        "current   "
        + ("working tree (uncommitted changes on " if dirty else "")
        + _git("log", "-1", "--format=%h %s", "HEAD")
        + (")" if dirty else "")
    )

    watched = [p for p in changed_paths(base) if is_watched(p)]
    if not watched:
        print("no watched paths changed since the baseline; nothing to check")
        return 0

    units = sorted(available_architectures()) + [SEARCH_UNIT]
    marked = marked_commits(base)
    for m in marked:
        unknown = sorted(m.names - set(units) - {ALL_UNITS})
        if unknown:
            print(f"warning: {TRAILER} in {m.sha[:9]} names unknown {unknown}")
    plan = plan_segments(units, base, head, marked)

    needed: dict[str, set[str]] = {}
    for unit, segments in plan.items():
        for pair in segments:
            for source in pair:
                needed.setdefault(source, set()).add(unit)

    tmp = tempfile.mkdtemp(prefix="check-goldens-")
    worktrees = []
    captures: dict[str, Capture] = {}
    try:
        for source in sorted(needed, key=lambda s: (s != base, s == head, s)):
            label = _short(source)
            out_dir = os.path.join(tmp, "captures", label.replace(" ", "-"))
            if source == head:
                src, allow_own = toplevel, False
            else:
                src = os.path.join(tmp, "worktrees", label)
                _git("worktree", "add", "--detach", src, source)
                worktrees.append(src)
                allow_own = True
            captures[source] = capture_source(
                src, label, needed[source], out_dir, allow_own
            )

        print()
        for capture in captures.values():
            torch_note = f", torch {_torch_version(capture)}" if capture.units else ""
            print(f"  {capture.label:<12} {capture.env}{torch_note}")
        print()

        compared = failed = 0
        for unit in units:
            span = exempt_span(unit, marked)
            if span is not None:
                print(
                    f"exempt  {unit:<32} {_short(span[0])} -> {_short(span[1])}"
                    f"  ({TRAILER})"
                )
            for before, after in plan[unit]:
                a, b = captures[before], captures[after]
                pair = f"{a.label} -> {b.label}"
                missing = [c.label for c in (a, b) if unit not in c.units]
                if missing:
                    print(f"absent  {unit:<32} {pair}  (absent at {missing[0]})")
                    continue
                compared += 1
                problems = compare_unit(unit, a, b)
                if problems:
                    failed += 1
                    print(f"FAIL    {unit:<32} {pair}")
                    for problem in problems:
                        print(f"          {problem}")
                else:
                    print(f"ok      {unit:<32} {pair}")
        if failed:
            print(f"\n{failed} comparison(s) not bit-identical.{FAILURE_HINT}")
            return 1
        if compared == 0:
            print("\nnothing compared: no unit is present on both sides")
        else:
            print(f"\nall {compared} comparisons bit-identical")
        return 0
    except CaptureFailed as e:
        print(f"\nerror: {e}")
        return 2
    finally:
        for worktree in worktrees:
            _git_try("worktree", "remove", "--force", worktree)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
