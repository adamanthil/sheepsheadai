#!/usr/bin/env python3
"""Planning logic of the commit-against-commit golden check: which paths
trigger it, how the Changes-Network-Outputs trailer is read, and which
(before, after) pairs each unit is compared over. The capture/compare half
needs git worktrees and full captures and is exercised by CI's goldens job."""

from sheepshead.devtools.capture_search_goldens import ARCH as SEARCH_ARCH
from sheepshead.devtools.check_goldens import (
    SEARCH_UNIT,
    MarkedCommit,
    exempt_span,
    is_watched,
    parse_trailer,
    plan_segments,
)

BASE, HEAD = "base", "head"


def _marked(sha, parent, *names):
    return MarkedCommit(sha, parent, frozenset(names))


class TestIsWatched:
    def test_package_code_is_watched(self):
        assert is_watched("sheepshead/agent/architectures/actors.py")
        assert is_watched("sheepshead/search/ismcts.py")

    def test_tests_and_analysis_are_not(self):
        assert not is_watched("sheepshead/tests/test_arch_golden.py")
        assert not is_watched("sheepshead/eval/rigorous_eval.py")
        assert not is_watched("sheepshead/research/leaster_scan.py")
        assert not is_watched("sheepshead/devtools/check_goldens.py")
        assert not is_watched("app/server/main.py")
        assert not is_watched("notebooks/notes.md")

    def test_capture_scripts_are(self):
        assert is_watched("sheepshead/devtools/capture_arch_goldens.py")
        assert is_watched("sheepshead/devtools/capture_search_goldens.py")


class TestParseTrailer:
    def test_names_and_reason(self):
        assert parse_trailer("perceiver-recall, full-recall (re-init, again)") == {
            "perceiver-recall",
            "full-recall",
        }

    def test_repeated_trailers_joined_by_git(self):
        assert parse_trailer("a (why),b") == {"a", "b"}

    def test_search_rides_with_its_architecture(self):
        assert parse_trailer(SEARCH_ARCH) == {SEARCH_ARCH, SEARCH_UNIT}
        assert SEARCH_UNIT not in parse_trailer("full")


class TestPlanSegments:
    def test_unmarked_unit_compares_base_to_head(self):
        plan = plan_segments(["x"], BASE, HEAD, [_marked("c2", "c1", "y")])
        assert plan["x"] == [(BASE, HEAD)]

    def test_marked_unit_is_split_around_the_commit(self):
        plan = plan_segments(["x"], BASE, HEAD, [_marked("c2", "c1", "x")])
        assert plan["x"] == [(BASE, "c1"), ("c2", HEAD)]
        assert exempt_span("x", [_marked("c2", "c1", "x")]) == ("c1", "c2")

    def test_first_and_last_marks_bound_the_exemption(self):
        marked = [_marked("c2", "c1", "x"), _marked("c4", "c3", "x")]
        assert plan_segments(["x"], BASE, HEAD, marked)["x"] == [
            (BASE, "c1"),
            ("c4", HEAD),
        ]
        assert exempt_span("x", marked) == ("c1", "c4")

    def test_all_marks_every_unit(self):
        plan = plan_segments(["x", "y"], BASE, HEAD, [_marked("c2", "c1", "all")])
        split = [(BASE, "c1"), ("c2", HEAD)]
        assert plan == {"x": split, "y": split}

    def test_degenerate_pairs_are_dropped(self):
        # Marked commit directly on the baseline, and it is also HEAD.
        plan = plan_segments(["x"], BASE, "c1", [_marked("c1", BASE, "x")])
        assert plan["x"] == []
