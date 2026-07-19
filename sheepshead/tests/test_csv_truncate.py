#!/usr/bin/env python3
"""Crash-resume telemetry dedupe: truncate_csv_rows_past_episode."""

import csv

from sheepshead.training.training_utils import truncate_csv_rows_past_episode


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


class TestTruncate:
    HEADER = ["episode", "pick_rate"]

    def test_drops_only_rows_past_episode(self, tmp_path):
        p = str(tmp_path / "t.csv")
        write_csv(
            p,
            self.HEADER,
            [["2000000", "0.2"], ["2050000", "0.21"], ["2050184", "0.22"], ["2083314", "0.19"]],
        )
        dropped = truncate_csv_rows_past_episode(p, 2_050_000)
        assert dropped == 2
        rows = read_csv(p)
        assert rows == [self.HEADER, ["2000000", "0.2"], ["2050000", "0.21"]]

    def test_clean_start_is_noop(self, tmp_path):
        p = str(tmp_path / "t.csv")
        body = [["1000000", "0.3"], ["2000000", "0.25"]]
        write_csv(p, self.HEADER, body)
        assert truncate_csv_rows_past_episode(p, 2_000_000) == 0
        assert read_csv(p) == [self.HEADER] + body

    def test_missing_file_and_zero_episode(self, tmp_path):
        assert truncate_csv_rows_past_episode(str(tmp_path / "absent.csv"), 100) == 0
        p = str(tmp_path / "t.csv")
        write_csv(p, self.HEADER, [["50", "0.1"]])
        # episode 0 = fresh run, never truncate
        assert truncate_csv_rows_past_episode(p, 0) == 0

    def test_malformed_rows_kept(self, tmp_path):
        p = str(tmp_path / "t.csv")
        write_csv(p, self.HEADER, [["not_a_number", "x"], ["150", "0.2"]])
        assert truncate_csv_rows_past_episode(p, 100) == 1
        assert read_csv(p) == [self.HEADER, ["not_a_number", "x"]]

    def test_missing_episode_column_is_noop(self, tmp_path):
        p = str(tmp_path / "t.csv")
        write_csv(p, ["generation", "edge"], [["1", "0.1"], ["2", "0.2"]])
        assert truncate_csv_rows_past_episode(p, 1) == 0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
