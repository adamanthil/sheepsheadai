"""Trick-indexed committee budget schedule for the distillation corpus
(CE_Teacher_Design §20.13 addendum 29 + amendment)."""

from __future__ import annotations

import pytest

from sheepshead.training.distill_corpus import parse_iters_schedule


def test_parse_schedule_keys_by_trick_and_lead():
    assert parse_iters_schedule("t0-lead:1024,t1-lead:512") == {
        (0, True): 1024,
        (1, True): 512,
    }
    assert parse_iters_schedule(" t2-follow:384 ") == {(2, False): 384}


def test_parse_schedule_empty_means_default_everywhere():
    assert parse_iters_schedule(None) == {}
    assert parse_iters_schedule("") == {}


def test_parse_schedule_rejects_malformed_entries():
    with pytest.raises(SystemExit):
        parse_iters_schedule("lead0:1024")
    with pytest.raises(SystemExit):
        parse_iters_schedule("t0-bid:1024")
