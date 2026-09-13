"""h2h_duplicate's sharded evaluation must be bit-identical to the serial
loop (CE_Teacher_Design §20.13: 8000-deal certs are the standard read).
Needs two real checkpoints; skipped when they are absent (CI)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from sheepshead.analysis.league_progress_eval import h2h_duplicate

CAND = "runs/policy_iteration_202609/iter11/distill_epoch7.pt"
ANCHOR = "runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt"


@pytest.mark.slow
def test_parallel_h2h_matches_serial_bitwise():
    if not (os.path.exists(CAND) and os.path.exists(ANCHOR)):
        pytest.skip("real checkpoints not available")
    serial = h2h_duplicate(CAND, ANCHOR, n_deals_per_mode=12, n_boot=50, workers=1)
    par = h2h_duplicate(CAND, ANCHOR, n_deals_per_mode=12, n_boot=50, workers=4)
    for m in range(2):
        assert np.array_equal(
            np.array(serial["per_deal"][m]), np.array(par["per_deal"][m])
        )
        assert serial["per_deal_leaster_hands"][m] == par["per_deal_leaster_hands"][m]
    assert serial["edge"] == par["edge"] and serial["se"] == par["se"]
    for key in ("leaster", "non_leaster"):
        a, b = serial[key], par[key]
        assert a["hands"] == b["hands"]
        if a["hands"]:
            assert a["edge"] == b["edge"] and a["se"] == b["se"]
