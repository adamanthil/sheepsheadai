"""DeploySampling: argmax at T = 0, tie-preserving at any T, band exclusion,
monotone coupling across temperatures, reseed reproducibility, picklable."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from sheepshead.agent.sampling import HEAD_GROUPS, DeploySampling, head_of
from sheepshead.game import ACTIONS

PLAY_IDS = [i + 1 for i, a in enumerate(ACTIONS) if a.startswith("PLAY ")]
PICK_IDS = [ACTIONS.index("PICK") + 1, ACTIONS.index("PASS") + 1]


def _log_probs(valid: list[int], logits: dict[int, float]) -> np.ndarray:
    """Masked log-probabilities over the full action vector from per-action logits."""
    z = np.full(len(ACTIONS), -np.inf)
    for a in valid:
        z[a - 1] = logits.get(a, 0.0)
    w = np.exp(z - z[np.isfinite(z)].max(), where=np.isfinite(z), out=np.zeros_like(z))
    return np.log(np.clip(w / w.sum(), 1e-30, None))


def test_head_of_groups():
    assert head_of(PICK_IDS) == "pick"
    assert head_of(PLAY_IDS[:3]) == "play"
    assert head_of([ACTIONS.index("CALL AC") + 1]) == "partner"
    assert head_of([ACTIONS.index("BURY 7C") + 1]) == "bury"


def test_zero_temperature_is_argmax():
    valid = PLAY_IDS[:4]
    lp = _log_probs(valid, {valid[2]: 2.0})
    s = DeploySampling()
    assert s.sample(lp, valid) == valid[2] - 1
    assert s.distribution(lp, valid)[valid[2] - 1] == 1.0


def test_ties_stay_mixed_and_gaps_collapse():
    valid = PLAY_IDS[:3]
    lp = _log_probs(valid, {valid[0]: 1.0, valid[1]: 1.0, valid[2]: -4.0})
    for t in (0.25, 0.5, 1.0):
        p = DeploySampling(temperature={"play": t}).distribution(lp, valid)
        assert p[valid[0] - 1] == pytest.approx(p[valid[1] - 1])
    p_cold = DeploySampling(temperature={"play": 0.25}).distribution(lp, valid)
    p_warm = DeploySampling(temperature={"play": 1.0}).distribution(lp, valid)
    assert p_cold[valid[2] - 1] < p_warm[valid[2] - 1] < 0.01


def test_band_excludes_far_actions_only():
    valid = PLAY_IDS[:3]
    lp = _log_probs(valid, {valid[0]: 0.0, valid[1]: -0.5, valid[2]: -3.0})
    p = DeploySampling(temperature={"play": 1.0}, band=1.0).distribution(lp, valid)
    assert p[valid[2] - 1] == 0.0
    assert p[valid[1] - 1] > 0.0
    assert p.sum() == pytest.approx(1.0)


def test_other_heads_stay_argmax():
    lp = _log_probs(PICK_IDS, {PICK_IDS[0]: 0.1})
    p = DeploySampling(temperature={"play": 1.0}).distribution(lp, PICK_IDS)
    assert p[PICK_IDS[0] - 1] == 1.0


def test_monotone_coupling_across_temperatures():
    valid = PLAY_IDS[:5]
    rng = np.random.default_rng(0)
    for _ in range(200):
        lp = _log_probs(valid, {a: float(rng.normal()) for a in valid})
        seed = int(rng.integers(1 << 30))
        best = int(np.argmax(lp))
        draws = {}
        for t in (0.25, 0.5, 1.0):
            s = DeploySampling(temperature={"play": t})
            s.reseed(seed)
            draws[t] = s.sample(lp, valid)
        # A colder arm that leaves the argmax implies the warmer arms did too
        # (same uniform, nested tails).
        if draws[0.25] != best:
            assert draws[0.5] != best and draws[1.0] != best
        if draws[0.5] != best:
            assert draws[1.0] != best


def test_reseed_reproduces_and_pickles():
    valid = PLAY_IDS[:6]
    lp = _log_probs(valid, {a: 0.0 for a in valid})
    s = DeploySampling(temperature={"play": 1.0}, seed=7)
    s.reseed(123)
    a = [s.sample(lp, valid) for _ in range(20)]
    s.reseed(123)
    assert [s.sample(lp, valid) for _ in range(20)] == a
    s2 = pickle.loads(pickle.dumps(s))
    s2.reseed(123)
    assert [s2.sample(lp, valid) for _ in range(20)] == a
    assert len(set(a)) > 1
    assert s.label() == "play=1"
    assert DeploySampling().label() == "argmax"
    assert set(HEAD_GROUPS) == {"pick", "partner", "bury", "play"}
