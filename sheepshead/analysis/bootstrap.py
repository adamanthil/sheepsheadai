#!/usr/bin/env python3
"""Bootstrap-over-deals primitives (deal = the independent unit).

Shared statistical machinery for the paired-deal instruments
(league_progress_eval) and the extended-league stop policy
(sheepshead/training/league_stopping.py). Mirrors the bootstrap
conventions of rigorous_eval.py — boot_idx arrays are interchangeable
between these modules — but stays import-light (pure numpy) on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def bootstrap_deal_indices(
    n_deals: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n_deals) array of resampled deal indices (with replacement)."""
    return rng.integers(0, n_deals, size=(n_boot, n_deals))


@dataclass
class IntervalStat:
    """A per-deal mean with its bootstrap interval."""

    mean: float
    lo: float
    hi: float
    se: float
    p_value: float  # two-sided bootstrap p that the mean == 0


def bootstrap_interval(
    per_deal: np.ndarray, boot_idx: np.ndarray, ci: float = 0.95
) -> IntervalStat:
    boots = per_deal[boot_idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    frac_le = float(np.mean(boots <= 0.0))
    frac_ge = float(np.mean(boots >= 0.0))
    return IntervalStat(
        mean=float(per_deal.mean()),
        lo=float(lo),
        hi=float(hi),
        se=float(boots.std(ddof=1)),
        p_value=min(1.0, 2.0 * min(frac_le, frac_ge)),
    )


def interval_to_dict(s: IntervalStat) -> Dict[str, float]:
    return {"mean": s.mean, "lo": s.lo, "hi": s.hi, "se": s.se, "p_value": s.p_value}
