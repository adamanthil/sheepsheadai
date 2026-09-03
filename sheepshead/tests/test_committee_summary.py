"""``pfsp_runtime.summarize_committee`` / ``CommitteeSummary`` (CE_Teacher_Design
§20.3): the pooled committee evidence every target builder reads.

The load-bearing property is numerical identity with the pre-§20 target
builder — the tilt semantics of ``search_target`` must not move under the
refactor. ``_legacy_build_ce_search_target`` is that builder, copied
verbatim from the pre-refactor module as the reference.
"""

import random

import numpy as np
import pytest

from sheepshead.ismcts import minmax_unit
from sheepshead.training.pfsp_runtime import (
    CommitteeSummary,
    build_ce_search_target,
    summarize_committee,
    tilt_summary_to_target,
)

SHRINK: dict = dict(shrink_nu=4.0, shrink_s2_global=6.95e-4)
GUMBEL: dict = dict(gumbel_c_visit=50.0, gumbel_c_scale=0.1)


def _legacy_build_ce_search_target(
    replicates,
    valid_actions,
    *,
    shrink_nu,
    shrink_s2_global,
    gumbel_c_visit,
    gumbel_c_scale,
    base_prior=None,
):
    acts = sorted(valid_actions)
    usable = [
        r
        for r in replicates
        if r["ok"] and r.get("root_q") is not None and r.get("root_prior") is not None
    ]
    if len(usable) < 2:
        return None
    q_obs = {
        a: [float(r["root_q"][a]) for r in usable if r["root_n"].get(a, 0.0) > 0.0]
        for a in acts
    }
    n_pool = {
        a: float(np.mean([r["root_n"].get(a, 0.0) for r in usable])) for a in acts
    }
    visited = [a for a in acts if q_obs[a]]
    if not visited:
        return None
    q_mean = {a: float(np.mean(q_obs[a])) for a in visited}
    v_mix = float(
        sum(n_pool[a] * q_mean[a] for a in visited)
        / max(sum(n_pool[a] for a in visited), 1e-12)
    )
    q_bar = np.array([q_mean.get(a, v_mix) for a in acts], dtype=np.float64)

    def pooled_mean_variance(obs):
        n_obs = len(obs)
        s2_node = float(np.var(obs, ddof=1)) if n_obs >= 2 else 0.0
        s2_blend = (shrink_nu * shrink_s2_global + (n_obs - 1) * s2_node) / (
            shrink_nu + n_obs - 1
        )
        return s2_blend / n_obs

    noise_var = float(np.mean([pooled_mean_variance(q_obs[a]) for a in visited]))
    signal_var = float(np.var(q_bar))
    shrink_w = max(0.0, 1.0 - noise_var / signal_var) if signal_var > 0.0 else 0.0
    if base_prior is not None:
        prior = np.asarray(base_prior, dtype=np.float64)
        prior = prior / max(prior.sum(), 1e-12)
    else:
        prior = np.array(
            [np.mean([r["root_prior"][a] for r in usable]) for a in acts],
            dtype=np.float64,
        )
    scale = (
        gumbel_c_visit
        + float(np.mean([max(r["root_n"].values() or [0.0]) for r in usable]))
    ) * gumbel_c_scale
    logits = np.log(np.clip(prior, 1e-12, None)) + scale * shrink_w * minmax_unit(q_bar)
    target = np.exp(logits - logits.max())
    target /= target.sum()
    q_sorted = np.sort(q_bar)[::-1]
    info = {
        "w": shrink_w,
        "spread": float(q_bar.max() - q_bar.min()),
        "gap": float(q_sorted[0] - q_sorted[1]) if len(q_sorted) >= 2 else 0.0,
    }
    return target.astype(np.float32), info


def _random_replicates(rng, valid, replicates=3, q_scale=0.05, unvisited_frac=0.2):
    """Committee tables with realistic structure: a shared node mean, per-
    action deviations at ``q_scale``, replicate noise, some actions left
    unvisited by some replicates (root_n = 0), one replicate optionally
    failed."""
    base = {a: rng.gauss(0.0, q_scale) for a in valid}
    out = []
    for _ in range(replicates):
        root_q, root_n, root_prior = {}, {}, {}
        raw_prior = {a: rng.random() + 1e-3 for a in valid}
        z = sum(raw_prior.values())
        for a in valid:
            visited = rng.random() > unvisited_frac
            root_n[a] = float(rng.randint(1, 400)) if visited else 0.0
            root_q[a] = base[a] + rng.gauss(0.0, 0.02)
            root_prior[a] = raw_prior[a] / z
        out.append(
            {"ok": True, "root_q": root_q, "root_n": root_n, "root_prior": root_prior}
        )
    return out


@pytest.mark.parametrize("seed", range(8))
def test_summary_tilt_matches_legacy_builder_bitwise(seed):
    rng = random.Random(seed)
    valid = sorted(rng.sample(range(1, 40), rng.randint(2, 7)))
    reps = _random_replicates(rng, valid)
    if seed % 3 == 0:
        reps.append({"ok": False, "root_q": None, "root_n": {}, "root_prior": None})
    base_prior = None
    if seed % 2:
        base_prior = np.array([rng.random() + 0.01 for _ in valid])
    legacy = _legacy_build_ce_search_target(
        reps, valid, **SHRINK, **GUMBEL, base_prior=base_prior
    )
    new = build_ce_search_target(reps, valid, **SHRINK, **GUMBEL, base_prior=base_prior)
    assert legacy is not None and new is not None
    assert np.array_equal(legacy[0], new[0])
    assert legacy[1] == new[1]


def test_summary_fields_and_completion():
    rng = random.Random(3)
    valid = [5, 9, 12, 20]
    reps = _random_replicates(rng, valid, unvisited_frac=0.0)
    # Make action 20 unvisited in every replicate: it must be completed with
    # the visit-weighted mean of the others and carry n_mean = 0.
    for r in reps:
        r["root_n"][20] = 0.0
    s = summarize_committee(reps, valid, **SHRINK)
    assert s is not None
    assert isinstance(s, CommitteeSummary)
    assert s.actions == (5, 9, 12, 20)
    assert s.n_mean[3] == 0.0
    visited = [0, 1, 2]
    v_mix = sum(s.n_mean[i] * s.q_mean[i] for i in visited) / sum(
        s.n_mean[i] for i in visited
    )
    assert s.q_mean[3] == pytest.approx(v_mix)
    # Unvisited action: one replicate's worth of the global variance.
    assert s.q_var[3] == pytest.approx(SHRINK["shrink_s2_global"])
    assert s.noise_var == pytest.approx(float(np.mean(s.q_var[visited])))
    assert s.spread == pytest.approx(float(s.q_mean.max() - s.q_mean.min()))
    top2 = np.sort(s.q_mean)[::-1][:2]
    assert s.gap == pytest.approx(float(top2[0] - top2[1]))
    assert 0.0 <= s.w <= 1.0
    assert s.prior.sum() == pytest.approx(1.0, abs=1e-9)
    fields = s.as_row_fields()
    assert fields["search_stats_source"] == "committee"
    assert len(fields["search_q"]) == len(valid) == len(fields["search_q_var"])
    assert all(isinstance(x, float) for x in fields["search_q"])


def test_within_noise_spread_shrinks_to_flat_target():
    """A node whose Q spread is inside replicate noise gets w = 0, and the
    tilt then returns the prior exactly (abstention = target fixed point)."""
    valid = [1, 2, 3]
    reps = []
    for rep in range(3):
        reps.append(
            {
                "ok": True,
                "root_q": {a: 0.1 + 1e-5 * ((a + rep) % 3) for a in valid},
                "root_n": {a: 100.0 for a in valid},
                "root_prior": {1: 0.5, 2: 0.3, 3: 0.2},
            }
        )
    s = summarize_committee(reps, valid, **SHRINK)
    assert s is not None
    assert s.w == 0.0
    target, info = tilt_summary_to_target(s, **GUMBEL)
    assert info["w"] == 0.0
    assert np.allclose(target, [0.5, 0.3, 0.2], atol=1e-6)


def test_fewer_than_two_usable_replicates_is_none():
    valid = [1, 2]
    reps = [
        {
            "ok": True,
            "root_q": {1: 0.0, 2: 0.1},
            "root_n": {1: 5.0, 2: 5.0},
            "root_prior": {1: 0.5, 2: 0.5},
        },
        {"ok": False, "root_q": None, "root_n": {}, "root_prior": None},
    ]
    assert summarize_committee(reps, valid, **SHRINK) is None
    assert build_ce_search_target(reps, valid, **SHRINK, **GUMBEL) is None
