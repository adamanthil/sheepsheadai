"""``recover_search_q``: schema-1 corpus rows -> recovered pooled Q
(CE_Teacher_Design §20.3).

A real agent generates a schema-2 corpus with the scripted committee, the
schema-2 evidence is stripped back to what schema 1 stored, and the
recovery must reproduce the pooled Q vector (up to its per-node offset),
the node noise variance and the act-time anchor from the stored target,
the replayed policy and the telemetry alone.
"""

import numpy as np
import pytest

from sheepshead.tests.test_distill_pipeline import _fresh_agent, _generate_game
from sheepshead.training.recover_search_q import (
    RECOVERED,
    RECOVERY_FAILED,
    UNRECOVERABLE,
    RecoveryStats,
    load_telemetry_index,
    recover_episodes,
    recover_row_q,
    telemetry_key,
)

SCHEMA2_FIELDS = (
    "search_q",
    "search_q_var",
    "search_n",
    "search_prior",
    "search_noise_var",
    "search_spread",
    "search_stats_source",
)


def _strip_to_schema1(episodes):
    """What the §17 generator stored: no pooled evidence, anchors only on
    endorsed/retention rows. Returns the stripped-away originals keyed by
    (episode, event) for the comparison."""
    originals = {}
    for ep_idx, ep in enumerate(episodes):
        for ev_idx, ev in enumerate(ep):
            if ev.get("kind") != "action":
                continue
            if "search_q" in ev:
                originals[(ep_idx, ev_idx)] = {k: ev[k] for k in SCHEMA2_FIELDS}
                originals[(ep_idx, ev_idx)]["anchor_probs"] = ev["anchor_probs"]
            for k in SCHEMA2_FIELDS:
                ev.pop(k, None)
            if ev["distill_set"] not in ("endorsed", "retention"):
                ev["anchor_probs"] = None
    return originals


def _index_from_generated(res, tmp_path):
    path = tmp_path / "nodes.jsonl"
    import json

    with open(path, "w") as f:
        for row in res["telemetry"]:
            f.write(json.dumps(row) + "\n")
    return load_telemetry_index(str(path))


def test_recovery_round_trip_on_generated_corpus(tmp_path):
    agent = _fresh_agent()
    res = _generate_game(agent, game_idx=3)
    episodes = res["episodes"]
    originals = _strip_to_schema1(episodes)
    assert originals, "the scripted committee is always material"
    index = _index_from_generated(res, tmp_path)
    stats = RecoveryStats()
    recover_episodes(
        agent,
        episodes,
        [3] * len(episodes),
        index,
        tol=1e-4,
        batch_segments=8,
        stats=stats,
    )
    assert stats.rows[RECOVERED] == len(originals)
    assert stats.rows[RECOVERY_FAILED] == 0
    for (ep_idx, ev_idx), orig in originals.items():
        ev = episodes[ep_idx][ev_idx]
        assert ev["search_stats_source"] == RECOVERED
        q_orig = np.asarray(orig["search_q"])
        q_rec = np.asarray(ev["search_q"])
        # Offset-free identity: the recovery pins max = spread, min = 0.
        assert np.allclose(q_rec - q_rec.max(), q_orig - q_orig.max(), atol=1e-5)
        assert ev["search_spread"] == pytest.approx(orig["search_spread"])
        # noise_var = (1 - w) * Var(q) is the shrinkage identity.
        assert ev["search_noise_var"] == pytest.approx(
            orig["search_noise_var"], rel=1e-3, abs=1e-9
        )
        assert ev["anchor_source"] == "replay"
        assert np.allclose(ev["anchor_probs"], orig["anchor_probs"], atol=1e-5)
        assert "search_n" not in ev and "search_prior" not in ev
    # Sanity of the verification instruments on a clean corpus.
    summary = stats.summary()
    assert summary["min_residual_max"] < 1e-4
    assert 5.0 <= summary["tilt_per_w_p50"] <= 430.0


def test_endorsed_rows_are_unrecoverable_and_keep_spread(tmp_path):
    agent = _fresh_agent()
    res = _generate_game(agent, game_idx=3)
    episodes = res["episodes"]
    # Turn one override row into what a w = 0 row looked like in schema 1.
    target_row = next(
        e
        for ep in episodes
        for e in ep
        if e.get("kind") == "action" and e["distill_set"] == "override"
    )
    tele = next(
        r
        for r in res["telemetry"]
        if telemetry_key(3, r["class"], r["w"], r["gap"])
        == telemetry_key(
            3,
            target_row["node_class"],
            target_row["search_w"],
            target_row["search_gap"],
        )
    )
    target_row["distill_set"] = "endorsed"
    target_row["has_search_target"] = False
    target_row["search_target"] = None
    _strip_to_schema1(episodes)
    index = _index_from_generated(res, tmp_path)
    stats = RecoveryStats()
    recover_episodes(
        agent,
        episodes,
        [3] * len(episodes),
        index,
        tol=1e-4,
        batch_segments=8,
        stats=stats,
    )
    assert stats.rows[UNRECOVERABLE] == 1
    assert target_row["search_stats_source"] == UNRECOVERABLE
    assert target_row["search_spread"] == pytest.approx(tele["spread"])
    assert "search_q" not in target_row
    assert target_row["anchor_probs"] is not None


def test_recover_row_q_checks():
    """Synthetic single-row checks of the pinning and the guards."""
    rng = np.random.default_rng(0)
    q = np.array([0.30, 0.28, 0.10, 0.0])  # spread 0.3, gap 0.02, top pair (0, 1)
    p = rng.dirichlet(np.ones(4))
    w = 0.7
    c = (50.0 + 300.0) * 0.1 * w
    minmax = (q - q.min()) / (q.max() - q.min())
    logits = np.log(p) + c * minmax
    t = np.exp(logits - logits.max())
    t /= t.sum()
    rec, tilt_per_w, reason = recover_row_q(
        t.astype(np.float32),
        p,
        w=w,
        gap=0.02,
        spread=0.3,
        top_pair_idx=(0, 1),
        tol=1e-4,
    )
    assert reason == "ok" and rec is not None
    assert np.allclose(rec, q, atol=1e-5)
    assert tilt_per_w == pytest.approx(35.0, rel=1e-3)
    # Underflow of the worst card must not break the top-pair pinning.
    big_c = 120.0
    logits = np.log(p) + big_c * minmax
    t = np.exp(logits - logits.max())
    t /= t.sum()
    t32 = t.astype(np.float32)
    assert t32[3] == 0.0, "the test needs a genuine float32 underflow"
    rec, _, reason = recover_row_q(
        t32, p, w=1.0, gap=0.02, spread=0.3, top_pair_idx=(0, 1), tol=1e-4
    )
    assert reason == "ok" and rec is not None
    assert np.allclose(rec[:3], q[:3], atol=1e-4) and rec[3] == 0.0
    # A wrong prior breaks the min-residual check rather than passing
    # silently (on a non-underflowed target, where the check is armed).
    logits = np.log(p) + c * minmax
    t = np.exp(logits - logits.max())
    t /= t.sum()
    wrong_p = rng.dirichlet(np.ones(4))
    _, _, reason = recover_row_q(
        t.astype(np.float32),
        wrong_p,
        w=w,
        gap=0.02,
        spread=0.3,
        top_pair_idx=(0, 1),
        tol=1e-4,
    )
    assert reason != "ok"
    # Non-material rows are refused outright.
    assert (
        recover_row_q(
            t32, p, w=0.0, gap=0.02, spread=0.3, top_pair_idx=(0, 1), tol=1e-4
        )[2]
        == "not_material"
    )
