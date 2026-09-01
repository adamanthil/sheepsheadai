#!/usr/bin/env python3
"""Lift a schema-1 distillation corpus to schema-2-equivalent shards by
recovering the pooled committee Q vector from the stored pi_gumbel target
(CE_Teacher_Design §20.3).

Schema-1 override rows (``distill_corpus.py`` before row schema 2) stored
only the tilted target

    t = softmax( log p + c · minmax(q̄) ),   c = (c_visit + max N) · c_scale · w

with p theta_k's act-time policy over the legal set and c a per-node
constant that was NOT stored. But ``u = log t − log p = c · minmax(q̄) +
const``, and the node telemetry (``nodes.jsonl``) kept the pooled Q's
``spread`` (max − min) and top-2 ``gap`` independently. So:

1. ``p`` is theta_k's REPLAYED policy at the row (the trainer's recurrent
   unroll reproduces the act-time stash to float noise, §17.12).
2. The scale ``c`` is pinned by the telemetry gap: for the top pair
   (a1, a2), ``u[a1] − u[a2] = c · gap / spread``. Pinning on the top pair
   rather than on min-max keeps the recovery exact even when the worst
   card's target underflowed float32 (c > ~87 nats).
3. ``minmax(q̄)[a] = (u[a] − u[a1]) / c + 1``, and ``q̄ = spread ·
   minmax(q̄)`` up to an additive per-node offset that no consumer needs
   (advantages are centered per node).
4. The node's noise variance follows from the stored w by definition:
   ``noise_var = (1 − w) · Var(q̄)``.

Each row is VERIFIED before it is trusted: the recovered top pair must be
the telemetry's top pair in order, the recovered min-max minimum must be 0
within ``--tol`` (the min is not used by the pinning, so this checks the
whole chain — replayed prior, float precision, telemetry join), and the
recovered scale ``c / w`` must lie inside the engine's possible tilt range.
Rows failing any check are marked ``search_stats_source =
"recovery_failed"`` with a reason and carry no Q. Endorsed rows (w = 0)
never stored a target and are marked ``"unrecoverable"``; they keep gap /
spread and receive their §20 targets from the pooled model alone.

Per-action sampling variances and visit counts are not recoverable; the
recovered rows carry the node noise variance in every ``search_q_var``
slot and no ``search_n`` / ``search_prior``. Override rows also gain
``anchor_probs`` (= the replayed p), which schema 1 stored only on
endorsed/retention rows.

Usage:
  uv run python -m sheepshead.training.recover_search_q \\
      --corpus-dir runs/distill_corpus_q_202608 \\
      --telemetry runs/distill_corpus_q_202608/nodes.jsonl \\
      --ckpt runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt \\
      --out-dir runs/distill_corpus_q_202608_recovered
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from sheepshead.agent.ppo import load_agent
from sheepshead.training.corpus_rows import (
    dense_to_sorted_valid,
    iter_row_batches,
    load_shard,
    replayed_policy_rows,
    shard_paths,
    store_corpus_episodes,
)
from sheepshead.training.distill_corpus import ROW_SCHEMA_VERSION

# The engine's tilt scale is (c_visit + max N) * c_scale * w with c_visit 50,
# c_scale 0.1 and 1 <= max N <= iters; the recovered c / w must sit inside
# this window (a generous margin on both sides for float noise).
TILT_PER_W_MIN = 50.0 * 0.1 * 0.5
TILT_PER_W_MAX = (50.0 + 4096.0) * 0.1 * 1.05

# Rows whose stored target fails the replay/telemetry consistency checks.
RECOVERY_FAILED = "recovery_failed"
# w = 0 rows: no target was ever stored, nothing to recover.
UNRECOVERABLE = "unrecoverable"
RECOVERED = "recovered"


def telemetry_key(game: int, cls: str, w: float, gap: float) -> tuple:
    """Join key between a corpus row and its telemetry line: the generator
    wrote both from the same ``info`` dict, so w and gap match exactly."""
    return (int(game), str(cls), round(float(w or 0.0), 9), round(float(gap or 0.0), 9))


def load_telemetry_index(path: str) -> dict[tuple, list[dict]]:
    """``telemetry_key -> [rows]``; a key maps to several rows only when two
    searched nodes of one game hit identical (class, w, gap), which the
    recovery treats as ambiguous unless their spread / top pair agree."""
    index: dict[tuple, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("w") is None:
                continue  # committee failure: nothing was stored for it
            index[
                telemetry_key(row["game"], row["class"], row["w"], row["gap"])
            ].append(row)
    return index


@dataclass
class RecoveryStats:
    rows: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    min_residuals: list = field(default_factory=list)  # |min minmax| on ok rows
    tilt_per_w: list = field(default_factory=list)

    def summary(self) -> dict:
        res = np.array(self.min_residuals) if self.min_residuals else np.zeros(0)
        return {
            "rows": dict(self.rows),
            "failure_reasons": dict(self.reasons),
            "min_residual_p50": float(np.median(res)) if res.size else None,
            "min_residual_max": float(res.max()) if res.size else None,
            "tilt_per_w_p50": (
                float(np.median(self.tilt_per_w)) if self.tilt_per_w else None
            ),
        }


def recover_row_q(
    target: np.ndarray,
    prior: np.ndarray,
    *,
    w: float,
    gap: float,
    spread: float,
    top_pair_idx: tuple[int, int] | None,
    tol: float,
) -> tuple[np.ndarray | None, float | None, str]:
    """Recover ``q̄`` (offset-free: max = spread, min ≈ 0) from one stored
    target. Returns ``(q, tilt_per_w, reason)``; ``q`` is None on failure
    and ``reason`` names the failed check ("ok" otherwise).

    ``top_pair_idx`` are the telemetry top-2 positions in sorted-valid
    order; when the telemetry gap is ~0 the scale is pinned by min-max
    instead (an exact top tie leaves no pair to pin on)."""
    if w <= 0.0 or spread <= 0.0:
        return None, None, "not_material"
    t = np.asarray(target, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    underflow = t <= 0.0
    u = np.full_like(t, -np.inf)
    u[~underflow] = np.log(t[~underflow]) - np.log(p[~underflow])
    finite = np.isfinite(u)
    if finite.sum() < 2:
        return None, None, "underflow"

    if gap > 1e-9 and top_pair_idx is not None:
        a1, a2 = top_pair_idx
        if not (finite[a1] and finite[a2]):
            return None, None, "top_pair_underflow"
        du = u[a1] - u[a2]
        if du <= 0.0:
            return None, None, "top_pair_order"
        c = du * spread / gap
        u_top = u[a1]
    else:
        # Exact top tie: pin on min-max (only valid when nothing underflowed).
        if underflow.any():
            return None, None, "underflow"
        c = u.max() - u.min()
        if c <= 0.0:
            return None, None, "flat_target"
        u_top = u.max()

    minmax = np.zeros_like(u)
    minmax[finite] = (u[finite] - u_top) / c + 1.0
    # Consistency: without underflow the minimum must sit at 0 (the min-max
    # floor) — it is not used by the pinning, so this checks the whole
    # chain. With underflow the true minimum is the underflowed card, so
    # only the [0, 1] range of the surviving entries can be checked.
    min_resid = float(minmax[finite].min())
    if not underflow.any() and abs(min_resid) > tol:
        return None, None, "min_residual"
    if min_resid < -tol:
        return None, None, "min_residual"
    if minmax.max() > 1.0 + tol:
        return None, None, "max_residual"
    tilt_per_w = c / w
    if not (TILT_PER_W_MIN <= tilt_per_w <= TILT_PER_W_MAX):
        return None, None, "tilt_range"
    q = np.clip(minmax, 0.0, 1.0) * spread
    if top_pair_idx is not None and gap > 1e-9:
        order = np.argsort(-q)
        if (int(order[0]), int(order[1])) != (top_pair_idx[0], top_pair_idx[1]):
            return None, None, "top_pair_mismatch"
    return q, tilt_per_w, "ok"


def _telemetry_match(index, source_event, game_idx) -> tuple[dict | None, str]:
    """The telemetry row for a corpus event, or (None, reason). Two searched
    nodes of one game can share (class, w, gap); the row's own legal set
    then disambiguates (the telemetry's n_valid and top pair must fit it),
    and survivors that still disagree on spread / top pair are refused."""
    key = telemetry_key(
        game_idx,
        source_event["node_class"],
        source_event.get("search_w"),
        source_event.get("search_gap"),
    )
    cands = index.get(key)
    if not cands:
        return None, "no_telemetry"
    valid = set(source_event["valid_actions"])
    fitting = [
        c
        for c in cands
        if c.get("n_valid", len(valid)) == len(valid)
        and all(a in valid for a in (c.get("top_pair") or []))
    ]
    if not fitting:
        return None, "telemetry_legal_set_mismatch"
    spreads = {round(c["spread"], 9) for c in fitting}
    pairs = {tuple(c.get("top_pair") or ()) for c in fitting}
    if len(spreads) > 1 or len(pairs) > 1:
        return None, "ambiguous_telemetry"
    return fitting[0], "ok"


def recover_episodes(
    agent,
    episodes: list,
    game_indices: list[int],
    index: dict,
    *,
    tol: float,
    batch_segments: int,
    stats: RecoveryStats,
) -> None:
    """Recover in place: every searched action row of ``episodes`` gains
    the schema-2 fields it can. ``game_indices[i]`` is the corpus game
    index of episode ``i`` (five consecutive episodes share a game)."""
    agent.reset_storage()
    source_map = store_corpus_episodes(agent, episodes)
    for rows in iter_row_batches(agent, batch_segments=batch_segments):
        probs = replayed_policy_rows(agent, rows)
        for r, ev_idx in enumerate(rows.event_indices):
            src_ref = source_map[ev_idx]
            assert src_ref is not None
            ep_idx, ev_pos = src_ref
            src = episodes[ep_idx][ev_pos]
            dset = src.get("distill_set", "none")
            if dset not in ("override", "endorsed"):
                continue
            if src.get("search_stats_source") == "committee":
                stats.rows["already_schema2"] += 1
                continue
            valid = src["valid_actions"]
            replay_p = dense_to_sorted_valid(probs[r], valid)
            if src.get("anchor_probs") is None:
                src["anchor_probs"] = replay_p
                src["anchor_source"] = "replay"
            tele, reason = _telemetry_match(index, src, game_indices[ep_idx])
            if tele is not None:
                src["search_spread"] = float(tele["spread"])
            if dset == "endorsed":
                src["search_stats_source"] = UNRECOVERABLE
                stats.rows[UNRECOVERABLE] += 1
                continue
            if tele is None:
                src["search_stats_source"] = RECOVERY_FAILED
                src["search_recovery_reason"] = reason
                stats.rows[RECOVERY_FAILED] += 1
                stats.reasons[reason] += 1
                continue
            acts = sorted(valid)
            pair = tele.get("top_pair") or []
            top_pair_idx = (
                (acts.index(pair[0]), acts.index(pair[1]))
                if len(pair) == 2 and pair[0] in acts and pair[1] in acts
                else None
            )
            q, tilt_per_w, reason = recover_row_q(
                np.asarray(src["search_target"], dtype=np.float64),
                np.asarray(replay_p, dtype=np.float64),
                w=float(src["search_w"]),
                gap=float(tele["gap"]),
                spread=float(tele["spread"]),
                top_pair_idx=top_pair_idx,
                tol=tol,
            )
            if q is None:
                src["search_stats_source"] = RECOVERY_FAILED
                src["search_recovery_reason"] = reason
                stats.rows[RECOVERY_FAILED] += 1
                stats.reasons[reason] += 1
                continue
            w = float(src["search_w"])
            noise_var = (1.0 - w) * float(np.var(q))
            src["search_q"] = [float(x) for x in q]
            src["search_q_var"] = [noise_var] * len(acts)
            src["search_noise_var"] = noise_var
            src["search_stats_source"] = RECOVERED
            stats.rows[RECOVERED] += 1
            stats.tilt_per_w.append(tilt_per_w)
            stats.min_residuals.append(abs(float(np.min(q))))
    agent.reset_storage()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--telemetry", required=True, help="the corpus's nodes.jsonl")
    ap.add_argument("--ckpt", required=True, help="theta_k (the corpus's --ckpt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--tol", type=float, default=1e-4, help="min-max residual tolerance"
    )
    ap.add_argument("--buffer-episodes", type=int, default=250)
    ap.add_argument("--batch-segments", type=int, default=32)
    ap.add_argument(
        "--shards", type=int, default=0, help="only the first N shards (0 = all)"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.corpus_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("row_schema", 1) >= ROW_SCHEMA_VERSION:
        raise SystemExit("corpus already carries row schema 2; nothing to recover")
    ckpt_manifest = manifest.get("ckpt")
    if ckpt_manifest and os.path.abspath(ckpt_manifest) != os.path.abspath(args.ckpt):
        print(
            f"WARNING: --ckpt {args.ckpt} differs from the manifest's "
            f"{ckpt_manifest}; the replayed prior must be theta_k's",
            flush=True,
        )

    agent = load_agent(args.ckpt)
    for net in (agent.encoder, agent.actor, agent.critic):
        for p in net.parameters():
            p.requires_grad_(False)
    index = load_telemetry_index(args.telemetry)
    stats = RecoveryStats()
    paths = shard_paths(args.corpus_dir)
    if args.shards:
        paths = paths[: args.shards]
    for path in paths:
        shard = load_shard(path)
        episodes = shard["episodes"]
        games = shard["games"]
        if len(episodes) != 5 * len(games):
            raise SystemExit(f"{path}: {len(episodes)} episodes for {len(games)} games")
        game_indices = [g["game"] for g in games for _ in range(5)]
        for start in range(0, len(episodes), args.buffer_episodes):
            chunk = episodes[start : start + args.buffer_episodes]
            recover_episodes(
                agent,
                chunk,
                game_indices[start : start + args.buffer_episodes],
                index,
                tol=args.tol,
                batch_segments=args.batch_segments,
                stats=stats,
            )
        out_path = os.path.join(args.out_dir, os.path.basename(path))
        torch.save({"episodes": episodes, "games": games}, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
        print(f"{os.path.basename(path)}: {stats.summary()['rows']}", flush=True)

    manifest["row_schema"] = ROW_SCHEMA_VERSION
    manifest["recovered_from"] = os.path.abspath(args.corpus_dir)
    manifest["recovery"] = {"tol": args.tol, **stats.summary()}
    manifest["shards"] = manifest["shards"][: len(paths)]
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.copy(args.telemetry, os.path.join(args.out_dir, "nodes.jsonl"))
    print(f"DONE: {json.dumps(stats.summary())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
