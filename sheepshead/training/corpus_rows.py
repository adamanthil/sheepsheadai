"""Corpus row plumbing shared by the §20 tools (CE_Teacher_Design §20.3):
``recover_search_q.py`` and ``train_policy_iteration.py``.

Both need to walk a distillation corpus (``distill_corpus.py`` shards)
through the agent's own replayed recurrent unroll and get back, PER ACTION
ROW, the things the search-Q policy-iteration recipe reads:

- the source corpus event the row came from (so recovered / built fields
  can be written back onto it),
- theta_k's replayed policy at that row (the act-time stash reproduces
  under replay to float noise — CE_Teacher_Design §17.12 — so replay is
  the reference wherever a stash is missing),
- the frozen encoder outputs the play pointer reads (the 256-d shared
  readout and the 8 post-reasoning hand tokens), which the advantage
  model regresses on.

Row order is the agent's flatten order (``PPOAgent._flatten_action_steps``):
segments in batch order, action events in chronological order within a
segment. Every helper here returns rows in that order so tensors and
source references line up by index.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F

from sheepshead.agent import ppo as ppo_module
from sheepshead.agent.ppo import MinibatchTensors, PPOAgent
from sheepshead.training.training_utils import RETURN_SCALE

HAND_SLOTS = 8  # the encoder's fixed hand width

# Annotates a freshly stored event record with fields from its source corpus
# event (the trainer's distill channels, the recovery's row references).
RecordAnnotator = Callable[[dict, dict], None]


def store_corpus_episodes(
    agent: PPOAgent,
    episodes: Sequence[list],
    annotate: RecordAnnotator | None = None,
) -> list[tuple[int, int] | None]:
    """Store corpus episodes into ``agent.events`` and return the source map:
    for every stored event index, ``(episode_idx, event_idx)`` into
    ``episodes`` (``None`` only if the agent ever stores an event with no
    source, which ``store_episode_events`` does not do).

    ``store_episode_events`` appends action and observation records in the
    exact order of the source events, so the map is a positional walk.
    ``annotate(record, source_event)`` runs on every stored ACTION record
    with its source event, for callers that need extra per-row fields.
    """
    source_map: list[tuple[int, int] | None] = []
    for ep_idx, episode in enumerate(episodes):
        start = len(agent.events)
        agent.store_episode_events(episode)
        stored = agent.events[start:]
        if len(stored) != len(episode):
            raise RuntimeError(
                f"store_episode_events stored {len(stored)} records for an "
                f"episode of {len(episode)} events; the source map assumes 1:1"
            )
        for ev_idx, (record, source) in enumerate(zip(stored, episode)):
            source_map.append((ep_idx, ev_idx))
            if record["kind"] != "action":
                continue
            # Supervised-phase defaults the minibatch builder requires:
            # MC terminal return as the value target (gamma = 1), no
            # advantages (PG is off), oracle targets mirroring the return.
            ret = record["final_return"] / RETURN_SCALE
            record["return"] = ret
            record["advantage"] = 0.0
            record["return_oracle"] = ret
            record["value_oracle"] = 0.0
            if annotate is not None:
                annotate(record, source)
    return source_map


def action_rows_in_flatten_order(batch, kinds) -> list[int]:
    """Event indices of the action rows of ``batch`` (a list of
    ``(seg_start, seg_end)`` segments), in the agent's flatten order."""
    return [
        i
        for seg_start, seg_end in batch
        for i in range(seg_start, seg_end + 1)
        if kinds[i] == "action"
    ]


@dataclass
class RowBatch:
    """One minibatch of corpus rows, everything indexed in flatten order."""

    segments: list
    minibatch: MinibatchTensors
    kinds: list
    event_indices: list[int]  # into agent.events, one per action row

    def __len__(self) -> int:
        return len(self.event_indices)


def iter_row_batches(
    agent: PPOAgent,
    *,
    batch_segments: int,
    shuffle: bool = False,
    rng: random.Random | None = None,
) -> Iterator[RowBatch]:
    """Yield minibatches over the episodes currently stored in
    ``agent.events`` (see ``store_corpus_episodes``). A segment is one
    seat's full episode; ``batch_segments`` of them form a minibatch."""
    states, masks_t, kinds = agent._prepare_training_views()
    segments = agent._segments_from_events(kinds)
    order = list(range(len(segments)))
    if shuffle:
        (rng or random).shuffle(order)
    for start in range(0, len(order), batch_segments):
        batch = [segments[i] for i in order[start : start + batch_segments]]
        minibatch = agent._build_minibatch_tensors(batch, states, masks_t, kinds)
        yield RowBatch(
            segments=batch,
            minibatch=minibatch,
            kinds=kinds,
            event_indices=action_rows_in_flatten_order(batch, kinds),
        )


def replayed_policy_rows(agent: PPOAgent, rows: RowBatch) -> torch.Tensor:
    """``(R, action_size)`` masked policy probabilities of ``agent`` at each
    action row, from its replayed recurrent unroll. Zero off the legal set
    (illegal logits are -1e8 before the softmax)."""
    with torch.no_grad():
        forward = agent._forward_vectorized(
            rows.minibatch.states_seqs, rows.minibatch.masks_bt
        )
        flat = agent._flatten_action_steps(rows.minibatch, forward)
    if flat is None:
        return torch.zeros((0, agent.action_size), device=ppo_module.device)
    return F.softmax(flat.logits_flat, dim=-1)


@dataclass
class EncodedRows:
    """Frozen-encoder outputs at each action row of a ``RowBatch``: what the
    play pointer reads (CE_Teacher_Design §20.2, "features")."""

    features: torch.Tensor  # (R, d_model) shared readout, pre-adapter
    hand_tokens: torch.Tensor  # (R, HAND_SLOTS, d_token) post-reasoning
    hand_ids: torch.Tensor  # (R, HAND_SLOTS) card ids, 0 on empty slots
    masks: torch.Tensor  # (R, action_size) legal-action mask


def encode_rows(agent: PPOAgent, rows: RowBatch) -> EncodedRows:
    """Run the agent's encoder over the batch's replayed sequences and
    select the action rows. Mirrors the encode half of
    ``PPOAgent._forward_vectorized`` (zero initial memory per segment,
    ``encode_sequences`` carries the recurrent state across steps) without
    the actor/critic heads. Gradient flows if the caller enables it."""
    device = ppo_module.device
    states = rows.minibatch.states_seqs
    B = len(states)
    T = rows.minibatch.masks_bt.size(1)
    memory_init = torch.zeros((B, agent.state_size), device=device)
    out = agent.encoder.encode_sequences(states, memory_in=memory_init, device=device)
    hand_ids_bt = torch.zeros((B, T, HAND_SLOTS), dtype=torch.long, device=device)
    for b, seq in enumerate(states):
        for t, state in enumerate(seq):
            if t >= T:
                break
            ids = torch.as_tensor(state["hand_ids"], dtype=torch.long, device=device)
            ids = ids.view(-1)
            hand_ids_bt[b, t, : min(HAND_SLOTS, ids.numel())] = ids[:HAND_SLOTS]
    flat_mask = rows.minibatch.is_action_bt.view(-1)
    features = out["features"].reshape(B * T, -1)[flat_mask]
    hand_tokens = out["hand_tokens"].reshape(B * T, HAND_SLOTS, -1)[flat_mask]
    hand_ids = hand_ids_bt.reshape(B * T, HAND_SLOTS)[flat_mask]
    masks = rows.minibatch.masks_bt.reshape(B * T, -1)[flat_mask]
    return EncodedRows(
        features=features, hand_tokens=hand_tokens, hand_ids=hand_ids, masks=masks
    )


def dense_to_sorted_valid(dense: torch.Tensor, valid_actions) -> list[float]:
    """Pick the entries of a dense ``(action_size,)`` vector at
    ``sorted(valid_actions)`` (1-indexed action ids)."""
    return [float(dense[a - 1]) for a in sorted(valid_actions)]


def sorted_valid_to_dense(values, valid_actions, action_size: int) -> list[float]:
    """Inverse of ``dense_to_sorted_valid``: zeros off the legal set."""
    dense = [0.0] * action_size
    for a, v in zip(sorted(valid_actions), values):
        dense[a - 1] = float(v)
    return dense


def load_shard(path: str) -> dict:
    """One corpus shard: ``{"episodes": [...], "games": [...]}``. Shards
    carry numpy scalars/arrays in their event dicts, so the weights-only
    unpickler is off."""
    return torch.load(path, map_location="cpu", weights_only=False)


def shard_paths(corpus_dir: str) -> list[str]:
    paths = sorted(
        os.path.join(corpus_dir, f)
        for f in os.listdir(corpus_dir)
        if f.startswith("corpus_") and f.endswith(".pt")
    )
    if not paths:
        raise SystemExit(f"no corpus shards in {corpus_dir}")
    return paths
