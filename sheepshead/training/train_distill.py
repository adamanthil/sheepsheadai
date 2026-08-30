#!/usr/bin/env python3
"""Supervised distillation trainer for the phased-offline pilot
(CE_Teacher_Design §17.4).

PG is OFF by construction: no importance ratios, no advantages, no
entropy controller — a plain supervised loop over a fixed corpus written
by ``distill_corpus.py``. This is the phase-separation answer to the
§16.9 mechanism: CE toward search targets is itself a KL-regularized
policy improvement step (Grill et al. 2020, arXiv:2007.12509), and
running it CONCURRENTLY with PG on a shared trunk gave two proximal
operators with different centers (attempts 11/12). Here the improvement
operator runs alone against fixed targets (AlphaGo Zero / AlphaZero
projection — Silver et al. 2017/2018; Expert Iteration — Anthony et al.
2017), and PG returns, if at all, only in a later separately-certified
phase.

Per-row policy loss by partition (§16.9 addendum 6):

  override   lambda_ce * omega * CE(t || pi_theta): t is the shrink-and-
             tilt committee target; omega = min(exp(gap/beta),
             omega_max)/omega_max — advantage/evidence-weighted
             regression in the AWR/AWAC/CRR family (Peng et al. 2019,
             arXiv:1910.00177; Nair et al. 2020, arXiv:2006.09359;
             Wang et al. 2020, arXiv:2006.15134).
  endorsed   lambda_end * tau^2 * KL(anchor_tau || pi_tau): knowledge-
             distillation anchor to theta_k's act-time distribution
             (Hinton et al. 2015, arXiv:1503.02531) — the Learning-
             without-Forgetting recipe (Li & Hoiem 2016,
             arXiv:1606.09282) applied where search spoke and endorsed.
  retention  identical form, separate coefficient + telemetry: bidding
             heads and leaster play (alone play is searched). Endorsed-
             KL rising = taught-region play drift; retention-KL rising =
             attempt-9/12-style collateral onset in untaught heads —
             the two pre-registered early-warning instruments.
  none       no policy loss (eligible-unsearched play, forced nodes).

Every action row regresses the value head on the Monte-Carlo terminal
return (gamma=1; no GAE without PG) plus the standard aux heads; the
privileged oracle critic trains the same way when the corpus carries
oracle states (kept calibrated for the next phase's search leaves /
certified PG). Per-partition MEANS are combined under the lambdas, so
gradient share is decoupled from row counts (§16.9 addendum 4) and the
lambda grid is the pre-registered mixture sweep.

Usage:
  uv run python -m sheepshead.training.train_distill \\
      --corpus-dir runs/distill_corpus_202608 \\
      --ckpt runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt \\
      --out-dir runs/distill_pilot_202608 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from sheepshead import ACTIONS, TRUMP
from sheepshead.agent import ppo as ppo_module
from sheepshead.agent.ppo import load_agent
from sheepshead.training.training_utils import RETURN_SCALE, greedy_health_probe

# Partition codes for the per-row channel (0 = no policy loss).
SET_CODES = {"none": 0, "override": 1, "endorsed": 2, "retention": 3}


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_shards(corpus_dir: str) -> list:
    """All episodes from every corpus shard, in shard order."""
    paths = sorted(
        os.path.join(corpus_dir, f)
        for f in os.listdir(corpus_dir)
        if f.startswith("corpus_") and f.endswith(".pt")
    )
    if not paths:
        raise SystemExit(f"no corpus shards in {corpus_dir}")
    episodes = []
    for p in paths:
        # Shards are this pipeline's own artifacts: event dicts carry numpy
        # arrays/scalars, which the weights-only unpickler rejects.
        episodes.extend(
            torch.load(p, map_location="cpu", weights_only=False)["episodes"]
        )
    return episodes


def split_by_game(episodes: list, holdout_frac: float, seed: int) -> tuple[list, list]:
    """(train, holdout) split at GAME granularity.

    Every corpus game contributes five per-seat episodes (contiguous in
    shard order) that share one deal and one outcome — splitting at the
    episode level would put siblings of the same deal on both sides and
    bias the holdout telemetry optimistic. Games are the exchangeable
    unit, so the shuffle and the cut both happen over 5-episode groups.
    Seed-deterministic: every sweep arm sees the identical split."""
    if len(episodes) % 5 != 0:
        raise SystemExit(
            f"corpus episode count {len(episodes)} is not a multiple of 5 "
            "(all-seat collection guarantees 5 per game)"
        )
    games = [episodes[i : i + 5] for i in range(0, len(episodes), 5)]
    random.Random(seed).shuffle(games)
    n_holdout_games = int(len(games) * holdout_frac)
    holdout = [ep for game in games[:n_holdout_games] for ep in game]
    train = [ep for game in games[n_holdout_games:] for ep in game]
    return train, holdout


def densify(probs: list | None, valid_actions, action_size: int) -> list:
    """Anchor distribution over sorted(valid) -> fixed-width vector
    (zeros off the legal set), mirroring the search-target densification
    in ``store_episode_events``."""
    dense = [0.0] * action_size
    if probs is not None:
        for action_id, p in zip(sorted(valid_actions), probs):
            dense[action_id - 1] = float(p)
    return dense


def store_episodes(agent, episodes: list, gap_floor: float = 0.0) -> None:
    """Store corpus episodes into the agent's event buffer and annotate the
    stored action records with the distill channels. Reuses the exact
    ``store_episode_events`` schema (masks, target densification, aux
    labels), then walks the appended records in step with the source
    events — action records correspond 1:1 in order.

    ``gap_floor`` (§18 label pruning): override rows with a resolved
    top-2 gap BELOW the floor are demoted to NO-LOSS ("none") — search
    spoke and materially disagreed, so anchoring them to theta_k would
    anti-teach, but their labels sit inside the committee's near-tie
    noise band (§12.8) and are excluded from the CE stream.

    Also writes the supervised value targets: return = final_score /
    RETURN_SCALE at EVERY action row (MC target, terminal gamma=1), and
    the oracle equivalents when oracle states are present."""
    for ep in episodes:
        start = len(agent.events)
        agent.store_episode_events(ep)
        src_actions = iter([e for e in ep if e["kind"] == "action"])
        for rec in agent.events[start:]:
            if rec["kind"] != "action":
                continue
            src = next(src_actions)
            dset = src.get("distill_set", "none")
            if (
                dset == "override"
                and float(src.get("search_gap", 0.0) or 0.0) < gap_floor
            ):
                dset = "none"
            rec["distill_set"] = SET_CODES.get(dset, 0)
            rec["search_gap"] = float(src.get("search_gap", 0.0) or 0.0)
            rec["node_class"] = src.get("node_class", "")
            rec["conv_cs_ids"] = src.get("conv_cs_ids")
            rec["anchor_probs"] = densify(
                src.get("anchor_probs"), src["valid_actions"], agent.action_size
            )
            ret = rec["final_return"] / RETURN_SCALE
            rec["return"] = ret
            rec["advantage"] = 0.0
            rec["return_oracle"] = ret
            rec["value_oracle"] = 0.0


# --------------------------------------------------------------------------- #
# Distill channels (aligned with the agent's pad/flatten path)
# --------------------------------------------------------------------------- #
def flat_channels(agent, batch, kinds):
    """(set_flat, gap_flat, anchor_flat) aligned with
    ``_flatten_action_steps``: same per-segment padding, same
    is-action row selection order."""
    device = ppo_module.device
    lengths = []
    set_list, gap_list, anchor_list, is_act_list = [], [], [], []
    for seg_start, seg_end in batch:
        ev_range = range(seg_start, seg_end + 1)
        lengths.append(seg_end - seg_start + 1)
        sets, gaps, anchors, is_act = [], [], [], []
        for i in ev_range:
            ev = agent.events[i]
            action = kinds[i] == "action"
            is_act.append(action)
            sets.append(float(ev.get("distill_set", 0)) if action else 0.0)
            gaps.append(float(ev.get("search_gap", 0.0)) if action else 0.0)
            anchor = ev.get("anchor_probs") if action else None
            anchors.append(
                torch.tensor(
                    anchor or [0.0] * agent.action_size,
                    dtype=torch.float32,
                    device=device,
                )
            )
        set_list.append(torch.tensor(sets, dtype=torch.float32, device=device))
        gap_list.append(torch.tensor(gaps, dtype=torch.float32, device=device))
        anchor_list.append(torch.stack(anchors, dim=0))
        is_act_list.append(torch.tensor(is_act, dtype=torch.bool, device=device))
    pad = agent._pad_to_bt
    flat_mask = pad(is_act_list, lengths, False).view(-1)
    set_flat = pad(set_list, lengths, 0.0).view(-1)[flat_mask]
    gap_flat = pad(gap_list, lengths, 0.0).view(-1)[flat_mask]
    anchor_bt = pad(anchor_list, lengths, 0.0)
    anchor_flat = anchor_bt.view(-1, anchor_bt.size(-1))[flat_mask]
    return set_flat, gap_flat, anchor_flat


_IS_PLAY = [name.startswith("PLAY ") for name in ACTIONS]
_IS_TRUMP_PLAY = [name.startswith("PLAY ") and name[5:] in TRUMP for name in ACTIONS]


def convention_rows(agent, batch, kinds):
    """Per-action-row convention-telemetry annotations, aligned with the
    flatten path's row order: a list (one entry per flat action row) of
    ``(conv_name, trick, tracked_action_id_set)`` tuples (empty when the
    row is convention-ineligible; a called-suit-eligible defender lead
    carries BOTH its instruments, matching the battery's independent
    definitions).

    The three §17.4-amendment conventions, battery orientations:
      - def_trump_lead: standard-game defender lead holding both classes;
        tracked = trump leads (rate LOWER = better, t0 = the historical
        leak metric).
      - partner_trump_lead: partner lead holding both; tracked = trump
        (higher = better).
      - called_suit_lead: generation-time eligibility (the called card is
        not reconstructible from the stored row); tracked = the adherent
        ids the corpus generator stored in ``conv_cs_ids`` (higher =
        better); rows from before that field existed don't contribute.
    Eligibility for the mask-derived pair: ``node_class`` says
    std|t{k}-{role}-lead and the legal set (= the hand, at a lead)
    contains both a trump and a fail play."""
    rows = []
    for seg_start, seg_end in batch:
        for i in range(seg_start, seg_end + 1):
            if kinds[i] != "action":
                continue
            ev = agent.events[i]
            cls = ev.get("node_class", "") or ""
            entries = []
            if cls.startswith("std|") and cls.endswith("-lead"):
                trick = int(cls.split("|")[1].split("-")[0][1:])
                mask = ev["mask"]
                valid = [a + 1 for a in range(len(mask)) if bool(mask[a])]
                trump_plays = {a for a in valid if _IS_TRUMP_PLAY[a - 1]}
                fail_plays = {a for a in valid if _IS_PLAY[a - 1]} - trump_plays
                both = bool(trump_plays and fail_plays)
                if "-defender-" in cls:
                    if both:
                        entries.append(("def_trump_lead", trick, trump_plays))
                    if ev.get("conv_cs_ids"):
                        entries.append(
                            ("called_suit_lead", trick, set(ev["conv_cs_ids"]))
                        )
                elif "-partner-" in cls and both:
                    entries.append(("partner_trump_lead", trick, trump_plays))
            rows.append(entries)
    return rows


def accumulate_conventions(conv_counts, conv_rows, logits_flat):
    """Fold one minibatch's greedy convention behavior into
    ``conv_counts[(name, trick)] = [eligible, led_tracked_class]`` (masked
    logits argmax = the greedy action, the same semantics as the
    battery)."""
    greedy = logits_flat.argmax(dim=-1)
    for row_idx, entries in enumerate(conv_rows):
        for name, trick, tracked in entries:
            bin_ = conv_counts.setdefault((name, trick), [0, 0])
            bin_[0] += 1
            bin_[1] += int(int(greedy[row_idx].item()) + 1 in tracked)


def convention_report(conv_counts) -> dict:
    """Battery-oriented rates from the accumulated counts: per convention,
    the pooled rate plus t0 (the deployable-priority bin)."""
    out = {}
    for name in ("def_trump_lead", "partner_trump_lead", "called_suit_lead"):
        pooled = [0, 0]
        t0 = [0, 0]
        for (n, trick), (elig, led) in conv_counts.items():
            if n != name:
                continue
            pooled[0] += elig
            pooled[1] += led
            if trick == 0:
                t0[0] += elig
                t0[1] += led
        if pooled[0]:
            out[f"{name}_rate"] = 100.0 * pooled[1] / pooled[0]
            out[f"{name}_n"] = pooled[0]
        if t0[0]:
            out[f"t0_{name}_rate"] = 100.0 * t0[1] / t0[0]
            out[f"t0_{name}_n"] = t0[0]
    return out


def omega_weights(gap_flat: torch.Tensor, beta: float, omega_max: float):
    """§17.4 evidence weight: omega = min(exp(gap/beta), omega_max) /
    omega_max — bounded, monotone in the resolved Q-gap, 1 at strong
    evidence (AWR-family regret weighting with the AWR weight clip —
    Peng et al. 2019)."""
    return torch.exp(gap_flat / beta).clamp(max=omega_max) / omega_max


def kd_kl(anchor_flat: torch.Tensor, logits_flat: torch.Tensor, tau: float):
    """Per-row tau^2 * KL(anchor_tau || pi_tau) over the legal set.

    The anchor is a masked probability vector (zeros off-legal); its
    tau-softening softmax(log a / tau) over the legal set equals
    softmax(z_ref / tau) on the reference's masked logits, so storing
    probabilities loses nothing (Hinton et al. 2015)."""
    legal = anchor_flat > 0.0
    log_a = torch.where(
        legal,
        torch.log(anchor_flat.clamp(min=1e-12)),
        torch.full_like(anchor_flat, -1e9),
    )
    p_ref = F.softmax(log_a / tau, dim=-1)
    logp_cur = F.log_softmax(logits_flat / tau, dim=-1)
    log_p_ref = torch.log(p_ref.clamp(min=1e-12))
    return (tau * tau) * (p_ref * (log_p_ref - logp_cur)).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def distill_losses(agent, minibatch, forward, flat, dchan, args):
    """Total loss + telemetry scalars for one minibatch. Policy terms per
    partition (means, lambda-combined); value/aux terms on all rows."""
    set_flat, gap_flat, anchor_flat = dchan
    stats = {}

    logp = F.log_softmax(flat.logits_flat, dim=-1)
    zero = flat.logits_flat.new_zeros(())

    ov = set_flat == SET_CODES["override"]
    if ov.any():
        target = flat.search_target_flat[ov]
        ce = -(target * logp[ov]).sum(dim=-1)
        omega = omega_weights(gap_flat[ov], args.beta, args.omega_max)
        override_loss = (omega * ce).mean()
        with torch.no_grad():
            ent = -(target.clamp(min=1e-12) * target.clamp(min=1e-12).log()).sum(-1)
            stats["override_ce"] = float(ce.mean())
            stats["override_kl"] = float((ce - ent).mean())
            stats["override_omega"] = float(omega.mean())
    else:
        override_loss = zero
    stats["override_rows"] = int(ov.sum())

    en = set_flat == SET_CODES["endorsed"]
    if en.any():
        endorsed_loss = kd_kl(anchor_flat[en], flat.logits_flat[en], args.kd_tau).mean()
        stats["endorsed_kl"] = float(endorsed_loss.detach()) / (args.kd_tau**2)
    else:
        endorsed_loss = zero
    stats["endorsed_rows"] = int(en.sum())

    rt = set_flat == SET_CODES["retention"]
    if rt.any():
        retention_loss = kd_kl(
            anchor_flat[rt], flat.logits_flat[rt], args.kd_tau
        ).mean()
        stats["retention_kl"] = float(retention_loss.detach()) / (args.kd_tau**2)
    else:
        retention_loss = zero
    stats["retention_rows"] = int(rt.sum())

    value_loss = F.mse_loss(flat.values_flat, flat.returns_flat)
    stats["value_mse"] = float(value_loss.detach())

    total = (
        args.lambda_ce * override_loss
        + args.lambda_end * endorsed_loss
        + args.lambda_ret * retention_loss
    )
    # §17.12 floor-attribution ablation: --no-value-aux drops every
    # value-stream term (value MSE, critic aux heads; --no-oracle covers
    # the oracle) so only the policy losses touch the shared trunk.
    if getattr(args, "no_value_aux", False):
        return total, stats
    # §17.13 lever 1 (--stop-grad-value): the value MSE is EXCLUDED from
    # the joint backward — run_epoch routes its gradient to critic
    # parameters only (value_head_grads), so the shared trunk never sees
    # it while the aux ballast below keeps flowing.
    if not getattr(args, "stop_grad_value", False):
        total = total + agent.value_loss_coeff * value_loss

    if agent.critic.has_aux_heads:
        win_loss = F.binary_cross_entropy_with_logits(
            flat.win_logits_flat, flat.win_labels_flat
        )
        return_loss = F.smooth_l1_loss(
            flat.returns_pred_flat / RETURN_SCALE,
            flat.final_returns_labels_flat / RETURN_SCALE,
        )
        secret_loss = F.binary_cross_entropy_with_logits(
            flat.secret_logits_flat, flat.secret_labels_flat
        )
        points_pred = forward.points_pred_bt.view(-1, forward.points_pred_bt.size(-1))[
            minibatch.is_action_bt.view(-1)
        ]
        points_lbl = minibatch.points_bt.view(-1, minibatch.points_bt.size(-1))[
            minibatch.is_action_bt.view(-1)
        ]
        points_loss = F.smooth_l1_loss(
            points_pred / ppo_module.POINTS_SCALE, points_lbl / ppo_module.POINTS_SCALE
        )
        seen_loss = F.binary_cross_entropy_with_logits(
            flat.seen_trump_mask_logits_flat, flat.seen_trump_mask_labels_flat
        )
        unseen_loss = F.binary_cross_entropy_with_logits(
            flat.unseen_trump_higher_than_hand_logits_flat,
            flat.unseen_trump_higher_than_hand_labels_flat,
        )
        total = total + (
            agent.win_loss_coeff * win_loss
            + agent.return_loss_coeff * return_loss
            + agent.secret_loss_coeff * secret_loss
            + agent.points_loss_coeff * points_loss
            + agent.seen_trump_mask_loss_coeff * seen_loss
            + agent.unseen_trump_higher_than_hand_loss_coeff * unseen_loss
        )
    return total, stats


def oracle_loss_for_batch(agent, batch, kinds, minibatch):
    """Plain-MSE oracle value regression toward the MC return (+ oracle aux
    losses), mirroring the PPO oracle path without the value clip (no old
    values in a supervised phase). Returns None when inactive."""
    if agent.oracle_critic is None:
        return None
    if any(
        "oracle_state" not in agent.events[i] for s, e in batch for i in range(s, e + 1)
    ):
        return None
    oracle_seqs, returns_oracle_bt, _ = agent._build_oracle_minibatch(batch, kinds)
    values_bt, trunk_bt = agent.oracle_critic.forward_sequences_full(
        oracle_seqs, device=ppo_module.device
    )
    flat_idx = minibatch.is_action_bt.view(-1)
    loss = F.mse_loss(
        values_bt.reshape(-1)[flat_idx], returns_oracle_bt.reshape(-1)[flat_idx]
    )
    if agent.oracle_critic.has_aux_heads:
        membership, points = agent.oracle_critic.aux_losses(
            trunk_bt, oracle_seqs, minibatch.is_action_bt
        )
        loss = loss + (
            agent.oracle_membership_coeff * membership
            + agent.oracle_points_coeff * points
        )
    return loss


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def value_head_grads(agent, flat):
    """Gradient of the (coeff-scaled) value MSE w.r.t. CRITIC parameters
    only (§17.13 lever 1). Because grads w.r.t. critic params never route
    through encoder params, adding these by hand after the joint backward
    trains the value head/trunk normally while the SHARED trunk receives
    zero value-stream gradient — the surgical version of freezing the
    value path without giving up critic calibration."""
    scaled = agent.value_loss_coeff * F.mse_loss(flat.values_flat, flat.returns_flat)
    params = [p for p in agent.critic.parameters() if p.requires_grad]
    grads = torch.autograd.grad(scaled, params, retain_graph=True, allow_unused=True)
    return params, grads


def run_epoch(agent, episodes, args, train: bool, frozen=None):
    """One pass over ``episodes`` in buffer-sized chunks. Returns the
    row-weighted mean telemetry.

    ``frozen`` (§17.11 recomputed anchors): a frozen theta_k agent whose
    forward pass over the SAME minibatch tensors replaces the stored
    act-time anchor stashes. Target and student then see byte-identical
    replayed streams, so the anchor loss's zero point is exactly
    theta_k-as-replayed (init KL == 0, init gradient == 0 by
    construction) — eliminating the act-time-vs-replay zero-point
    corruption that §17.11 promoted to lead hypothesis for the
    lambda-independent EV floor."""
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    conv_counts: dict = {}
    steps = 0
    order = list(range(len(episodes)))
    if train:
        random.shuffle(order)
    for chunk_start in range(0, len(order), args.buffer_episodes):
        chunk = [
            episodes[i] for i in order[chunk_start : chunk_start + args.buffer_episodes]
        ]
        agent.reset_storage()
        store_episodes(agent, chunk, gap_floor=getattr(args, "gap_floor", 0.0))
        states, masks_t, kinds = agent._prepare_training_views()
        segments = agent._segments_from_events(kinds)
        seg_order = list(range(len(segments)))
        if train:
            random.shuffle(seg_order)
        for mb_start in range(0, len(seg_order), args.batch_segments):
            batch = [
                segments[i]
                for i in seg_order[mb_start : mb_start + args.batch_segments]
            ]
            minibatch = agent._build_minibatch_tensors(batch, states, masks_t, kinds)
            with torch.set_grad_enabled(train):
                forward = agent._forward_vectorized(
                    minibatch.states_seqs, minibatch.masks_bt
                )
                flat = agent._flatten_action_steps(minibatch, forward)
                if flat is None:
                    continue
                dchan = flat_channels(agent, batch, kinds)
                if frozen is not None:
                    with torch.no_grad():
                        f_fwd = frozen._forward_vectorized(
                            minibatch.states_seqs, minibatch.masks_bt
                        )
                        f_flat = agent._flatten_action_steps(minibatch, f_fwd)
                        # Masked logits: softmax is ~0 off-legal, so the
                        # kd_kl legal-set reconstruction (anchor > 0) holds.
                        dchan = (
                            dchan[0],
                            dchan[1],
                            F.softmax(f_flat.logits_flat, dim=-1),
                        )
                accumulate_conventions(
                    conv_counts,
                    convention_rows(agent, batch, kinds),
                    flat.logits_flat.detach(),
                )
                total, stats = distill_losses(
                    agent, minibatch, forward, flat, dchan, args
                )
                o_loss = (
                    oracle_loss_for_batch(agent, batch, kinds, minibatch)
                    if args.train_oracle
                    else None
                )
                if o_loss is not None:
                    total = total + agent.oracle_value_loss_coeff * o_loss
                    stats["oracle_loss"] = float(o_loss.detach())
            if train:
                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                if agent.oracle_optimizer is not None:
                    agent.oracle_optimizer.zero_grad()
                sg_value = getattr(args, "stop_grad_value", False)
                if sg_value:
                    v_params, v_grads = value_head_grads(agent, flat)
                total.backward()
                if sg_value:
                    for p, g in zip(v_params, v_grads):
                        if g is not None:
                            p.grad = g if p.grad is None else p.grad + g
                torch.nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), agent.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    agent.encoder.parameters(), agent.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    agent.critic.parameters(), agent.max_grad_norm
                )
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()
                if o_loss is not None and agent.oracle_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(
                        agent.oracle_critic.parameters(), agent.max_grad_norm
                    )
                    agent.oracle_optimizer.step()
                agent.optimizer_steps_total += 1
            steps += 1
            for key in (
                "override_ce",
                "override_kl",
                "override_omega",
                "endorsed_kl",
                "retention_kl",
                "value_mse",
                "oracle_loss",
            ):
                if key in stats:
                    w = {
                        "override_ce": stats["override_rows"],
                        "override_kl": stats["override_rows"],
                        "override_omega": stats["override_rows"],
                        "endorsed_kl": stats["endorsed_rows"],
                        "retention_kl": stats["retention_rows"],
                    }.get(key, 1)
                    if w:
                        totals[key] = totals.get(key, 0.0) + stats[key] * w
                        weights[key] = weights.get(key, 0.0) + w
            for key in ("override_rows", "endorsed_rows", "retention_rows"):
                totals[key] = totals.get(key, 0.0) + stats[key]
                weights[key] = 1.0
    agent.reset_storage()
    out = {k: totals[k] / max(weights.get(k, 1.0), 1e-9) for k in totals}
    out.update(convention_report(conv_counts))
    return out, steps


def fmt_stats(stats: dict) -> str:
    parts = []
    for k in (
        "override_ce",
        "override_kl",
        "override_omega",
        "endorsed_kl",
        "retention_kl",
        "value_mse",
        "oracle_loss",
    ):
        if k in stats:
            parts.append(f"{k} {stats[k]:.4f}")
    parts.append(
        "rows ov/en/rt {:.0f}/{:.0f}/{:.0f}".format(
            stats.get("override_rows", 0),
            stats.get("endorsed_rows", 0),
            stats.get("retention_rows", 0),
        )
    )
    for k in (
        "t0_def_trump_lead_rate",
        "def_trump_lead_rate",
        "partner_trump_lead_rate",
        "t0_called_suit_lead_rate",
        "called_suit_lead_rate",
    ):
        if k in stats:
            parts.append(f"{k} {stats[k]:.1f} (n={stats[k[:-5] + '_n']:.0f})")
    return "  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="theta_k init checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--buffer-episodes", type=int, default=512)
    ap.add_argument("--batch-segments", type=int, default=16)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    # §17.4 loss knobs (the pre-registered sweep grid)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-end", type=float, default=0.5)
    ap.add_argument("--lambda-ret", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.03)
    ap.add_argument("--omega-max", type=float, default=float(np.e))
    ap.add_argument("--kd-tau", type=float, default=1.0)
    ap.add_argument(
        "--recomputed-anchors",
        action="store_true",
        help="anchor targets from a frozen theta_k forward over the "
        "trainer's own replayed unroll instead of the stored act-time "
        "stashes (§17.11: init KL == 0 by construction)",
    )
    ap.add_argument(
        "--gap-floor",
        dest="gap_floor",
        type=float,
        default=0.0,
        help="demote override rows with top-2 gap below this to no-loss "
        "(§18 near-tie label pruning; 0 = keep all)",
    )
    ap.add_argument(
        "--stop-grad-value",
        dest="stop_grad_value",
        action="store_true",
        help="value MSE trains the critic only — zero value-stream "
        "gradient reaches the shared trunk; aux ballast unaffected "
        "(§17.13 iteration-4 lever 1)",
    )
    ap.add_argument(
        "--no-value-aux",
        dest="no_value_aux",
        action="store_true",
        help="drop value + critic-aux losses (§17.12 floor attribution; "
        "combine with --no-oracle for a pure policy-stream arm)",
    )
    ap.add_argument("--no-oracle", dest="train_oracle", action="store_false")
    ap.add_argument(
        "--probe-games",
        type=int,
        default=500,
        help="greedy_health_probe size at each epoch end (0 disables)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = load_agent(args.ckpt)
    # Supervised phase: the corpus is fixed, so PPO's staleness-driven LR
    # schedule is irrelevant — flat distill LR on both optimizer paths.
    agent.set_learning_rates(actor_lr=args.lr, critic_lr=args.lr)

    frozen = None
    if args.recomputed_anchors:
        frozen = load_agent(args.ckpt)
        for net in (frozen.encoder, frozen.actor, frozen.critic):
            for p in net.parameters():
                p.requires_grad_(False)

    episodes = load_shards(args.corpus_dir)
    train_eps, holdout = split_by_game(episodes, args.holdout_frac, args.seed)
    print(
        f"corpus: {len(train_eps)} train / {len(holdout)} holdout episodes "
        f"(game-level split)",
        flush=True,
    )

    log_path = os.path.join(args.out_dir, "distill_log.jsonl")
    log_f = open(log_path, "a")

    def log_row(row: dict):
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

    log_row({"kind": "config", **{k: str(v) for k, v in vars(args).items()}})

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats, steps = run_epoch(
            agent, train_eps, args, train=True, frozen=frozen
        )
        print(
            f"[epoch {epoch}] train ({steps} steps, "
            f"{(time.time() - t0) / 60:.1f} min): {fmt_stats(train_stats)}",
            flush=True,
        )
        log_row({"kind": "train", "epoch": epoch, **train_stats})
        if holdout:
            hold_stats, _ = run_epoch(agent, holdout, args, train=False, frozen=frozen)
            print(f"[epoch {epoch}] holdout: {fmt_stats(hold_stats)}", flush=True)
            log_row({"kind": "holdout", "epoch": epoch, **hold_stats})
        if args.probe_games:
            probe = greedy_health_probe(agent, n_games=args.probe_games, seed=0)
            print(
                f"[epoch {epoch}] probe: "
                + "  ".join(f"{k} {v:.1f}" for k, v in sorted(probe.items())),
                flush=True,
            )
            log_row({"kind": "probe", "epoch": epoch, **probe})
        ckpt_path = os.path.join(args.out_dir, f"distill_epoch{epoch}.pt")
        agent.save(ckpt_path)
        print(f"[epoch {epoch}] saved {ckpt_path}", flush=True)
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
