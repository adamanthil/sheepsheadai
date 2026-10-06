#!/usr/bin/env python3
"""Does the oracle critic need history, and does recurrence deliver it?

Offline probe for notebooks/Oracle_History_Probe_202610.md. The oracle sees
every hand, so the only thing a single full-information snapshot s is
missing is the PUBLIC HISTORY: who played which card, in which trick, in
which order (the bidding is already implied by s — every seat before the
picker passed, and the call/alone/under flags are context scalars). The
actors' policies condition on that history, so the sound target is the
history-state value U(h, s) (Baisero & Amato 2022); the production oracle
recovers h through a GRU over its event stream. This probe asks whether h
carries value information at all, and whether the GRU recovers it.

Three arms, trained from scratch on the SAME frozen-policy dataset
(``pretrain_oracle generate`` output), the same split, recipe, and
early-stopping rule as ``pretrain_oracle pretrain``:

  recurrent  the production OracleValueNetwork over each hero event stream
             (fresh zero memory per stream) — today's design.
  stateless  the same network on each decision row ALONE (T=1, zero memory):
             an estimate of V(s), no history at all.
  history    the candidate production layout: feed-forward, no recurrence,
             one token per deck card (32) plus the context token. Each card
             token carries its LOCATION — in seat X's hand, blind, bury
             (picker's), in the current trick, or played in an earlier
             trick — with the holding/playing seat, that seat's role, the
             trick index, order-in-trick, and an under flag. Every card
             appears exactly once, so the full public history and the full
             hidden state fit in 33 tokens (vs the production 51).

Read-out: the target G is the realized return, unbiased for U(h, s) under
the data policy, so for any arm  MSE = E[(v - U)^2] + E[Var(G | h, s)] and
the paired difference  dMSE(A, B) = E[(v_A - U)^2] - E[(v_B - U)^2]  is a
pure bias-squared difference (the irreducible return noise cancels). It is
reported overall and per stratum with an episode-clustered SE. Train-step
throughput per arm answers the cost side of the redesign question.

Usage:
  uv run python -m sheepshead.analysis.oracle_history_probe \\
      --dataset runs/202609_recall_rc/oracle/dataset.pt \\
      --reference-weights runs/202609_recall_rc/league/checkpoints/checkpoint_7700000.pt \\
      --seeds 42,43 --recurrent-seeds 42 \\
      --out runs/oracle_history_probe/bootstrap_policy.json

``--reference-weights`` takes a bare oracle state_dict (``pretrain_oracle``
output) or a full training checkpoint (its ``oracle_state_dict``). Arms
that share seeds are compared seed-for-seed; otherwise every seed of one
arm is compared against every seed of the other.

  # smoke test
  uv run python -m sheepshead.analysis.oracle_history_probe \\
      --dataset runs/202609_recall_rc/oracle/dataset.pt --max-episodes 400 \\
      --max-epochs 1 --out /tmp/ohp_smoke.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

from sheepshead.agent.encoder import PAD_CARD_ID, CardEmbeddingConfig
from sheepshead.agent.oracle import OracleCriticEncoder, OracleValueNetwork
from sheepshead.game import UNDER_CARD_ID
from sheepshead.training.pretrain_oracle import (
    STRATA_ORDER,
    _batches,
    _masked_mse,
    stratum_report,
)

ARMS = ("recurrent", "stateless", "history")
PAIRS = (
    ("stateless", "recurrent"),
    ("history", "recurrent"),
    ("history", "stateless"),
    ("history", "reference"),
    ("recurrent", "reference"),
    ("stateless", "reference"),
)
TOKENS = {"recurrent": 51, "stateless": 51, "history": 33}

# Card-location table: one row per deck card (id - 1), columns below.
N_CARDS = 32
LOC_NONE, LOC_HAND, LOC_BLIND, LOC_BURY, LOC_UNDER, LOC_TRICK, LOC_PLAYED = range(7)
COL_LOC, COL_SEAT, COL_TRICK, COL_ORDER, COL_UNDER = range(5)


# --------------------------------------------------------------------------- #
# Card locations, rebuilt from the hero's oracle event stream
# --------------------------------------------------------------------------- #
def build_card_locations(ep: dict) -> list[np.ndarray | None]:
    """Per event: a (32, 5) uint8 table of (location, seat_rel, trick+1,
    order+1, under) indexed by card id - 1 (None on observation events).

    All dicts in a hero stream share the hero's relative-seat frame. Each
    completed trick arrives as an observation event carrying that trick's
    cards; a trick's leader is the ``leader_rel`` the hero saw while acting
    in it (the hero plays every trick). Later sources override earlier ones
    — blind, bury, held-under, hands, then played — so a picked-up blind
    card sits in the picker's hand and a buried one in the bury. A called
    under leaves the picker's hand when declared, so until it is played it
    sits in its own location (held face-down by the picker); once played,
    the face-down trick card (id 33) resolves to the true ``under_card_id``. Asserts every
    card is located and 5 cards per completed trick."""
    obs, is_action = ep["obs"], ep["is_action"]
    leaders: dict[int, int] = {}
    for o, a in zip(obs, is_action):
        if a and int(o["play_started"]):
            leaders.setdefault(int(o["current_trick"]), int(o["leader_rel"]))

    def trick_cards(o, k: int, lead: int):
        for r, cid in enumerate(np.asarray(o["trick_card_ids"]).ravel(), start=1):
            cid = int(cid)
            if cid == UNDER_CARD_ID:
                cid = int(o["under_card_id"])
            if cid != PAD_CARD_ID:
                yield cid, r, k + 1, (((r - lead) % 5) + 1 if lead else 0)

    played: list[tuple[int, int, int, int]] = []
    out: list[np.ndarray | None] = []
    for o, a in zip(obs, is_action):
        k = int(o["current_trick"])
        if not a:
            out.append(None)
            played.extend(trick_cards(o, k, leaders.get(k, 0)))
            continue
        if int(o["play_started"]) and len(played) != 5 * k:
            raise ValueError(
                f"history rebuild: {len(played)} cards before trick {k} "
                "(expected 5 per completed trick)"
            )
        table = np.zeros((N_CARDS, 5), dtype=np.uint8)

        def put(cid: int, loc: int, seat: int, trick: int = 0, order: int = 0):
            if cid != PAD_CARD_ID:
                table[cid - 1, :4] = (loc, seat, trick, order)

        for cid in np.asarray(o["blind_ids"]).ravel():
            put(int(cid), LOC_BLIND, 0)
        for cid in np.asarray(o["bury_ids"]).ravel():
            put(int(cid), LOC_BURY, int(o["picker_rel"]))
        under = int(o["under_card_id"])
        if under not in (PAD_CARD_ID, UNDER_CARD_ID):
            put(under, LOC_UNDER, int(o["picker_rel"]))
            table[under - 1, COL_UNDER] = 1
        for cid in np.asarray(o["hand_ids"]).ravel():
            put(int(cid), LOC_HAND, 1)
        for i, row in enumerate(np.asarray(o["opp_hand_ids"]).reshape(4, 8)):
            for cid in row:
                put(int(cid), LOC_HAND, i + 2)
        if int(o["play_started"]):
            for cid, seat, trick, order in trick_cards(o, k, int(o["leader_rel"])):
                put(cid, LOC_TRICK, seat, trick, order)
        for cid, seat, trick, order in played:
            put(cid, LOC_PLAYED, seat, trick, order)
        missing = int((table[:, COL_LOC] == LOC_NONE).sum())
        if missing:
            raise ValueError(f"card-location rebuild: {missing} cards unlocated")
        out.append(table)
    return out


class CardLocationEncoder(OracleCriticEncoder):
    """Feed-forward full-information encoder: [context] + 32 card tokens.

    Reuses the production oracle's card embedding (informed init), seat and
    role embeddings, context MLP (header + points + secret partner + called
    and under cards), and transformer; drops every recurrent / per-bag
    module. Each card token = MLP(card, location, seat, role, trick, order,
    under). Role is that seat's picker / secret-partner bits, as on the
    production opponent-hand tokens."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_card, d_token = self.d_card_dim, self.d_token_dim
        for name in (
            "card_type",
            "memory_in_proj",
            "memory_gru",
            "token_mlp_hand",
            "token_mlp_trick",
            "token_mlp_simple",
            "token_mlp_opp",
        ):
            delattr(self, name)
        self.loc_emb = nn.Embedding(7, 8)
        self.trick_emb = nn.Embedding(7, 4)  # 0 = none, 1..6 = trick
        self.order_emb = nn.Embedding(6, 4)  # 0 = none, 1..5 = lead..last
        self.under_emb = nn.Embedding(2, 4)
        self.token_mlp_card = nn.Sequential(
            nn.Linear(d_card + 8 + 4 + 4 + 4 + 4 + 4, d_token), nn.SiLU()
        )

    def _context_token(self, batch: List[Dict[str, Any]], dev) -> torch.Tensor:
        """OracleCriticEncoder.encode_batch's context token, verbatim."""
        fields = [
            "partner_mode",
            "is_leaster",
            "play_started",
            "current_trick",
            "alone_called",
            "called_under",
            "picker_rel",
            "partner_rel",
            "leader_rel",
            "picker_position",
        ]
        header = torch.cat([self._stack_scalar(batch, k) for k in fields], dim=1).to(
            dev
        )
        norm = torch.tensor(
            [1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0], device=dev
        )
        points = self._stack_uint8(batch, "points_taken_rel", 5).float().to(dev) / 120.0
        secret = torch.as_tensor(
            [float(s["secret_partner_rel"]) / 5.0 for s in batch], device=dev
        ).view(-1, 1)
        called = torch.as_tensor(
            [int(s["called_card_id"]) for s in batch], dtype=torch.long, device=dev
        )
        under = torch.as_tensor(
            [int(s["under_card_id"]) for s in batch], dtype=torch.long, device=dev
        )
        return self.context_mlp(
            torch.cat(
                [header / norm, points, secret, self.card(called), self.card(under)],
                dim=1,
            )
        )

    def encode_rows(
        self, batch: List[Dict[str, Any]], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(all_tokens (B, 33, d_token), all_mask (B, 33)) after reasoning."""
        table = torch.as_tensor(
            np.stack([s["cardloc"] for s in batch]), dtype=torch.long
        )
        if device is not None:
            table = table.to(device)
        dev = table.device
        B = table.size(0)
        loc, seat, trick, order, under = table.unbind(-1)
        picker = torch.as_tensor(
            [int(s["picker_rel"]) for s in batch], device=dev
        ).view(B, 1)
        secret = torch.as_tensor(
            [int(s["secret_partner_rel"]) for s in batch], device=dev
        ).view(B, 1)
        seated = seat.ne(0)
        role = (seated & seat.eq(picker)).long() + (seated & seat.eq(secret)).long() * 2
        ids = torch.arange(1, N_CARDS + 1, device=dev).view(1, N_CARDS).expand(B, -1)
        card_tok = self.token_mlp_card(
            torch.cat(
                [
                    self.card(ids),
                    self.loc_emb(loc),
                    self.seat(seat),
                    self.role(role),
                    self.trick_emb(trick),
                    self.order_emb(order),
                    self.under_emb(under),
                ],
                dim=-1,
            )
        )
        tokens = torch.cat([self._context_token(batch, dev).unsqueeze(1), card_tok], 1)
        mask = torch.cat(
            [torch.ones((B, 1), dtype=torch.bool, device=dev), loc.ne(LOC_NONE)], 1
        )
        return self.card_reasoner(tokens, mask), mask


class CardLocationOracleValueNetwork(OracleValueNetwork):
    """OracleValueNetwork's readout / trunk / heads over CardLocationEncoder."""

    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__(d_model=d_model, **kwargs)
        self.encoder = CardLocationEncoder(
            card_config=CardEmbeddingConfig(), d_model=d_model
        )

    def forward_rows(
        self, rows: List[Dict[str, Any]], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Values (B, 1) and trunk features (B, 1, d_model), one per row."""
        assert isinstance(self.encoder, CardLocationEncoder)
        tokens, mask = self.encoder.encode_rows(rows, device=device)
        trunk = self.value_trunk(self._readout(tokens, mask))
        return self.value_head(trunk).view(-1, 1), trunk.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Arm plumbing: every arm maps a list of episodes to (B, T) values/targets
# --------------------------------------------------------------------------- #
Forward = Callable[[OracleValueNetwork, list, torch.device], tuple]


def _recurrent_forward(net, eps, dev):
    seqs = [ep["obs"] for ep in eps]
    vals, trunk = net.forward_sequences_full(seqs, device=dev)
    B, T = vals.shape
    target = torch.zeros((B, T), device=vals.device)
    mask = torch.zeros((B, T), dtype=torch.bool, device=vals.device)
    for b, ep in enumerate(eps):
        for t, (is_a, g) in enumerate(zip(ep["is_action"], ep["g"])):
            if is_a:
                target[b, t] = g
                mask[b, t] = True
    return seqs, vals, trunk, target, mask


def _flat_forward(card_locations: bool):
    def fwd(net, eps, dev):
        rows, gs = [], []
        for ep in eps:
            for t, is_a in enumerate(ep["is_action"]):
                if is_a:
                    o = ep["obs"][t]
                    rows.append(
                        {**o, "cardloc": ep["cardloc"][t]} if card_locations else o
                    )
                    gs.append(ep["g"][t])
        seqs = [[r] for r in rows]
        if card_locations:
            vals, trunk = net.forward_rows(rows, device=dev)
        else:
            vals, trunk = net.forward_sequences_full(seqs, device=dev)
        target = torch.as_tensor(gs, dtype=torch.float32, device=vals.device).view(
            -1, 1
        )
        mask = torch.ones_like(target, dtype=torch.bool)
        return seqs, vals, trunk, target, mask

    return fwd


def make_arm(name: str) -> tuple[OracleValueNetwork, Forward]:
    if name == "recurrent":
        return OracleValueNetwork(use_aux_heads=True), _recurrent_forward
    if name == "stateless":
        return OracleValueNetwork(use_aux_heads=True), _flat_forward(False)
    if name == "history":
        return CardLocationOracleValueNetwork(use_aux_heads=True), _flat_forward(True)
    raise ValueError(name)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def predict(net, fwd, eps, dev, batch_size) -> np.ndarray:
    """Test-row predictions in canonical order (episode, then event)."""
    net.eval()
    preds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(eps), batch_size):
            chunk = eps[i : i + batch_size]
            _, vals, _, _, mask = fwd(net, chunk, dev)
            preds.extend(vals[mask].tolist())
    return np.asarray(preds)


def train_arm(name, seed, train, val, args, dev) -> tuple:
    """pretrain_oracle's recipe: value MSE + aux heads, Adam, clip 0.5,
    early stop on validation value-MSE. Batches are EPISODES for every arm,
    so each gradient step sees the same decision rows across arms."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    net, fwd = make_arm(name)
    net = net.to(dev)
    opt = torch.optim.Adam(net.param_groups(args.lr))
    rng = random.Random(seed)
    best_val, best_state, bad, curve = float("inf"), None, 0, []
    step_time, step_rows, steps = 0.0, 0, 0
    for epoch in range(args.max_epochs):
        net.train()
        t_epoch = time.time()
        for batch_idx in _batches(list(range(len(train))), args.batch_size, rng):
            eps = [train[i] for i in batch_idx]
            _sync(dev)
            t0 = time.perf_counter()
            seqs, vals, trunk, target, mask = fwd(net, eps, dev)
            m_loss, p_loss = net.aux_losses(trunk, seqs, mask)
            loss = (
                _masked_mse(vals, target, mask)
                + args.aux_partner_coeff * m_loss
                + args.aux_points_coeff * p_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            _sync(dev)
            step_time += time.perf_counter() - t0
            step_rows += int(mask.sum())
            steps += 1
        se, n = 0.0, 0
        net.eval()
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                _, vals, _, target, mask = fwd(net, val[i : i + args.batch_size], dev)
                se += float(((vals - target) ** 2)[mask].sum())
                n += int(mask.sum())
        val_mse = se / max(n, 1)
        curve.append(val_mse)
        print(
            f"  [{name} s{seed}] epoch {epoch + 1}: val MSE {val_mse:.5f} "
            f"({time.time() - t_epoch:.0f}s)",
            flush=True,
        )
        if val_mse < best_val - 1e-6:
            best_val, bad = val_mse, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad > args.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    timing = {
        "train_steps": steps,
        "sec_per_step": step_time / max(steps, 1),
        "decision_rows_per_sec": step_rows / max(step_time, 1e-9),
    }
    return net, fwd, {"best_val_mse": best_val, "val_curve": curve, "timing": timing}


# --------------------------------------------------------------------------- #
# Read-out
# --------------------------------------------------------------------------- #
def paired_dmse(va, vb, g, ep_ids) -> dict:
    """Mean of per-row (va-g)^2 - (vb-g)^2 with an episode-clustered SE."""
    d = (va - g) ** 2 - (vb - g) ** 2
    n = len(d)
    mean = float(d.mean())
    resid = d - mean
    cluster = np.bincount(ep_ids, weights=resid)
    se = float(np.sqrt((cluster**2).sum()) / n)
    return {"n": n, "dmse": mean, "se": se, "z": mean / se if se > 0 else 0.0}


def arm_report(v, g, strata) -> dict:
    rows = [{"g": gi, "v": vi, "strata": s} for gi, vi, s in zip(g, v, strata)]
    rep = stratum_report(rows)
    for s, cell in rep.items():
        idx = [i for i, ss in enumerate(strata) if s in ss]
        cell["mse"] = float(np.mean((v[idx] - g[idx]) ** 2))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="42", help="comma list; one run per arm")
    ap.add_argument(
        "--recurrent-seeds",
        default=None,
        help="comma list overriding --seeds for the (slow) recurrent arm",
    )
    ap.add_argument(
        "--reference-weights",
        default=None,
        help="untrained-here production oracle: a bare oracle state_dict or a "
        "full training checkpoint (reads its oracle_state_dict)",
    )
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--aux-partner-coeff", type=float, default=0.1)
    ap.add_argument("--aux-points-coeff", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sheepshead.agent.ppo import device as default_device

    dev = torch.device(args.device) if args.device else default_device
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    arm_seeds = {name: seeds for name in arms}
    if args.recurrent_seeds is not None and "recurrent" in arm_seeds:
        arm_seeds["recurrent"] = [int(s) for s in args.recurrent_seeds.split(",") if s]

    data = torch.load(args.dataset, weights_only=False)
    episodes = data["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    for ep in episodes:
        ep["cardloc"] = build_card_locations(ep)
    # pretrain_oracle's split, so the reference oracle's test set is ours.
    test = [ep for i, ep in enumerate(episodes) if i % 10 == 0]
    val = [ep for i, ep in enumerate(episodes) if i % 10 == 1]
    train = [ep for i, ep in enumerate(episodes) if i % 10 >= 2]
    g = np.asarray(
        [ep["g"][t] for ep in test for t, a in enumerate(ep["is_action"]) if a]
    )
    strata = [
        ep["strata"][t] for ep in test for t, a in enumerate(ep["is_action"]) if a
    ]
    ep_ids = np.asarray(
        [b for b, ep in enumerate(test) for a in ep["is_action"] if a], dtype=np.int64
    )
    print(
        f"{len(episodes)} episodes (gamma={data.get('gamma')}, ckpt={data.get('ckpt')}): "
        f"train {len(train)}, val {len(val)}, test {len(test)} ({len(g)} rows); "
        f"device={dev}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    keys: dict[str, list[str]] = {}
    if args.reference_weights:
        blob = torch.load(args.reference_weights, map_location=dev, weights_only=False)
        if "oracle_state_dict" in blob:
            if blob["oracle_state_dict"] is None:
                raise ValueError(f"{args.reference_weights} carries no oracle")
            state = blob["oracle_state_dict"]
            aux = bool(blob.get("oracle_aux_heads", False))
        else:
            state = blob
            aux = any(k.startswith("team_membership.") for k in state)
        ref = OracleValueNetwork(use_aux_heads=aux).to(dev)
        ref.load_state_dict(state)
        keys["reference"] = ["reference"]
        preds["reference"] = predict(
            ref, _recurrent_forward, test, dev, args.batch_size
        )
        results["reference"] = {
            "weights": args.reference_weights,
            "params": sum(p.numel() for p in ref.parameters()),
            "test": arm_report(preds["reference"], g, strata),
        }

    for name in arms:
        keys[name] = []
        for seed in arm_seeds[name]:
            key = f"{name}_s{seed}"
            keys[name].append(key)
            net, fwd, info = train_arm(name, seed, train, val, args, dev)
            preds[key] = predict(net, fwd, test, dev, args.batch_size)
            info["test"] = arm_report(preds[key], g, strata)
            info["params"] = sum(p.numel() for p in net.parameters())
            info["tokens"] = TOKENS[name]
            results[key] = info
            if args.save_weights:
                torch.save(net.state_dict(), out_path.with_suffix(f".{key}.pt"))

    comparisons: dict[str, dict] = {}
    for a, b in PAIRS:
        if a not in keys or b not in keys:
            continue
        shared = [k for k in arm_seeds.get(a, []) if k in arm_seeds.get(b, [])]
        if shared:
            pairs = [(f"{a}_s{s}", f"{b}_s{s}") for s in shared]
        else:
            pairs = [(ka, kb) for ka in keys[a] for kb in keys[b]]
        for ka, kb in pairs:
            cell = {"all": paired_dmse(preds[ka], preds[kb], g, ep_ids)}
            for s in STRATA_ORDER[1:]:
                idx = np.asarray([i for i, ss in enumerate(strata) if s in ss])
                if len(idx) >= 20:
                    cell[s] = paired_dmse(
                        preds[ka][idx], preds[kb][idx], g[idx], ep_ids[idx]
                    )
            comparisons[f"{ka} - {kb}"] = cell

    print(f"\n{'arm':<18}{'test EV':>9}{'test MSE':>10}{'rows/s':>9}{'params':>10}")
    for key, info in results.items():
        t = info.get("timing", {})
        print(
            f"{key:<18}{info['test']['all']['ev']:>9.4f}{info['test']['all']['mse']:>10.5f}"
            f"{t.get('decision_rows_per_sec', float('nan')):>9.0f}"
            f"{info.get('params', 0):>10}"
        )
    print("\ndMSE = MSE(A) - MSE(B) = bias^2(A) - bias^2(B); negative favors A")
    for label, cell in comparisons.items():
        print(f"  {label}")
        for s, c in cell.items():
            print(
                f"    {s:<30} n={c['n']:>6}  dMSE {c['dmse']:+.5f} "
                f"(SE {c['se']:.5f}, z {c['z']:+.1f})"
            )

    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "dataset": args.dataset,
                    "data_ckpt": data.get("ckpt"),
                    "gamma": data.get("gamma"),
                    "episodes": len(episodes),
                    "test_rows": len(g),
                    "seeds": arm_seeds,
                    "reference": args.reference_weights,
                    "device": str(dev),
                    "recipe": {
                        k: getattr(args, k)
                        for k in (
                            "lr",
                            "batch_size",
                            "max_epochs",
                            "patience",
                            "aux_partner_coeff",
                            "aux_points_coeff",
                        )
                    },
                },
                "arms": results,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
