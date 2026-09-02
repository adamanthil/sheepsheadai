"""Pooled search-advantage model for search-Q policy iteration
(CE_Teacher_Design §20.2, stages 1, 1b and 2).

The problem this solves: a committee search gives every searched node a
noisy scalar Q per action (replicate SE ~0.015 Q at 1024/1 x R=3) while
the effects that matter — the conventions — are ~0.01 Q per opportunity
(§20.1). No per-node label can see them; pooling across similar states
can. This module pools by REGRESSION: fit a small advantage head on the
policy's own frozen features, so "similar" means similar for the decision
the policy makes and the pooling structure is discovered rather than
hand-labeled (a Fay-Herriot small-area estimator — Fay & Herriot 1979 —
with the covariates supplied by theta_k's encoder).

Stage 1   ``fit_advantage_model``: heteroscedastic weighted least squares
          of the centered pooled Q (q̄_a − mean over the legal set) on the
          frozen features, per-row weight 1 / noise_var, held-out early
          stopping. Capacity rungs (``AdvantageModel``): ``pointer`` (a
          fresh pointer scorer over the frozen adapter output), ``adapter``
          (fresh adapter MLP + pointer), ``trunk`` (encoder unfrozen).
Stage 1b  ``estimate_residual_variance`` + ``blend_advantages``: the
          Fay-Herriot combination. sigma_u^2 = E[r^2] − E[noise_var] on
          held-out rows; gamma_n = sigma_u^2 / (sigma_u^2 + noise_var_n);
          a_hat = gamma * a_obs + (1 − gamma) * a_model; posterior variance
          gamma * noise_var (rows with Q) or sigma_u^2 (rows without).
Stage 2   ``build_tilt_target``: t(a) ∝ p_theta_k(a) * exp(clip(a_hat /
          (kappa * sqrt(v_post)))) — the mirror-descent / advantage-
          weighted step (Vieillard et al. 2020; Peng 2019; Nair 2020;
          Wang 2020) at a temperature set by the posterior SE, so a one-SE
          edge is one nat and a within-noise row stays at the prior.

Everything here is training-time only: the advantage model never ships,
it manufactures the CE targets the projection stage trains the policy on.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sheepshead.agent import ppo as ppo_module
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.corpus_rows import (
    EncodedRows,
    encode_rows,
    iter_row_batches,
    sorted_valid_to_dense,
    store_corpus_episodes,
)

CAPACITIES = ("pointer", "adapter", "trunk")
POINTER_HIDDEN = 64  # matches the actor's pointer scorer
# Heteroscedastic head: log sigma_u^2 init (~ the global residual variance
# measured on corpus q) and the clamp that keeps the NLL well-conditioned.
LOGVAR_INIT = 1e-3
LOGVAR_MIN, LOGVAR_MAX = math.log(1e-7), math.log(1.0)

# Corpus rows the §20 recipe can TARGET: searched play rows, whatever the
# committee concluded there (the endorsed/override split dissolves).
TARGETABLE_SETS = ("override", "endorsed")
# ... and the subset that carries usable pooled Q for Stage 1.
Q_SOURCES = ("committee", "recovered")


# --------------------------------------------------------------------------- #
# Row evidence
# --------------------------------------------------------------------------- #
@dataclass
class RowEvidence:
    """What one corpus row contributes, in dense action_size vectors."""

    targetable: bool
    has_q: bool
    advantage: list[float]  # q̄ centered over the legal set; 0 elsewhere
    label_mask: list[bool]  # legal actions carrying Q (all False if !has_q)
    noise_var: float
    prior: list[float]  # theta_k's act-time policy (anchor_probs), dense
    node_class: str


def row_evidence(source_event: dict, action_size: int) -> RowEvidence | None:
    """Evidence of a corpus action row, or None for rows the recipe never
    targets (bidding heads, leaster play, forced and unsearched rows)."""
    if source_event.get("distill_set") not in TARGETABLE_SETS:
        return None
    valid = source_event["valid_actions"]
    anchor = source_event.get("anchor_probs")
    if anchor is None:
        raise ValueError(
            "targetable row without anchor_probs — run recover_search_q on "
            "schema-1 corpora first (it stores the replayed prior)"
        )
    prior = sorted_valid_to_dense(anchor, valid, action_size)
    has_q = (
        source_event.get("search_stats_source") in Q_SOURCES
        and source_event.get("search_q") is not None
    )
    if has_q:
        q = np.asarray(source_event["search_q"], dtype=np.float64)
        centered = q - q.mean()
        advantage = sorted_valid_to_dense(centered, valid, action_size)
        label_mask = [False] * action_size
        for a in valid:
            label_mask[a - 1] = True
        noise_var = float(source_event["search_noise_var"])
    else:
        advantage = [0.0] * action_size
        label_mask = [False] * action_size
        noise_var = float("nan")
    return RowEvidence(
        targetable=True,
        has_q=has_q,
        advantage=advantage,
        label_mask=label_mask,
        noise_var=noise_var,
        prior=prior,
        node_class=str(source_event.get("node_class", "")),
    )


# --------------------------------------------------------------------------- #
# Row table (cached frozen encodings + evidence, in corpus order)
# --------------------------------------------------------------------------- #
@dataclass
class RowTable:
    """Targetable rows of a corpus with their frozen encodings, evidence and
    a reference back to the source event (shard, episode, event) so Stage 2
    can write targets onto the corpus."""

    features: torch.Tensor  # (R, d_model)
    hand_tokens: torch.Tensor  # (R, 8, d_token)
    hand_ids: torch.Tensor  # (R, 8)
    masks: torch.Tensor  # (R, A) legal
    advantage: torch.Tensor  # (R, A)
    label_mask: torch.Tensor  # (R, A) bool
    noise_var: torch.Tensor  # (R,) nan where !has_q
    prior: torch.Tensor  # (R, A)
    has_q: torch.Tensor  # (R,) bool
    node_class: list[str]
    refs: list[tuple[int, int, int]]  # (shard_idx, episode_idx, event_idx)
    game: torch.Tensor  # (R,) corpus game index (for game-level splits)

    def __len__(self) -> int:
        return int(self.features.size(0))

    def subset(self, idx: torch.Tensor) -> "RowTable":
        idx_list = idx.tolist()
        return RowTable(
            features=self.features[idx],
            hand_tokens=self.hand_tokens[idx],
            hand_ids=self.hand_ids[idx],
            masks=self.masks[idx],
            advantage=self.advantage[idx],
            label_mask=self.label_mask[idx],
            noise_var=self.noise_var[idx],
            prior=self.prior[idx],
            has_q=self.has_q[idx],
            node_class=[self.node_class[i] for i in idx_list],
            refs=[self.refs[i] for i in idx_list],
            game=self.game[idx],
        )

    def encoded(self, idx: torch.Tensor) -> EncodedRows:
        return EncodedRows(
            features=self.features[idx],
            hand_tokens=self.hand_tokens[idx],
            hand_ids=self.hand_ids[idx],
            masks=self.masks[idx],
        )

    @staticmethod
    def concat(tables: list["RowTable"]) -> "RowTable":
        tables = [t for t in tables if len(t)]
        if not tables:
            raise ValueError("no targetable rows in corpus")
        cat = lambda name: torch.cat([getattr(t, name) for t in tables], dim=0)  # noqa: E731
        return RowTable(
            features=cat("features"),
            hand_tokens=cat("hand_tokens"),
            hand_ids=cat("hand_ids"),
            masks=cat("masks"),
            advantage=cat("advantage"),
            label_mask=cat("label_mask"),
            noise_var=cat("noise_var"),
            prior=cat("prior"),
            has_q=cat("has_q"),
            node_class=[c for t in tables for c in t.node_class],
            refs=[r for t in tables for r in t.refs],
            game=cat("game"),
        )

    def save(self, path: str) -> None:
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path: str) -> "RowTable":
        return RowTable(**torch.load(path, map_location="cpu", weights_only=False))


def build_row_table(
    agent: PPOAgent,
    episodes: list,
    *,
    shard_idx: int,
    game_indices: list[int],
    buffer_episodes: int = 250,
    batch_segments: int = 32,
) -> RowTable:
    """Encode ``episodes`` through the FROZEN agent and collect every
    targetable row. ``game_indices[i]`` is episode i's corpus game."""
    pieces: list[RowTable] = []
    action_size = agent.action_size
    for start in range(0, len(episodes), buffer_episodes):
        chunk = episodes[start : start + buffer_episodes]
        chunk_games = game_indices[start : start + buffer_episodes]
        agent.reset_storage()
        source_map = store_corpus_episodes(agent, chunk)
        for rows in iter_row_batches(agent, batch_segments=batch_segments):
            with torch.no_grad():
                enc = encode_rows(agent, rows)
            keep, evidence, refs, games = [], [], [], []
            for r, ev_idx in enumerate(rows.event_indices):
                ref = source_map[ev_idx]
                assert ref is not None
                ep_idx, ev_pos = ref
                ev = row_evidence(chunk[ep_idx][ev_pos], action_size)
                if ev is None:
                    continue
                keep.append(r)
                evidence.append(ev)
                refs.append((shard_idx, start + ep_idx, ev_pos))
                games.append(chunk_games[ep_idx])
            if not keep:
                continue
            idx = torch.tensor(keep, dtype=torch.long, device=enc.features.device)
            pieces.append(
                RowTable(
                    features=enc.features[idx].cpu(),
                    hand_tokens=enc.hand_tokens[idx].cpu(),
                    hand_ids=enc.hand_ids[idx].cpu(),
                    masks=enc.masks[idx].cpu(),
                    advantage=torch.tensor([e.advantage for e in evidence]),
                    label_mask=torch.tensor([e.label_mask for e in evidence]),
                    noise_var=torch.tensor([e.noise_var for e in evidence]),
                    prior=torch.tensor([e.prior for e in evidence]),
                    has_q=torch.tensor([e.has_q for e in evidence]),
                    node_class=[e.node_class for e in evidence],
                    refs=refs,
                    game=torch.tensor(games, dtype=torch.long),
                )
            )
    agent.reset_storage()
    return RowTable.concat(pieces)


def split_rows_by_game(
    table: RowTable, holdout_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(train_idx, holdout_idx): a game-level split — five seats of one deal
    share an outcome, so rows of one game never straddle the cut."""
    games = sorted(set(table.game.tolist()))
    random.Random(seed).shuffle(games)
    n_hold = int(len(games) * holdout_frac)
    hold = set(games[:n_hold])
    is_hold = torch.tensor([g in hold for g in table.game.tolist()])
    return torch.nonzero(~is_hold).squeeze(1), torch.nonzero(is_hold).squeeze(1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AdvantageModel(nn.Module):
    """A twin of the policy's play pointer that predicts advantages.

    Reads exactly what the pointer actor reads — the shared readout through
    the actor adapter and the post-reasoning hand tokens — and scores each
    hand slot with v'·tanh(W'_g h + W'_t token). Slot scores are scattered
    onto action ids with the actor's own card-id maps, so the output is a
    dense ``(R, action_size)`` advantage aligned with policy logits.

    Capacity rungs:
      pointer  fresh pointer scorer; adapter and encoder frozen (~20k params)
      adapter  fresh adapter MLP + pointer; encoder frozen (~150k params)
      trunk    a trainable COPY of the encoder + fresh adapter + pointer
    Only ``trunk`` re-encodes rows during training; the frozen rungs train
    on a cached ``RowTable``.

    The policy's own centered log-probability over the legal set enters as
    a covariate with one learned scale (``prior_scale``, Q per nat): theta_k
    already orders actions in rough agreement with the committee (its top
    card is the committee's ~58% of the time on corpus q), so the head
    learns the RESIDUAL the prior does not explain instead of relearning
    the prior from scratch — the natural Fay-Herriot covariate.

    ``heteroscedastic`` (§20.8): a second output, the per-ROW log residual
    variance log sigma_u^2(s), trained by Gaussian negative log-likelihood
    with the known measurement noise added (Kendall & Gal 2017). Each row is
    then standardized by its own learned scale in the loss — cells whose
    true advantages vary a lot no longer drown cells whose whole signal is
    small, with no cell taxonomy — and the learned variance is the per-node
    Fay-Herriot sigma_u^2 the blend uses.
    """

    def __init__(self, agent: PPOAgent, capacity: str, heteroscedastic: bool = False):
        super().__init__()
        if capacity not in CAPACITIES:
            raise ValueError(f"capacity must be one of {CAPACITIES}, got {capacity!r}")
        self.capacity = capacity
        actor = agent.actor
        d_model = actor._d_model
        d_token = actor._d_token
        if capacity == "pointer":
            self.adapter = copy.deepcopy(actor.actor_adapter)
            for p in self.adapter.parameters():
                p.requires_grad_(False)
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
            )
        self.encoder = copy.deepcopy(agent.encoder) if capacity == "trunk" else None
        self.pointer_Wg = nn.Linear(d_model, POINTER_HIDDEN)
        self.pointer_Wt = nn.Linear(d_token, POINTER_HIDDEN)
        self.pointer_v = nn.Linear(POINTER_HIDDEN, 1, bias=False)
        self.prior_scale = nn.Parameter(torch.tensor(0.01))
        self.heteroscedastic = bool(heteroscedastic)
        if self.heteroscedastic:
            self.logvar_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.constant_(self.logvar_head.bias, math.log(LOGVAR_INIT))
        self.action_size = int(agent.action_size)
        for name in ("play", "under", "bury"):
            self.register_buffer(
                f"cid_to_{name}",
                getattr(actor, f"_map_cid_to_{name}_action_index").clone(),
                persistent=True,
            )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self, enc: EncodedRows, prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Dense ``(R, action_size)`` advantage. ``prior`` (dense theta_k
        probabilities, zeros off-legal) adds the scaled centered log-prior
        covariate; omit it for the bare head (the scatter-alignment test)."""
        return self.forward_with_variance(enc, prior)[0]

    def forward_with_variance(
        self, enc: EncodedRows, prior: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(advantage (R, A), sigma_u^2 (R,) or None for a homoscedastic
        model)."""
        h = self.adapter(enc.features)
        g = self.pointer_Wg(h).unsqueeze(1)  # (R, 1, hidden)
        t = self.pointer_Wt(enc.hand_tokens)  # (R, 8, hidden)
        slot = self.pointer_v(torch.tanh(g + t)).squeeze(-1)  # (R, 8)
        R = slot.size(0)
        wide = slot.new_zeros((R, self.action_size + 1))
        cids = enc.hand_ids.long()
        for name in ("bury", "under", "play"):
            dest = getattr(self, f"cid_to_{name}")[cids]
            dest = torch.where(
                dest.ge(0), dest, torch.full_like(dest, self.action_size)
            )
            wide = wide.scatter(1, dest, slot)
        out = wide[:, : self.action_size]
        if prior is not None:
            out = out + self.prior_scale * centered_log_prior(prior, enc.masks.bool())
        if not self.heteroscedastic:
            return out, None
        logvar = self.logvar_head(h).squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return out, torch.exp(logvar)


def centered_log_prior(prior: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """log p centered over the legal set (0 off-legal) — the prior covariate."""
    logp = torch.log(prior.clamp(min=1e-12))
    logp = torch.where(legal, logp, torch.zeros_like(logp))
    n = legal.sum(dim=1, keepdim=True).clamp(min=1)
    centered = logp - logp.sum(dim=1, keepdim=True) / n
    return torch.where(legal, centered, torch.zeros_like(centered))


# --------------------------------------------------------------------------- #
# Stage 1: fit
# --------------------------------------------------------------------------- #
def weighted_rows_mse(
    pred: torch.Tensor,
    advantage: torch.Tensor,
    label_mask: torch.Tensor,
    noise_var: torch.Tensor,
    var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss, per_row_mse): per row the mean squared error over labeled
    actions; the loss is the 1/noise_var-weighted mean over rows (the
    heteroscedastic least-squares objective)."""
    sq = (pred - advantage) ** 2 * label_mask
    n_lab = label_mask.sum(dim=1).clamp(min=1)
    per_row = sq.sum(dim=1) / n_lab
    w = 1.0 / (noise_var + var_floor)
    return (w * per_row).sum() / w.sum(), per_row


def gaussian_nll_rows(
    pred: torch.Tensor,
    advantage: torch.Tensor,
    label_mask: torch.Tensor,
    noise_var: torch.Tensor,
    sigma_u2: torch.Tensor,
    var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss, per_row_mse): Gaussian negative log-likelihood of the labeled
    advantages under total variance noise_var + sigma_u^2(s) (both per row),
    averaged over labeled actions per row and then over rows. Standardizes
    each row by its own learned scale — the taxonomy-free replacement for
    the per-cell Fay-Herriot weights."""
    total = (noise_var + sigma_u2 + var_floor).unsqueeze(1)
    sq = (pred - advantage) ** 2
    nll = 0.5 * (sq / total + torch.log(total))
    n_lab = label_mask.sum(dim=1).clamp(min=1)
    per_row_nll = (nll * label_mask).sum(dim=1) / n_lab
    per_row_mse = (sq * label_mask).sum(dim=1) / n_lab
    return per_row_nll.mean(), per_row_mse


@dataclass
class FitReport:
    capacity: str
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_holdout_mse: float = float("inf")
    holdout_noise_floor: float = float("nan")
    per_class: dict = field(default_factory=dict)
    sigma_u2: float = float("nan")
    # Per-class residual variances (§20.6, shrunk toward sigma_u2 by row
    # count); filled by ``class_residual_variances`` after the fit.
    sigma_u2_by_class: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def _batches(idx: torch.Tensor, batch_rows: int, shuffle: bool, rng: random.Random):
    order = idx.tolist()
    if shuffle:
        rng.shuffle(order)
    for s in range(0, len(order), batch_rows):
        yield torch.tensor(order[s : s + batch_rows], dtype=torch.long)


@dataclass
class HoldoutEval:
    """``evaluate_rows`` output: per-class diagnostics (``"__all__"`` = the
    pooled row) plus the unweighted means the Fay-Herriot estimate needs."""

    per_class: dict
    residual_sq_mean: float
    noise_var_mean: float

    @property
    def pooled(self) -> dict:
        return self.per_class.get("__all__", {})


def evaluate_rows(
    model: AdvantageModel,
    table: RowTable,
    idx: torch.Tensor,
    *,
    var_floor: float,
    batch_rows: int = 2048,
) -> HoldoutEval:
    """Held-out diagnostics on rows WITH Q: weighted MSE vs the noise floor
    (the weighted mean noise variance = a perfect model's expected MSE),
    per-class breakdown, and top-card agreement with the committee draw
    for the model vs the prior (the pooling diagnostic of §20.4)."""
    model.eval()
    device = ppo_module.device
    q_idx = idx[table.has_q[idx]]
    sums: dict = defaultdict(
        lambda: {
            "w": 0.0,
            "wmse": 0.0,
            "wnoise": 0.0,
            "n": 0,
            "agree_model": 0,
            "agree_prior": 0,
            "su2_sum": 0.0,
        }
    )
    residual_sq, noise_list = [], []
    with torch.no_grad():
        for b in _batches(q_idx, batch_rows, False, random.Random(0)):
            enc = table.encoded(b)
            enc = EncodedRows(
                features=enc.features.to(device),
                hand_tokens=enc.hand_tokens.to(device),
                hand_ids=enc.hand_ids.to(device),
                masks=enc.masks.to(device),
            )
            pred, su2 = model.forward_with_variance(enc, table.prior[b].to(device))
            pred = pred.cpu()
            su2 = su2.cpu() if su2 is not None else None
            adv, lab, nv = table.advantage[b], table.label_mask[b], table.noise_var[b]
            _, per_row = weighted_rows_mse(pred, adv, lab, nv, var_floor)
            w = 1.0 / (nv + var_floor)
            neg = torch.full_like(pred, -1e9)
            obs_top = torch.where(lab, adv, neg).argmax(dim=1)
            model_top = torch.where(lab, pred, neg).argmax(dim=1)
            prior_top = torch.where(lab, table.prior[b], neg).argmax(dim=1)
            for i, row in enumerate(b.tolist()):
                cls = table.node_class[row]
                for key in (cls, "__all__"):
                    s = sums[key]
                    s["w"] += float(w[i])
                    s["wmse"] += float(w[i] * per_row[i])
                    s["wnoise"] += float(w[i] * nv[i])
                    s["n"] += 1
                    s["agree_model"] += int(model_top[i] == obs_top[i])
                    s["agree_prior"] += int(prior_top[i] == obs_top[i])
                    if su2 is not None:
                        s["su2_sum"] += float(su2[i])
            residual_sq.extend(per_row.tolist())
            noise_list.extend(nv.tolist())
    per_class = {}
    for key, s in sums.items():
        if s["n"] == 0:
            continue
        per_class[key] = {
            "n": s["n"],
            "weighted_mse": s["wmse"] / max(s["w"], 1e-12),
            "noise_floor": s["wnoise"] / max(s["w"], 1e-12),
            "top_agree_model": s["agree_model"] / s["n"],
            "top_agree_prior": s["agree_prior"] / s["n"],
            "sigma_u2_head_mean": s["su2_sum"] / s["n"],
        }
    return HoldoutEval(
        per_class=per_class,
        residual_sq_mean=float(np.mean(residual_sq)) if residual_sq else float("nan"),
        noise_var_mean=float(np.mean(noise_list)) if noise_list else float("nan"),
    )


def fit_advantage_model(
    model: AdvantageModel,
    table: RowTable,
    train_idx: torch.Tensor,
    holdout_idx: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_rows: int,
    var_floor: float,
    patience: int,
    seed: int,
    log=print,
    extra_var: torch.Tensor | None = None,
) -> FitReport:
    """Stage 1 on a cached ``RowTable`` (frozen rungs). Early-stops on the
    held-out weighted MSE and restores the best epoch's weights.

    ``extra_var`` (per row) is added to the noise variance in the training
    weight only: the Fay-Herriot generalized-least-squares weight is
    1 / (noise_var + sigma_u^2 of the row's cell), which keeps cells whose
    true advantages vary a lot (picker follows, ~2e-3 Q^2) from dominating
    the fit over cells whose whole signal is ~1e-4 Q^2 (lead conventions).
    Held-out diagnostics keep the plain 1 / noise_var weight so reports
    stay comparable across iterations."""
    if model.capacity == "trunk":
        raise ValueError("the trunk rung re-encodes rows; use fit_advantage_model_live")
    device = ppo_module.device
    model.to(device)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=lr, weight_decay=weight_decay
    )
    report = FitReport(capacity=model.capacity)
    q_train = train_idx[table.has_q[train_idx]]
    train_var = table.noise_var if extra_var is None else table.noise_var + extra_var
    best_state = copy.deepcopy(model.state_dict())
    since_best = 0
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in _batches(q_train, batch_rows, True, rng):
            enc = table.encoded(b)
            enc = EncodedRows(
                features=enc.features.to(device),
                hand_tokens=enc.hand_tokens.to(device),
                hand_ids=enc.hand_ids.to(device),
                masks=enc.masks.to(device),
            )
            pred, su2 = model.forward_with_variance(enc, table.prior[b].to(device))
            if su2 is not None:
                loss, _ = gaussian_nll_rows(
                    pred,
                    table.advantage[b].to(device),
                    table.label_mask[b].to(device),
                    table.noise_var[b].to(device),
                    su2,
                    var_floor,
                )
            else:
                loss, _ = weighted_rows_mse(
                    pred,
                    table.advantage[b].to(device),
                    table.label_mask[b].to(device),
                    train_var[b].to(device),
                    var_floor,
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        h_all = evaluate_rows(model, table, holdout_idx, var_floor=var_floor).pooled
        row = {
            "epoch": epoch,
            "train_weighted_mse": tot / max(n, 1),
            "holdout_weighted_mse": h_all.get("weighted_mse", float("nan")),
            "holdout_noise_floor": h_all.get("noise_floor", float("nan")),
            "holdout_top_agree_model": h_all.get("top_agree_model", float("nan")),
            "holdout_top_agree_prior": h_all.get("top_agree_prior", float("nan")),
        }
        report.epochs.append(row)
        log(
            f"[fit {model.capacity} epoch {epoch}] train wMSE {row['train_weighted_mse']:.3e} "
            f"holdout wMSE {row['holdout_weighted_mse']:.3e} "
            f"(noise floor {row['holdout_noise_floor']:.3e}) "
            f"top-agree model {row['holdout_top_agree_model']:.3f} "
            f"prior {row['holdout_top_agree_prior']:.3f}"
        )
        if row["holdout_weighted_mse"] < report.best_holdout_mse:
            report.best_holdout_mse = row["holdout_weighted_mse"]
            report.best_epoch = epoch
            report.holdout_noise_floor = row["holdout_noise_floor"]
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                log(f"[fit {model.capacity}] early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    final = evaluate_rows(model, table, holdout_idx, var_floor=var_floor)
    report.per_class = final.per_class
    report.sigma_u2 = estimate_residual_variance(
        final.residual_sq_mean, final.noise_var_mean
    )
    report.sigma_u2_by_class = class_residual_variances(
        report.per_class, report.sigma_u2, CLASS_SHRINK_ROWS
    )
    return report


def fit_advantage_model_iterated(
    make_model,
    table: RowTable,
    train_idx: torch.Tensor,
    holdout_idx: torch.Tensor,
    *,
    fh_iterations: int,
    class_shrink_rows: float,
    var_floor: float,
    log=print,
    **fit_kwargs,
) -> tuple[AdvantageModel, FitReport]:
    """Fay-Herriot iterated weighted least squares (§20.6 arm 3): fit with
    plain 1 / noise_var weights, estimate the per-cell residual variance on
    every row with Q, refit with 1 / (noise_var + sigma_u^2_cell) weights,
    repeat. ``make_model()`` returns a fresh model each round (the fit is
    re-run from scratch so early stopping stays honest). Returns the last
    round's model and report, the report's ``sigma_u2_by_class`` being the
    all-rows estimate the target stage should use."""
    extra = None
    model = make_model()
    report = FitReport(capacity=model.capacity)
    for it in range(fh_iterations):
        model = make_model()
        report = fit_advantage_model(
            model,
            table,
            train_idx,
            holdout_idx,
            var_floor=var_floor,
            log=log,
            extra_var=extra,
            **fit_kwargs,
        )
        all_rows = evaluate_rows(
            model, table, torch.arange(len(table)), var_floor=var_floor
        )
        report.sigma_u2_by_class = class_residual_variances(
            all_rows.per_class, report.sigma_u2, class_shrink_rows
        )
        extra = sigma_u2_rows(
            table.node_class, report.sigma_u2_by_class, report.sigma_u2
        )
        log(
            f"[fit {model.capacity} FH round {it + 1}/{fh_iterations}] best epoch "
            f"{report.best_epoch}, holdout wMSE {report.best_holdout_mse:.3e}, "
            f"sigma_u2 {report.sigma_u2:.3e}"
        )
    return model, report


# --------------------------------------------------------------------------- #
# Stage 1b: Fay-Herriot combination
# --------------------------------------------------------------------------- #
SIGMA_U2_FLOOR = 1e-7  # Q^2; keeps gamma defined when the model fits to the floor
# Row-count weight of the global sigma_u^2 when shrinking per-class values
# (§20.6): a cell with this many held-out rows counts as much as the prior.
CLASS_SHRINK_ROWS = 50.0


def estimate_residual_variance(residual_sq_mean: float, noise_var_mean: float) -> float:
    """Method-of-moments sigma_u^2: what the model does NOT explain beyond
    measurement noise (held-out E[r^2] − E[noise_var]), floored."""
    if not (math.isfinite(residual_sq_mean) and math.isfinite(noise_var_mean)):
        return SIGMA_U2_FLOOR
    return max(residual_sq_mean - noise_var_mean, SIGMA_U2_FLOOR)


def class_residual_variances(
    per_class: dict, global_sigma_u2: float, shrink_rows: float
) -> dict[str, float]:
    """Per-class sigma_u^2 (§20.6): the held-out identity E[r^2] = sigma_u^2
    + E[noise] applied within each telemetry cell (weighted MSE minus noise
    floor), then shrunk toward the global value by row count,
    (n * local + k * global) / (n + k) with k = ``shrink_rows`` — the
    empirical-Bayes step that keeps thin cells from swinging on their own
    sampling error. The cell is only a VARIANCE bucket here; it never says
    which card is right."""
    out = {}
    for cls, r in per_class.items():
        if cls == "__all__":
            continue
        local = max(r["weighted_mse"] - r["noise_floor"], SIGMA_U2_FLOOR)
        n = float(r["n"])
        out[cls] = (n * local + shrink_rows * global_sigma_u2) / (n + shrink_rows)
    return out


def sigma_u2_rows(
    node_class: list[str], by_class: dict[str, float], global_sigma_u2: float
) -> torch.Tensor:
    """Per-row sigma_u^2 from the class table; classes never seen on the
    held-out set fall back to the global value."""
    return torch.tensor(
        [by_class.get(c, global_sigma_u2) for c in node_class], dtype=torch.float32
    )


def blend_advantages(
    a_obs: torch.Tensor,
    a_model: torch.Tensor,
    has_q: torch.Tensor,
    noise_var: torch.Tensor,
    sigma_u2: "float | torch.Tensor",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(a_hat, v_post, gamma) per row. Rows with Q take the precision-
    weighted blend; rows without take the model alone with variance
    sigma_u^2. ``sigma_u2`` is one value for every row or a per-row
    tensor (the §20.6 per-class variances)."""
    su2 = torch.as_tensor(sigma_u2, dtype=noise_var.dtype).expand_as(noise_var)
    nv = torch.where(has_q, noise_var, torch.zeros_like(noise_var))
    gamma = torch.where(has_q, su2 / (su2 + nv), torch.zeros_like(nv))
    a_hat = gamma.unsqueeze(1) * a_obs + (1.0 - gamma).unsqueeze(1) * a_model
    v_post = torch.where(has_q, gamma * nv, su2)
    return a_hat, v_post, gamma


# --------------------------------------------------------------------------- #
# Stage 2: target
# --------------------------------------------------------------------------- #
def build_tilt_target(
    prior: torch.Tensor,
    a_hat: torch.Tensor,
    v_post: torch.Tensor,
    legal: torch.Tensor,
    *,
    kappa: float,
    tilt_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target, z): t ∝ prior * exp(z) over the legal set with
    z = clip(a_hat / (kappa * sqrt(v_post)), ±tilt_max). Identities: a_hat
    = 0 gives the prior back exactly; a large positive edge on one action
    gives ~one-hot; the clip bounds any single update."""
    scale = kappa * torch.sqrt(v_post.clamp(min=1e-12)).unsqueeze(1)
    z = (a_hat / scale).clamp(-tilt_max, tilt_max)
    z = torch.where(legal, z, torch.zeros_like(z))
    logits = torch.log(prior.clamp(min=1e-12)) + z
    logits = torch.where(legal, logits, torch.full_like(logits, -1e9))
    return F.softmax(logits, dim=-1), z


def targets_for_table(
    model: AdvantageModel,
    table: RowTable,
    *,
    sigma_u2: "float | torch.Tensor",
    kappa: float,
    tilt_max: float,
    batch_rows: int = 2048,
    node_variance: bool = False,
) -> dict:
    """Stage 1b + 2 over every targetable row. Returns dense tensors
    (targets, z, gamma, v_post, a_hat) aligned with ``table``.
    ``sigma_u2``: one value, or a per-row tensor aligned with ``table``;
    ``node_variance`` uses the heteroscedastic head's per-row sigma_u^2
    instead (ignored for a homoscedastic model)."""
    su2_rows = (
        sigma_u2
        if isinstance(sigma_u2, torch.Tensor)
        else torch.full((len(table),), float(sigma_u2))
    )
    model.eval()
    device = ppo_module.device
    out_t, out_z, out_g, out_v, out_a = [], [], [], [], []
    with torch.no_grad():
        for b in _batches(
            torch.arange(len(table)), batch_rows, False, random.Random(0)
        ):
            enc = table.encoded(b)
            enc = EncodedRows(
                features=enc.features.to(device),
                hand_tokens=enc.hand_tokens.to(device),
                hand_ids=enc.hand_ids.to(device),
                masks=enc.masks.to(device),
            )
            a_model, su2_head = model.forward_with_variance(
                enc, table.prior[b].to(device)
            )
            a_model = a_model.cpu()
            legal = table.masks[b].bool()
            # The model predicts over hand slots; center it over the legal
            # set so it lives on the same zero as the observed advantages.
            a_model = torch.where(legal, a_model, torch.zeros_like(a_model))
            n_legal = legal.sum(dim=1).clamp(min=1).unsqueeze(1)
            a_model = a_model - (a_model.sum(dim=1, keepdim=True) / n_legal)
            a_model = torch.where(legal, a_model, torch.zeros_like(a_model))
            su2_b = (
                su2_head.cpu()
                if (su2_head is not None and node_variance)
                else su2_rows[b]
            )
            a_hat, v_post, gamma = blend_advantages(
                table.advantage[b],
                a_model,
                table.has_q[b],
                table.noise_var[b],
                su2_b,
            )
            t, z = build_tilt_target(
                table.prior[b], a_hat, v_post, legal, kappa=kappa, tilt_max=tilt_max
            )
            out_t.append(t)
            out_z.append(z)
            out_g.append(gamma)
            out_v.append(v_post)
            out_a.append(a_hat)
    return {
        "target": torch.cat(out_t),
        "z": torch.cat(out_z),
        "gamma": torch.cat(out_g),
        "v_post": torch.cat(out_v),
        "a_hat": torch.cat(out_a),
    }


# --------------------------------------------------------------------------- #
# Stage 1, trunk rung: live re-encoding
# --------------------------------------------------------------------------- #
def fit_advantage_model_live(
    model: AdvantageModel,
    agent: PPOAgent,
    shards: list[dict],
    table: RowTable,
    train_idx: torch.Tensor,
    holdout_idx: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    buffer_episodes: int,
    batch_segments: int,
    var_floor: float,
    patience: int,
    seed: int,
    log=print,
) -> FitReport:
    """Stage 1 for the ``trunk`` rung: the encoder copy is trainable, so
    rows are re-encoded every epoch instead of read from the cache. The
    evidence and the train/holdout split come from ``table`` (keyed by
    row reference); held-out evaluation re-encodes with the trained copy
    too, through a temporary table swap."""
    if model.capacity != "trunk":
        raise ValueError("fit_advantage_model_live is the trunk rung's fit")
    assert model.encoder is not None
    device = ppo_module.device
    model.to(device)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=lr, weight_decay=weight_decay
    )
    report = FitReport(capacity="trunk")
    ref_to_row = {ref: r for r, ref in enumerate(table.refs)}
    train_rows = set(train_idx.tolist())
    best_state = copy.deepcopy(model.state_dict())
    since_best = 0

    def encoded_table(train_mode: bool) -> RowTable:
        """Re-encode every targetable row with the model's encoder copy."""
        feats, toks, ids, masks, order = [], [], [], [], []
        for shard_idx, shard in enumerate(shards):
            episodes = shard["episodes"]
            for start in range(0, len(episodes), buffer_episodes):
                chunk = episodes[start : start + buffer_episodes]
                agent.reset_storage()
                source_map = store_corpus_episodes(agent, chunk)
                for rows in iter_row_batches(agent, batch_segments=batch_segments):
                    with torch.no_grad():
                        enc = encode_rows(agent, rows, encoder=model.encoder)
                    for r, ev_idx in enumerate(rows.event_indices):
                        ref = source_map[ev_idx]
                        assert ref is not None
                        row = ref_to_row.get((shard_idx, start + ref[0], ref[1]))
                        if row is None:
                            continue
                        order.append(row)
                        feats.append(enc.features[r].cpu())
                        toks.append(enc.hand_tokens[r].cpu())
                        ids.append(enc.hand_ids[r].cpu())
                        masks.append(enc.masks[r].cpu())
        agent.reset_storage()
        idx = torch.tensor(order, dtype=torch.long)
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(len(idx))
        return RowTable(
            features=torch.stack(feats)[inv],
            hand_tokens=torch.stack(toks)[inv],
            hand_ids=torch.stack(ids)[inv],
            masks=torch.stack(masks)[inv],
            advantage=table.advantage,
            label_mask=table.label_mask,
            noise_var=table.noise_var,
            prior=table.prior,
            has_q=table.has_q,
            node_class=table.node_class,
            refs=table.refs,
            game=table.game,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for shard_idx, shard in enumerate(shards):
            episodes = shard["episodes"]
            for start in range(0, len(episodes), buffer_episodes):
                chunk = episodes[start : start + buffer_episodes]
                agent.reset_storage()
                source_map = store_corpus_episodes(agent, chunk)
                for rows in iter_row_batches(
                    agent, batch_segments=batch_segments, shuffle=True, rng=rng
                ):
                    keep, tab_rows = [], []
                    for r, ev_idx in enumerate(rows.event_indices):
                        ref = source_map[ev_idx]
                        assert ref is not None
                        row = ref_to_row.get((shard_idx, start + ref[0], ref[1]))
                        if (
                            row is None
                            or row not in train_rows
                            or not bool(table.has_q[row])
                        ):
                            continue
                        keep.append(r)
                        tab_rows.append(row)
                    if not keep:
                        continue
                    enc = encode_rows(agent, rows, encoder=model.encoder)
                    sel = torch.tensor(keep, dtype=torch.long, device=device)
                    enc = EncodedRows(
                        features=enc.features[sel],
                        hand_tokens=enc.hand_tokens[sel],
                        hand_ids=enc.hand_ids[sel],
                        masks=enc.masks[sel],
                    )
                    b = torch.tensor(tab_rows, dtype=torch.long)
                    pred = model(enc, table.prior[b].to(device))
                    loss, _ = weighted_rows_mse(
                        pred,
                        table.advantage[b].to(device),
                        table.label_mask[b].to(device),
                        table.noise_var[b].to(device),
                        var_floor,
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach()) * len(b)
                    n += len(b)
        live = encoded_table(train_mode=False)
        h_all = evaluate_rows(model, live, holdout_idx, var_floor=var_floor).pooled
        row = {
            "epoch": epoch,
            "train_weighted_mse": tot / max(n, 1),
            "holdout_weighted_mse": h_all.get("weighted_mse", float("nan")),
            "holdout_noise_floor": h_all.get("noise_floor", float("nan")),
            "holdout_top_agree_model": h_all.get("top_agree_model", float("nan")),
            "holdout_top_agree_prior": h_all.get("top_agree_prior", float("nan")),
        }
        report.epochs.append(row)
        log(
            f"[fit trunk epoch {epoch}] train wMSE {row['train_weighted_mse']:.3e} "
            f"holdout wMSE {row['holdout_weighted_mse']:.3e} "
            f"(noise floor {row['holdout_noise_floor']:.3e})"
        )
        if row["holdout_weighted_mse"] < report.best_holdout_mse:
            report.best_holdout_mse = row["holdout_weighted_mse"]
            report.best_epoch = epoch
            report.holdout_noise_floor = row["holdout_noise_floor"]
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    live = encoded_table(train_mode=False)
    final = evaluate_rows(model, live, holdout_idx, var_floor=var_floor)
    report.per_class = final.per_class
    report.sigma_u2 = estimate_residual_variance(
        final.residual_sq_mean, final.noise_var_mean
    )
    report.sigma_u2_by_class = class_residual_variances(
        report.per_class, report.sigma_u2, CLASS_SHRINK_ROWS
    )
    # Stage 2 reads encodings from the saved table; the trunk rung's
    # encoder differs from theta_k's, so refresh the cached encodings.
    table.features, table.hand_tokens = live.features, live.hand_tokens
    return report
