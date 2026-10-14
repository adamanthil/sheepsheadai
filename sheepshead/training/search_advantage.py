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
          fresh pointer scorer over the frozen adapter output) and
          ``adapter`` (fresh adapter MLP + pointer; the standing recipe).
          The unfrozen-encoder rung of the §20 screen never beat the
          adapter on held-out MSE and was retired.
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

CAPACITIES = ("pointer", "adapter")
POINTER_HIDDEN = 64  # matches the actor's pointer scorer
# Heteroscedastic head: log sigma_u^2 init (~ the global residual variance
# measured on corpus q) and the clamp that keeps the NLL well-conditioned.
LOGVAR_INIT = 1e-3
LOGVAR_MIN, LOGVAR_MAX = math.log(1e-7), math.log(1.0)

# Corpus rows the §20 recipe can TARGET: searched play rows, whatever the
# committee concluded there (the resolved/unresolved split dissolves).
TARGETABLE_SETS = ("resolved", "unresolved")
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
    noise_var: float  # the corpus's pooled-mean noise of a RAW Q (row mean)
    noise_q_var: list[float]  # per action, the pooled-mean noise of its raw Q; nan
    noise_s2c: list[float]  # per-replicate variance of the CENTERED Q, dense; nan
    noise_nobs: list[int]  # replicates that visited the action, dense; 0
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
        noise_q_var = sorted_valid_to_dense(
            source_event["search_q_var"], valid, action_size, fill=float("nan")
        )
        s2c = source_event.get("search_q_s2_centered")
        nobs = source_event.get("search_q_n_obs")
        if s2c is not None and nobs is not None:
            noise_s2c = sorted_valid_to_dense(
                s2c, valid, action_size, fill=float("nan")
            )
            noise_nobs = [
                int(x) for x in sorted_valid_to_dense(nobs, valid, action_size)
            ]
        else:  # a corpus written before the centered field (2026-10-07)
            noise_s2c = [float("nan")] * action_size
            noise_nobs = [0] * action_size
    else:
        advantage = [0.0] * action_size
        label_mask = [False] * action_size
        noise_var = float("nan")
        noise_q_var = [float("nan")] * action_size
        noise_s2c = [float("nan")] * action_size
        noise_nobs = [0] * action_size
    return RowEvidence(
        targetable=True,
        has_q=has_q,
        advantage=advantage,
        label_mask=label_mask,
        noise_var=noise_var,
        noise_q_var=noise_q_var,
        noise_s2c=noise_s2c,
        noise_nobs=noise_nobs,
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
    noise_var: torch.Tensor  # (R,) WORKING label noise, row mean (apply_noise_model)
    prior: torch.Tensor  # (R, A)
    has_q: torch.Tensor  # (R,) bool
    node_class: list[str]
    refs: list[tuple[int, int, int]]  # (shard_idx, episode_idx, event_idx)
    game: torch.Tensor  # (R,) corpus game index (for game-level splits)
    noise_var_raw: torch.Tensor  # (R,) the corpus's raw-Q noise, nan where !has_q
    noise_q_var: torch.Tensor  # (R, A) per-action raw-Q pooled-mean noise, nan
    noise_s2c: torch.Tensor  # (R, A) centered per-replicate variance, nan
    noise_nobs: torch.Tensor  # (R, A) replicates per action, 0
    noise_var_action: torch.Tensor  # (R, A) WORKING label noise per action, nan

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
            noise_var_raw=self.noise_var_raw[idx],
            noise_q_var=self.noise_q_var[idx],
            noise_s2c=self.noise_s2c[idx],
            noise_nobs=self.noise_nobs[idx],
            noise_var_action=self.noise_var_action[idx],
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

        def cat(name: str) -> torch.Tensor:
            return torch.cat([getattr(t, name) for t in tables], dim=0)

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
            noise_var_raw=cat("noise_var_raw"),
            noise_q_var=cat("noise_q_var"),
            noise_s2c=cat("noise_s2c"),
            noise_nobs=cat("noise_nobs"),
            noise_var_action=cat("noise_var_action"),
        )

    def save(self, path: str) -> None:
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path: str) -> "RowTable":
        d = torch.load(path, map_location="cpu", weights_only=False)
        if "noise_var_raw" not in d:  # a cache from before the noise fields
            d["noise_var_raw"] = d["noise_var"].clone()
            d["noise_s2c"] = torch.full_like(d["advantage"], float("nan"))
            d["noise_nobs"] = torch.zeros_like(d["advantage"], dtype=torch.long)
        if "noise_q_var" not in d:  # ... or before the per-action fields
            lab = d["label_mask"]
            d["noise_q_var"] = torch.where(
                lab, d["noise_var_raw"].unsqueeze(1).expand_as(lab), torch.nan
            )
            d["noise_var_action"] = d["noise_q_var"].clone()
        return RowTable(**d)


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
                    noise_var_raw=torch.tensor([e.noise_var for e in evidence]),
                    noise_q_var=torch.tensor([e.noise_q_var for e in evidence]),
                    noise_s2c=torch.tensor([e.noise_s2c for e in evidence]),
                    noise_nobs=torch.tensor(
                        [e.noise_nobs for e in evidence], dtype=torch.long
                    ),
                    noise_var_action=torch.tensor([e.noise_q_var for e in evidence]),
                )
            )
    agent.reset_storage()
    return RowTable.concat(pieces)


NOISE_MODELS = ("centered", "independent", "raw")


NOISE_POOLS = ("class", "corpus")
# df weight of the corpus-wide centered variance when a class's own mean is
# the shrinkage target: a class with this many replicate degrees of freedom
# counts as much as the corpus. Only thin classes feel it (a 3,000-row
# class carries ~30,000 df).
NOISE_POOL_DF = 200.0


def apply_noise_model(
    table: RowTable, mode: str, shrink_nu: float, pool: str = "class"
) -> dict:
    """Set the table's WORKING label noise — ``noise_var_action`` (R, A), the
    measurement noise of each labeled action's label (the pooled Q centered
    over the legal set), and ``noise_var`` (R,), its mean over the labeled
    actions (the Fay-Herriot blend's gamma and the fit weights are per
    row) — from the corpus fields:

    raw          the corpus's row-mean pooled noise of a RAW Q on every
                 action (the recipe before 2026-10-07: it over-subtracts,
                 since centering removes 1/n of independent noise and all
                 common-mode noise, and on the pooled defender/partner cells
                 that bias exceeded the sigma_u^2 it produced 10-30x —
                 gamma ~0.01 was an estimator artifact).
    independent  the exact centering under independent per-card noise from
                 the per-action search_q_var: var(y_a) = s_a^2 (1 - 2/n) +
                 sum_b s_b^2 / n^2 over the n labeled actions. A lower bound
                 on the correction when replicates share noise.
    centered     per action, the measured across-replicate variance of the
                 centered Q, shrunk with ``shrink_nu`` toward a pooled
                 (df-weighted) mean as the corpus does for raw Q, over the
                 replicate count; actions (or rows) without the field — a
                 corpus written before it — take ``independent``.

    ``pool`` names the shrinkage target of the centered model. With three
    replicates a card carries 2 df against shrink_nu = 4, so the target
    takes two thirds of every label's noise:

    corpus       the corpus-wide mean (the 2026-10-07 recipe). Quiet
                 classes inherit the loud ones' noise — the recall run's
                 iteration-1 table pooled at 1.46e-3 while defender follows
                 measured 0.04-0.39 of that, so their noise was overstated
                 by up to 25x, the held-out residual sat BELOW the noise
                 floor, and sigma_u^2 came out at the floor in every one;
                 picker follows (1.7-2.9x the mean) ran the other way.
    class        the row's telemetry class mean (tens of thousands of df in
                 the big classes), itself shrunk toward the corpus-wide mean
                 by df with ``NOISE_POOL_DF``. Classes fix the search
                 budget, so this is per budget too.
    Returns counts and the pooled centered variances for the log."""
    if mode not in NOISE_MODELS:
        raise ValueError(f"noise model must be one of {NOISE_MODELS}, got {mode!r}")
    if pool not in NOISE_POOLS:
        raise ValueError(f"noise pool must be one of {NOISE_POOLS}, got {pool!r}")
    lab = table.label_mask
    raw = table.noise_var_raw
    n = lab.sum(dim=1, keepdim=True).to(raw.dtype)
    stats = {"mode": mode, "rows_with_q": int(table.has_q.sum())}
    nan = torch.full_like(table.noise_q_var, float("nan"))
    if mode == "raw":
        per_action = torch.where(lab, raw.unsqueeze(1).expand_as(lab), nan)
    else:
        q_var = torch.where(lab, table.noise_q_var, torch.zeros_like(nan))
        total = q_var.sum(dim=1, keepdim=True)
        independent = q_var * (1.0 - 2.0 / n.clamp(min=1)) + total / n.clamp(min=1) ** 2
        independent = torch.where(
            lab & (n >= 2), independent, torch.where(lab, q_var, nan)
        )
        per_action = independent
        if mode == "centered":
            nobs = table.noise_nobs.to(raw.dtype)
            ok = lab & (nobs >= 2) & torch.isfinite(table.noise_s2c)
            df = torch.where(ok, nobs - 1.0, torch.zeros_like(nobs))
            s2c = torch.where(ok, table.noise_s2c, torch.zeros_like(nobs))
            df_total = float(df.sum())
            if df_total > 0.0:
                s2_global = float((df * s2c).sum() / df_total)
                target = torch.full_like(s2c, s2_global)
                by_class: dict[str, float] = {}
                if pool == "class":
                    names = sorted(set(table.node_class))
                    index = {c: i for i, c in enumerate(names)}
                    cls = torch.tensor([index[c] for c in table.node_class])
                    df_c = torch.zeros(len(names), dtype=df.dtype).index_add_(
                        0, cls, df.sum(dim=1)
                    )
                    s2_c = torch.zeros(len(names), dtype=df.dtype).index_add_(
                        0, cls, (df * s2c).sum(dim=1)
                    )
                    pooled = (s2_c + NOISE_POOL_DF * s2_global) / (df_c + NOISE_POOL_DF)
                    target = pooled[cls].unsqueeze(1).expand_as(s2c)
                    by_class = {
                        c: float(pooled[i]) for c, i in index.items() if df_c[i] > 0
                    }
                blend = (shrink_nu * target + df * s2c) / (shrink_nu + df)
                per_action = torch.where(ok, blend / nobs.clamp(min=1), independent)
                use = table.has_q & ok.any(dim=1)
                stats.update(
                    rows_centered=int(use.sum()),
                    rows_independent=int((table.has_q & ~use).sum()),
                    s2_centered_global=s2_global,
                    s2_centered_by_class=by_class,
                )
            else:
                stats.update(rows_centered=0, rows_independent=stats["rows_with_q"])
    table.noise_var_action = per_action
    row = torch.where(lab, per_action, torch.zeros_like(per_action)).sum(
        dim=1
    ) / n.squeeze(1).clamp(min=1)
    table.noise_var = torch.where(table.has_q, row, raw)
    stats["noise_mean"] = (
        float(table.noise_var[table.has_q].mean()) if bool(table.has_q.any()) else None
    )
    stats["noise_raw_mean"] = (
        float(raw[table.has_q].mean()) if bool(table.has_q.any()) else None
    )
    return stats


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
    Both train on a cached ``RowTable`` of frozen encodings.

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

    def __init__(
        self,
        agent: PPOAgent,
        capacity: str,
        heteroscedastic: bool = False,
        bilinear: bool = False,
    ):
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
        self.pointer_Wg = nn.Linear(d_model, POINTER_HIDDEN)
        self.pointer_Wt = nn.Linear(d_token, POINTER_HIDDEN)
        self.pointer_v = nn.Linear(POINTER_HIDDEN, 1, bias=False)
        self.prior_scale = nn.Parameter(torch.tensor(0.01))
        # §20.8: a bilinear state x card term. The pointer's tanh(W_g h + W_t
        # token) is additive inside the nonlinearity and gates a card
        # attribute by the state only weakly; (U h) . (V token) scores a card
        # attribute conditionally on the state (e.g. "called-suit card, but
        # only at a defender lead before the suit was led"). Zero-initialized
        # on the state side so the model starts as the plain pointer.
        self.bilinear = bool(bilinear)
        if self.bilinear:
            self.bilinear_U = nn.Linear(d_model, POINTER_HIDDEN, bias=False)
            self.bilinear_V = nn.Linear(d_token, POINTER_HIDDEN, bias=False)
            nn.init.zeros_(self.bilinear_U.weight)
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
        if self.bilinear:
            u = self.bilinear_U(h).unsqueeze(1)  # (R, 1, hidden)
            v = self.bilinear_V(enc.hand_tokens)  # (R, 8, hidden)
            slot = slot + (u * v).sum(-1) / POINTER_HIDDEN**0.5
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
def center_over(pred: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    """``pred`` minus its mean over the labeled actions of each row (0 off
    the labeled set). The labels are centered over the legal set and the
    tilt ignores a per-row constant, so the fit must not spend capacity on
    one — before 2026-10-07 it did, and any offset it failed to cancel sat
    in the residuals sigma_u^2 is built from."""
    lab = label_mask.to(pred.dtype)
    n = lab.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (pred * lab).sum(dim=1, keepdim=True) / n
    return (pred - mean) * lab


def weighted_rows_mse(
    pred: torch.Tensor,
    advantage: torch.Tensor,
    label_mask: torch.Tensor,
    noise_var: torch.Tensor,
    var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss, per_row_mse): per row the mean squared error over labeled
    actions of the prediction CENTERED over them; the loss is the
    1/noise_var-weighted mean over rows (the heteroscedastic least-squares
    objective)."""
    pred = center_over(pred, label_mask)
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
    advantages under total variance noise_var + sigma_u^2(s) (``noise_var``
    per row, or per row AND action as ``RowTable.noise_var_action``),
    the prediction centered over the labeled actions, averaged over labeled
    actions per row and then over rows. Standardizes each row by its own
    learned scale — the taxonomy-free replacement for the per-cell
    Fay-Herriot weights."""
    pred = center_over(pred, label_mask)
    nv = noise_var if noise_var.dim() == 2 else noise_var.unsqueeze(1)
    nv = torch.where(label_mask, nv.nan_to_num(0.0), torch.zeros_like(pred))
    total = nv + (sigma_u2 + var_floor).unsqueeze(1)
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
    # Cross-fitting (CE_Teacher_Design §22.5 item 2): folds, the fixed epoch
    # count, and the OUT-OF-FOLD residual variance (global + per class) —
    # honest on every row, the blend's sigma_u^2 when present.
    cross_fit: dict = field(default_factory=dict)

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
    oof: "tuple[torch.Tensor, torch.Tensor | None] | None" = None,
) -> HoldoutEval:
    """Held-out diagnostics on rows WITH Q: weighted MSE vs the noise floor
    (the weighted mean noise variance = a perfect model's expected MSE),
    per-class breakdown, and top-card agreement with the committee draw
    for the model vs the prior (the pooling diagnostic of §20.4). With
    ``oof`` (out-of-fold a_model and per-row sigma_u^2 aligned with the
    table) those stand in for the model's own predictions."""
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
            if oof is not None:
                pred = oof[0][b]
                su2 = oof[1][b] if oof[1] is not None else None
            else:
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
    fixed_epochs: bool = False,
) -> FitReport:
    """Stage 1 on a cached ``RowTable`` (frozen rungs). Early-stops on the
    held-out weighted MSE and restores the best epoch's weights — unless
    ``fixed_epochs``: then every epoch runs and the LAST weights are kept,
    so ``holdout_idx`` selects nothing (a cross-fitting fold model must not
    pick its epoch on the rows it will predict).

    ``extra_var`` (per row) is added to the noise variance in the training
    weight only: the Fay-Herriot generalized-least-squares weight is
    1 / (noise_var + sigma_u^2 of the row's cell), which keeps cells whose
    true advantages vary a lot (picker follows, ~2e-3 Q^2) from dominating
    the fit over cells whose whole signal is ~1e-4 Q^2 (lead conventions).
    It applies to the homoscedastic objective only: a heteroscedastic
    model learns that per-row variance itself and its NLL already
    standardizes every row by it, so passing ``extra_var`` there is an
    error rather than a silent no-op. Held-out diagnostics keep the plain
    1 / noise_var weight so reports stay comparable across iterations."""
    if extra_var is not None and model.heteroscedastic:
        raise ValueError(
            "extra_var is the homoscedastic GLS weight; a heteroscedastic model "
            "learns its own per-row variance (nothing to reweight)"
        )
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
                    table.noise_var_action[b].to(device),
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
        if fixed_epochs:
            report.best_holdout_mse = row["holdout_weighted_mse"]
            report.best_epoch = epoch
            report.holdout_noise_floor = row["holdout_noise_floor"]
            continue
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
    if not fixed_epochs:
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
    all-rows estimate the target stage should use.

    A heteroscedastic model (§20.8) is fit ONCE whatever ``fh_iterations``
    says: its NLL learns the per-row variance the reweighting rounds were
    estimating per cell, so a second round would be a fresh-init refit of
    the same objective (the standing recipe ran that inert round until
    2026-10-06)."""
    extra = None
    model = make_model()
    report = FitReport(capacity=model.capacity)
    if model.heteroscedastic and fh_iterations > 1:
        log(
            f"[fit {model.capacity}] heteroscedastic model: one round "
            f"(fh_iterations {fh_iterations} applies to the homoscedastic fit)"
        )
        fh_iterations = 1
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


def predict_rows(
    model: AdvantageModel, table: RowTable, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(a_model, sigma_u2_head) for rows ``b``: the model's advantages
    centered over the legal set (the zero the observed advantages live on;
    0 off-legal) and the heteroscedastic head's per-row variance (None for a
    homoscedastic model). CPU tensors."""
    device = ppo_module.device
    enc = table.encoded(b)
    enc = EncodedRows(
        features=enc.features.to(device),
        hand_tokens=enc.hand_tokens.to(device),
        hand_ids=enc.hand_ids.to(device),
        masks=enc.masks.to(device),
    )
    a_model, su2_head = model.forward_with_variance(enc, table.prior[b].to(device))
    a_model = a_model.cpu()
    legal = table.masks[b].bool()
    a_model = torch.where(legal, a_model, torch.zeros_like(a_model))
    n_legal = legal.sum(dim=1).clamp(min=1).unsqueeze(1)
    a_model = a_model - (a_model.sum(dim=1, keepdim=True) / n_legal)
    a_model = torch.where(legal, a_model, torch.zeros_like(a_model))
    return a_model, (su2_head.cpu() if su2_head is not None else None)


def fold_rows_by_game(table: RowTable, folds: int, seed: int) -> torch.Tensor:
    """Fold id per row, assigned at GAME granularity (the five seats of a
    deal share an outcome) by a seeded shuffle of the games."""
    games = sorted(set(table.game.tolist()))
    random.Random(seed).shuffle(games)
    fold_of_game = {g: i % folds for i, g in enumerate(games)}
    return torch.tensor(
        [fold_of_game[g] for g in table.game.tolist()], dtype=torch.long
    )


def cross_fit_predictions(
    make_model,
    table: RowTable,
    *,
    folds: int,
    epochs: int,
    var_floor: float,
    seed: int,
    log=print,
    **fit_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, list[dict]]:
    """Out-of-fold advantages for every row (CE_Teacher_Design §22.5 item
    2). ``folds`` models, each trained on the other folds' games for
    exactly ``epochs`` epochs (the count the ordinary held-out fit chose;
    no early stopping, so nothing is selected on the predicted fold), and
    each row's a_model and per-row sigma_u^2(s) come from the model that
    never saw its game. The blend's two inputs then carry independent
    errors, and the residual on every row is an honest one for sigma_u^2.
    Returns (a_oof (R, A), su2_oof (R,) or None, fold per row, the fold
    models' state dicts)."""
    if folds < 2:
        raise ValueError(f"cross-fitting needs at least 2 folds, got {folds}")
    fold = fold_rows_by_game(table, folds, seed)
    a_oof = torch.zeros_like(table.advantage)
    su2_oof: torch.Tensor | None = None
    states = []
    for i in range(folds):
        train_idx = torch.nonzero(fold != i).squeeze(1)
        pred_idx = torch.nonzero(fold == i).squeeze(1)
        torch.manual_seed(seed + 1000 + i)
        model = make_model()
        report = fit_advantage_model(
            model,
            table,
            train_idx,
            pred_idx,
            epochs=epochs,
            var_floor=var_floor,
            seed=seed + i,
            log=lambda *_: None,
            fixed_epochs=True,
            **fit_kwargs,
        )
        log(
            f"[cross-fit {i + 1}/{folds}] {len(train_idx)} train rows -> "
            f"{len(pred_idx)} out-of-fold rows, {epochs} epochs, fold wMSE "
            f"{report.best_holdout_mse:.3e}"
        )
        model.eval()
        with torch.no_grad():
            for b in _batches(pred_idx, 2048, False, random.Random(0)):
                a_b, su2_b = predict_rows(model, table, b)
                a_oof[b] = a_b
                if su2_b is not None:
                    if su2_oof is None:
                        su2_oof = torch.zeros(len(table), dtype=su2_b.dtype)
                    su2_oof[b] = su2_b
        states.append({k: v.cpu() for k, v in model.state_dict().items()})
    return a_oof, su2_oof, fold, states


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


def floor_class_variances(
    all_rows: dict[str, float], holdout: dict[str, float]
) -> dict[str, float]:
    """Per-class sigma_u^2 for the blend: the all-rows estimate (§20.6,
    stable in thin cells) floored at the held-out one. On the rows it was
    fit to the model's residual can sit BELOW the search noise floor, so
    the all-rows subtraction bottoms out at SIGMA_U2_FLOOR and the class
    value collapses to the shrinkage prior alone (std defender follows:
    3e-6 all-rows vs 1.3e-5 held-out, rc_validate_v2 iteration 1) — a
    posterior SE set by ``CLASS_SHRINK_ROWS``, not by data. The held-out
    residual variance is the model's actual out-of-sample error on the
    class (its u AND its own estimation error, the Prasad-Rao g2 term a
    flexible model has no closed form for), so the blend's v_post never
    claims less than it."""
    return {cls: max(v, holdout.get(cls, v)) for cls, v in all_rows.items()}


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
    eta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target, z): t ∝ prior * exp(z) over the legal set with
    z = clip(a_hat / scale, ±tilt_max). Identities: a_hat = 0 gives the
    prior back exactly; a large positive edge on one action gives ~one-hot;
    the clip bounds any single update.

    ``scale`` is the tilt temperature. Default: kappa * sqrt(v_post), the
    posterior SE (§20.9) — a row's temperature shrinks as its posterior
    tightens, so the quiet late-trick rows with the least at stake get the
    sharpest targets (Arm_D_Review §6.1). With ``eta`` the temperature is
    ONE value in advantage units for every row (the KL-regularized policy-
    improvement step, MPO/AWR form): the posterior enters through a_hat
    alone, and a row's target moves by how much its decision is worth."""
    scale = tilt_scale(v_post, kappa=kappa, eta=eta)
    z = (a_hat / scale).clamp(-tilt_max, tilt_max)
    z = torch.where(legal, z, torch.zeros_like(z))
    logits = torch.log(prior.clamp(min=1e-12)) + z
    logits = torch.where(legal, logits, torch.full_like(logits, -1e9))
    return F.softmax(logits, dim=-1), z


def tilt_scale(
    v_post: torch.Tensor, *, kappa: float, eta: float | None
) -> torch.Tensor:
    """(R, 1) tilt temperature of ``build_tilt_target``: ``eta`` on every
    row, or kappa * sqrt(v_post) when ``eta`` is None."""
    if eta is not None:
        return torch.full((v_post.size(0), 1), float(eta), dtype=v_post.dtype)
    return kappa * torch.sqrt(v_post.clamp(min=1e-12)).unsqueeze(1)


CE_WEIGHTS = ("precision", "uniform")
TILT_SCALES = ("posterior", "global")


def ce_weights(v_post: torch.Tensor, mode: str, weight_max: float) -> torch.Tensor:
    """Per-row CE weights, mean 1 over the targeted rows (the same dose as
    the uniform loss). ``precision``: 1 / v_post (§20.9), capped at
    ``weight_max`` then re-normalized — allocates the dose by posterior
    precision, i.e. toward quiet rows. ``uniform``: every row 1."""
    if mode not in CE_WEIGHTS:
        raise ValueError(f"weights must be one of {CE_WEIGHTS}, got {mode!r}")
    if mode == "uniform":
        return torch.ones_like(v_post)
    prec = 1.0 / v_post.clamp(min=1e-9)
    weights = (prec / prec.mean()).clamp(max=weight_max)
    return weights / weights.mean()


def clipped_disagreement(
    target: torch.Tensor,
    z_abs: torch.Tensor,
    table: RowTable,
    *,
    tilt_max: float,
    committee_se_mult: float = 2.0,
) -> dict:
    """Over-confidence read on the rows whose tilt hit the clip (a near
    one-hot target): how often the target's top card is not the committee
    draw's top card — overall, and restricted to rows where the committee
    itself was confident (top-2 Q gap above ``committee_se_mult`` noise
    SEs; a disagreement there is a directional error, not a tie) — and
    how often it is not the prior's top card (the rest is the prior's own
    argmax sharpened). Counts only rows with Q and at least two labeled
    actions; the fractions are None when no row qualifies."""
    lab = table.label_mask
    eligible = table.has_q & (lab.sum(dim=1) >= 2)
    clipped = eligible & (z_abs >= tilt_max - 1e-6)
    neg = torch.finfo(table.advantage.dtype).min
    adv = torch.where(lab, table.advantage, torch.full_like(table.advantage, neg))
    top2 = adv.topk(2, dim=1).values
    gap = top2[:, 0] - top2[:, 1]
    confident = clipped & (
        gap > committee_se_mult * torch.sqrt(table.noise_var.nan_to_num(0.0))
    )
    t_top = target.argmax(dim=1)
    q_top = adv.argmax(dim=1)
    p_top = torch.where(lab, table.prior, torch.zeros_like(table.prior)).argmax(dim=1)
    dis_q = t_top != q_top
    dis_p = t_top != p_top

    def frac(mask: torch.Tensor, hit: torch.Tensor):
        n = int(mask.sum())
        return float(hit[mask].float().mean()) if n else None

    return {
        "n_eligible": int(eligible.sum()),
        "n_clipped": int(clipped.sum()),
        "n_clipped_committee_confident": int(confident.sum()),
        "clipped_frac": frac(eligible, clipped),
        "clipped_disagree_committee": frac(clipped, dis_q),
        "clipped_disagree_committee_confident": frac(confident, dis_q),
        "clipped_disagree_prior": frac(clipped, dis_p),
    }


PRIOR_CONFIDENT = 0.8  # theta_k top-card probability treated as confident
PRIOR_TOP_BINS = (0.0, 0.5, 0.8, 0.95)
EXTRA_KL_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)


def _mean_z(x: torch.Tensor) -> dict:
    """Mean of ``x`` with its z against 0 (None below two rows)."""
    n = int(x.numel())
    if n < 2:
        return {"mean": float(x.mean()) if n else None, "z": None}
    se = float(x.std()) / math.sqrt(n)
    return {"mean": float(x.mean()), "z": float(x.mean()) / se if se > 0 else None}


def model_push_diagnostics(
    table: RowTable,
    a_model: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    noise_var: torch.Tensor,
    *,
    tilt_max: float,
    out_of_fold: bool,
) -> dict:
    """Is the advantage model's part of the target supported by the search
    labels? (Arm_D_Review_202610 §7.3, §7.4, §8.) Three reads, all in
    advantage units (score / RETURN_SCALE), judged by each row's own
    centered search label y:

    ``top_card_residual``  per class, mean y − a_model at theta_k's top
        card. theta_k's choice is independent of the label noise, so with
        out-of-fold a_model a nonzero mean is a clean calibration lean
        (§7.4: positive = the model under-rates theta_k's card).
    ``override_support``  rows where a_model's top card is not theta_k's:
        the edge the model claims for its card against the edge the labels
        show (§7.3). Conditioned on the model's own argmax, so the claim
        carries the winner's curse; a support ratio below 1 is partly that.
    ``push``  the target split in log space into a search-only tilt
        t_s (gamma·y) and a model-only tilt t_m ((1 − gamma)·a_model), each
        valued as sum((t − prior) · y):
          model_gain   t_m's value; unbiased for its true search-Q value
                       when a_model is out of fold.
          model_claim  t_m valued by a_model itself.
          full, search the target's and t_s's value, Stein-corrected for
                       their dependence on y (label noise covariance
                       noise_var · (I − 11'/k) on the legal set, Gaussian).
          marginal     full − search: what the model adds inside the
                       target.
          stein        the corrections subtracted from full and search:
                       how far those two reads lean on the noise model.
          prior_top_lost  probability theta_k's top card loses under the
                       target and under t_s alone.
          reversed     the target's top card is not t_s's;
                       ``reversed_vs_evidence`` additionally with the
                       label favouring t_s's card by > 2 SE.
        Reported overall, per class, by theta_k's top-card probability, by
        the model's extra movement KL(target || t_s), and per class within
        the top extra-movement bin (``push_top_extra_by_class``).

    ``scale`` is the (R, 1) tilt temperature (``tilt_scale``) and
    ``noise_var`` the WORKING row label noise the blend used. Only rows
    with Q and at least two labeled actions count."""
    legal = table.masks.bool()
    rows = table.has_q & (table.label_mask.sum(dim=1) >= 2)
    idx = rows.nonzero()[:, 0]
    legal = legal[idx]
    y = torch.where(
        legal,
        table.advantage[idx],
        torch.zeros_like(legal, dtype=table.advantage.dtype),
    )
    m = torch.where(legal, a_model[idx], torch.zeros_like(y))
    prior = table.prior[idx]
    g = gamma[idx].unsqueeze(1)
    sc = scale[idx]
    s2 = noise_var[idx]
    k = legal.sum(dim=1).to(y.dtype)
    log_p = torch.log(prior.clamp(min=1e-12))
    neg = torch.full_like(y, -1e9)

    def tilt(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zr = u / sc
        z = zr.clamp(-tilt_max, tilt_max)
        t = F.softmax(torch.where(legal, log_p + z, neg), dim=-1)
        return t, legal & (zr.abs() < tilt_max)

    def stein(t: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
        # E[sum_a (t_a − p_a) eps_a] = sum_ab C_ab dt_a/dy_b with
        # dt_a/dy_b = (gamma / scale)(t_a d_ab − t_a t_b) [b unclipped]
        # and C = s2 (I − 11'/k): s2 (gamma/scale) (tr J − 1'J1 / k).
        tf = t * free
        tr = (tf - t * tf).sum(dim=1)
        tot = tf.sum(dim=1) - t.sum(dim=1) * tf.sum(dim=1)
        return s2 * (g / sc).squeeze(1) * (tr - tot / k)

    def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * (a.clamp(min=1e-12).log() - b.clamp(min=1e-12).log())).sum(dim=1)

    t_full, free_full = tilt(g * y + (1 - g) * m)
    t_s, free_s = tilt(g * y)
    t_m, _ = tilt((1 - g) * m)
    stein_full, stein_s = stein(t_full, free_full), stein(t_s, free_s)
    gain_full = ((t_full - prior) * y).sum(dim=1) - stein_full
    gain_s = ((t_s - prior) * y).sum(dim=1) - stein_s
    gain_m = ((t_m - prior) * y).sum(dim=1)
    claim_m = ((t_m - prior) * m).sum(dim=1)
    marginal = gain_full - gain_s
    kl_prior, kl_extra = kl(t_full, prior), kl(t_full, t_s)
    top_s = torch.where(legal, t_s, neg).argmax(dim=1)
    top_f = torch.where(legal, t_full, neg).argmax(dim=1)
    rev = top_s != top_f
    label_gap = y.gather(1, top_s[:, None])[:, 0] - y.gather(1, top_f[:, None])[:, 0]
    rev_strong = rev & (label_gap > 2 * torch.sqrt(2 * s2))

    p_top = torch.where(legal, prior, neg).argmax(dim=1)
    p_top_prob = prior.gather(1, p_top[:, None])[:, 0]
    m_top = torch.where(legal, m, neg).argmax(dim=1)
    resid_top = (y - m).gather(1, p_top[:, None])[:, 0]
    lost_full = p_top_prob - t_full.gather(1, p_top[:, None])[:, 0]
    lost_s = p_top_prob - t_s.gather(1, p_top[:, None])[:, 0]
    over = m_top != p_top
    claimed = m.gather(1, m_top[:, None])[:, 0] - m.gather(1, p_top[:, None])[:, 0]
    shown = y.gather(1, m_top[:, None])[:, 0] - y.gather(1, p_top[:, None])[:, 0]

    def push(sel: torch.Tensor) -> dict:
        n = int(sel.sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "kl_target_prior": float(kl_prior[sel].mean()),
            "kl_extra": float(kl_extra[sel].mean()),
            "model_gain": _mean_z(gain_m[sel]),
            "model_claim": float(claim_m[sel].mean()),
            "marginal": _mean_z(marginal[sel]),
            "full": _mean_z(gain_full[sel]),
            "search": _mean_z(gain_s[sel]),
            "stein": {
                "full": float(stein_full[sel].mean()),
                "search": float(stein_s[sel].mean()),
            },
            "reversed": float(rev[sel].float().mean()),
            "reversed_vs_evidence": float(rev_strong[sel].float().mean()),
            "prior_top_lost": {
                "target": float(lost_full[sel].mean()),
                "search": float(lost_s[sel].mean()),
            },
        }

    def support(sel: torch.Tensor) -> dict:
        sel = sel & over
        n = int(sel.sum())
        if n == 0:
            return {"n": 0}
        c = float(claimed[sel].mean())
        s = _mean_z(shown[sel])
        return {
            "n": n,
            "claimed": c,
            "shown": s,
            "ratio": s["mean"] / c if c > 0 else None,
        }

    every = torch.ones_like(rev)
    confident = p_top_prob >= PRIOR_CONFIDENT
    classes: dict[str, list[int]] = defaultdict(list)
    for j, r in enumerate(idx.tolist()):
        classes[table.node_class[r]].append(j)
    per_class = {}
    for cls, js in sorted(classes.items()):
        sel = torch.zeros_like(rev)
        sel[torch.tensor(js)] = True
        per_class[cls] = {
            "top_card_residual": {"n": len(js), **_mean_z(resid_top[sel])},
            "override_support": support(sel),
            "push": push(sel),
        }
    bins = [*PRIOR_TOP_BINS, 1.0 + 1e-6]
    by_prior = {
        f"[{lo},{min(hi, 1.0)}]": push((p_top_prob >= lo) & (p_top_prob < hi))
        for lo, hi in zip(bins[:-1], bins[1:])
    }
    by_extra = {}
    if len(idx):
        cuts = kl_extra.quantile(torch.tensor(EXTRA_KL_QUANTILES, dtype=kl_extra.dtype))
        edges = [float("-inf"), *cuts.tolist(), float("inf")]
        labels = [
            "<q50",
            *[
                f"q{int(100 * a)}-q{int(100 * b)}"
                for a, b in zip(EXTRA_KL_QUANTILES[:-1], EXTRA_KL_QUANTILES[1:])
            ],
            f">=q{int(100 * EXTRA_KL_QUANTILES[-1])}",
        ]
        for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
            entry = push((kl_extra >= lo) & (kl_extra < hi))
            entry["kl_extra_from"] = lo if math.isfinite(lo) else 0.0
            by_extra[lab] = entry
    # The top bin by class: where the model's largest moves sit (§8.3).
    tail_by_class = {}
    if len(idx):
        tail = kl_extra >= float(
            kl_extra.quantile(
                torch.tensor(EXTRA_KL_QUANTILES[-1], dtype=kl_extra.dtype)
            )
        )
        for cls, js in sorted(classes.items()):
            sel = torch.zeros_like(rev)
            sel[torch.tensor(js)] = True
            if bool((sel & tail).any()):
                tail_by_class[cls] = push(sel & tail)
    gated = [
        c
        for c, d in per_class.items()
        if d["top_card_residual"]["z"] is not None
        and abs(d["top_card_residual"]["z"]) >= 3.0
    ]
    return {
        "units": "advantage (score / RETURN_SCALE)",
        "a_model_out_of_fold": out_of_fold,
        "rows": int(len(idx)),
        "prior_confident": PRIOR_CONFIDENT,
        "push": push(every),
        "push_by_prior_top": by_prior,
        "push_by_extra_kl": by_extra,
        "push_top_extra_by_class": tail_by_class,
        "override_support": support(every),
        "override_support_prior_confident": support(confident),
        "top_card_residual_classes_abs_z_ge_3": sorted(gated),
        "per_class": per_class,
    }


def format_model_diagnostics(diag: dict, points: float) -> list[str]:
    """Log lines for ``model_push_diagnostics``; values times ``points``
    (RETURN_SCALE: game points per decision)."""

    def v(d: dict) -> str:
        if d.get("mean") is None:
            return "n/a"
        z = d.get("z")
        return f"{points * d['mean']:+.4f}" + (
            f" (z {z:+.1f})" if z is not None else ""
        )

    def push(name: str, p: dict) -> str:
        if not p.get("n"):
            return f"    {name:28s} n=0"
        return (
            f"    {name:28s} n={p['n']:6d} KL {p['kl_target_prior']:.3f} "
            f"model-extra {p['kl_extra']:.3f} | model-only {v(p['model_gain'])} "
            f"claims {points * p['model_claim']:+.4f} | marginal {v(p['marginal'])} "
            f"of target {v(p['full'])} | top card changed "
            f"{100 * p['reversed']:.1f}% vs >2SE label {100 * p['reversed_vs_evidence']:.2f}%"
        )

    def support(name: str, s: dict) -> str:
        if not s.get("n"):
            return f"    {name}: none"
        ratio = "n/a" if s["ratio"] is None else f"{s['ratio']:+.2f}"
        return (
            f"    {name}: n={s['n']} claimed {points * s['claimed']:+.4f} "
            f"labels {v(s['shown'])} ratio {ratio}"
        )

    oof = (
        "out-of-fold"
        if diag["a_model_out_of_fold"]
        else "IN-SAMPLE (model-only biased up)"
    )
    lines = [
        f"[model] model push vs the search labels ({diag['rows']} rows, a_model "
        f"{oof}; game points per decision):",
        push("all", diag["push"]),
    ]
    lines += [push(f"theta_k top {k}", p) for k, p in diag["push_by_prior_top"].items()]
    lines += [push(f"model-extra {k}", p) for k, p in diag["push_by_extra_kl"].items()]
    lines += [
        "[model] overrides (a_model's top card != theta_k's):",
        support("all", diag["override_support"]),
        support(
            f"theta_k >= {diag['prior_confident']}",
            diag["override_support_prior_confident"],
        ),
    ]
    gated = diag["top_card_residual_classes_abs_z_ge_3"]
    lines.append(
        f"[model] top-card residual |z| >= 3 in {len(gated)} of "
        f"{len(diag['per_class'])} classes"
    )
    for cls in sorted(diag["per_class"]):
        d = diag["per_class"][cls]
        if d["push"].get("n", 0) >= 100:
            lines.append(
                push(cls, d["push"]) + f" | top-card resid {v(d['top_card_residual'])}"
            )
    return lines


def targets_for_table(
    model: AdvantageModel,
    table: RowTable,
    *,
    sigma_u2: "float | torch.Tensor",
    kappa: float,
    tilt_max: float,
    batch_rows: int = 2048,
    node_variance: bool = False,
    oof: "tuple[torch.Tensor, torch.Tensor | None] | None" = None,
    eta: float | None = None,
) -> dict:
    """Stage 1b + 2 over every targetable row. Returns dense tensors
    (targets, z, gamma, v_post, a_hat, a_model) aligned with ``table``.
    ``sigma_u2``: one value, or a per-row tensor aligned with ``table``;
    ``node_variance`` uses the heteroscedastic head's per-row sigma_u^2
    instead (ignored for a homoscedastic model); ``oof`` supplies
    out-of-fold a_model / per-row sigma_u^2 in place of the model's;
    ``eta`` is the global value-unit tilt temperature (build_tilt_target)."""
    su2_rows = (
        sigma_u2
        if isinstance(sigma_u2, torch.Tensor)
        else torch.full((len(table),), float(sigma_u2))
    )
    model.eval()
    out_t, out_z, out_g, out_v, out_a, out_m = [], [], [], [], [], []
    with torch.no_grad():
        for b in _batches(
            torch.arange(len(table)), batch_rows, False, random.Random(0)
        ):
            legal = table.masks[b].bool()
            if oof is not None:
                a_model = oof[0][b]
                su2_head = oof[1][b] if oof[1] is not None else None
            else:
                a_model, su2_head = predict_rows(model, table, b)
            su2_b = (
                su2_head if (su2_head is not None and node_variance) else su2_rows[b]
            )
            a_hat, v_post, gamma = blend_advantages(
                table.advantage[b],
                a_model,
                table.has_q[b],
                table.noise_var[b],
                su2_b,
            )
            t, z = build_tilt_target(
                table.prior[b],
                a_hat,
                v_post,
                legal,
                kappa=kappa,
                tilt_max=tilt_max,
                eta=eta,
            )
            out_t.append(t)
            out_z.append(z)
            out_g.append(gamma)
            out_v.append(v_post)
            out_a.append(a_hat)
            out_m.append(a_model)
    return {
        "target": torch.cat(out_t),
        "z": torch.cat(out_z),
        "gamma": torch.cat(out_g),
        "v_post": torch.cat(out_v),
        "a_hat": torch.cat(out_a),
        "a_model": torch.cat(out_m),
    }
