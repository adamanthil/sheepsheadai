"""Model metadata and card-embedding geometry for the analyze UI.

Everything here is static per checkpoint, so results are cached on
(path, mtime) the same way ai_loader caches the checkpoint dict.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np

from server.api.schemas import (
    AnalyzeCardEmbeddingEntry,
    AnalyzeCardEmbeddings,
    AnalyzeModelResponse,
)
from server.config import get_settings
from server.services.ai_loader import load_agent
from sheepshead import DECK_IDS
from sheepshead.game import UNDER_CARD_ID, UNDER_TOKEN

_ID_TO_CARD = {v: k for k, v in DECK_IDS.items()}
_ID_TO_CARD[UNDER_CARD_ID] = UNDER_TOKEN


def _build_card_embeddings(agent) -> Optional[AnalyzeCardEmbeddings]:
    card_table = getattr(agent.encoder, "card", None)
    if card_table is None:  # onehot-ff has no card-embedding table
        return None

    weight = card_table.weight.detach().cpu().numpy()
    vectors = weight[1:]  # drop the padding row; rows 1..33 = cards + UNDER
    ids = list(range(1, vectors.shape[0] + 1))

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-8, None)
    cosine_sim = normed @ normed.T

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = singular_values**2
    explained = variance[:2] / max(float(variance.sum()), 1e-12)

    return AnalyzeCardEmbeddings(
        dims=int(vectors.shape[1]),
        cards=[
            AnalyzeCardEmbeddingEntry(
                id=card_id,
                card=_ID_TO_CARD[card_id],
                vector=[float(x) for x in vectors[i]],
            )
            for i, card_id in enumerate(ids)
        ],
        cosineSim=[[float(x) for x in row] for row in cosine_sim],
        pcaCoords=[[float(x), float(y)] for x, y in coords],
        pcaExplainedVariance=[float(x) for x in explained],
    )


@lru_cache(maxsize=2)
def _model_info(model_path: str, mtime: float, label: str) -> AnalyzeModelResponse:
    agent = load_agent(model_path)
    return AnalyzeModelResponse(
        modelLabel=label,
        arch=agent.arch_name,
        criticMode=getattr(agent, "critic_mode", "limited"),
        hasAuxHeads=bool(agent.critic.has_aux_heads),
        hasOracle=getattr(agent, "oracle_critic", None) is not None,
        gamma=float(agent.gamma),
        cardEmbeddings=_build_card_embeddings(agent),
    )


def get_model_info() -> AnalyzeModelResponse:
    settings = get_settings()
    path = settings.sheepshead_model_path
    return _model_info(
        path, os.path.getmtime(path), settings.sheepshead_model_label
    )
