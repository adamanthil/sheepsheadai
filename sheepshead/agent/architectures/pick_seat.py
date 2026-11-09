"""Zero-initialized seat residual for the pick decision.

``perceiver-recall-pickseat`` surgery onto a trained seat-blind trunk: the
frozen trunk keeps seeing ``leader_rel = 0`` before play, and this module
adds the seat directly at the pick head and limited critic. Both of its
output paths start at zero, so a migrated checkpoint is bit-identical until
training moves it.
"""

import torch
import torch.nn as nn

# 0 = not a pick decision; 1..10 = (leader_rel - 1) * 2 + partner_mode + 1.
PICK_CONTEXT_COUNT = 11


class PickSeatResidual(nn.Module):
    """Per-(seat, partner mode) offset plus a small MLP over the head's
    features and the context, giving one scalar per row. Rows that are not
    pick decisions (context 0) get exactly zero."""

    def __init__(self, d_input: int, d_context: int = 16, hidden: int = 64):
        super().__init__()
        self.offset = nn.Embedding(PICK_CONTEXT_COUNT, 1)
        self.context = nn.Embedding(PICK_CONTEXT_COUNT, d_context)
        first = nn.Linear(d_input + d_context, hidden)
        last = nn.Linear(hidden, 1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_input + d_context), first, nn.SiLU(), last
        )
        nn.init.zeros_(self.offset.weight)
        nn.init.orthogonal_(first.weight, gain=1.0)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, feat: torch.Tensor, pick_context: torch.Tensor) -> torch.Tensor:
        """feat: (..., d_input); pick_context: (...) long. Returns (...)."""
        offset = self.offset(pick_context).squeeze(-1)
        joined = torch.cat([feat, self.context(pick_context)], dim=-1)
        return (offset + self.mlp(joined).squeeze(-1)) * (pick_context > 0)
