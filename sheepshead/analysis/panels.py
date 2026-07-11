"""Frozen evaluation panels shared by orchestrators and instruments.

PANEL-A membership, ordering, and deal budgets are FROZEN (pre-registered
2026-07; see notebooks/Architecture_Ablation_202607.md). Do not edit without
an explicit pre-registration update — every recorded panel number depends on
this exact anchor set. Paths are repo-root-relative: run panel commands from
the repository root.
"""

PANEL_A = [
    "final_pfsp_swish_ppo.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt",
    "runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt",
    "runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt",
]
