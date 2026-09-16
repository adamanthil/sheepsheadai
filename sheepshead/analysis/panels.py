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

# PANEL-B (TENTATIVE, 2026-09-16; Training_Program_Redesign §5.3): the
# strongest artifacts across two architectures and two training regimes —
# the production 30M, the perceiver-shared-v2 lineage's release (theta_3
# after its bidding phase), its iteration-1 checkpoint and its 8M league
# seed. PANEL-A's anchors are weak relative to current candidates, so their
# differences compress against it; PANEL-B keeps discriminating at strong
# skill and reads robustness across ecologies. RECORDED ONLY: every
# pre-registered bar stays on PANEL-A; a PANEL-B bar, if any, is defined
# from what the fresh run shows. Membership is provisional until the v2
# release's reads vs the 30M land.
PANEL_B = [
    "final_pfsp_swish_ppo.pt",
    "runs/rc_validate_v2/final/release.pt",
    "runs/policy_iteration_202609/iter11/distill_epoch7.pt",
    "runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt",
]
