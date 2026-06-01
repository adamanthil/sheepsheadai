#!/usr/bin/env python3
"""Unified training hyperparameters for the PFSP trainers.

All tunable knobs for the population-based trainers live here so the entry-point
scripts (``train_pfsp_ppo.py`` shaped baseline, ``train_pfsp_exit.py`` ISMCTS
hybrid) stay thin. ``PFSPHyperparams`` is shared by both; ``reward_mode`` and the
nested optional ``search`` select the strategy:

* shaped baseline:  reward_mode="shaped",  search=None   (uses shaping schedules +
  entropy bumps)
* ISMCTS hybrid:    reward_mode="terminal", search=SearchConfig(...)  (terminal-only
  return + search-distillation; shaping schedules / entropy bumps inactive)
"""

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """ISMCTS soft-teacher search controls (used only by the ExIt hybrid trainer).

    ``head_search_fractions`` is the per-head probability that a training-agent
    decision is searched. The bidding heads default to **1.0** and play to **0.10**: the
    bidding decisions are the most collapse-prone (always-pick / always-pass /
    always-call), so searching them every time is the principled replacement for
    the stripped shaping + epsilon-floor crutches, and they are cheap — shallow
    ``max_depth=1`` roots, at most a couple per game for the training agent —
    relative to the deep (``max_depth=6``) play tree, which only needs a thin
    correction for the trick-0 leak. Pre-pick (PICK / PASS) determinization is
    supported via ``Game._sample_prepick_deal`` (P4); PARTNER / BURY ride the
    post-pick determinizer (a picker exists); leasters via
    ``Game._sample_leaster_deal`` (no picker / called card / bury). Leaster PLAY
    decisions ARE searched (head "play", at the play frac): with the per-trick
    reward + leaster bonus gone, the pass->leaster branch the bidding EV rides on
    is only win-likelihood-driven if the agent plays leasters well, which needs a
    teacher signal there.

    ``t_full`` / ``d_short`` set the trick-indexed rollout-depth schedule: roll to
    (near) terminal for tricks ``0..t_full`` where the critic is blind to the
    trick-0 leak, then bootstrap ``d_short`` plies later once the value head is
    trustworthy. ``t_full=1`` / ``d_short=2`` are validated by the critic-calibration
    probe (``t_full_probe.py``): a search at trick ``t`` bootstraps at ~``t+d_short``,
    so this lands every bootstrap at trick >= 4, where the best-possible value head
    reaches R^2 ~0.73+ (vs ~0.26 at trick 0). The trick-0 defender-lead leak states
    are always rolled to terminal (0 <= t_full). Leasters are forced to terminal
    rollout in the runtime regardless of t_full (their outcomes barely calibrate,
    R^2 <= 0.21).

    ``searched_ppo_weight`` is the plan-§4 A/B knob for how searched transitions are
    trained: it is the weight on the PPO clip term for searched transitions (NOT a
    teacher weight). 0.0 = hard PG-mask (drop the PPO clip term; distillation owns
    those states) — the default; 1.0 = additive form (keep PPO AND add distillation);
    0<w<1 = a residual PPO weight (fallback if the hard mask proves too noisy at low
    ESS). Distillation toward pi' is applied on searched transitions either way at
    its own ``search_distill_coeff``; only the PPO term's weight on them changes here.
    Unsearched transitions always keep full PG.
    """

    head_search_fractions: dict = field(
        default_factory=lambda: {"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 0.10}
    )
    t_full: int = 1
    d_short: int = 2
    searched_ppo_weight: float = 0.0
    enabled: bool = True


@dataclass
class PFSPHyperparams:
    # Strategy selectors (shared trainer; see module docstring)
    reward_mode: str = "shaped"  # "shaped" | "terminal"
    search: SearchConfig | None = None

    # Parallel game-generation workers (Lever 1). None => auto: parallelize the
    # expensive ISMCTS ExIt generation (reward_mode="terminal") across
    # min(cpu_count-1, 8) workers, keep the cheap shaped PPO baseline sequential.
    # 1 (or <=1) forces the original in-process sequential loop.
    num_workers: int | None = None

    # Adaptive exploration for pick head (rate-based bump scheduling)
    low_pick_rate_threshold: float = 20.0  # percent
    high_pick_rate_threshold: float = 60.0  # percent
    pick_entropy_bump: float = 0.04  # added to base decayed pick entropy
    pick_entropy_bump_duration: int = 25000  # episodes

    # PASS-floor epsilon controller (shaped mode only).
    # Ensures minimum PASS probability on pick steps if picker average score is low.
    high_pick_rate_ceiling: float = (
        80.0  # Alter distribution to force PASS after this threshold
    )
    pass_floor_eps_base: float = 0.0
    pass_floor_eps_target: float = 0.08
    pass_floor_eps_step_up: float = 0.02
    pass_floor_eps_step_down: float = 0.02
    pass_floor_eps_picker_avg_threshold: float = -0.75

    # PICK-floor epsilon controller (shaped mode only).
    # Ensures minimum PICK probability on pick steps if overall pick rate is low.
    low_pick_rate_floor: float = 8.0  # percent
    pick_floor_eps_base: float = 0.0
    pick_floor_eps_target: float = 0.05
    pick_floor_eps_step_up: float = 0.02
    pick_floor_eps_step_down: float = 0.02

    # Adaptive exploration for partner head (ALONE decision; bump scheduling)
    low_alone_rate_threshold: float = 2.5  # percent
    high_alone_rate_threshold: float = 30.0  # percent
    partner_entropy_bump: float = 0.04  # added to base decayed partner entropy
    partner_entropy_bump_duration: int = 25000  # episodes

    # Partner CALL mixture epsilon controller (shaped mode only).
    # Probability floor over CALL actions when picker average score is low.
    high_alone_rate_ceiling: float = (
        60.0  # Alter distribution to force partner calls after this threshold
    )
    partner_call_eps_base: float = 0.0
    partner_call_eps_max_mid: float = (
        0.05  # when picker avg <= mid_picker_avg_threshold
    )
    partner_call_eps_mid_picker_avg_threshold: float = -0.75
    partner_call_eps_max_high: float = (
        0.10  # when picker avg <= high_picker_avg_threshold
    )
    partner_call_eps_high_picker_avg_threshold: float = -2
    partner_call_eps_step_up: float = 0.02
    partner_call_eps_step_down: float = 0.02

    # Adaptive exploration for bury head (bury decisions quality)
    # If bury_quality_rate drops below a threshold, temporarily bump bury entropy.
    low_bury_quality_threshold: float = 85.0  # percent
    bury_entropy_bump: float = 0.04  # added to base decayed bury entropy
    bury_entropy_bump_duration: int = 19000  # episodes

    # Entropy schedules (start -> end)
    entropy_pick_start: float = 0.05
    entropy_pick_end: float = 0.005
    entropy_partner_start: float = 0.05
    entropy_partner_end: float = 0.005
    entropy_bury_start: float = 0.04
    entropy_bury_end: float = 0.002
    entropy_play_start: float = 0.05
    entropy_play_end: float = 0.005

    # Shaped reward schedules (percent -> weight). Used only when reward_mode="shaped".
    shaping_schedule_pick: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 50: 1.0, 60: 0}
    )
    shaping_schedule_partner: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 50: 1.0, 55: 0}
    )
    shaping_schedule_bury: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 50: 1.0, 55: 0}
    )
    shaping_schedule_play: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 50: 1.0, 70: 0}
    )

    # Learning rate schedules (percent -> learning rate).
    lr_schedule_actor: dict[int, float] = field(
        default_factory=lambda: {0: 1.5e-4, 100: 5e-5}
    )
    lr_schedule_critic: dict[int, float] = field(
        default_factory=lambda: {0: 1.5e-4, 100: 5e-5}
    )

    # Opponent scheduling (PFSP mixture vs anchor/pressure/support specials)
    anchor_block_start_prob: float = 0.03
    anchor_block_len_min: int = 6
    anchor_block_len_max: int = 20
    anchor_slots_in_block: int = 3
    pressure_slot_prob: float = 0.12
    support_slot_prob: float = 0.06


DEFAULT_HYPERPARAMS = PFSPHyperparams()
