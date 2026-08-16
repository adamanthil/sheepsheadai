#!/usr/bin/env python3
"""Central training hyperparameters.

Consumers:

* ``train_league_ppo.py`` reads ``PFSPHyperparams`` (instantiated once as
  ``PFSP_HYPERPARAMS``) for the entropy + learning-rate decay schedules and the
  greedy-health collapse gates. Everything else the league trainer needs is a
  per-run CLI flag (workers, anchor, eval cadence), not a tuning constant.
* ``train_selfplay_ppo.py`` reads ``SelfPlayHyperparams`` (instantiated once as
  ``SELFPLAY_HYPERPARAMS``) for the bootstrap run's fixed learning rates and
  entropy schedule. Its values intentionally differ from the league trainer's,
  hence a separate dataclass.
* The CE search-teacher path (``pfsp_runtime.play_population_game`` +
  ``ismcts.py``) reads ``SearchConfig`` for node eligibility, the committee
  budget, and the shrinkage constants; the league trainer builds one when
  ``--teacher`` is on (terminal-reward mode only).

The shaped-reward controllers, opponent-block scheduling, and the standalone
ExIt trainer that this module used to configure were removed in the June 2026
league consolidation.
"""

from dataclasses import dataclass, field


@dataclass
class PFSPHyperparams:
    """League-trainer schedules and collapse gates (see module docstring)."""

    # Entropy schedules (start -> end), decayed linearly over the schedule horizon.
    entropy_pick_start: float = 0.05
    entropy_pick_end: float = 0.005
    entropy_partner_start: float = 0.05
    entropy_partner_end: float = 0.005
    entropy_bury_start: float = 0.04
    entropy_bury_end: float = 0.002
    entropy_play_start: float = 0.015
    entropy_play_end: float = 0.001

    # Learning rate schedules (percent progress -> learning rate).
    lr_schedule_actor: dict[int, float] = field(
        default_factory=lambda: {0: 1.5e-4, 100: 5e-5}
    )
    lr_schedule_critic: dict[int, float] = field(
        default_factory=lambda: {0: 1.5e-4, 100: 5e-5}
    )

    # Greedy self-play health gates (collapse guard; percent units except the
    # play-head logit spread). Stochastic training-time rates masked the run-2
    # collapse for 586k episodes: a flattened policy still *samples* ~30% PICK
    # while its argmax is PASS. The greedy probe (training_utils.greedy_health_probe)
    # plays argmax self-play and warns when any rate crosses these gates.
    greedy_gate_min_pick: float = 15.0
    # 20% ALONE (of partner decisions) can still be clean play; much above
    # that usually means weak defender-field collaboration, which league
    # training itself should repair. The extended-league orchestrator
    # additionally applies this gate relative to the resume checkpoint's own
    # baseline (max(gate, baseline + margin)) so a high-alone warm start
    # doesn't trip it while regression still does.
    greedy_gate_max_alone: float = 20.0
    greedy_gate_max_trump_lead: float = 8.0
    greedy_gate_min_play_spread: float = 0.5


@dataclass
class SelfPlayHyperparams:
    """Bootstrap self-play trainer (``train_selfplay_ppo.py``) schedule.

    This trainer produces the ~100k-episode seed model that warm-starts league
    training so the league need not bootstrap from scratch. Its fixed learning
    rates and entropy schedule intentionally differ from the league trainer's
    ``PFSPHyperparams`` (higher / flatter exploration suited to a from-scratch
    run), so the two are kept as separate dataclasses rather than shared values.
    """

    # Fixed learning rates (constant over the bootstrap run; no schedule).
    lr_actor: float = 1.0e-4
    lr_critic: float = 1.0e-4

    # Entropy schedules (start -> end), decayed linearly over the run length.
    entropy_pick_start: float = 0.08
    entropy_pick_end: float = 0.05
    entropy_partner_start: float = 0.05
    entropy_partner_end: float = 0.04
    entropy_bury_start: float = 0.04
    entropy_bury_end: float = 0.03
    entropy_play_start: float = 0.05
    entropy_play_end: float = 0.05


@dataclass
class SearchConfig:
    """CE search-teacher SCHEDULING (CE_Teacher_Design §1-§2): which
    decisions get searched, the committee budget, and the shrinkage noise
    model that turns committee Q tables into CE targets. The engine physics
    (PUCT constants, belief pool, batching, leaf/readout choices) live in
    ``sheepshead.ismcts.ISMCTSConfig`` — the split is deliberate: the
    trainer owns coverage and the target construction, the engine owns one
    search.

    Emission is CLASS-BLIND (no cell taxonomy, no confidence trigger —
    §13.3: a top-2-gap trigger captures only ~35% of policy-wrong t0
    called-suit nodes) and abstention lives in the TARGET, not a gate: the
    committee's completed-Q vector is James-Stein-shrunk toward flat by the
    replicate noise model, and a flat shrunk vector reproduces the expert's
    label-time prior — near-zero CE gradient by construction. See
    ``pfsp_runtime.build_ce_search_target`` for the construction and
    CE_Teacher_Design §1.1-§1.2 for the properties each piece replaces
    (ε-gates, pair emission, incumbent tax).

    Literature: the loop is Expert Iteration (Anthony et al. 2017) on
    on-policy states (DAgger, Ross et al. 2011) with a FROZEN per-
    generation expert; the target is Gumbel MuZero's completed-Q policy
    improvement (Danihelka et al. 2022; Grill et al. 2020); the shrinkage
    is positive-part James-Stein with a hierarchical variance blend.

    The resolved-pair hinge gate this replaces (gate_pair_* / gate_cells /
    gate_emit_margin) was removed 2026-08 with the §12 program (attempts
    5a-10, all retired); git tag ``pre-ce-teacher`` archives it.
    """

    enabled: bool = True
    teacher_iters: int = 1024  # calibrated budget (E9; §12.8 re-validated cheap)
    teacher_d_rollout: int = 1  # shallow + oracle leaves (variance-min; E9 §7)
    teacher_replicates: int = 3  # R: committee size (lockstep search_committee)
    teacher_prob: float = 0.1  # subsample of eligible nodes (the budget knob)
    # Shrinkage noise model (CE_Teacher_Design §1.2): per-action replicate
    # variance at R=3 has 2 dof, so blend it with a global replicate-noise
    # calibration:  s2_a <- (nu*s2_global + (n_obs-1)*s2_node_a)/(nu + n_obs - 1).
    shrink_nu: float = 4.0
    # Per-replicate per-action Q variance at the 1024/1 budget, measured by
    # the §1.2 calibration gate (analysis/calibrate_shrinkage.py on the
    # archived §12.8 deflead gating study: 144 nodes x 6 reps, pooled mean
    # over 720 action cells; per-action per-replicate SD ~0.026 Q). See
    # CE_Teacher_Design §10 for the recorded gate results.
    shrink_s2_global: float = 6.95e-4


@dataclass
class LeagueConfig:
    """Knobs for roster management and table sampling (plan §3.3/§8).

    Consumed by ``league.League`` (roster management + table sampling) and
    ``exploiter.py`` (frozen-main league construction). The generation
    schedule and gate thresholds are per-run CLI flags on the trainers.
    """

    max_past_mains: int = 30
    hof_quota: int = 6
    protect_newest: int = 5  # newest past_mains immune to skill pruning
    # Exploiter seat share: cap * clip(max_active_gate_edge / edge_full, 0, 1).
    # Driven by the FROZEN gate edge (settlement score/deal), not the live binary
    # EMA, so it can't ratchet to zero when the table EMA dips below neutral.
    exploiter_seat_cap: float = 0.30
    exploiter_edge_full: float = 0.30  # settlement score/deal that earns the full cap
    self_play_share: float = 0.15
    hof_floor_prob: float = 0.05  # chance a PFSP seat is forced to a HOF anchor
    # PFSP win-rate curriculum over past mains (kept from the old design —
    # the principled part). x = exploitation EMA.
    pfsp_variable_weight: float = 0.7
    pfsp_hard_weight: float = 0.3
    pfsp_hard_power: float = 2.0
    pfsp_uniform_mix: float = 0.1
    pfsp_conf_scale: float = 5.0
    # Exploiter retirement: demote to past_main purely on age. Guarantees every
    # inserted exploiter exploiter_retire_generations of seat time (the floor).
    exploiter_retire_generations: int = 3
    # Exploit-patched retirement: demote an exploiter to past_main once its
    # live outcome EMA vs the training agent shows the exploit no longer wins
    # (EMA below this with >= exploiter_patched_min_samples). Without it the
    # FROZEN gate-edge seat share keeps burning episodes for the full age
    # floor after the hero adapts. None = disabled (historical behavior).
    exploiter_patched_ema: float | None = None
    exploiter_patched_min_samples: int = 200
