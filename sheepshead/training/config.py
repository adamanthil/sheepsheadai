#!/usr/bin/env python3
"""Central training hyperparameters (Training_Program_Redesign_202609 §4).

Consumers:

* ``train_ppo.py`` reads ``BootstrapHyperparams`` for the shaped self-play
  bootstrap (phase 0) and ``LeagueHyperparams`` for the terminal-only league
  phases (phase 2 and the bidding-only phase). Everything else the trainer
  needs is a per-run CLI flag (workers, cadence), not a tuning constant.
* ``league.League`` reads ``LeagueConfig`` for roster management and table
  sampling.
* ``distill_corpus.py`` and the search instruments read ``CommitteeConfig``
  for the committee budget and the shrinkage noise model that turns
  committee Q tables into corpus rows.
"""

from dataclasses import dataclass


@dataclass
class BootstrapHyperparams:
    """Phase 0: shaped self-play from scratch (the retired self-play trainer's
    validated settings — Architecture_Ablation_202607 regime).

    Fixed learning rates and a flat-ish per-head entropy schedule decayed
    linearly over the bootstrap's own length. Higher, flatter exploration
    than the league phases: a from-scratch run must escape the all-PASS
    attractor (Architecture_Ablation §4.5) and learn trick play under dense
    shaped rewards, and the update cadence is four times the league's.
    """

    lr_actor: float = 1.0e-4
    lr_critic: float = 1.0e-4
    update_interval: int = 4096  # transitions per PPO update
    ppo_epochs: int = 4
    minibatch_episodes: int = 256

    entropy_pick_start: float = 0.08
    entropy_pick_end: float = 0.05
    entropy_partner_start: float = 0.05
    entropy_partner_end: float = 0.04
    entropy_bury_start: float = 0.04
    entropy_bury_end: float = 0.03
    entropy_play_start: float = 0.05
    entropy_play_end: float = 0.05


@dataclass
class LeagueHyperparams:
    """Phase 2 (terminal-only league PG) and the bidding-only phase: the
    retention-run configuration (Learning_System_Redesign §7.9) with the
    clock schedules removed.

    Learning rate is CONSTANT: the historical 20M-episode decay never
    annealed inside a real run, and the lineage-best run trained at an
    effectively flat 1.4e-4 (Learning_System_Redesign §8.7).

    Entropy coefficients: generation 1 runs the fixed START values below
    (the legacy schedule's start, at which the retention run's gens 1-2 were
    trained — its decay was clock-based and inert); from generation 2 the
    target-entropy controller (entropy_controller.py) owns the coefficients,
    attaching bumplessly to the measured operating point.
    """

    lr_actor: float = 1.5e-4
    lr_critic: float = 1.5e-4
    update_interval: int = 16_384  # transitions per PPO update
    ppo_epochs: int = 4
    minibatch_episodes: int = 128  # with gradient accumulation: one step / epoch
    oracle_extra_epochs: int = 4

    entropy_pick: float = 0.05
    entropy_partner: float = 0.05
    entropy_bury: float = 0.04
    entropy_play: float = 0.015

    # Greedy self-play health gates (collapse guard; percent units except the
    # play-head logit spread). Stochastic training-time rates masked the run-2
    # collapse for 586k episodes: a flattened policy still *samples* ~30% PICK
    # while its argmax is PASS. The greedy probe (training_utils.greedy_health_probe)
    # plays argmax self-play and warns when any rate crosses these gates.
    greedy_gate_min_pick: float = 15.0
    # 20% ALONE (of partner decisions) can still be clean play; much above
    # that usually means weak defender-field collaboration, which league
    # training itself should repair. The orchestrator applies this gate
    # relative to the phase's starting checkpoint (max(gate, baseline +
    # margin)) so a high-alone warm start doesn't trip it while regression
    # still does.
    greedy_gate_max_alone: float = 20.0
    greedy_gate_max_trump_lead: float = 8.0
    greedy_gate_min_play_spread: float = 0.5


@dataclass
class CommitteeConfig:
    """The ISMCTS committee that labels corpus rows (CE_Teacher_Design
    §1.2, §20) and the shrinkage noise model built on its replicates. The
    engine physics (PUCT constants, belief pool, batching, leaf/readout
    choices) live in ``sheepshead.ismcts.ISMCTSConfig``; this owns one
    committee's budget and the noise calibration.

    Literature: the loop is Expert Iteration (Anthony et al. 2017) on the
    student's own states (DAgger, Ross et al. 2011) with a frozen expert
    per iteration; the shrinkage is positive-part James-Stein with a
    hierarchical variance blend.
    """

    iters: int = 1024  # calibrated budget (E9; §12.8 re-validated cheap)
    d_rollout: int = 1  # shallow + oracle leaves (variance-min; E9 §7)
    replicates: int = 3  # R: committee size (lockstep search_committee)
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
    """Roster management and table sampling (``league.League``).

    The validated per-seat mixture (Learning_System_Redesign §7.9, §9):
    each opponent seat is the current agent with ``self_play_share``, else a
    PFSP draw over past mains with a HOF floor. Exploiters left the loop
    in the September 2026 redesign (8/8 gates failed in the retention run;
    exploitability is a one-time post-hoc audit now).
    """

    max_past_mains: int = 30
    hof_quota: int = 6
    protect_newest: int = 5  # newest past_mains immune to skill pruning
    self_play_share: float = 0.15
    hof_floor_prob: float = 0.05  # chance a PFSP seat is forced to a HOF anchor
    # PFSP win-rate curriculum over past mains (Vinyals et al. 2019).
    # x = the member's EMA of P(outscores the training agent). Measured
    # near-uniform in practice (Learning_System_Redesign §2.2); kept because
    # it is validated and cheap.
    pfsp_variable_weight: float = 0.7
    pfsp_hard_weight: float = 0.3
    pfsp_hard_power: float = 2.0
    pfsp_uniform_mix: float = 0.1
    pfsp_conf_scale: float = 5.0
