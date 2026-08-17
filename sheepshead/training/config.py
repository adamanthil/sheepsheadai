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
* The gated search-teacher path (``pfsp_runtime.play_population_game`` +
  ``ismcts.py``) reads ``SearchConfig`` for node eligibility and the
  committee-gate knobs; the league trainer builds one when ``--search-teacher``
  is on (terminal-reward mode only).

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
    """Agreement-gated ISMCTS teacher SCHEDULING (Search_Teacher_Design §9):
    which decisions get searched and how the committee gate decides whether a
    label is emitted. The engine physics (PUCT constants, belief pool,
    batching, leaf/readout choices) live in ``sheepshead.ismcts.ISMCTSConfig``
    — the split is deliberate: the trainer owns coverage and the gate, the
    engine owns one search.

    The gate (resolved-pair emission, Search_Teacher_Design §12.7/§12.8):
    run the same cheap search ``gate_replicates`` times with independent
    RNG, collect each replicate's completed-Q table, and emit PAIRWISE
    ordering constraints a>b only where the committee statistically
    resolves them: the per-replicate paired Q-difference is
    sign-consistent across ALL replicates AND its mean clears
    max(gate_pair_eps, gate_pair_z * s / sqrt(R)). Pairs the live policy
    already satisfies by the teaching margin are not emitted
    (self-retirement); near-tied actions never produce a constraint at
    all — the §12.8 study measured true top gaps (0.004-0.007 Q) AT the
    noise floor of any affordable budget, so abstention on ties is the
    designed common case and the student's relative mass over unresolved
    sets is left to PG + the entropy bonus (max-ent completion).
    Literature: the loop is Expert Iteration (Anthony et al. 2017) on
    on-policy states (DAgger, Ross et al. 2011) with a FROZEN per-
    generation expert (§12.2); pairwise constraints instead of
    distribution targets are preference learning (Bradley-Terry;
    Christiano et al. 2017; the anchored pair-gap is DPO's implicit
    reward, Rafailov et al. 2023); emit-only-when-resolved is racing /
    best-arm elimination (Maron & Moore 1994); the Q tables are Gumbel
    MuZero's completed-Q (Danihelka et al. 2022).

    A legacy per-head-fraction ExIt scheduler (dense forward-KL visit
    targets) was removed 2026-08 (§10/§11); the exact-card 2-of-3
    one-hot gate (``gate_agreement``/``gate_target``) was removed
    2026-08-16 after §12.8 showed its labels ~50% non-reproducible at
    defender leads (the attempt-8 incumbent-tax mechanism, §12.6).
    """

    enabled: bool = True
    gate_iters: int = 1024  # calibrated budget (E9; §12.8 re-validated cheap)
    gate_d_rollout: int = 1  # shallow + oracle leaves (variance-min; E9 §7)
    gate_replicates: int = 5  # R: committee size; replicates beat iterations
    gate_node_prob: float = 0.02  # subsample of eligible nodes (budget knob)
    # Resolved-pair emission rule (§12.7; constants calibrated §12.8: per-
    # node paired-diff SE ~0.006 at 1024/1, so z=2 with eps floor = harm_eps).
    gate_pair_eps: float = 0.01  # Q-units floor (E9 harm epsilon)
    gate_pair_z: float = 2.0  # required mean/SE ratio
    gate_max_pairs: int = 8  # strongest-evidence cap per node (t-stat order)
    # Teaching filter: a resolved pair is emitted only if the LIVE policy's
    # log-prob gap log pi(a) - log pi(b) is below this margin (mirror of the
    # loss margin, wired from --search-teacher-margin). Satisfied pairs are
    # counted for self-retirement telemetry but not emitted.
    gate_emit_margin: float = 0.3
    # Node classes searched at all (trick x role x lead/follow). From the E9
    # matrix: every play cell whose mean headroom was >= ~0.003 Q — the gate
    # supplies per-node reliability, the cell set only excludes classes where
    # search would confirm the policy at pure cost. t5 has no decisions
    # (forced card); leaster / alone games are ineligible upstream.
    gate_cells: frozenset = frozenset(
        {
            "t0-defender-follow",
            "t0-defender-lead",
            "t0-partner-follow",
            "t0-picker-follow",
            "t0-picker-lead",
            "t1-defender-lead",
            "t1-partner-follow",
            "t1-partner-lead",
            "t1-picker-follow",
            "t1-picker-lead",
            "t2-defender-follow",
            "t2-defender-lead",
            "t2-partner-follow",
            "t2-partner-lead",
            "t2-picker-follow",
            "t2-picker-lead",
            "t3-defender-lead",
            "t3-partner-follow",
            "t3-picker-follow",
            "t4-defender-follow",
            "t4-partner-follow",
            "t4-partner-lead",
            "t4-picker-lead",
        }
    )


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
