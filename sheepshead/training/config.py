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

    The gate: run the same cheap search ``gate_replicates`` times with
    independent RNG and emit a distillation target only when
    >= ``gate_agreement`` replicates pick the SAME action and it differs
    from the policy's greedy choice (the search root prior's argmax).
    Certification: gated labels measured +0.0112 mean uplift with 0/22 harm
    vs +0.0010 / 11% harm ungated (E9 §8).
    Literature: the loop is Expert Iteration (Anthony et al. 2017) with
    AlphaZero-style search targets on on-policy states (Silver et al.
    2017; DAgger, Ross et al. 2011); the readout and its small-budget
    policy-improvement guarantee are Gumbel MuZero's completed-Q
    (Danihelka et al. 2022; Grill et al. 2020 for the regularized-PI view);
    replicate averaging is root parallelization (Chaslot et al. 2008); the
    agreement gate is query-by-committee data selection (Seung et al. 1992)
    with the committee = independent stochastic runs of one searcher.

    A legacy per-head-fraction ExIt scheduler (dense forward-KL visit
    targets, ``head_search_fractions``/``t_full``/``d_short``) lived here
    until 2026-08; it was removed after the gated margin form superseded it
    (Search_Teacher_Design §10, Exit_Arms_202606).
    """

    enabled: bool = True
    gate_iters: int = 1024  # calibrated budget; changing it voids the E9 cert
    gate_d_rollout: int = 1  # shallow + oracle leaves (variance-min; E9 §7)
    gate_replicates: int = 3
    gate_agreement: int = 2  # strict exact-action 2-of-3 (E9 §8.2)
    gate_node_prob: float = 0.02  # subsample of eligible nodes (budget knob)
    # Label form. "agreed_onehot" (default): 1-eps on the committee's agreed
    # action, eps spread over the node's other legal actions — the exact
    # semantics the E9 certification validated (+0.0112 uplift was for the
    # AGREED ACTION). "avg_gumbel" (the original §9 design) is retained for
    # study only: at near-tie nodes the completed-Q soft target is close to
    # uniform, so forward-KL toward it is an entropy-injection term with a
    # label-count-independent loss scale — observed to flatten the play head
    # globally within ~25k episodes (branch run attempt 3, 2026-08-12).
    gate_target: str = "agreed_onehot"
    gate_target_smooth: float = 0.05
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
