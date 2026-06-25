#!/usr/bin/env python3
"""Central training hyperparameters.

Consumers:

* ``train_league_ppo.py`` reads ``PFSPHyperparams`` (instantiated once as
  ``_SCHED``) for the entropy + learning-rate decay schedules and the
  greedy-health collapse gates. Everything else the league trainer needs is a
  per-run CLI flag (workers, anchor, eval cadence), not a tuning constant.
* ``train_selfplay_ppo.py`` reads ``SelfPlayHyperparams`` (instantiated once as
  ``_HP``) for the bootstrap run's fixed learning rates and entropy schedule.
  Its values intentionally differ from the league trainer's, hence a separate
  dataclass.
* The deploy/audit ISMCTS search path (``pfsp_runtime.play_population_game`` +
  ``ismcts.py``) reads ``SearchConfig`` for the per-head search coverage and the
  rollout-depth schedule. The league/exploiter trainers run terminal-reward only
  with no teacher, so the search path is reachable only from the probes and the
  regression tests.

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
    greedy_gate_min_pick: float = 25.0
    greedy_gate_max_alone: float = 12.0
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
    """ISMCTS soft-teacher search controls (deploy/audit search path).

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
    """

    head_search_fractions: dict = field(
        default_factory=lambda: {"pick": 0, "partner": 0, "bury": 0, "play": 0.30}
    )
    t_full: int = 1
    d_short: int = 2
    enabled: bool = True
