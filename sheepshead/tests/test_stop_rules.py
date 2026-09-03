"""The program's decision rules (Training_Program_Redesign §5) on recorded
numbers, including the July retention run's actual h2h series."""

import pytest

from sheepshead.training.stop_rules import (
    HandoffRuleConfig,
    IterationRuleConfig,
    decide_handoff,
    gain_improving,
    generation_verdict,
    iteration_stop,
    settled_generation,
)

CFG = HandoffRuleConfig()

# runs/league_retention_pg/orchestrator/generations.csv: h2h edge / se per gen
RETENTION_H2H = [
    (0.0781, 0.0134),
    (0.1013, 0.0129),
    (-0.0046, 0.0130),
    (0.0011, 0.0133),
    (0.1069, 0.0136),
    (0.0178, 0.0105),
    (0.0192, 0.0107),
    (-0.0053, 0.0123),
]


def test_marginal_bar_on_the_retention_run():
    flags = [gain_improving(e, s, CFG) for e, s in RETENTION_H2H]
    assert flags == [True, True, False, False, True, False, False, False]


def test_verdict_confirmation_path():
    v = generation_verdict(3, -0.005, 0.013, CFG)
    assert not v.improving and v.needs_confirmation
    v = generation_verdict(3, 0.015, 0.012, CFG, confirm=(0.03, 0.012))
    assert v.improving and v.confirm_edge == 0.03
    v = generation_verdict(3, 0.015, 0.012, CFG, confirm=(0.01, 0.012))
    assert not v.improving and not v.needs_confirmation
    assert generation_verdict(1, 0.08, 0.013, CFG).improving


def test_handoff_sequence_on_the_retention_run():
    """Replaying the retention run: gens 1-2 continue, gen 3 (first failure
    at the floor) fires the step, gen 4 (second failure) hands off."""
    flags = [gain_improving(e, s, CFG) for e, s in RETENTION_H2H]
    step = None
    actions = []
    for g in range(1, 5):
        d = decide_handoff(flags[:g], g, step, CFG)
        actions.append(d.action)
        if d.action == "entropy_step":
            step = g
        if d.action == "handoff":
            break
    assert actions == ["continue", "continue", "entropy_step", "handoff"]
    assert settled_generation(4, step_generation=3) == 3  # pre-step boundary
    assert settled_generation(5, step_generation=3) == 5


def test_floor_and_cap():
    assert decide_handoff([False], 1, None, CFG).action == "continue"
    # A one-generation floor still cannot step before the controller exists.
    one = HandoffRuleConfig(min_generations=1)
    assert decide_handoff([False], 1, None, one).action == "continue"
    assert decide_handoff([False, False], 2, None, one).action == "entropy_step"
    assert decide_handoff([False, False], 2, None, CFG).action == "continue"
    assert decide_handoff([False] * 3, 3, None, CFG).action == "entropy_step"
    cap = HandoffRuleConfig(max_generations=3)
    assert decide_handoff([True] * 3, 3, None, cap).action == "handoff"
    # An improving generation after the step keeps going; the next failure
    # hands off.
    assert decide_handoff([False] * 3 + [True], 4, 3, CFG).action == "continue"
    assert decide_handoff([False] * 3 + [True, False], 5, 3, CFG).action == "handoff"


def test_history_length_must_match():
    with pytest.raises(ValueError):
        decide_handoff([True], 2, None, CFG)


def test_iteration_stop():
    cfg = IterationRuleConfig()
    assert iteration_stop([(0.026, 0.007)], cfg) == (False, "gain still resolvable")
    assert iteration_stop([(0.026, 0.007), (0.005, 0.007)], cfg)[0] is False
    assert iteration_stop([(0.005, 0.007), (0.004, 0.007)], cfg)[0] is True
    assert iteration_stop([(0.03, 0.007)] * 5, cfg)[0] is True
