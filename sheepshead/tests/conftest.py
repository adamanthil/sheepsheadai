"""Training/research test suite for the sheepshead package.

Scope: game engine rules and invariants, agent architectures, league/PPO
training runtime, ISMCTS search, and analysis probes. Product/server tests
live in app/server/tests and run against the FastAPI app.

Conventions:
  * Long-running files carry a module-level ``pytestmark = pytest.mark.slow``
    (registered in pyproject.toml); ``-m "not slow"`` gives a fast local run.
  * Shared game-driving helpers live in ``sheepshead.tests.game_test_utils``.
  * Golden architecture fixtures live in ``fixtures/arch_golden/`` and are
    captured/checked by ``sheepshead.analysis.capture_arch_goldens``.
"""
