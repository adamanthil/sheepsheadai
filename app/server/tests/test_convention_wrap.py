"""Convention-wrap plumbing (server.api.games.build_table_agent).

Hermetic: load_agent is stubbed; only the settings-driven wrap decision is
under test. The wrapper's own behavior is covered by
tests/test_convention_wrapper.py at repo root.
"""

from __future__ import annotations

import pytest

import server.api.games as games_module
from server.api.games import build_table_agent
from server.config import Settings
from sheepshead.agent.convention_wrapper import ConventionWrapper


class _StubAgent:
    def act(self, state, valid_actions, player_id, deterministic=True):
        return sorted(valid_actions)[0], None, None

    def observe(self, state, player_id):
        pass

    def reset_recurrent_state(self):
        pass


def _settings(wrap: str) -> Settings:
    return Settings(
        sheepshead_model_path="unused.pt",
        sheepshead_model_label="test",
        database_url="postgresql://unused",
        sheepshead_convention_wrap=wrap,
    )


@pytest.fixture
def stub_loader(monkeypatch):
    stub = _StubAgent()
    monkeypatch.setattr(games_module, "load_agent", lambda path: stub)
    return stub


def test_default_is_unwrapped(stub_loader):
    agent = build_table_agent(_settings(""), table_id="t1")
    assert agent is stub_loader
    assert not isinstance(agent, ConventionWrapper)


@pytest.mark.parametrize(
    "wrap,c1,c2", [("c1", True, False), ("c2", False, True), ("c1c2", True, True)]
)
def test_wrap_values_apply_the_mask(stub_loader, wrap, c1, c2):
    agent = build_table_agent(_settings(wrap), table_id="t1")
    assert isinstance(agent, ConventionWrapper)
    assert agent.c1 is c1
    assert agent.c2 is c2
    # The wrapped agent still exposes the loader's underlying instance.
    assert agent._agent is stub_loader


def test_unknown_wrap_fails_fast(stub_loader):
    with pytest.raises(ValueError):
        build_table_agent(_settings("c3"), table_id="t1")
