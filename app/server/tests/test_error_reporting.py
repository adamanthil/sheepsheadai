"""Errors the server survives still reach Sentry when a DSN is configured."""

from __future__ import annotations

import logging

import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport
from sentry_sdk.types import Event

from server.config import Settings
from server.error_reporting import init_error_reporting


class _Capture(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        for item in envelope.items:
            if item.type == "event":
                event = item.get_event()
                assert event is not None
                self.events.append(event)


def _settings(**overrides) -> Settings:
    return Settings(
        sheepshead_model_path="unused",
        sheepshead_model_label="test-model",
        database_url="postgresql://unused",
        **overrides,
    )


@pytest.fixture
def capture():
    transport = _Capture()
    yield transport
    sentry_sdk.init()  # back to a disabled client for the rest of the suite


def test_no_dsn_leaves_reporting_off():
    assert init_error_reporting(_settings()) is False


def test_a_logged_exception_is_reported(capture):
    started = init_error_reporting(
        _settings(
            sentry_dsn="https://key@sentry.invalid/1",
            env="production",
            sentry_release="abc123",
        )
    )
    assert started
    sentry_sdk.get_client().transport = capture

    try:
        raise ValueError("persist failed")
    except ValueError:
        logging.getLogger("server.test").exception("persist_trick_completed failed")
    logging.getLogger("server.test").warning("only a warning")
    sentry_sdk.flush()

    (event,) = capture.events
    assert event["environment"] == "production"
    assert event["release"] == "abc123"
    assert event["exception"]["values"][-1]["type"] == "ValueError"
    assert "user" not in event or "ip_address" not in event["user"]
