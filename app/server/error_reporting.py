"""Ship server errors to Sentry when SENTRY_DSN is set.

With no DSN (development, tests, CI) nothing is initialised and nothing
leaves the process. With one, three paths report:

* Unhandled exceptions in HTTP and websocket handlers (the FastAPI and
  Starlette integrations, enabled automatically for those packages).
* Every ``logging.error``/``logging.exception`` record. The server's rule is
  that a caught failure it chooses to survive is logged with its traceback
  (persistence writes, the AI turn loop, background tasks in
  server.runtime.tasks), so the logging integration turns each of those into
  an event without any call site importing Sentry.
* Exceptions that escape an asyncio task nobody awaits (the asyncio
  integration), as a backstop for tasks not started through spawn().

Only errors are sent: no tracing or profiling, which keeps the free plan's
quota for what matters. Data collection stays at the SDK's PII-off
default, so cookies, user identity and client IPs are not attached, and
request bodies (passwords, emails, chat) are never read. Two further
defaults would undo that, so both are overridden:

* Stack frames would carry their local variables: a handler's parsed
  request model (email and password in its repr) and the ASGI scope with
  the Authorization header. ``include_local_variables`` is off.
* The websocket handshake's ``Sec-WebSocket-Protocol`` header carries the
  session token (server.api.auth) and is not one the SDK filters, so
  ``_drop_session_token`` removes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from server.config import Settings

if TYPE_CHECKING:
    from sentry_sdk.types import Event, Hint

# Compared case-insensitively: ASGI lower-cases header names, but nothing
# here should depend on that.
_TOKEN_HEADER = "sec-websocket-protocol"


def _drop_session_token(event: Event, hint: Hint) -> Event:
    """Remove the header that carries the websocket session token."""
    headers = event.get("request", {}).get("headers")
    if isinstance(headers, dict):
        for name in [n for n in headers if n.lower() == _TOKEN_HEADER]:
            del headers[name]
    return event


def init_error_reporting(settings: Settings) -> bool:
    """Start the Sentry client if configured; return whether it started."""
    if not settings.sentry_dsn:
        return False
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.env,
        # The api image's content tag (app/scripts/deploy_tags.sh), so an
        # event names the exact deploy it came from.
        release=settings.sentry_release or None,
        max_request_body_size="never",
        include_local_variables=False,
        before_send=_drop_session_token,
        integrations=[AsyncioIntegration()],
    )
    return True
