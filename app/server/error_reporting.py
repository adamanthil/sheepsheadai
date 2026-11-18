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
request bodies (passwords, emails, chat) are never read.
"""

from __future__ import annotations

import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from server.config import Settings


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
        integrations=[AsyncioIntegration()],
    )
    return True
