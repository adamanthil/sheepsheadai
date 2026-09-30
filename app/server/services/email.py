"""Transactional email through Resend (https://resend.com/docs/api-reference).

Callers send from a background task (server.runtime.tasks.spawn) rather
than awaiting: a request's response time must not depend on whether an
email went out, or it would reveal which addresses have accounts.

Without RESEND_API_KEY (development, tests) the message is logged instead,
links included, so the flows can be exercised locally.
"""

from __future__ import annotations

import logging

import httpx

from server.config import get_settings

RESEND_URL = "https://api.resend.com/emails"
_TIMEOUT = httpx.Timeout(10.0)


async def send_email(to: str, subject: str, text: str, html: str) -> None:
    settings = get_settings()
    if not settings.resend_api_key:
        logging.info(
            "email (not sent, no RESEND_API_KEY) to %s: %s\n%s", to, subject, text
        )
        return
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": settings.email_from,
                "to": [to],
                "subject": subject,
                "text": text,
                "html": html,
            },
        )
    # A rejected send (bad key, unverified domain) is an operator problem;
    # raising lets spawn() log it with the response body.
    if response.is_error:
        raise RuntimeError(
            f"Resend rejected email ({response.status_code}): {response.text}"
        )
