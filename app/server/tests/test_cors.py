"""The dev CORS regex must only match genuinely local origins."""

from __future__ import annotations

import re

from server.app import DEV_CORS_ORIGIN_REGEX


def test_dev_cors_regex_matches_local_origins():
    for origin in (
        "http://localhost:3000",
        "http://localhost",
        "https://localhost:3001",
        "http://127.0.0.1:3000",
    ):
        assert re.match(DEV_CORS_ORIGIN_REGEX, origin), origin


def test_dev_cors_regex_rejects_foreign_origins():
    for origin in (
        "http://evil.com:3000",
        "http://localhost.evil.com:3000",
        "http://evil.com/localhost:3000",
        "https://notlocalhost:3000",
    ):
        assert not re.match(DEV_CORS_ORIGIN_REGEX, origin), origin
