#!/usr/bin/env python
"""Dump the FastAPI OpenAPI schema to web/openapi.json.

Deterministic (sorted keys, fixed dummy settings) so CI can regenerate and
diff to catch drift between the server schemas and the committed TS types.

Run from the repo root:  uv run python app/scripts/export_openapi.py

With --check it writes nothing and exits 1 if the committed file is stale
(the prek hook; CI regenerates and diffs to the same end).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from unittest import mock

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REGENERATE = (
    "uv run python app/scripts/export_openapi.py && (cd app/web && npm run gen:api)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if web/openapi.json is out of date, writing nothing",
    )
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".pt") as stub_model:
        os.environ["SHEEPSHEAD_MODEL_PATH"] = stub_model.name
        os.environ["SHEEPSHEAD_MODEL_LABEL"] = "schema-export"
        os.environ["DATABASE_URL"] = "postgresql://x:x@127.0.0.1:1/x"
        os.environ["ENV"] = "development"

        import server.app as app_module
        from server.config import get_settings

        get_settings.cache_clear()
        with mock.patch.object(app_module, "load_agent", lambda path: object()):
            app = app_module.create_app()
        schema = app.openapi()

    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    out_path = os.path.join(APP_DIR, "web", "openapi.json")
    if args.check:
        with open(out_path) as f:
            if f.read() == text:
                return
        # A route's docstring is its OpenAPI description, so even a comment
        # edit in server/api can land here.
        print(
            "app/web/openapi.json is out of date with the server's routes "
            f"and schemas. Regenerate it and the TS types:\n  {REGENERATE}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
