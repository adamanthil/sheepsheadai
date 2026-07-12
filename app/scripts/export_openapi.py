#!/usr/bin/env python
"""Dump the FastAPI OpenAPI schema to web/openapi.json.

Deterministic (sorted keys, fixed dummy settings) so CI can regenerate and
diff to catch drift between the server schemas and the committed TS types.

Run from the repo root:  uv run python app/scripts/export_openapi.py
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
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

    out_path = os.path.join(APP_DIR, "web", "openapi.json")
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
