"""Build the single-file 3D architecture HTML by embedding the JSON
captured by dump_forward_pass.py — plus the vendored three.js sources,
so the output works fully offline — into the template."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "ppo_forward_pass.json"
TEMPLATE_PATH = HERE / "ppo_3d_template.html"
OUT_PATH = HERE / "ppo_architecture_3d.html"
VENDOR = HERE / "vendor"
THREE_PATH = VENDOR / "three.module.min.js"
ORBIT_PATH = VENDOR / "OrbitControls.js"


def js_string(s: str) -> str:
    """JSON-encode a string for embedding in an inline <script>, escaping
    the sequence that would otherwise terminate the script tag."""
    return json.dumps(s).replace("</", "<\\/")


def main():
    data = json.loads(JSON_PATH.read_text())
    template = TEMPLATE_PATH.read_text()
    embedded = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = (
        template
        .replace("__THREE_SRC_JSON__", js_string(THREE_PATH.read_text()))
        .replace("__ORBIT_SRC_JSON__", js_string(ORBIT_PATH.read_text()))
        .replace("__DATA_JSON__", embedded)
    )
    OUT_PATH.write_text(out)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
