"""Build the single-file 3D architecture HTML by embedding the JSON
captured by dump_forward_pass.py into a Three.js scene."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "ppo_forward_pass.json"
TEMPLATE_PATH = HERE / "ppo_3d_template.html"
OUT_PATH = HERE / "ppo_architecture_3d.html"


def main():
    data = json.loads(JSON_PATH.read_text())
    template = TEMPLATE_PATH.read_text()
    embedded = json.dumps(data, separators=(",", ":"))
    out = template.replace("__DATA_JSON__", embedded)
    OUT_PATH.write_text(out)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
