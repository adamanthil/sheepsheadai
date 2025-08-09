#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
MODEL_PATH=""

# Usage: bash server/run_server.sh [--model path/to/checkpoint.pth]
if [[ "${1:-}" == "--model" && -n "${2:-}" ]]; then
  MODEL_PATH="${2}"
  export SHEEPSHEAD_MODEL_PATH="${MODEL_PATH}"
  echo "Using model: ${MODEL_PATH}"
fi

# If uv is available, prefer it to manage dependencies from pyproject/uv.lock
if command -v uv >/dev/null 2>&1; then
  echo "Using uv to run the server with project dependencies"
  exec uv run -- ${PYTHON} -m uvicorn server.main:app --host 0.0.0.0 --port 9000 --reload
fi

echo "uv not found. You can install it: https://docs.astral.sh/uv/"
echo "Falling back to current environment. Ensure dependencies (fastapi, uvicorn, numpy, torch) are installed."
exec ${PYTHON} -m uvicorn server.main:app --host 0.0.0.0 --port 9000 --reload


