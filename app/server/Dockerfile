# Build from the REPO ROOT: docker build -f app/server/Dockerfile .
# The server imports the sheepshead package (game engine, agent, shared
# training utilities), so the build context must be the repo root.
#
# The model checkpoint is NOT baked in — mount it and set
# SHEEPSHEAD_MODEL_PATH (e.g. -v ./final_pfsp_swish_ppo.pt:/models/model.pt
# and SHEEPSHEAD_MODEL_PATH=/models/model.pt).

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependency layer first so code edits don't re-resolve torch (~200MB even
# as CPU wheels — the pyproject uv index pins Linux to the CPU-only build).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra server --no-dev --no-install-project

COPY sheepshead ./sheepshead
COPY app/server ./server

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=2).status == 200 else 1)"]

# Exactly ONE worker: all live game state (tables, websockets, locks) lives
# in this process's memory. More workers would shard tables across processes
# and break the app. Scale up, not out.
CMD ["uvicorn", "server.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "9000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--timeout-graceful-shutdown", "25"]
