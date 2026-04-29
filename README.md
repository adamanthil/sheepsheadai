# sheepsheadai

Deep learning AI for the Sheepshead card game.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Node.js 22+ and npm — for the web frontend

---

## Running a CLI game

Install Python dependencies and run a single game played by the agent:

```bash
uv sync
uv run play.py
```

---

## Web UI and Multiplayer Tables

The web UI consists of a FastAPI backend server and a Next.js frontend.

### 1. Backend server

Install the server dependencies (includes FastAPI, uvicorn, asyncpg, etc.):

```bash
uv sync --extra server
```

Start the backend (pass the path to your trained model checkpoint):

```bash
./server/run_server.sh --model final_pfsp_swish_ppo.pt
```

The server listens on `http://localhost:9000` by default.

**Required environment variables:**

| Variable | Description | Example |
|---|---|---|
| `SHEEPSHEAD_MODEL_PATH` | Path to the `.pt` model file | `./final_pfsp_swish_ppo.pt` |

> `SHEEPSHEAD_MODEL_PATH` can be set via the `--model` flag in `run_server.sh` or as an env var directly.

### 2. Frontend

Install Node dependencies (first time only, or after dependency changes):

```bash
cd web
npm install
```

Start the dev server:

```bash
npm run dev
```

The frontend is available at `http://localhost:3000`.

**Other frontend scripts:**

```bash
npm run build      # production build
npm run lint       # ESLint check
npm run typecheck  # TypeScript type check (tsc --noEmit)
```

---

## Upgrading Dependencies

### Python

Upgrade all Python dependencies to their latest allowed versions and update `uv.lock`:

```bash
uv sync --extra server --upgrade
```

### Node

Upgrade Node dependencies and update `package-lock.json`:

```bash
cd web
npm install
```

To upgrade to new major versions, edit the version ranges in `web/package.json` first, then re-run `npm install`.
