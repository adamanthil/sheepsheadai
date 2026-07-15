# sheepsheadai

Deep learning AI for the Sheepshead card game.

## Repository layout

```
sheepshead/         installable RL core (uv sync installs it editable)
  game.py           game engine (rules, deals, scoring)
  scripted_agent.py rules baseline    ismcts.py  search / ExIt teacher
  agent/            ppo, encoder, oracle, architectures registry
  training/         self-play + league trainers, exploiter, orchestrators
  analysis/         measurement instruments (rigorous_eval, probes, panels)
  validation/       historical one-off gate checks
  tests/            training/research test suite (kept out of the wheel)
app/                the hosted product: server/ (FastAPI), web/ (Next.js),
                    db/ (graphile-migrate), deploy/, scripts/, compose files
                    (server tests live in app/server/tests)
visualizations/     network visualization artifacts
notebooks/ docs/    research journals and operator docs
play.py             CLI game entry point
```

Python imports use the `sheepshead.*` (and `server.*`) package names from any
cwd once `uv sync` has run. Trainer entry points: `uv run train-selfplay`,
`uv run train-league`, `uv run extended-league` (or `python -m
sheepshead.training.<module>`).

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

### 1. Database

The server requires Postgres at runtime. A local dev instance is provided
via Docker Compose:

```bash
docker compose -f app/docker-compose.yml up -d postgres
```

This boots Postgres 18 on `localhost:5433` (host port chosen to avoid colliding
with a local Postgres on 5432) with both the `sheepshead` database and a
`sheepshead_shadow` database (used by graphile-migrate). Credentials match
`.env.example`.

All database commands run from the `app/db/` directory (or use `npm --prefix app/db`
from the repo root). Apply the schema with
[graphile-migrate](https://github.com/graphile/migrate):

```bash
# First time only — install graphile-migrate locally:
npm --prefix db install

# Make sure DATABASE_URL / SHADOW_DATABASE_URL / ROOT_DATABASE_URL are exported
# in the current shell (graphile-migrate reads them directly):
set -a && source .env && set +a

# Apply committed migrations to the DB pointed at by DATABASE_URL:
npm --prefix db run migrate

# During schema iteration:
npm --prefix db run watch                 # re-apply current.sql on save (shadow + dev)
npm --prefix db run commit -- -m "msg"    # freeze current.sql as a committed migration
npm --prefix db run reset                 # drop & recreate, re-run migrations + afterReset.sql
```

Equivalently, `cd app/db && npm run migrate` etc.

See [docs/database-migrations.md](docs/database-migrations.md) for the full
migration workflow (writing new migrations, deploying to production, common
pitfalls).

Reference seed data (`suit`, `card`) is generated from the `sheepshead` package by
`app/scripts/gen_card_seed.py` and lives in `app/db/fixtures/afterReset.sql`. Re-run
the script after any change to `DECK` / `SUIT_NAMES` and commit the result.

### 2. Backend server


Install the server dependencies (includes FastAPI, uvicorn, asyncpg, etc.):

```bash
uv sync --extra server
```

Start the backend (pass the path to your trained model checkpoint):

```bash
./app/server/run_server.sh --model final_pfsp_swish_ppo.pt
```

The server listens on `http://localhost:9000` by default.

**Environment variables:**

| Variable | Required | Description | Example |
|---|---|---|---|
| `SHEEPSHEAD_MODEL_PATH` | Yes | Path to the trained `.pt` model file. Must point to a file owned and reviewed by you — never load untrusted checkpoints. | `./final_pfsp_swish_ppo.pt` |
| `DATABASE_URL` | Yes | Postgres connection string. Server fails fast at startup if missing. | `postgres://sheepshead:sheepshead@localhost:5433/sheepshead` |
| `SHADOW_DATABASE_URL` | Dev/CI | Shadow DB used by graphile-migrate `watch` / `reset`. Never set in production. | `postgres://...:5433/sheepshead_shadow` |
| `ROOT_DATABASE_URL` | Dev/CI | Superuser DB used by graphile-migrate to create/drop the shadow. | `postgres://...:5433/postgres` |
| `SHEEPSHEAD_CORS_ORIGINS` | In production | Comma-separated list of allowed CORS origins. Required when `ENV=production`; omit in dev (localhost:3000 is allowed automatically). | `https://example.com` |
| `ENV` | No | Set to `production` to enable production-mode CORS and logging defaults. | `development` |
| `LOG_FORMAT` | No | Set to `json` for structured JSON logs (recommended in production). Defaults to `text`. | `json` |

Copy `.env.example` to `.env` and fill in your values. `.env` is never committed.

> `SHEEPSHEAD_MODEL_PATH` can be set via the `--model` flag in `run_server.sh` or as an env var directly.

### 3. Frontend

Install Node dependencies (first time only, or after dependency changes):

```bash
cd app/web
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
npm run gen:api    # regenerate lib/api.gen.ts from openapi.json
```

### 4. Tests and generated types

```bash
uv run pytest                           # both suites (training + server)
uv run pytest sheepshead/tests          # training/research suite (~2 min)
uv run pytest sheepshead/tests -m "not slow"   # fast subset (~15s)
uv run pytest app/server/tests          # hermetic server tests (no DB needed)

# Full API-flow tests against a real Postgres:
docker exec sheepshead_postgres psql -U sheepshead -c "CREATE DATABASE sheepshead_test;"
(cd app/db && DATABASE_URL=postgres://sheepshead:sheepshead@localhost:5433/sheepshead_test npx graphile-migrate migrate)
TEST_DATABASE_URL=postgres://sheepshead:sheepshead@localhost:5433/sheepshead_test uv run pytest app/server/tests
```

REST types are generated from the server's OpenAPI schema. After changing
`app/server/api/schemas.py` or any route signature:

```bash
uv run python app/scripts/export_openapi.py   # refresh app/web/openapi.json
cd app/web && npm run gen:api                 # refresh lib/api.gen.ts
```

CI (`.github/workflows/ci.yml`) runs three jobs — `server` (ruff + server
tests against Postgres + schema drift), `training` (ruff over the package +
the full training suite), and `web` (typecheck/lint/build + generated-type
drift).

### 5. Deployment

See [docs/deploy.md](docs/deploy.md) — single-VPS Docker Compose stack
(Caddy TLS + web + api + Postgres + nightly backups). The API is a single
process by design; the runbook covers deploys, drains, and restores.

---

## Training the AI

Training has two stages — a self-play bootstrap followed by league-based PPO —
plus an orchestrator that runs the league stage end-to-end with an automatic
stopping rule. All build on the shared game primitives in `pfsp_runtime.py`,
the hyperparameters in `config.py`, and the architecture registry in
`architectures.py` (`--arch` on every trainer; checkpoints record their
architecture and are rebuilt to match on load). All artifacts for a run —
checkpoints, the final model, plots, CSVs, and the league roster — are written
under `runs/<run-name>/` (gitignored).

### Step 1 — Self-play PPO (bootstrap)

Train a single agent by self-play. This is the starting point; it needs no
population. Defaults to 100k episodes and writes `<arch>_checkpoint_<N>.pt`
snapshots plus an anchored strength curve (`anchored_eval.csv`, paired CRN
edges vs three frozen yardsticks) under `runs/selfplay_ppo/`:

```bash
uv run train-selfplay --episodes 100000
```

Useful flags: `--arch` (architecture variant), `--leaster-watchdog` (guards the
always-PASS collapse that affects all from-scratch shaped self-play runs).

### Step 2 — League PPO

`sheepshead/training/train_league_ppo.py` is the main trainer: one agent improves under a
terminal-reward PPO objective against a **league** of its own past snapshots
plus, optionally, best-response *exploiters*. The usual practice is to **resume
the policy from the final self-play checkpoint and seed the initial league from
the self-play snapshots** produced in step 1 (`--resume` for the weights,
`--seed-checkpoints` for the opponent roster), matching that starting point:

```bash
uv run train-league \
  --resume runs/selfplay_ppo/full_checkpoint_100000.pt \
  --seed-checkpoints "runs/selfplay_ppo/full_checkpoint_*.pt" \
  --league-dir runs/league_ppo/league --run-name league_ppo \
  --generations 6 --main-episodes 5000000
```

Each generation trains the main agent against league tables (past-main
snapshots, hot exploiters, and self-play seats), then trains and gates a
best-response exploiter, appending its measured edge to `exploitability.csv` —
the empirical-exploitability trend that certifies the run. Artifacts go under
`runs/league_ppo/`. Instead of `--seed-checkpoints`, pass `--migrate-from
<legacy population dir>` to ingest an old PFSP population, or neither to
cold-start the league from pure self-play. Pass `--resume <checkpoint>` pointing
at a later checkpoint to continue an interrupted run.

Notable flags: `--critic-mode oracle` trains a privileged full-information
critic as the GAE baseline (CTDE / asymmetric actor-critic; the deployed policy
never sees hidden state, and league snapshots strip the oracle before
insertion); `--anchor-coeff` + `--anchor-ref` enable a bidding-head KL anchor
for warm-start safety.

### Step 3 — Extended league run with automatic stopping

`sheepshead/training/run_extended_league.py` wraps step 2 into a fully instrumented,
crash-resumable campaign that decides for itself when learning has concluded
(design pre-registered in `notebooks/Extended_League_202607.md`). It
calibrates the gen-1 KL-anchor coefficient with short probes, runs generation
1 anchored and later generations unanchored (one trainer subprocess per
generation), evaluates every generation on the frozen PANEL-A gauntlet
(4000-deal composite endpoint + head-to-head vs the previous generation), and
stops after two consecutive statistically flat generations confirmed on a
fresh deal set:

```bash
uv run extended-league \
  --arch full --resume runs/selfplay_ppo/final_full.pt \
  --seed-checkpoints "runs/selfplay_ppo/full_checkpoint_*.pt" \
  --run-name ext_league --critic-mode oracle
```

Progress lands in `runs/<run-name>/orchestrator/`: `generations.csv`,
`report.md`, `generations_curve.png` (strength + trick-0/1 defender
trump-lead-leak trends), and `state.json` (re-invoke with the same arguments
to resume after any interruption). `--smoke` runs the whole loop in minutes;
`--dry-run` prints the planned commands without training.

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
cd app/web
npm install
```

To upgrade to new major versions, edit the version ranges in `app/web/package.json` first, then re-run `npm install`.
