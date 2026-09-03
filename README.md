# sheepsheadai

Deep learning AI for the Sheepshead card game.

## Repository layout

```
sheepshead/         installable RL core (uv sync installs it editable)
  game.py           game engine (rules, deals, scoring)
  scripted_agent.py rules baseline    ismcts.py  search / ExIt teacher
  agent/            ppo, encoder, oracle, architectures registry
  training/         the PPO trainer, oracle pretraining, corpus + policy
                    iteration, and the program orchestrator
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
cwd once `uv sync` has run. Trainer entry points: `uv run train-ppo`,
`uv run training-program` (or `python -m
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

CI (`.github/workflows/ci.yml`) runs four jobs — `lint` (the pre-commit
gate over the whole tree: ruff format/check + basedpyright), `server`
(server tests against Postgres + schema drift), `training` (the full
training suite), and `web` (typecheck/lint/prettier/build + generated-type
drift).

Python formatting is ruff. `ruff format` does the layout; import order is a
separate concern it deliberately leaves alone, so isort (`I`) is enabled under
`[tool.ruff.lint]` in `pyproject.toml` and rides along on the lint pass —
`ruff check` reports unsorted imports as I001 and `ruff check --fix` sorts
them. `sheepshead` and `server` are declared first-party there, since the
latter is rooted at `app/` and cannot be inferred:

```bash
uv run ruff format .        # layout
uv run ruff check --fix .   # lint + import order
```

Python type checking is basedpyright, configured in `pyrightconfig.json`
and held at **zero errors** by the pre-commit gate and the `lint` CI job:

```bash
uv run basedpyright              # whole tree
uv run basedpyright path/to.py   # one file (~1s)
```

`typeCheckingMode` is `basic` with the Unknown-* family off — the research
code is deliberately only partially annotated, and basedpyright's default
"recommended" mode reports ~29k warnings about it. `pythonVersion` is pinned
to 3.14 (the repo uses PEP 758 `except A, B:`, which an inferred older
version rejects) and `pythonPlatform` is `All`, so the macOS hook and the
Linux CI job agree on what a clean tree is. The version in the `dev` extra is
pinned exactly and must match the editor's language server; bump both
together.

Frontend formatting is prettier, scoped to `app/web` TypeScript and
JavaScript (`npm run format` to fix, `npm run format:check` to verify).
`lib/api.gen.ts` is excluded because `npm run gen:api` regenerates it in
openapi-typescript's own style; `design/` and `visualizations/vendor/` are
outside the scope on purpose — the former carries `EDITMODE` markers that
tooling rewrites, the latter is vendored upstream code.

### 5. Pre-commit hook (optional)

Hooks are managed by [prek](https://github.com/j178/prek), a single-binary
reimplementation of `pre-commit` that reads the same config format. `git
clone` does not install hooks, so enable the gate once per clone:

```bash
uv tool install prek   # or: brew install prek
prek install
```

Each commit is then checked by `ruff format --check`, `ruff check` and
`basedpyright` on Python, and `eslint` plus `prettier --check` on `app/web`
TypeScript. prek stashes unstaged changes first, so a partially staged file
is judged by what is actually being committed. Nothing is rewritten under
you — a failing hook prints the command to run — and `git commit
--no-verify` skips it.

The hooks are `language: system`, running this repo's own `.venv` and
`app/web/node_modules` rather than versions prek pins itself, so the hook,
CI and your editor never disagree by a patch release. The frontend hooks
skip rather than fail when `node_modules` is missing, so a fresh clone can
still commit; CI is the real backstop.

Two config files, because eslint and prettier have to resolve
`eslint.config.mjs` and `tsconfig.json` from `app/web`: the root
`.pre-commit-config.yaml` carries the Python hooks and
`app/web/.pre-commit-config.yaml` the frontend ones. prek's workspace mode
discovers both and runs each with its own directory as the working
directory.

```bash
prek run --all-files   # the whole tree (~12s), same as the lint CI job
prek run <hook-id>     # one hook, e.g. `prek run basedpyright`
```

### 6. Deployment

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

### The training program

`sheepshead/training/run_training_program.py` runs the whole release-candidate
program end to end (design and pre-registration in
`notebooks/Training_Program_Redesign_202609.md`), crash-resumable from an
atomic `state.json`, with one config dataclass (`program_config.py`) that
doubles as the pre-registration artifact:

```bash
uv run training-program --run-name rc_202609        # the pre-registered run
uv run training-program --smoke --run-name _smoke   # every phase in minutes
uv run training-program --config my_config.json     # a modified program
```

The five phases, each also runnable on its own:

| phase | command | what it does |
|---|---|---|
| 0 bootstrap | `uv run train-ppo --phase bootstrap --arch perceiver-recall --run-name r/bootstrap --until 400000` | shaped self-play from scratch on an empty population, limited critic, leaster watchdog |
| 1 oracle | `uv run python -m sheepshead.training.pretrain_oracle generate/pretrain ...` | supervised pretraining of the privileged critic on the bootstrap policy's games |
| 2 league | `uv run train-ppo --phase league --resume ... --seed-checkpoints ... --until <g x 1M>` | terminal-only PPO with the oracle GAE baseline against a PFSP population of snapshots; one generation per invocation; the orchestrator's marginal-value rule (`stop_rules.py`) decides when to hand off to search |
| 3 policy iteration | `distill_corpus` -> `policy_iteration all` -> `policy_iteration cert` -> `train-ppo --phase bidding` | search-Q regularized policy iteration: an ISMCTS-committee corpus from the frozen policy, the pooled advantage fit, the tilted targets, the supervised projection, the certification battery, then a bidding-only PG phase |
| 4 final | (orchestrator) | duplicate h2h vs the reference agents, a one-time exploitability audit (`analysis/exploitability_audit.py`), `release.pt` |

Every phase writes under `runs/<run-name>/...`; the program's own record
(`state.json`, `config.json`, `generations.csv`, `report.md`, per-step logs)
lives in `runs/<run-name>/program/`. `--dry-run` prints the configuration and
the generation-1 trainer command without training.

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
