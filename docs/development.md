# Development

Day-to-day commands for working on the repo. For getting the app or the
trainer running the first time, see the [README](../README.md).

## Setup

```bash
uv sync --extra server --extra dev   # Python: core, server and dev tools
npm --prefix app/web install         # frontend
npm --prefix app/db install          # graphile-migrate
cp .env.example .env                 # server and migration settings
uv tool install prek && prek install # pre-commit hooks (optional)
```

## Configuration

The server reads `.env` from the repo root. `.env.example` documents every
variable; `app/server/config.py` is the authoritative list with defaults.

| Variable | Needed | Purpose |
|---|---|---|
| `SHEEPSHEAD_MODEL_PATH` | always | Checkpoint to serve. `run_server.sh --model` sets it. Only load checkpoints you trust. |
| `SHEEPSHEAD_MODEL_LABEL` | always | Human-readable name for the loaded model. |
| `SHEEPSHEAD_TEMPERATURE_*`, `SHEEPSHEAD_LOGIT_BAND` | optional | Sampling for the table AI, tuned per checkpoint. Defaults to argmax. |
| `DATABASE_URL` | always | Postgres connection string. |
| `SHEEPSHEAD_CORS_ORIGINS` | production | Comma-separated allowed origins. In development it adds to the built-in localhost allowance. |
| `SHADOW_DATABASE_URL`, `ROOT_DATABASE_URL` | migrations | Used by graphile-migrate in dev and CI. Never set in production. |
| `ENV`, `LOG_FORMAT` | optional | `production` / `json` in production. |

The frontend's own settings are in `app/web/.env.example`.

## Tests

```bash
uv run pytest                                   # both Python suites
uv run pytest sheepshead/tests                  # training suite (~2 min)
uv run pytest sheepshead/tests -m "not slow"    # fast subset (~15 s)
uv run pytest app/server/tests                  # server suite, no database needed
```

The server's API-flow tests run only when `TEST_DATABASE_URL` points at a
migrated Postgres:

```bash
docker exec sheepshead_postgres psql -U sheepshead -c "CREATE DATABASE sheepshead_test;"
export TEST_DATABASE_URL=postgres://sheepshead:sheepshead@localhost:5433/sheepshead_test
DATABASE_URL=$TEST_DATABASE_URL npm --prefix app/db run migrate
uv run pytest app/server/tests
```

Frontend, from `app/web`:

```bash
npm test       # Vitest unit tests
npm run e2e    # Playwright; starts its own API, web build and Postgres container
```

The e2e run needs Docker and `models/pfsp-swish-30m.pt`.

Changes to network outputs or search are checked bit-for-bit against the
upstream branch:

```bash
uv run python -m sheepshead.devtools.check_goldens
```

A commit that changes outputs on purpose says so with a
`Changes-Network-Outputs:` trailer. See
[sheepshead/tests/README.md](../sheepshead/tests/README.md) for the suite map
and the trailer format.

## Generated API types

The frontend's REST types come from the server's OpenAPI schema. After
changing `app/server/api/schemas.py`, a route signature or a route docstring:

```bash
uv run python app/scripts/export_openapi.py    # refresh app/web/openapi.json
npm --prefix app/web run gen:api               # refresh app/web/lib/api.gen.ts
```

## Format, lint and type-check

```bash
uv run ruff format .          # Python layout
uv run ruff check --fix .     # Python lint and import order
uv run basedpyright           # Python types; the tree is held at zero errors
```

From `app/web`:

```bash
npm run format       # prettier (format:check to verify only)
npm run lint         # eslint
npm run typecheck    # tsc
```

Notes:

- basedpyright is pinned exactly in the `dev` extra. Keep your editor's
  language server on the same version.
- Python is pinned to 3.14 and the code uses PEP 758 `except A, B:`, so run
  tools through `uv run`, not a system Python.
- prettier skips `lib/api.gen.ts` (generated), `design/` and
  `visualizations/vendor/`.

## Pre-commit hooks

Hooks run through [prek](https://github.com/j178/prek) and are opt-in per
clone (`prek install`). They check and never rewrite; a failing hook prints
the command that fixes it.

| Config | Hooks |
|---|---|
| `.pre-commit-config.yaml` | `ruff-format`, `ruff-check`, `basedpyright`, `openapi-schema` |
| `app/web/.pre-commit-config.yaml` | `eslint`, `prettier`, `vitest`, `api-types` |

```bash
prek run --all-files    # whole tree (~12 s), same as the lint CI job
prek run basedpyright   # one hook
```

The hooks use the repo's own `.venv` and `app/web/node_modules`. The frontend
hooks skip when `node_modules` is missing.

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | Checks |
|---|---|
| `lint` | the prek hooks over the whole tree |
| `server` | server tests against Postgres, schema drift |
| `training` | the full training suite |
| `goldens` | bit-exact network and search outputs against the previous commit |
| `web` | generated-type drift, typecheck, eslint, prettier, Vitest, build |
| `e2e` | Playwright against the real API, web build and Postgres |
| `images` | builds the production images; `main` only, after every job above passes |
| `deploy` | deploys the images; only when `DEPLOY_ENABLED` is set |

Deployment is covered in [deploy.md](deploy.md).

## Database

Schema changes go through graphile-migrate; see
[database-migrations.md](database-migrations.md).

## Upgrading dependencies

```bash
uv sync --extra server --extra dev --upgrade   # Python, rewrites uv.lock
npm --prefix app/web update                    # Node, within package.json ranges
```

For a new Node major version, edit the range in `app/web/package.json` and
run `npm install`.
