# sheepsheadai

A deep reinforcement learning agent for the card game
[Sheepshead](docs/Sheepshead.md), with a web app for playing against it.

## Repository layout

```
sheepshead/        the RL package, installed editable by `uv sync`
  game.py          game engine: rules, deals, scoring
  agent/           PPO agent, encoder, oracle critic, architectures
  search/          ISMCTS search
  rl/              training building blocks
  eval/            evaluation instruments and model fetching
  training/        trainers and the training-program orchestrator
  research/        one-off experiments
  devtools/        golden gate, benchmarks
  tests/           training test suite
app/               the web product
  server/          FastAPI backend (tests in server/tests)
  web/             Next.js frontend
  db/              Postgres migrations (graphile-migrate)
  deploy/          production deployment files
models/            checkpoint manifest; weights are downloaded here
notebooks/         research journals
docs/              guides
design/            UI design prototypes
visualizations/    network visualizations
play.py            CLI game
```

Inside `sheepshead/`, the packages from `agent/` to `training/` are layered:
each imports only from those listed above it. Nothing imports `research/` or
`devtools/`.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs
  the pinned Python 3.14 for you.
- Node.js 22+ and npm, for the web app.
- Docker, for the web app's Postgres.

## CLI game

Watch the agent play a game:

```bash
uv sync
uv run python -m sheepshead.eval.fetch_models   # download checkpoints to models/
uv run play.py
```

## Web app

A FastAPI server on port 9000 and a Next.js frontend on port 3000, backed by
Postgres on port 5433.

```bash
# one-time setup
cp .env.example .env
uv sync --extra server
uv run python -m sheepshead.eval.fetch_models
npm --prefix app/db install
npm --prefix app/web install

# database
docker compose -f app/docker-compose.yml up -d postgres
set -a && source .env && set +a
npm --prefix app/db run migrate

# backend, in one terminal
./app/server/run_server.sh --model models/pfsp-swish-30m.pt

# frontend, in another
npm --prefix app/web run dev
```

Then open <http://localhost:3000>.

## Training

```bash
uv sync --extra dev
uv run training-program --smoke --run-name _smoke   # every phase on a toy config, ~5 min
uv run training-program --run-name my_run           # the full program
```

The full program takes weeks on a 10-core M1 Max. It is resumable: re-run the
same command to continue. Output goes to `runs/<run-name>/`.

See the [training guide](docs/training.md) for the phases, the commands
behind each one and how to read the logs.

## Documentation

- [Development](docs/development.md): tests, linting, hooks, CI,
  configuration, dependency upgrades
- [Training guide](docs/training.md)
- [Database migrations](docs/database-migrations.md)
- [Deployment](docs/deploy.md)
- [Sheepshead rules](docs/Sheepshead.md)
- `notebooks/`: the research record behind the design
