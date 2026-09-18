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

The agent is produced by one **training program**: a shaped self-play
bootstrap, supervised pretraining of a privileged critic, terminal-reward
league policy gradient with an automatic handoff rule, search-Q regularized
policy iteration to convergence, and a final certification. The design, the
pre-registered expectations and the decision log are in
[`notebooks/Training_Program_Redesign_202609.md`](notebooks/Training_Program_Redesign_202609.md);
the policy-iteration recipe and the investigation that produced it are in
[`notebooks/CE_Teacher_Design_202608.md`](notebooks/CE_Teacher_Design_202608.md)
(§20.14 is the pinned recipe, §21 the close-out summary). This section is
the operator's guide: what to run, what appears on disk, what the logs say,
and which numbers decide each phase.

Everything below was developed and timed on a 10-core M1 Max with CPU game
workers (`--num-workers 8`). `runs/` is gitignored; every artifact of a run
lives under `runs/<run-name>/`.

### Setup

```bash
uv sync --extra dev                            # torch, the sheepshead package, test tools
uv run pytest sheepshead/tests -m "not slow"   # ~15 s sanity check
uv run python -m sheepshead.analysis.capture_arch_goldens --check   # encoder goldens
uv run training-program --smoke --run-name _smoke                    # every phase, ~5 min
```

The smoke run exercises every phase and stage of the orchestrator on a toy
configuration and must end with `PROGRAM FINISHED`; it proves the plumbing,
not the learning (its policy never leaves the initial weights). Reference
checkpoints used by the review gates (the 8M league seed of the July run,
the iteration-1 policy-iteration checkpoint) are training artifacts and are
not distributed; the orchestrator skips a reference read with a log line
when the file is absent. The one reference that ships with the repository
is `final_pfsp_swish_ppo.pt`, the 30M-episode production model the web app
serves — point `final.references` at it to compare a fresh run against it.

### The orchestrator

`sheepshead/training/run_training_program.py` runs the whole program,
crash-resumable, from one configuration tree (`program_config.py`) whose
defaults ARE the pre-registered run:

```bash
uv run training-program --run-name rc_202609            # the pre-registered run (all defaults)
uv run training-program --config runs/x/config.json     # a modified program
uv run training-program --dry-run --run-name rc_202609  # print the config and the first command
uv run training-program --smoke --run-name _smoke       # minutes-long end-to-end check
```

`--config` takes a JSON file with the same shape as
`ProgramConfig` (any omitted field keeps its default). The run's own record
is written under `runs/<run-name>/program/`:

| file | what |
|---|---|
| `config.json` | the configuration actually run — the pre-registration artifact |
| `state.json` | atomic resume state: every recorded number of every phase |
| `program.log` | timestamped, one line per event: ruled banners at phase boundaries (with each phase's start time and, on completion, its wall-clock and stage hours), dashed sub-headers per generation / iteration, `▶` stage starts with the full command indented beneath, `✔` completions with elapsed hours, `★` decisions, `↷` stages a resume found complete, `✖` NEEDS REVIEW |
| `<stage>.log` | the stdout of each stage subprocess (`bootstrap.log`, `oracle.log`, `league_gen<g>.log`, `pi_iter<k>.log`, `final.log`) |
| `generations.csv` | one row per league generation (h2h, panel, conventions, decision) |
| `report.md` | regenerated at every phase boundary: generation table, iteration table, final reads, event log |

**Resuming.** Every stage is skipped when its output already exists (the
bootstrap's `final.pt`, `oracle_init.pt`, a generation's boundary
checkpoint, a corpus manifest with all games kept, `distill_best.json`,
`cert.json`, a bidding phase's `final.pt`), so re-running the same command
continues where the run stopped — including mid-phase, since the trainers
resume from their last checkpoint. A gate failure or a stage that exits
non-zero stops the program with exit code 2 and a `NEEDS REVIEW: ...` line
in `program.log`; fix the cause and re-run the same command (the status
flips back to running). Exit code 0 means `PROGRAM FINISHED`.

### Phase by phase

Each phase is a standalone command the orchestrator issues; the commands
below are the ones it runs (`uv run python -m ...` outside the orchestrator,
which calls the venv's python directly). `uv run train-ppo` is an alias for
`python -m sheepshead.training.train_ppo`.

**Phase 0 — shaped self-play bootstrap** (400k episodes; 6–8 h estimated)

```bash
uv run train-ppo --phase bootstrap --arch perceiver-recall \
    --run-name rc_202609/bootstrap --until 400000 \
    --save-interval 50000 --greedy-eval-interval 50000 --greedy-eval-games 200 \
    --num-workers 8 --seed 42
```

Intermediate trick rewards plus a leaster bonus (`reward_shaping.py`), the
limited critic, an empty population (every seat is the training agent), the
leaster watchdog against the all-PASS attractor. Artifacts under
`runs/rc_202609/bootstrap/`: `checkpoints/checkpoint_<episode>.pt`,
`checkpoints/training_progress.csv` (one row per PPO update),
`checkpoints/greedy_health.csv` (one row per greedy probe) and `final.pt`.
The only gate is health: the last greedy probe's leaster rate must be below
50% (`bootstrap.max_final_leaster_rate`) — there is no strength bar on the
bootstrap by design.

**Phase 1 — oracle pretraining** (~3 h)

```bash
uv run python -m sheepshead.training.pretrain_oracle generate \
    --ckpt runs/rc_202609/bootstrap/final.pt --episodes 40000 --workers 8 \
    --gamma 1.0 --seed 42 --out runs/rc_202609/oracle/dataset.pt
uv run python -m sheepshead.training.pretrain_oracle pretrain \
    --dataset runs/rc_202609/oracle/dataset.pt --max-epochs 25 --patience 3 \
    --seed 42 --out runs/rc_202609/oracle/oracle_init.pt
```

Fits the privileged (full-information) critic and its two aux heads on the
bootstrap policy's own terminal-reward games, so the league phase has a
calibrated GAE baseline from its first update. `oracle.log` records the
per-epoch validation MSE and the per-stratum explained variance.

**Phase 2 — terminal-only league policy gradient** (1M episodes per
generation, ~2 days each estimated; 3–8 generations)

```bash
# generation g trains to the ABSOLUTE episode g x 1,000,000
uv run train-ppo --phase league --resume <previous boundary checkpoint> \
    --run-name rc_202609/league --league-dir runs/rc_202609/league/league \
    --until 1000000 --save-interval 50000 --snapshot-interval 50000 \
    --greedy-eval-interval 50000 --greedy-eval-games 200 \
    --entropy-play-floor 0.28 --num-workers 8 --seed 42 \
    --worker-device mps --worker-compile default --aux-det-scale 2.5 \
    --seed-checkpoints 'runs/rc_202609/seeds/*.pt' \
    --oracle-init runs/rc_202609/oracle/oracle_init.pt --no-entropy-controller
```

`--aux-det-scale 2.5` trains the four deterministic aux heads (seen-trump
mask, unseen-trump-higher, known points, secret partner) at 2.5× their base
loss coefficients; win and return keep theirs. The orchestrator passes it to
every stage that trains those heads (bootstrap, league, the distill trunk
epochs, the bidding phase) from `ProgramConfig.aux_det_scale`.

Generation 1 seeds the population with four copies of the bootstrap final
(`runs/rc_202609/seeds/`), loads the pretrained oracle and runs with fixed
entropy coefficients; from generation 2 the target-entropy controller owns
them (`checkpoints/entropy_controller.json`). The league generations run
their game workers' inference on MPS with a compiled encoder (bit-exact vs
CPU, ~1.36x at 8 workers); the other phases keep CPU workers. Terminal reward only, the
oracle as GAE baseline, a per-seat PFSP population of snapshots with a
self-play share, seat-rotated deal-paired collection. Each generation ends
at its boundary checkpoint (`checkpoints/checkpoint_<g000000>.pt`), which
is also promoted to a hall-of-fame anchor in the population.

After every generation the orchestrator records, in `state.json` and
`generations.csv`:

- the duplicate-deal h2h vs the previous boundary (2,000 deals per partner
  mode, seed 42; a fresh-seed confirmation when the read is near the bar);
- the PANEL-A anchored gauntlet (3,996 deals) as the absolute yardstick,
  and PANEL-B (the 30M, the v2 release, iter11 P1, the v2 8M seed: a
  strong-skill, cross-ecology field) recorded beside it, gating nothing;
- the convention battery (4 × 1,000 greedy games): partner trump lead,
  defender trick-0 trump lead, called-suit lead, pick and leaster rates,
  and the deterministic aux heads' reads from the same games (seen-trump
  accuracy / false-seen / recall by trick, unseen-higher, known points,
  secret partner).

The **handoff rule** (`stop_rules.py`, §5.1 of the notebook): a generation
is *improving* when its h2h gain is at least +0.02 with the 2-SE lower
bound above zero. After the 3-generation floor, the first non-improving
generation fires the single play-entropy step; the second hands off to
search with θ₀ = the boundary checkpoint of the last generation wholly at a
settled entropy target. The cap is 8 generations. The handoff has a
**readiness precondition** (§5.4): the four deterministic aux heads must
be essentially never wrong on the battery (seen-trump accuracy ≥ 99.5%,
false-seen ≤ 0.5% overall and per trick 1–5, recall ≥ 99% per trick 1–5;
unseen-higher ≥ 99%; known points mean absolute error ≤ 1.0 point; secret
partner ≥ 99.5%),
because the league is the only phase that builds the trunk memory they
read. A handoff the rule would make is deferred (`continue`, "handoff
deferred, aux heads not ready") until they clear; at the cap without
readiness the program exits NEEDS REVIEW. Two review gates stop the
program for the operator instead of deciding: the generation-2 panel must
read at least +0.06, and the handoff checkpoint's h2h vs the July 8M
reference must have a lower bound above −0.02 (skipped when the reference
is absent). The B2 bounds (partner trump lead ≥ 50%, defender trick-0 trump
lead ≤ 10%) are hard health checks at every generation. The log carries an
`aux readiness gen g: READY` / `NOT READY: <every failing bar>` line and
the decision line reads:

```
gen 3: h2h +0.0312±0.0118 improving=True panel +0.0840 aux_ready=True -> continue (...)
```

**Phase 3 — search-Q regularized policy iteration** (~3 days per
iteration; up to 5)

One iteration k from θ_k, everything under `runs/rc_202609/pi/iter<k>/`:

```bash
# 1. corpus: 8,000 committee-acted games, every play node searched (~60 h)
uv run python -m sheepshead.training.distill_corpus --ckpt <theta_k> \
    --out-dir runs/rc_202609/pi/iter1/corpus --games 8000 --workers 8 --seed 20260903 \
    --p-base 1.0 --boost-lead 1.0 --boost-cs 1.5 --p-min 0.05 --p-max 1.0 \
    --committee-act-frac 1.0 --iters 256 --iters-schedule t0-lead:1024,t1-lead:512 \
    --replicates 3 --node-telemetry runs/rc_202609/pi/iter1/corpus/nodes.jsonl \
    --routed-encoder mps
# 2. advantage fit, tilted targets, supervised projection (~2 h)
uv run python -m sheepshead.training.policy_iteration all \
    --corpus-dir runs/rc_202609/pi/iter1/corpus --ckpt <theta_k> \
    --out-dir runs/rc_202609/pi/iter1 --trunk-epochs 6 --lr 3e-5 --lambda-ret 10.0 --head-epochs 4
# 3. certification (~1.2 h)
uv run python -m sheepshead.training.policy_iteration cert --ckpt <theta_k> \
    --out-dir runs/rc_202609/pi/iter1 --cert-games 1000 --cert-seeds 4 --h2h-deals 8000
# 4. bidding-only PG phase from the certified candidate (~8 h) and its cert (~0.4 h)
uv run train-ppo --phase bidding --resume runs/rc_202609/pi/iter1/distill_epoch<N>.pt \
    --run-name rc_202609/pi/iter1/bidding --league-dir runs/rc_202609/league/league \
    --until 200000 --save-interval 200000 --snapshot-interval 0 --greedy-eval-interval 0 \
    --num-workers 8 --seed 42
uv run python -m sheepshead.training.policy_iteration cert --ckpt runs/rc_202609/pi/iter1/distill_epoch<N>.pt \
    --out-dir runs/rc_202609/pi/iter1/bidding_cert --candidate runs/rc_202609/pi/iter1/bidding/final.pt \
    --cert-games 1000 --cert-seeds 4 --h2h-deals 8000 --no-routed-reads
```

What each step does and leaves behind:

1. **Corpus.** The frozen θ_k plays 8,000 games; at every play node an
   ISMCTS committee (3 replicates, 256 iterations; 1,024 at trick-0 leads
   and 512 at trick-1 leads, one-ply rollouts to oracle-valued leaves)
   scores the legal cards, and the committee's choice is played
   (`--committee-act-frac 1.0`). Rows are partitioned into *override*
   (search disagrees with the policy), *endorsed* and *retention* (bidding
   and leaster nodes, unsearched). Output: `corpus/corpus_<shard>.pt`,
   `corpus/manifest.json` (per-class counts), `corpus/nodes.jsonl`
   (per-node telemetry incl. the budget used). Progress line:
   `[1150/8000 games, 0.04 g/s] searched 19199 override 9386 endorsed 9723 failed 90`,
   ending in `DONE: 8000 games kept, 40000 episodes, 40 shards -> ...`.
   An interrupted corpus resumes with `--start-game <n>`.
2. **fit / target / distill** (`policy_iteration all`). *fit*: a
   heteroscedastic advantage model (an adapter twin of the play pointer with
   the log-prior as covariate) is fit to the committee Q on a 10% game-level
   holdout → `advantage_model.pt`, `fit_report.json`, `row_table.pt`.
   *target*: per-row Fay–Herriot posterior variances and the tilted class-
   mode targets t(a) ∝ p_θk(a) · exp(Â(a)/√v), precision weights capped at
   5 → `targeted/`, `target_report.json`. *distill*: six trunk epochs at
   3e-5 (everything trains: weighted CE on searched rows, retention KL ×10
   to θ_k on bidding and leaster rows, value / oracle regression), then
   bilinear-only head epochs at 1e-3 → `distill_epoch<e>.pt`,
   `distill_log.jsonl`, `distill_best.json` naming the candidate: the
   LAST epoch of the schedule (held-out target KL is logged per epoch as
   a fidelity check but selects nothing — CE_Teacher add. 31). Per-epoch
   lines: `[distill epoch 6] train (1152 steps, 15.3 min): override_ce 0.3855  override_kl 0.0949 ... retention_kl 0.0014 ...`,
   a `holdout:` line, a `probe:` line (500 greedy games) and `saved ...`.
3. **cert** — the adoption battery vs θ_k, `cert.json`: four 1,000-game
   convention probes (seeds 98765–98768), the duplicate-deal h2h at 8,000
   deals per partner mode (sharded across processes, ~20 min), and two
   head-routed reads at the same size: **play-only** (bidding heads from
   θ_k, play from the candidate — the *compounding statistic*) and
   **bidding-only** (the drift guard). Adoption = non-inferiority on the
   full checkpoint (edge + 2·SE ≥ 0), bidding route ≥ −0.003, partner trump
   lead ≥ 96.5%, defender trick-0 trump lead ≤ 1.0%, play logit spread ≥
   3.6. Lines: `[cert] probe seed 98765 (2 min): called_suit_lead_rate 54.6 ...`,
   `[cert] h2h vs theta_k (19 min): edge +0.0029 se 0.0025 (called +0.0043 / jd +0.0014); leaster hands -0.0077 se 0.0102 (n=6586)`,
   `[cert] routed play_only vs theta_k (23 min): +0.0036 se 0.0023`.
   A failed cert raises `NEEDS REVIEW` (the walk-back is the operator's).
4. **Bidding phase.** PPO under terminal reward against the league
   population with the encoder, actor adapter, play pointer and play-under
   head frozen: only the pick / partner / call heads and both critics
   train, so search's play is untouched while bidding and the value stream
   re-ground on fresh on-policy games. Its `final.pt` is certified against
   the candidate on the same battery (no routed reads) and adopted on
   non-inferiority; θ_{k+1} is the adopted checkpoint. `program.log`:
   `iter 1 bidding phase: h2h vs candidate +0.0054±0.0036 -> ADOPTED`.

The iteration's summary line and the stop rule:

```
iter 1: compounding gain (play_only) +0.0036±0.0023; full h2h +0.0029±0.0025; theta_1 = runs/.../bidding/final.pt
policy iteration STOP: <reason>
```

Policy iteration stops when the play-only gain has been below 2 SE for two
consecutive iterations, or at `max_iterations` (5). One corpus per
iteration, generated by the current θ_k; earlier corpora are never reused
(they read at the harm line, CE_Teacher add. 30).

**Phase 4 — final certification.** `runs/rc_202609/final/release.pt` is
the adopted checkpoint; `h2h_vs_<name>.json` at 8,000 deals per mode for
every entry of `final.references` that exists on disk (the v2 lineage's
release, iter11 P1, the production 30M); the convention battery; and a
one-time exploitability audit (`analysis/exploitability_audit.py`: 50k
episodes of best-response PPO against the frozen release, gated on a
3,000-deal duplicate edge → `final/exploit/gate_result.json`). `report.md`
is regenerated with everything above. Golden capture for the release
(`capture_arch_goldens`) and the export are manual steps. To serve the
model, point the API's `SHEEPSHEAD_MODEL_PATH` at `release.pt`.

### Validating the final phases on an existing lineage

`start_phase: "policy_iteration"` runs phase 3 and 4 from an external
checkpoint, sampling the bidding phase's opponents from an existing league
population. This is how the pipeline was validated before the fresh run
(`runs/rc_validate_v2/config.json`, reproduced here in full):

```json
{
  "run_name": "rc_validate_v2",
  "arch": "perceiver-shared-v2",
  "start_phase": "policy_iteration",
  "policy_iteration": {
    "theta_0": "runs/policy_iteration_202609/iter29_d8k_t6/distill_epoch10.pt",
    "league_dir": "runs/league_retention_pg/league",
    "bidding_first": true,
    "max_iterations": 1,
    "corpus_seed_base": 20260913
  },
  "final": {"references": {}, "exploit_episodes": 0},
  "gates": {"gen2_panel_min": -10.0, "handoff_reference": "", "handoff_h2h_lower_min": -10.0}
}
```

With `bidding_first` the external θ₀ (a certified distill candidate) gets
its bidding phase as iteration 0, and iteration 1's corpus comes from the
adopted checkpoint. Measured on the perceiver-shared-v2 lineage
(September 2026; details in CE_Teacher §21):

| stage | wall time | read vs the previous checkpoint |
|---|---|---|
| bidding phase, 200k episodes | 8.0 h | +0.0054 ± 0.0036 (adopted); leaster hands +0.024 ± 0.009 |
| corpus, 8,000 games | 60.6 h | 133k searched / 64.5k override rows |
| fit + target + distill | 2.1 h | held-out target KL 0.124 → 0.100 |
| cert with routed reads | 1.2 h | play-only +0.0036 ± 0.0023, bidding-only −0.0008 ± 0.0008 |

### Running the instruments by hand

```bash
# the adoption battery on any checkpoint pair
uv run python -m sheepshead.training.policy_iteration cert --ckpt <reference.pt> \
    --out-dir <dir> --candidate <candidate.pt> --h2h-deals 8000
# a head-routed chimera read (bidding from one checkpoint, play from another)
uv run python -m sheepshead.analysis.head_routed_h2h --bid-ckpt <a.pt> --play-ckpt <b.pt> --deals-per-mode 8000
# duplicate-deal h2h of two league boundaries
uv run python -m sheepshead.analysis.league_progress_eval --h2h <gen.pt> <prev.pt>
# the anchored PANEL-A gauntlet / rigorous paired comparison
uv run python -m sheepshead.analysis.rigorous_eval --help
```

### Reading the trainer log

Every `train-ppo` phase prints one line per PPO update:

```
Ep 41,594 | picker_avg +1.54 | pick 19% | leaster 6.9% | advσ all/pick/play 0.123/0.154/0.108 | 6.8 eps/s  ev O/L 0.67/0.54 | Hn 0.04/0.11/0.15/0.44
```

`picker_avg` is the training agent's mean score over its last 3,000 picked
hands, `pick`/`leaster` the rates in the training games (against the sampled
population, so not comparable to the greedy self-play probes), `advσ` the
advantage standard deviations, `ev O/L` the oracle and limited critics'
explained variance of the empirical return, and `Hn` the normalized
entropies of the pick, partner, bury and play heads. The same numbers land
in `checkpoints/training_progress.csv`; the greedy probes (pick, leaster,
conventions, logit spread, seen-trump memory) in
`checkpoints/greedy_health.csv`. The `seen-trump` clause of the probe line
reads the aux critic's seen-trump head at every seat's play node: `acc` over
all 14 trumps, `recall` over the trumps the seat can only know from memory
(played in an earlier trick, or the picker's bury/discarded blind; nothing
in the hand or on the table), the same per trick `t0-5` so forgetting shows
as decay, and `false-seen` on trumps the seat has not seen, overall and per
trick (rising recall at flat false-seen is memory; at rising false-seen it
is a head that says "seen" for everything); then `secret` (the
secret-partner self label), `points exact` (known points per seat, exact
after rounding, with the MAE) and `unseen-higher`, the other three
deterministic aux heads at the same nodes. In the bootstrap and the policy
iteration certs these are informational; at the league boundaries they are
the handoff's readiness precondition (above). The unweighted aux losses of
every update land in `training_progress.csv` as `aux_loss_*`. To re-probe
saved checkpoints (for instance after a probe change) run
`uv run python -m sheepshead.analysis.reprobe_checkpoints <run>/checkpoints/checkpoint_*.pt --out <run>/checkpoints/greedy_health_recall.csv`.

### Where the methodology is documented

- `notebooks/Training_Program_Redesign_202609.md` — the program: objectives, architecture, phases, stop rules, pre-registration, decision log.
- `notebooks/CE_Teacher_Design_202608.md` — §20 policy iteration: the corpus, advantage model, targets, projection; §20.13 the 31 addenda of the stall diagnosis and the compounding finding; §20.14 the pinned recipe; §21 the close-out.
- `notebooks/Learning_System_Redesign_202607.md` — the league trainer's terminal-reward / oracle-critic / retention design.
- `notebooks/Architecture_Ablation_202607.md`, `Blind_Bury_Ablation_202608.md` — how `perceiver-shared-v2` and the recall constraint were chosen.
- `notebooks/Evaluation_Harnesses_202607.md` — the duplicate-deal instrument, PANEL-A, the probes.

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
