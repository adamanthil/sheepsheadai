# Repository Reorganization Plan — July 2026

**Status: EXECUTED 2026-07-11** on branch `repo-reorg` (commit series
starting at `91a15bb`), after the 400k extension batch finished and its ext2
panels were read and recorded (stage-0 closed 2026-07-11, commit `45cdd7e`).
All gates passed, including golden 34/34 bit-identity, full pytest, a
byte-identical CRN reproduction of the v2 200k called-mode panel, docker
build + in-image import smoke, server tests, and the web build. The document
below is the plan as approved; deviations, if any, are noted in the commit
messages.

Operator decisions locked 2026-07-10:

- Product umbrella directory is named **`app/`**.
- `analysis/` and `validation/` become **subpackages of `sheepshead/`**
  (not a separate `research/` package — see Rationale).
- `visualizations/` stays top-level (separate artifact type).
- Core package is named **`sheepshead`**, absorbing `sheepshead.py` as
  `sheepshead/game.py`.

## Motivation

The repo is four projects sharing one flat namespace: the RL core (~14 loose
root modules), research instruments (`analysis/`, `validation/`, with 21+
`sys.path.insert` hacks), the hosted product (`server/`, `web/`, `db/`,
`deploy/`, `scripts/`), and artifacts/journals. Two name collisions confuse
navigation (`league.py` roster vs `train_league_ppo.py` trainer; root
`config.py` vs `server/config.py`), and `analysis/` contains two files that
are training orchestrators, not analyses (`run_extended_league.py`,
`run_ablation_matrix.py`).

Organizing principle: **group by audience and dependency direction.**
`sheepshead` is the importable core; `app/` consumes it; `tests/`,
`visualizations/`, and `play.py` consume it; nothing imports back into
consumers.

### Rationale: why analysis lives inside the package

The orchestrators and instruments are coupled in-process:
`run_extended_league` imports `league_progress_eval` + `league_stopping`;
`league_progress_eval` imports `rigorous_eval` and `trump_lead_probe`
machinery; `PANEL_A` is shared between `run_ablation_matrix` and
`league_progress_eval`. A separate top-level `research/` package would force
`sheepshead.training` to import from `research/`, inverting the layering.
One package with internal layering avoids that. Internal import rule after
the move: **`sheepshead.analysis` must never import `sheepshead.training`**
(the reverse is allowed — orchestrators sit on top of instruments). `PANEL_A`
moves to a neutral `sheepshead/analysis/panels.py` to satisfy this.

## Target layout

```
sheepshead/                     # installable research core (uv editable)
  __init__.py                   #   re-exports game API (see §Import strategy)
  game.py                       #   ← sheepshead.py   (the only file rename)
  scripted_agent.py             #   rules baseline (no torch)
  ismcts.py                     #   search / ExIt teacher
  agent/
    __init__.py
    ppo.py  encoder.py  oracle.py
    architectures/              #   ← existing architectures/ package
  training/
    __init__.py
    train_selfplay_ppo.py  train_league_ppo.py  exploiter.py
    league.py                   #   roster/PFSP membership (collision resolved by context)
    pfsp_runtime.py  training_utils.py  config.py
    run_extended_league.py      #   ← analysis/
    run_ablation_matrix.py      #   ← analysis/ (it launches trainers)
  analysis/                     # measurement instruments
    __init__.py
    panels.py                   #   NEW — PANEL_A (extracted from run_ablation_matrix)
    rigorous_eval.py  scripted_probe.py  trump_lead_probe.py
    league_progress_eval.py  league_stopping.py
    aggregate_ablation.py  ablation_report.py  capture_arch_goldens.py
    tournament_eval.py  model_comparison.py  model_picking_report.py
    policy_kl_compare.py  critic_calibration.py  leaster_scan.py
    analyze_card_embeddings.py
    analyze_defender_trump_leads.py  scan_defender_trump_leads.py
    compare_partner_defender_leads.py  counterfactual_trump_leads.py
    targeted_trump_lead_search.py  tune_deploy_search.py
    diagnostics/                #   ← analysis/diagnostics/
  validation/                   # historical one-off gate checks (ISMCTS era)
    __init__.py
    <all 18 files unchanged>
app/                            # the hosted product
  server/                       #   ← server/  (Python package, import name stays `server`)
  web/                          #   ← web/    (Next.js)
  db/                           #   ← db/     (graphile-migrate)
  deploy/                       #   ← deploy/ (Caddyfile, .env.prod.example)
  scripts/                      #   ← scripts/ (export_openapi.py, gen_card_seed.py)
  docker-compose.yml            #   ← root
  docker-compose.prod.yml       #   ← root
visualizations/                 # stays top-level (operator decision)
tests/                          # stays top-level (covers core; server tests live in app/server/tests)
notebooks/  docs/  design/      # stay top-level (journals/docs/design artifacts)
runs/  model-history/           # gitignored data, untouched
play.py                         # human-facing entry point, stays at root
final_pfsp_swish_ppo.pt         # DOES NOT MOVE (frozen PANEL-A anchor, ~20 path refs)
README.md  pyproject.toml  uv.lock  .gitignore  .github/
```

### Deliberate non-moves

- `final_pfsp_swish_ppo.pt` — frozen eval anchor; referenced by path in ~20
  files including frozen instruments and notebooks.
- `runs/`, `model-history/` — gitignored artifact stores; old watcher shell
  scripts inside `runs/` reference pre-reorg paths and must NOT be restarted
  after the reorg without path fixes (they will all be finished by then).
- `notebooks/` — journals are referenced by name from many places (including
  operator memory); moving them buys nothing.
- `tests/` at top level — pytest convention; imports update mechanically.

## Import strategy

- `sheepshead/__init__.py` re-exports the game API (explicit list of the
  public names currently imported via `from sheepshead import ...` — Game,
  Player, ACTIONS, card helpers, etc., derived by grep at implementation
  time). This keeps all **~72** `from sheepshead import ...` sites working
  unchanged. No star-import; explicit names.
- All other flat imports rewrite mechanically (~50 files, one line each):
  - `import ppo` / `from ppo import X` → `from sheepshead.agent import ppo` /
    `from sheepshead.agent.ppo import X`
  - `encoder` → `sheepshead.agent.encoder`; `oracle` → `sheepshead.agent.oracle`
  - `architectures` → `sheepshead.agent.architectures`
  - `league` → `sheepshead.training.league`; `pfsp_runtime`, `training_utils`,
    `config` → `sheepshead.training.*`
  - `scripted_agent` → `sheepshead.scripted_agent`; `ismcts` → `sheepshead.ismcts`
  - `from analysis.X import` → `from sheepshead.analysis.X import`
- Use absolute `sheepshead.*` imports throughout (repo style), no relative
  imports.
- Delete every `sys.path.insert` hack in `analysis/` and `validation/`
  (21+ sites) — the editable install makes them unnecessary.
- No back-compat shim modules at root: we control all callers; a clean break
  plus repo-wide rewrite is less confusing than lingering `ppo.py` stubs.
- Checkpoints are move-safe: `ppo.save()` writes state_dicts + plain metadata
  only (no class pickling), so existing `.pt` files load unchanged.

## Packaging mechanics

- `pyproject.toml`: add a `[build-system]` (hatchling) and package mapping:

  ```toml
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [tool.hatch.build.targets.wheel]
  packages = ["sheepshead", "app/server"]
  ```

  The `app/server` entry installs the server package under the import name
  `server`, so all `from server.x import ...` lines (server-internal, tests,
  `app/scripts/export_openapi.py`) keep working, and the Docker image layout
  (`/app/server`) is unchanged. `uv sync` then installs the project editable;
  `uv run` resolves `sheepshead.*` and `server.*` from any cwd.
- Optional sugar (recommended): `[project.scripts]` entries so trainer
  commands get *shorter* than today:

  ```toml
  [project.scripts]
  train-selfplay = "sheepshead.training.train_selfplay_ppo:main"
  train-league = "sheepshead.training.train_league_ppo:main"
  extended-league = "sheepshead.training.run_extended_league:main"
  ```

  (Requires each script's argparse body to be wrapped in a `main()` — verify
  at implementation time; skip any file where that refactor isn't trivial and
  use `python -m` for it instead.)
- Subprocess launch sites switch from script paths to `-m` module invocation
  (cwd-robust):
  - `train_league_ppo.py:628` → `[sys.executable, "-m", "sheepshead.training.exploiter", ...]`
  - `run_extended_league.py:452` → `-m sheepshead.training.train_league_ppo`
  - `run_ablation_matrix.py` (~:118/:150/:164/:237/:277) →
    `-m sheepshead.training.train_selfplay_ppo`,
    `-m sheepshead.analysis.rigorous_eval`,
    `-m sheepshead.analysis.aggregate_ablation`

## app/ mechanics

- `server/Dockerfile` (stays at `app/server/Dockerfile`, build context stays
  repo root):
  - `COPY sheepshead.py ppo.py encoder.py training_utils.py ./` →
    `COPY sheepshead ./sheepshead` (drags analysis/validation along — a few
    hundred KB of source, acceptable; optionally exclude via `.dockerignore`
    later).
  - `COPY server ./server` → `COPY app/server ./server` (image layout and
    `uvicorn server.app:create_app` CMD unchanged).
  - Update the header comment (the "engine extraction into a package is
    deferred" note comes true).
- Server import updates (5 lines):
  - `server/runtime/models.py:12` `from ppo import PPOAgent`
  - `server/services/ai_loader.py:8-9` `import ppo` / `from ppo import ...`
  - `server/services/analyze.py:19` `from training_utils import ...`
  → `sheepshead.agent.ppo` / `sheepshead.training.training_utils`.
  (Side note for later, out of scope: `analyze.py` importing reward internals
  from training_utils is a boundary worth revisiting.)
- Compose files move to `app/`; their `build.context` becomes `..` (repo
  root) with `dockerfile: app/server/Dockerfile`; volume paths (e.g.
  `db/init`) get `../`-adjusted or the compose files use repo-root-relative
  paths — verify against both compose files at implementation time.
- `.github/workflows/ci.yml`: path updates — `ruff check app/server`,
  `compileall app/server`, pytest target, node steps' `working-directory`
  for `app/db` and `app/web`, any path filters.
- Grep `package.json` files, `deploy/*.sh`, `server/run_server.sh`, and
  `docs/deploy.md` + `docs/database-migrations.md` for hardcoded `server/`,
  `web/`, `db/`, `scripts/` paths and update.

## Documentation updates (same branch, atomic)

- Rewrite command blocks in the **live** journals:
  `Architecture_Ablation_202607.md`, `Extended_League_202607.md`,
  `Evaluation_Harnesses_202607.md` (e.g. `uv run analysis/rigorous_eval.py`
  → `uv run python -m sheepshead.analysis.rigorous_eval`; trainer commands
  → console scripts or `-m`). The mechanical-executability guarantee for the
  ablation notebook is preserved by updating commands in the same merge.
- Historical notebooks (ISMCTS-era etc.): add a one-line header note "paths
  predate the 2026-07 repo reorg" instead of rewriting history.
- README: new layout section + updated commands.
- `.gitignore`: `analysis/card_embedding_analysis/` →
  `sheepshead/analysis/card_embedding_analysis/`.

## Root clutter cleanup (final commit)

- Delete stray `ruff/` directory (abandoned venv-like cache: bin/, lib/,
  pyvenv.cfg, CACHEDIR.TAG — not a config dir).
- Move `rigorous_strength.png`, `final_pfsp_swish_training.png`,
  `tournament_results_30M.xlsx` → `docs/assets/` (verified: nothing
  references them by path).
- The session transcript `2026-07-08-134813-claude-session-*.txt` is
  untracked operator material — operator moves or deletes it manually.

## Verification gates (every commit on the branch must pass 1, 2, 6; the

full set runs before merge)

1. **Golden gate**: `uv run python -m sheepshead.analysis.capture_arch_goldens --check`
   — construction + update hashes must match (pure moves are behaviorally
   invisible; this is the standing gate for any arch/ppo structural change).
2. **Full test suite**: `uv run pytest tests/` (imports updated), plus
   `app/server/tests` via the CI job locally.
3. **CRN determinism end-to-end**: re-run one completed rigorous_eval panel
   command (e.g. the v2 200k called-mode panel) with identical args and diff
   the CSV against the existing artifact — must reproduce to 4 decimals
   (established property of the instrument; proves the whole
   load→game→eval path survived the move).
4. **Trainer smokes**: `tests/test_league_smoke.py`; plus a ~200-episode
   selfplay run via the new entry point; plus one exploiter-subprocess
   launch (exercises the `-m` subprocess rewrite).
5. **Product**: `docker build -f app/server/Dockerfile .` succeeds; server
   test suite green; CI workflow green on the branch; web `npm run build`.
6. **Lint**: `ruff format --check` + `ruff check` clean.

## Commit series (single branch `repo-reorg`, merge as a unit)

1. Packaging: pyproject build-system + hatch mapping; `git mv` core modules
   into `sheepshead/` (`sheepshead.py`→`game.py`, agent/, training/);
   `__init__` re-exports; repo-wide import rewrite; subprocess `-m` rewrite.
   Gates 1, 2, 6.
2. `analysis/` + `validation/` → `sheepshead/`; `PANEL_A` →
   `analysis/panels.py`; delete sys.path hacks; enforce
   analysis-never-imports-training. Gates 1–3, 6.
3. `app/` umbrella: `git mv` the five dirs + compose files; Dockerfile,
   server imports, CI, path greps. Gate 5.
4. Docs: notebooks command rewrite, README, historical-notebook notes,
   `.gitignore`.
5. Root clutter cleanup.

Rollback: the branch is dominated by `git mv` + mechanical one-line import
rewrites; dropping the branch (pre-merge) or reverting the merge restores
the old layout exactly. Estimated effort: ~half a day including gates.

## Sequencing

Execute only after the 400k extension batch completes AND its ext2 panels
are read and folded into the ablation notebook. If that leaves no room
before operator access ends (2026-07-12), this plan is designed to be
mechanically executable later — nothing in it depends on conversation
context.
