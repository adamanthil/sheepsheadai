# Continuous Deployment to sheepshead.ai — Plan (August 2026, revised October 2026)

**Status (2026-10-27): PLAN ONLY — revised after the `202609_recall_rc`
program concluded; no code written, no server provisioned.** The August
decisions were audited against the repository as it stands (four CI
jobs, per-model deployment sampling in the app, the release checkpoint
and two frozen evaluation panels that live under gitignored `runs/`).
Changes from the August text are marked **[rev]** and the new decisions
are D9–D12. Companion doc: `docs/deploy.md` (manual deploy; stale — it
still describes a Hetzner box that builds images locally — and is
rewritten in §9).

## TL;DR — what this is

Ship the `app/` stack (Next.js + FastAPI + Postgres behind Caddy, see
`app/docker-compose.prod.yml`) to a $10/mo Vultr VPS in Chicago, with
GitHub Actions continuous deployment on every push to `main` (the
default branch is renamed from `master` before any CD work — §9 step
0). Four properties the design guarantees:

1. **The Python API container is NOT restarted unless code it actually
   runs has changed.** Live games hold all state in that one process;
   research commits (`sheepshead/analysis/`, `training/`, notebooks)
   must not kill them. Mechanism: per-component image tags derived from
   `git log` over an explicit path list, plus docker compose's native
   "only recreate changed services" behavior (§3).
2. **Images are built in CI, never on the box.** The 1 vCPU / 2 GB VPS
   only ever pulls and runs.
3. **Every model artifact the repository depends on — the deployed
   checkpoint AND the seven frozen panel agents — is a GitHub Release
   asset, pinned by one committed manifest (sha256 + size + provenance),
   and fetched by one tool.** [rev] Deploying a new model is a two-step:
   upload release, commit manifest (§5). Running the panels from a fresh
   clone is one command (§5.4).
4. **Per-checkpoint deployment sampling travels with the model.** [rev]
   The four head temperatures and the logit band (Training_Program_
   Redesign §4.5, app migration 000008) are part of the model manifest
   entry, not hand-set on the box (§5.2).

## 0. Decisions and rationale

* **D1 — Host: Vultr, Chicago (ORD), 1 vCPU / 2 GB / 55 GB, ~$10/mo.**
  Primary audience is Wisconsin (~10 ms RTT). Hetzner's US locations lost
  their price edge in 2026 ($13.49 for the same specs); EU locations add
  ~120 ms to every WebSocket round trip. Serverless (Lambda) would
  require externalizing the in-process game state — a rearchitecture
  with no payoff at this price point. Fly.io scale-to-zero was
  considered and rejected for vendor risk; it remains the documented
  cost fallback if real usage shows the box idle 95%+ of the time.
  [rev] Re-check the price and the 2 GB plan's availability at
  provisioning time; the decision does not depend on the exact number.
* **D2 — Per-component deploy tags from git path history** (§3), not
  reproducible-build digest comparison (Docker builds aren't
  reproducible) and not a hand-tracked "last deployed" state file
  (stateless is simpler and rollback-friendly).
* **D3 — The api path filter is narrowed inside `sheepshead/`** to the
  server's actual import closure, and a CI guard recomputes that closure
  and fails if the committed list is stale (§4). Without the guard,
  narrowing risks silently missing restarts after a reorg; with it,
  staleness is a loud CI failure. [rev] The server's direct `sheepshead`
  imports today are `game`, `agent.ppo`, `agent.observation`,
  `agent.sampling`, `agent.convention_wrapper`, `agent` (the package)
  and `training.reward_shaping`; `training/training_utils.py` arrives
  via `agent/ppo.py`. The guard computes the closure, so this list is
  informational.
* **D4 — No `.dockerignore` narrowing of the image contents.** It would
  have to be kept in sync with `api-paths.txt` by hand; the dead
  `analysis/` code in the image is harmless. Accepted consequence: two
  images with the same api tag can differ in unreachable code.
* **D5 — Model artifacts via GitHub Release assets from day one**, not
  committed `.pt` blobs and not Git LFS. [rev, widened] The repository
  is public, so release assets download over plain HTTPS with no token
  (2 GB per asset, no storage or bandwidth quota); Git LFS on a public
  repository would burn its 1 GB/month bandwidth quota on every clone of
  ~130 MB of panels; committing the blobs would add 130 MB to every
  clone forever. [rev, operator 10-27] `final_pfsp_swish_ppo.pt` at
  the repo root was a stopgap and is **removed from the tree**
  (`git rm`) once it is published as an asset like every other model;
  the ~50 scripts, README lines, the Dockerfile comment,
  `.env.prod.example` and `test_arch_golden`'s legacy-load test that
  name the root path point at `models/pfsp-swish-30m.pt` instead (the
  test keeps its skip-when-absent guard). The blob stays in history
  (added in 21954a9, 10.6 MB) — no history rewrite on a public
  repository with sixteen branches; clones carry it until someone
  decides that is worth a force-push.
* **D6 — Migrations run only when `app/db` changed**, and db-only
  migrations must stay compatible with the running (not-restarted) API
  process (§7, rule 2).
* **D7 — v1 deploys do NOT drain.** A Python-code deploy interrupts any
  live games (in-memory state; unavoidable without snapshot/restore).
  Traffic will be near-zero at launch; the existing
  `set_draining()` flag (`app/server/runtime/lifecycle.py`) is the
  future hook, stubbed into the deploy script behind an env flag but
  not wired to an endpoint yet (§10).
* **D8 — Off-site backups via rclone to object storage.** The Vultr
  plan has no provider backups; the `db-backup` container's daily dumps
  land on the same disk that would die with the box. [rev] The same
  rclone remote also mirrors the model artifact cache (§5.5), so a
  deleted release is recoverable from two places besides the operator's
  `model-history/` archive.
* **D9 — Rename the default branch `master` → `main` before any CD
  work** (August decision, reaffirmed by the operator 10-27 after a
  revision proposed dropping it). Cost: every local clone and worktree
  re-points once (§9 step 0 lists the commands; the training worktrees
  and `sheepsheadai-rc` repeat them on next use). The deploy job still
  triggers on `github.event.repository.default_branch` rather than a
  literal name, so the workflow is correct before and after the rename
  and never has to be edited for it.
* **D10 — One manifest for all model artifacts, one fetch tool.** [rev]
  `models/manifest.json` (committed) lists every artifact with its
  release tag, asset name, sha256, size, role and provenance;
  `sheepshead.analysis.fetch_models` downloads and verifies into
  `models/` (gitignored). The deployed model is the entry with
  `role: deploy`; the panels are the entries with `role: panel-a` /
  `panel-b`. The box runs the same verification logic in shell (sha256
  pin from `app/deploy/model.env`, cross-checked against the manifest in
  CI) so the box needs no Python outside the container.
* **D11 — Release assets are made deletion-resistant three ways.** [rev]
  (1) Enable the repository's **immutable releases** setting
  (Settings → General → Releases; verify the toggle exists for this
  account at implementation time), which blocks asset deletion and
  re-upload once a release is published; (2) a **tag ruleset** on
  `artifact/*` that forbids deletion and non-fast-forward updates; (3)
  the sha256 pin in git, which turns any tampering into a loud failure
  rather than a silent swap. Belt: the rclone mirror (D8) and the
  operator's `model-history/`.
* **D12 — Panel identity is the bytes, not the path.** [rev]
  `sheepshead/analysis/panels.py` stops naming `runs/...` paths and
  names manifest keys; the panel loader resolves a key to
  `models/<asset>` and refuses to run if the file's sha256 does not
  match the manifest. The frozen pre-registration (membership, order,
  deal seed, deal hash) is unchanged; only the on-disk location moves.
  Every recorded panel number remains comparable.

## 1. Current state (what already exists, 2026-10-27)

* `app/docker-compose.prod.yml`: caddy (TLS, same-origin routing) → web
  (Next.js) + api (FastAPI) → postgres, plus one-shot `migrate` service
  (graphile-migrate) and a `db-backup` sidecar writing daily dumps to
  repo-root `backups/`. The api service already takes `MODEL_FILE` as a
  read-only mount source and the five sampling env vars
  (`SHEEPSHEAD_TEMPERATURE_{PICK,PARTNER,BURY,PLAY}`,
  `SHEEPSHEAD_LOGIT_BAND`). [rev]
* `app/server/Dockerfile`: python:3.14-slim + uv, CPU-only torch,
  **exactly one uvicorn worker** (all game state in-process — never
  scale out), built-in `HEALTHCHECK` against `/health`,
  25 s graceful shutdown; compose adds `stop_grace_period: 30s`.
* The model checkpoint is a read-only bind mount
  (`SHEEPSHEAD_MODEL_PATH=/models/model.pt`); `SHEEPSHEAD_MODEL_LABEL`
  is required at boot and upserted into `ai_model`; the AI player's
  identity is (model, five sampling columns) in `ai_player` (migration
  000008), so a re-tempered deployment is a different player for stats.
  [rev]
* CI (`.github/workflows/ci.yml`): **four** jobs [rev] — `lint` (prek:
  ruff format/check + whole-project basedpyright), `server` (compileall,
  real migrations against postgres:18, pytest, OpenAPI drift check),
  `training` (sheepshead tests), `web` (generated-API drift, typecheck,
  lint, prettier, build).
* Model artifacts: the release `runs/202609_recall_rc/final/release.pt`
  (= `pi/iter6/bidding/final.pt`, byte-identical), PANEL-A (four files)
  and PANEL-B (four files, one shared with A) — seven distinct files,
  six under gitignored `runs/` and the 30M at the repo root as a
  committed stopgap, ~106 MB total.
  No release has ever been created on the repository (`gh release list`
  is empty). `git` is 70 MB; a blobless clone is cheap.

## 2. Target architecture

```
push to default branch ──► CI (lint / server / training / web, unchanged)
                               │ all green
                               ▼
                          deploy job (GitHub Actions, environment: production)
                               │ 1. compute API/WEB/DB tags from git paths
                               │ 2. build+push ghcr.io images IF tag missing
                               │ 3. ssh vultr → app/scripts/deploy.sh <args>
                               ▼
   Vultr ORD box: /opt/sheepsheadai (git checkout, --filter=blob:none)
                  /opt/sheepsheadai-models/<asset>.pt   (release assets, sha256-verified)
                  docker compose pull + up -d   (recreates ONLY changed services)

research clone:  uv run python -m sheepshead.analysis.fetch_models      → models/*.pt
                 uv run python -m sheepshead.analysis.rigorous_eval ...  (panels resolve to models/)
```

## 3. Core mechanism: content-derived per-service tags

Tag each image with the short sha of the **last commit touching that
component's paths**:

```bash
API_TAG=$(git log -1 --format=%h -- $(cat app/deploy/api-paths.txt))
WEB_TAG=$(git log -1 --format=%h -- app/web)
DB_SHA=$(git log -1 --format=%h -- app/db)
```

Properties this buys:

* **Deterministic & stateless** — recomputable at any commit; no
  "last deployed" bookkeeping. Requires `fetch-depth: 0` checkout.
* **Build skipping** — if `ghcr.io/...:api-<sha>` already exists
  (`docker manifest inspect` succeeds), skip the build entirely.
* **No-op restarts** — a commit touching only `app/web` leaves
  `API_TAG` unchanged; `docker compose up -d` sees an identical image
  reference + config for the api service and does not touch the
  running container. Games survive.
* **Rollback = redeploy old ref** — `workflow_dispatch` with a `ref`
  input recomputes that ref's tags; the images already exist in GHCR,
  so it goes straight to the ssh step.

`app/deploy/api-paths.txt` (initial contents — CI-verified, see §4):

```
app/server
pyproject.toml
uv.lock
sheepshead/__init__.py
sheepshead/game.py
sheepshead/agent
sheepshead/training/reward_shaping.py
sheepshead/training/training_utils.py
```

(`app/server` covers `app/server/Dockerfile`. `sheepshead/agent`
covers `agent/architectures/` and `agent/sampling.py`.) [rev] The
server also imports `sheepshead.game` constants in `app/server/tests`;
tests are under `app/server` and therefore already covered, but a test
change rebuilds the api image — accept it (D4's logic) rather than
exclude `app/server/tests`, which would make the list a maintenance
item.

## 4. CI guard: `app/scripts/check_api_paths.py`

Purpose: make D3's narrowing safe. Run as a step in the existing
`server` CI job (`uv run python app/scripts/check_api_paths.py`).

Behavior spec (implementable directly):

1. Parse every `*.py` under `app/server/` with `ast`; collect imported
   module names matching `sheepshead` or `sheepshead.*` (handle
   `import x.y`, `from x.y import z` — including the case where `z` is
   itself a submodule — and imports nested inside functions/methods;
   AST walk catches those, which is why we do NOT "just import the app
   and inspect sys.modules": lazy imports would be missed).
2. Resolve each module name to a repo file (`sheepshead/foo/bar.py` or
   `sheepshead/foo/bar/__init__.py`; a package resolves to its
   `__init__.py`). `from sheepshead.x import name` must be resolved by
   checking whether `sheepshead/x/name.py` exists (submodule import)
   and otherwise falling back to `sheepshead/x`'s file (attribute
   import).
3. Transitively repeat over each newly reached `sheepshead/` file until
   fixpoint (the closure).
4. Load `app/deploy/api-paths.txt`. Check **coverage**: every file in
   the closure must be equal to, or under a directory listed in, the
   file. Coverage is ⊇, not equality — a conservative extra entry is
   fine (over-restart is a nuisance; under-restart is a bug).
5. Also check every listed path exists in the tree (catches stale
   entries after a reorg).
6. On failure: print the uncovered files and the exact line(s) to add,
   exit 1.

## 5. Model artifacts: release assets + one committed manifest [rev]

**Release = blob store; manifest commit = the deploy.** Code deploys
never create releases; releases are created only when publishing an
artifact. Panels and the deployed model share the mechanism; they differ
only in `role`.

### 5.1 Tag namespace and assets

* Tag namespace `artifact/<name>`. The tag's commit is irrelevant;
  nobody checks out artifact tags. One release may carry several assets
  (the panel bundle), the manifest keys each asset individually.
* **Releases are immutable** (D11, §7 rule 5): a corrected or
  re-trained artifact gets a new name, never a re-upload.
* Bootstrap releases (operator runbook §8.2; hashes measured 10-27 on
  the files as they sit under `runs/`):

| key | release tag | asset | sha256 (prefix) | bytes | role | provenance |
|---|---|---|---|---|---|---|
| `recall-rc-theta6` | `artifact/recall-rc-theta6` | `recall-rc-theta6.pt` | `d96a6224ecc3` | 21,140,707 | deploy | `runs/202609_recall_rc/final/release.pt` (= `pi/iter6/bidding/final.pt`); Training_Program_Redesign decision log 10-25 |
| `pfsp-swish-30m` | `artifact/panels-2026-10` | `pfsp-swish-30m.pt` | `cc644b7109d5` | 10,629,409 | panel-a, panel-b | `final_pfsp_swish_ppo.pt` (repo root until the bootstrap release; removed from the tree in §9 step 2) |
| `pfsp-swish-15m` | `artifact/panels-2026-10` | `pfsp-swish-15m.pt` | `6951cd42c52a` | 10,635,253 | panel-a | `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt` |
| `pfsp-swish-5m` | `artifact/panels-2026-10` | `pfsp-swish-5m.pt` | `c8ec0d1df248` | 10,634,675 | panel-a | `.../pfsp_swish_checkpoint_5000000.pt` |
| `selfplay-swish-100k` | `artifact/panels-2026-10` | `selfplay-swish-100k.pt` | `7f5426b68d3e` | 10,619,055 | panel-a | `runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt` |
| `v2-release` | `artifact/panels-2026-10` | `v2-release.pt` | `6ce80a5e6bc5` | 21,155,651 | panel-b | `runs/rc_validate_v2/final/release.pt` (CE_Teacher §21) |
| `v2-iter11-p1` | `artifact/panels-2026-10` | `v2-iter11-p1.pt` | `348490d8df02` | 21,202,037 | panel-b | `runs/policy_iteration_202609/iter11/distill_epoch7.pt` |
| `v2-league-8m` | `artifact/panels-2026-10` | `v2-league-8m.pt` | `b56ba26c5c09` | 20,979,519 | panel-b | `runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt` |

Full hashes go in the manifest at implementation time (recompute with
`shasum -a 256`; the prefixes above are for cross-checking).

### 5.2 The manifest: `models/manifest.json` (committed)

```json
{
  "release_base": "https://github.com/adamanthil/sheepsheadai/releases/download",
  "artifacts": {
    "recall-rc-theta6": {
      "tag": "artifact/recall-rc-theta6",
      "asset": "recall-rc-theta6.pt",
      "sha256": "d96a6224ecc3…",
      "bytes": 21140707,
      "roles": ["deploy"],
      "arch": "perceiver-recall",
      "provenance": "runs/202609_recall_rc/final/release.pt; program concluded 2026-10-25",
      "sampling": {"pick": 0.5, "partner": 0.5, "bury": 0.0, "play": 0.5, "band": 1.0}
    },
    "pfsp-swish-30m": {
      "tag": "artifact/panels-2026-10", "asset": "pfsp-swish-30m.pt",
      "sha256": "cc644b7109d5…", "bytes": 10629409,
      "roles": ["panel-a", "panel-b"], "arch": "full",
      "provenance": "legacy swish PFSP 30M (+ flush update); project_final_model_provenance"
    },
    "...": "one entry per row of §5.1"
  },
  "panels": {
    "PANEL-A": ["pfsp-swish-30m", "pfsp-swish-15m", "pfsp-swish-5m", "selfplay-swish-100k"],
    "PANEL-B": ["pfsp-swish-30m", "v2-release", "v2-iter11-p1", "v2-league-8m"]
  }
}
```

* `sampling` on the deploy entry is the per-checkpoint deployment
  sampler (§4.5 of the program notebook). **Decided by the operator
  10-27**: T 0.5 on pick, partner and play, bury 0 (the 10-26 split
  attributed the bidding cost to the bury head), global logit band 1.0
  nats. Measured cost vs the release's own argmax field ≈ −0.0155, about
  17% of deals changed; vs the 30M still ≈ +0.09.
  The deploy script turns it into the five env vars, so the sampler
  travels with the model and a re-temper is a manifest commit, not a
  box edit. A re-temper without a weight change recreates only the api
  container (env changed) and creates a new `ai_player` row by design.
* `panels` is the frozen membership, in order. `panels.py` reads it.

### 5.3 `app/deploy/model.env` (committed; the box's shell-sourceable view)

```
SHEEPSHEAD_MODEL_LABEL=recall-rc-theta6
MODEL_ASSET_TAG=artifact/recall-rc-theta6
MODEL_ASSET_NAME=recall-rc-theta6.pt
MODEL_SHA256=d96a6224ecc3…
SHEEPSHEAD_TEMPERATURE_PICK=0.5
SHEEPSHEAD_TEMPERATURE_PARTNER=0.5
SHEEPSHEAD_TEMPERATURE_BURY=0
SHEEPSHEAD_TEMPERATURE_PLAY=0.5
SHEEPSHEAD_LOGIT_BAND=1.0
```

Two files describe the deployed model so the box needs no JSON parser;
CI keeps them identical: a step in the `server` job
(`uv run python app/scripts/check_model_manifest.py`) asserts
`model.env`'s label is a manifest key with role `deploy`, that tag,
asset, sha256 and the five sampling values match the manifest entry
exactly, that every manifest sha256 is 64 hex chars, and that
`panels` lists only known keys.
Optionally (stricter, slower, `workflow_dispatch` only): download the
deploy asset and `torch.load` it against the architecture registry so
an incompatible checkpoint fails CI, not the boot on the box.

### 5.4 The fetch tool: `sheepshead/analysis/fetch_models.py`

```
uv run python -m sheepshead.analysis.fetch_models               # everything in the manifest
uv run python -m sheepshead.analysis.fetch_models --role panel-a --role panel-b
uv run python -m sheepshead.analysis.fetch_models --key recall-rc-theta6
uv run python -m sheepshead.analysis.fetch_models --check        # verify only, no network
```

Spec: stdlib only (`urllib.request`, `hashlib`, `json`; no new
dependency). For each selected key: if `models/<asset>` exists and
hashes to the manifest, skip; else download
`<release_base>/<tag>/<asset>` to a temp file in `models/`, verify
sha256 and size, `os.replace` into place; a mismatch deletes the temp
file and exits 1 naming the key. `--check` is what the panel loader calls before
every panel read, so a stale or corrupted cache fails loudly instead of
producing a number. `models/` is gitignored except `manifest.json`.

Panel resolution (D12): `panels.py` exposes
`panel_paths(name) -> list[Path]` built from the manifest, replacing
the two path lists; `program_config.py`, `rigorous_eval`'s CLI,
`league_progress_eval.eval_endpoint` and `build_panel_reference` call
it. The `checkpoints` column of `notebooks/panel_reference.csv` records
whatever path was used, so old rows keep their `runs/...` strings and
new rows record `models/...`; the `protocol` column, deal seed and deal
hash are unaffected, which is what makes the rows comparable.

### 5.5 Deploy-script side

Models are cached under `/opt/sheepsheadai-models/` (OUTSIDE the
checkout — survives `git checkout --force`). If `<asset>` is absent or
its sha256 doesn't match `model.env`, download
`https://github.com/adamanthil/sheepsheadai/releases/download/$MODEL_ASSET_TAG/$MODEL_ASSET_NAME`
with `curl -fL` (public repository: no token, no `gh` on the box)
into a temp file, verify sha256, `mv` atomically into place. **Hash
mismatch aborts the deploy loudly** — this also defends against a
re-uploaded asset. The nightly rclone job (D8) syncs this directory
alongside `backups/`.

Compose wiring: the api service mounts
`${MODEL_FILE}:/models/model.pt:ro` where `MODEL_FILE` is the absolute
cache path, written by the deploy script into `.env.deploy` (§6). A
model change therefore changes the api service's mount source +
`SHEEPSHEAD_MODEL_LABEL` env → compose recreates the api container
(necessary — that's how weights load) and nothing else.
`api-paths.txt` deliberately does NOT include `app/deploy/model.env` or
`models/manifest.json`: a model swap needs no image rebuild.

Rollback: revert the manifest + `model.env` commit (old asset still
exists) or `workflow_dispatch` at the old ref.

## 6. The moving parts to implement

### 6.1 Repo changes

1. `app/docker-compose.prod.yml`: api/web services switch from `build:`
   to `image: ghcr.io/<owner>/sheepshead-api:${API_TAG}` /
   `...-web:${WEB_TAG}`. The model mount and the sampling env vars are
   already parameterised; `.env.prod.example` drops
   `SHEEPSHEAD_MODEL_LABEL` and `MODEL_FILE` (now `model.env` +
   `.env.deploy`). Keep a `docker-compose.build.yml` override with the
   `build:` blocks for local use. [rev] Bump `db-backup` to a
   `postgres-backup-local` tag whose `pg_dump` is ≥ 18 — the pinned
   `:17` client cannot dump a Postgres 18 server (`pg_dump` refuses
   newer servers), so the current sidecar would fail nightly on the
   real box.
2. New: `app/deploy/api-paths.txt` (§3), `app/deploy/model.env` (§5.3),
   `models/manifest.json` + `models/.gitignore` (§5.2).
3. New: `app/scripts/check_api_paths.py` (§4),
   `app/scripts/check_model_manifest.py` (§5.3),
   `sheepshead/analysis/fetch_models.py` (§5.4); `panels.py` → manifest
   keys (D12) and its four callers.
4. New: `app/scripts/deploy.sh` (§6.3).
5. `.github/workflows/ci.yml`: add the two guard steps to `server` and
   the `deploy` job (§6.2).
6. `/health` reports the sampling label beside the model label [rev]
   (`SamplingSpec.label()` already exists) so a re-temper is verifiable
   from outside; `HealthResponse` gains `sampling: str` → OpenAPI export
   + `gen:api` regen (the CI drift checks require it).
7. `docs/deploy.md`: rewrite "Deploying an update" to point here; the
   initial-provisioning section gains §6.4's steps; drop the
   build-on-box commands and the Hetzner sizing.

### 6.2 The deploy job (append to `ci.yml`)

```yaml
deploy:
  needs: [lint, server, training, web]
  if: >-
    (github.event_name == 'push'
       && github.ref == format('refs/heads/{0}', github.event.repository.default_branch))
    || github.event_name == 'workflow_dispatch'
  runs-on: ubuntu-latest
  environment: production
  concurrency: { group: production-deploy, cancel-in-progress: false }
  steps:
    - uses: actions/checkout@v5
      with:
        fetch-depth: 0                     # tags need full path history
        ref: ${{ inputs.ref || github.sha }}
    - id: tags
      run: |
        echo "api=$(git log -1 --format=%h -- $(cat app/deploy/api-paths.txt))" >> "$GITHUB_OUTPUT"
        echo "web=$(git log -1 --format=%h -- app/web)" >> "$GITHUB_OUTPUT"
        echo "db=$(git log -1 --format=%h -- app/db)"   >> "$GITHUB_OUTPUT"
        echo "sha=$(git rev-parse HEAD)"                >> "$GITHUB_OUTPUT"
    - uses: docker/setup-buildx-action@v3
    - uses: docker/login-action@v3
      with: { registry: ghcr.io, username: "${{ github.actor }}",
              password: "${{ secrets.GITHUB_TOKEN }}" }
    - name: build api if missing
      run: |
        IMG=ghcr.io/${{ github.repository_owner }}/sheepshead-api:api-${{ steps.tags.outputs.api }}
        docker manifest inspect "$IMG" >/dev/null 2>&1 || \
          docker buildx build -f app/server/Dockerfile -t "$IMG" \
            --cache-from type=gha --cache-to type=gha,mode=max --push .
    - name: build web if missing
      run: |
        IMG=ghcr.io/${{ github.repository_owner }}/sheepshead-web:web-${{ steps.tags.outputs.web }}
        docker manifest inspect "$IMG" >/dev/null 2>&1 || \
          docker buildx build -t "$IMG" \
            --cache-from type=gha --cache-to type=gha,mode=max --push app/web
    - name: deploy over ssh
      uses: appleboy/ssh-action@v1
      with:
        host: ${{ secrets.DEPLOY_HOST }}
        username: deploy
        key: ${{ secrets.DEPLOY_SSH_KEY }}
        script: >-
          /opt/sheepsheadai/app/scripts/deploy.sh
          ${{ steps.tags.outputs.sha }}
          api-${{ steps.tags.outputs.api }}
          web-${{ steps.tags.outputs.web }}
          ${{ steps.tags.outputs.db }}
```

Also add to the workflow header:
`workflow_dispatch: { inputs: { ref: { description: "ref to deploy (rollback)", required: false } } }`,
and `permissions: { contents: read, packages: write }` for the job
(GHCR push with the built-in token). The `environment: production`
scopes the two secrets and (initially) can carry a required-reviewer
approval gate; drop the gate once trusted. The web build must receive
`NEXT_PUBLIC_API_BASE=""` (same-origin), which is the Dockerfile's
default.

### 6.3 `app/scripts/deploy.sh <sha> <api_tag> <web_tag> <db_sha>`

Runs on the box as the `deploy` user. Spec:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /opt/sheepsheadai
COMPOSE="docker compose --env-file app/deploy/.env.prod \
  --env-file app/deploy/.env.deploy --env-file app/deploy/model.env \
  -f app/docker-compose.prod.yml"

# 0. Remember previous deploy state (first deploy: file absent)
PREV_DB_SHA=$(grep -s '^DB_SHA=' app/deploy/.env.deploy | cut -d= -f2 || true)
PREV_SHA=$(git rev-parse HEAD)

# 1. Sync tree to the requested commit
git fetch origin && git checkout --force "$1"

# 2. Ensure the manifest's model is cached & verified (see §5.5)
source app/deploy/model.env
MODEL_FILE=/opt/sheepsheadai-models/${MODEL_ASSET_NAME}
if ! echo "${MODEL_SHA256}  ${MODEL_FILE}" | sha256sum -c --quiet 2>/dev/null; then
  tmp=$(mktemp -d)
  curl -fL --retry 3 -o "$tmp/m.pt" \
    "https://github.com/adamanthil/sheepsheadai/releases/download/${MODEL_ASSET_TAG}/${MODEL_ASSET_NAME}"
  echo "${MODEL_SHA256}  $tmp/m.pt" | sha256sum -c --quiet   # aborts on mismatch
  mv "$tmp/m.pt" "$MODEL_FILE"
fi

# 3. Write the machine-owned env file (never hand-edit)
printf 'API_TAG=%s\nWEB_TAG=%s\nDB_SHA=%s\nMODEL_FILE=%s\n' \
  "$2" "$3" "$4" "$MODEL_FILE" > app/deploy/.env.deploy

# 4. Migrate only if app/db changed (compat rule: §7 rule 2)
[ "$4" != "$PREV_DB_SHA" ] && $COMPOSE run --rm migrate

# 5. Converge — recreates ONLY services whose image/config changed
$COMPOSE pull -q
$COMPOSE up -d --remove-orphans

# 6. Caddyfile content changes don't trigger recreation (bind mount);
#    graceful reload preserves open websockets
git diff --quiet "$PREV_SHA" HEAD -- app/deploy/Caddyfile || \
  $COMPOSE exec caddy caddy reload --config /etc/caddy/Caddyfile

# 7. Wait for api health (built-in HEALTHCHECK, start_period 60s)
for i in $(seq 60); do
  st=$(docker inspect -f '{{.State.Health.Status}}' "$($COMPOSE ps -q api)")
  [ "$st" = healthy ] && exit 0
  sleep 3
done
echo "api failed to become healthy" >&2; exit 1
```

Failure handling: a non-zero exit fails the Actions run (notification,
§10). Recovery from a bad deploy is the rollback runbook (§8.3), not
automated revert — with one operator and near-zero traffic, explicit
beats clever. [rev] The `migrate` service runs `npm ci` inside
`node:22-alpine` on every db deploy (~1 min on 1 vCPU); acceptable at
this cadence, and prebuilding a migrate image would add a third tag for
no restart benefit.

### 6.4 One-time provisioning (Vultr box, Ubuntu 24.04 LTS)

1. `deploy` user (in `docker` group), SSH key-only auth, root login off.
2. ufw: allow 22/80/443. unattended-upgrades on.
3. **1 GB swapfile** (2 GB RAM + torch: this is the OOM insurance).
4. Docker Engine + compose plugin; `curl`.
5. `docker login ghcr.io` with a fine-grained PAT scoped to
   `read:packages` only. [rev] No `gh` CLI and no `contents` scope on
   the box: release assets are public.
6. `git clone --filter=blob:none <repo> /opt/sheepsheadai` (lazy blobs:
   the box never downloads training-artifact history).
7. `mkdir /opt/sheepsheadai-models`
8. `app/deploy/.env.prod` by hand: `DOMAIN`, `POSTGRES_USER`,
   `POSTGRES_PASSWORD`, `RESEND_API_KEY`, `EMAIL_FROM`. Secrets never
   transit CI.
9. DNS: `sheepshead.ai` A record → box IP. Caddy handles TLS
   automatically once reachable.
10. First deploy: run the workflow (or `deploy.sh` by hand with args
    from §3's commands run locally).
11. **Off-site backups (D8)**: install rclone, configure a B2/S3
    remote, cron nightly
    `rclone sync /opt/sheepsheadai/backups remote:sheepsheadai-backups`
    and `rclone sync /opt/sheepsheadai-models remote:sheepsheadai-models`.
12. Uptime monitor (UptimeRobot free) on `https://sheepshead.ai/health`
    — with sporadic usage nobody organically notices downtime.

GitHub side: create the `production` environment with secrets
`DEPLOY_HOST`, `DEPLOY_SSH_KEY` (a dedicated keypair; public half in
the box's `authorized_keys`); enable immutable releases; add the
`artifact/*` tag ruleset (D11).

## 7. Invariants (write these into docs/deploy.md too)

1. **Never build images on the box.** Even for debugging. A torch build
   on 1 vCPU grinds ~20 min and the memory pressure can OOM the api
   mid-game. The box pulls; CI builds.
2. **A db-only migration must be compatible with the running API
   code**, because the api deliberately does not restart (its process
   may run against the new schema indefinitely). Migrations that ship
   in the same commit range as api changes are exempt — the api
   recreates in the same deploy.
3. **Exactly one api container, one uvicorn worker.** Unchanged from
   the compose file's warnings; nothing in CD may scale it out.
4. **`.env.deploy` is machine-owned** (written by deploy.sh);
   `.env.prod` is human-owned (secrets); `model.env` and
   `models/manifest.json` are git-owned and CI keeps them consistent.
   Never cross the streams.
5. **Artifact releases are immutable**: never re-upload an asset under
   an existing `artifact/<name>` tag — mint a new name. The sha256 pin
   turns violations into loud deploy failures; the repository setting
   and tag ruleset (D11) make them impossible by accident.
6. **Python-code deploys interrupt live games** (D7). Deploy off-peak
   when it matters; the restart matrix (§8.4) says when it applies.
7. **Panel membership is frozen by hash.** [rev] A panel read against a
   file that does not hash to the manifest is not a panel read; the
   loader refuses rather than records.

## 8. Operator runbooks

### 8.1 Deploy code
Push to the default branch (or merge). CI green → deploy runs. Nothing
else.

### 8.2 Publish artifacts (bootstrap, once) and deploy a new model

```bash
# Bootstrap: the seven panel agents in one immutable release
mkdir -p /tmp/panels && cd /tmp/panels
cp final_pfsp_swish_ppo.pt pfsp-swish-30m.pt            # … one copy per §5.1 row, renamed
gh release create artifact/panels-2026-10 *.pt \
    --title "artifact: evaluation panels (PANEL-A, PANEL-B), 2026-10" \
    --notes "Frozen panel members; provenance per models/manifest.json. Immutable."

# The deployed model
cp runs/202609_recall_rc/final/release.pt recall-rc-theta6.pt
gh release create artifact/recall-rc-theta6 recall-rc-theta6.pt \
    --title "artifact: recall-rc-theta6" \
    --notes "202609_recall_rc release (theta_6); +0.046 vs v2 release, +0.092 vs 30M"

# Then: fill models/manifest.json (shasum -a 256 each file), write
# app/deploy/model.env to match, commit (own commit), push. CD does the rest.
uv run python -m sheepshead.analysis.fetch_models --check   # proves the manifest
```

Verify after a model deploy: `https://sheepshead.ai/health` reports the
new label and sampling label. A later model: one new
`artifact/<name>` release, one manifest + `model.env` commit.

### 8.3 Roll back
Actions → CI → "Run workflow" → `ref` = last good commit sha. Images
already exist for old tags, so this skips builds. For a bad model:
revert the manifest commit instead (or include it in the ref).

### 8.4 What restarts when

| Commit touches                          | api (games!) | web | caddy | migrate |
|-----------------------------------------|--------------|-----|-------|---------|
| `sheepshead/analysis|training|validation` (excl. the two files), notebooks, docs, `models/manifest.json` panel entries | — | — | — | — |
| `app/web/**` only                        | —            | ✓   | —     | —       |
| `api-paths.txt` closure (server py + tests, engine, agent, lockfile) | ✓ | — | — | — |
| `app/db/**` only                         | —            | —   | —     | ✓       |
| `app/deploy/Caddyfile`                   | —            | —   | graceful reload (WS survive) | — |
| `app/deploy/model.env` (weights or sampling) | ✓ (new weights / new ai_player) | — | — | — |
| compose env/config for a service         | that service | …   | …     | —       |

### 8.5 Disaster recovery (box dies)
Provision per §6.4 on a fresh box, restore latest dump from the rclone
remote into postgres (`docker compose run --rm ... psql < dump`),
repoint DNS, run the deploy workflow. Nothing else lives on the box:
code = git, images = GHCR, models = releases (+ rclone mirror +
`model-history/`), secrets = `.env.prod` (keep an offline copy), db =
backups.

### 8.6 Run the panels from a fresh clone [rev]

```bash
git clone https://github.com/adamanthil/sheepsheadai && cd sheepsheadai
uv sync --frozen --extra dev
uv run python -m sheepshead.analysis.fetch_models --role panel-a --role panel-b
uv run python -m sheepshead.analysis.rigorous_eval ... --panel PANEL-A
```

Same deal seed and deal hash as every row in `notebooks/panel_reference.csv`.

## 9. Implementation order (each its own commit, per house rules)

0. **Rename the default branch `master` → `main`** (D9; before any CD
   work, so nothing below is ever wired to the old name):
   * On GitHub: `gh api -X POST repos/{owner}/{repo}/branches/master/rename -f new_name=main`
     (GitHub retargets open PRs and branch protection automatically,
     and old-name pushes/fetches get a redirect notice).
   * Locally: `git branch -m master main && git fetch origin &&
     git branch -u origin/main main &&
     git remote set-head origin -a`.
   * In-repo references (one commit): `.github/workflows/ci.yml`
     push trigger `branches: [master]` → `[main]`;
     `docs/database-migrations.md` line mentioning `master`.
   * Any other clones/worktrees (training worktrees, `sheepsheadai-rc`)
     repeat the local steps on next use.
1. `api-paths.txt` + `check_api_paths.py` + CI guard step (safe,
   standalone; proves the closure list before anything depends on it).
2. `models/manifest.json` + `fetch_models.py` + `check_model_manifest.py`
   + CI step; `panels.py` → manifest keys and its callers; **publish the
   two bootstrap releases first** (§8.2) so `--check` passes in CI
   (the deploy entry's `sampling` block is the operator's 10-27
   decision, §5.2); then
   `git rm final_pfsp_swish_ppo.pt` and re-point its references to
   `models/pfsp-swish-30m.pt` (D5) in the same commit, so the tree never
   has both copies.
3. `app/deploy/model.env`.
4. Compose: `image:` switch, `db-backup` client version, build override
   file, `.env.prod.example` trim.
5. `/health` sampling label + OpenAPI/TS regen.
6. `deploy.sh`.
7. `deploy` job in `ci.yml` (gated on the `production` environment —
   inert until the secrets exist).
8. `docs/deploy.md` rewrite.
9. GitHub settings: immutable releases, `artifact/*` tag ruleset,
   `production` environment + secrets.
10. Provision box (§6.4), first deploy, DNS cutover.

Steps 1–2 are valuable on their own even if the box is never
provisioned: they make the panels reproducible for anyone and put the
release checkpoint somewhere that is not one disk under `runs/`.

## 10. Deferred / future work

* **Drain-before-deploy**: admin endpoint flipping `set_draining()`;
  deploy.sh (behind `DRAIN_ON_DEPLOY=1`) calls it and polls table count
  with a timeout before `up -d` when the api will restart. Do when
  there's traffic worth protecting.
* **Game-state snapshot/restore across restarts** — the only true fix
  for D7; large; revisit only if deploys visibly annoy players.
* **Fly.io scale-to-zero migration** (D1 fallback) if usage data shows
  the box ~always idle: same containers, Fly proxy replaces Caddy,
  Neon replaces the postgres container. Revisit-not-before: 3 months of
  real traffic.
* **Release-checkpoint regression fixture** (load `recall-rc-theta6`
  through the app's load path, fixed probe states, bit-identical
  logits; skipped when the asset is absent). Declined 10-27 as not
  worth carrying; the manifest hash check covers the bytes, not the
  load path. Re-raise if a checkpoint-load regression ever bites.
* Staging environment: overkill for now; CI's real-migration +
  container-boot coverage is the gate.
* Further artifacts (future releases, new panel members) follow §5.1
  with new names; a PANEL-C would be a new `panels` list, never an edit
  of A or B.
