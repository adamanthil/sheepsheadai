# Continuous Deployment to sheepshead.ai — Plan (August 2026)

**Status (2026-08-06): PLAN ONLY — no code written, no server provisioned.
All decisions below are final (agreed with the operator); a future session
can implement this without further design work. Judgment calls are marked
D1–D8 with rationale. Companion doc: `docs/deploy.md` (manual deploy —
several sections there are superseded by this plan and should be updated
during implementation, see §9).**

## TL;DR — what this is

Ship the `app/` stack (Next.js + FastAPI + Postgres behind Caddy, see
`app/docker-compose.prod.yml`) to a $10/mo Vultr VPS in Chicago, with
GitHub Actions continuous deployment on every push to `main` (the
default branch is renamed from `master` before any CD work — §9 step 0).
Three
properties the design guarantees:

1. **The Python API container is NOT restarted unless code it actually
   runs has changed.** Live games hold all state in that one process;
   research commits (`sheepshead/analysis/`, `training/`, notebooks)
   must not kill them. Mechanism: per-component image tags derived from
   `git log` over an explicit path list, plus docker compose's native
   "only recreate changed services" behavior (§3).
2. **Images are built in CI, never on the box.** The 1 vCPU / 2 GB VPS
   only ever pulls and runs.
3. **Model checkpoints are distributed as GitHub Release assets**, pinned
   by a committed manifest (label + URL + sha256). Deploying a new model
   is a two-step: upload release, commit manifest (§5).

## 0. Decisions and rationale

* **D1 — Host: Vultr, Chicago (ORD), 1 vCPU / 2 GB / 55 GB, ~$10/mo.**
  Primary audience is Wisconsin (~10 ms RTT). Hetzner's US locations lost
  their price edge in 2026 ($13.49 for the same specs); EU locations add
  ~120 ms to every WebSocket round trip. Serverless (Lambda) would
  require externalizing the in-process game state — a rearchitecture
  with no payoff at this price point. Fly.io scale-to-zero was
  considered and rejected for vendor risk; it remains the documented
  cost fallback if real usage shows the box idle 95%+ of the time.
* **D2 — Per-component deploy tags from git path history** (§3), not
  reproducible-build digest comparison (Docker builds aren't
  reproducible) and not a hand-tracked "last deployed" state file
  (stateless is simpler and rollback-friendly).
* **D3 — The api path filter is narrowed inside `sheepshead/`** to the
  server's actual import closure, and a CI guard recomputes that closure
  and fails if the committed list is stale (§4). Without the guard,
  narrowing risks silently missing restarts after a reorg; with it,
  staleness is a loud CI failure. (The closure today already includes
  two non-obvious files: `training/reward_shaping.py` — imported
  directly by the server — and `training/training_utils.py` — pulled in
  via `agent/ppo.py`.)
* **D4 — No `.dockerignore` narrowing of the image contents.** It would
  have to be kept in sync with `api-paths.txt` by hand; the dead
  `analysis/` code in the image is harmless. Accepted consequence: two
  images with the same api tag can differ in unreachable code.
* **D5 — Models via GitHub Release assets from day one**, not committed
  `.pt` blobs. The manifest commit (not the release upload) is what
  deploys. Note: `final_pfsp_swish_ppo.pt` at the repo root **stays in
  git** — it is the frozen PANEL-A anchor for `rigorous_eval` and other
  research tooling. It just stops being a distribution channel; the
  production copy of the same weights is a release asset.
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
  land on the same disk that would die with the box.

## 1. Current state (what already exists)

* `app/docker-compose.prod.yml`: caddy (TLS, same-origin routing) → web
  (Next.js) + api (FastAPI) → postgres, plus one-shot `migrate` service
  (graphile-migrate) and a `db-backup` sidecar writing daily dumps to
  repo-root `backups/`.
* `app/server/Dockerfile`: python:3.14-slim + uv, CPU-only torch,
  **exactly one uvicorn worker** (all game state in-process — never
  scale out), built-in `HEALTHCHECK` against `/health`,
  25 s graceful shutdown; compose adds `stop_grace_period: 30s`.
* The model checkpoint is a read-only bind mount
  (`SHEEPSHEAD_MODEL_PATH=/models/model.pt`); `SHEEPSHEAD_MODEL_LABEL`
  is required at boot, upserted into the DB (`_upsert_ai_model`) so
  every recorded game is attributed to the model that played it.
* CI (`.github/workflows/ci.yml`): three jobs — `server` (ruff, pytest
  against a postgres service with real migrations applied, OpenAPI
  drift check), `training`, `web` (typecheck/lint/build, generated-API
  drift check).

## 2. Target architecture

```
push to main ────► CI (server / training / web jobs, unchanged)
                        │ all green
                        ▼
                   deploy job (GitHub Actions, environment: production)
                        │ 1. compute API/WEB/DB tags from git paths
                        │ 2. build+push ghcr.io images IF tag missing
                        │ 3. ssh vultr → app/scripts/deploy.sh <args>
                        ▼
   Vultr ORD box: /opt/sheepsheadai (git checkout, --filter=blob:none)
                  /opt/sheepsheadai-models/<label>.pt   (release assets)
                  docker compose pull + up -d   (recreates ONLY changed services)
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
covers `agent/architectures/`.)

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

## 5. Model distribution: GitHub Release assets + committed manifest

**Release = blob store; manifest commit = the deploy.** Code deploys
never create releases; releases are created only when promoting a model.

* Tag namespace: `model/<label>`, e.g. `model/pfsp-swish-30m`. The tag's
  commit is irrelevant; nobody checks out model tags. The asset filename
  is `<label>.pt` and the label doubles as `SHEEPSHEAD_MODEL_LABEL`.
* Committed manifest `app/deploy/model.env` (read by compose as an
  env file AND parsed by the deploy script):

  ```
  SHEEPSHEAD_MODEL_LABEL=pfsp-swish-30m
  MODEL_ASSET_TAG=model/pfsp-swish-30m
  MODEL_SHA256=<sha256 of the .pt>
  ```

* Promoting a new model (operator runbook, §8.2):

  ```bash
  sha256sum league-gen3.pt                       # note the hash
  gh release create model/league-gen3 league-gen3.pt \
      --title "model: league-gen3" --notes "provenance: <run/checkpoint>"
  # then edit app/deploy/model.env (all three lines), commit (its own
  # commit), push. CD does the rest.
  ```

* Deploy-script side (§6): models are cached under
  `/opt/sheepsheadai-models/` (OUTSIDE the checkout — survives
  `git checkout --force`). If `<label>.pt` is absent or its sha256
  doesn't match the manifest, download via
  `gh release download "$MODEL_ASSET_TAG" --pattern '*.pt'` into a temp
  file, verify sha256, `mv` atomically into place. **Hash mismatch
  aborts the deploy loudly** — this also defends against a re-uploaded
  asset.
* Compose wiring: the api service mounts
  `${MODEL_FILE}:/models/model.pt:ro` where `MODEL_FILE` is the
  absolute cache path, written by the deploy script into
  `.env.deploy` (§6). A model change therefore changes the api
  service's mount source + `SHEEPSHEAD_MODEL_LABEL` env → compose
  recreates the api container (necessary — that's how weights load)
  and nothing else. `api-paths.txt` deliberately does NOT include
  `app/deploy/model.env`: a model swap needs no image rebuild.
* CI manifest check (step in the `server` job): parse `model.env`,
  assert all three keys present, label == asset tag suffix, and
  `gh release view "$MODEL_ASSET_TAG"` succeeds (works with the
  built-in `GITHUB_TOKEN`). Optionally (stricter, slower): download and
  `torch.load` it against the architecture registry so an incompatible
  checkpoint fails CI, not the boot on the box.
* Rollback: revert the manifest commit (old asset still exists) or
  workflow_dispatch at the old ref.
* Bootstrap: create `model/pfsp-swish-30m` from
  `final_pfsp_swish_ppo.pt` (same bytes; keep the repo file — D5).

## 6. The moving parts to implement

### 6.1 Repo changes

1. `app/docker-compose.prod.yml`: api/web services switch from `build:`
   to `image: ghcr.io/<owner>/sheepshead-api:${API_TAG}` /
   `...-web:${WEB_TAG}`; api's model mount becomes
   `${MODEL_FILE}:/models/model.pt:ro`; drop the hand-set
   `SHEEPSHEAD_MODEL_LABEL` from `.env.prod` docs (now in
   `model.env`). Keep a `docker-compose.build.yml` override with the
   `build:` blocks for local use.
2. New: `app/deploy/api-paths.txt` (§3), `app/deploy/model.env` (§5).
3. New: `app/scripts/check_api_paths.py` (§4) + CI steps in the
   `server` job: closure guard + manifest check.
4. New: `app/scripts/deploy.sh` (§6.3).
5. `.github/workflows/ci.yml`: add the `deploy` job (§6.2).
6. `docs/deploy.md`: rewrite "Deploying an update" to point here; the
   initial-provisioning section gains §6.4's steps.

### 6.2 The deploy job (append to `ci.yml`)

```yaml
deploy:
  needs: [server, training, web]
  if: >-
    (github.event_name == 'push' && github.ref == 'refs/heads/main')
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
`workflow_dispatch: { inputs: { ref: { description: "ref to deploy (rollback)", required: false } } }`.
The `environment: production` scopes the two secrets and (initially)
can carry a required-reviewer approval gate; drop the gate once trusted.

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

# 2. Ensure the manifest's model is cached & verified (see §5)
source app/deploy/model.env
MODEL_FILE=/opt/sheepsheadai-models/${SHEEPSHEAD_MODEL_LABEL}.pt
if ! echo "${MODEL_SHA256}  ${MODEL_FILE}" | sha256sum -c --quiet 2>/dev/null; then
  tmp=$(mktemp -d)
  gh release download "$MODEL_ASSET_TAG" --pattern '*.pt' -O "$tmp/m.pt"
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
beats clever.

### 6.4 One-time provisioning (Vultr box, Ubuntu 24.04 LTS)

1. `deploy` user (in `docker` group), SSH key-only auth, root login off.
2. ufw: allow 22/80/443. unattended-upgrades on.
3. **1 GB swapfile** (2 GB RAM + torch: this is the OOM insurance).
4. Docker Engine + compose plugin.
5. `gh` CLI; `gh auth login` with a fine-grained PAT: `contents: read`
   (release assets) + `read:packages` (GHCR). Same PAT does
   `docker login ghcr.io`.
6. `git clone --filter=blob:none <repo> /opt/sheepsheadai` (lazy blobs:
   the box never downloads training-artifact history).
7. `mkdir /opt/sheepsheadai-models`
8. `app/deploy/.env.prod` by hand: `DOMAIN`, `POSTGRES_USER`,
   `POSTGRES_PASSWORD`. (Model vars now come from `model.env` +
   `.env.deploy`.) Secrets never transit CI.
9. DNS: `sheepshead.ai` A record → box IP. Caddy handles TLS
   automatically once reachable.
10. First deploy: run the workflow (or `deploy.sh` by hand with args
    from §3's commands run locally).
11. **Off-site backups (D8)**: install rclone, configure a B2/S3
    remote, cron nightly
    `rclone sync /opt/sheepsheadai/backups remote:sheepsheadai-backups`.
    The `db-backup` container already produces the dumps.
12. Uptime monitor (UptimeRobot free) on `https://sheepshead.ai/health`
    — with sporadic usage nobody organically notices downtime.

GitHub side: create the `production` environment with secrets
`DEPLOY_HOST`, `DEPLOY_SSH_KEY` (a dedicated keypair; public half in
the box's `authorized_keys`).

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
   `.env.prod` is human-owned (secrets); `model.env` is git-owned.
   Never cross the streams.
5. **Model releases are immutable**: never re-upload an asset under an
   existing `model/<label>` tag — mint a new label. The sha256 pin
   turns violations into loud deploy failures.
6. **Python-code deploys interrupt live games** (D7). Deploy off-peak
   when it matters; the restart matrix (§8.4) says when it applies.

## 8. Operator runbooks

### 8.1 Deploy code
Push to `main` (or merge). CI green → deploy runs. Nothing else.

### 8.2 Deploy a new model
§5's three commands: `sha256sum`, `gh release create model/<label>`,
then commit `app/deploy/model.env` (own commit) and push. Verify after:
`https://sheepshead.ai/health` reports the new label.

### 8.3 Roll back
Actions → CI → "Run workflow" → `ref` = last good commit sha. Images
already exist for old tags, so this skips builds. For a bad model:
revert the `model.env` commit instead (or include it in the ref).

### 8.4 What restarts when

| Commit touches                          | api (games!) | web | caddy | migrate |
|-----------------------------------------|--------------|-----|-------|---------|
| `sheepshead/analysis|training|validation` (excl. the two files), notebooks, docs | — | — | — | — |
| `app/web/**` only                        | —            | ✓   | —     | —       |
| `api-paths.txt` closure (server py, engine, agent, lockfile) | ✓ | — | — | — |
| `app/db/**` only                         | —            | —   | —     | ✓       |
| `app/deploy/Caddyfile`                   | —            | —   | graceful reload (WS survive) | — |
| `app/deploy/model.env`                   | ✓ (new weights) | — | —  | —       |
| compose env/config for a service         | that service | …   | …     | —       |

### 8.5 Disaster recovery (box dies)
Provision per §6.4 on a fresh box, restore latest dump from the rclone
remote into postgres (`docker compose run --rm ... psql < dump`),
repoint DNS, run the deploy workflow. Nothing else lives on the box:
code = git, images = GHCR, models = releases, secrets = `.env.prod`
(keep an offline copy), db = backups.

## 9. Implementation order (each its own commit, per house rules)

0. **Rename the default branch `master` → `main`** (before any CD
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
   * Any other clones/worktrees (e.g. training boxes) repeat the
     local steps on next use.
1. `api-paths.txt` + `check_api_paths.py` + CI guard step (safe,
   standalone; proves the closure list before anything depends on it).
2. `model.env` + CI manifest check + create the bootstrap release
   `model/pfsp-swish-30m` from `final_pfsp_swish_ppo.pt`.
3. Compose: `image:` switch + model-mount change + build override file.
4. `deploy.sh`.
5. `deploy` job in `ci.yml` (gated on the `production` environment —
   inert until the secrets exist).
6. `docs/deploy.md` update.
7. Provision box (§6.4), first deploy, DNS cutover.

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
* Staging environment: overkill for now; CI's real-migration +
  container-boot coverage is the gate.
