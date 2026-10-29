# Continuous Deployment to sheepshead.ai — Plan (August 2026, revised October 2026)

**Status (2026-10-27): PLAN ONLY — no code written, no server
provisioned.** Revised twice on 10-27: once after the `202609_recall_rc`
program concluded (marked **[rev]**), then after an audit against the
repository (marked **[audit]**). The audit fixed four defects in the
deploy mechanics (floating third-party tags restarted by `pull`, a
migration that was never retried after failure, a deploy script that
ran one deploy stale, short-sha tags that change length), filled
operational gaps (disk, ssh, backups, rollback caveats), and simplified
(content-hash tags, always-run migrate, one model description instead
of two, public images).

**Prerequisite: `notebooks/Model_Artifacts_Plan_202610.md`** — the
model-artifact work (release assets, `models/manifest.json`,
`fetch_models`, panels by hash) moved there [audit]. This plan consumes
its manifest; it does not define it. Companion doc: `docs/deploy.md`
(manual deploy; stale — it still describes a Hetzner box that builds
images locally — and is rewritten in §9).

## TL;DR — what this is

Ship the `app/` stack (Next.js + FastAPI + Postgres behind Caddy, see
`app/docker-compose.prod.yml`) to a $10/mo Vultr VPS in Chicago, with
GitHub Actions continuous deployment on every push to `main` (the
default branch is renamed from `master` before any CD work — §9 step
0). Four properties the design guarantees:

1. **Nothing a live game depends on restarts unless it actually
   changed.** Live games hold all state in the one api process and in
   WebSockets through Caddy; research commits (`sheepshead/research/`,
   `training/`, notebooks) must not kill either. Mechanism: per-service
   image tags hashed from the contents of an explicit path list, only
   our own images are pulled, third-party images are pinned, and docker
   compose's native "only recreate changed services" behavior (§3).
   [audit: the pull/pin half]
2. **Images are built in CI, never on the box.** The 1 vCPU / 2 GB VPS
   only ever pulls and runs.
3. **The deployed model is whatever `models/manifest.json` gives the
   `deploy` role** (Model Artifacts plan). Deploying a new model is a
   release upload plus a manifest commit; the box reads the manifest
   directly (§5). [audit]
4. **Per-checkpoint deployment sampling travels with the model.** [rev]
   The four head temperatures and the logit band (Training_Program_
   Redesign §4.5, app migration 000008) are on the manifest's deploy
   entry, not hand-set on the box (§5).

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
* **D2 — Per-service tags hashed from file contents** (§3). [audit]
  The tag is a hash of `git ls-tree` over the service's paths, not the
  sha of the last commit touching them: the August `git log -1
  --format=%h` scheme broke when git lengthens abbreviations (the repo
  is at ~12k objects; abbreviations go from 7 to 8 chars around 16k,
  which would change every tag at once and restart the api for
  nothing, and a clone with a different object count — the box's
  blobless clone, a laptop — computes different strings). Content
  hashes are length-stable, need no history (`fetch-depth: 0` is
  gone), and a revert reproduces the old tag, so reverting reuses the
  old image instead of rebuilding. Not reproducible-build digest
  comparison (Docker builds aren't reproducible) and not a
  hand-tracked "last deployed" state file.
* **D3 — The api path filter is narrowed inside `sheepshead/`** to the
  server's actual import closure, and a CI guard recomputes that closure
  and fails if the committed list is stale (§4). Without the guard,
  narrowing risks silently missing restarts after a reorg; with it,
  staleness is a loud CI failure. This is the most intricate piece of
  the plan and it earns its keep: most commits in this repository are
  research commits that must not restart games. [rev] The server's
  direct `sheepshead` imports today are `game`, `views`, `agent.ppo`,
  `agent.observation`, `agent.sampling`, `agent.convention_wrapper`,
  `agent.traced_game`, `agent` (the package) and `rewards`. [2026-10 restructure] The
  package layering test (`sheepshead/tests/test_import_layering.py`)
  now enforces that the app imports only the core and agent layers, so
  the closure cannot reach training code. The guard still computes the
  closure, so this list is informational.
* **D4 — No `.dockerignore` narrowing of the image contents.** It would
  have to be kept in sync with `api-paths.txt` by hand; the dead
  research and training code in the image is harmless. Accepted consequence: two
  images with the same api tag can differ in unreachable code.
* **D5 — Model artifacts: see the Model Artifacts plan** (formerly D5,
  D10–D12 here). [audit] This plan relies on exactly three of its
  properties: the manifest has exactly one `deploy` entry with a
  `sampling` block (CI-linted), assets are immutable, and assets are
  public HTTPS downloads.
* **D6 — Migrations run on every deploy.** [audit] graphile-migrate is
  a no-op when nothing is pending, so the August "only when `app/db`
  changed" bookkeeping (`DB_SHA`, `PREV_DB_SHA`) bought nothing and had
  a bug: it recorded the new `DB_SHA` before migrating, so a failed
  migration was skipped on the retry. The compatibility rule stands: a
  migration that lands without an api change must work with the
  running, not-restarted api process (§7 rule 2).
* **D7 — v1 deploys do NOT drain.** A Python-code deploy interrupts any
  live games (in-memory state; unavoidable without snapshot/restore).
  Traffic will be near-zero at launch; the existing
  `set_draining()` flag (`app/server/runtime/lifecycle.py`) is the
  future hook (§10).
* **D8 — Off-site backups that survive the box and survive mistakes.**
  The Vultr plan has no provider backups; the `db-backup` container's
  daily dumps land on the same disk that would die with the box. rclone
  pushes them, plus the model cache, to a **Backblaze B2** bucket.
  [audit] B2 because the data (megabytes of dumps plus ~21 MB per
  deployed model) fits its 10 GB free tier, it is a different vendor
  from the box (a Vultr outage, billing problem or account suspension
  cannot take the backups with it — which rules out Vultr Object
  Storage), it keeps prior file versions by default, and rclone has a
  native `b2` backend with a single bucket-scoped application key — no
  IAM policy as on S3. With
  three guards the August `rclone sync` lacked: `rclone copy` into a
  **versioned bucket with a lifecycle rule** (a `sync` would copy an
  emptied or corrupted `backups/` to the remote and delete the good
  copies); a **`crypt` remote** (dumps hold PII: account emails and
  `player.last_ip`); and a **dead-man ping** at the end of the cron job
  (a silently failing backup cron is the normal failure mode). A
  restore drill is part of provisioning (§6.4), not an afterthought.
* **D9 — Rename the default branch `master` → `main` before any CD
  work** (August decision, reaffirmed by the operator 10-27). Cost:
  every local clone and worktree re-points once (§9 step 0 lists the
  commands; the training worktrees and `sheepsheadai-rc` repeat them on
  next use). The deploy job triggers on
  `github.event.repository.default_branch` rather than a literal name,
  so the workflow is correct before and after the rename.
* **D10 — Third-party images are pinned to exact versions, and deploys
  pull only our own images.** [audit] The compose file uses floating
  tags (`caddy:2`, `postgres:18`, `node:22-alpine`), and the August
  deploy script pulled everything on every deploy — including
  notebook-only commits. An upstream patch release would then recreate
  Caddy (dropping every open WebSocket) or Postgres (yanking the db
  from under the api): exactly the restart property #1 promises not to
  cause. Pins (`caddy:2.x.y`, `postgres:18.x`, `node:22.x-alpine`, the
  `db-backup` image, and the `uv` image copied in the api Dockerfile)
  are bumped in their own commits, which then restart that service
  deliberately (§8.4).
* **D11 — GHCR packages are public; the box holds no registry
  credential.** [audit] The August plan put a fine-grained PAT with
  `read:packages` on the box, but GHCR accepts only classic PATs, which
  are broad and expire. The repository is public, so the images may as
  well be: set both packages public once after their first push (§9).
  Nothing to rotate, nothing to leak.
* **D12 — The deploy key can only deploy.** [audit] The `deploy` user
  is in the `docker` group, which is root-equivalent, so its key is
  restricted in `authorized_keys` to a forced command
  (`restrict,command="/opt/sheepsheadai/app/scripts/deploy.sh"`) that
  validates its three arguments. CI connects with plain `ssh` and a
  pinned `known_hosts` (a `DEPLOY_KNOWN_HOSTS` secret), not a
  third-party action that is handed the key and by default skips host
  verification.

## 1. Current state (what already exists, 2026-10-27)

* `app/docker-compose.prod.yml`: caddy (TLS, same-origin routing) → web
  (Next.js) + api (FastAPI) → postgres, plus one-shot `migrate` service
  (graphile-migrate, `profiles: ["tools"]`) and a `db-backup` sidecar
  writing daily dumps to repo-root `backups/` (gitignored). The api
  service already takes `MODEL_FILE` as a read-only mount source and
  the five sampling env vars (`SHEEPSHEAD_TEMPERATURE_{PICK,PARTNER,
  BURY,PLAY}`, `SHEEPSHEAD_LOGIT_BAND`). [rev] [audit] Gaps found:
  third-party tags float; no `logging:` limits (Docker's json-file logs
  grow without bound); `db-backup` is pinned to the `:17` client, which
  cannot dump a Postgres 18 server.
* `app/server/Dockerfile`: python:3.14-slim + uv (copied from
  `ghcr.io/astral-sh/uv:latest` — floating [audit]), CPU-only torch,
  **exactly one uvicorn worker** (all game state in-process — never
  scale out), built-in `HEALTHCHECK` against `/health`, 25 s graceful
  shutdown; compose adds `stop_grace_period: 30s`.
* The model checkpoint is a read-only bind mount
  (`SHEEPSHEAD_MODEL_PATH=/models/model.pt`); `SHEEPSHEAD_MODEL_LABEL`
  is required at boot and upserted into `ai_model`; the AI player's
  identity is (model, five sampling columns) in `ai_player` (migration
  000008), so a re-tempered deployment is a different player for stats.
  `/health` returns `status`, `db` and `model` (the label). [rev]
* CI (`.github/workflows/ci.yml`): **four** jobs [rev] — `lint` (prek:
  ruff format/check + whole-project basedpyright), `server` (compileall,
  real migrations against postgres:18, pytest, OpenAPI drift check),
  `training` (sheepshead tests), `web` (generated-API drift, typecheck,
  lint, prettier, build). Triggers: push to `master`, pull requests.
* `.gitignore` covers `app/deploy/.env.prod` and `backups/`, not the
  `.env.deploy` this plan introduces. [audit]

## 2. Target architecture

```
push to default branch ──► CI (lint / server / training / web, unchanged)
                               │ all green
                               ▼
                          images job: compute content tags, build+push to GHCR if missing
                               │
                               ▼
                          deploy job (environment: production)
                               │ ssh deploy@box "<sha> <api_tag> <web_tag>"
                               ▼
   Vultr ORD box: forced command → app/scripts/deploy.sh
                  /opt/sheepsheadai (git checkout, --filter=blob:none)
                  /opt/sheepsheadai-models/<asset>.pt  (from models/manifest.json, sha256-verified)
                  docker compose pull api web + migrate + up -d  (recreates ONLY changed services)
```

## 3. Core mechanism: content-derived per-service tags [audit]

`app/scripts/deploy_tags.sh` — the one place tags are computed; CI and
a hand deploy run the same script, so they cannot disagree:

```bash
#!/usr/bin/env bash
# Content-derived image tags: the same files give the same tag at any commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tag() { git ls-tree -r HEAD -- "$@" | git hash-object --stdin | cut -c1-12; }
echo "api=$(tag $(cat app/deploy/api-paths.txt))"
echo "web=$(tag app/web)"
echo "sha=$(git rev-parse HEAD)"
```

(`git hash-object` rather than `sha256sum`: identical on macOS and
Linux with no extra tool.)

Properties this buys:

* **Deterministic & stateless** — recomputable at any commit from the
  tree alone; no history needed, no "last deployed" bookkeeping.
* **Build skipping** — if `ghcr.io/...:<tag>` already exists
  (`docker manifest inspect` succeeds), skip the build entirely.
* **No-op restarts** — a commit touching only `app/web` leaves the api
  tag unchanged; `docker compose up -d` sees an identical image
  reference + config for the api service and does not touch the
  running container. Games survive.
* **Rollback reuses images** — redeploying an old ref, or reverting a
  commit, reproduces the old tag; the image already exists in GHCR.

`app/deploy/api-paths.txt` (initial contents — CI-verified, see §4):

```
app/server
pyproject.toml
uv.lock
sheepshead/__init__.py
sheepshead/game.py
sheepshead/agent
sheepshead/rewards.py
sheepshead/views.py
```

(`app/server` covers `app/server/Dockerfile`. `sheepshead/agent`
covers `agent/architectures/` and `agent/sampling.py`.) [rev] A change
under `app/server/tests` rebuilds the api image — accepted (D4's logic)
rather than excluding tests, which would make the list a maintenance
item. A listed path that does not exist would silently hash as empty;
the guard's existence check (§4 step 5) is what prevents that.

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

## 5. Model deploy: the box reads the manifest [audit]

There is no `app/deploy/model.env` and no `check_model_manifest.py`:
the box has `jq` (one apt package, §6.4) and reads
`models/manifest.json` from its checkout, so the model is described in
exactly one file. The manifest's shape — exactly one `deploy` entry,
a five-key `sampling` block, 64-hex hashes — is enforced by
`fetch_models --lint` in CI (Model Artifacts plan §4).

`deploy.sh` (§6.3) selects the `deploy` entry, ensures
`/opt/sheepsheadai-models/<asset>` exists and hashes to the manifest
(downloading `<release_base>/<tag>/<asset>` with `curl -fL` into a temp
file **in that directory**, so the final `mv` is a same-filesystem
atomic rename), and writes the label, `MODEL_FILE` and the five
sampling values into the machine-owned `.env.deploy`. **Hash mismatch
aborts the deploy loudly** — this also defends against a re-uploaded
asset. The models directory lives OUTSIDE the checkout (survives
`git checkout --force`); the nightly rclone job (D8) copies it
alongside `backups/`.

A model change (weights or sampling) changes the api service's mount
source and/or environment → compose recreates the api container
(necessary — that's how weights load) and nothing else. `api-paths.txt`
deliberately does NOT include `models/manifest.json`: a model swap
needs no image rebuild. Changes to panel entries in the manifest leave
`.env.deploy` byte-identical and restart nothing.

## 6. The moving parts to implement

### 6.1 Repo changes

1. `app/docker-compose.prod.yml`:
   * api/web switch from `build:` to
     `image: ghcr.io/adamanthil/sheepshead-api:${API_TAG}` /
     `...-web:${WEB_TAG}`; keep a `docker-compose.build.yml` override
     with the `build:` blocks for local use.
   * [audit] Pin `postgres`, `caddy`, `node` (migrate) to exact
     versions (D10); bump `db-backup` to a `postgres-backup-local` tag
     whose `pg_dump` is ≥ 18 [rev].
   * [audit] An `x-logging` anchor
     (`json-file`, `max-size: 10m`, `max-file: "3"`) applied to every
     service.
   * [audit] A named `npm-cache` volume at `/root/.npm` on `migrate`,
     so the always-run `npm ci` (D6) installs from local cache instead
     of the network.
   * `.env.prod.example` drops `SHEEPSHEAD_MODEL_LABEL` and
     `MODEL_FILE` (now written into `.env.deploy`).
2. `app/server/Dockerfile`: pin the `uv` image to an exact version
   (D10). [audit]
3. `.gitignore`: add `app/deploy/.env.deploy`. [audit]
4. New: `app/deploy/api-paths.txt` (§3),
   `app/scripts/check_api_paths.py` (§4),
   `app/scripts/deploy_tags.sh` (§3), `app/scripts/deploy.sh` (§6.3).
5. `.github/workflows/ci.yml`: the guard step in `server`; the
   `images` and `deploy` jobs (§6.2); a `workflow_dispatch` trigger.
6. `/health` reports the sampling label beside the model label [rev]
   (`SamplingSpec.label()` already exists) so a re-temper is verifiable
   from outside; `HealthResponse` gains `sampling: str` → OpenAPI export
   + `gen:api` regen (the CI drift checks require it).
7. `docs/deploy.md`: rewrite "Deploying an update" to point here; the
   initial-provisioning section gains §6.4's steps; drop the
   build-on-box commands and the Hetzner sizing.

### 6.2 The `images` and `deploy` jobs (append to `ci.yml`)

Two jobs, so images can be built and published before a box exists
(the packages must exist to be made public, D11) and the job that holds
the production secrets does nothing but ssh. [audit]

```yaml
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
    inputs:
      ref: { description: "ref to deploy (rollback)", required: false }

# … existing jobs …

images:
  needs: [lint, server, training, web]
  if: >-
    (github.event_name == 'push'
       && github.ref == format('refs/heads/{0}', github.event.repository.default_branch))
    || github.event_name == 'workflow_dispatch'
  runs-on: ubuntu-latest
  permissions: { contents: read, packages: write }
  outputs:
    sha: ${{ steps.tags.outputs.sha }}
    api: ${{ steps.tags.outputs.api }}
    web: ${{ steps.tags.outputs.web }}
  steps:
    - uses: actions/checkout@v5
      with:
        ref: ${{ inputs.ref || github.sha }}
    - id: tags
      run: app/scripts/deploy_tags.sh >> "$GITHUB_OUTPUT"
    - uses: docker/setup-buildx-action@v3
    - uses: docker/login-action@v3
      with: { registry: ghcr.io, username: "${{ github.actor }}",
              password: "${{ secrets.GITHUB_TOKEN }}" }
    - name: build api if missing
      run: |
        IMG=ghcr.io/${{ github.repository_owner }}/sheepshead-api:${{ steps.tags.outputs.api }}
        docker manifest inspect "$IMG" >/dev/null 2>&1 || \
          docker buildx build -f app/server/Dockerfile -t "$IMG" \
            --cache-from type=gha,scope=api --cache-to type=gha,scope=api,mode=max --push .
    - name: build web if missing
      run: |
        IMG=ghcr.io/${{ github.repository_owner }}/sheepshead-web:${{ steps.tags.outputs.web }}
        docker manifest inspect "$IMG" >/dev/null 2>&1 || \
          docker buildx build -t "$IMG" \
            --cache-from type=gha,scope=web --cache-to type=gha,scope=web,mode=max --push app/web

deploy:
  needs: images
  runs-on: ubuntu-latest
  environment: production
  concurrency: { group: production-deploy, cancel-in-progress: false }
  steps:
    - name: deploy over ssh
      env:
        HOST: ${{ secrets.DEPLOY_HOST }}
        KEY: ${{ secrets.DEPLOY_SSH_KEY }}
        KNOWN_HOSTS: ${{ secrets.DEPLOY_KNOWN_HOSTS }}
      run: |
        mkdir -p ~/.ssh && printf '%s\n' "$KNOWN_HOSTS" > ~/.ssh/known_hosts
        printf '%s\n' "$KEY" > key && chmod 600 key
        ssh -i key -o StrictHostKeyChecking=yes "deploy@$HOST" \
          "${{ needs.images.outputs.sha }} ${{ needs.images.outputs.api }} ${{ needs.images.outputs.web }}"
```

Notes: the two `--cache-*` scopes keep the api and web builds from
evicting each other's layer cache (both default to one scope)
[audit]. The `production` environment scopes the three secrets and can
carry a required-reviewer gate at first; drop the gate once trusted.
The web build must receive `NEXT_PUBLIC_API_BASE=""` (same-origin),
which is the Dockerfile's default. The `deploy` job lands only once the
box exists (§9), otherwise every push goes red at the ssh step.

### 6.3 `app/scripts/deploy.sh <sha> <api_tag> <web_tag>` [audit]

The deploy key's forced command; also runnable by hand as the `deploy`
user with the output of `deploy_tags.sh`. Spec:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /opt/sheepsheadai

# Arguments come from SSH_ORIGINAL_COMMAND under the forced command
# (§6.4), or from the command line by hand. Validate strictly: this is
# the only thing the deploy key can run.
read -r SHA API_TAG WEB_TAG EXTRA <<<"${SSH_ORIGINAL_COMMAND:-$*}"
[[ $SHA =~ ^[0-9a-f]{40}$ && $API_TAG =~ ^[0-9a-f]{12}$ \
   && $WEB_TAG =~ ^[0-9a-f]{12}$ && -z ${EXTRA:-} ]] ||
  { echo "usage: deploy.sh <sha40> <api_tag12> <web_tag12>" >&2; exit 2; }

# 0. Sync the tree, then re-exec so this deploy runs the NEW script
#    (otherwise a deploy.sh change would take effect one deploy late)
if [ -z "${DEPLOY_REEXEC:-}" ]; then
  PREV_SHA=$(git rev-parse HEAD); export PREV_SHA
  git fetch -q origin && git checkout -q --force "$SHA"
  DEPLOY_REEXEC=1 exec app/scripts/deploy.sh "$SHA" "$API_TAG" "$WEB_TAG"
fi

COMPOSE="docker compose --env-file app/deploy/.env.prod \
  --env-file app/deploy/.env.deploy -f app/docker-compose.prod.yml"
MODELS=/opt/sheepsheadai-models
M=models/manifest.json

# 1. Resolve the deploy model from the manifest (§5)
KEY=$(jq -r '.artifacts | to_entries[] | select(.value.roles | index("deploy")) | .key' "$M")
A=".artifacts[\"$KEY\"]"
SUM=$(jq -r "$A.sha256" "$M")
MODEL_FILE="$MODELS/$(jq -r "$A.asset" "$M")"
URL="$(jq -r .release_base "$M")/$(jq -r "$A.tag" "$M")/$(jq -r "$A.asset" "$M")"

# 2. Ensure it is cached & verified (temp file on the same filesystem)
if ! echo "$SUM  $MODEL_FILE" | sha256sum -c --quiet 2>/dev/null; then
  tmp=$(mktemp -p "$MODELS")
  curl -fsSL --retry 3 -o "$tmp" "$URL"
  echo "$SUM  $tmp" | sha256sum -c --quiet || { rm -f "$tmp"; exit 1; }
  mv "$tmp" "$MODEL_FILE"
fi

# 3. Write the machine-owned env file (never hand-edit)
{
  echo "API_TAG=$API_TAG"
  echo "WEB_TAG=$WEB_TAG"
  echo "MODEL_FILE=$MODEL_FILE"
  echo "SHEEPSHEAD_MODEL_LABEL=$KEY"
  jq -r "$A.sampling
    | \"SHEEPSHEAD_TEMPERATURE_PICK=\(.pick)\",
      \"SHEEPSHEAD_TEMPERATURE_PARTNER=\(.partner)\",
      \"SHEEPSHEAD_TEMPERATURE_BURY=\(.bury)\",
      \"SHEEPSHEAD_TEMPERATURE_PLAY=\(.play)\",
      \"SHEEPSHEAD_LOGIT_BAND=\(.band)\"" "$M"
} > app/deploy/.env.deploy

# 4. Pull ONLY our images (third-party images are pinned, D10); before
#    migrating, so an unpullable tag aborts before the schema changes
$COMPOSE pull -q api web

# 5. Migrate (no-op when nothing is pending; compat rule §7 rule 2)
$COMPOSE run --rm migrate

# 6. Converge — recreates ONLY services whose image/config changed
$COMPOSE up -d --remove-orphans

# 7. Caddyfile content changes don't trigger recreation (bind mount);
#    graceful reload preserves open websockets
git diff --quiet "$PREV_SHA" HEAD -- app/deploy/Caddyfile ||
  $COMPOSE exec -T caddy caddy reload --config /etc/caddy/Caddyfile

# 8. Wait for api health (built-in HEALTHCHECK, start_period 60s), then
#    confirm it serves the manifest's model, not a stale container
api=$($COMPOSE ps -q api)
for _ in $(seq 60); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' "$api")" = healthy ] && break
  sleep 3
done
[ "$(docker inspect -f '{{.State.Health.Status}}' "$api")" = healthy ] ||
  { echo "api failed to become healthy" >&2; exit 1; }
$COMPOSE exec -T api python -c \
  "import json,sys,urllib.request as u; sys.exit(json.load(u.urlopen('http://127.0.0.1:9000/health', timeout=2))['model'] != sys.argv[1])" \
  "$KEY" || { echo "api is not serving $KEY" >&2; exit 1; }

# 9. Reclaim disk: each api deploy leaves a ~1 GB image behind. A week's
#    images stay for quick rollback; older ones re-pull from GHCR.
docker image prune -af --filter until=168h >/dev/null
```

Failure handling: a non-zero exit fails the Actions run (notification,
§10). Recovery from a bad deploy is the rollback runbook (§8.3), not
automated revert — with one operator and near-zero traffic, explicit
beats clever. Because nothing in `.env.deploy` is read back as
"previous state", a retry after any failure redoes every step.
`migrate` costs an `npm ci` per deploy; with the cache volume this is
local work only. If it ever shows up as CPU contention with live games,
the upgrade is a prebuilt migrate image (§10).

### 6.4 One-time provisioning (Vultr box, Ubuntu 24.04 LTS)

1. `deploy` user (in `docker` group), SSH key-only auth, root login off.
2. ufw: allow 22/80/443. unattended-upgrades on, with
   `Unattended-Upgrade::Automatic-Reboot "false"` [audit]: an automatic
   reboot kills live games; kernel reboots are done by hand off-peak
   (every service is `restart: unless-stopped`, so the stack returns).
3. **1 GB swapfile** (2 GB RAM + torch: this is the OOM insurance).
4. Docker Engine + compose plugin; `curl`, `jq` [audit]. No
   `docker login`: the GHCR packages are public (D11). [audit]
5. `git clone --filter=blob:none <repo> /opt/sheepsheadai` (lazy blobs:
   the box never downloads training-artifact history).
6. `mkdir /opt/sheepsheadai-models`
7. `app/deploy/.env.prod` by hand: `DOMAIN`, `POSTGRES_USER`,
   `POSTGRES_PASSWORD`, `RESEND_API_KEY`, `EMAIL_FROM`. Secrets never
   transit CI. Keep an offline copy.
8. Deploy key [audit]: a dedicated keypair; the public half goes in
   `~deploy/.ssh/authorized_keys` as
   `restrict,command="/opt/sheepsheadai/app/scripts/deploy.sh" ssh-ed25519 …`
   (D12). Record the box's host key — `ssh-keyscan <ip>`, cross-checked
   against the fingerprint on the Vultr console — for the
   `DEPLOY_KNOWN_HOSTS` secret. The operator's own key stays
   unrestricted.
9. DNS: `sheepshead.ai` A record → box IP. Caddy handles TLS
   automatically once reachable.
10. First deploy, by hand: `app/scripts/deploy_tags.sh` locally, then
    as `deploy` on the box `app/scripts/deploy.sh <sha> <api> <web>`.
11. **Off-site backups (D8)** [audit]: a private Backblaze B2 bucket
    (versions kept by default) with a lifecycle rule deleting hidden
    (noncurrent) versions 90 days after hiding
    (`daysFromHidingToDeleting: 90`); a B2 application key restricted
    to that bucket; an rclone `b2` remote with an rclone `crypt` remote
    over it; nightly cron
    `rclone copy /opt/sheepsheadai/backups crypt:backups &&
    rclone copy /opt/sheepsheadai-models crypt:models &&
    curl -fsS -m 10 https://hc-ping.com/<uuid>` with a free
    healthchecks.io check that alerts when the ping stops arriving.
12. **Restore drill** [audit], once now and after any Postgres major
    upgrade: pull the latest dump back through the crypt remote,
    restore it into a throwaway `postgres` container, and spot-check
    row counts (`player`, `ai_player`, game records) against the live
    db.
13. Uptime monitor (UptimeRobot free) on `https://sheepshead.ai/health`
    — with sporadic usage nobody organically notices downtime.

GitHub side: create the `production` environment with secrets
`DEPLOY_HOST`, `DEPLOY_SSH_KEY`, `DEPLOY_KNOWN_HOSTS`; set the
`sheepshead-api` and `sheepshead-web` packages public (after their
first push).

## 7. Invariants (write these into docs/deploy.md too)

1. **Never build images on the box.** Even for debugging. A torch build
   on 1 vCPU grinds ~20 min and the memory pressure can OOM the api
   mid-game. The box pulls; CI builds.
2. **A migration must be compatible with the api code already
   running**, because it may land without an api restart (its process
   may run against the new schema indefinitely). Migrations that ship
   with api changes still run before the new api starts, so the old
   process briefly sees the new schema too: the same rule covers both.
   Migrations are forward-only; this rule is also what makes rollback
   past a migration safe (§8.3).
3. **Exactly one api container, one uvicorn worker.** Unchanged from
   the compose file's warnings; nothing in CD may scale it out.
4. **`.env.deploy` is machine-owned** (written by deploy.sh, gitignored);
   `.env.prod` is human-owned (secrets); `models/manifest.json` is
   git-owned. Never cross the streams.
5. **Third-party images move only in their own commits** (D10), never
   as a side effect of a deploy.
6. **Python-code deploys interrupt live games** (D7). Deploy off-peak
   when it matters; the restart matrix (§8.4) says when it applies.

## 8. Operator runbooks

### 8.1 Deploy code
Push to the default branch (or merge). CI green → deploy runs. Nothing
else.

### 8.2 Deploy a new model or a re-temper
Model Artifacts plan §6.2: upload the release (new model only), commit
the manifest change, push. CD does the rest. Verify:
`https://sheepshead.ai/health` reports the new model and sampling
labels (deploy.sh already checks the model label).

### 8.3 Roll back
* **Durable:** `git revert` the bad commit(s) and push. Content tags
  make the reverted tree reproduce the old tags, so no build runs.
* **Fast, temporary:** Actions → CI → "Run workflow" → `ref` = last good
  commit sha. Caveats [audit]: the four CI jobs still run first (on the
  dispatched branch head, not on `ref`), so this takes as long as a
  normal deploy; and the next push to `main` redeploys HEAD, so follow
  up with the revert.
* **Across a migration:** there are no down-migrations. Rolling code
  back past a migration is safe only because of §7 rule 2; if that rule
  was broken, fix forward instead.
* **Bad model:** revert the manifest commit (the old asset still
  exists).

### 8.4 What restarts when

| Commit touches                          | api (games!) | web | caddy | postgres | migrate |
|-----------------------------------------|--------------|-----|-------|----------|---------|
| `sheepshead/` outside the api closure (`search`, `rl`, `eval`, `training`, `research`, `devtools`), notebooks, docs, manifest panel entries | — | — | — | — | runs, no-op |
| `app/web/**` only                        | —            | ✓   | —     | —        | runs, no-op |
| `api-paths.txt` closure (server py + tests, engine, agent, lockfile) | ✓ | — | — | — | runs, no-op |
| `app/db/**`                              | —            | —   | —     | —        | applies |
| `app/deploy/Caddyfile`                   | —            | —   | graceful reload (WS survive) | — | runs, no-op |
| manifest deploy entry (weights or sampling) | ✓ (new weights / new ai_player) | — | — | — | runs, no-op |
| a pinned third-party image bumped (D10)  | —            | —   | ✓ if caddy (drops WS) | ✓ if postgres (api reconnects) | — |
| compose env/config for a service         | that service | …   | …     | …        | — |

### 8.5 Disaster recovery (box dies)
Provision per §6.4 on a fresh box, restore the latest dump from the
crypt remote into postgres (`docker compose run --rm ... psql < dump`),
repoint DNS, run the deploy workflow. Nothing else lives on the box:
code = git, images = GHCR, models = releases (+ rclone copy +
`model-history/`), secrets = `.env.prod` (offline copy), db = backups.

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
1. **The Model Artifacts plan, complete** (its §7). [audit]
2. `api-paths.txt` + `check_api_paths.py` + CI guard step (safe,
   standalone; proves the closure list before anything depends on it).
3. `deploy_tags.sh`.
4. Compose and Dockerfile, one commit each: pin third-party images +
   `uv`; `x-logging`; `db-backup` client version; `npm-cache` volume;
   `image:` switch + build override + `.env.prod.example` trim +
   `.gitignore` for `.env.deploy`.
5. `/health` sampling label + OpenAPI/TS regen.
6. `deploy.sh`.
7. `images` job + `workflow_dispatch` trigger in `ci.yml`; after its
   first run, set both GHCR packages public (D11).
8. Provision the box (§6.4), first deploy by hand, DNS cutover, restore
   drill.
9. `production` environment + secrets, then the `deploy` job in
   `ci.yml`.
10. `docs/deploy.md` rewrite.

## 10. Deferred / future work

* **Drain-before-deploy**: admin endpoint flipping `set_draining()`;
  deploy.sh (behind `DRAIN_ON_DEPLOY=1`) calls it and polls table count
  with a timeout before `up -d` when the api will restart. Do when
  there's traffic worth protecting.
* **Game-state snapshot/restore across restarts** — the only true fix
  for D7; large; revisit only if deploys visibly annoy players.
* **Prebuilt migrate image** (a third content tag over `app/db`) if the
  per-deploy `npm ci` ever contends with live games for the one vCPU.
* **Fly.io scale-to-zero migration** (D1 fallback) if usage data shows
  the box ~always idle: same containers, Fly proxy replaces Caddy,
  Neon replaces the postgres container. Revisit-not-before: 3 months of
  real traffic.
* Staging environment: overkill for now; CI's real-migration +
  container-boot coverage is the gate.
