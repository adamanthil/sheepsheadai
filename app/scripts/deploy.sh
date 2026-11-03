#!/usr/bin/env bash
# Deploy one commit to the production box: deploy.sh <sha40> <api_tag12> <web_tag12>
#
# The deploy ssh key's forced command (arguments arrive in
# SSH_ORIGINAL_COMMAND); also runnable by hand as the `deploy` user with the
# output of app/scripts/deploy_tags.sh. Every step is redone on every run, so
# a retry after any failure is safe. Recovery from a bad deploy is the
# rollback runbook, not an automatic revert.
# See notebooks/CD_Deployment_Plan_202608.md section 6.3.
set -euo pipefail
cd /opt/sheepsheadai

# Validate strictly: this is the only thing the deploy key can run.
read -r SHA API_TAG WEB_TAG EXTRA <<<"${SSH_ORIGINAL_COMMAND:-$*}"
[[ ${SHA:-} =~ ^[0-9a-f]{40}$ && ${API_TAG:-} =~ ^[0-9a-f]{12}$ &&
  ${WEB_TAG:-} =~ ^[0-9a-f]{12}$ && -z ${EXTRA:-} ]] ||
  { echo "usage: deploy.sh <sha40> <api_tag12> <web_tag12>" >&2; exit 2; }

# 0. One deploy at a time (CI and a hand deploy could overlap). Sync the
#    tree, then re-exec so this deploy runs the NEW script; otherwise a
#    deploy.sh change would take effect one deploy late. The lock's file
#    descriptor survives the exec.
if [ -z "${DEPLOY_REEXEC:-}" ]; then
  exec 9>/tmp/sheepshead-deploy.lock
  flock -n 9 || { echo "another deploy is running" >&2; exit 1; }
  PREV_SHA=$(git rev-parse HEAD)
  export PREV_SHA
  git fetch -q origin
  git checkout -q --force "$SHA"
  DEPLOY_REEXEC=1 exec app/scripts/deploy.sh "$SHA" "$API_TAG" "$WEB_TAG"
fi

COMPOSE=(docker compose --env-file app/deploy/.env.prod
  --env-file app/deploy/.env.deploy -f app/docker-compose.prod.yml)
MODELS=/opt/sheepsheadai-models
M=models/manifest.json

# 1. Resolve the deploy model from the manifest (fetch_models --lint in CI
#    guarantees exactly one deploy entry with a sampling block).
KEY=$(jq -r '.artifacts | to_entries[] | select(.value.roles | index("deploy")) | .key' "$M")
[[ $KEY =~ ^[A-Za-z0-9._-]+$ ]] ||
  { echo "manifest must have exactly one deploy entry, got: $KEY" >&2; exit 1; }
A=".artifacts[\"$KEY\"]"
SUM=$(jq -r "$A.sha256" "$M")
ASSET=$(jq -r "$A.asset" "$M")
MODEL_FILE="$MODELS/$ASSET"
URL="$(jq -r .release_base "$M")/$(jq -r "$A.tag" "$M")/$ASSET"

# 2. Ensure it is cached and verified. The download lands in a temp file in
#    the same directory, so the final mv is an atomic rename. A hash
#    mismatch aborts loudly (it would also catch a re-uploaded asset).
if ! echo "$SUM  $MODEL_FILE" | sha256sum -c --quiet 2>/dev/null; then
  echo "fetching $URL"
  tmp=$(mktemp -p "$MODELS")
  curl -fsSL --retry 3 -o "$tmp" "$URL"
  echo "$SUM  $tmp" | sha256sum -c --quiet ||
    { rm -f "$tmp"; echo "$ASSET does not match the manifest hash" >&2; exit 1; }
  chmod 644 "$tmp"
  mv "$tmp" "$MODEL_FILE"
fi

# 3. Write the machine-owned env file (never hand-edit).
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
      \"SHEEPSHEAD_LOGIT_BAND=\(.band // \"\")\"" "$M"
} >app/deploy/.env.deploy

# 4. Pull ONLY our images (third-party images are pinned); before
#    migrating, so an unpullable tag aborts before the schema changes.
"${COMPOSE[@]}" pull -q api web

# 5. Migrate (a no-op when nothing is pending). A migration must work with
#    the api code already running, which may not restart this deploy.
"${COMPOSE[@]}" run --rm migrate

# 6. Converge: recreates ONLY services whose image or config changed.
"${COMPOSE[@]}" up -d --remove-orphans

# 7. A Caddyfile content change does not recreate caddy (it is a bind
#    mount); a graceful reload keeps open websockets.
git diff --quiet "$PREV_SHA" HEAD -- app/deploy/Caddyfile ||
  "${COMPOSE[@]}" exec -T caddy caddy reload --config /etc/caddy/Caddyfile

# 8. Wait for api health (HEALTHCHECK start period 60s), then confirm it
#    serves the manifest's model rather than a stale container.
api=$("${COMPOSE[@]}" ps -q api)
health() { docker inspect -f '{{.State.Health.Status}}' "$api"; }
for _ in $(seq 60); do
  [ "$(health)" = healthy ] && break
  sleep 3
done
[ "$(health)" = healthy ] || { echo "api failed to become healthy" >&2; exit 1; }
"${COMPOSE[@]}" exec -T api python -c \
  "import json,sys,urllib.request as u; sys.exit(json.load(u.urlopen('http://127.0.0.1:9000/health', timeout=2))['model'] != sys.argv[1])" \
  "$KEY" || { echo "api is not serving $KEY" >&2; exit 1; }

# 9. Reclaim disk: each api deploy leaves a ~1 GB image behind. A week of
#    images stays for quick rollback; older ones re-pull from GHCR.
docker image prune -af --filter until=168h >/dev/null
echo "deployed $SHA (api $API_TAG, web $WEB_TAG, model $KEY)"
