#!/usr/bin/env bash
# Restore drill: prove the off-site copy restores. Run on the box as
# `deploy` at provisioning and after any Postgres major upgrade
# (docs/deploy.md).
#
# Pulls the newest dump back from B2 through the crypt remote (the same
# path a dead box's replacement would take), restores it into a throwaway
# postgres container of the production version, and prints its row counts
# beside the live database's. The live counts should be equal or a little
# higher (rows added since the dump). Fails on any restore error or an
# empty player table.
set -euo pipefail
cd /opt/sheepsheadai

ENV_FILE=app/deploy/.env.backup
RCLONE_IMAGE=rclone/rclone:1.75.2
# The production pin, so the drill restores with the server it would meet.
PG_IMAGE=$(sed -n 's/^ *image: \(postgres:.*\)$/\1/p' app/docker-compose.prod.yml)
PG_USER=$(sed -n 's/^POSTGRES_USER=//p' app/deploy/.env.prod)
PG_USER=${PG_USER:-sheepshead}
COMPOSE=(docker compose --env-file app/deploy/.env.prod
  --env-file app/deploy/.env.deploy -f app/docker-compose.prod.yml)
TABLES=(player account ai_player game_table game game_player trick trick_card)
NAME=sheepshead-restore-drill

work=$(mktemp -d)
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT

rclone() {
  docker run --rm --user "$(id -u):$(id -g)" --env-file "$ENV_FILE" \
    -v "$work:/data" "$RCLONE_IMAGE" --config "" "$@"
}

dump=$(rclone lsf --files-only --include 'sheepshead-*.sql.gz' crypt:backups/last |
  grep -v latest | sort | tail -1)
[ -n "$dump" ] || { echo "no dump in crypt:backups/last" >&2; exit 1; }
echo "restoring $dump into $PG_IMAGE"
rclone copy "crypt:backups/last/$dump" /data

docker run -d --name "$NAME" -e POSTGRES_USER="$PG_USER" \
  -e POSTGRES_PASSWORD=drill -e POSTGRES_DB=sheepshead "$PG_IMAGE" >/dev/null
# The image's init runs a temporary server first; wait for the real one.
for _ in $(seq 60); do
  docker exec "$NAME" pg_isready -q -h 127.0.0.1 -U "$PG_USER" 2>/dev/null && break
  sleep 1
done
gunzip -c "$work/$dump" |
  docker exec -i "$NAME" psql -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d sheepshead >/dev/null

query="select $(printf "(select count(*) from %s)||' '||" "${TABLES[@]}")''"
restored=$(docker exec "$NAME" psql -tA -U "$PG_USER" -d sheepshead -c "$query")
live=$("${COMPOSE[@]}" exec -T postgres psql -tA -U "$PG_USER" -d sheepshead -c "$query")

read -ra r <<<"$restored"
read -ra l <<<"$live"
printf '%-14s %10s %10s\n' table restored live
for i in "${!TABLES[@]}"; do
  printf '%-14s %10s %10s\n' "${TABLES[$i]}" "${r[$i]}" "${l[$i]}"
done
[ "${r[0]}" -gt 0 ] || { echo "restored player table is empty" >&2; exit 1; }
echo "restore drill passed"
