#!/usr/bin/env bash
# Copy the database dumps off the box, encrypted, to Backblaze B2; nightly
# from the `deploy` user's crontab (docs/deploy.md).
#
# The db-backup container writes the dumps to /opt/sheepsheadai/backups on
# this disk, which dies with the box. This pushes them through an rclone
# crypt remote (they hold account emails and IPs) into a versioned bucket.
# The models need no copy: they are immutable GitHub release assets. It uses `copy`, never
# `sync`: a sync would mirror an emptied or corrupted backups/ and delete
# the good copies. Every run's log goes to a healthchecks.io check, which
# alerts when a success ping stops arriving -- a silently failing backup
# cron is the normal failure mode, so this is the alarm for the db-backup
# container too: a missing or stale dump fails the run.
# See notebooks/CD_Deployment_Plan_202608.md, D8.
#
# Config: app/deploy/.env.backup (human-owned, gitignored; see
# app/deploy/.env.backup.example). rclone runs from a pinned image, so the
# box needs no rclone install.
set -euo pipefail
cd /opt/sheepsheadai

ENV_FILE=app/deploy/.env.backup
RCLONE_IMAGE=rclone/rclone:1.75.2
BACKUPS=/opt/sheepsheadai/backups
# db-backup dumps @daily; anything older than this means it stopped.
MAX_DUMP_AGE_MIN=$((26 * 60))

PING_URL=$(sed -n 's/^BACKUP_PING_URL=//p' "$ENV_FILE")
[ -n "$PING_URL" ] || { echo "BACKUP_PING_URL missing from $ENV_FILE" >&2; exit 2; }

# Log to a file, then ship it with the ping (healthchecks.io keeps the
# body) and echo it for a hand run. /fail alerts at once rather than
# after the grace period.
log=$(mktemp)
exec 3>&1 >"$log" 2>&1
finish() {
  status=$?
  url=$PING_URL
  [ "$status" -eq 0 ] || url="$PING_URL/fail"
  curl -fsS -m 10 --retry 3 --data-binary @"$log" "$url" >/dev/null ||
    echo "ping to $url failed" >>"$log"
  cat "$log" >&3
  rm -f "$log"
  exit "$status"
}
trap finish EXIT

echo "offsite backup $(date -u +%FT%TZ)"

fresh=$(find "$BACKUPS/last" -name 'sheepshead-*.sql.gz' -type f \
  -size +1k -mmin -"$MAX_DUMP_AGE_MIN" | sort | tail -1)
[ -n "$fresh" ] ||
  { echo "no dump in $BACKUPS/last newer than ${MAX_DUMP_AGE_MIN}m: is db-backup running?"; exit 1; }
echo "newest dump: $fresh"

rclone() {
  docker run --rm --user "$(id -u):$(id -g)" --env-file "$ENV_FILE" \
    -v "$BACKUPS:/data/backups:ro" \
    "$RCLONE_IMAGE" --config "" --stats-one-line --stats-log-level NOTICE "$@"
}

# --skip-links: the *-latest symlinks only name the newest dated file.
rclone copy --skip-links /data/backups crypt:backups
echo ok
