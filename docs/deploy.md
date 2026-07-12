# Deploying Sheepshead AI

Target: **one small VPS** (Hetzner CX32 / DigitalOcean 4GB class) running
Docker Compose. The API keeps all live game state in one process's memory, so
exactly one API container with one uvicorn worker is a hard requirement —
autoscaling platforms are the wrong shape for this app. Restarting the API
drops live hands (completed hands are already persisted); clients see a
"server restarting" notice and auto-reconnect.

## One-time provisioning

1. Provision a VPS (2+ cores, 4GB RAM), install Docker + the compose plugin,
   open ports 80/443, and point your domain's A/AAAA records at it.
2. Clone the repo and copy the model checkpoint onto the host.
3. `cp app/deploy/.env.prod.example app/deploy/.env.prod` and fill in `DOMAIN`,
   `POSTGRES_PASSWORD` (long + random), `SHEEPSHEAD_MODEL_LABEL`, `MODEL_FILE`.
4. First boot:

   ```sh
   docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml build
   docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml up -d postgres
   docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml run --rm migrate
   docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml up -d
   ```

   Caddy obtains TLS certificates automatically on first request.

5. Smoke-check: `curl https://$DOMAIN/health` → `{"status":"ok","db":true,...}`,
   then play a hand in the browser.

## Deploying an update

```sh
git pull
docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml build
docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml run --rm migrate
docker compose --env-file app/deploy/.env.prod -f app/docker-compose.prod.yml up -d
```

`up -d` recreates changed containers. On SIGTERM the API broadcasts
`server_restart` to every table, refuses new tables/games (503), waits ~2s,
then shuts down gracefully (compose allows 30s). Live hands are lost by
design — deploy off-peak.

## Backups

- The `db-backup` sidecar writes nightly `pg_dump`s to `./backups`
  (14 daily + 8 weekly retained).
- Copy them off the box (cron + rclone to B2/S3 or similar):
  `rclone sync ./backups remote:sheepshead-backups`
- **Restore drill** (run one quarterly):

  ```sh
  gunzip -c backups/daily/sheepshead-<date>.sql.gz | \
    docker compose -f app/docker-compose.prod.yml exec -T postgres \
      psql -U sheepshead -d sheepshead
  ```

## Operational notes

- **Never scale the api service** past one replica; tables would shard across
  processes and websockets would break.
- Logs: `docker compose -f app/docker-compose.prod.yml logs -f api` (JSON format).
- The uptime probe is `GET /health`; `db:false` with a 200 means Postgres is
  down but the process is alive.
- Rate limits and connection caps are in-process (see
  `app/server/api/ratelimit.py`, `MAX_TABLES`, `MAX_SOCKETS_PER_IP`) — correct for
  the single-instance deployment.
- Dev setup remains: `docker compose -f app/docker-compose.yml up -d` (Postgres only) +
  `bash app/server/run_server.sh` + `npm run dev` in `app/web/`.
