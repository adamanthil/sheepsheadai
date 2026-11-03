# Deploying Sheepshead AI

Production is one small VPS (Vultr, Chicago; 1 vCPU / 2 GB) running the
Docker Compose stack in `app/docker-compose.prod.yml`: Caddy (TLS,
same-origin routing) in front of the Next.js `web` and FastAPI `api`
services, with Postgres behind them. Every push to `main` that passes CI
deploys itself. The design and its reasoning are in
`notebooks/CD_Deployment_Plan_202608.md`; this page is the operator's
reference.

The API keeps all live game state in one process's memory, so exactly one
api container with one uvicorn worker is a hard requirement. Restarting
the api drops live hands (completed hands are already persisted); clients
see a "server restarting" notice and reconnect.

## How a deploy works

1. CI runs the test jobs. On a push to `main`, the `images` job computes
   content tags (`app/scripts/deploy_tags.sh`: a hash of each service's
   files) and builds and pushes `ghcr.io/adamanthil/sheepshead-{api,web}`
   only when that tag does not exist yet.
2. The `deploy` job sshes to the box as `deploy`, whose key can run only
   `app/scripts/deploy.sh <sha> <api_tag> <web_tag>`.
3. `deploy.sh` checks out the commit, fetches and hash-checks the model
   that `models/manifest.json` gives the `deploy` role, writes
   `app/deploy/.env.deploy`, pulls the two images, runs migrations, runs
   `docker compose up -d`, and checks that the api serves the expected
   model.

`docker compose up -d` recreates only services whose image or config
changed, and a tag changes only when its files do, so most commits
restart nothing:

| Commit touches | api (live games!) | web | caddy | postgres | migrate |
|---|---|---|---|---|---|
| research, training, eval, notebooks, docs, manifest panel entries | — | — | — | — | runs, no-op |
| `app/web/**` only | — | ✓ | — | — | runs, no-op |
| a path in `app/deploy/api-paths.txt` | ✓ | — | — | — | runs, no-op |
| `app/db/**` | — | — | — | — | applies |
| `app/deploy/Caddyfile` | — | — | graceful reload (websockets survive) | — | runs, no-op |
| the manifest's deploy entry (weights or sampling) | ✓ | — | — | — | runs, no-op |
| a pinned third-party image | — | — | ✓ if caddy (drops websockets) | ✓ if postgres | — |

## Rules

1. **Never build images on the box**, even to debug. A torch build on one
   vCPU takes ~20 minutes and can OOM the live api. CI builds; the box
   pulls.
2. **A migration must work with the api code already running.** It may
   land without an api restart, and even with one the old process sees the
   new schema briefly. Migrations are forward-only; this rule is what
   makes rolling back past one safe.
3. **Exactly one api container, one uvicorn worker.**
4. **Who owns which file:** `app/deploy/.env.prod` is yours (secrets, by
   hand); `app/deploy/.env.deploy` is `deploy.sh`'s (never edit it);
   `models/manifest.json` is git's.
5. **Third-party images move only in their own commits.** Every image in
   the compose file and Dockerfiles is pinned to an exact version.
6. **Deploys that touch the api interrupt live games.** Push those
   off-peak when it matters.
7. **`app/deploy/api-paths.txt` must cover the api's imports.** CI's
   `check_api_paths.py` fails when it doesn't, and prints the lines to add.

## Runbooks

**Deploy code.** Push to `main`. Nothing else.

**Deploy a new model or new sampling.** Upload the release asset (new
weights only) and commit the manifest change; see
`notebooks/Model_Artifacts_Plan_202610.md` §6.2. Then check
`https://sheepshead.ai/health`: `model` and `sampling` should show the new
values.

**Roll back.**
- Durable: `git revert` the bad commit and push. The reverted tree
  reproduces the old content tags, so no image is rebuilt.
- Fast and temporary: Actions → CI → Run workflow, with `ref` = the last
  good commit. The test jobs run first, so it takes as long as a normal
  deploy, and the next push to `main` redeploys HEAD, so follow up with a
  revert.
- Past a migration: there are no down-migrations; rule 2 is what makes
  this safe. If rule 2 was broken, fix forward.
- Bad model: revert the manifest commit (the old asset still exists).

**Deploy by hand** (first deploy, or with CI down). Locally run
`app/scripts/deploy_tags.sh`, make sure CI has published those tags, then
on the box as `deploy`: `/opt/sheepsheadai/app/scripts/deploy.sh <sha>
<api> <web>`.

**Box dies.** Provision a new one (below), restore the latest dump from
the off-site copy, repoint DNS, deploy. Nothing else lives on the box:
code is in git, images in GHCR, models in GitHub releases, secrets in the
offline copy of `.env.prod`, and the database in the backups.

## Prerelease mode

The first deployment is a playtest shared by link, not a public launch.
`app/deploy/Caddyfile` serves a `robots.txt` that disallows all crawling
and adds `X-Robots-Tag: noindex, nofollow` to every response. At public
launch, delete that block (marked PRERELEASE) and push; Caddy reloads
gracefully.

## One-time provisioning

On the box (Ubuntu 26.04 LTS). 26.04 ships the Rust rewrites of coreutils
and sudo by default; `deploy.sh` has been run against them (`sha256sum -c`,
`mktemp -p`, `seq`). If a GNU-only flag ever bites, `apt install
coreutils-from-gnu` switches back.

1. A `deploy` user in the `docker` group; ssh key-only auth; root login
   off.
2. `ufw` allowing 22, 80 and 443. `unattended-upgrades` on, with
   `Unattended-Upgrade::Automatic-Reboot "false";` (a reboot kills live
   games; reboot by hand off-peak, and the stack comes back on its own).
3. A 1 GB swapfile (torch on 2 GB of RAM).
4. Docker Engine with the compose plugin; `curl`, `jq`. No `docker login`:
   the images are public.
5. `git clone --filter=blob:none https://github.com/adamanthil/sheepsheadai.git /opt/sheepsheadai`,
   owned by `deploy`.
6. `mkdir /opt/sheepsheadai-models`, owned by `deploy`.
7. `cp app/deploy/.env.prod.example app/deploy/.env.prod` and fill in
   `DOMAIN`, `POSTGRES_PASSWORD` (long and random), `RESEND_API_KEY` and
   `EMAIL_FROM`. These never pass through CI. Keep an offline copy.
8. The deploy key: generate a dedicated keypair and put the public half in
   `~deploy/.ssh/authorized_keys` as
   `restrict,command="/opt/sheepsheadai/app/scripts/deploy.sh" ssh-ed25519 …`.
   Your own key stays unrestricted. Record the host key with
   `ssh-keyscan <ip>`, checked against the fingerprint on the Vultr
   console.
9. DNS: an `A` record for the domain pointing at the box. Caddy gets the
   TLS certificate on its own once the name resolves.
10. First deploy by hand (above), then check
    `curl https://$DOMAIN/health` → `{"status":"ok","db":true,...}` and
    play a hand in a browser.
11. Off-site backups: a private Backblaze B2 bucket with a lifecycle rule
    deleting hidden versions 90 days after hiding; an application key
    restricted to that bucket; an rclone `b2` remote with a `crypt` remote
    over it (the dumps hold account emails and IPs). Nightly cron as
    `deploy`:

    ```sh
    rclone copy /opt/sheepsheadai/backups crypt:backups &&
      rclone copy /opt/sheepsheadai-models crypt:models &&
      curl -fsS -m 10 https://hc-ping.com/<uuid>
    ```

    with a free healthchecks.io check that alerts when the ping stops.
    Use `copy`, never `sync`: a `sync` would mirror an emptied `backups/`
    and delete the good copies. (rclone skips the backup tool's `*-latest`
    symlinks; the dated files are the real dumps.)
12. Restore drill, now and after any Postgres major upgrade: pull the
    latest dump back through the crypt remote, restore it into a
    throwaway postgres container, and compare row counts (`player`,
    `ai_player`, game records) with the live database:

    ```sh
    rclone copy crypt:backups/daily/ ./drill/
    dump=$(ls -t drill/*.sql.gz | head -1)
    docker run -d --name restore-drill -e POSTGRES_PASSWORD=x postgres:18.6
    sleep 5
    gunzip -c "$dump" |
      docker exec -i restore-drill psql -q -U postgres
    docker exec restore-drill psql -U postgres -c 'select count(*) from player'
    docker rm -f restore-drill
    ```
13. An uptime monitor (UptimeRobot, free) on `https://$DOMAIN/health`.

On GitHub:

1. After the first `images` run, set the `sheepshead-api` and
   `sheepshead-web` packages public (Packages → package settings → change
   visibility).
2. Create the `production` environment with secrets `DEPLOY_HOST`,
   `DEPLOY_SSH_KEY` (the private half of the deploy key) and
   `DEPLOY_KNOWN_HOSTS` (the `ssh-keyscan` output). Optionally require a
   reviewer at first.
3. Set the repository variable `DEPLOY_ENABLED` to `true`.

## Operational notes

- Logs: on the box,
  `docker compose --env-file app/deploy/.env.prod --env-file app/deploy/.env.deploy -f app/docker-compose.prod.yml logs -f api`
  (JSON). Each container keeps at most 3 × 10 MB.
- `GET /health` returns `status`, `db`, `model` and `sampling`.
  `db: false` with a 200 means Postgres is down but the process is alive.
- Rate limits and connection caps are in-process (see
  `app/server/api/ratelimit.py`, `MAX_TABLES`, `MAX_SOCKETS_PER_IP`), which
  is correct for a single instance.
- `deploy.sh` prunes images older than a week; older rollbacks re-pull
  from GHCR.
- To run the production stack locally from source, add
  `-f app/docker-compose.build.yml` (see its header).
- Dev setup: `docker compose -f app/docker-compose.yml up -d` (Postgres
  only), `bash app/server/run_server.sh`, and `npm run dev` in `app/web/`.
