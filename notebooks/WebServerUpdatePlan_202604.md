# Web Server Update Plan — April 2026

This plan covers four coordinated workstreams against `/server` (FastAPI + Python)
and `/web` (Next.js + TypeScript):

1. Code cleanup for a release-quality baseline (maintainability, security).
2. Dependency upgrades across Python + Node.
3. Postgres persistence for game stats/history, schema already drafted in
   `server/database/sheepshead-ai-db-schema.sql`, to be managed with
   [graphile-migrate](https://github.com/graphile/migrate).
4. Persistent player identity: browser-stored player UUID, plus persisted
   display names when the user overrides a default.

It is written to be followed by a junior engineer. Each task lists the
concrete files to touch, the expected outcome, and verification steps.

> **Terminology.** The schema calls each row in the `game` table a *game*,
> but semantically that is one **hand** (a single deal played to its end) —
> a 1:1 correspondence with the `sheepshead.Game` Python class. A table
> (`game_table`) hosts one or more hands across its lifetime.
>
> All design choices previously tagged `⚠️ CLARIFY` have been resolved by
> Andrew; see [Appendix A: Clarifications](#appendix-a-clarifications)
> for the authoritative record.

---

## Table of Contents

1. [Current Architecture Snapshot](#current-architecture-snapshot)
2. [Phase 1 — Dependency Upgrades](#phase-1--dependency-upgrades)
3. [Phase 2 — Code Cleanup & Security Hardening](#phase-2--code-cleanup--security-hardening)
4. [Phase 3 — Database: Migrations & Infrastructure](#phase-3--database-migrations--infrastructure)
5. [Phase 4 — Persistent Player Identity & Display Name](#phase-4--persistent-player-identity--display-name)
6. [Phase 5 — Game & Table Persistence (Writes)](#phase-5--game--table-persistence-writes)
7. [Phase 6 — Stats / History Read Paths (MVP)](#phase-6--stats--history-read-paths-mvp)
8. [Phase 7 — Ops, CI, Release Prep](#phase-7--ops-ci-release-prep)
9. [Milestones & Suggested Sequencing](#milestones--suggested-sequencing)
10. [Appendix A: Clarifications](#appendix-a-clarifications)
11. [Appendix B: Proposed Data-Mapping Table](#appendix-b-proposed-data-mapping-table)

---

## Current Architecture Snapshot

| Layer | Location | Notes |
|---|---|---|
| Game engine (pure Python) | `sheepshead.py` | `Game` + `Player`. Owns hand/blind/bury state, tricks, scoring. No persistence. |
| RL/PPO agent | `ppo.py`, `encoder.py`, `training_utils.py` | Loaded at server startup from `SHEEPSHEAD_MODEL_PATH` (`.pt` file). |
| FastAPI server | `server/main.py` (~1350 lines) | In-memory `TableManager` holds all live state. REST + single websocket per table. |
| Model loader | `server/services/ai_loader.py` | Thin wrapper around `PPOAgent.load`. |
| Analyze endpoint | `server/services/analyze.py` | Single-shot game simulation + trace, used by `/analyze` UI. |
| API schemas | `server/api/schemas.py` | Pydantic v2 models for request/response bodies. |
| Frontend | `web/` (Next.js 14, React 18) | `/` (lobby), `/waiting/[id]`, `/table/[id]`, `/analyze`. |
| Client identity today | Per-table `client_id` generated on `POST /api/tables/:id/join`, stored in `localStorage` under `sheepshead_client_id_${tableId}`. No cross-table identity. |

**Known smells that this plan will clean up:**

- `server/main.py` is one monolithic file with FastAPI routes, WebSocket handler,
  AI loop, seat/reservation logic, autoclose/grace-period scheduling, and
  broadcasting all intermixed.
- `client_id` is used as an auth token but is passed in URL query strings and
  request bodies. There is no separate session layer and no rate limiting.
- `Table.to_public_dict` leaks `hostId` (host's client_id / auth token) in the
  public table payload. Other humans at the table can see it.
  (See `server/main.py` line 172.)
- CORS config uses a literal `"https://yourdomain.com"` placeholder
  (`server/main.py` line 227).
- Several bare `except Exception: pass` / no-op branches swallow errors
  (`server/main.py` around lines 729–732, 978–979, 1246–1247).
- Chat messages are only length-limited (500 chars) and are rendered as text
  by `ChatPanel.tsx` — good — but the server does not sanitise content or
  rate-limit posts.
- Display-name generation in `web/app/page.tsx` uses Hobbit names and changes on
  every reload — the user's intent (a name they typed) is lost.
- No tests for the server layer. No CI. No type-checking step in the web build.

---

## Phase 1 — Dependency Upgrades

**Goal:** move Python + Node dependencies to latest stable. This runs first so
later cleanup is written against current APIs.

### 1.1 Python (`pyproject.toml`, `uv.lock`)

Currently pinned (`pyproject.toml`):

```
fastapi>=0.129.0
uvicorn[standard]>=0.41.0
pydantic>=2.12.5
websockets>=16.0
```

1. Add the following server-extras dependencies:
   - `asyncpg` — async Postgres driver (per C11).
   - `pydantic-settings` — typed settings loaded from env/.env.
   - UUIDs: use the built-in `uuid` module. No extra dep needed.
2. Run `uv sync --extra server --upgrade` (or `uv-upgrade` per `README.md`).
3. Bump each lib to current stable; fix any deprecations flagged by uvicorn
   startup or Pydantic v2 migration warnings.
4. Verify `./server/run_server.sh --model …` still boots cleanly.
5. Commit `pyproject.toml` + `uv.lock` together.

**Acceptance:** server boots; `GET /api/health` returns `{ "status": "ok" }`;
no deprecation warnings in stderr.

### 1.2 Node (`web/package.json`, `web/package-lock.json`)

Currently pinned:

```
next ^14.2.32
react / react-dom 18.2.0
typescript 5.4.5
eslint 8.57.0
```

1. Upgrade to:
   - `next@^15` (App Router is stable, expected drop-in).
   - `react@^19`, `react-dom@^19`.
   - `typescript@^5.6` (or latest).
   - `eslint@^9` + `eslint-config-next@^15` (flat config).
   - `@types/react@^19`, `@types/react-dom@^19`, `@types/node@^22`.
2. Delete stale `web/tsconfig.tsbuildinfo` and `.next/`; `rm -rf node_modules`
   and reinstall.
3. React 19 may change some hook behaviors; verify:
   - `useResponsive` dynamic ref math in `web/app/table/[id]/hooks/useResponsive.ts`.
   - WebSocket effect cleanup in `useTableSocket.ts`.
4. Run `npm run build` and `npm run lint`; fix any type errors.
5. Manually smoke-test: create table → join → auto-fill AI → play a hand → redeal.

**Acceptance:** `npm run build` passes; no runtime errors in browser console
when playing a full hand against AI.

### 1.3 (Optional) Add `npm run typecheck`

Add a `typecheck` script that runs `tsc --noEmit` so type errors fail in CI
even if lint does not catch them.

---

## Phase 2 — Code Cleanup & Security Hardening

**Goal:** restructure `server/main.py` for maintainability; remove security
smells before we ship.

### 2.1 Split `server/main.py` into modules

Target layout (no behaviour changes in this step):

```
server/
├── __init__.py
├── app.py                 # FastAPI app factory + CORS + startup hooks
├── config.py              # typed settings (pydantic-settings)
├── api/
│   ├── schemas.py         # (existing)
│   ├── tables.py          # /api/tables, /join, /seat, /rules, /close
│   ├── games.py           # /start, /redeal, /action, /fill_ai, /start_waiting
│   ├── actions.py         # /api/actions (action_lookup)
│   ├── analyze.py         # /api/analyze/simulate
│   └── players.py         # NEW — /api/players (identity), see Phase 4
├── realtime/
│   ├── websocket.py       # /ws/table/{id} handler
│   ├── chat.py            # add_chat_message, broadcast_chat_append
│   └── broadcast.py       # broadcast_table_state, broadcast_table_event
├── runtime/
│   ├── tables.py          # Table, TableManager, Occupant, ClientConn
│   ├── ai_loop.py         # ai_take_turns, schedule_ai_turns, ai_observe_all
│   ├── seating.py         # seat helpers, reservation, disconnect replacement
│   └── lifecycle.py       # close_table, schedule_autoclose_if_no_humans
└── services/
    ├── ai_loader.py
    ├── analyze.py
    └── persistence/       # NEW — see Phase 3 / 5
        ├── __init__.py
        ├── pool.py
        ├── players.py
        ├── games.py
        └── cards.py
```

Do this as a mechanical move (one PR per sub-module if possible) so review
stays tractable. After every sub-module move, run the full manual smoke test.

### 2.2 Security hardening

Do these in a separate PR after the split is green.

1. **Stop leaking host's client_id.** In `Table.to_public_dict` (currently
   `server/main.py:172`), remove `hostId`. Keep `host` (the display name).
   The host's own client is the only one that needs to know their ID, and it
   already has it locally. Update `web/app/waiting/[id]/page.tsx` and
   `web/app/table/[id]/page.tsx` (`isHost` check) to instead derive host-ness
   by comparing the displayed `host` name to the current player's display
   name — or better, expose `you.isHost: boolean` in per-client websocket
   payloads so we do not leak any identifier.
2. **Introduce an explicit player session token** separate from the cross-table
   `player_id`. See Phase 4.
3. **CORS.** Move the allowed-origins list to `Settings` (env var
   `SHEEPSHEAD_CORS_ORIGINS`, comma-separated). Remove the
   `"https://yourdomain.com"` placeholder. Document in README.
4. **Rate-limit chat.** In `realtime/chat.py`, cap player chat at e.g.
   5 messages / 5 seconds per `client_id`; drop silently with a debug log
   when exceeded.
5. **Length limits.** Enforce server-side: `display_name` ≤ 32 chars,
   `table.name` ≤ 48 chars, chat message ≤ 500 chars (already enforced —
   move it into a schema validator rather than inline `if` in the websocket
   loop).
6. **Input validation.** Pydantic validators for:
   - `CreateTableRequest.name` — trimmed, non-empty, ≤48 chars.
   - `JoinTableRequest.display_name` — trimmed, non-empty, ≤32 chars.
   - `rules` — restrict keys to `{partnerMode: 0|1, doubleOnTheBump: bool}`;
     reject anything else (today it is an open dict).
7. **Remove broad `except Exception: pass`** noted above. Replace with narrow
   excepts + `logging.exception(...)`.
8. **Treat pickled model loads as trusted local input only.** Document that
   `SHEEPSHEAD_MODEL_PATH` must point to a file owned/reviewed by us — no
   change in code, just README wording.
9. **Structured logging.** Configure `uvicorn` access + app logs to JSON in
   production via `LOG_FORMAT=json`; keep human-readable in dev. Include
   `table_id` / `client_id` as structured fields in table-scoped logs.
10. **Secrets.** No DB URL goes in source; read `DATABASE_URL` from env via
    `Settings`. Commit a `.env.example`.

### 2.3 Frontend cleanup

1. Remove debug `console.log`s in `web/app/page.tsx` (there are several —
   lines 124, 141, 149, 162, 188, 197).
2. Remove dead `setJoinInfo` state in `web/app/page.tsx` (declared, never read).
3. Extract the `API_BASE` derivation into `web/lib/apiBase.ts` — it is
   duplicated across `page.tsx`, `waiting/[id]/page.tsx`, `useTableSocket.ts`.
4. Extract the `localStorage` keys (`sheepshead_client_id_*`, and the new
   `sheepshead_player_id` / `sheepshead_display_name` from Phase 4) into
   `web/lib/storage.ts` constants so typos cannot desync keys.
5. Define a typed `TableState` / `TableView` in `web/lib/types.ts` to replace
   the `any` escape hatches (`TableStateMsg.table: any`, `view: any`).

---

## Phase 3 — Database: Migrations & Infrastructure

**Goal:** a working local Postgres with schema applied via graphile-migrate.
No application writes yet — this phase just produces a migrated, seeded DB.

### 3.1 Local dev Postgres

1. Add `docker-compose.yml` at repo root with a single `postgres:18` service
   on port 5432, volume-mounted, env `POSTGRES_USER=sheepshead`,
   `POSTGRES_PASSWORD=sheepshead`, `POSTGRES_DB=sheepshead`. Document in README.
2. `.env.example`:
   ```
   DATABASE_URL=postgres://sheepshead:sheepshead@localhost:5432/sheepshead
   SHEEPSHEAD_MODEL_PATH=./final_pfsp_swish_ppo.pt
   SHEEPSHEAD_CORS_ORIGINS=http://localhost:3000
   ENV=development
   LOG_FORMAT=text
   ```
3. Add `.env` to `.gitignore` (already excludes most build artefacts; confirm).

### 3.2 graphile-migrate setup

graphile-migrate is a Node tool so it lives under `/db`:

```
db/
├── .gmrc              # graphile-migrate config (committed)
├── migrations/
│   ├── committed/     # squashed, reviewed migrations (committed)
│   └── current.sql    # working migration (committed, empty or WIP)
├── fixtures/
│   ├── afterReset.sql # optional: seed data (card, suit, ai_model)
│   └── afterAllMigrations.sql
└── package.json       # dev-dep on graphile-migrate
```

1. `cd db && npm init -y && npm install --save-dev graphile-migrate`.
2. Create `.gmrc` (committed):
   ```jsonc
   {
     "connectionString": "$DATABASE_URL",
     "shadowConnectionString": "$SHADOW_DATABASE_URL",
     "rootConnectionString": "$ROOT_DATABASE_URL",
     "pgSettings": { "search_path": "public" },
     "placeholders": { ":DATABASE_AUTHENTICATOR": "!ENV" },
     "afterReset": ["!afterReset.sql"],
     "afterAllMigrations": []
   }
   ```
3. Add npm scripts in `db/package.json`:
   - `"migrate": "graphile-migrate migrate"`
   - `"watch": "graphile-migrate watch"`
   - `"commit": "graphile-migrate commit"`
   - `"reset": "graphile-migrate reset --erase"`
4. Provide a shadow DB for graphile-migrate (create a second empty DB
   `sheepshead_shadow` in docker-compose or document `createdb`).

### 3.3 Initial migration

1. Translate `server/database/sheepshead-ai-db-schema.sql` into the
   `current.sql` of a first migration. Adjustments to apply during translation
   (the raw `.sql` file is a reference; the migration is the source of truth):
   - Replace `BIGINT NOT NULL` primary keys on `game_player`, `cardset`,
     `cardset_card`, `trick`, `trick_card`, `ai_model`, `ai_player` with
     `BIGSERIAL` (or `GENERATED ALWAYS AS IDENTITY`) so inserts don't have
     to supply ids.
   - **Add `cardset.cards_hash TEXT NOT NULL UNIQUE`** — enables dedup per
     Q6. Format: sorted ascending `card_id`s joined by `,` (e.g.
     `"1,4,7,12,19,24"` for a six-card hand). See §5.3 for the upsert
     pattern.
   - Add missing indexes for lookup: `trick(game_id, index)`,
     `trick_card(trick_id, index)`, `game_player(game_id)`,
     `game(game_table_id)`, `game_player(player_id)`,
     `game_player(ai_player_id)`.
   - Add `UNIQUE (ai_model_id, is_deterministic)` on `ai_player` so the
     startup upsert in §5.4 has a conflict target.
   - `game.is_called_partner BOOLEAN NOT NULL` stays as-is. It reflects the
     **table-level partner-selection mode** for the hand (true = called-ace,
     false = Jack-of-Diamonds), not whether a partner was actually called.
     Leasters are valid under either mode.
   - `trick.winning_player_id` is set for every persisted (completed) trick.
     We don't persist in-progress tricks, so NULL should never appear in
     practice. Keep the column nullable per the schema.
   - Confirm all FKs validate in the shadow DB.
2. `npm --prefix db run commit -- -m "initial_schema"` to freeze it.

### 3.4 Seed data (afterReset.sql)

Populate reference tables that never change at runtime. These are the
`suit` and `card` catalogues used by every game.

1. `suit` — four rows matching `sheepshead.py` `SUIT_NAMES`:
   ```
   (1, 'C', 'Clubs')
   (2, 'S', 'Spades')
   (3, 'H', 'Hearts')
   (4, 'D', 'Diamonds')
   ```
2. `card` — 32 rows, `card_id` aligned to `sheepshead.DECK_IDS` (1..32) so
   the Python side can insert FKs without a lookup round trip. Use the
   `DECK` ordering in `sheepshead.py` (TRUMP first, then FAIL). `code` is
   the short code ("QC", "AD"), `name` is from `CARD_FULL_NAMES`.
   Write a one-off Python generator (`scripts/gen_card_seed.py`) that reads
   `sheepshead.DECK` and emits the INSERT SQL so the seed stays in sync with
   the game constants.
3. `ai_model` — inserted at application startup instead (see Phase 5), not in
   seed, because the label depends on the loaded model.

### 3.5 Runtime DB pool (Python side)

1. `server/services/persistence/pool.py` — initialises an `asyncpg` pool on
   FastAPI startup; closes it on shutdown. Use `app.state.db_pool`.
2. `Settings.database_url` is the single source of truth.
3. The pool is **not** created in the analyze endpoint path — analyze stays
   a pure in-memory simulation and never touches the DB.
4. If `DATABASE_URL` is missing, the server should **fail fast** at startup
   (same pattern as `SHEEPSHEAD_MODEL_PATH`). Persistence is now a hard
   requirement, not optional.
5. **All queries must use parameterized placeholders** (`$1, $2, …` per
   `asyncpg`). Never build SQL with f-strings or `%`-formatting on user- or
   game-sourced values. Add a review checklist item in `CONTRIBUTING.md` (or
   equivalent) that calls this out, and lint any direct-string SQL literal
   containing non-constant values during code review. Applies equally to
   Phase 6 read paths.

**Acceptance for Phase 3:**
- `docker compose up -d postgres` then `npm --prefix db run migrate` creates
  all tables.
- `psql $DATABASE_URL -c "\dt"` lists every table from the schema SQL.
- `SELECT count(*) FROM suit;` → 4. `SELECT count(*) FROM card;` → 32.
- `\d cardset` shows `cards_hash TEXT NOT NULL` with a unique index.
- `\d ai_player` shows a unique index on `(ai_model_id, is_deterministic)`.
- `\d game` shows `time_closed` nullable (used as the in-progress sentinel).
- Server boots with `DATABASE_URL` + `SHEEPSHEAD_MODEL_LABEL` set; errors
  clearly if either is missing.

---

## Phase 4 — Persistent Player Identity & Display Name

**Goal:** each browser gets a stable `player_id` it keeps across tables, and
custom names are persisted. Default (hobbit) names remain ephemeral.

### 4.1 Data model reuse

The schema already has:

```sql
CREATE TABLE player(
    player_id UUID NOT NULL PRIMARY KEY,
    name TEXT NULL,
    time_created TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,
    last_updated TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL
);
```

- `name IS NULL` → default (ephemeral) user; we never persisted a custom name.
- `name IS NOT NULL` → user chose a name. This is what we display server-side
  when persisting game history.

### 4.2 Frontend: localStorage keys and flow

Two new stable localStorage keys (see `web/lib/storage.ts`):

| Key | Value | Written when | Read when |
|---|---|---|---|
| `sheepshead_player_id` | UUID string | on first successful `/join` response (lazy — not on landing page visits) | every subsequent join so the server can reuse the id |
| `sheepshead_display_name` | string or absent | only when user explicitly types a non-empty display name (see 4.4) | every load of `/` |

The existing per-table key `sheepshead_client_id_${tableId}` stays as-is.
`client_id` remains the short-lived session/auth token for a specific table;
`player_id` is the long-lived account-like identifier.

### 4.3 Backend: player provisioning endpoint

**Decision (Q1 resolved):** server-generated, but **lazily on first
`POST /api/tables/:id/join`** — we do not create a `player` row for every
landing-page visit. This keeps the `player` table clean of bots, scrapers,
and window-shoppers.

No dedicated `POST /api/players` endpoint. The player surface is just two
endpoints:

1. `GET /api/players/:id` — returns `{ player_id, name }`. 404 if unknown.
   The frontend calls this on load *only if* a `player_id` exists in
   localStorage, to hydrate `player.name` back into the display-name input.
   A 404 is a benign "your id was forgotten" signal: the client clears
   its localStorage `player_id` and falls back to "no id yet" state.
2. `PATCH /api/players/:id` — body `{ name: string | null }`. Sets the
   display name. Only meaningful after a first join has minted an id.

`POST /api/tables/:id/join` performs the actual creation:
- If the request has no `player_id`, insert a new row
  `(player_id = uuid4(), name = NULL)` and return it in the response.
- If the request has a `player_id` **that is not in `player`**, defensively
  upsert it with `name = NULL` so a DB reset / partial migration does not
  brick existing browsers that have a stale id in localStorage.
- Always echo `player_id` in the response so the client can persist it.

**Note on trust:** `player_id` is self-asserted. A bad actor can claim any
known UUID. We accept this — it is only used for non-sensitive
"who played in that game" attribution. Never promote `player_id` to an auth
credential.

### 4.4 Frontend: name-change detection

Current code in `web/app/page.tsx` already distinguishes "typed a name" from
"using placeholder". The effective display name is
`displayNameInput.trim() || displayPlaceholder`. Persist only when
`displayNameInput.trim()` is non-empty.

New flow (pseudocode for `HomePage`, lazy-join variant):

```ts
// On mount:
let playerId = localStorage.getItem('sheepshead_player_id') ?? null;
let savedName = localStorage.getItem('sheepshead_display_name') ?? '';

if (playerId) {
  // Hydrate persisted name from server (in case they cleared localStorage
  // but the DB still has it).
  const res = await fetch(`${API_BASE}/api/players/${playerId}`);
  if (res.status === 404) {
    // Server doesn't know this id — forget it, we'll remint on first join.
    localStorage.removeItem('sheepshead_player_id');
    playerId = null;
  } else if (res.ok) {
    const { name } = await res.json();
    if (name) {
      savedName = name;
      localStorage.setItem('sheepshead_display_name', name);
    }
  }
}

setDisplayNameInput(savedName);       // pre-fill input if previously saved
if (!savedName) {
  setDisplayPlaceholder(getRandomItem(HOBBIT_NAMES));  // ephemeral default
}

// On submit (create/join):
const isCustom = displayNameInput.trim().length > 0;
const finalName = isCustom ? displayNameInput.trim() : displayPlaceholder!;

// Join first — this is what mints the player_id if we don't have one yet.
const joinRes = await fetch(`${API_BASE}/api/tables/${tid}/join`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    display_name: finalName,
    player_id: playerId,        // null on first ever join
  }),
});
const joined = await joinRes.json();
if (joined.player_id && joined.player_id !== playerId) {
  playerId = joined.player_id;
  localStorage.setItem('sheepshead_player_id', playerId);
}

// If they typed a custom name, persist it now (post-join, id is guaranteed).
if (isCustom && savedName !== displayNameInput.trim()) {
  await fetch(`${API_BASE}/api/players/${playerId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: displayNameInput.trim() }),
  });
  localStorage.setItem('sheepshead_display_name', displayNameInput.trim());
}
```

Two small consequences of doing this post-join:
- The very first game of a brand-new user who typed a custom name has
  `player.name IS NULL` at the moment the game starts. That's fine for live
  play (the custom name is in `ClientConn.display_name` and flows into
  `game_player.name` as a snapshot). The PATCH updates `player.name` before
  the next game.
- If the user types a name and *cancels* the join (closes the tab without
  actually creating/joining a table), nothing is persisted. That matches the
  "don't pollute the player table" goal.

- If the user clears the input (back to placeholder), we do **not** null
  out the persisted `player.name` (per C2). They can navigate elsewhere
  and the saved name is still there on the next load. Hydration on mount
  (§4.4) repopulates the input from the server.
- Placeholder rotation on reload is unchanged when no custom name is saved.

### 4.5 Backend: wire `player_id` into `/join`

1. Extend `JoinTableRequest`:
   ```python
   class JoinTableRequest(BaseModel):
       display_name: str
       seat: Optional[int] = None
       player_id: Optional[UUID] = None   # NEW
   ```
2. On join:
   - If `player_id` is `None`, create a player row with `name = NULL` and
     **return the new id in the response** so legacy clients can catch up.
   - If `player_id` exists but no row is found in DB, upsert with `name =
     NULL`. This handles DB resets gracefully.
   - Store the `player_id` on the in-memory `ClientConn` so
     game-end persistence (Phase 5) can write it into `game_player.player_id`.
3. `JoinTableRequest.display_name` still flows through unchanged for the live
   table display. It is independent from `player.name` — a user can ephemerally
   play as "Bilbo" for one table and "Frodo" for another; `player.name`
   only reflects their explicitly-chosen name.

### 4.6 AI players at joined tables

AI occupant ids are ephemeral UUIDs (`server/main.py:578`). They never
correspond to `player` rows. They map to `ai_player` rows — see Phase 5.

**Acceptance for Phase 4:**
- Fresh browser → hits `/`, acquires a `player_id`, `player` row exists in DB.
- Typing a display name → `PATCH /api/players/:id` → `player.name` updated,
  `player.last_updated` bumped, localStorage reflects it.
- Clearing localStorage and refreshing → new player UUID is provisioned.
- Default-placeholder-only users: one `player` row per browser, all with
  `name IS NULL`, consistent with instruction (3.).

---

## Phase 5 — Game & Table Persistence (Writes)

**Goal:** persist everything needed to reconstruct or stat a past game.

### 5.1 When to write `game_table`

Per C4: write on first `POST /api/tables/:id/start`, not on table creation.
Tables that are created but never start a hand produce no DB rows. Rationale:
avoids empty-table clutter without costing anything useful for stats.
Implementation:

1. `services/persistence/games.py::ensure_game_table(conn, table)`:
   - Called at the top of `start_game`.
   - Uses `table.id` as the UUID pk (already a UUID — keep it the same).
   - Sets `time_created = now()`, `time_closed = NULL`.
   - If row already exists (resume / redeal-then-start), it's a no-op.
2. When `close_table` runs, set `game_table.time_closed = now()`.

### 5.2 When to write `game`

Persistence mirrors the in-memory `Game` lifecycle: each state transition
that produces durable information pushes the corresponding DB delta. The
hand is fully represented in the DB at every point after the deal, so a
future "live games" view can read straight from Postgres.

`time_closed` is the single source of truth for completeness:
`time_closed IS NULL` ⇒ in progress, `time_closed IS NOT NULL` ⇒ finished.
No separate boolean.

Lifecycle hooks:

| # | Hook | When | What's written |
|---|---|---|---|
| 1 | `persist_started_game` | end of `start_game` (after `Game.deal()`) | blind + 5 starting-hand cardsets; `game` row (deal-time fields, `time_closed=NULL`); 5 `game_player` rows (identifiers + `starting_hand_id`) |
| 2 | `persist_pick_resolved` | once the pick phase resolves — either someone picks, or all five pass | UPDATE `game.is_leaster`; if not leaster, UPDATE `game_player.is_picker` for the picker's seat |
| 3 | `persist_picker_decisions` | after the picker's bury is locked in (and any called-ace / under-call decisions are made) | upsert bury cardset; UPDATE `game.bury_id`, `is_alone`, `called_card_id`, `under_card_id` |
| 4 | `persist_partner_revealed` | when `Game.partner` becomes non-None in memory (immediate for JD mode at deal time, deferred to ace-played for called-ace, never for alone) | UPDATE `game_player.is_partner = true` for the partner's seat |
| 5 | `persist_trick_completed` | when a trick resolves | INSERT `trick` row + 5 `trick_card` rows for that trick |
| 6 | `persist_finalize_game` | `results_counted` transitions to `True` | UPDATE `game.time_closed = now()`; UPDATE `game_player.score` for all 5 seats |

Each hook is its own DB transaction. They can each fail independently
without compromising live play (§5.5).

Implementation notes:

1. Each `persist_*` function lives in `services/persistence/games.py`.
2. `persist_started_game` stashes the freshly-minted `game_id` on the
   in-memory `Table` (e.g. `Table.current_game_id`) so subsequent hooks
   can find the row without a lookup. It's cleared by
   `persist_finalize_game` so the next deal starts fresh.
3. Hooks 2–5 short-circuit if `Table.current_game_id` is `NULL` (hook 1
   never ran or failed). They log a debug line and return — there is no
   defensive fallback path. Simple and predictable: either every hook
   for a hand runs in order, or that hand is absent from the DB.
4. Hooks 2 and 3 may run as a single combined update for hands where the
   information arrives together (e.g. AI picker decides everything in
   one synchronous block); structure the call sites so each hook is
   *idempotent for the field set it touches* and either hook can be
   skipped without breaking the others.
5. Hook call sites in `server/main.py` (paths from current code; will
   move with the Phase 2.1 split):
   - hook 1: `start_game` end (`server/main.py:1063` area).
   - hooks 2–4: in the action handlers (`post_action`, `ai_take_turns`)
     where pick / call / bury / partner-reveal mutations happen.
   - hook 5: at trick-complete points in `post_action` and
     `ai_take_turns`.
   - hook 6: end-of-game blocks (`server/main.py:474–497` and
     `server/main.py:1223–1248`).

### 5.3 What to write

The full persisted shape is the union of what each hook writes. Per-hook
detail below.

**Hook 1 — `persist_started_game` (post-deal):**

1. **cardset** rows for the blind (2 cards) and each player's starting
   hand (5 rows × 6 cards). Use the upsert pattern below.
2. **game** row:
   ```
   game_id               = uuid4()
   game_table_id         = table.id
   is_double_on_the_bump = table.rules.doubleOnTheBump (default True)
   is_called_partner     = rules.partnerMode == PARTNER_BY_CALLED_ACE (1)
                           -- Table-level mode; independent of leaster.
   blind_id              = cardset_id for Game.blind
   time_created          = now()
   time_closed           = NULL                          -- in-progress sentinel
   is_alone, is_leaster, called_card_id, under_card_id, bury_id = NULL
   ```
3. **game_player** rows (one per seat, 5 total):
   - `game_id` — from 2.
   - `player_id` — from `ClientConn.player_id` for humans, `NULL` for AI.
   - `ai_player_id` — for AI (see §5.4), `NULL` for humans.
   - `name` — current `display_name` at deal time (snapshot).
   - `position` — seat 1..5.
   - `starting_hand_id` — cardset_id for that seat's 6-card starting hand.
   - `is_picker` / `is_partner` / `score` — `NULL` (set by later hooks).

**Hook 2 — `persist_pick_resolved`:**

```sql
UPDATE game SET is_leaster = $1 WHERE game_id = $2;
-- if not leaster:
UPDATE game_player SET is_picker = true
 WHERE game_id = $1 AND position = $2;
```

Pass `is_leaster = true` if all five players passed. Otherwise `false` and
also flip `is_picker` for the picker's seat. (Other seats keep `is_picker
IS NULL` — that's fine; it will stay NULL forever, distinguishable from
`false` if anyone cares, but in practice readers should query
`is_picker = true`.)

**Hook 3 — `persist_picker_decisions`:**

1. Upsert bury cardset (if non-leaster) via the helper.
2. ```sql
   UPDATE game
      SET bury_id        = $1,        -- NULL if leaster
          is_alone       = $2,        -- Game.alone_called, NULL if leaster
          called_card_id = $3,        -- DECK_IDS[Game.called_card] or NULL
          under_card_id  = $4         -- DECK_IDS[Game.under_card] if
                                      -- Game.is_called_under and
                                      -- Game.under_card else NULL
    WHERE game_id = $5;
   ```
   `Game.under_card` is the face-down card (e.g. "AC"), distinct from the
   called ace.

**Hook 4 — `persist_partner_revealed`:**

```sql
UPDATE game_player SET is_partner = true
 WHERE game_id = $1 AND position = $2;
```

The picker's row also gets `is_partner = true` if the picker plays the
secret-partner / alone-with-self case — match `Game.partner` exactly.
Hook is a no-op for alone hands (no partner) and for hands where
`Game.partner` is not yet revealed.

**Hook 5 — `persist_trick_completed`:**

1. INSERT one **trick** row:
   - `game_id`, `index` (0-based to match code).
   - `lead_player_id` — `game_player_id` of the leader (look up by seat).
   - `winning_player_id` — `game_player_id` of the winner.
   - `points` — `Game.trick_points[i]`.
2. INSERT 5 **trick_card** rows for that trick:
   - `trick_id` (from 1), `card_id`, `game_player_id`, `index` (0..4).

**Hook 6 — `persist_finalize_game`:**

```sql
UPDATE game SET time_closed = now() WHERE game_id = $1;
UPDATE game_player SET score = $1 WHERE game_id = $2 AND position = $3;
-- (one score UPDATE per seat, or a single CASE expression)
```

That's all hook 6 does. Bury, decisions, leaster status, tricks, and
partner have all been written by earlier hooks.

**Cardset upsert** (used by hooks 1 and 3):

```python
async def upsert_cardset(conn, card_ids: list[int]) -> int:
    cards_hash = ",".join(str(cid) for cid in sorted(card_ids))
    # Insert-then-fallback avoids a double round trip on the hot path.
    # With ON CONFLICT DO NOTHING, RETURNING only yields rows for inserts.
    row = await conn.fetchrow(
        "INSERT INTO cardset (cards_hash) VALUES ($1) "
        "ON CONFLICT (cards_hash) DO NOTHING "
        "RETURNING cardset_id",
        cards_hash,
    )
    if row is not None:
        cardset_id = row["cardset_id"]
        await conn.executemany(
            "INSERT INTO cardset_card (cardset_id, card_id) VALUES ($1, $2)",
            [(cardset_id, cid) for cid in card_ids],
        )
        return cardset_id
    return await conn.fetchval(
        "SELECT cardset_id FROM cardset WHERE cards_hash = $1",
        cards_hash,
    )
```

Only insert `cardset_card` rows on a *new* insert — detected by whether
the first `fetchrow` returned anything. Writes scale with *novel* hands,
not total hands played.

### 5.4 AI model + AI player bookkeeping

**Decision (Q7 resolved):** `ai_model.label` is sourced from a required
env var `SHEEPSHEAD_MODEL_LABEL`. If unset at startup, the server refuses
to boot (same fail-fast pattern as `SHEEPSHEAD_MODEL_PATH`). This prevents
two unrelated `.pt` files with the same filename from silently colliding
on the same row.

Implementation:

1. **Config.** Add `SHEEPSHEAD_MODEL_LABEL` to `Settings` and to
   `.env.example`. Document in README a naming convention, e.g.
   `pfsp-swish-v30M-2025-12` (model family + variant + size + date).
2. **Model row (upserted on startup, after `load_agent`):**
   - `INSERT ... ON CONFLICT (label) DO UPDATE SET label = EXCLUDED.label
      RETURNING ai_model_id;` so re-boots with the same label re-use the row.
   - Cache the resulting `ai_model_id` in `app.state.ai_model_id`.
3. **AI player row per (model, deterministic) pair:**
   - `is_deterministic` records whether the model ran with greedy/argmax
     action selection (`True`) or stochastic sampling (`False`) for that
     game. The current server calls `act(..., deterministic=True)` in
     `ai_take_turns` (see `server/main.py:425`).
   - On startup, upsert one `ai_player` row with
     `(ai_model_id = app.state.ai_model_id, is_deterministic = true)` — the
     current default. Use `ON CONFLICT (ai_model_id, is_deterministic) DO
     UPDATE SET ... RETURNING ai_player_id;` (add a unique index on this
     pair to the migration).
   - Cache the resulting `ai_player_id` in `app.state.ai_player_id`.
   - At game start time, pick `app.state.ai_player_id` based on the
     then-current `deterministic` flag. All five AI seats at the table
     reference the same `ai_player_id` — the server runs exactly one model
     variant at a time.
   - If we later add per-table or runtime determinism toggles, upsert a
     second `ai_player` row `(same model, is_deterministic = false)` lazily
     on first use and cache it alongside.

### 5.5 Error handling for persistence

- If a DB write fails in **any** hook, **log and continue**. In-memory
  game state is authoritative for the live session. Persistence must not
  break play.
- Hook 1 (start) failure: `Table.current_game_id` stays `NULL`. Hooks 2–6
  short-circuit — the hand is absent from the DB entirely. No fallback,
  no partial writes. Simple and predictable.
- Hook 2–5 failure: the hand has partial state (e.g. `is_picker` set but
  `bury_id` missing). Live play continues. The hand may or may not be
  finalized later depending on whether hook 6 succeeds.
- Hook 6 failure: the hand stays as a stale `time_closed IS NULL` row.
  A future janitor task can sweep these (e.g. `time_closed IS NULL AND
  time_created < now() - interval '1 day'`) — out of scope for this plan,
  but the row layout supports it.
- Each hook logs a structured line on failure including `table_id`,
  `game_id`, hook name, and the failing step so we can backfill manually
  if needed.
- (Optional, later) add a dead-letter file that captures failed game
  payloads to `./data/failed_games/*.json` for manual retry.

### 5.6 Schema questions carried into implementation

See **Appendix B** for a row-by-row mapping and **Appendix A** for the list of
unresolved questions.

**Acceptance for Phase 5:**
- Dealing a hand (post-deal, no actions yet) → one `game` row with
  `time_closed IS NULL`, `is_leaster IS NULL`, decision fields all NULL;
  five `game_player` rows with `starting_hand_id` populated and
  `is_picker / is_partner / score` all NULL. No `trick` rows.
- After the pick phase resolves with someone picking → `game.is_leaster
  = false` and exactly one `game_player.is_picker = true`.
- After all five pass → `game.is_leaster = true`; no `is_picker` updates.
- After the picker buries → `game.bury_id`, `is_alone`, `called_card_id`,
  `under_card_id` reflect their final values. Bury cardset row exists.
- After the partner is revealed (called-ace played, or JD played, or
  immediate for JD mode) → `game_player.is_partner = true` for the
  partner's seat. Alone hands never get an `is_partner = true` row.
- After each trick completes → one new `trick` row + 5 `trick_card` rows
  exist. Mid-hand inspection: `SELECT count(*) FROM trick WHERE game_id =
  $1` matches the number of completed tricks.
- Hand end (standard) → `time_closed` populated, all 5 `game_player.score`
  set, and the row is internally consistent (six tricks, thirty
  trick_cards, `is_leaster = false`, decision fields populated).
- Hand end (leaster) → same, but `is_leaster = true`, `bury_id` NULL,
  `is_alone` NULL, `called_card_id` NULL, `under_card_id` NULL, no
  `is_picker = true` row, no `is_partner = true` row.
- AI-only table → all five `game_player.ai_player_id` populated; all
  `player_id` NULL.
- Mixed human/AI table → correct `player_id` per seat.
- `game_player.position` + `starting_hand_id` correctly round-trips — you can
  reconstruct each starting hand via `cardset_card` joined on `card`.
- Server crash mid-hand → the stale `time_closed IS NULL` row remains;
  live play is unaffected on reboot. (Cleanup job is out of scope.)

---

## Phase 6 — Stats / History Read Paths (MVP)

This phase is *scoped out* of the stated deliverables — the user's instructions
cover **recording** games, not yet **displaying** them. We still scaffold the
minimum that is useful:

1. `GET /api/players/:id/games` — last N games the player was in, with their
   score, picker/partner status, and hand timestamp. **Filter
   `WHERE time_closed IS NOT NULL` by default** — in-progress hands have
   NULL results and would clutter the list. Optional
   `?include_in_progress=1` query param to surface them when the future
   "live games" view needs it.
2. `GET /api/tables/:id/history` — full `resultsHistory` equivalent, but read
   from DB rather than in-memory (so you can see a closed table later). Same
   `time_closed IS NOT NULL` filtering.

Defer UI work here; unblock it by ensuring the read APIs exist.

---

## Phase 7 — Ops, CI, Release Prep

### 7.1 CI

Add `.github/workflows/ci.yml` (or equivalent):

1. **Node**: `npm --prefix web ci`, `npm --prefix web run lint`,
   `npm --prefix web run typecheck`, `npm --prefix web run build`.
2. **Python**: `uv sync --extra server`, `uv run ruff check .`,
   `uv run pytest server/` (after we add tests).
3. **DB**: spin up postgres service container, run
   `npm --prefix db run migrate`, fail on dirty diff of
   `db/migrations/committed/`.

### 7.2 Minimal server tests

Add `server/tests/test_tables.py` using FastAPI `TestClient`:

- create table → list tables includes it;
- join table → client_id returned, seat assigned, player_id echoed;
- start game with <5 seats and no AI fill → 400;
- `/api/players` roundtrip;
- action endpoint rejects not-your-turn.

### 7.3 README updates

- New "Database" section: docker-compose up, migrate commands.
- New "Environment variables" section (`.env.example`).
- Remove outdated CORS note; document real prod-mode config.

### 7.4 Deployment (self-managed Postgres)

Initial production deployment is small. Plan assumes self-managed Postgres
(a single VPS or container, co-located with the server if footprint allows):

1. Provision a Postgres 18 instance. Enable daily base backups + WAL
   archiving to a bucket you control (`pgbackrest` or `wal-g`).
2. `DATABASE_URL` lives only in the deploy target's secret store (env var,
   not checked in).
3. Restrict Postgres to the server's private IP; no public port.
4. `SHADOW_DATABASE_URL` and `ROOT_DATABASE_URL` exist only in dev / CI.
   Production runs `graphile-migrate migrate` — no `reset`, no shadow DB.
5. Revisit (Neon / Supabase / RDS) once traffic or operational burden makes
   self-hosting painful.

### 7.5 Release checklist

- [ ] All three test suites green.
- [ ] `docker compose up` from clean checkout boots the full stack.
- [ ] Smoke: 2-human + 3-AI full hand → row in `game`, rows in
      `game_player`/`trick`/`trick_card`, custom display name persisted.
- [ ] Playing the same hand twice in a row (same seed) → only one new
      `cardset` row per logical set (blind/bury/hands), existing `cardset_id`s
      re-used on the second game.
- [ ] No `client_id`/`player_id` in URL access logs (only in request bodies).
- [ ] CORS allowlist reflects real production domain.
- [ ] Grep of persistence code reveals zero SQL built with f-strings /
      `%`-formatting / concatenation on non-constant values.

---

## Milestones & Suggested Sequencing

Recommended PR order — each PR is reviewable on its own and keeps `master`
deployable.

| # | PR | Scope | Depends on |
|---|---|---|---|
| 1 | Python deps upgrade | Phase 1.1 | — |
| 2 | Node deps upgrade | Phase 1.2 + 1.3 | — |
| 3 | server/main.py split | Phase 2.1 (mechanical) | 1, 2 |
| 4 | Security + cleanup | Phase 2.2 + 2.3 | 3 |
| 5 | DB infra + migrations (no app writes) | Phase 3 | 4 |
| 6 | Player identity (read/write players only) | Phase 4 | 5 |
| 7 | Game persistence writes | Phase 5 | 6 |
| 8 | Read API stubs + CI | Phase 6 + 7 | 7 |

Each PR should include the relevant README / `.env.example` changes so reviewers
can reproduce locally.

---

## Appendix A: Clarifications

Authoritative answers from Andrew (2026-04-21) to the ambiguities raised
while drafting this plan. Every design choice elsewhere in the plan is
consistent with these.

**C1. `player_id` generation — server-side, lazy.**
The server generates the UUID. Creation happens **lazily** on first
`POST /api/tables/:id/join` (not on landing-page loads), so browsers that
never join a table do not produce `player` rows. Subsequent joins include
the stored `player_id`; the server upserts defensively. See §4.3–§4.5.

**C2. Name-reset UX.**
Clearing the display-name input does **not** nullify the persisted
`player.name`. The input simply falls back to the rotating hobbit
placeholder for the session. A future "Reset name" control can expose
explicit nullification if needed.

**C3. `game.is_called_partner` — table-level, not per-hand.**
`is_called_partner = true` means the table was configured with
`PARTNER_BY_CALLED_ACE`; `false` means `PARTNER_BY_JD`. This is set from
`Table.rules["partnerMode"]` and is **independent of whether the hand
ended up as a leaster** — leasters occur under either partner mode.

**C4. `game_table` row written on first `/api/tables/:id/start`.**
No DB activity for tables that are created but never start a hand.

**C5. `trick.winning_player_id` populated for every completed trick.**
Including leaster tricks. The column stays nullable in the schema (it
must, for in-progress tricks) but we never persist a trick until it is
resolved.

**C6. `cardset` deduped via `cards_hash`.**
Identical logical card sets (blind, bury, starting hand) share a single
`cardset_id` across all games so that stats can group by identical deals.
A new `cardset.cards_hash TEXT NOT NULL UNIQUE` column (sorted card_ids
joined by `,`) drives the upsert. `cardset_card` rows are written once per
novel cardset. See §5.3 upsert pattern.

**C7. `ai_model.label`.**
Sourced from a **required** env var `SHEEPSHEAD_MODEL_LABEL`. Server
refuses to boot without it. Naming convention documented in README
(e.g. `pfsp-swish-v30M-2025-12`).

**C8. `ai_player` cardinality.**
One row per `(ai_model_id, is_deterministic)`. The `is_deterministic` flag
records whether the model ran with greedy (`act(..., deterministic=True)`,
current default) or stochastic action selection for that game. All five AI
seats at a table reference the same `ai_player_id` — the server runs a
single model variant at a time. Future multi-model support would add
per-model rows at that time, not now. See §5.4.

**C9. `game.game_id` — fresh UUID per hand.**
One `game` row per `sheepshead.Game()` instance. The table-↔-hand
relationship lives on `game.game_table_id`.

**C10. `game.under_card_id` — the face-down card.**
When a picker uses the under-call variant (`Game.is_called_under == true`),
`under_card_id = DECK_IDS[Game.under_card]`. That is the fail card placed
face-down, **not** the called ace (which is already captured by
`called_card_id`). Sourced from `sheepshead.py:238,710–711`.

**C11. DB driver & SQL safety.**
`asyncpg` directly, no ORM. **All queries are parameterized** — use
positional placeholders (`$1, $2, …`). No f-strings / `%`-formatting /
concatenation in SQL built from user or game data. This is a hard rule;
code review rejects any violation. See §3.5 and Phase 6.

**C12. Production Postgres — self-managed.**
Small initial deployment. Self-managed Postgres 18 co-located with (or
network-adjacent to) the server, with automated base backups and WAL
archiving. Revisit managed options once traffic or ops burden justifies
the move. See §7.4.

**C13. In-progress hands are persisted with live updates (added
2026-05-02).**
Persistence mirrors the in-memory `Game` lifecycle: a `game` row and 5
`game_player` rows are written at deal time, then **each meaningful state
transition pushes its own DB delta** — pick resolution flips
`is_leaster` and `is_picker`; the picker's bury/decisions update
`bury_id`, `is_alone`, `called_card_id`, `under_card_id`; partner reveal
flips `is_partner`; each completed trick inserts its `trick` +
`trick_card` rows. The end-of-hand step only stamps `time_closed = now()`
and writes scores. This unblocks a future "see games in progress" view
without piggy-backing on the in-memory `TableManager`. Tables that are
*created but never start a hand* still produce no DB rows (C4 stands —
start is the trigger, not creation). `time_closed` is the single source
of truth for completeness — `IS NULL` means in progress, `IS NOT NULL`
means finished — so we did **not** add a separate `is_complete` boolean
(rejected to avoid two sources of truth). On any persistence failure we
log and continue; subsequent hooks for that hand short-circuit if the
start hook never produced a row, and there is **no defensive
fallback** — every hand either flows through the full hook sequence or
is absent from the DB. Stale `time_closed IS NULL` rows left by crashes
or partial-write failures are tolerated; a janitor sweep is out of
scope. See §5.2, §5.3, §5.5.

---

## Appendix B: Proposed Data-Mapping Table

This is the reference a junior engineer can cross-check while writing
persistence code in Phase 5. Columns are *output*; values are *source*.

### `game_table`

| column | source |
|---|---|
| `game_table_id` | `Table.id` (existing UUID) |
| `name` | `Table.name` |
| `time_created` | set on `ensure_game_table` (first start) |
| `time_closed` | set on `close_table` |

### `game`

`hook` indicates which §5.2 lifecycle hook populates the column.
1=start, 2=pick-resolved, 3=picker-decisions, 6=finalize.

| column | hook | source |
|---|---|---|
| `game_id` | 1 | fresh `uuid4()` per hand (C9) |
| `game_table_id` | 1 | `Table.id` |
| `is_double_on_the_bump` | 1 | `Table.rules["doubleOnTheBump"]`, default `True` |
| `is_called_partner` | 1 | `Table.rules["partnerMode"] == 1` (PARTNER_BY_CALLED_ACE). Table-level mode, independent of leaster. |
| `is_leaster` | 2 | `Game.is_leaster` (decided once the pick phase resolves) |
| `is_alone` | 3 | `Game.alone_called` (NULL if leaster) |
| `called_card_id` | 3 | `DECK_IDS[Game.called_card]` or NULL |
| `under_card_id` | 3 | `DECK_IDS[Game.under_card]` if `Game.is_called_under` and `Game.under_card` else NULL |
| `bury_id` | 3 | newly inserted `cardset.cardset_id` for `Game.bury`, NULL if leaster |
| `blind_id` | 1 | newly inserted `cardset.cardset_id` for `Game.blind` |
| `time_created` | 1 | `now()` at hand start |
| `time_closed` | 6 | `now()` at finalize. NULL while in progress — the single source of truth for completeness (C13). |

### `game_player` (5 rows per game)

| column | hook | source |
|---|---|---|
| `game_player_id` | 1 | auto |
| `game_id` | 1 | from `game` |
| `player_id` | 1 | `ClientConn.player_id` for humans, NULL for AI |
| `ai_player_id` | 1 | `app.state.ai_player_id` for AI, NULL for humans |
| `name` | 1 | `ClientConn.display_name` / `Occupant.display_name` snapshot at deal time |
| `position` | 1 | 1..5 |
| `starting_hand_id` | 1 | cardset_id for that seat's initial 6-card hand |
| `is_picker` | 2 | set to `true` for the picker's seat once pick resolves; stays NULL elsewhere |
| `is_partner` | 4 | set to `true` for `Game.partner`'s seat when the partner is revealed; stays NULL elsewhere (and forever for alone hands) |
| `score` | 6 | `Player.get_score()` |

### `trick` (one per completed trick — written by hook 5 as each trick resolves)

| column | source |
|---|---|
| `trick_id` | auto |
| `game_id` | from `game` |
| `index` | 0..5 |
| `lead_player_id` | `game_player_id` of seat `Game.leaders[i]` |
| `winning_player_id` | `game_player_id` of seat `Game.trick_winners[i]` |
| `points` | `Game.trick_points[i]` |

### `trick_card` (5 rows per completed trick — written by hook 5)

| column | source |
|---|---|
| `trick_card_id` | auto |
| `trick_id` | from `trick` |
| `card_id` | `DECK_IDS[Game.history[trick_index][seat_play_order]]` |
| `game_player_id` | `game_player_id` of the playing seat |
| `index` | 0..4 (order played within the trick) |

### `cardset` / `cardset_card`

Deduped across all games. Each logical set (blind, bury, starting hand)
produces one `cardset` row only the *first* time it is seen — identified by
`cards_hash` (sorted comma-joined `card_id`s, UNIQUE). Subsequent identical
sets re-use the existing `cardset_id`, so stats queries naturally group by
identical hands. `cardset_card` rows are written only on first insert.

---

*End of plan.*
