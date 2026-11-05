import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

// End-to-end harness. Requires Docker (a throwaway Postgres is started by
// e2e/database.mjs) unless E2E_DATABASE_URL points at a reachable database,
// which is migrated and used as is. Run from app/web: `npm run e2e`.

const repoRoot = path.resolve(__dirname, "../..");
const apiBase = "http://127.0.0.1:9100";

// Its own port, never the dev database's (docker-compose maps that to 5433):
// a reachable database here is used as is, so a shared port would have the
// suite migrate and fill the dev database.
const E2E_DATABASE_URL =
  process.env.E2E_DATABASE_URL ??
  "postgres://sheepshead:sheepshead@127.0.0.1:5544/sheepshead";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  globalTeardown: "./e2e/teardown.mjs",
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      // The database comes up first: webServers start before globalSetup.
      command:
        "node app/web/e2e/database.mjs && uv run -- python -m uvicorn server.app:create_app --factory --host 127.0.0.1 --port 9100",
      cwd: repoRoot,
      url: `${apiBase}/health`,
      timeout: 240_000,
      reuseExistingServer: !process.env.CI,
      env: {
        SHEEPSHEAD_MODEL_PATH: path.join(repoRoot, "models/pfsp-swish-30m.pt"),
        SHEEPSHEAD_MODEL_LABEL: "e2e-smoke",
        DATABASE_URL: E2E_DATABASE_URL,
        ENV: "development",
        // Every spec creates and joins from 127.0.0.1, which a whole run
        // would push past the per-IP table cap and rate limits (both have
        // their own server tests).
        SHEEPSHEAD_MAX_TABLES_PER_IP: "100",
        RATELIMIT_ENABLED: "false",
        // A short turn clock so specs that sit out turns finish quickly.
        SHEEPSHEAD_TURN_TIMEOUT_SECONDS: "2",
        // And a short hold on a dropped player's seat, for the same reason.
        SHEEPSHEAD_DISCONNECT_GRACE_SECONDS: "5",
      },
    },
    {
      // Production build + start: NEXT_PUBLIC_API_BASE is inlined at build
      // time, and dev-mode streaming hydration is flaky under the headless
      // shell, so the harness tests the real build.
      command: "npx next build && npx next start -p 3100",
      cwd: __dirname,
      url: "http://127.0.0.1:3100",
      timeout: 360_000,
      reuseExistingServer: !process.env.CI,
      env: {
        NEXT_PUBLIC_API_BASE: apiBase,
      },
    },
  ],
});
