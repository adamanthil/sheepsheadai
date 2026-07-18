import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

// End-to-end smoke harness. Requires Docker (an ephemeral Postgres is started
// by e2e/global-setup.ts) unless E2E_DATABASE_URL points at a migrated
// database. Run from app/web: `npm run e2e`.

const repoRoot = path.resolve(__dirname, "../..");
const apiBase = "http://127.0.0.1:9100";

export const E2E_DATABASE_URL =
  process.env.E2E_DATABASE_URL ??
  "postgres://sheepshead:sheepshead@127.0.0.1:5433/sheepshead";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command:
        "uv run -- python -m uvicorn server.app:create_app --factory --host 127.0.0.1 --port 9100",
      cwd: repoRoot,
      url: `${apiBase}/health`,
      timeout: 240_000,
      reuseExistingServer: !process.env.CI,
      env: {
        SHEEPSHEAD_MODEL_PATH: path.join(repoRoot, "final_pfsp_swish_ppo.pt"),
        SHEEPSHEAD_MODEL_LABEL: "e2e-smoke",
        DATABASE_URL: E2E_DATABASE_URL,
        ENV: "development",
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
