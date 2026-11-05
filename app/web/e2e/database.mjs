// Brings up the e2e database before the API server starts: Playwright
// launches its webServers before globalSetup, so this runs as the first half
// of the API server's command instead (playwright.config.ts).
//
// A database already reachable at DATABASE_URL is used as is (CI's service
// container, or one you pointed E2E_DATABASE_URL at); otherwise a throwaway
// Postgres container is started, which e2e/teardown.mjs removes. Either way
// it is migrated to the current schema.
import { execSync, spawnSync } from "node:child_process";
import { connect } from "node:net";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const CONTAINER = "sheepshead-e2e-pg";

const here = path.dirname(fileURLToPath(import.meta.url));
const dbDir = path.resolve(here, "../../db");

function portReachable(host, port) {
  return new Promise((resolve) => {
    const socket = connect({ host, port, timeout: 1500 });
    const done = (ok) => {
      socket.destroy();
      resolve(ok);
    };
    socket.once("connect", () => done(true));
    socket.once("error", () => done(false));
    socket.once("timeout", () => done(false));
  });
}

async function main() {
  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) throw new Error("DATABASE_URL is not set");
  const url = new URL(databaseUrl);
  const port = Number(url.port || 5432);

  if (!(await portReachable(url.hostname, port))) {
    execSync(
      `docker run -d --rm --name ${CONTAINER} ` +
        `-e POSTGRES_USER=${url.username} -e POSTGRES_PASSWORD=${url.password} ` +
        `-e POSTGRES_DB=${url.pathname.slice(1)} -p ${port}:5432 postgres:18`,
      { stdio: "inherit" },
    );
    for (let attempt = 0; ; attempt++) {
      const ready =
        spawnSync("docker", [
          "exec",
          CONTAINER,
          "pg_isready",
          "-U",
          url.username,
        ]).status === 0;
      if (ready) break;
      if (attempt >= 60)
        throw new Error("postgres container never became ready");
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  execSync("npx graphile-migrate migrate", {
    stdio: "inherit",
    cwd: dbDir,
    env: { ...process.env, DATABASE_URL: databaseUrl },
  });
}

// Run as a script; imported (by e2e/teardown.mjs) for CONTAINER only.
if (import.meta.url === pathToFileURL(process.argv[1]).href) await main();
