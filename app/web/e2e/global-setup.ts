import { execSync, spawnSync } from "node:child_process";
import { connect } from "node:net";
import path from "node:path";
import { E2E_DATABASE_URL } from "../playwright.config";

const CONTAINER = "sheepshead-e2e-pg";
const dbDir = path.resolve(__dirname, "../../db");

function sh(command: string, options: { cwd?: string; env?: NodeJS.ProcessEnv } = {}) {
  execSync(command, { stdio: "inherit", ...options });
}

function portReachable(host: string, port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = connect({ host, port, timeout: 1500 });
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    const fail = () => {
      socket.destroy();
      resolve(false);
    };
    socket.once("error", fail);
    socket.once("timeout", fail);
  });
}

export default async function globalSetup() {
  const url = new URL(E2E_DATABASE_URL);
  const port = Number(url.port || 5432);
  let startedContainer = false;

  if (!(await portReachable(url.hostname, port))) {
    sh(
      `docker run -d --rm --name ${CONTAINER} ` +
        "-e POSTGRES_USER=sheepshead -e POSTGRES_PASSWORD=sheepshead " +
        `-e POSTGRES_DB=sheepshead -p ${port}:5432 postgres:18`,
    );
    startedContainer = true;
    for (let attempt = 0; ; attempt++) {
      const ready =
        spawnSync("docker", ["exec", CONTAINER, "pg_isready", "-U", "sheepshead"])
          .status === 0;
      if (ready) break;
      if (attempt >= 60) throw new Error("postgres container never became ready");
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  sh("npx graphile-migrate migrate", {
    cwd: dbDir,
    env: { ...process.env, DATABASE_URL: E2E_DATABASE_URL },
  });

  return () => {
    if (startedContainer) {
      spawnSync("docker", ["rm", "-f", CONTAINER]);
    }
  };
}
