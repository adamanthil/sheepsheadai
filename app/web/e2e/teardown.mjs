// Removes the throwaway database e2e/database.mjs starts, if it started one.
import { spawnSync } from "node:child_process";
import { CONTAINER } from "./database.mjs";

export default function globalTeardown() {
  spawnSync("docker", ["rm", "-f", CONTAINER], { stdio: "ignore" });
}
