import { defineConfig } from "vitest/config";

// Unit tests for the frontend's pure modules (state selectors, message
// parsing, optimistic moves), colocated as *.test.ts. Components and flows
// are the Playwright suite's (e2e/).
export default defineConfig({
  test: {
    include: ["**/*.test.ts"],
    exclude: ["node_modules/**", ".next/**", "e2e/**"],
    environment: "node",
  },
});
