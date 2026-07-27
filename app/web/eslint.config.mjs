import nextPlugin from "@next/eslint-plugin-next";
import tseslint from "typescript-eslint";

// Flat config lints only **/*.{js,mjs,cjs} unless a `files` glob says
// otherwise, so every block below opts the TypeScript sources in explicitly.
// Without that this config silently linted 2 .mjs files and none of the app.
const SOURCES = ["**/*.{js,mjs,cjs,ts,tsx}"];

export default [
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      // Generated from openapi.json by `npm run gen:api`; drift is caught in
      // CI by a git diff against the regenerated file, not by lint.
      "lib/api.gen.ts",
    ],
  },
  ...tseslint.configs.recommended.map((c) => ({ ...c, files: SOURCES })),
  {
    ...nextPlugin.configs["core-web-vitals"],
    files: SOURCES,
  },
];
