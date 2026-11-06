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
      // Third-party scripts copied verbatim (see app/components/Analytics.tsx).
      "public/vendor/**",
    ],
  },
  ...tseslint.configs.recommended.map((c) => ({ ...c, files: SOURCES })),
  {
    ...nextPlugin.configs["core-web-vitals"],
    files: SOURCES,
  },
  {
    // CSS-module class names are a build detail: their format changed under
    // the specs once already and broke four of them silently. Find elements
    // by role and accessible name (adding them to the component if missing),
    // or by a data-* attribute as a last resort.
    files: ["e2e/**/*.ts"],
    rules: {
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "Literal[value=/\\[class/], TemplateElement[value.raw=/\\[class/]",
          message:
            "No class selectors in e2e specs: use getByRole/getByLabel/getByText, or a data-* attribute.",
        },
      ],
    },
  },
];
