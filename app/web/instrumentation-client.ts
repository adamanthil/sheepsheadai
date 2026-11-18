// Browser error reporting; see lib/errorReporting.ts for why the DSN is a
// placeholder. Runs before the app hydrates, so it sees errors from the
// very first render. Errors only: no tracing, replay or profiling.
import * as Sentry from "@sentry/nextjs";

import {
  DATA_COLLECTION,
  PLACEHOLDER_DSN,
  TUNNEL_PATH,
} from "./lib/errorReporting";

Sentry.init({
  dsn: PLACEHOLDER_DSN,
  tunnel: TUNNEL_PATH,
  // `next dev` would only fill the console with failed tunnel posts; a
  // production build reports, and the server decides whether anything is
  // forwarded.
  enabled: process.env.NODE_ENV === "production",
  dataCollection: DATA_COLLECTION,
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
