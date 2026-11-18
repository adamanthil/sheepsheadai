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
  // The default rewrites a failed fetch's message in place to "Failed to
  // fetch (host)", for every catch in the app to see. Keep errors as the
  // browser threw them; the host is still added to what Sentry receives.
  enhanceFetchErrorMessages: "report-only",
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
