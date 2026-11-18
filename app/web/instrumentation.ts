// Server-side error reporting for the Next.js server (rendering, route
// handlers). Off unless SENTRY_DSN is set at runtime, as it is in the
// production compose file; read here, never inlined at build time.
import * as Sentry from "@sentry/nextjs";

import { DATA_COLLECTION } from "./lib/errorReporting";

export function register() {
  if (process.env.NEXT_RUNTIME === "nodejs" && process.env.SENTRY_DSN) {
    Sentry.init({
      dsn: process.env.SENTRY_DSN,
      environment: process.env.SENTRY_ENVIRONMENT || "production",
      dataCollection: DATA_COLLECTION,
    });
  }
}

export const onRequestError = Sentry.captureRequestError;
