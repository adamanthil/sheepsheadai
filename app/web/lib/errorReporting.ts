// Shared pieces of the error-reporting setup (instrumentation*.ts and
// app/monitoring/route.ts).
//
// The web image is built once in CI and tagged by content, so nothing
// deployment-specific is baked into it. That includes the Sentry DSN: the
// browser SDK is initialised with a fixed placeholder and sends every
// envelope to TUNNEL_PATH on our own origin, where the Next server swaps in
// the real DSN from its runtime SENTRY_DSN (or drops the envelope when there
// is none). As side benefits the browser never talks to a third-party host,
// so neither the CSP nor ad blockers get in the way.

export const TUNNEL_PATH = "/monitoring";

// Parses as a DSN so the SDK runs, but names no real project or host.
export const PLACEHOLDER_DSN = "https://browser@tunnel.invalid/0";

// What the SDKs may attach to an event, for both the browser and the Next
// server: no user identity, cookies (session tokens), headers or bodies
// (passwords, emails, chat). Stack traces and breadcrumbs carry the rest.
export const DATA_COLLECTION: {
  userInfo: boolean;
  cookies: boolean;
  httpHeaders: boolean;
  httpBodies: [];
} = {
  userInfo: false,
  cookies: false,
  httpHeaders: false,
  httpBodies: [],
};
