// Resolution order:
// 1. NEXT_PUBLIC_API_BASE, if set non-empty at build time (inlined by Next).
// 2. Production builds: "" — same-origin, for the reverse-proxy layout where
//    Caddy routes /api and /ws to the backend on the same domain.
// 3. Dev: the API on port 9000 of whatever host the browser is using.
const explicit = process.env.NEXT_PUBLIC_API_BASE;

export const API_BASE =
  explicit !== undefined && explicit !== ""
    ? explicit
    : process.env.NODE_ENV === "production"
      ? ""
      : typeof window === "undefined"
        ? "http://localhost:9000"
        : `${window.location.protocol}//${window.location.hostname}:9000`;
