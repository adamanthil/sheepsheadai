// The pure half of the error-reporting tunnel (app/monitoring/route.ts):
// re-addressing a browser envelope to the server's real DSN. See
// lib/errorReporting.ts for why the browser doesn't hold that DSN.

/** Sentry's envelope endpoint for `dsn`, or null if it doesn't parse. */
export function envelopeUrl(dsn: string): string | null {
  let url: URL;
  try {
    url = new URL(dsn);
  } catch {
    return null;
  }
  // https://<key>@<host>[/<path>]/<project id>
  const segments = url.pathname.split("/").filter(Boolean);
  const projectId = segments.pop();
  if (!url.username || !projectId || !/^\d+$/.test(projectId)) return null;
  const prefix = segments.length ? `/${segments.join("/")}` : "";
  const query = new URLSearchParams({
    sentry_key: url.username,
    sentry_version: "7",
  });
  return `${url.protocol}//${url.host}${prefix}/api/${projectId}/envelope/?${query}`;
}

/**
 * The envelope with its header's DSN replaced by `dsn`, or null if it isn't
 * an envelope. An envelope is a JSON header line followed by its items,
 * which pass through byte for byte.
 */
export function readdressEnvelope(
  body: Uint8Array,
  dsn: string,
): Uint8Array<ArrayBuffer> | null {
  const newline = body.indexOf(0x0a);
  const headerBytes = newline === -1 ? body : body.subarray(0, newline);
  let header: unknown;
  try {
    header = JSON.parse(new TextDecoder().decode(headerBytes));
  } catch {
    return null;
  }
  if (typeof header !== "object" || header === null || Array.isArray(header)) {
    return null;
  }
  const head = new TextEncoder().encode(JSON.stringify({ ...header, dsn }));
  const rest = newline === -1 ? new Uint8Array(0) : body.subarray(newline);
  const out = new Uint8Array(head.length + rest.length);
  out.set(head);
  out.set(rest, head.length);
  return out;
}
