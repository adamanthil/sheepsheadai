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

/**
 * The whole of `stream` if it is at most `maxBytes`, else null. Stops
 * reading at the limit, so a body with no (or a false) Content-Length
 * cannot make the server buffer more than that.
 */
export async function readCapped(
  stream: ReadableStream<Uint8Array> | null,
  maxBytes: number,
): Promise<Uint8Array | null> {
  if (!stream) return new Uint8Array(0);
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > maxBytes) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const out = new Uint8Array(size);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

/**
 * The key a client's envelopes are counted under, from the address the
 * reverse proxy saw. As in the api (server/api/ratelimit.py), IPv6 counts
 * per /64, since one subscriber is routinely handed a whole /64, and
 * IPv4-mapped IPv6 counts as its IPv4 address.
 */
export function clientKey(address: string): string {
  const addr = address.trim().toLowerCase();
  if (!addr.includes(":")) return addr;
  const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/.exec(addr);
  if (mapped) return mapped[1];
  const [head, tail] = addr.split("::");
  const groups = head ? head.split(":") : [];
  if (tail !== undefined) {
    const rest = tail ? tail.split(":") : [];
    groups.push(
      ...Array(Math.max(0, 8 - groups.length - rest.length)).fill("0"),
    );
    groups.push(...rest);
  }
  if (groups.length !== 8) return addr; // not an address; count it as given
  return `${groups
    .slice(0, 4)
    .map((g) => g.replace(/^0+(?=.)/, ""))
    .join(":")}::/64`;
}

/**
 * A fixed-window counter: `allow(key, now)` is true for the first `limit`
 * calls per key in each `windowMs`. Every key is forgotten when the window
 * turns, so memory is bounded by the clients seen in one window.
 */
export function windowLimiter(limit: number, windowMs: number) {
  const counts = new Map<string, number>();
  let windowStart = -Infinity;
  return {
    allow(key: string, now: number): boolean {
      if (now - windowStart >= windowMs) {
        counts.clear();
        windowStart = now;
      }
      const used = counts.get(key) ?? 0;
      if (used >= limit) return false;
      counts.set(key, used + 1);
      return true;
    },
    /** Seconds until the current window ends, for Retry-After. */
    retryAfter(now: number): number {
      return Math.max(1, Math.ceil((windowStart + windowMs - now) / 1000));
    },
  };
}
