// Forwards the browser's Sentry envelopes, re-addressed to the runtime
// SENTRY_DSN (see lib/errorReporting.ts). Without a DSN, as in development
// and CI, envelopes are accepted and dropped.
//
// Anyone can post here, and what is forwarded spends the Sentry project's
// event quota, so each client gets a small budget and a size cap.
import {
  clientKey,
  envelopeUrl,
  readCapped,
  readdressEnvelope,
  windowLimiter,
} from "../../lib/sentryTunnel";

// Browser error events are a few KB, tens with a deep stack and a full
// breadcrumb trail; anything this size isn't one of ours. The Caddyfile
// holds the same cap, so an oversized post normally never gets here.
const MAX_BODY_BYTES = 256 * 1024;

// Per client per minute. The SDK drops repeats of an error, so a page
// that is merely broken stays far below this.
const ENVELOPES_PER_MINUTE = 30;
const limiter = windowLimiter(ENVELOPES_PER_MINUTE, 60_000);

// The address Caddy saw. It replaces any X-Forwarded-For the client sent
// (no trusted_proxies are configured), so the last entry is the peer's.
// Without a proxy (development) every client shares one budget.
function client(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  const last = forwarded?.split(",").pop()?.trim();
  return last ? clientKey(last) : "direct";
}

// Sentry's answers the browser SDK reads, to back off when rate-limited.
const PASSED_BACK = ["content-type", "retry-after", "x-sentry-rate-limits"];

export async function POST(request: Request): Promise<Response> {
  const dsn = process.env.SENTRY_DSN;
  if (!dsn) return new Response(null, { status: 204 });
  const upstream = envelopeUrl(dsn);
  if (!upstream) {
    console.error("SENTRY_DSN is not a valid DSN; dropping browser errors");
    return new Response(null, { status: 204 });
  }

  const now = Date.now();
  if (!limiter.allow(client(request), now)) {
    // The browser SDK reads Retry-After on a 429 and holds its events.
    return new Response("Too many envelopes", {
      status: 429,
      headers: { "retry-after": String(limiter.retryAfter(now)) },
    });
  }

  const declared = Number(request.headers.get("content-length") ?? 0);
  if (declared > MAX_BODY_BYTES) {
    return new Response("Envelope too large", { status: 413 });
  }
  const body = await readCapped(request.body, MAX_BODY_BYTES);
  if (!body) return new Response("Envelope too large", { status: 413 });
  const envelope = readdressEnvelope(body, dsn);
  if (!envelope) return new Response("Invalid envelope", { status: 400 });

  let answer: Response;
  try {
    answer = await fetch(upstream, {
      method: "POST",
      headers: {
        "content-type": "application/x-sentry-envelope",
        // Lets Sentry name the browser and OS. The client's IP is
        // deliberately not forwarded.
        "user-agent": request.headers.get("user-agent") ?? "",
      },
      body: envelope,
    });
  } catch (err) {
    console.error("forwarding a browser error to Sentry failed:", err);
    return new Response("Upstream unreachable", { status: 502 });
  }
  const headers = new Headers();
  for (const name of PASSED_BACK) {
    const value = answer.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(await answer.arrayBuffer(), {
    status: answer.status,
    headers,
  });
}
