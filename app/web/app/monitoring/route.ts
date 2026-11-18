// Forwards the browser's Sentry envelopes, re-addressed to the runtime
// SENTRY_DSN (see lib/errorReporting.ts). Without a DSN, as in development
// and CI, envelopes are accepted and dropped.
import { envelopeUrl, readdressEnvelope } from "../../lib/sentryTunnel";

// Browser error events are a few KB; anything this size isn't one of ours.
const MAX_BODY_BYTES = 1024 * 1024;

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

  const declared = Number(request.headers.get("content-length") ?? 0);
  if (declared > MAX_BODY_BYTES) {
    return new Response("Envelope too large", { status: 413 });
  }
  const body = new Uint8Array(await request.arrayBuffer());
  if (body.length > MAX_BODY_BYTES) {
    return new Response("Envelope too large", { status: 413 });
  }
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
