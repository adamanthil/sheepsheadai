import { describe, expect, it } from "vitest";

import { envelopeUrl, readdressEnvelope } from "./sentryTunnel";

const DSN = "https://abc123@o42.ingest.us.sentry.io/4507";

describe("envelopeUrl", () => {
  it("builds the ingest endpoint from a DSN", () => {
    expect(envelopeUrl(DSN)).toBe(
      "https://o42.ingest.us.sentry.io/api/4507/envelope/?sentry_key=abc123&sentry_version=7",
    );
  });

  it("keeps a self-hosted path prefix", () => {
    expect(envelopeUrl("https://k@errors.example.com/sentry/7")).toBe(
      "https://errors.example.com/sentry/api/7/envelope/?sentry_key=k&sentry_version=7",
    );
  });

  it("rejects what isn't a DSN", () => {
    expect(envelopeUrl("")).toBeNull();
    expect(envelopeUrl("https://o42.ingest.sentry.io/4507")).toBeNull();
    expect(envelopeUrl("https://k@host/not-a-project")).toBeNull();
  });
});

describe("readdressEnvelope", () => {
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  it("replaces the header DSN and keeps the items byte for byte", () => {
    const items = '\n{"type":"event"}\n{"message":"boom"}\n';
    const body = enc.encode(
      `{"dsn":"https://browser@tunnel.invalid/0","sent_at":"t"}${items}`,
    );
    const out = readdressEnvelope(body, DSN);
    expect(out).not.toBeNull();
    const text = dec.decode(out!);
    const [head, ...rest] = text.split("\n");
    expect(JSON.parse(head)).toEqual({ dsn: DSN, sent_at: "t" });
    expect("\n" + rest.join("\n")).toBe(items);
  });

  it("rejects a body whose first line isn't a JSON object", () => {
    expect(readdressEnvelope(enc.encode("not json\n{}"), DSN)).toBeNull();
    expect(readdressEnvelope(enc.encode("[1]\n{}"), DSN)).toBeNull();
  });
});
