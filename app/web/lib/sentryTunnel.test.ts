import { describe, expect, it } from "vitest";

import {
  clientKey,
  envelopeUrl,
  readCapped,
  readdressEnvelope,
  windowLimiter,
} from "./sentryTunnel";

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

describe("readCapped", () => {
  const streamOf = (chunks: number[], onPull?: () => void) => {
    const queue = [...chunks];
    return new ReadableStream<Uint8Array>({
      pull(controller) {
        onPull?.();
        const next = queue.shift();
        if (next === undefined) controller.close();
        else controller.enqueue(new Uint8Array(next).fill(7));
      },
    });
  };

  it("returns a body within the limit whole", async () => {
    const out = await readCapped(streamOf([3, 4]), 7);
    expect(out).toEqual(new Uint8Array(7).fill(7));
  });

  it("treats a missing body as empty", async () => {
    expect(await readCapped(null, 10)).toEqual(new Uint8Array(0));
  });

  it("stops reading once the limit is passed", async () => {
    let pulls = 0;
    const stream = streamOf(Array(100).fill(10), () => pulls++);
    expect(await readCapped(stream, 25)).toBeNull();
    expect(pulls).toBeLessThan(10);
  });
});

describe("clientKey", () => {
  it("keeps an IPv4 address", () => {
    expect(clientKey("203.0.113.9")).toBe("203.0.113.9");
    expect(clientKey("::ffff:203.0.113.9")).toBe("203.0.113.9");
  });

  it("folds IPv6 to its /64", () => {
    const key = "2001:db8:0:1::/64";
    expect(clientKey("2001:db8:0:1::1")).toBe(key);
    expect(clientKey("2001:0DB8:0000:0001:aaaa:bbbb:cccc:dddd")).toBe(key);
    expect(clientKey("2001:db8:0:2::1")).not.toBe(key);
    expect(clientKey("::1")).toBe("0:0:0:0::/64");
  });

  it("passes through what isn't an address", () => {
    expect(clientKey("a:b")).toBe("a:b");
  });
});

describe("windowLimiter", () => {
  it("allows the limit per key per window", () => {
    const limiter = windowLimiter(2, 1000);
    expect(limiter.allow("a", 0)).toBe(true);
    expect(limiter.allow("a", 10)).toBe(true);
    expect(limiter.allow("a", 20)).toBe(false);
    expect(limiter.allow("b", 20)).toBe(true);
    expect(limiter.retryAfter(20)).toBe(1);
    expect(limiter.allow("a", 1000)).toBe(true);
  });
});
