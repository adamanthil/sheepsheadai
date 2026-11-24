// The config entry point: the package root resolves to CommonJS under Node,
// whose named exports an ES module config cannot import.
import { withSentryConfig } from "@sentry/nextjs/config";

const isProd = process.env.NODE_ENV === "production";

// The API may be same-origin (behind the reverse proxy) or a separate origin
// named by NEXT_PUBLIC_API_BASE; connect-src must allow both the HTTP and
// websocket forms. Dev stays permissive because the API host is whatever
// hostname the browser used (see lib/apiBase.ts).
function connectSrc() {
  // GoatCounter's counting endpoint (app/components/Analytics.tsx).
  const sources = ["'self'", "https://sheepsheadai.goatcounter.com"];
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (base) {
    sources.push(base, base.replace(/^http/, "ws"));
  }
  if (!isProd) {
    sources.push("http:", "https:", "ws:", "wss:");
  }
  return sources.join(" ");
}

const csp = [
  "default-src 'self'",
  // Next.js injects inline bootstrap scripts; nonce-based CSP is a later
  // upgrade. Dev additionally needs eval for react-refresh.
  `script-src 'self' 'unsafe-inline'${isProd ? "" : " 'unsafe-eval'"}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  `connect-src ${connectSrc()}`,
  "object-src 'none'",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

// Hostnames (no scheme, no port) permitted to reach dev-server internals.
// Next allows localhost only, so the HMR websocket -- the one dev request that
// carries an Origin header -- 403s when a phone loads the app over the LAN.
// Opt in with DEV_ALLOWED_ORIGINS=andrews-macbook-pro.local; dev-server-only,
// with no effect on `next build` or production.
const allowedDevOrigins = (process.env.DEV_ALLOWED_ORIGINS ?? "")
  .split(",")
  .map((h) => h.trim())
  .filter(Boolean);

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle for the Docker image.
  output: "standalone",
  allowedDevOrigins,
  experimental: {},
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Content-Security-Policy", value: csp },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
        ],
      },
    ];
  },
};

// Source maps reach Sentry only from a build that has SENTRY_AUTH_TOKEN (the
// CI image build, as a BuildKit secret; see Dockerfile): they are uploaded,
// then deleted, so the image never serves them. Any other build skips the
// upload. The release is the image's content tag, as the api's is.
const sentryAuthToken = process.env.SENTRY_AUTH_TOKEN;

export default withSentryConfig(nextConfig, {
  org: "sheepshead-ai",
  project: "sheepshead-web",
  authToken: sentryAuthToken,
  // No commit association: the image builds from app/web alone, without git.
  // CI's deploy job attaches the commits to this release once it is live.
  release: {
    name: process.env.NEXT_DEPLOYMENT_ID || undefined,
    setCommits: false,
  },
  sourcemaps: {
    disable: !sentryAuthToken,
    deleteSourcemapsAfterUpload: true,
  },
  silent: !sentryAuthToken,
  telemetry: false,
});
