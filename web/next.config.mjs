const isProd = process.env.NODE_ENV === "production";

// The API may be same-origin (behind the reverse proxy) or a separate origin
// named by NEXT_PUBLIC_API_BASE; connect-src must allow both the HTTP and
// websocket forms. Dev stays permissive because the API host is whatever
// hostname the browser used (see lib/apiBase.ts).
function connectSrc() {
  const sources = ["'self'"];
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

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
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

export default nextConfig;
