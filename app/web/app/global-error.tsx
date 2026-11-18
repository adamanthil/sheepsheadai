"use client";

// The last-resort boundary: an error in the root layout itself, which
// app/error.tsx sits inside and so cannot catch. It replaces the whole
// document, so it renders its own <html> and <body> and leans on nothing
// from the layout (fonts, global CSS).
import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";

export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  useEffect(() => {
    console.error("Unhandled root error:", error);
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 16,
          padding: 24,
          textAlign: "center",
          fontFamily: "system-ui, sans-serif",
          background: "#f3f1ea",
          color: "#1a1a1a",
        }}
      >
        <h2 style={{ margin: 0 }}>Something went wrong</h2>
        {/* A hard navigation: nothing of the crashed document survives. */}
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
        <a href="/" style={{ padding: "8px 16px" }}>
          Back to lobby
        </a>
      </body>
    </html>
  );
}
