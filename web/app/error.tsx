"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Unhandled page error:", error);
  }, [error]);

  return (
    <div
      style={{
        minHeight: "60vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 16,
        padding: 24,
        textAlign: "center",
      }}
    >
      <h2 style={{ margin: 0 }}>Something went wrong</h2>
      <p style={{ margin: 0, opacity: 0.7, fontSize: 14 }}>
        The table may still be live — try again, or head back to the lobby.
      </p>
      <div style={{ display: "flex", gap: 12 }}>
        <button onClick={reset} style={{ padding: "8px 16px" }}>
          Try again
        </button>
        <a href="/" style={{ padding: "8px 16px" }}>
          Back to lobby
        </a>
      </div>
    </div>
  );
}
