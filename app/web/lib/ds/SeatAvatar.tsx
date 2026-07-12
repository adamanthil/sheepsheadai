import React from "react";

export type SeatTone = "default" | "picker" | "partner" | "you";

interface SeatAvatarProps {
  name?: string;
  isAI?: boolean;
  size?: number;
  tone?: SeatTone;
}

const TONES: Record<SeatTone, { bg: string; border: string; fg: string }> = {
  default: {
    bg: "var(--bg-page-deep)",
    border: "var(--rule-strong)",
    fg: "var(--ink)",
  },
  picker: {
    bg: "var(--accent)",
    border: "var(--accent)",
    fg: "var(--card-paper)",
  },
  partner: {
    bg: "var(--gold)",
    border: "var(--gold)",
    fg: "var(--card-paper)",
  },
  you: { bg: "var(--card-paper)", border: "var(--ink)", fg: "var(--ink)" },
};

/** Initial-in-a-disc avatar, toned by role, with an optional AI corner badge. */
export default function SeatAvatar({
  name,
  isAI,
  size = 44,
  tone = "default",
}: SeatAvatarProps) {
  const initial = (name || "?").slice(0, 1).toUpperCase();
  const s = TONES[tone] ?? TONES.default;
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: s.bg,
        border: "1px solid " + s.border,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "var(--font-display)",
        fontSize: size * 0.5,
        color: s.fg,
        position: "relative",
        flexShrink: 0,
      }}
    >
      {initial}
      {isAI && (
        <span
          style={{
            position: "absolute",
            bottom: -3,
            right: -4,
            fontFamily: "var(--font-ui)",
            fontSize: 9,
            fontWeight: 600,
            letterSpacing: "0.12em",
            background: "var(--ink)",
            color: "var(--bg-page)",
            padding: "1px 4px",
            borderRadius: 2,
          }}
        >
          AI
        </span>
      )}
    </div>
  );
}
