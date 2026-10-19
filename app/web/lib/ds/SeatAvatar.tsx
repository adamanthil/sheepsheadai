import React from "react";

export type SeatTone = "default" | "picker" | "partner" | "you";

interface SeatAvatarProps {
  name?: string;
  isAI?: boolean;
  /** Verified account username; draws the account seal in the AI badge's
   * corner (the two never coexist: an AI seat has no account). */
  account?: string | null;
  /** Draws the dealer button in the top-left corner, clear of the AI badge
   * and account seal. */
  dealer?: boolean;
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

/** Initial-in-a-disc avatar, toned by role, with an optional AI or
 * account corner badge and dealer button. */
export default function SeatAvatar({
  name,
  isAI,
  account,
  dealer,
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
      {!isAI && account && <AccountSeal username={account} size={size} />}
      {dealer && <DealerButton size={size} />}
    </div>
  );
}

/** Small grey "D" button marking the hand's dealer. Kept quieter than the
 * role badges: it is context for the pick decision, not a side in the hand. */
function DealerButton({ size }: { size: number }) {
  const d = Math.max(12, Math.round(size * 0.4));
  return (
    <span
      role="img"
      aria-label="Dealer"
      title="Dealer"
      style={{
        position: "absolute",
        top: -3,
        left: -4,
        width: d,
        height: d,
        boxSizing: "border-box",
        borderRadius: "50%",
        background: "var(--muted)",
        border: "1.5px solid var(--bg-page)",
        color: "var(--card-paper)",
        fontFamily: "var(--font-ui)",
        fontSize: Math.round(d * 0.6),
        fontWeight: 600,
        lineHeight: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      D
    </span>
  );
}

/** Gold seal marking a seat held by a verified account; hover or a screen
 * reader gives the username. */
function AccountSeal({ username, size }: { username: string; size: number }) {
  const d = Math.max(12, Math.round(size * 0.4));
  return (
    <span
      role="img"
      aria-label={`Account @${username}`}
      title={`@${username}`}
      style={{
        position: "absolute",
        bottom: -3,
        right: -4,
        width: d,
        height: d,
        borderRadius: "50%",
        background: "var(--gold)",
        border: "1.5px solid var(--bg-page)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <svg
        viewBox="0 0 12 12"
        width={d * 0.62}
        height={d * 0.62}
        aria-hidden="true"
      >
        <path
          d="M2.5 6.3 5 8.6 9.6 3.6"
          fill="none"
          stroke="var(--card-paper)"
          strokeWidth="1.9"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}
