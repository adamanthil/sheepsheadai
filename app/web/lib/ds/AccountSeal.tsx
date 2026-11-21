import type React from "react";

/** Gold seal marking a verified account; hover or a screen reader gives
 * the username. Sits on a seat avatar's corner or inline in text. */
export default function AccountSeal({
  username,
  size,
  style,
}: {
  username: string;
  /** Diameter in px. */
  size: number;
  style?: React.CSSProperties;
}) {
  return (
    <span
      role="img"
      aria-label={`Account @${username}`}
      title={`@${username}`}
      style={{
        width: size,
        height: size,
        flex: "none",
        borderRadius: "50%",
        background: "var(--gold)",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        ...style,
      }}
    >
      <svg
        viewBox="0 0 12 12"
        width={size * 0.62}
        height={size * 0.62}
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
