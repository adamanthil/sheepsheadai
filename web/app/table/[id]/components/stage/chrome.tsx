import React from "react";
import { SeatAvatar, ds } from "../../../../../lib/ds";
import type { SeatRole } from "../../lib/phase";
import type { SeatView } from "./types";
import styles from "../Stage.module.css";

export function roleBadge(role: SeatRole, small?: boolean) {
  const fs = small ? { fontSize: 8, padding: "1px 5px" } : undefined;
  if (role === "PICKER")
    return (
      <span className={`${ds.badge} ${ds.badgeAccent}`} style={fs}>
        Picker
      </span>
    );
  if (role === "PARTNER")
    return (
      <span className={`${ds.badge} ${ds.badgeGold}`} style={fs}>
        Partner
      </span>
    );
  if (role === "PASS")
    return (
      <span className={`${ds.badge} ${ds.badgeQuiet}`} style={fs}>
        Pass
      </span>
    );
  if (role === "PENDING")
    return (
      <span
        className={`${ds.badge} ${ds.badgeQuiet}`}
        style={{ ...fs, animation: "ssPulse 1.4s ease-in-out infinite" }}
      >
        Deciding
      </span>
    );
  return null;
}

export function tone(role: SeatRole): "default" | "picker" | "partner" {
  return role === "PICKER"
    ? "picker"
    : role === "PARTNER"
      ? "partner"
      : "default";
}

// Compact name-plate that floats off a seat's played card toward the table rim,
// always reading on the outside of the ellipse: outer seats below their card,
// inner top seats out to the side. `compact` trims it for the tighter mobile
// ring (smaller avatar, no seat number).
export function RingChip({
  seat,
  plate,
  compact,
  badgeSide,
}: {
  seat: SeatView;
  plate: "above" | "below" | "left" | "right";
  compact?: boolean;
  // When set, the role badge floats to this side of the name block (absolutely
  // positioned) instead of stacking below it, so its presence never shifts the
  // name's vertical position. Used for the inner mobile seats.
  badgeSide?: "left" | "right";
}) {
  const plateClass =
    plate === "above"
      ? styles.chipAbove
      : plate === "left"
        ? styles.chipLeft
        : plate === "right"
          ? styles.chipRight
          : styles.chipBelow;
  const badge = roleBadge(seat.role, compact);
  const badgeSideClass = badgeSide
    ? badgeSide === "left"
      ? styles.chipBadgesLeft
      : styles.chipBadgesRight
    : "";
  return (
    <div
      className={`${styles.chip} ${plateClass} ${compact ? styles.chipCompact : ""}`}
    >
      <SeatAvatar
        name={seat.name}
        isAI={seat.isAI}
        tone={tone(seat.role)}
        size={compact ? 26 : 32}
      />
      <div className={styles.chipText}>
        <div
          className={`${styles.chipName} ${compact ? styles.chipNameSm : ""}`}
        >
          {seat.name}
        </div>
        {!compact && (
          <div className={ds.overline} style={{ fontSize: 9 }}>
            Seat {seat.absSeat}
          </div>
        )}
      </div>
      {badge && (
        <div className={`${styles.chipBadges} ${badgeSideClass}`}>{badge}</div>
      )}
    </div>
  );
}
