import React from "react";
import { SeatAvatar, ds } from "../../../../lib/ds";
import styles from "./Seat.module.css";

export interface SeatInfo {
  seat: number;
  name: string;
  isAI: boolean;
  isYou: boolean;
  /** The seated human's verified account username, or null. */
  account: string | null;
  /** The seated human's client id, when the viewer (the host) may remove
   * them; null otherwise. */
  removableId: string | null;
  /** Whether the viewer may move into this seat (an AI holds it). */
  takeable: boolean;
  /** Typing a chat message right now. */
  typing?: boolean;
}

interface SeatProps {
  seat: SeatInfo;
  onTake: (seat: number) => void;
  /** The viewer's move into this seat is on its way to the server. */
  taking?: boolean;
  /** Asks to remove the seat's player; the layout confirms it. */
  onRemove: (seat: SeatInfo) => void;
}

function avatar(seat: SeatInfo, size: number) {
  return (
    <SeatAvatar
      name={seat.name}
      isAI={seat.isAI}
      account={seat.account}
      size={size}
      typing={seat.typing}
      tone={seat.isYou ? "you" : "default"}
    />
  );
}

/** Desktop: one compact tile in the row of five. An AI seat offers Take
 * over; another player's seat offers the host Remove. */
export function SeatTile({ seat, onTake, taking, onRemove }: SeatProps) {
  const sub = [
    seat.isYou ? "You" : null,
    `Seat ${seat.seat}`,
    seat.account ? `@${seat.account}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div
      className={`${styles.tile} ${seat.isYou ? styles.yours : ""}`}
      role="group"
      aria-label={`Seat ${seat.seat}`}
    >
      {avatar(seat, 40)}
      <div className={styles.who}>
        <div className={styles.name}>{seat.name}</div>
        <div className={styles.sub}>{sub}</div>
      </div>
      {seat.takeable && !seat.isYou && (
        <button
          type="button"
          className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${taking ? ds.btnBusy : ""}`}
          onClick={() => onTake(seat.seat)}
          disabled={taking}
          aria-busy={taking}
        >
          Take over
        </button>
      )}
      {seat.removableId && (
        <button
          type="button"
          className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.remove}`}
          onClick={() => onRemove(seat)}
        >
          Remove
        </button>
      )}
    </div>
  );
}

/** Mobile: one seat in the strip of five. Tapping an AI seat sits there;
 * the host taps another player's seat to remove them. */
export function SeatChip({ seat, onTake, taking, onRemove }: SeatProps) {
  const body = (
    <>
      {avatar(seat, 42)}
      <span className={styles.chipName}>{seat.name}</span>
      <span className={`${styles.sub} ${seat.isYou ? styles.you : ""}`}>
        {seat.isYou ? "You" : `Seat ${seat.seat}`}
      </span>
    </>
  );
  if (seat.takeable && !seat.isYou) {
    return (
      <button
        type="button"
        className={`${styles.chip} ${taking ? ds.btnBusy : ""}`}
        onClick={() => onTake(seat.seat)}
        disabled={taking}
        aria-busy={taking}
        aria-label={`Sit in ${seat.name}’s seat, seat ${seat.seat}`}
      >
        {body}
      </button>
    );
  }
  if (seat.removableId) {
    return (
      <button
        type="button"
        className={styles.chip}
        onClick={() => onRemove(seat)}
        aria-label={`Remove ${seat.name}`}
      >
        {body}
      </button>
    );
  }
  return <div className={styles.chip}>{body}</div>;
}
