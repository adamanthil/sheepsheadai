import React, { useState } from "react";
import { ds } from "../../../../lib/ds";
import { useBusy } from "../../../../lib/hooks/useBusy";
import type { TableView } from "../../../../lib/types";
import { formatClock } from "../lib/tableSelectors";
import { isAiSeat, isTakeableSeat, nameForSeat } from "../utils/seatMath";
import styles from "./SpectatorBar.module.css";

/** Time left on the acting player's turn clock, red for the last 5s. */
function TurnClock({ secondsLeft }: { secondsLeft: number | null }) {
  if (secondsLeft === null) return null;
  return (
    <span
      className={`${styles.clock} ${secondsLeft <= 5 ? styles.clockUrgent : ""}`}
      aria-label={`${secondsLeft} seconds left to move`}
    >
      {formatClock(secondsLeft)}
    </span>
  );
}

interface SpectatorBarProps {
  table: TableView;
  actorName: string;
  /** Resolves false when the server refused the seat. */
  onTakeSeat: (seat: number) => Promise<boolean>;
  isMobile: boolean;
  secondsLeft: number | null;
  /** Set when the turn clock moved this spectator out of a seat. */
  homeOccupant?: string | null;
  /** Set for a newcomer who joined mid-hand: the AI they take over from
   * when the next hand is dealt. */
  queuedOccupant?: string | null;
}

/** Stands in for the hand while you watch: one button per AI seat
 * that was dealt to the AI, each taking over that AI's hand where it
 * stands. A player the turn clock moved out is offered only their own seat
 * back while its AI still holds it (the server enforces both rules). A
 * newcomer who joined mid-hand is told whose seat they get at the next
 * deal, with Take seat now as the quieter option. */
export default function SpectatorBar(props: SpectatorBarProps) {
  // Held on success: the bar goes away when the seat's state arrives.
  const taking = useBusy();
  const [pressed, setPressed] = useState<number | null>(null);
  const seatOf = (occ: string | null | undefined) =>
    [1, 2, 3, 4, 5].find(
      (s) => occ && props.table.seatOccupants[String(s)] === occ,
    );
  const homeSeat = seatOf(props.homeOccupant);
  const queuedSeat =
    homeSeat === undefined ? seatOf(props.queuedOccupant) : undefined;
  const aiSeats =
    homeSeat !== undefined
      ? [homeSeat]
      : queuedSeat !== undefined
        ? [queuedSeat].filter((s) => isTakeableSeat(s, props.table))
        : [1, 2, 3, 4, 5].filter(
            (s) => isAiSeat(s, props.table) && isTakeableSeat(s, props.table),
          );
  const queuedNote = queuedSeat !== undefined && (
    <span className={styles.helper}>
      You take over from {nameForSeat(queuedSeat, props.table)} next hand
    </span>
  );
  const status = props.actorName
    ? `Watching · waiting for ${props.actorName}…`
    : "Watching";

  const seatButtons = aiSeats.map((seat) => (
    <button
      key={seat}
      className={`${ds.btn} ${seat === queuedSeat ? ds.btnGhost : ds.btnAccent} ${taking.busy && seat === pressed ? ds.btnBusy : ""}`}
      onClick={() => {
        setPressed(seat);
        void taking.run(() => props.onTakeSeat(seat), { hold: true });
      }}
      disabled={taking.busy}
      aria-busy={taking.busy && seat === pressed}
    >
      {seat === homeSeat
        ? `Take back seat ${seat}`
        : seat === queuedSeat
          ? "Take seat now"
          : `Take seat ${seat} · ${nameForSeat(seat, props.table)}`}
    </button>
  ));
  if (props.isMobile) {
    return (
      <div className={styles.mob}>
        <div className={styles.mobHelper}>
          {status}
          <TurnClock secondsLeft={props.secondsLeft} />
        </div>
        <div className={styles.mobRow}>
          {queuedNote}
          {seatButtons}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.desk}>
      <span className={styles.waiting}>
        {status}
        <TurnClock secondsLeft={props.secondsLeft} />
      </span>
      <div className={styles.deskRight}>
        {queuedNote}
        {seatButtons.length || queuedNote ? (
          seatButtons
        ) : (
          <span className={styles.helper}>No seats to take over this hand</span>
        )}
      </div>
    </div>
  );
}
