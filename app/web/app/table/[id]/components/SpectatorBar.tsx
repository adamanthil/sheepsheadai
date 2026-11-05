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
}

/** Stands in for the hand while you watch: one button per AI seat
 * that was dealt to the AI, each taking over that AI's hand where it
 * stands. A player the turn clock moved out is offered only their own seat
 * back while its AI still holds it (the server enforces both rules). */
export default function SpectatorBar(props: SpectatorBarProps) {
  // Held on success: the bar goes away when the seat's state arrives.
  const taking = useBusy();
  const [pressed, setPressed] = useState<number | null>(null);
  const homeSeat = [1, 2, 3, 4, 5].find(
    (s) =>
      props.homeOccupant &&
      props.table.seatOccupants[String(s)] === props.homeOccupant,
  );
  const aiSeats =
    homeSeat !== undefined
      ? [homeSeat]
      : [1, 2, 3, 4, 5].filter(
          (s) => isAiSeat(s, props.table) && isTakeableSeat(s, props.table),
        );
  const status = props.actorName
    ? `Watching · waiting for ${props.actorName}…`
    : "Watching";

  const seatButtons = aiSeats.map((seat) => (
    <button
      key={seat}
      className={`${ds.btn} ${ds.btnAccent} ${taking.busy && seat === pressed ? ds.btnBusy : ""}`}
      onClick={() => {
        setPressed(seat);
        void taking.run(() => props.onTakeSeat(seat), { hold: true });
      }}
      disabled={taking.busy}
      aria-busy={taking.busy && seat === pressed}
    >
      {seat === homeSeat
        ? `Take back seat ${seat}`
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
        <div className={styles.mobRow}>{seatButtons}</div>
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
        {seatButtons.length ? (
          seatButtons
        ) : (
          <span className={styles.helper}>No seats to take over this hand</span>
        )}
      </div>
    </div>
  );
}
