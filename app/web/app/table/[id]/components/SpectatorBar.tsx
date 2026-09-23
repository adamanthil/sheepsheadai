import React from "react";
import { ds } from "../../../../lib/ds";
import type { TableView } from "../../../../lib/types";
import { isAiSeat, nameForSeat } from "../utils/seatMath";
import { TurnClock } from "./ActionBar";
import styles from "./ActionBar.module.css";

interface SpectatorBarProps {
  table: TableView;
  actorName: string;
  onTakeSeat: (seat: number) => void;
  onShowScores: () => void;
  isMobile: boolean;
  secondsLeft: number | null;
}

/** Stands in for the ActionBar while you watch: one button per AI seat,
 * each taking over that AI's hand where it stands. */
export default function SpectatorBar(props: SpectatorBarProps) {
  const aiSeats = [1, 2, 3, 4, 5].filter((s) => isAiSeat(s, props.table));
  const status = props.actorName
    ? `Watching · waiting for ${props.actorName}…`
    : "Watching";

  const seatButtons = aiSeats.map((seat) => (
    <button
      key={seat}
      className={`${ds.btn} ${ds.btnAccent}`}
      onClick={() => props.onTakeSeat(seat)}
    >
      Take seat {seat} · {nameForSeat(seat, props.table)}
    </button>
  ));
  const scores = (
    <button
      className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
      onClick={props.onShowScores}
    >
      Scores
    </button>
  );

  if (props.isMobile) {
    return (
      <div className={styles.mob}>
        <div className={styles.mobHelper}>
          {status}
          <TurnClock secondsLeft={props.secondsLeft} />
        </div>
        <div className={styles.mobRow}>{seatButtons}</div>
        <div className={styles.mobRow}>{scores}</div>
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
          <span className={styles.helper}>No AI seats to take over</span>
        )}
        {scores}
      </div>
    </div>
  );
}
