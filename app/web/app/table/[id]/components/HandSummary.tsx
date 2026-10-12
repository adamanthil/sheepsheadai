import React from "react";
import { ds } from "../../../../lib/ds";
import { nameForSeat } from "../utils/seatMath";
import type { FinalState, TableView } from "../../../../lib/types";
import styles from "./HandSummary.module.css";

const SEATS = [1, 2, 3, 4, 5];

/** One value per seat, under the seat's name. */
export function SeatGrid({
  table,
  values,
  label,
}: {
  table: TableView;
  values: number[] | undefined;
  label: string;
}) {
  return (
    <>
      <div className={styles.gridLabel}>{label}</div>
      <div className={styles.scoresGrid}>
        {SEATS.map((seat) => (
          <div key={seat} className={styles.scoreCell}>
            <div className={styles.scoreName}>{nameForSeat(seat, table)}</div>
            <div className={styles.scoreValue}>
              {(values && values[seat - 1]) || 0}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

/** Who picked and partnered (or won the leaster), and the points split. */
export default function HandSummary({
  final,
  table,
}: {
  final: FinalState;
  table: TableView;
}) {
  const multiplier = final.multiplier ?? 1;

  // Names run inline here ("Picker: X · Partner: Y"); clamp them so a long
  // one ellipsizes rather than wrapping the line.
  const name = (seat: number) => {
    const label = nameForSeat(seat, table);
    return (
      <strong className={`${ds.nameClamp} ${styles.subName}`} title={label}>
        {label}
      </strong>
    );
  };

  return (
    <>
      {multiplier > 1 && (
        <div className={styles.sub}>
          <span className={`${ds.badge} ${ds.badgeAccent}`}>
            {multiplier}× Doubler
          </span>
        </div>
      )}
      {final.mode === "leaster" ? (
        <div className={styles.sub}>Winner: {name(final.winner || 0)}</div>
      ) : (
        <>
          <div className={styles.sub}>
            Picker: {name(final.picker || 0)} · Partner:{" "}
            {name(final.partner || 0)}
          </div>
          <div className={styles.sub}>
            Picker score: <strong>{final.picker_score}</strong> · Defenders:{" "}
            <strong>{final.defender_score}</strong>
          </div>
        </>
      )}
    </>
  );
}
