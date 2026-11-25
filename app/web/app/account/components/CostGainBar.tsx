import React from "react";
import styles from "./CostGainBar.module.css";

/** Points of expected score that fill one side of the bar. The same for
 * every row, so hands compare down a column; a longer bar stops at full
 * width and ends in a point, to say that it runs on. */
const FULL = 3;

/** What a hand's choices cost (red, growing left from the centre line)
 * and gained (green, growing right) against the AI's. Decorative: the
 * figures are given in words by whatever labels the row. */
export default function CostGainBar({
  cost,
  gained,
  quiet = false,
}: {
  cost: number;
  gained: number;
  /** Drawn faintly, for a hand that does not stand out. */
  quiet?: boolean;
}) {
  const side = (value: number) => `${Math.min(value / FULL, 1) * 50}%`;
  return (
    <span
      className={`${styles.bar} ${quiet ? styles.quiet : ""}`}
      aria-hidden="true"
    >
      <span
        className={`${styles.cost} ${cost > FULL ? styles.costOver : ""}`}
        style={{ width: side(cost) }}
      />
      <span
        className={`${styles.gained} ${gained > FULL ? styles.gainedOver : ""}`}
        style={{ width: side(gained) }}
      />
      <span className={styles.centre} />
    </span>
  );
}
