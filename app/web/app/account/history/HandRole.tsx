import React from "react";
import { ROLE_LABEL } from "../../../lib/roles";
import type { HandHistoryRow } from "../../../lib/types";
import styles from "./history.module.css";

/** The player's part in a hand, as a history row names it. */
export default function HandRole({ hand }: { hand: HandHistoryRow }) {
  return (
    <>
      {ROLE_LABEL[hand.role]}
      {hand.alone && hand.role === "picker" && (
        <span className={styles.muted}> · alone</span>
      )}
    </>
  );
}
