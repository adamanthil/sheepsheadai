import React from "react";
import styles from "./Strapline.module.css";

/** The rules of the game in one italic line. Font scales with viewport so it
 *  stays on a single line from mobile to desktop. */
export default function Strapline() {
  return (
    <span className={styles.strapline}>
      <span className={styles.club}>♣</span> Queens high
      <span className={styles.dot}> · </span>
      <span className={styles.diamond}>♦</span> Diamonds are trump
      <span className={styles.dot}> · </span>
      Take 61
    </span>
  );
}
