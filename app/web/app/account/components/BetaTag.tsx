import React from "react";
import styles from "./BetaTag.module.css";

/** Marks a feature that is newer and less settled than the rest of the
 * site. Sits after the feature's name wherever it is entered. */
export default function BetaTag() {
  return <span className={styles.beta}>Beta</span>;
}
