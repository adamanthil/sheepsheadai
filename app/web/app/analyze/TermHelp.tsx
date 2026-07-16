import React from "react";
import styles from "./TermHelp.module.css";

interface TermProps {
  /** The term as it appears in the UI (e.g. "Brier"). */
  label: React.ReactNode;
  /** Full Wikipedia URL for readers who want the formal treatment. */
  wiki?: string;
  /** Anchor the popover's right edge to the term instead of its left —
   * use for terms near the right edge of a clipping container. */
  align?: "left" | "right";
  /** Plain-language definition, written for someone with only a basic
   * stats/ML background. */
  children: React.ReactNode;
}

/** A technical term that expands into a plain-language definition.
 *
 * Renders as the term with a dotted underline; clicking it opens a small
 * definition card (native <details>, so it needs no JS state and closes
 * independently of other terms).
 */
export default function Term({
  label,
  wiki,
  align = "left",
  children,
}: TermProps) {
  return (
    <details className={styles.term}>
      <summary className={styles.summary}>
        {label}
        <span className={styles.marker} aria-hidden>
          ?
        </span>
      </summary>
      <div
        className={`${styles.popover} ${
          align === "right" ? styles.popoverRight : ""
        }`}
      >
        <div className={styles.definition}>{children}</div>
        {wiki && (
          <a
            className={styles.wikiLink}
            href={wiki}
            target="_blank"
            rel="noreferrer"
          >
            Wikipedia ↗
          </a>
        )}
      </div>
    </details>
  );
}
