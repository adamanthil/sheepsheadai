import React from "react";
import type { Tone } from "../../../lib/review";
import styles from "./BigFigures.module.css";

/** A row of headline figures in the display face: the totals at the top
 * of the AI review and admin pages, and a decision's three scores. */
export function BigFigures({
  compact = false,
  minWidth,
  children,
}: {
  /** Three across under a rule, a size down: inside a decision's detail. */
  compact?: boolean;
  /** The narrowest a figure's column may get before the row wraps, in
   * pixels, where the default is too tight for what the figures say. */
  minWidth?: number;
  children: React.ReactNode;
}) {
  return (
    <dl
      className={`${styles.figures} ${compact ? styles.compact : ""}`}
      style={
        minWidth
          ? ({ "--figure-min": `${minWidth}px` } as React.CSSProperties)
          : undefined
      }
    >
      {children}
    </dl>
  );
}

/** One figure: a small-caps label, the figure, and a line under it. */
export function BigFigure({
  label,
  note,
  tone,
  children,
}: {
  label: string;
  note?: React.ReactNode;
  /** Colours the figure as a cost or a gain against the AI. */
  tone?: Tone;
  children: React.ReactNode;
}) {
  return (
    <div>
      <dt>{label}</dt>
      <dd data-tone={tone}>{children}</dd>
      {note != null && <dd className={styles.note}>{note}</dd>}
    </div>
  );
}
