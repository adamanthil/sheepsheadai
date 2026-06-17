import React from "react";
import { parseCard, suitSymbol, isRedSuit } from "./cardUtils";
import styles from "./PlayingCard.module.css";

export interface PlayingCardProps {
  /** Card code: "QC", "10H", "AD", "__" (face down), "UNDER" (buried-under placeholder). */
  code: string;
  /** Card width in px; height is derived as w * 1.45. */
  w?: number;
  playable?: boolean;
  dim?: boolean;
  ariaHidden?: boolean;
  className?: string;
}

export default function PlayingCard({
  code,
  w = 64,
  playable,
  dim,
  ariaHidden,
  className,
}: PlayingCardProps) {
  const { rank, suit, faceDown, special } = parseCard(code);
  const cssVars = { ["--pc-w" as string]: `${w}px` } as React.CSSProperties;

  if (special === "UNDER") {
    return (
      <div
        className={`${styles.pc} ${styles.inset} ${className ?? ""}`}
        style={cssVars}
        aria-hidden={ariaHidden}
      >
        <span className={styles.insetLabel}>under</span>
      </div>
    );
  }

  if (faceDown) {
    return (
      <div
        className={`${styles.pc} ${styles.back} ${dim ? styles.dim : ""} ${className ?? ""}`}
        style={cssVars}
        aria-hidden={ariaHidden}
      />
    );
  }

  const sym = suitSymbol(suit);
  const classNames = [
    styles.pc,
    isRedSuit(suit) ? styles.red : "",
    playable ? styles.playable : "",
    dim ? styles.dim : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classNames} style={cssVars} aria-hidden={ariaHidden}>
      <div className={styles.corner}>
        <div className={styles.rank}>{rank}</div>
        <div className={styles.suitSm}>{sym}</div>
      </div>
      <div
        className={`${styles.center} ${suit === "H" ? styles.centerHeart : suit === "D" ? styles.centerDiamond : ""}`}
      >
        {sym}
      </div>
      <div className={`${styles.corner} ${styles.cornerBr}`}>
        <div className={styles.rank}>{rank}</div>
        <div className={styles.suitSm}>{sym}</div>
      </div>
    </div>
  );
}
