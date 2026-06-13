import React from "react";
import { ds } from "../../../../lib/ds";
import { nameForSeat } from "../utils/seatMath";
import type { FinalState, TableView } from "../../../../lib/types";
import styles from "./GameOverBanner.module.css";

interface GameOverBannerProps {
  final: FinalState;
  table: TableView;
  onRedeal: () => void;
  onShowScores: () => void;
}

export default function GameOverBanner({
  final,
  table,
  onRedeal,
  onShowScores,
}: GameOverBannerProps) {
  const seats = [1, 2, 3, 4, 5];

  const scoresGrid = (scores: number[] | undefined, label?: string) => (
    <>
      {label && <div className={styles.gridLabel}>{label}</div>}
      <div className={styles.scoresGrid}>
        {seats.map((seat) => (
          <div key={seat} className={styles.scoreCell}>
            <div className={styles.scoreName}>{nameForSeat(seat, table)}</div>
            <div className={styles.scoreValue}>
              {(scores && scores[seat - 1]) || 0}
            </div>
          </div>
        ))}
      </div>
    </>
  );

  const isLeaster = final.mode === "leaster";

  return (
    <div className={styles.overlay}>
      <div className={styles.banner}>
        <div className={styles.header}>
          {isLeaster ? "Game Over · Leaster" : "Game Over"}
        </div>
        {isLeaster ? (
          <div className={styles.sub}>
            Winner: <strong>{nameForSeat(final.winner || 0, table)}</strong>
          </div>
        ) : (
          <>
            <div className={styles.sub}>
              Picker: <strong>{nameForSeat(final.picker || 0, table)}</strong> ·
              Partner: <strong>{nameForSeat(final.partner || 0, table)}</strong>
            </div>
            <div className={styles.sub}>
              Picker score: <strong>{final.picker_score}</strong> · Defenders:{" "}
              <strong>{final.defender_score}</strong>
            </div>
          </>
        )}
        {scoresGrid(final.scores, "Scores by player")}
        {isLeaster && scoresGrid(final.points_taken, "Points taken")}
        <div className={styles.buttons}>
          <button className={`${ds.btn} ${ds.btnAccent}`} onClick={onRedeal}>
            Redeal →
          </button>
          <button className={`${ds.btn} ${ds.btnGhost}`} onClick={onShowScores}>
            Show scores
          </button>
        </div>
      </div>
    </div>
  );
}
