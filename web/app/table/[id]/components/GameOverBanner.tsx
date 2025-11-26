import React from 'react';
import { nameForSeat } from '../utils/seatMath';
import styles from '../page.module.css';

interface FinalState {
  mode?: 'leaster' | 'normal';
  winner?: number;
  picker?: number;
  partner?: number;
  picker_score?: number;
  defender_score?: number;
  scores?: number[];
  points_taken?: number[];
}

interface GameOverBannerProps {
  final: FinalState;
  table: any;
  onRedeal: () => void;
  onShowScores: () => void;
}

export default function GameOverBanner({
  final,
  table,
  onRedeal,
  onShowScores,
}: GameOverBannerProps) {
  const seats = Array.from({ length: 5 }, (_, i) => i + 1);

  const renderScoresGrid = (scores: number[] | undefined, label?: string) => (
    <>
      {label && <div className={`${styles.muted} ${styles.mt8}`}>{label}</div>}
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

  if (final.mode === 'leaster') {
    return (
      <div className={styles.finalBanner}>
        <div className={styles.finalHeader}>Game Over · Leaster</div>
        <div className={styles.finalSubheading}>
          Winner: <strong>{nameForSeat(final.winner || 0, table)}</strong>
        </div>
        <div className={styles.finalSubheading}>Scores by player</div>
        {renderScoresGrid(final.scores)}
        {renderScoresGrid(final.points_taken, 'Points taken by player')}
        <div className={styles.finalButtons}>
          <button onClick={onRedeal}>Redeal</button>
          <button onClick={onShowScores}>Show scores</button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.finalBanner}>
      <div className={styles.finalHeader}>Game Over</div>
      <div className={styles.muted}>
        Picker: <strong>{nameForSeat(final.picker || 0, table)}</strong>
        {' · '}
        Partner: <strong>{nameForSeat(final.partner || 0, table)}</strong>
      </div>
      <div className={`${styles.muted} ${styles.mt4}`}>
        Picker score: <strong>{final.picker_score}</strong> · Defenders score:{' '}
        <strong>{final.defender_score}</strong>
      </div>
      {renderScoresGrid(final.scores, 'Scores by player')}
      <div className={styles.finalButtons}>
        <button onClick={onRedeal}>Redeal</button>
        <button onClick={onShowScores}>Show scores</button>
      </div>
    </div>
  );
}

