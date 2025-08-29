import React from 'react';
import styles from './page.module.css';

interface DividerProps {
  type: 'phase' | 'trick';
  fromPhase?: string;
  toPhase?: string;
  trickNumber?: number;
}

export default function Divider({ type, fromPhase, toPhase, trickNumber }: DividerProps) {
  if (type === 'phase' && fromPhase && toPhase) {
    return (
      <div className={styles.phaseDivider}>
        <div className={styles.phaseDividerLine}></div>
        <div className={styles.phaseDividerLabel}>
          {fromPhase} → {toPhase}
        </div>
        <div className={styles.phaseDividerLine}></div>
      </div>
    );
  }

  if (type === 'trick' && trickNumber) {
    return (
      <div className={styles.trickDivider}>
        <div className={styles.trickDividerLine}></div>
        <div className={styles.trickDividerLabel}>
          Trick {trickNumber}
        </div>
        <div className={styles.trickDividerLine}></div>
      </div>
    );
  }

  return null;
}
