import React from 'react';
import { AnalyzeProbability } from '../../lib/analyzeTypes';
import { CardText } from '../../lib/components';
import styles from './page.module.css';

interface ProbabilityBarProps {
  probability: AnalyzeProbability;
  maxProb: number;
}

export default function ProbabilityBar({ probability, maxProb }: ProbabilityBarProps) {
  const percentage = maxProb > 0 ? (probability.prob / maxProb) * 100 : 0;

  return (
    <div className={styles.probabilityItem}>
      <div className={styles.probabilityAction}>
        <CardText>{probability.action}</CardText>
      </div>
      <div className={styles.probabilityBarContainer}>
        <div
          className={styles.probabilityBar}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className={styles.probabilityValue}>
        {(probability.prob * 100).toFixed(1)}%
      </div>
    </div>
  );
}
