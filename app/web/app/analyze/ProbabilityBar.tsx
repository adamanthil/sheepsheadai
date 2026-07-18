import React from "react";
import { AnalyzeProbability } from "../../lib/analyzeTypes";
import { CardText } from "../../lib/ds";
import styles from "./ProbabilityBar.module.css";

interface ProbabilityBarProps {
  probability: AnalyzeProbability;
  maxProb: number;
}

export default function ProbabilityBar({
  probability,
  maxProb,
}: ProbabilityBarProps) {
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
        <span
          className={styles.probabilityLogit}
          title={`Logit ${probability.logit.toFixed(4)}: the model's raw score for this action before scores are converted to probabilities (softmax). A gap of 1 between two logits ≈ one action being ~2.7x as likely as the other.`}
        >
          ℓ {probability.logit.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
