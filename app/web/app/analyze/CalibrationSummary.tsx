import React from "react";
import { AnalyzeCalibrationSummary } from "../../lib/analyzeTypes";
import styles from "./CalibrationSummary.module.css";

interface CalibrationSummaryProps {
  calibration: AnalyzeCalibrationSummary;
}

const SPARK_W = 56;
const SPARK_H = 20;
const SPARK_PAD = 4;

function Trajectory({ first, last }: { first: number; last: number }) {
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const y = (p: number) =>
    SPARK_H - SPARK_PAD - clamp(p) * (SPARK_H - 2 * SPARK_PAD);
  const x0 = SPARK_PAD;
  const x1 = SPARK_W - SPARK_PAD;

  return (
    <svg
      className={styles.trajectory}
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      role="img"
      aria-label={`Win probability from ${(first * 100).toFixed(0)}% to ${(last * 100).toFixed(0)}%`}
    >
      <title>{`First decision: ${(first * 100).toFixed(0)}% → last decision: ${(last * 100).toFixed(0)}%`}</title>
      <line
        x1={x0}
        y1={SPARK_H - SPARK_PAD}
        x2={x1}
        y2={SPARK_H - SPARK_PAD}
        className={styles.trajectoryBaseline}
      />
      <line
        x1={x0}
        y1={y(first)}
        x2={x1}
        y2={y(last)}
        className={styles.trajectoryLine}
      />
      <circle cx={x0} cy={y(first)} r={2.5} className={styles.trajectoryDot} />
      <circle
        cx={x1}
        cy={y(last)}
        r={2.5}
        className={`${styles.trajectoryDot} ${styles.trajectoryDotEnd}`}
      />
    </svg>
  );
}

export default function CalibrationSummary({
  calibration,
}: CalibrationSummaryProps) {
  const { overallBrier, overallPointsMae, seats, trumpMaskAccuracy, trumpMaskCount } =
    calibration;
  const sortedSeats = [...seats].sort((a, b) => a.seat - b.seat);

  return (
    <div className={styles.calibrationSummary}>
      <div className={styles.header}>
        <h3 className={styles.title}>Prediction Calibration</h3>
        <p className={styles.subhead}>
          How well the model&rsquo;s own predictions (win probability, points,
          trump tracking) matched what actually happened.
        </p>
      </div>

      <div className={styles.tableWrap}>
        <div className={`${styles.row} ${styles.headRow}`}>
          <div className={styles.cellSeat}>Seat</div>
          <div className={styles.cellResult}>Result</div>
          <div className={styles.cellCount}>Decisions</div>
          <div className={styles.cellTrend}>Win % (first &rarr; last)</div>
          <div className={styles.cellNum}>Mean Win %</div>
          <div
            className={styles.cellNum}
            title="Brier score: 0 = perfect, 0.25 = uninformed"
          >
            Brier
          </div>
        </div>

        {sortedSeats.map((s) => (
          <div key={s.seat} className={styles.row}>
            <div className={styles.cellSeat}>{s.seatName}</div>
            <div className={styles.cellResult}>
              <span
                className={`${styles.resultMarker} ${
                  s.won ? styles.resultWon : styles.resultLost
                }`}
              >
                {s.won ? "Won" : "Lost"}
              </span>
            </div>
            <div className={styles.cellCount}>{s.decisionCount}</div>
            <div className={styles.cellTrend}>
              <Trajectory first={s.firstWinProb} last={s.lastWinProb} />
              <span className={styles.trendText}>
                {(s.firstWinProb * 100).toFixed(0)}% &rarr;{" "}
                {(s.lastWinProb * 100).toFixed(0)}%
              </span>
            </div>
            <div className={styles.cellNum}>
              {(s.meanWinProb * 100).toFixed(1)}%
            </div>
            <div className={styles.cellNum}>{s.brierScore.toFixed(3)}</div>
          </div>
        ))}
      </div>

      <div className={styles.footer}>
        <div
          className={styles.footerStat}
          title="Brier score: 0 = perfect, 0.25 = uninformed"
        >
          <span className={styles.footerLabel}>Overall Brier</span>
          <span className={styles.footerValue}>{overallBrier.toFixed(3)}</span>
        </div>
        <div className={styles.footerStat}>
          <span className={styles.footerLabel}>
            Point prediction MAE (pts)
          </span>
          <span className={styles.footerValue}>
            {overallPointsMae.toFixed(1)}
          </span>
        </div>
        <div className={styles.footerStat}>
          <span className={styles.footerLabel}>Trump-mask accuracy</span>
          <span className={styles.footerValue}>
            {typeof trumpMaskAccuracy === "number"
              ? `${(trumpMaskAccuracy * 100).toFixed(1)}%`
              : "—"}
            <span className={styles.footerSub}> (n={trumpMaskCount})</span>
          </span>
        </div>
      </div>
    </div>
  );
}
