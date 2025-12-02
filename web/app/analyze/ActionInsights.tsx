import React, { useMemo } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import styles from './page.module.css';

interface ActionInsightsProps {
  action: AnalyzeActionDetail;
}

export default function ActionInsights({ action }: ActionInsightsProps) {
  const { secretPartnerProb, pointEstimates } = action;

  const sortedPointEstimates = useMemo(() => {
    if (!pointEstimates) return [];
    return [...pointEstimates].sort((a, b) => a.seat - b.seat);
  }, [pointEstimates]);

  const showSecret = typeof secretPartnerProb === 'number';
  const showPoints = sortedPointEstimates.length > 0;

  if (!showSecret && !showPoints) {
    return null;
  }

  const secretPct = showSecret
    ? Math.round(Math.max(0, Math.min(1, secretPartnerProb as number)) * 100)
    : 0;

  return (
    <div className={styles.actionInsightsSection}>
      <div className={styles.insightsGrid}>
        {showPoints && (
          <div className={`${styles.insightCard} ${styles.pointsInsight}`}>
            <div className={styles.insightTitle}>Point Predictions</div>
            <div className={styles.pointsList}>
              {sortedPointEstimates.map((pt) => {
                const widthPct = Math.min(100, (pt.points / 120) * 100);
                return (
                  <div key={pt.seat} className={styles.pointRow}>
                    <div className={styles.pointSeat}>
                      Seat {pt.seat}
                      <span>{pt.seatName}</span>
                    </div>
                    <div className={styles.pointBarWrapper}>
                      <div className={styles.pointBarTrack}>
                        <div className={styles.pointBarFill} style={{ width: `${widthPct}%` }} />
                      </div>
                      <div className={styles.pointValue}>{pt.points.toFixed(1)} pts</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {showSecret && (
          <div className={styles.insightCard}>
            <div className={styles.insightTitle}>Secret Partner</div>
            <div className={styles.secretMini}>
              <div className={styles.secretMiniValue}>{secretPct}%</div>
              <div className={styles.secretMiniLabel}>confidence</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

