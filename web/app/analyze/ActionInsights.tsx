import React, { useMemo } from 'react';
import { AnalyzeActionDetail, AnalyzePointEstimate } from '../../lib/analyzeTypes';
import styles from './page.module.css';

interface ActionInsightsProps {
  action: AnalyzeActionDetail;
}

export default function ActionInsights({ action }: ActionInsightsProps) {
  const {
    secretPartnerProb,
    pointEstimates,
    pointActuals,
    highestTrumpActual,
    highestTrumpTopk,
  } = action;

  const sortedPointEstimates = useMemo(() => {
    if (!pointEstimates) return [];
    return [...pointEstimates].sort((a, b) => a.seat - b.seat);
  }, [pointEstimates]);

  const actualPointsBySeat = useMemo(() => {
    const map = new Map<number, AnalyzePointEstimate>();
    (pointActuals || []).forEach((pt) => {
      map.set(pt.seat, pt);
    });
    return map;
  }, [pointActuals]);

  const showSecret = typeof secretPartnerProb === 'number';
  const showPoints = sortedPointEstimates.length > 0;
  const trumpDistribution = highestTrumpTopk ?? [];
  const showTrump = trumpDistribution.length > 0 || typeof highestTrumpActual === 'string';

  const formatTrumpLabel = (card?: string | null) => {
    if (!card) return 'Unknown';
    if (card === 'ALL_SEEN') return 'All Seen';
    if (card.trim().length === 0) return 'Unknown';
    return card;
  };

  if (!showSecret && !showPoints && !showTrump) {
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
            <div className={styles.insightTitle}>Points: Prediction vs Actual</div>
                <div className={styles.legendContainer}>
                    <div className={styles.pointsLegend}>
                    <div className={styles.legendItem}>
                        <span className={styles.predictionSwatch} />
                        Prediction
                    </div>
                    <div className={styles.legendItem}>
                        <span className={styles.actualSwatch} />
                        Actual
                    </div>
                    </div>
                    <div className={styles.valueLegend}>Prediction <br/> Actual</div>
                </div>
            <div className={styles.pointsList}>
              {sortedPointEstimates.map((pt) => {
                const widthPct = Math.min(100, (pt.points / 120) * 100);
                const actualPoint = actualPointsBySeat.get(pt.seat);
                const actualValue = typeof actualPoint?.points === 'number' ? actualPoint.points : null;
                const actualPct = actualValue !== null ? Math.min(100, (actualValue / 120) * 100) : null;
                const actualRounded = actualValue !== null ? Math.round(actualValue) : null;
                const diff = actualValue !== null ? actualValue - pt.points : null;
                const predictionClass =
                  diff === null
                    ? styles.pointValueNeutral
                    : diff > 0
                      ? styles.pointValuePositive
                      : diff < 0
                        ? styles.pointValueNegative
                        : styles.pointValueNeutral;
                return (
                  <div key={pt.seat} className={styles.pointRow}>
                    <div className={styles.pointSeat}>
                      Seat {pt.seat}
                      <span>{pt.seatName}</span>
                    </div>
                    <div className={styles.pointBarWrapper}>
                      <div className={styles.pointBarTrack}>
                        <div className={styles.pointBarFill} style={{ width: `${widthPct}%` }} />
                        {actualPct !== null && (
                          <div
                            className={styles.pointActualIndicator}
                            style={{ left: `${actualPct}%` }}
                            title={`Actual ${actualValue?.toFixed(1)} pts`}
                          />
                        )}
                      </div>
                      <div className={styles.pointValueStack}>
                        <div className={`${styles.pointValue} ${predictionClass}`}>{pt.points.toFixed(1)} pts</div>
                        {actualRounded !== null && (
                          <div className={styles.pointActualValue}>{actualRounded} pts</div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {showTrump && (
          <div className={`${styles.insightCard}`}>
            <div className={styles.insightTitle}>Highest Remaining Trump (Top 5 Predictions)</div>
            <div className={styles.trumpList}>
              {trumpDistribution.length > 0 ? (
                trumpDistribution.map((entry) => (
                  <div key={`${entry.card}-${entry.probability}`} className={styles.trumpEntry}>
                    <div className={styles.trumpCardLabel}>{formatTrumpLabel(entry.card)}</div>
                    <div className={styles.trumpProbBar}>
                      <div
                        className={styles.trumpProbFill}
                        style={{ width: `${Math.min(100, entry.probability * 100)}%` }}
                      />
                    </div>
                    <div className={styles.trumpProbValue}>{(entry.probability * 100).toFixed(1)}%</div>
                  </div>
                ))
              ) : (
                <div className={styles.trumpEmpty}>No prediction available</div>
              )}
            </div>
            {highestTrumpActual && (
              <div className={styles.trumpActualRow}>
                Actual highest: <span>{formatTrumpLabel(highestTrumpActual)}</span>
              </div>
            )}
          </div>
        )}
      </div>
      {showSecret && (
        <div className={styles.secretRow}>
          <div className={`${styles.insightCard} ${styles.secretRowCard}`}>
            <div className={styles.insightTitle}>Secret Partner</div>
            <div className={styles.secretMini}>
              <div className={styles.secretMiniValue}>{secretPct}%</div>
              <div className={styles.secretMiniLabel}>confidence</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

