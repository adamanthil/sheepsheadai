import React, { useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionDetails from './ActionDetails';
import styles from './page.module.css';

interface ActionRowProps {
  action: AnalyzeActionDetail;
  picker?: string;
  partner?: string;
  normalizedValue: number; // 0-1 scale for color gradient (value)
  normalizedReward: number; // 0-1 scale for color gradient (reward)
  normalizedStepReward?: number; // 0-1 scale for color gradient (immediate step reward)
}

export default function ActionRow({ action, picker, partner, normalizedValue, normalizedReward, normalizedStepReward }: ActionRowProps) {
  const [expanded, setExpanded] = useState(false);

  const handleClick = () => {
    setExpanded(!expanded);
  };

  const formatValue = (value: number) => {
    return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
  };

  const getPlayerRole = () => {
    if (action.seatName === picker) return 'picker';
    if (action.seatName === partner) return 'partner';
    return 'other';
  };

  const getSeatBadgeClass = () => {
    const role = getPlayerRole();
    switch (role) {
      case 'picker':
        return `${styles.seatBadge} ${styles.seatBadgePicker}`;
      case 'partner':
        return `${styles.seatBadge} ${styles.seatBadgePartner}`;
      default:
        return `${styles.seatBadge} ${styles.seatBadgeOther}`;
    }
  };

  const getHueStyle = (normalized: number) => {
    const hue = normalized * 120; // 0 (red) .. 120 (green)
    return { '--value-hue': `${hue}deg` } as React.CSSProperties;
  };

  return (
    <div className={styles.actionRow}>
      <div
        className={styles.actionSummary}
        onClick={handleClick}
      >
        <div className={styles.actionBasic}>
          <div className={styles.stepIndex}>
            {action.stepIndex}
          </div>

          <div className={getSeatBadgeClass()}>
            {action.seatName}
          </div>

          <div className={styles.actionInfo}>
            <div className={styles.actionText}>
              {action.action}
            </div>
            <div className={styles.phaseText}>
              {action.phase} Head
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div
            className={styles.valueEstimate}
            style={getHueStyle(normalizedValue)}
          >
            Value Estimate: {formatValue(action.valueEstimate)}
          </div>

          {typeof action.discountedReturn === 'number' && (
            <div
              className={styles.valueEstimate}
              style={getHueStyle(normalizedReward)}
            >
              Return: {formatValue(action.discountedReturn)}
            </div>
          )}

          {typeof action.stepReward === 'number' && (
            <div
              className={styles.valueEstimate}
              style={getHueStyle(typeof normalizedStepReward === 'number' ? normalizedStepReward : 0.5)}
            >
              Step Reward: {formatValue(action.stepReward)}
            </div>
          )}

          <div className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}>
            ▼
          </div>
        </div>
      </div>

      {expanded && (
        <ActionDetails action={action} />
      )}
    </div>
  );
}
