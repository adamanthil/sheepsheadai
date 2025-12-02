import React, { useState } from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionDetails from './ActionDetails';
import ActionInsights from './ActionInsights';
import ActionStateVector from './ActionStateVector';
import { CardText } from '../../lib/components';
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

  const formatSigned = (value: number, digits = 3) => {
    return value >= 0 ? `+${value.toFixed(digits)}` : value.toFixed(digits);
  };

  function MetricChip({
    label,
    value,
    colorStyle,
    tooltip,
    suffix = ''
  }: {
    label: React.ReactNode;
    value: string;
    colorStyle?: React.CSSProperties;
    tooltip?: string;
    suffix?: string;
  }) {
    return (
      <div className={styles.metricChip} title={tooltip} style={colorStyle}>
        <span className={styles.metricLabel}>{label}</span>
        <span className={styles.metricValue}>{value}{suffix}</span>
      </div>
    );
  }

  function MetricsBar({
    value,
    valueHue,
    discounted,
    discountedHue,
    step,
    stepHue,
    winPct,
    expectedFinal,
  }: {
    value: number;
    valueHue: number;
    discounted?: number;
    discountedHue?: number;
    step?: number;
    stepHue?: number;
    winPct?: number;
    expectedFinal?: number;
  }) {
    return (
      <div className={styles.metricsBar}>
        <MetricChip
          label={<>V</>}
          value={formatSigned(value)}
          colorStyle={getHueStyle(valueHue)}
          tooltip="Value estimate (critic)"
        />

        {typeof discounted === 'number' && (
          <MetricChip
            label={<>G<sub>t</sub></>}
            value={formatSigned(discounted)}
            colorStyle={getHueStyle(typeof discountedHue === 'number' ? discountedHue : 0.5)}
            tooltip="Discounted return"
          />
        )}

        {typeof step === 'number' && (
          <MetricChip
            label={<>r</>}
            value={formatSigned(step)}
            colorStyle={getHueStyle(typeof stepHue === 'number' ? stepHue : 0.5)}
            tooltip="Immediate step reward"
          />
        )}

        {typeof winPct === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Win probability">
            <span className={styles.metricLabel}>Win</span>
            <span className={styles.metricValue}>{`${Math.round(winPct)}%`}</span>
          </div>
        )}

        {typeof expectedFinal === 'number' && (
          <div className={`${styles.metricChip} ${styles.metricChipNeutral}`} title="Expected final return (undiscounted)">
            <span className={styles.metricLabel}>E[Ret]</span>
            <span className={styles.metricValue}>{expectedFinal.toFixed(2)}</span>
          </div>
        )}

      </div>
    );
  }

  return (
    <div className={styles.actionRow}>
      <div
        className={styles.actionSummary}
        onClick={handleClick}
      >
        <div className={styles.actionBasic}>
          <div className={styles.stepIndex}>
            {action.stepIndex + 1}
          </div>

          <div className={getSeatBadgeClass()}>
            {action.seatName}
          </div>

          <div className={styles.actionInfo}>
            <div className={styles.actionText}>
              <CardText>{action.action}</CardText>
            </div>
            <div className={styles.phaseText}>
              {action.phase} Head
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <MetricsBar
            value={action.valueEstimate}
            valueHue={normalizedValue}
            discounted={typeof action.discountedReturn === 'number' ? action.discountedReturn : undefined}
            discountedHue={normalizedReward}
            step={typeof action.stepReward === 'number' ? action.stepReward : undefined}
            stepHue={typeof normalizedStepReward === 'number' ? normalizedStepReward : undefined}
            winPct={typeof action.winProb === 'number' ? (action.winProb * 100) : undefined}
            expectedFinal={typeof action.expectedFinalReturn === 'number' ? action.expectedFinalReturn : undefined}
          />

          <div className={`${styles.expandIcon} ${expanded ? styles.expanded : ''}`}>▼</div>
        </div>
      </div>

      {expanded && (
        <>
          <ActionDetails action={action} />
          <ActionInsights action={action} />
          <ActionStateVector action={action} />
        </>
      )}
    </div>
  );
}
