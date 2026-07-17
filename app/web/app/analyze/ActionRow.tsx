import React, { useState } from "react";
import { AnalyzeActionDetail } from "../../lib/analyzeTypes";
import ActionDetails from "./ActionDetails";
import ActionInsights from "./ActionInsights";
import { CardText } from "../../lib/ds";
import styles from "./ActionRow.module.css";

interface ActionRowProps {
  action: AnalyzeActionDetail;
  picker?: string;
  partner?: string;
  normalizedValue: number; // 0-1 scale for color gradient (value)
  normalizedReward: number; // 0-1 scale for color gradient (reward)
  normalizedStepReward?: number; // 0-1 scale for color gradient (immediate step reward)
}

const formatSigned = (value: number, digits = 3) =>
  value >= 0 ? `+${value.toFixed(digits)}` : value.toFixed(digits);

/** Map a 0-1 normalized metric onto the red→green chip gradient. */
const hueStyle = (normalized: number) =>
  ({ "--value-hue": `${normalized * 120}deg` }) as React.CSSProperties;

function MetricChip({
  label,
  value,
  hue,
  tooltip,
}: {
  label: React.ReactNode;
  value: string;
  /** 0-1 position on the red→green gradient; omit for a neutral chip. */
  hue?: number;
  tooltip?: string;
}) {
  const className =
    typeof hue === "number"
      ? styles.metricChip
      : `${styles.metricChip} ${styles.metricChipNeutral}`;
  return (
    <div
      className={className}
      title={tooltip}
      style={typeof hue === "number" ? hueStyle(hue) : undefined}
    >
      <span className={styles.metricLabel}>{label}</span>
      <span className={styles.metricValue}>{value}</span>
    </div>
  );
}

function MetricsBar({
  action,
  normalizedValue,
  normalizedReward,
  normalizedStepReward,
}: {
  action: AnalyzeActionDetail;
  normalizedValue: number;
  normalizedReward: number;
  normalizedStepReward?: number;
}) {
  const {
    valueEstimate,
    oracleValue,
    discountedReturn,
    stepReward,
    winProb,
    expectedFinalReturn,
  } = action;
  return (
    <div className={styles.metricsBar}>
      <MetricChip
        label="V"
        value={formatSigned(valueEstimate)}
        hue={normalizedValue}
        tooltip="Value estimate: the critic's prediction, from this seat's point of view, of the reward it will end the game with"
      />

      {typeof oracleValue === "number" && (
        <MetricChip
          label="V*"
          value={formatSigned(oracleValue)}
          tooltip={`Oracle value: the same prediction from a privileged critic that sees every hidden card (Δ ${formatSigned(oracleValue - valueEstimate)} vs V — the gap is what hidden information is worth here)`}
        />
      )}

      {typeof discountedReturn === "number" && (
        <MetricChip
          label={
            <>
              G<sub>t</sub>
            </>
          }
          value={formatSigned(discountedReturn)}
          hue={normalizedReward}
          tooltip="Discounted return: the reward actually accumulated from this point to the end of the game, with later rewards weighted down by gamma per step"
        />
      )}

      {typeof stepReward === "number" && (
        <MetricChip
          label="r"
          value={formatSigned(stepReward)}
          hue={typeof normalizedStepReward === "number" ? normalizedStepReward : 0.5}
          tooltip="Immediate step reward"
        />
      )}

      {typeof winProb === "number" && (
        <MetricChip
          label="Win"
          value={`${Math.round(winProb * 100)}%`}
          tooltip="Win probability"
        />
      )}

      {typeof expectedFinalReturn === "number" && (
        <MetricChip
          label="E[Ret]"
          value={expectedFinalReturn.toFixed(2)}
          tooltip="The model's prediction of this seat's final game score (undiscounted, from an auxiliary head)"
        />
      )}
    </div>
  );
}

export default function ActionRow({
  action,
  picker,
  partner,
  normalizedValue,
  normalizedReward,
  normalizedStepReward,
}: ActionRowProps) {
  const [expanded, setExpanded] = useState(false);

  const role =
    action.seatName === picker
      ? styles.seatBadgePicker
      : action.seatName === partner
        ? styles.seatBadgePartner
        : styles.seatBadgeOther;

  return (
    <div className={styles.actionRow}>
      <div
        className={styles.actionSummary}
        onClick={() => setExpanded(!expanded)}
      >
        <div className={styles.actionBasic}>
          <div className={styles.stepIndex}>{action.stepIndex + 1}</div>

          <div className={`${styles.seatBadge} ${role}`}>{action.seatName}</div>

          <div className={styles.actionInfo}>
            <div className={styles.actionText}>
              <CardText>{action.action}</CardText>
            </div>
            <div className={styles.phaseText}>{action.phase} Head</div>
          </div>
        </div>

        <div className={styles.summaryRight}>
          <MetricsBar
            action={action}
            normalizedValue={normalizedValue}
            normalizedReward={normalizedReward}
            normalizedStepReward={normalizedStepReward}
          />

          <div
            className={`${styles.expandIcon} ${expanded ? styles.expanded : ""}`}
          >
            ▼
          </div>
        </div>
      </div>

      {expanded && (
        <>
          <ActionDetails action={action} />
          <ActionInsights action={action} />
        </>
      )}
    </div>
  );
}
