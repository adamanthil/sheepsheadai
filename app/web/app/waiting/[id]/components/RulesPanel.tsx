import React from "react";
import { ds } from "../../../../lib/ds";
import styles from "./RulesPanel.module.css";

interface SegmentedProps {
  options: string[];
  value: number;
  onChange: (index: number) => void;
  disabled?: boolean;
}

function Segmented({ options, value, onChange, disabled }: SegmentedProps) {
  return (
    <div
      className={styles.segmented}
      style={{ gridTemplateColumns: `repeat(${options.length}, 1fr)` }}
    >
      {options.map((o, i) => (
        <button
          key={o}
          type="button"
          className={`${styles.segBtn} ${i === value ? styles.segActive : ""}`}
          onClick={() => onChange(i)}
          disabled={disabled}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

interface RulesPanelProps {
  partnerMode: number; // 1 = Called Ace, 0 = Jack of Diamonds
  scoringMode: number; // 1 = Double on the Bump, 0 = Symmetric
  onPartnerMode: (mode: number) => void;
  onScoringMode: (mode: number) => void;
  variant?: "panel" | "inline";
}

/**
 * House-rules controls bound to the table's partner/scoring modes. The
 * segmented option order is [secondary, primary] to match the prototype, so
 * the stored mode (1 = primary) maps to segmented index `mode === 1 ? 0 : 1`.
 */
export default function RulesPanel({
  partnerMode,
  scoringMode,
  onPartnerMode,
  onScoringMode,
  variant = "panel",
}: RulesPanelProps) {
  const inline = variant === "inline";
  const partnerIndex = partnerMode === 1 ? 0 : 1;
  const scoringIndex = scoringMode === 1 ? 0 : 1;

  return (
    <div className={inline ? styles.inline : `${ds.panel} ${styles.panel}`}>
      {!inline && (
        <div>
          <div className={ds.overline} style={{ marginBottom: 4 }}>
            House Rules · Host decides
          </div>
          <div className={styles.title}>Game Mode</div>
        </div>
      )}

      <div className={styles.group}>
        <div className={styles.groupLabel}>
          {inline ? "Partner" : "Partner selection"}
        </div>
        <Segmented
          options={["Called Ace", "Jack of Diamonds"]}
          value={partnerIndex}
          onChange={(i) => onPartnerMode(i === 0 ? 1 : 0)}
        />
        {!inline && (
          <div className={styles.desc}>
            {partnerMode === 1
              ? "The picker names a fail‑suit ace; whoever holds it is their secret partner until the card is played."
              : "The player holding the Jack of Diamonds is the picker’s partner."}
          </div>
        )}
      </div>

      <div className={styles.group}>
        <div className={styles.groupLabel}>Scoring</div>
        <Segmented
          options={["Double on Bump", "Symmetric"]}
          value={scoringIndex}
          onChange={(i) => onScoringMode(i === 0 ? 1 : 0)}
        />
        {!inline && (
          <div className={styles.desc}>
            {scoringMode === 1
              ? "The picking team loses double when they fail to take 60. Wins are scored normally."
              : "Wins and losses are scored symmetrically — no double on the bump."}
          </div>
        )}
      </div>
    </div>
  );
}
