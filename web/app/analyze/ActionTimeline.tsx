import React from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionRow from './ActionRow';
import styles from './page.module.css';

interface ActionTimelineProps {
  trace: AnalyzeActionDetail[];
  picker?: string;
  partner?: string;
}

export default function ActionTimeline({ trace, picker, partner }: ActionTimelineProps) {
  if (trace.length === 0) {
    return (
      <div className={styles.emptyState}>
        <h3>No actions to display</h3>
        <p>The simulation produced no action trace.</p>
      </div>
    );
  }

  // Calculate min/max values for color normalization
  const valueEstimates = trace.map(action => action.valueEstimate);
  const minValue = Math.min(...valueEstimates);
  const maxValue = Math.max(...valueEstimates);

  const normalizeValue = (value: number) => {
    if (maxValue === minValue) return 0.5; // Handle edge case where all values are the same
    return (value - minValue) / (maxValue - minValue);
  };

  return (
    <div>
      {trace.map((action, index) => (
        <ActionRow
          key={index}
          action={action}
          picker={picker}
          partner={partner}
          normalizedValue={normalizeValue(action.valueEstimate)}
        />
      ))}
    </div>
  );
}
