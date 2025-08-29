import React from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionRow from './ActionRow';
import Divider from './Divider';
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

  // Calculate min/max for discounted returns (rewards), independent scale
  const rewardValues = trace
    .map(action => (typeof action.discountedReturn === 'number' ? action.discountedReturn : null))
    .filter((v): v is number => v !== null);
  const minReward = rewardValues.length > 0 ? Math.min(...rewardValues) : 0;
  const maxReward = rewardValues.length > 0 ? Math.max(...rewardValues) : 0;

  // Calculate min/max for step rewards (immediate), independent scale
  const stepRewardValues = trace
    .map(action => (typeof action.stepReward === 'number' ? action.stepReward : null))
    .filter((v): v is number => v !== null);
  const minStepReward = stepRewardValues.length > 0 ? Math.min(...stepRewardValues) : 0;
  const maxStepReward = stepRewardValues.length > 0 ? Math.max(...stepRewardValues) : 0;

  const normalize = (value: number | undefined, min: number, max: number) => {
    if (typeof value !== 'number') return 0.5;
    if (max === min) return 0.5; // Handle edge case where all values are the same
    return (value - min) / (max - min);
  };

    // Sheepshead always has exactly 5 players
  const PLAYERS_PER_TRICK = 5;

  // Helper function to check if we need a divider before this action
  const needsDivider = (currentIndex: number) => {
    if (currentIndex === 0) return false;

    const currentAction = trace[currentIndex];
    const prevAction = trace[currentIndex - 1];

    // Phase change divider
    if (currentAction.phase !== prevAction.phase) {
      return { type: 'phase', fromPhase: prevAction.phase, toPhase: currentAction.phase };
    }

    // Trick divider (only in play phase)
    if (currentAction.phase === 'play' && prevAction.phase === 'play') {
      // Count how many play actions have occurred before this one
      const playActionsBefore = trace.slice(0, currentIndex).filter(action => action.phase === 'play').length;

      // If we have a multiple of 5 actions, this starts a new trick
      if (playActionsBefore > 0 && playActionsBefore % PLAYERS_PER_TRICK === 0) {
        const trickNumber = Math.floor(playActionsBefore / PLAYERS_PER_TRICK) + 1;
        return { type: 'trick', trickNumber };
      }
    }

    return false;
  };

  return (
    <div>
      {trace.map((action, index) => {
        const dividerInfo = needsDivider(index);
        return (
          <React.Fragment key={index}>
            {dividerInfo && (
              <Divider
                type={dividerInfo.type as 'phase' | 'trick'}
                fromPhase={dividerInfo.fromPhase}
                toPhase={dividerInfo.toPhase}
                trickNumber={dividerInfo.trickNumber}
              />
            )}
            <ActionRow
              action={action}
              picker={picker}
              partner={partner}
              normalizedValue={normalize(action.valueEstimate, minValue, maxValue)}
              normalizedReward={normalize(action.discountedReturn, minReward, maxReward)}
              normalizedStepReward={normalize(action.stepReward, minStepReward, maxStepReward)}
            />
          </React.Fragment>
        );
      })}
    </div>
  );
}
