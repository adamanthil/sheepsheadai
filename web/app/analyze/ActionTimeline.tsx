import React from 'react';
import { AnalyzeActionDetail } from '../../lib/analyzeTypes';
import ActionRow from './ActionRow';
import Divider from './Divider';
import styles from './page.module.css';

interface ActionTimelineProps {
  trace: AnalyzeActionDetail[];
  picker?: string;
  partner?: string;
  shapingWeightPercent?: number; // 0-100 (applies to pick/partner/bury/play shaping)
  gamma?: number;
}

export default function ActionTimeline({ trace, picker, partner, shapingWeightPercent = 100, gamma = 0.95 }: ActionTimelineProps) {
  if (trace.length === 0) {
    return (
      <div className={styles.emptyState}>
        <h3>No actions to display</h3>
        <p>The simulation produced no action trace.</p>
      </div>
    );
  }

  const shapingWeight = Math.max(0, Math.min(1, shapingWeightPercent / 100));

  const displayedStepRewards = trace.map((action, i) => {
    const base = action.stepRewardBase;
    const head = action.stepRewardHeadShaping;
    if (typeof base === 'number' && typeof head === 'number') return base + (shapingWeight * head);
    if (typeof action.stepReward !== 'number') return action.stepReward;
    const h = typeof head === 'number' ? head : 0;
    return action.stepReward - h + (shapingWeight * h);
  });

  // Recompute discounted returns from the displayed step rewards using the same per-player reset logic.
  const idxsBySeat: Record<number, number[]> = {};
  trace.forEach((action, idx) => {
    idxsBySeat[action.seat] = idxsBySeat[action.seat] || [];
    idxsBySeat[action.seat].push(idx);
  });
  const discountedByIndex: Record<number, number> = {};
  Object.values(idxsBySeat).forEach((idxs) => {
    let ret = 0;
    for (let k = idxs.length - 1; k >= 0; k--) {
      const idx = idxs[k];
      if (k === idxs.length - 1) ret = 0;
      const step = displayedStepRewards[idx];
      const r = typeof step === 'number' ? step : 0;
      ret = r + gamma * ret;
      discountedByIndex[idx] = ret;
    }
  });

  const displayedTrace: AnalyzeActionDetail[] = trace.map((action, i) => ({
    ...action,
    stepReward: displayedStepRewards[i],
    discountedReturn: typeof discountedByIndex[i] === 'number' ? discountedByIndex[i] : action.discountedReturn,
  }));

  // Calculate min/max values for color normalization
  const valueEstimates = displayedTrace.map(action => action.valueEstimate);
  const minValue = Math.min(...valueEstimates);
  const maxValue = Math.max(...valueEstimates);

  // Calculate min/max for discounted returns (rewards), independent scale
  const rewardValues = displayedTrace
    .map(action => (typeof action.discountedReturn === 'number' ? action.discountedReturn : null))
    .filter((v): v is number => v !== null);
  const minReward = rewardValues.length > 0 ? Math.min(...rewardValues) : 0;
  const maxReward = rewardValues.length > 0 ? Math.max(...rewardValues) : 0;

  // Calculate min/max for step rewards (immediate), independent scale
  const stepRewardValues = displayedTrace
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
      {displayedTrace.map((action, index) => {
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
