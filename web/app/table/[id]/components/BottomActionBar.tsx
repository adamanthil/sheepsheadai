import React, { useMemo } from 'react';
import { CardText } from '../../../../lib/components';
import { nameForSeat } from '../utils/seatMath';
import styles from '../page.module.css';

interface BottomActionBarProps {
  yourSeat: number;
  actorSeat: number | null;
  table: any;
  isYourTurn: boolean;
  validActions: number[];
  actionLookup: Record<string, string>;
  showPrev: boolean;
  hasLastTrick: boolean;
  isHost: boolean;
  confirmClose: boolean;
  onTogglePrev: () => void;
  onShowScores: () => void;
  onCloseTable: () => void;
  onConfirmClose: (confirm: boolean) => void;
  onTakeAction: (actionId: number) => void;
}

export default function BottomActionBar({
  yourSeat,
  actorSeat,
  table,
  isYourTurn,
  validActions,
  actionLookup,
  showPrev,
  hasLastTrick,
  isHost,
  confirmClose,
  onTogglePrev,
  onShowScores,
  onCloseTable,
  onConfirmClose,
  onTakeAction,
}: BottomActionBarProps) {
  const yourName = nameForSeat(yourSeat, table);
  const actorName = nameForSeat(actorSeat, table);

  // Filter action buttons - show non-PLAY/BURY actions, plus "PLAY UNDER"
  const actionButtons = useMemo(() => {
    return validActions
      .filter((aid) => {
        const label = actionLookup[String(aid)];
        if (!label) return false;
        // Show if it's not PLAY/BURY or if it's specifically "PLAY UNDER"
        return (
          (!label.startsWith('PLAY') && !label.startsWith('BURY')) ||
          label === 'PLAY UNDER'
        );
      })
      .map((aid) => ({
        id: aid,
        label: actionLookup[String(aid)] || `Action ${aid}`,
      }));
  }, [validActions, actionLookup]);

  const hasActions = actionButtons.length > 0;

  return (
    <div className={styles.bottomBar}>
      <div className={styles.bottomBarInner}>
        <div className={styles.bottomTopRow}>
          {/* Player info - hidden on mobile via CSS */}
          <div className={`${styles.muted} ${styles.smallText} ${styles.playerInfo}`}>
            {yourName} {isYourTurn ? '· Your turn' : ''}
          </div>

          {/* Previous trick toggle */}
          {hasLastTrick && (
            <button
              className={styles.noFlex}
              onClick={onTogglePrev}
              aria-label={showPrev ? 'Hide previous trick' : 'Show previous trick'}
            >
              {showPrev ? 'Hide prev' : 'Show prev'}
            </button>
          )}

          {/* Action buttons or waiting message */}
          {hasActions ? (
            <div className={styles.actionButtons}>
              {actionButtons.map((action) => (
                <button
                  key={action.id}
                  onClick={() => onTakeAction(action.id)}
                  className={styles.actionButton}
                >
                  <CardText>{action.label}</CardText>
                </button>
              ))}
            </div>
          ) : (
            <div className={styles.dimmed}>
              Waiting for {actorName || `Seat ${actorSeat || ''}`}…
            </div>
          )}

          {/* Utility buttons */}
          <div className={styles.bottomButtonRow}>
            <button onClick={onShowScores}>Scores</button>
            {isHost && (
              <>
                {confirmClose ? (
                  <span className={styles.confirmCloseRow}>
                    <button className={styles.dangerButton} onClick={onCloseTable}>
                      Confirm close
                    </button>
                    <button onClick={() => onConfirmClose(false)}>Cancel</button>
                  </span>
                ) : (
                  <button onClick={() => onConfirmClose(true)}>Close table</button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

