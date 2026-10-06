import React, { useState } from "react";
import { Wordmark, MiniCardMark, ds } from "../../../../lib/ds";
import styles from "./TableHeader.module.css";

interface TableHeaderProps {
  roomName: string;
  rulesBadge: string | null;
  stakeBadge: string | null;
  handNumber: number;
  phaseLabel: string;
  connected: boolean;
  isMobile: boolean;
  onLeave: () => void;
  /** Seated with a hand in play: Leave offers to wait for the hand's end. */
  canLeaveAfterHand: boolean;
  leavingAfterHand: boolean;
  onLeaveAfterHand: (on: boolean) => void;
  onShowScores: () => void;
  onShowLog?: () => void;
}

export default function TableHeader({
  roomName,
  rulesBadge,
  stakeBadge,
  handNumber,
  phaseLabel,
  isMobile,
  onLeave,
  canLeaveAfterHand,
  leavingAfterHand,
  onLeaveAfterHand,
  onShowScores,
  onShowLog,
}: TableHeaderProps) {
  const [choosing, setChoosing] = useState(false);
  const linkClass = isMobile ? styles.mobLink : ds.link;
  const leaveClass = isMobile ? styles.mobLeave : `${ds.link} ${styles.leave}`;
  const fontSize = isMobile ? undefined : 12;

  // Mid-hand, Leave first asks when: now forfeits the hand to the AI, after
  // the hand waits for it to end. Once asked for, the wait can be undone.
  let leave: React.ReactNode;
  if (canLeaveAfterHand && leavingAfterHand) {
    leave = (
      <span className={styles.leaveRow}>
        <span className={styles.armed}>
          {isMobile ? "Leaving after hand" : "Leaving after this hand"}
        </span>
        <a
          className={linkClass}
          style={{ fontSize }}
          onClick={() => onLeaveAfterHand(false)}
        >
          Undo
        </a>
      </span>
    );
  } else if (canLeaveAfterHand && choosing) {
    leave = (
      <span className={styles.leaveRow}>
        <a
          className={linkClass}
          style={{ fontSize }}
          onClick={() => {
            setChoosing(false);
            onLeaveAfterHand(true);
          }}
        >
          After this hand
        </a>
        <a className={leaveClass} style={{ fontSize }} onClick={onLeave}>
          Leave now
        </a>
        <a
          className={linkClass}
          style={{ fontSize }}
          onClick={() => setChoosing(false)}
        >
          Cancel
        </a>
      </span>
    );
  } else {
    leave = (
      <a
        className={leaveClass}
        style={{ fontSize }}
        onClick={canLeaveAfterHand ? () => setChoosing(true) : onLeave}
      >
        Leave
      </a>
    );
  }
  const leaveExpanded = canLeaveAfterHand && choosing && !leavingAfterHand;

  if (isMobile) {
    return (
      <div className={styles.mob}>
        {/* The choices need the room; the table name comes back after. */}
        {leaveExpanded ? null : (
          <div className={styles.mobLeft}>
            <MiniCardMark h={20} />
            <div className={styles.mobRoom}>{roomName}</div>
            {/* Makes room for "Leaving after hand"; the hand still shows. */}
            {!(canLeaveAfterHand && leavingAfterHand) && (
              <div className={styles.mobMeta}>
                H{handNumber} · {phaseLabel}
              </div>
            )}
            {stakeBadge && (
              <span className={`${ds.badge} ${ds.badgeAccent} ${styles.stake}`}>
                {stakeBadge}
              </span>
            )}
          </div>
        )}
        {leave}
      </div>
    );
  }

  return (
    <div className={styles.desk}>
      <div className={styles.deskLeft}>
        <Wordmark size="sm" />
        <div className={styles.sep} />
        <div className={styles.room}>{roomName}</div>
        {rulesBadge && (
          <span className={`${ds.badge} ${ds.badgeQuiet}`}>{rulesBadge}</span>
        )}
        {stakeBadge && (
          <span className={`${ds.badge} ${ds.badgeAccent} ${styles.stake}`}>
            {stakeBadge}
          </span>
        )}
      </div>
      <div className={styles.deskRight}>
        <div className={styles.stat}>
          <span className={ds.overline}>Hand</span>
          <span className={styles.statNum}>{handNumber}</span>
        </div>
        <div className={styles.stat}>
          <span className={ds.overline}>Phase</span>
          <span className={styles.phase}>{phaseLabel}</span>
        </div>
        <div className={styles.links}>
          <a
            className={ds.link}
            style={{ fontSize: 12 }}
            onClick={onShowScores}
          >
            Scores
          </a>
          {onShowLog && (
            <a className={ds.link} style={{ fontSize: 12 }} onClick={onShowLog}>
              Chat
            </a>
          )}
          {leave}
        </div>
      </div>
    </div>
  );
}
