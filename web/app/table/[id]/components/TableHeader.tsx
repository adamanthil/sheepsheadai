import React from "react";
import { Wordmark, MiniCardMark, ds } from "../../../../lib/ds";
import styles from "./TableHeader.module.css";

interface TableHeaderProps {
  roomName: string;
  rulesBadge: string | null;
  handNumber: number;
  phaseLabel: string;
  connected: boolean;
  isMobile: boolean;
  onLeave: () => void;
  onShowScores: () => void;
  onShowLog?: () => void;
}

export default function TableHeader({
  roomName,
  rulesBadge,
  handNumber,
  phaseLabel,
  connected,
  isMobile,
  onLeave,
  onShowScores,
  onShowLog,
}: TableHeaderProps) {
  if (isMobile) {
    return (
      <div className={styles.mob}>
        <div className={styles.mobLeft}>
          <MiniCardMark h={20} />
          <div className={styles.mobRoom}>{roomName}</div>
          <div className={styles.mobMeta}>
            H{handNumber} · {phaseLabel}
          </div>
        </div>
        <a className={styles.mobLeave} onClick={onLeave}>
          Leave
        </a>
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
          <a
            className={`${ds.link} ${styles.leave}`}
            style={{ fontSize: 12 }}
            onClick={onLeave}
          >
            Leave
          </a>
        </div>
      </div>
    </div>
  );
}
