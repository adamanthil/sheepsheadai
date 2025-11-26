import React from 'react';
import { PlayingCard } from '../../../../lib/components';
import ui from '../../../styles/ui.module.css';
import styles from '../page.module.css';

type PlayerStatus = 'PASS' | 'PICK' | 'PICKER' | 'PENDING' | 'PARTNER' | null;

interface TrickCardProps {
  card: string;
  relSeat: number;
  name: string;
  isAi: boolean;
  highlight?: boolean;
  playerStatus?: PlayerStatus;
  centerSize: { w: number; h: number };
}

export default function TrickCard({
  card,
  relSeat,
  name,
  isAi,
  highlight = false,
  playerStatus,
  centerSize,
}: TrickCardProps) {
  // Map relSeat to CSS class for positioning
  const spotClass = `spotR${relSeat}` as 'spotR0' | 'spotR1' | 'spotR2' | 'spotR3' | 'spotR4';

  // Render pick status badge (not shown for user's position r=0)
  const renderStatusBadge = () => {
    if (!playerStatus || relSeat === 0) return null;

    const badgeClass =
      playerStatus === 'PICK' || playerStatus === 'PICKER'
        ? styles.badgePick
        : playerStatus === 'PASS'
          ? styles.badgePass
          : playerStatus === 'PENDING'
            ? styles.badgePending
            : styles.badgePartner;

    return <span className={`${styles.badge} ${badgeClass}`}>{playerStatus}</span>;
  };

  const renderMeta = () => {
    const nameContent = (
      <span className={ui.nameInline}>
        <span>{name}</span>
        {isAi && <span className={ui.aiTag}>AI</span>}
      </span>
    );

    const badgeContent = renderStatusBadge();
    const badgeWrapper =
      badgeContent && <div className={styles.statusContainer}>{badgeContent}</div>;

    if (relSeat === 2) {
      return (
        <div className={styles.nameLeftOfCard}>
          {nameContent}
          {badgeWrapper}
        </div>
      );
    }

    if (relSeat === 3) {
      return (
        <div className={styles.nameRightOfCard}>
          {nameContent}
          {badgeWrapper}
        </div>
      );
    }

    const belowClass = relSeat === 0 ? styles.metaBelowSelf : styles.metaBelow;

    return (
      <div className={belowClass}>
        <div className={styles.nameBelow}>{nameContent}</div>
        {badgeWrapper}
      </div>
    );
  };

  return (
    <div className={`${styles.trickSpot} ${styles[spotClass]}`}>
      <PlayingCard
        label={card || '__'}
        width={centerSize.w}
        height={centerSize.h}
        highlight={highlight}
      />
      {renderMeta()}
    </div>
  );
}

