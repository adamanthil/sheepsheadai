import React from 'react';
import { PlayingCard } from '../../../../lib/components';
import ui from '../../../styles/ui.module.css';
import styles from '../page.module.css';

type PickStatus = 'PASS' | 'PICK' | 'PENDING' | null;

interface TrickCardProps {
  card: string;
  relSeat: number;
  name: string;
  isAi: boolean;
  highlight?: boolean;
  pickStatus?: PickStatus;
  centerSize: { w: number; h: number };
}

export default function TrickCard({
  card,
  relSeat,
  name,
  isAi,
  highlight = false,
  pickStatus,
  centerSize,
}: TrickCardProps) {
  // Map relSeat to CSS class for positioning
  const spotClass = `spotR${relSeat}` as 'spotR0' | 'spotR1' | 'spotR2' | 'spotR3' | 'spotR4';

  // Determine name label position based on relative seat
  const renderNameLabel = () => {
    const nameContent = (
      <span className={ui.nameInline}>
        <span>{name}</span>
        {isAi && <span className={ui.aiTag}>AI</span>}
      </span>
    );

    if (relSeat === 2) {
      return <div className={styles.nameLeftOfCard}>{nameContent}</div>;
    } else if (relSeat === 3) {
      return <div className={styles.nameRightOfCard}>{nameContent}</div>;
    } else {
      return <div className={styles.nameBelow}>{nameContent}</div>;
    }
  };

  // Render pick status badge (not shown for user's position r=0)
  const renderStatusBadge = () => {
    if (!pickStatus || relSeat === 0) return null;

    const badgeClass =
      pickStatus === 'PICK'
        ? styles.badgePick
        : pickStatus === 'PASS'
          ? styles.badgePass
          : styles.badgePending;

    return (
      <div className={styles.statusContainer}>
        <span className={`${styles.badge} ${badgeClass}`}>{pickStatus}</span>
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
      {renderNameLabel()}
      {renderStatusBadge()}
    </div>
  );
}

