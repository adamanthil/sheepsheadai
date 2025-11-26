import React from 'react';
import { PlayingCard } from '../../../../lib/components';
import styles from '../page.module.css';

type PickStatus = 'PASS' | 'PICK' | 'PENDING' | null;

interface PlayerHandProps {
  hand: string[];
  handSize: { w: number; h: number };
  handTopMargin: number;
  handRowRef: React.RefObject<HTMLDivElement>;
  isYourTurn: boolean;
  validActionStrings: Set<string>;
  onCardClick: (card: string) => void;
  userPickStatus: PickStatus;
}

export default function PlayerHand({
  hand,
  handSize,
  handTopMargin,
  handRowRef,
  isYourTurn,
  validActionStrings,
  onCardClick,
  userPickStatus,
}: PlayerHandProps) {
  const isCardClickable = (card: string): boolean => {
    if (!isYourTurn) return false;
    return (
      validActionStrings.has(`PLAY ${card}`) ||
      validActionStrings.has(`BURY ${card}`) ||
      validActionStrings.has(`UNDER ${card}`)
    );
  };

  const handleCardClick = (card: string) => {
    if (isCardClickable(card)) {
      onCardClick(card);
    }
  };

  // Render user pick status badge
  const renderStatusBadge = () => {
    if (!userPickStatus) return null;

    const badgeClass =
      userPickStatus === 'PICK'
        ? styles.badgePick
        : userPickStatus === 'PASS'
          ? styles.badgePass
          : styles.badgePending;

    return (
      <span className={`${styles.badge} ${badgeClass} ${styles.ml8}`}>
        {userPickStatus}
      </span>
    );
  };

  return (
    <div
      className={styles.handTopSpacer}
      style={{ ['--handTop' as any]: `${handTopMargin}px` }}
    >
      <div className={styles.sectionTitle}>
        Your hand
        {renderStatusBadge()}
      </div>
      <div ref={handRowRef} className={styles.handRow}>
        {hand.map((card, idx) => {
          const clickable = isCardClickable(card);
          return (
            <div
              key={idx}
              onClick={() => handleCardClick(card)}
              className={clickable ? styles.clickable : undefined}
            >
              <PlayingCard
                label={card}
                highlight={clickable}
                width={handSize.w}
                height={handSize.h}
                bigMarks
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

