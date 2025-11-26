import React from 'react';
import { PlayingCard } from '../../../../lib/components';
import styles from '../page.module.css';

type PlayerStatus = 'PASS' | 'PICK' | 'PICKER' | 'PENDING' | 'PARTNER' | null;

interface PlayerHandProps {
  hand: string[];
  handSize: { w: number; h: number };
  handTopMargin: number;
  handRowRef: React.RefObject<HTMLDivElement>;
  isYourTurn: boolean;
  validActionStrings: Set<string>;
  onCardClick: (card: string) => void;
  userStatus: PlayerStatus;
}

export default function PlayerHand({
  hand,
  handSize,
  handTopMargin,
  handRowRef,
  isYourTurn,
  validActionStrings,
  onCardClick,
  userStatus,
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

  return (
    <div
      className={styles.handTopSpacer}
      style={{ ['--handTop' as any]: `${handTopMargin}px` }}
    >
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
      {userStatus && (
        <div className={styles.handStatusBadge}>
          <span
            className={`${styles.badge} ${
              userStatus === 'PICK' || userStatus === 'PICKER'
                ? styles.badgePick
                : userStatus === 'PASS'
                  ? styles.badgePass
                  : userStatus === 'PENDING'
                    ? styles.badgePending
                    : styles.badgePartner
            }`}
          >
            {userStatus}
          </span>
        </div>
      )}
    </div>
  );
}

