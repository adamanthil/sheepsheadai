import React from 'react';
import TrickCard from './TrickCard';
import CollectOverlay from './CollectOverlay';
import { relSeat, nameForSeat, isAiSeat } from '../utils/seatMath';
import type { AnimTrick } from '../hooks/useTrickAnimation';
import type { Callout } from '../hooks/useCallout';
import styles from '../page.module.css';

type PlayerStatus = 'PASS' | 'PICK' | 'PICKER' | 'PENDING' | 'PARTNER' | null;

interface TrickAreaProps {
  cards: string[];
  yourSeat: number;
  table: any;
  showPrev: boolean;
  lastTrick: string[] | null;
  lastTrickWinner: number | null;
  lastTrickPoints?: number;
  animTrick: AnimTrick | null;
  callout: Callout | null;
  centerSize: { w: number; h: number };
  trickBoxRef: React.RefObject<HTMLDivElement>;
  getPlayerStatus: (absSeat: number) => PlayerStatus;
}

export default function TrickArea({
  cards,
  yourSeat,
  table,
  showPrev,
  lastTrick,
  lastTrickWinner,
  lastTrickPoints,
  animTrick,
  callout,
  centerSize,
  trickBoxRef,
  getPlayerStatus,
}: TrickAreaProps) {
  const isPrev = !!showPrev;
  const displayCards: string[] = isPrev && lastTrick ? lastTrick : cards;
  const winnerSeat = isPrev ? lastTrickWinner : null;

  // Calculate trick container height based on center card size
  const trickHeight = Math.max(400, Math.floor(centerSize.h * 1.7));

  return (
    <div
      id="trick-container"
      ref={trickBoxRef}
      className={styles.trickContainer}
      style={{
        ['--trickH' as any]: `${trickHeight}px`,
        ['--cardW' as any]: `${centerSize.w}px`,
        ['--cardH' as any]: `${centerSize.h}px`,
      }}
    >
      {displayCards.map((card, idx) => {
        const absSeat = idx + 1;
        const rel = relSeat(absSeat, yourSeat);
        const highlight = isPrev && winnerSeat === absSeat;
        const name = nameForSeat(absSeat, table);
        const isAi = isAiSeat(absSeat, table);
        const playerStatus = getPlayerStatus(absSeat);

        return (
          <TrickCard
            key={idx}
            card={card}
            relSeat={rel}
            name={name}
            isAi={isAi}
            highlight={highlight}
            playerStatus={playerStatus}
            centerSize={centerSize}
          />
        );
      })}

      {/* Previous trick info banner */}
      {showPrev && !animTrick && lastTrick && (
        <div className={styles.prevBanner}>
          Previous trick · Winner: {nameForSeat(lastTrickWinner, table)} · Points:{' '}
          {lastTrickPoints}
        </div>
      )}

      {/* Trick collect animation overlay */}
      {animTrick && (
        <CollectOverlay
          containerRef={trickBoxRef}
          yourSeat={yourSeat}
          winner={animTrick.winner}
          cards={animTrick.cards}
          centerSize={centerSize}
        />
      )}

      {/* Callout overlay */}
      {callout && (
        <div className={styles.callout}>
          <div className={styles.calloutTitle}>{callout.message}</div>
        </div>
      )}
    </div>
  );
}

