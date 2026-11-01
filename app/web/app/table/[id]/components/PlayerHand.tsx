import React from "react";
import { PlayingCard, cardName } from "../../../../lib/ds";
import styles from "./PlayerHand.module.css";

interface PlayerHandProps {
  hand: string[];
  /** The picker still holds a face-down under card (drawn last in the fan). */
  hasUnder?: boolean;
  isYourTurn: boolean;
  validActionStrings: Set<string>;
  onCardClick: (card: string) => void;
  // Cards staged in the center (bury/under selection): hidden from the fan
  // until they are confirmed or put back.
  stagedCards?: string[];
  isMobile: boolean;
  uiScale?: number;
}

// Stands in for the under card in the fan; tapping it sends PLAY UNDER.
const UNDER = "UNDER";

export default function PlayerHand({
  hand,
  hasUnder,
  isYourTurn,
  validActionStrings,
  onCardClick,
  stagedCards,
  isMobile,
  uiScale = 1,
}: PlayerHandProps) {
  // Card size lives in --pc-w on the fan: desktop sets it from uiScale here,
  // mobile leaves it to the container-query rule in PlayerHand.module.css.
  // The fan spacing derives from the same variable.
  const fanVars = isMobile
    ? undefined
    : ({
        ["--pc-w" as string]: `${Math.round(96 * uiScale)}px`,
      } as React.CSSProperties);

  const isClickable = (card: string) =>
    isYourTurn &&
    (card === UNDER
      ? validActionStrings.has(`PLAY ${UNDER}`)
      : validActionStrings.has(`PLAY ${card}`) ||
        validActionStrings.has(`BURY ${card}`) ||
        validActionStrings.has(`UNDER ${card}`));

  const staged = stagedCards ?? [];
  const shownHand = hand.filter((card) => !staged.includes(card));
  const fanCards = hasUnder ? [...shownHand, UNDER] : shownHand;

  return (
    <div
      className={`${styles.wrap} ${isMobile ? styles.mobWrap : styles.deskWrap}`}
    >
      <div className={styles.fan} style={fanVars}>
        {/* Every card is a button so the playable ones can be tabbed to;
            the rest are disabled, which also takes them out of tab order. */}
        {fanCards.map((card, i) => {
          const clickable = isClickable(card);
          return (
            <button
              type="button"
              key={card + i}
              className={`${styles.cardSlot} ${clickable ? styles.clickable : ""}`}
              style={{ zIndex: i }}
              disabled={!clickable}
              data-clickable={clickable || undefined}
              aria-label={
                card === UNDER
                  ? "Your under card, played face down"
                  : cardName(card)
              }
              onClick={() => onCardClick(card)}
            >
              {card === UNDER ? (
                <PlayingCard code="__" label="under" playable={clickable} />
              ) : (
                <PlayingCard code={card} playable={clickable} />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
