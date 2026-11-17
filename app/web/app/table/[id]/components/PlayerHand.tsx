import React, { useEffect, useState } from "react";
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

// How long cards picked up from the blind stay marked. Matches the halo's
// fade in PlayerHand.module.css.
const FROM_BLIND_MS = 5000;
const NONE: ReadonlySet<string> = new Set();

/**
 * The cards that just joined the hand. A hand only grows when its owner picks
 * up the blind, so whatever appears in a non-empty hand came from the blind;
 * a hand that starts from empty (a new deal, taking a seat) marks nothing.
 * Derived during render rather than in an effect, so the new cards mount
 * with their class already on and their animation starts on the first frame.
 */
function useFromBlind(hand: string[]): ReadonlySet<string> {
  const [prevHand, setPrevHand] = useState(hand);
  const [fromBlind, setFromBlind] = useState(NONE);
  if (hand !== prevHand) {
    setPrevHand(hand);
    const added =
      prevHand.length > 0 ? hand.filter((c) => !prevHand.includes(c)) : [];
    if (added.length > 0) setFromBlind(new Set(added));
  }
  useEffect(() => {
    if (fromBlind.size === 0) return;
    const t = setTimeout(() => setFromBlind(NONE), FROM_BLIND_MS);
    return () => clearTimeout(t);
  }, [fromBlind]);
  return fromBlind;
}

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

  const fromBlind = useFromBlind(hand);
  const staged = stagedCards ?? [];
  const shownHand = hand.filter((card) => !staged.includes(card));
  const fanCards = hasUnder ? [...shownHand, UNDER] : shownHand;

  return (
    <div
      className={`${styles.wrap} ${isMobile ? styles.mobWrap : styles.deskWrap}`}
    >
      <div
        className={styles.fan}
        style={fanVars}
        role="group"
        aria-label="Your hand"
      >
        {/* Every card is a button so the playable ones can be tabbed to;
            the rest are disabled, which also takes them out of tab order. */}
        {fanCards.map((card, i) => {
          const clickable = isClickable(card);
          const isNew = fromBlind.has(card);
          return (
            <button
              type="button"
              // Keyed by card alone so a card keeps its element when the
              // blind's cards sort in around it.
              key={card}
              className={`${styles.cardSlot} ${clickable ? styles.clickable : ""} ${isNew ? styles.fromBlind : ""}`}
              style={{ zIndex: i }}
              disabled={!clickable}
              data-clickable={clickable || undefined}
              aria-label={
                card === UNDER
                  ? "Your under card, played face down"
                  : isNew
                    ? `${cardName(card)}, from the blind`
                    : cardName(card)
              }
              onClick={() => onCardClick(card)}
            >
              {card === UNDER ? (
                <PlayingCard code="__" label="under" playable={clickable} />
              ) : (
                <PlayingCard
                  code={card}
                  playable={clickable}
                  className={isNew ? styles.blindGlow : undefined}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
