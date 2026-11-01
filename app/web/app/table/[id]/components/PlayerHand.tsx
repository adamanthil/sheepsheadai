import React from "react";
import { PlayingCard, cardName, ds } from "../../../../lib/ds";
import type { TablePhase, InterludeMode, YourRole } from "../lib/phase";
import { roleBadge } from "./stage/chrome";
import styles from "./PlayerHand.module.css";

interface PlayerHandProps {
  hand: string[];
  /** The picker still holds a face-down under card (drawn last in the fan). */
  hasUnder?: boolean;
  isYourTurn: boolean;
  phase: TablePhase;
  yourMode: InterludeMode;
  validActionStrings: Set<string>;
  onCardClick: (card: string) => void;
  // Cards staged in the center (bury/under selection): hidden from the fan
  // until they are confirmed or put back.
  stagedCards?: string[];
  // Your own role from private info; shown as a persistent badge so a player
  // taking over mid-hand can tell at a glance which side they're on.
  yourRole?: YourRole;
  // You dealt this hand; noted in the overline since your seat has no avatar
  // to carry the dealer button.
  isDealer?: boolean;
  isMobile: boolean;
  uiScale?: number;
}

const META: Record<string, string> = {
  pick: "Your starting hand · waiting on the blind",
  bury: "Choose 2 cards to bury, then confirm",
  call: "Picked the blind · now call your partner",
  under: "Called under · choose the card to tuck under",
  play: "Tap a highlighted card to play",
  done: "Hand complete",
};

// Prompt badge shown while an interlude mode wants card taps.
const INTERLUDE_BADGE: Partial<Record<InterludeMode, string>> = {
  bury: "tap to bury",
  under: "choose under",
};

// Stands in for the under card in the fan; tapping it sends PLAY UNDER.
const UNDER = "UNDER";

export default function PlayerHand({
  hand,
  hasUnder,
  isYourTurn,
  phase,
  yourMode,
  validActionStrings,
  onCardClick,
  stagedCards,
  yourRole,
  isDealer,
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

  const metaKey = phase === "interlude" ? yourMode : phase;
  const meta = META[metaKey] ?? META.play;

  const interludeBadge =
    phase === "interlude" ? INTERLUDE_BADGE[yourMode] : undefined;

  return (
    <div
      className={`${styles.wrap} ${isMobile ? styles.mobWrap : styles.deskWrap}`}
    >
      <div className={styles.metaRow}>
        <div style={{ minWidth: 0 }}>
          <div
            className={`${ds.overline} ${styles.overline}`}
            style={{ fontSize: isMobile ? 9 : 11 }}
          >
            Your hand · {fanCards.length} cards
            {isDealer && " · Dealer"}
          </div>
          {/* Mobile drops the prompt: the action bar below already carries
              the same instruction, and the line's height goes to the ring. */}
          {!isMobile && (
            <div className={styles.metaText} style={{ fontSize: 16 }}>
              {meta}
            </div>
          )}
        </div>
        <div className={styles.badges}>
          {isYourTurn && phase === "play" && (
            <span
              className={`${ds.badge} ${ds.badgeAccent2}`}
              style={{ fontSize: 10 }}
            >
              ● Your turn
            </span>
          )}
          {interludeBadge && (
            <span
              className={`${ds.badge} ${ds.badgeAccent}`}
              style={{ fontSize: 10 }}
            >
              {interludeBadge}
            </span>
          )}
          {yourRole && roleBadge(yourRole)}
        </div>
      </div>

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
