import React from "react";
import { PlayingCard, ds } from "../../../../lib/ds";
import type { TablePhase, InterludeMode } from "../lib/phase";
import styles from "./PlayerHand.module.css";

interface PlayerHandProps {
  hand: string[];
  isYourTurn: boolean;
  phase: TablePhase;
  yourMode: InterludeMode;
  validActionStrings: Set<string>;
  onCardClick: (card: string) => void;
  isMobile: boolean;
  uiScale?: number;
}

const META: Record<string, string> = {
  pick: "Your starting hand · waiting on the blind",
  bury: "Picked the blind · tap 2 cards to bury",
  call: "Buried · now call your partner",
  play: "Tap a highlighted card to play",
  done: "Hand complete",
};

export default function PlayerHand({
  hand,
  isYourTurn,
  phase,
  yourMode,
  validActionStrings,
  onCardClick,
  isMobile,
  uiScale = 1,
}: PlayerHandProps) {
  const w = isMobile ? 60 : Math.round(96 * uiScale);
  const overlap = Math.round(w * 0.32);

  const isClickable = (card: string) =>
    isYourTurn &&
    (validActionStrings.has(`PLAY ${card}`) ||
      validActionStrings.has(`BURY ${card}`) ||
      validActionStrings.has(`UNDER ${card}`));

  const metaKey = phase === "interlude" ? yourMode : phase;
  const meta = META[metaKey] ?? META.play;

  const buryChosen = phase === "interlude" && yourMode === "bury";

  return (
    <div
      className={`${styles.wrap} ${isMobile ? styles.mobWrap : styles.deskWrap}`}
    >
      <div className={styles.metaRow}>
        <div style={{ minWidth: 0 }}>
          <div className={ds.overline} style={{ fontSize: isMobile ? 9 : 11 }}>
            Your hand · {hand.length} cards
          </div>
          <div
            className={styles.metaText}
            style={{ fontSize: isMobile ? 12 : 16 }}
          >
            {meta}
          </div>
        </div>
        {isYourTurn && phase === "play" && (
          <span
            className={`${ds.badge} ${ds.badgeAccent2}`}
            style={{ fontSize: 10 }}
          >
            ● Your turn
          </span>
        )}
        {buryChosen && (
          <span
            className={`${ds.badge} ${ds.badgeAccent}`}
            style={{ fontSize: 10 }}
          >
            tap to bury
          </span>
        )}
      </div>

      <div className={styles.fan}>
        {hand.map((card, i) => {
          const clickable = isClickable(card);
          return (
            <div
              key={card + i}
              className={`${styles.cardSlot} ${clickable ? styles.clickable : ""}`}
              style={{ marginLeft: i === 0 ? 0 : -overlap, zIndex: i }}
              data-clickable={clickable || undefined}
              onClick={() => clickable && onCardClick(card)}
            >
              <PlayingCard code={card} w={w} playable={clickable} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
