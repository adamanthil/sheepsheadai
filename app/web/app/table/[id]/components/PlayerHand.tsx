import React from "react";
import { PlayingCard, ds } from "../../../../lib/ds";
import type { TablePhase, InterludeMode, YourRole } from "../lib/phase";
import { roleBadge } from "./stage/chrome";
import styles from "./PlayerHand.module.css";

interface PlayerHandProps {
  hand: string[];
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

export default function PlayerHand({
  hand,
  isYourTurn,
  phase,
  yourMode,
  validActionStrings,
  onCardClick,
  stagedCards,
  yourRole,
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
    (validActionStrings.has(`PLAY ${card}`) ||
      validActionStrings.has(`BURY ${card}`) ||
      validActionStrings.has(`UNDER ${card}`));

  const staged = stagedCards ?? [];
  const shownHand = hand.filter((card) => !staged.includes(card));

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
          <div className={ds.overline} style={{ fontSize: isMobile ? 9 : 11 }}>
            Your hand · {shownHand.length} cards
          </div>
          <div
            className={styles.metaText}
            style={{ fontSize: isMobile ? 12 : 16 }}
          >
            {meta}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
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
        {shownHand.map((card, i) => {
          const clickable = isClickable(card);
          return (
            <div
              key={card + i}
              className={`${styles.cardSlot} ${clickable ? styles.clickable : ""}`}
              style={{ zIndex: i }}
              data-clickable={clickable || undefined}
              onClick={() => clickable && onCardClick(card)}
            >
              <PlayingCard code={card} playable={clickable} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
