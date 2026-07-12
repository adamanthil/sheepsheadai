import React from "react";
import { PlayingCard } from "../../../../../lib/ds";
import { RING_ANCHORS } from "../../lib/seatLayout";
import CollectOverlay from "../CollectOverlay";
import { CenterContent, seatCardContent } from "./CenterContent";
import { RingChip } from "./chrome";
import type { StageProps } from "./types";
import styles from "../Stage.module.css";

// ---------- Desktop ----------
export default function DesktopStage(props: StageProps) {
  const { seats, yourSeat } = props;
  const cardW = Math.round(104 * (props.uiScale ?? 1));
  const you = seats.find((s) => s.you);
  const youPlayed =
    (props.phase === "play" || props.phase === "done") && you
      ? props.displayCards[you.absSeat - 1] || ""
      : "";

  return (
    <div
      className={styles.deskStage}
      ref={props.trickBoxRef as React.RefObject<HTMLDivElement>}
    >
      <div className={styles.deskInner}>
        <svg
          className={styles.ellipse}
          viewBox="0 0 560 480"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <ellipse
            cx="280"
            cy="260"
            rx="260"
            ry="200"
            fill="none"
            stroke="var(--rule)"
            strokeDasharray="2 5"
            strokeWidth="1"
          />
        </svg>

        <div className={styles.center}>
          <CenterContent props={props} />
        </div>

        {seats
          .filter((s) => !s.you)
          .map((seat) => {
            const anchor = RING_ANCHORS[seat.rel];
            if (!anchor) return null;
            return (
              <div
                key={seat.absSeat}
                className={styles.ringSeat}
                style={{ left: `${anchor.cardX}%`, top: `${anchor.cardY}%` }}
              >
                <RingChip seat={seat} plate={anchor.plate} />
                <div>{seatCardContent(props, seat, cardW)}</div>
              </div>
            );
          })}

        {youPlayed && (
          <div
            className={styles.ringSeat}
            style={{
              left: `${RING_ANCHORS[0].cardX}%`,
              top: `${RING_ANCHORS[0].cardY}%`,
            }}
          >
            <PlayingCard code={youPlayed} w={cardW} />
            <span className={styles.youPlate}>You</span>
          </div>
        )}

        {props.showPrev && props.prevText && !props.animTrick && (
          <div className={styles.prevBanner}>{props.prevText}</div>
        )}
        {props.animTrick && (
          <CollectOverlay
            yourSeat={yourSeat}
            winner={props.animTrick.winner}
            cards={props.animTrick.cards}
            cardW={cardW}
            anchors={RING_ANCHORS}
          />
        )}
        {props.callout && <div className={styles.callout}>{props.callout}</div>}
      </div>
    </div>
  );
}
