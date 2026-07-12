import React from "react";
import { PlayingCard } from "../../../../../lib/ds";
import { MOBILE_RING_ANCHORS } from "../../lib/seatLayout";
import CollectOverlay from "../CollectOverlay";
import { CenterContent, seatCardContent } from "./CenterContent";
import { RingChip } from "./chrome";
import type { StageProps } from "./types";
import styles from "../Stage.module.css";

// ---------- Mobile ----------
export default function MobileStage(props: StageProps) {
  const { seats } = props;
  const MOB_CARD = 66;
  const you = seats.find((s) => s.you);
  const youPlayed =
    (props.phase === "play" || props.phase === "done") && you
      ? props.displayCards[you.absSeat - 1] || ""
      : "";
  return (
    <div
      className={styles.mobStage}
      ref={props.trickBoxRef as React.RefObject<HTMLDivElement>}
    >
      <svg
        className={styles.mobEllipse}
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
        aria-hidden="true"
      >
        <ellipse
          cx="50"
          cy="50"
          rx="48"
          ry="46"
          fill="none"
          stroke="var(--rule)"
          strokeDasharray="0.6 1.4"
          strokeWidth="0.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      <div className={styles.mobCenter}>
        <CenterContent props={props} mobile />
      </div>

      {seats
        .filter((s) => !s.you)
        .map((seat) => {
          const anchor = MOBILE_RING_ANCHORS[seat.rel];
          if (!anchor) return null;
          // Inner top seats (plate above) float their badge to the outer side
          // so it never pushes the name up out of alignment.
          const badgeSide =
            anchor.plate === "above"
              ? anchor.cardX < 50
                ? "left"
                : "right"
              : undefined;
          return (
            <div
              key={seat.absSeat}
              className={styles.ringSeat}
              style={{ left: `${anchor.cardX}%`, top: `${anchor.cardY}%` }}
            >
              <RingChip
                seat={seat}
                plate={anchor.plate}
                compact
                badgeSide={badgeSide}
              />
              <div>{seatCardContent(props, seat, MOB_CARD)}</div>
            </div>
          );
        })}

      {youPlayed && (
        <div
          className={styles.ringSeat}
          style={{
            left: `${MOBILE_RING_ANCHORS[0].cardX}%`,
            top: `${MOBILE_RING_ANCHORS[0].cardY}%`,
          }}
        >
          <PlayingCard code={youPlayed} w={MOB_CARD} />
          <span className={styles.youPlate}>You</span>
        </div>
      )}

      {props.animTrick && (
        <CollectOverlay
          yourSeat={props.yourSeat}
          winner={props.animTrick.winner}
          cards={props.animTrick.cards}
          cardW={MOB_CARD}
          anchors={MOBILE_RING_ANCHORS}
        />
      )}

      {props.callout && <div className={styles.callout}>{props.callout}</div>}
    </div>
  );
}
