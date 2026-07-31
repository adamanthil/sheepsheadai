import React from "react";
import { PlayingCard } from "../../../../../lib/ds";
import { MOBILE_RING_ANCHORS } from "../../lib/seatLayout";
import CollectOverlay from "../CollectOverlay";
import { CenterContent, seatCardContent } from "./CenterContent";
import { RingChip } from "./chrome";
import type { StageProps } from "./types";
import styles from "../Stage.module.css";

// ---------- Mobile ----------
// Card sizing is CSS-driven: the stage is a size container and Stage.module.css
// sets --pc-w on its children from the measured box, so the ring cards, chits
// and collect animation all follow without any JS measurement. The mobStagePlay
// class lets the played cards grow past the decision-phase chit size.
export default function MobileStage(props: StageProps) {
  const { seats } = props;
  const playing = props.phase === "play" || props.phase === "done";
  const you = seats.find((s) => s.you);
  const youPlayed =
    playing && you ? props.displayCards[you.absSeat - 1] || "" : "";
  return (
    <div
      className={`${styles.mobStage} ${playing ? styles.mobStagePlay : ""}`}
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
          const topRow = anchor.plate === "above";
          const badgeSide = topRow
            ? anchor.cardX < 50
              ? "left"
              : "right"
            : undefined;
          return (
            <div
              key={seat.absSeat}
              // The top row's y comes from .mobRowTop (it slides down off the
              // top edge on a short stage); the other rows sit at their anchor.
              className={`${styles.ringSeat} ${topRow ? styles.mobRowTop : ""}`}
              data-seat-rel={seat.rel}
              style={{
                left: `${anchor.cardX}%`,
                ...(topRow ? null : { top: `${anchor.cardY}%` }),
              }}
            >
              <RingChip
                seat={seat}
                plate={anchor.plate}
                compact
                badgeSide={badgeSide}
              />
              <div>{seatCardContent(props, seat)}</div>
            </div>
          );
        })}

      {youPlayed && (
        <div
          className={styles.ringSeat}
          data-seat-rel={0}
          style={{
            left: `${MOBILE_RING_ANCHORS[0].cardX}%`,
            top: `${MOBILE_RING_ANCHORS[0].cardY}%`,
          }}
        >
          <PlayingCard code={youPlayed} />
          <span className={styles.youPlate}>You</span>
        </div>
      )}

      {props.animTrick && (
        <CollectOverlay
          yourSeat={props.yourSeat}
          winner={props.animTrick.winner}
          cards={props.animTrick.cards}
          anchors={MOBILE_RING_ANCHORS}
        />
      )}

      {props.callout && <div className={styles.callout}>{props.callout}</div>}
    </div>
  );
}
