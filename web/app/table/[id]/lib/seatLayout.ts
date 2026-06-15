// Seat geometry, single-sourced for the desktop and mobile ring stages and the
// trick-collect animation. relSeat 0 is always you (bottom).

/** Convert an absolute seat (1-5) to a relative seat (0-4) from your view. */
export function relSeat(absSeat: number, mySeat: number): number {
  return (absSeat - mySeat + 5) % 5;
}

/**
 * Desktop ring anchor for a relative seat. The five played cards sit on an
 * elliptical ring inside the 560x480 stage so they spread evenly and never
 * crowd the center (relSeat 0 = you, bottom). `cardX`/`cardY` are the card's
 * CENTER as a percentage of the stage box; `side` is which way the name-plate
 * reads (and, for mid seats, which edge it hugs). The plate sits above each
 * card, toward the table rim. Values are tuned (see pentagon overlap check)
 * for a clean ~17px gap at full width.
 */
export interface RingAnchor {
  cardX: number;
  cardY: number;
  plate: "above" | "below"; // where the name-plate floats vs the card
}

export const RING_ANCHORS: Record<number, RingAnchor> = {
  0: { cardX: 50, cardY: 83, plate: "below" }, // you (bottom center)
  1: { cardX: 16, cardY: 55, plate: "above" }, // ml (mid-left)
  2: { cardX: 31, cardY: 20, plate: "below" }, // tl (upper-left)
  3: { cardX: 69, cardY: 20, plate: "below" }, // tr (upper-right)
  4: { cardX: 84, cardY: 55, plate: "above" }, // mr (mid-right)
};

/**
 * Mobile ring anchors. The portrait stage is tighter vertically, so the
 * name-plates fan toward the rim instead of converging in the middle: top
 * seats sit a touch lower with their plate above (toward the top edge), mid
 * seats keep their plate below their card. Tuned for ~64px cards.
 */
export const MOBILE_RING_ANCHORS: Record<number, RingAnchor> = {
  0: { cardX: 50, cardY: 82, plate: "below" }, // you (bottom center)
  1: { cardX: 16, cardY: 53, plate: "below" }, // ml (mid-left)
  2: { cardX: 30, cardY: 29, plate: "above" }, // tl (upper-left)
  3: { cardX: 70, cardY: 29, plate: "above" }, // tr (upper-right)
  4: { cardX: 84, cardY: 53, plate: "below" }, // mr (mid-right)
};
