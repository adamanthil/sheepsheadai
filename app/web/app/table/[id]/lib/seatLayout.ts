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
 * card; plates always read on the outside of the ellipse (outer seats below,
 * inner seats to the side) so they never crowd the center UI. Values are tuned
 * (see pentagon overlap check)
 * for a clean ~17px gap at full width.
 */
export interface RingAnchor {
  cardX: number;
  cardY: number;
  // Where the name-plate floats relative to the card. Plates always read on the
  // OUTSIDE of the ellipse so they never crowd the interior or the center UI:
  // the outer mid seats drop below their card, the inner top seats fan out to
  // the side (left/right toward the rim).
  plate: "above" | "below" | "left" | "right";
}

export const RING_ANCHORS: Record<number, RingAnchor> = {
  0: { cardX: 50, cardY: 83, plate: "below" }, // you (bottom center)
  1: { cardX: 16, cardY: 55, plate: "below" }, // ml (mid-left, outer)
  2: { cardX: 31, cardY: 20, plate: "left" }, // tl (upper-left, inner)
  3: { cardX: 69, cardY: 20, plate: "right" }, // tr (upper-right, inner)
  4: { cardX: 84, cardY: 55, plate: "below" }, // mr (mid-right, outer)
};

/**
 * Mobile ring anchors. The portrait stage is tighter vertically, so the
 * name-plates fan toward the rim instead of converging in the middle: top
 * seats sit a touch lower with their plate above (toward the top edge), mid
 * seats keep their plate below their card. Card size is CSS-driven and these
 * rows are tuned as a pair with the --mob-card-fit formula in Stage.module.css:
 * the formula returns the widest card these fixed rows can hold, so moving a
 * row means rederiving it. Top row splits the room between its plate above and
 * the center block; mid row sits a full card below the top row; you sits
 * between the center block and the bottom edge (with the "You" label).
 */
export const MOBILE_RING_ANCHORS: Record<number, RingAnchor> = {
  0: { cardX: 50, cardY: 75, plate: "below" }, // you (bottom center)
  1: { cardX: 16, cardY: 60, plate: "below" }, // ml (mid-left)
  2: { cardX: 30, cardY: 29, plate: "above" }, // tl (upper-left)
  3: { cardX: 70, cardY: 29, plate: "above" }, // tr (upper-right)
  4: { cardX: 84, cardY: 60, plate: "below" }, // mr (mid-right)
};
