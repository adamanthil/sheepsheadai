// Seat geometry, single-sourced for the desktop ring stage, the mobile 2x2
// grid, and the trick-collect animation. relSeat 0 is always you (bottom).
import type React from "react";

export type RingPos = "tl" | "tr" | "ml" | "mr";

/** Convert an absolute seat (1-5) to a relative seat (0-4) from your view. */
export function relSeat(absSeat: number, mySeat: number): number {
  return (absSeat - mySeat + 5) % 5;
}

/** The four opponent ring positions, by relative seat. (0 = you, bottom.) */
export function ringPosForRel(rel: number): RingPos | null {
  switch (rel) {
    case 1:
      return "ml";
    case 2:
      return "tl";
    case 3:
      return "tr";
    case 4:
      return "mr";
    default:
      return null; // rel 0 = you, rendered separately
  }
}

// Desktop ring: absolute placement inside the 560x480 stage box. `cardSide`
// is which side of the chip the played card sits (toward the table center).
export const RING_COORDS: Record<
  RingPos,
  { style: React.CSSProperties; cardSide: "left" | "right" }
> = {
  tl: { style: { left: 0, top: 10 }, cardSide: "right" },
  tr: { style: { right: 0, top: 10 }, cardSide: "left" },
  ml: { style: { left: 0, top: 280 }, cardSide: "right" },
  mr: { style: { right: 0, top: 280 }, cardSide: "left" },
};

/**
 * Percentage anchors (of the trick container) for the collect animation,
 * by relative seat. These approximate where each seat's played card sits so
 * the cards fly to the winner believably. Kept here so the stage and the
 * animation never drift apart.
 */
export function collectAnchorPct(rel: number): { left: number; top: number } {
  switch (rel) {
    case 0:
      return { left: 50, top: 86 }; // you (bottom center)
    case 1:
      return { left: 14, top: 60 }; // ml
    case 2:
      return { left: 18, top: 18 }; // tl
    case 3:
      return { left: 82, top: 18 }; // tr
    case 4:
      return { left: 86, top: 60 }; // mr
    default:
      return { left: 50, top: 50 };
  }
}

// Mobile 2x2 grid: which corner each relative seat occupies.
export type GridCorner = "tl" | "tr" | "bl" | "br";
export function gridCornerForRel(rel: number): GridCorner | null {
  switch (rel) {
    case 2:
      return "tl";
    case 3:
      return "tr";
    case 1:
      return "bl";
    case 4:
      return "br";
    default:
      return null;
  }
}
