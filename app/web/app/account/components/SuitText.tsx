import React from "react";
import { RANK_PATTERN } from "../../../lib/ds";

// A card as the review wording writes it: rank, then a red suit's symbol.
const RED_CARD = new RegExp(`(${RANK_PATTERN}[♥♦])`);

/** A sentence that names cards ("You led the Q♣. The AI leads the J♦."),
 * with the hearts and diamonds in the red suit colour like every card
 * elsewhere. Clubs and spades keep the colour of the text around them. */
export default function SuitText({ children }: { children: string }) {
  return (
    <>
      {children.split(RED_CARD).map((part, i) =>
        i % 2 === 1 ? (
          <span key={i} style={{ color: "var(--card-red)" }}>
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </>
  );
}
