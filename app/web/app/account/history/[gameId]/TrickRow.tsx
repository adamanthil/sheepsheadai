import React from "react";
import { PlayingCard, ds } from "../../../../lib/ds";
import type { DecisionReview, HandSeat } from "../../../../lib/types";
import styles from "./handReview.module.css";

/** The trick a card was played to, as the player saw it: the five seats
 * in playing order, the cards already down, the player's own turn, and
 * the seats still to come.
 *
 * A seat carries its role only where the table knew it. The picker is
 * always known. The partner is known once the card that names them has
 * been played, and to the partner themselves from the start. */
export default function TrickRow({
  decision,
  seat,
  seats,
}: {
  decision: DecisionReview;
  seat: number;
  seats: HandSeat[];
}) {
  const played = new Map(decision.table);
  const lead = decision.table.length ? decision.table[0][0] : seat;
  const order = [0, 1, 2, 3, 4].map((i) => ((lead - 1 + i) % 5) + 1);

  const roleOf = (n: number): string | null => {
    const role = seats.find((s) => s.seat === n)?.role;
    if (role === "picker") return "Picker";
    if (role === "partner" && (decision.partner_known || n === seat))
      return "Partner";
    return null;
  };

  return (
    <ol
      className={styles.trick}
      aria-label="The trick so far, in playing order"
    >
      {order.map((n) => {
        const card = played.get(n);
        const you = n === seat;
        const role = roleOf(n);
        return (
          <li key={n} className={you ? styles.trickYou : undefined}>
            {card ? (
              <PlayingCard code={card} largeIndex />
            ) : (
              <span
                className={`${styles.trickEmpty} ${you ? styles.trickTurn : ""}`}
                aria-label={you ? "Your turn" : "Yet to play"}
              />
            )}
            <span className={styles.trickSeat}>
              {you ? "You" : `Seat ${n}`}
            </span>
            {role && (
              <span
                className={`${ds.badge} ${role === "Picker" ? ds.badgeInk : ds.badgeQuiet} ${styles.trickRole}`}
              >
                {role}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
