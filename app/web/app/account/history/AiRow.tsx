import React, { useState } from "react";
import { accountErrorMessage, fetchHandReview } from "../../../lib/account";
import { shortDateTime, signed } from "../../../lib/format";
import { netTone, unreviewedReason } from "../../../lib/review";
import type { HandHistoryRow, HandReview } from "../../../lib/types";
import CostGainBar from "../components/CostGainBar";
import accountStyles from "../account.module.css";
import HandRole from "./HandRole";
import HandStrip from "./HandStrip";
import styles from "./history.module.css";

/** A hand in the AI comparison: what its choices cost and gained against
 * the AI's, and the net. Opens to a strip of its decisions. */
export default function AiRow({ hand }: { hand: HandHistoryRow }) {
  const [open, setOpen] = useState(false);
  const [review, setReview] = useState<HandReview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const status = hand.review?.status ?? null;
  const reviewed = status === "reviewed";
  const cost = hand.review?.cost ?? 0;
  const gained = hand.review?.gained ?? 0;
  const handNet = hand.review?.net ?? 0;
  // Most hands differ from the AI's line a little each way and come out
  // near even. Those are drawn quietly, so the few where a decision moved
  // the hand a full point are the ones the eye finds.
  const notable = hand.review?.notable ?? false;
  const lineTone = hand.counted_score < 0 ? styles.loss : "";

  const toggle = () => {
    setOpen(!open);
    if (!open && !review) {
      setError(null);
      fetchHandReview(hand.game_id).then(setReview, (err) =>
        setError(accountErrorMessage(err)),
      );
    }
  };
  const role = <HandRole hand={hand} />;

  return (
    <>
      {/* The whole row opens it; the button is the same control for the
          keyboard and for screen readers. */}
      <tr
        className={`${lineTone} ${open ? styles.openRow : ""} ${reviewed ? styles.opens : ""}`}
        onClick={reviewed ? toggle : undefined}
      >
        <td className={styles.when}>
          {shortDateTime(hand.time_closed)}
          <span className={`${styles.roleUnder} ${styles.tone}`}>{role}</span>
        </td>
        <td className={`${styles.tone} ${styles.roleCell}`}>{role}</td>
        <td className={`${styles.num} ${styles.tone}`}>{signed(hand.score)}</td>
        {reviewed ? (
          <>
            <td className={styles.barCell}>
              <CostGainBar cost={cost} gained={gained} quiet={!notable} />
            </td>
            <td
              className={`${styles.num} ${styles.net} ${notable ? styles.netNotable : ""}`}
              data-tone={netTone(handNet, notable)}
            >
              {cost === 0 && gained === 0 ? "—" : signed(handNet, 1)}
            </td>
            <td className={styles.expandCell}>
              <button
                type="button"
                className={styles.expand}
                aria-expanded={open}
                aria-label={`Cost ${cost.toFixed(1)}, gained ${gained.toFixed(1)}, net ${signed(handNet, 1)} points of expected score. ${open ? "Hide" : "Show"} decisions`}
                onClick={(event) => {
                  // The row would toggle it straight back.
                  event.stopPropagation();
                  toggle();
                }}
              >
                <span className={open ? styles.caretOpen : undefined}>▾</span>
              </button>
            </td>
          </>
        ) : (
          <td colSpan={3} className={styles.pending}>
            {status === "queued" ? (
              <span role="status">Waiting for review</span>
            ) : (
              <span title={status ? unreviewedReason(status) : "Not reviewed"}>
                —
              </span>
            )}
          </td>
        )}
      </tr>
      {open && (
        <tr className={styles.stripRow}>
          <td colSpan={6}>
            {review ? (
              <HandStrip review={review} />
            ) : (
              <p className={error ? accountStyles.error : accountStyles.note}>
                {error ?? "Loading…"}
              </p>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
