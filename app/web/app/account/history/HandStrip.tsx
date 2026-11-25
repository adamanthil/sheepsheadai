import React from "react";
import Link from "next/link";
import { CardText, ds } from "../../../lib/ds";
import {
  actionCodes,
  decisionLabel,
  decisionNote,
  handHeadline,
  tone,
} from "../../../lib/review";
import { signed } from "../../../lib/format";
import type { HandReview } from "../../../lib/types";
import SuitText from "../components/SuitText";
import styles from "./history.module.css";

/** Under an opened history row: the hand's headline, the AI's expected
 * score through it from the player's seat, and each decision in a cell.
 * A phone keeps the headline and the line and drops the cells. */
export default function HandStrip({ review }: { review: HandReview }) {
  const { decisions } = review;
  const headline = handHeadline(review);
  const n = decisions.length;
  const reach = Math.ceil(
    Math.max(4, ...decisions.map((d) => Math.abs(d.expected_score))),
  );
  const y = (value: number) => 32 - (value / reach) * 28;
  const points = decisions
    .map((d, i) => `${50 + 100 * i},${y(d.expected_score).toFixed(1)}`)
    .join(" ");

  return (
    <div className={styles.strip}>
      <div className={styles.stripHead}>
        <div>
          {headline.kicker && (
            <div className={ds.overline}>{headline.kicker}</div>
          )}
          <div className={styles.stripHeadline}>
            <SuitText>{headline.text}</SuitText>
          </div>
        </div>
        <Link
          href={`/account/history/${review.hand.game_id}`}
          className={`${ds.link} ${styles.stripLink}`}
        >
          Open full review →
        </Link>
      </div>
      {n > 0 && (
        <>
          <div className={styles.stripCaption}>
            The AI’s expected score through the hand, from your seat
          </div>
          <div className={styles.stripChart}>
            <span className={styles.stripZero} aria-hidden="true">
              0
            </span>
            <div className={styles.stripPlotColumn}>
              <div
                className={styles.stripPlot}
                role="img"
                aria-label="The AI's expected score before each of your decisions in this hand"
              >
                <svg
                  aria-hidden="true"
                  viewBox={`0 0 ${n * 100} 64`}
                  preserveAspectRatio="none"
                >
                  <line x1="0" y1="32" x2={n * 100} y2="32" />
                  <polyline points={points} />
                </svg>
                {decisions.map((d, i) => (
                  <span
                    key={d.index}
                    className={styles.stripDot}
                    data-tone={tone(d.standing)}
                    style={{
                      left: `${((i + 0.5) / n) * 100}%`,
                      top: `${(y(d.expected_score) / 64) * 100}%`,
                    }}
                  />
                ))}
              </div>
              <ol
                className={styles.stripCells}
                style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}
              >
                {decisions.map((d) => (
                  <li key={d.index}>
                    <span className={styles.cellLabel}>{decisionLabel(d)}</span>
                    <CardText className={styles.cellAction}>
                      {actionCodes(d.action)}
                    </CardText>
                    <span
                      className={styles.cellDifference}
                      data-tone={tone(d.standing)}
                    >
                      {d.difference != null ? signed(d.difference, 1) : "—"}
                    </span>
                    <span
                      className={styles.cellNote}
                      data-tone={tone(d.standing)}
                    >
                      {decisionNote(d)}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
