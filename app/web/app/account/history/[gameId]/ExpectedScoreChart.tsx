import React from "react";
import { decisionLabel, tone } from "../../../../lib/review";
import { signed } from "../../../../lib/format";
import type { DecisionReview } from "../../../../lib/types";
import styles from "./handReview.module.css";

/** The AI's expected score before each of the player's decisions, ending
 * at the hand's final score.
 *
 * Solid lines are the hand as it was played: from the player's seat
 * (black, a dot per decision) and with every card known (grey). The one
 * broken line is hypothetical: from a decision that cost or gained half a
 * point, a dotted line runs to where the estimate would have stood had
 * the player made the AI's choice, and a bar joins that to where it did
 * stand. The bar's length is the difference shown beside it. */
export default function ExpectedScoreChart({
  decisions,
  finalScore,
  selected,
  onSelect,
}: {
  decisions: DecisionReview[];
  finalScore: number;
  selected: number;
  onSelect: (index: number) => void;
}) {
  const n = decisions.length;
  // One column per decision plus one for the end of the hand; the SVG's
  // units are a hundred per column and a hundred each side of zero.
  const x = (i: number) => 50 + 100 * i;
  const left = (i: number) => `${((i + 0.5) / (n + 1)) * 100}%`;

  const actualAfter = (i: number) =>
    i + 1 < n ? decisions[i + 1].expected_score : finalScore;
  // Where the estimate would have stood after the AI's choice: the point
  // the hand really reached next, moved by the difference.
  const ghosts = decisions.flatMap((d, i) =>
    d.difference != null && tone(d.standing) !== "plain"
      ? [{ i, d, value: actualAfter(i) - d.difference, actual: actualAfter(i) }]
      : [],
  );

  const reach = Math.max(
    4,
    Math.abs(finalScore),
    ...decisions.map((d) => Math.abs(d.expected_score)),
    ...decisions.map((d) => Math.abs(d.oracle_score ?? 0)),
    ...ghosts.map((g) => Math.abs(g.value)),
  );
  const domain = Math.ceil(reach);
  const y = (value: number) => 100 - (value / domain) * 90;
  const top = (value: number) => `${y(value) / 2}%`;

  const end = `${x(n)},${y(finalScore).toFixed(1)}`;
  const seatLine = decisions
    .map((d, i) => `${x(i)},${y(d.expected_score).toFixed(1)}`)
    .concat(end)
    .join(" ");
  const hasOracle = decisions.every((d) => d.oracle_score != null);
  const oracleLine = decisions
    .map((d, i) => `${x(i)},${y(d.oracle_score ?? 0).toFixed(1)}`)
    .concat(end)
    .join(" ");
  const width = (n + 1) * 100;

  return (
    <div className={styles.chart}>
      <div className={styles.legend}>
        <span>
          <span className={`${styles.swatch} ${styles.swatchSeat}`} />
          From your seat
        </span>
        {hasOracle && (
          <span>
            <span className={`${styles.swatch} ${styles.swatchOracle}`} />
            With every card known
          </span>
        )}
        {ghosts.length > 0 && (
          <span>
            <span className={styles.swatchGhost}>
              <span />
              <span />
            </span>
            Had you made the AI’s choice
          </span>
        )}
      </div>
      <div className={styles.plotRow}>
        <div className={styles.axis} aria-hidden="true">
          <span style={{ top: top(domain) }}>+{domain}</span>
          <span style={{ top: top(0) }}>0</span>
          <span style={{ top: top(-domain) }}>−{domain}</span>
        </div>
        <div className={styles.plotColumn}>
          <div
            className={styles.plot}
            role="img"
            aria-label="Line chart of the AI's expected score before each of your decisions"
          >
            <svg
              aria-hidden="true"
              viewBox={`0 0 ${width} 200`}
              preserveAspectRatio="none"
            >
              <line className={styles.edge} x1="0" y1="10" x2={width} y2="10" />
              <line
                className={styles.zero}
                x1="0"
                y1="100"
                x2={width}
                y2="100"
              />
              <line
                className={styles.edge}
                x1="0"
                y1="190"
                x2={width}
                y2="190"
              />
              {ghosts.map((g) => (
                <React.Fragment key={g.i}>
                  <line
                    className={styles.ghost}
                    x1={x(g.i)}
                    y1={y(g.d.expected_score)}
                    x2={x(g.i + 1)}
                    y2={y(g.value)}
                  />
                  <line
                    className={styles.gap}
                    data-tone={tone(g.d.standing)}
                    x1={x(g.i + 1)}
                    y1={y(g.value)}
                    x2={x(g.i + 1)}
                    y2={y(g.actual)}
                  />
                </React.Fragment>
              ))}
              {hasOracle && (
                <polyline className={styles.oracle} points={oracleLine} />
              )}
              <polyline className={styles.seat} points={seatLine} />
            </svg>
            {ghosts.map((g) => (
              <React.Fragment key={g.i}>
                <span
                  className={styles.ring}
                  style={{ left: left(g.i + 1), top: top(g.value) }}
                />
                <span
                  className={styles.gapLabel}
                  data-tone={tone(g.d.standing)}
                  style={{
                    left: left(g.i + 1),
                    top: top((g.value + g.actual) / 2),
                  }}
                >
                  {signed(g.d.difference ?? 0, 1)}
                </span>
              </React.Fragment>
            ))}
            {decisions.map((d, i) => (
              <button
                key={d.index}
                type="button"
                className={`${styles.dot} ${d.index === selected ? styles.dotSelected : ""}`}
                data-tone={tone(d.standing)}
                style={{ left: left(i), top: top(d.expected_score) }}
                aria-label={`${decisionLabel(d)}: expected score ${signed(d.expected_score, 1)}`}
                aria-pressed={d.index === selected}
                onClick={() => onSelect(d.index)}
              />
            ))}
          </div>
          <div
            className={styles.ticks}
            style={{ gridTemplateColumns: `repeat(${n + 1}, minmax(0, 1fr))` }}
            aria-hidden="true"
          >
            {decisions.map((d) => (
              <span key={d.index}>{decisionLabel(d, true)}</span>
            ))}
            <span>End</span>
          </div>
        </div>
      </div>
    </div>
  );
}
