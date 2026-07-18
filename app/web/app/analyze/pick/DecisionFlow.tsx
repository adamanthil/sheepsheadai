"use client";

import React, { useMemo } from "react";
import type { AnalyzeActionDetail } from "../../../lib/analyzeTypes";
import { CardText } from "../../../lib/ds";
import styles from "./DecisionFlow.module.css";

interface DecisionFlowProps {
  decisions: AnalyzeActionDetail[];
  targetSeat: number;
}

/** Every option gets at least this share of the bar so its label stays
 * readable; the rest of the width is distributed by policy probability. */
const MIN_PCT = 9;

interface RowLayout {
  widths: number[]; // percent per option, summing to 100
  centers: number[]; // percent x-center per option
  chosenIndex: number;
}

function layoutRow(decision: AnalyzeActionDetail): RowLayout {
  const clamped = decision.probabilities.map((p) =>
    Math.max(p.prob * 100, MIN_PCT),
  );
  const total = clamped.reduce((a, b) => a + b, 0);
  const widths = clamped.map((w) => (w / total) * 100);
  const centers: number[] = [];
  let acc = 0;
  for (const w of widths) {
    centers.push(acc + w / 2);
    acc += w;
  }
  const chosenIndex = decision.probabilities.findIndex(
    (p) => p.actionId === decision.actionId,
  );
  return { widths, centers, chosenIndex };
}

function formatPct(prob: number): string {
  const pct = prob * 100;
  if (pct >= 99.5) return "100%";
  if (pct < 1) return "<1%";
  return `${pct >= 10 ? Math.round(pct) : pct.toFixed(1)}%`;
}

function formatSigned(value: number): string {
  return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
}

/** Phase heading for a row; bury rows are numbered when there are two. */
function rowLabel(
  decision: AnalyzeActionDetail,
  buryIndex: number,
  buryCount: number,
): string {
  if (decision.phase === "pick") return "Pick or pass";
  if (decision.phase === "partner") return "Partner call";
  const isUnder = decision.probabilities[0]?.action.startsWith("UNDER ");
  if (isUnder) return "Under placement";
  return buryCount > 1 ? `Bury ${buryIndex}` : "Bury";
}

function DecisionRow({
  decision,
  label,
  layout,
}: {
  decision: AnalyzeActionDetail;
  label: string;
  layout: RowLayout;
}) {
  return (
    <div className={styles.row}>
      <div className={styles.rowHeader}>
        <span className={styles.rowLabel}>{label}</span>
        <div className={styles.rowMetrics}>
          <span className={styles.metricChip} title="Value estimate (critic)">
            V {formatSigned(decision.valueEstimate)}
          </span>
          {typeof decision.winProb === "number" && (
            <span className={styles.metricChip} title="Win probability">
              Win {Math.round(decision.winProb * 100)}%
            </span>
          )}
        </div>
      </div>

      <div className={styles.segments}>
        {decision.probabilities.map((p, i) => (
          <div
            key={p.actionId}
            className={`${styles.segment} ${
              i === layout.chosenIndex ? styles.segmentChosen : ""
            }`}
            style={{ width: `${layout.widths[i]}%` }}
            title={`${p.action} — ${(p.prob * 100).toFixed(2)}%`}
          >
            <span className={styles.segmentAction}>
              <CardText>{p.action}</CardText>
            </span>
            <span className={styles.segmentProb}>{formatPct(p.prob)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Elbow arrow from the chosen option of one decision down to the chosen
 * option of the next. Coordinates are percent-based; the stroke keeps its
 * width via non-scaling-stroke and the head is a CSS triangle. */
function Connector({ fromPct, toPct }: { fromPct: number; toPct: number }) {
  return (
    <div className={styles.connector} aria-hidden="true">
      <svg
        className={styles.connectorSvg}
        viewBox="0 0 100 44"
        preserveAspectRatio="none"
      >
        <path
          className={styles.connectorPath}
          d={`M ${fromPct} 0 C ${fromPct} 26, ${toPct} 16, ${toPct} 37`}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <span
        className={styles.connectorHead}
        style={{ left: `calc(${toPct}% - 7px)` }}
      />
    </div>
  );
}

/** The target seat's pre-play decision chain: one proportional bar per
 * decision, sized by the policy's softmax, chained by arrows from each
 * chosen action to the decision it leads to. */
export default function DecisionFlow({
  decisions,
  targetSeat,
}: DecisionFlowProps) {
  const rows = useMemo(
    () => decisions.filter((d) => d.seat === targetSeat),
    [decisions, targetSeat],
  );
  const layouts = useMemo(() => rows.map(layoutRow), [rows]);

  if (rows.length === 0) return null;

  const buryCount = rows.filter(
    (d) =>
      d.phase === "bury" && d.probabilities[0]?.action.startsWith("BURY "),
  ).length;
  let buryIndex = 0;

  const passed = rows[0].action === "PASS";

  return (
    <section className={styles.flowPanel}>
      <h2 className={styles.flowTitle}>Seat {targetSeat} decision flow</h2>

      {rows.map((decision, i) => {
        if (
          decision.phase === "bury" &&
          decision.probabilities[0]?.action.startsWith("BURY ")
        ) {
          buryIndex += 1;
        }
        return (
          <React.Fragment key={decision.stepIndex}>
            {i > 0 && (
              <Connector
                fromPct={layouts[i - 1].centers[layouts[i - 1].chosenIndex]}
                toPct={layouts[i].centers[layouts[i].chosenIndex]}
              />
            )}
            <DecisionRow
              decision={decision}
              label={rowLabel(decision, buryIndex, buryCount)}
              layout={layouts[i]}
            />
          </React.Fragment>
        );
      })}

      {passed && (
        <p className={styles.passNote}>
          Seat {targetSeat} passes — no further decisions for this seat.
        </p>
      )}
    </section>
  );
}
