import React, { useMemo, useRef, useState } from "react";
import {
  AnalyzeActionDetail,
  AnalyzeMemoryObserve,
} from "../../lib/analyzeTypes";
import Term from "./TermHelp";
import styles from "./MemoryDriftChart.module.css";

interface MemoryDriftChartProps {
  trace: AnalyzeActionDetail[];
  /** Trick-completion memory updates (all seats), plotted as hollow
   * markers on the trick boundaries. */
  observes?: AnalyzeMemoryObserve[];
}

// Fixed categorical order (blue, green, magenta, yellow, aqua) — validated
// all-pairs CVD-safe with secondary encoding (legend + line-key), see the
// dataviz skill. Sheepshead never has more than 5 seats, so this never runs
// out.
const SEAT_COLORS = [
  "#2a78d6",
  "#008300",
  "#e87ba4",
  "#eda100",
  "#1baf7a",
];

const VB_WIDTH = 1000;
const VB_HEIGHT = 260;
const MARGIN = { top: 20, right: 18, bottom: 62, left: 48 };
const PLOT_WIDTH = VB_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = VB_HEIGHT - MARGIN.top - MARGIN.bottom;

// Sheepshead always has exactly 5 players.
const PLAYERS_PER_TRICK = 5;

/** Compact per-step label: the card for plays, prefixed forms otherwise. */
function shortAction(action: string): string {
  if (action.startsWith("PLAY ")) return action.slice(5);
  if (action.startsWith("BURY ")) return `B·${action.slice(5)}`;
  if (action.startsWith("CALL ")) return `C·${action.slice(5)}`;
  if (action.startsWith("UNDER ")) return `U·${action.slice(6)}`;
  if (action === "JD PARTNER") return "JD P";
  return action; // PICK / PASS / ALONE
}

interface SeriesPoint {
  x: number; // stepIndex for decisions, afterStepIndex + 0.5 for observes
  value: number;
  kind: "decision" | "observe";
}

interface Boundary {
  beforeStep: number; // the boundary sits just before this stepIndex
  label: string;
}

export default function MemoryDriftChart({
  trace,
  observes = [],
}: MemoryDriftChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const {
    seats,
    colorBySeat,
    stepInfo,
    observesByX,
    hoverXs,
    boundaries,
    minStep,
    maxStep,
    maxValue,
  } = useMemo(() => {
    const bySeat = new Map<
      number,
      { seat: number; seatName: string; points: SeriesPoint[] }
    >();
    const info = new Map<number, AnalyzeActionDetail>();
    let minS = Infinity;
    let maxS = -Infinity;
    let maxV = 0;

    const ensureSeat = (seat: number, seatName: string) => {
      if (!bySeat.has(seat)) {
        bySeat.set(seat, { seat, seatName, points: [] });
      }
      return bySeat.get(seat)!;
    };

    // Stable seat -> color mapping over ALL acting seats so labels, lines,
    // and observe markers agree.
    const actingSeats = [...new Set(trace.map((a) => a.seat))].sort(
      (a, b) => a - b,
    );
    const colors = new Map<number, string>();
    actingSeats.forEach((seat, i) =>
      colors.set(seat, SEAT_COLORS[i % SEAT_COLORS.length]),
    );

    for (const action of trace) {
      info.set(action.stepIndex, action);
      if (action.stepIndex < minS) minS = action.stepIndex;
      if (action.stepIndex > maxS) maxS = action.stepIndex;
      if (typeof action.memoryCosineDistance !== "number") continue;
      ensureSeat(action.seat, action.seatName).points.push({
        x: action.stepIndex,
        value: action.memoryCosineDistance,
        kind: "decision",
      });
      if (action.memoryCosineDistance > maxV) {
        maxV = action.memoryCosineDistance;
      }
    }

    const obsByX = new Map<number, AnalyzeMemoryObserve[]>();
    for (const obs of observes) {
      if (typeof obs.memoryCosineDistance !== "number") continue;
      // All five seats observe the completed trick at the same instant, so
      // every observe marker sits exactly on the boundary line.
      const x = obs.afterStepIndex + 0.5;
      ensureSeat(obs.seat, obs.seatName).points.push({
        x,
        value: obs.memoryCosineDistance,
        kind: "observe",
      });
      if (obs.memoryCosineDistance > maxV) maxV = obs.memoryCosineDistance;
      if (!obsByX.has(x)) obsByX.set(x, []);
      obsByX.get(x)!.push(obs);
    }

    bySeat.forEach((s) => s.points.sort((a, b) => a.x - b.x));

    // Phase / trick boundaries, mirroring the timeline's dividers.
    const bounds: Boundary[] = [];
    let playActionsBefore = 0;
    for (let i = 0; i < trace.length; i++) {
      const current = trace[i];
      const prev = i > 0 ? trace[i - 1] : null;
      if (prev && current.phase !== prev.phase) {
        bounds.push({ beforeStep: current.stepIndex, label: current.phase });
      } else if (
        prev &&
        current.phase === "play" &&
        prev.phase === "play" &&
        playActionsBefore > 0 &&
        playActionsBefore % PLAYERS_PER_TRICK === 0
      ) {
        bounds.push({
          beforeStep: current.stepIndex,
          label: `T${Math.floor(playActionsBefore / PLAYERS_PER_TRICK) + 1}`,
        });
      }
      if (current.phase === "play") playActionsBefore += 1;
    }

    const xs = new Set<number>();
    info.forEach((_a, step) => xs.add(step));
    obsByX.forEach((_o, x) => xs.add(x));

    return {
      seats: [...bySeat.values()].sort((a, b) => a.seat - b.seat),
      colorBySeat: colors,
      stepInfo: info,
      observesByX: obsByX,
      hoverXs: [...xs].sort((a, b) => a - b),
      boundaries: bounds,
      minStep: Number.isFinite(minS) ? minS : 0,
      maxStep: Number.isFinite(maxS) ? maxS : 1,
      maxValue: maxV,
    };
  }, [trace, observes]);

  const hasData = seats.length > 0;

  const xScale = (x: number) => {
    const span = maxStep + 0.5 - minStep || 1;
    return MARGIN.left + ((x - minStep) / span) * PLOT_WIDTH;
  };
  const yMax = maxValue > 0 ? maxValue * 1.15 : 1;
  const yScale = (value: number) => {
    return MARGIN.top + PLOT_HEIGHT - (value / yMax) * PLOT_HEIGHT;
  };

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!hasData || !svgRef.current || hoverXs.length === 0) return;
    // Screen -> viewBox via the SVG's own transform matrix, so the mapping
    // stays exact regardless of how CSS scales or letterboxes the element.
    const ctm = svgRef.current.getScreenCTM();
    if (!ctm) return;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(
      ctm.inverse(),
    );
    const span = maxStep + 0.5 - minStep || 1;
    const approx = minStep + ((pt.x - MARGIN.left) / PLOT_WIDTH) * span;
    let nearest = hoverXs[0];
    let best = Infinity;
    for (const x of hoverXs) {
      const d = Math.abs(x - approx);
      if (d < best) {
        best = d;
        nearest = x;
      }
    }
    setHoverX(nearest);
  };

  if (!hasData) return null;

  const yTicks = [0, yMax / 2, yMax];
  const hoverAction =
    hoverX !== null && Number.isInteger(hoverX)
      ? stepInfo.get(hoverX)
      : undefined;
  const hoverObserves =
    hoverX !== null && !Number.isInteger(hoverX)
      ? observesByX.get(hoverX)
      : undefined;

  return (
    <div className={styles.memoryDriftChart}>
      <div className={styles.header}>
        <div className={styles.title}>
          <Term
            label="Memory Drift"
            wiki="https://en.wikipedia.org/wiki/Gated_recurrent_unit"
          >
            Each seat carries a memory vector (256 dimensions) that a
            recurrent unit (GRU) rewrites at two kinds of update: filled
            markers are the seat&rsquo;s own decisions, where the encoder
            folds the current observation (its hand, the trick so far, and
            the public flags) into the memory; hollow markers on the trick
            boundaries are trick-completion observations, where all five
            seats fold the finished trick&rsquo;s outcome into memory. Each
            point is the cosine distance between the memory before and after
            that update: near 0 = little revision of the model&rsquo;s
            internal state; higher = a substantial one. Dashed vertical
            lines mark phase changes and trick boundaries, and each step is
            labeled with the action taken.
          </Term>
        </div>
        <div className={styles.legend}>
          {seats.map((s) => (
            <div key={s.seat} className={styles.legendItem}>
              <span
                className={styles.legendSwatch}
                style={{ background: colorBySeat.get(s.seat) }}
              />
              {s.seatName}
            </div>
          ))}
          <div className={styles.legendItem}>
            <span className={styles.legendDotDecision} /> decision
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendDotObserve} /> trick observe
          </div>
        </div>
      </div>

      <div className={styles.chartWrap}>
        <svg
          ref={svgRef}
          className={styles.svg}
          viewBox={`0 0 ${VB_WIDTH} ${VB_HEIGHT}`}
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverX(null)}
        >
          {/* y gridlines + labels */}
          {yTicks.map((t, i) => (
            <g key={i}>
              <line
                x1={MARGIN.left}
                x2={VB_WIDTH - MARGIN.right}
                y1={yScale(t)}
                y2={yScale(t)}
                className={styles.gridline}
              />
              <text
                x={MARGIN.left - 6}
                y={yScale(t)}
                className={styles.axisLabel}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {t.toFixed(2)}
              </text>
            </g>
          ))}

          {/* y-axis title */}
          <text
            className={styles.axisTitle}
            transform={`translate(${12}, ${MARGIN.top + PLOT_HEIGHT / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            memory Δ (cosine)
          </text>

          {/* phase / trick boundaries */}
          {boundaries.map((b, i) => {
            const x = (xScale(b.beforeStep - 1) + xScale(b.beforeStep)) / 2;
            return (
              <g key={i}>
                <line
                  x1={x}
                  x2={x}
                  y1={MARGIN.top - 4}
                  y2={MARGIN.top + PLOT_HEIGHT}
                  className={styles.boundary}
                />
                <text
                  x={x + 3}
                  y={MARGIN.top + 2}
                  className={styles.boundaryLabel}
                >
                  {b.label}
                </text>
              </g>
            );
          })}

          {/* x baseline */}
          <line
            x1={MARGIN.left}
            x2={VB_WIDTH - MARGIN.right}
            y1={MARGIN.top + PLOT_HEIGHT}
            y2={MARGIN.top + PLOT_HEIGHT}
            className={styles.axisLine}
          />

          {/* per-step action labels, colored by the acting seat */}
          {[...stepInfo.values()].map((action) => (
            <text
              key={action.stepIndex}
              className={styles.actionLabel}
              transform={`translate(${xScale(action.stepIndex)}, ${
                MARGIN.top + PLOT_HEIGHT + 8
              }) rotate(-55)`}
              textAnchor="end"
              fill={colorBySeat.get(action.seat)}
              opacity={
                hoverX === null || hoverX === action.stepIndex ? 1 : 0.55
              }
            >
              {shortAction(action.action)}
            </text>
          ))}

          {/* Series in two passes — every line first, every marker on top —
              so no seat's marker fill can erase another seat's line. */}
          {seats.map((s) => {
            const color = colorBySeat.get(s.seat);
            const path = s.points
              .map(
                (p, idx) =>
                  `${idx === 0 ? "M" : "L"} ${xScale(p.x)} ${yScale(p.value)}`,
              )
              .join(" ");
            return (
              s.points.length > 1 && (
                <path
                  key={s.seat}
                  d={path}
                  className={styles.line}
                  stroke={color}
                />
              )
            );
          })}
          {seats.map((s) => {
            const color = colorBySeat.get(s.seat);
            return (
              <g key={s.seat}>
                {s.points.map((p) => (
                  <circle
                    key={p.x}
                    cx={xScale(p.x)}
                    cy={yScale(p.value)}
                    r={hoverX === p.x ? 4.5 : p.kind === "observe" ? 2.8 : 3}
                    // Inline styles: the .marker stylesheet rule would
                    // otherwise override SVG presentation attributes and
                    // turn the observe rings white-on-white.
                    style={
                      p.kind === "decision"
                        ? { fill: color, stroke: "var(--an-panel)", strokeWidth: 1.5 }
                        : { fill: "var(--an-panel)", stroke: color, strokeWidth: 1.6 }
                    }
                    className={styles.marker}
                  />
                ))}
              </g>
            );
          })}

          {/* crosshair */}
          {hoverX !== null && (
            <line
              x1={xScale(hoverX)}
              x2={xScale(hoverX)}
              y1={MARGIN.top}
              y2={MARGIN.top + PLOT_HEIGHT}
              className={styles.crosshair}
            />
          )}
        </svg>

        {hoverAction && (
          <div
            className={styles.tooltip}
            style={{
              left: `${(xScale(hoverAction.stepIndex) / VB_WIDTH) * 100}%`,
            }}
          >
            <div className={styles.tooltipHeader}>
              Step {hoverAction.stepIndex + 1} · {hoverAction.seatName}
            </div>
            <div className={styles.tooltipRow}>
              <span
                className={styles.tooltipKey}
                style={{ background: colorBySeat.get(hoverAction.seat) }}
              />
              <span className={styles.tooltipSeat}>{hoverAction.action}</span>
              <span className={styles.tooltipValue}>
                {typeof hoverAction.memoryCosineDistance === "number"
                  ? hoverAction.memoryCosineDistance.toFixed(3)
                  : "first update"}
              </span>
            </div>
          </div>
        )}

        {hoverObserves && hoverObserves.length > 0 && (
          <div
            className={styles.tooltip}
            style={{ left: `${(xScale(hoverX!) / VB_WIDTH) * 100}%` }}
          >
            <div className={styles.tooltipHeader}>
              Trick {hoverObserves[0].trick + 1} observed (all seats)
            </div>
            {hoverObserves.map((obs) => (
              <div key={obs.seat} className={styles.tooltipRow}>
                <span
                  className={styles.tooltipKey}
                  style={{ background: colorBySeat.get(obs.seat) }}
                />
                <span className={styles.tooltipSeat}>{obs.seatName}</span>
                <span className={styles.tooltipValue}>
                  {typeof obs.memoryCosineDistance === "number"
                    ? obs.memoryCosineDistance.toFixed(3)
                    : "—"}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
