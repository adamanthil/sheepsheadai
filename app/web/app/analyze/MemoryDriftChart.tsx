import React, { useMemo, useRef, useState } from "react";
import { AnalyzeActionDetail } from "../../lib/analyzeTypes";
import Term from "./TermHelp";
import styles from "./MemoryDriftChart.module.css";

interface MemoryDriftChartProps {
  trace: AnalyzeActionDetail[];
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

const VB_WIDTH = 640;
const VB_HEIGHT = 214;
const MARGIN = { top: 18, right: 16, bottom: 56, left: 42 };
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

interface Boundary {
  beforeStep: number; // the boundary sits just before this stepIndex
  label: string;
}

export default function MemoryDriftChart({ trace }: MemoryDriftChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverStep, setHoverStep] = useState<number | null>(null);

  const { seats, colorBySeat, stepInfo, boundaries, minStep, maxStep, maxValue } =
    useMemo(() => {
      const bySeat = new Map<
        number,
        { seat: number; seatName: string; points: Map<number, number> }
      >();
      const info = new Map<number, AnalyzeActionDetail>();
      let minS = Infinity;
      let maxS = -Infinity;
      let maxV = 0;

      for (const action of trace) {
        info.set(action.stepIndex, action);
        if (action.stepIndex < minS) minS = action.stepIndex;
        if (action.stepIndex > maxS) maxS = action.stepIndex;
        if (typeof action.memoryCosineDistance !== "number") continue;
        if (!bySeat.has(action.seat)) {
          bySeat.set(action.seat, {
            seat: action.seat,
            seatName: action.seatName,
            points: new Map(),
          });
        }
        bySeat
          .get(action.seat)!
          .points.set(action.stepIndex, action.memoryCosineDistance);
        if (action.memoryCosineDistance > maxV) {
          maxV = action.memoryCosineDistance;
        }
      }

      // Stable seat -> color mapping over ALL acting seats (not only the
      // ones with drift points) so labels and lines agree.
      const actingSeats = [...new Set(trace.map((a) => a.seat))].sort(
        (a, b) => a - b,
      );
      const colors = new Map<number, string>();
      actingSeats.forEach((seat, i) =>
        colors.set(seat, SEAT_COLORS[i % SEAT_COLORS.length]),
      );

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

      return {
        seats: [...bySeat.values()].sort((a, b) => a.seat - b.seat),
        colorBySeat: colors,
        stepInfo: info,
        boundaries: bounds,
        minStep: Number.isFinite(minS) ? minS : 0,
        maxStep: Number.isFinite(maxS) ? maxS : 1,
        maxValue: maxV,
      };
    }, [trace]);

  const hasData = seats.length > 0;

  const xScale = (step: number) => {
    const span = maxStep - minStep || 1;
    return MARGIN.left + ((step - minStep) / span) * PLOT_WIDTH;
  };
  const yMax = maxValue > 0 ? maxValue * 1.15 : 1;
  const yScale = (value: number) => {
    return MARGIN.top + PLOT_HEIGHT - (value / yMax) * PLOT_HEIGHT;
  };

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!hasData || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * VB_WIDTH;
    const span = maxStep - minStep || 1;
    const approx = Math.round(
      minStep + ((relX - MARGIN.left) / PLOT_WIDTH) * span,
    );
    const clamped = Math.max(minStep, Math.min(maxStep, approx));
    setHoverStep(stepInfo.has(clamped) ? clamped : null);
  };

  if (!hasData) return null;

  const yTicks = [0, yMax / 2, yMax];
  const hoverAction = hoverStep !== null ? stepInfo.get(hoverStep) : undefined;

  return (
    <div className={styles.memoryDriftChart}>
      <div className={styles.header}>
        <div className={styles.title}>
          <Term
            label="Memory Drift"
            wiki="https://en.wikipedia.org/wiki/Gated_recurrent_unit"
          >
            Each seat carries a memory vector (256 dimensions) that a
            recurrent unit (GRU) updates whenever that seat receives new
            information: once at each of its own decisions — the encoder
            folds the current observation (its hand, the trick so far, and
            the public flags, which reflect everything other seats did since
            its last update) into the memory — and once after every
            completed trick (those updates are not plotted). Each point is
            the cosine distance between a seat&rsquo;s memory before and
            after its decision update: near 0 = the observation barely
            changed the model&rsquo;s internal state; higher = a substantial
            revision. Vertical lines mark phase changes and trick
            boundaries, and each step is labeled with the action taken.
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
        </div>
      </div>

      <div className={styles.chartWrap}>
        <svg
          ref={svgRef}
          className={styles.svg}
          viewBox={`0 0 ${VB_WIDTH} ${VB_HEIGHT}`}
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverStep(null)}
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
            const x =
              (xScale(b.beforeStep - 1) + xScale(b.beforeStep)) / 2;
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
                hoverStep === null || hoverStep === action.stepIndex ? 1 : 0.55
              }
            >
              {shortAction(action.action)}
            </text>
          ))}

          {/* series */}
          {seats.map((s) => {
            const color = colorBySeat.get(s.seat);
            const pts = [...s.points.entries()].sort((a, b) => a[0] - b[0]);
            const path = pts
              .map(
                ([step, value], idx) =>
                  `${idx === 0 ? "M" : "L"} ${xScale(step)} ${yScale(value)}`,
              )
              .join(" ");
            return (
              <g key={s.seat}>
                {pts.length > 1 && (
                  <path d={path} className={styles.line} stroke={color} />
                )}
                {pts.map(([step, value]) => (
                  <circle
                    key={step}
                    cx={xScale(step)}
                    cy={yScale(value)}
                    r={hoverStep === step ? 4.5 : 3}
                    fill={color}
                    className={styles.marker}
                  />
                ))}
              </g>
            );
          })}

          {/* crosshair */}
          {hoverStep !== null && (
            <line
              x1={xScale(hoverStep)}
              x2={xScale(hoverStep)}
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
      </div>
    </div>
  );
}
