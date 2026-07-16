import React, { useMemo, useRef, useState } from "react";
import { AnalyzeActionDetail } from "../../lib/analyzeTypes";
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
const VB_HEIGHT = 180;
const MARGIN = { top: 12, right: 16, bottom: 26, left: 42 };
const PLOT_WIDTH = VB_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = VB_HEIGHT - MARGIN.top - MARGIN.bottom;

export default function MemoryDriftChart({ trace }: MemoryDriftChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverStep, setHoverStep] = useState<number | null>(null);

  const { seats, allSteps, minStep, maxStep, maxValue } = useMemo(() => {
    const bySeat = new Map<number, { seat: number; seatName: string; points: Map<number, number> }>();
    let minS = Infinity;
    let maxS = -Infinity;
    let maxV = 0;

    for (const action of trace) {
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
      bySeat.get(action.seat)!.points.set(
        action.stepIndex,
        action.memoryCosineDistance,
      );
      if (action.memoryCosineDistance > maxV) maxV = action.memoryCosineDistance;
    }

    const seatList = [...bySeat.values()].sort((a, b) => a.seat - b.seat);
    const stepSet = new Set<number>();
    seatList.forEach((s) => s.points.forEach((_v, step) => stepSet.add(step)));
    const stepsSorted = [...stepSet].sort((a, b) => a - b);

    return {
      seats: seatList,
      allSteps: stepsSorted,
      minStep: Number.isFinite(minS) ? minS : 0,
      maxStep: Number.isFinite(maxS) ? maxS : 1,
      maxValue: maxV,
    };
  }, [trace]);

  const hasData = seats.length > 0 && allSteps.length > 0;

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
    const approxStep = minStep + ((relX - MARGIN.left) / PLOT_WIDTH) * span;
    let nearest = allSteps[0];
    let bestDist = Infinity;
    for (const s of allSteps) {
      const d = Math.abs(s - approxStep);
      if (d < bestDist) {
        bestDist = d;
        nearest = s;
      }
    }
    setHoverStep(nearest);
  };

  if (!hasData) return null;

  const yTicks = [0, yMax / 2, yMax];
  const xTickSteps = [minStep, Math.round((minStep + maxStep) / 2), maxStep];

  return (
    <div className={styles.memoryDriftChart}>
      <div className={styles.header}>
        <div className={styles.title}>Memory Drift</div>
        <div className={styles.legend}>
          {seats.map((s, i) => (
            <div key={s.seat} className={styles.legendItem}>
              <span
                className={styles.legendSwatch}
                style={{ background: SEAT_COLORS[i % SEAT_COLORS.length] }}
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

          {/* x baseline + tick labels */}
          <line
            x1={MARGIN.left}
            x2={VB_WIDTH - MARGIN.right}
            y1={MARGIN.top + PLOT_HEIGHT}
            y2={MARGIN.top + PLOT_HEIGHT}
            className={styles.axisLine}
          />
          {xTickSteps.map((t, i) => (
            <text
              key={i}
              x={xScale(t)}
              y={VB_HEIGHT - 6}
              className={styles.axisLabel}
              textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
            >
              step {t + 1}
            </text>
          ))}

          {/* series */}
          {seats.map((s, i) => {
            const color = SEAT_COLORS[i % SEAT_COLORS.length];
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

        {hoverStep !== null && (
          <div
            className={styles.tooltip}
            style={{
              left: `${(xScale(hoverStep) / VB_WIDTH) * 100}%`,
            }}
          >
            <div className={styles.tooltipHeader}>Step {hoverStep + 1}</div>
            {seats.map((s, i) => {
              const value = s.points.get(hoverStep);
              return (
                <div key={s.seat} className={styles.tooltipRow}>
                  <span
                    className={styles.tooltipKey}
                    style={{ background: SEAT_COLORS[i % SEAT_COLORS.length] }}
                  />
                  <span className={styles.tooltipSeat}>{s.seatName}</span>
                  <span className={styles.tooltipValue}>
                    {typeof value === "number" ? value.toFixed(3) : "—"}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
