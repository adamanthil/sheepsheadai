import React from "react";
import { SeatAvatar, ds } from "../../../../lib/ds";
import type { TableView } from "../../../../lib/types";

interface ScoreboardProps {
  table: TableView;
  yourSeat: number;
  compact?: boolean;
}

/** Running totals per seat, sourced from table.runningBySeat. */
export default function Scoreboard({
  table,
  yourSeat,
  compact,
}: ScoreboardProps) {
  const running = (table.runningBySeat || {}) as Record<string, number>;
  const seatIsAI = (table.seatIsAI || {}) as Record<string, boolean>;
  const handCount = table.resultsHistory?.length ?? 0;

  const rows = [1, 2, 3, 4, 5]
    .map((seat) => ({
      seat,
      name: table.seats?.[String(seat)] || `Seat ${seat}`,
      pts: running[String(seat)] ?? 0,
      isAI: !!seatIsAI[String(seat)],
      you: seat === yourSeat,
      seated: !!table.seats?.[String(seat)],
    }))
    .filter((r) => r.seated);

  return (
    <div>
      <div
        className={ds.overline}
        style={{ marginBottom: 10, fontSize: compact ? 9 : 11 }}
      >
        Running Scores
        {handCount ? ` · ${handCount} hand${handCount === 1 ? "" : "s"}` : ""}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map((r) => (
          <div
            key={r.seat}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <SeatAvatar
                name={r.name}
                isAI={r.isAI}
                tone={r.you ? "you" : "default"}
                size={compact ? 24 : 26}
              />
              <div
                style={{
                  fontFamily: "var(--font-display)",
                  fontSize: compact ? 16 : 16,
                  color: r.you ? "var(--ink)" : "var(--ink-soft)",
                  fontWeight: r.you ? 500 : 400,
                }}
              >
                {r.name}
              </div>
              {r.you && (
                <span className={ds.badge} style={{ fontSize: 9 }}>
                  You
                </span>
              )}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontVariantNumeric: "tabular-nums",
                fontSize: 16,
                fontWeight: 500,
                color:
                  r.pts > 0
                    ? "var(--accent)"
                    : r.pts < 0
                      ? "var(--accent-2)"
                      : "var(--muted)",
              }}
            >
              {r.pts > 0 ? "+" : ""}
              {r.pts}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
