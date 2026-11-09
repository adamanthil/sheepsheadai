import React from "react";
import type { FinalState, TableView, HandResult } from "../../../../lib/types";
import { ds } from "../../../../lib/ds";
import HandSummary, { SeatGrid } from "./HandSummary";
import styles from "./ScoresOverlay.module.css";

interface ScoresOverlayProps {
  onClose: () => void;
  table: TableView;
  title?: string;
  /** Replace the Close button, for a card the player can't dismiss. */
  actions?: React.ReactNode;
  /** The hand just finished, summarized above the totals: the table has
   * ended, so the game-over banner that would show it never appears. */
  lastHand?: FinalState;
}

export default function ScoresOverlay({
  onClose,
  table,
  title = "Running totals",
  actions,
  lastHand,
}: ScoresOverlayProps) {
  const history: HandResult[] = table?.resultsHistory || [];
  const seats = table?.seats || {};

  const rows = history.map((h, idx) => ({
    hand: h.hand || idx + 1,
    multiplier: h.multiplier || 1,
    bySeat: h.bySeat || {},
    sum: h.sum || 0,
    picker: h.picker ?? null,
  }));

  const initialOrder: string[] = (table?.initialSeatOrder || []).map((x) =>
    String(x),
  );
  const ids = table?.seatOccupants || {};

  const labelsById: Record<string, string> = {};
  for (let i = 1; i <= 5; i++) {
    const occ = String(ids[String(i)] || `seat-${i}`);
    labelsById[occ] = seats[String(i)] || `Seat ${i}`;
  }

  const columns: Array<{ id: string; label: string }> = (
    initialOrder.length === 5 ? initialOrder : Object.keys(labelsById)
  )
    // Someone no longer seated keeps the name they were dealt in under.
    .map((id: string) => ({
      id,
      label: labelsById[id] || table?.initialNames?.[id] || id,
    }));

  const scoreFor = (
    row: { bySeat?: Record<string, { id: string; score: number }> },
    id: string,
  ) => {
    const entries = row.bySeat || {};
    for (const key of Object.keys(entries)) {
      const v = entries[key];
      if (v && String(v.id) === String(id)) return v.score || 0;
    }
    return 0;
  };

  // The picking team lost: the picker's score is negative.
  const isSet = (row: (typeof rows)[number]) =>
    row.picker !== null && scoreFor(row, row.picker) < 0;

  // Each row shows the totals through that hand; the hand's own score
  // stays in the cell's tooltip.
  const runningById: Record<string, number> = {};
  const cumulative = rows.map((r) =>
    columns.map((c) => {
      const delta = scoreFor(r, c.id);
      runningById[c.id] = (runningById[c.id] ?? 0) + delta;
      return { delta, total: runningById[c.id] };
    }),
  );
  const signed = (n: number) => (n > 0 ? `+${n}` : String(n));

  return (
    <div className={styles.scoresOverlay}>
      <div className={styles.scoresBox}>
        <div className={styles.scoresHeader}>
          <div>
            <strong>{title}</strong>
          </div>
          <div className={`${styles.mlAuto} ${styles.actions}`}>
            {actions ?? (
              <button
                className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                onClick={onClose}
              >
                Close
              </button>
            )}
          </div>
        </div>
        <div className={styles.scoresBody}>
          {lastHand && (
            <div className={styles.lastHand}>
              <div className={styles.lastHandLabel}>
                {lastHand.mode === "leaster"
                  ? "Last hand · Leaster"
                  : "Last hand"}
              </div>
              <HandSummary final={lastHand} table={table} />
              {lastHand.mode === "leaster" && (
                <SeatGrid
                  table={table}
                  values={lastHand.points_taken}
                  label="Points taken"
                />
              )}
            </div>
          )}
          <div>
            <table className={styles.scoresTable}>
              <thead>
                <tr>
                  <th className={styles.thLeft}>Hand</th>
                  {columns.map((c, idx) => (
                    <th key={idx} className={styles.thRight}>
                      <span className={styles.thName} title={c.label}>
                        {c.label}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => (
                  <tr
                    key={idx}
                    className={[
                      isSet(r) && styles.set,
                      idx === rows.length - 1 && styles.latest,
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <td className={styles.tdHand}>
                      {r.hand}
                      {r.multiplier > 1 && (
                        <span
                          className={styles.handStake}
                          title={`Doubler — scored at ${r.multiplier}x`}
                        >
                          {r.multiplier}×
                        </span>
                      )}
                    </td>
                    {cumulative[idx].map(({ delta, total }, i) => (
                      <td
                        key={i}
                        className={styles.tdRight}
                        title={`This hand: ${signed(delta)}`}
                      >
                        {total}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
