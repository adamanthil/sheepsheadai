import React from "react";
import type { FinalState, TableView } from "../../../../lib/types";
import { ds } from "../../../../lib/ds";
import HandSummary, { SeatGrid } from "./HandSummary";
import { scoreBlocks } from "../lib/tableSelectors";
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
  const blocks = scoreBlocks(table);
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
            {/* One block per set of players: each restarts its running
                totals from 0 under its own names. */}
            <table className={styles.scoresTable}>
              {blocks.map((block, b) => (
                <tbody key={b} className={styles.block}>
                  <tr>
                    <th scope="col" className={styles.thLeft}>
                      Hand
                    </th>
                    {block.columns.map((c) => (
                      <th key={c.id} scope="col" className={styles.thRight}>
                        <span className={styles.thName} title={c.name}>
                          {c.name}
                        </span>
                      </th>
                    ))}
                  </tr>
                  {block.rows.map((r, idx) => (
                    <tr
                      key={r.hand}
                      className={[
                        r.set && styles.set,
                        b === blocks.length - 1 &&
                          idx === block.rows.length - 1 &&
                          styles.latest,
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
                      {r.cells.map(({ delta, total }, i) => (
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
              ))}
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
