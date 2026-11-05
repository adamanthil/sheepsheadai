import React, { useState } from "react";
import { ds } from "../../../../lib/ds";
import { useBusy } from "../../../../lib/hooks/useBusy";
import HandSummary, { SeatGrid } from "./HandSummary";
import type { FinalState, TableView } from "../../../../lib/types";
import styles from "./HandSummaryBanner.module.css";

interface HandSummaryBannerProps {
  final: FinalState;
  table: TableView;
  /** Resolves false when the server refused the deal. */
  onNextHand: () => Promise<boolean>;
  onShowScores: () => void;
  isHost: boolean;
  onCloseTable: () => Promise<boolean>;
  onLeave: () => void;
}

export default function HandSummaryBanner({
  final,
  table,
  onNextHand,
  onShowScores,
  isHost,
  onCloseTable,
  onLeave,
}: HandSummaryBannerProps) {
  const [confirmClose, setConfirmClose] = useState(false);
  // Both are held busy on success: the new deal (or the close) replaces
  // this banner when the server's state arrives.
  const dealing = useBusy();
  const closing = useBusy();
  const isLeaster = final.mode === "leaster";

  return (
    <div className={styles.overlay}>
      <div className={styles.banner}>
        <div className={styles.header}>
          {isLeaster ? "Hand Summary · Leaster" : "Hand Summary"}
        </div>
        <HandSummary final={final} table={table} />
        <SeatGrid
          table={table}
          values={final.scores}
          label="Scores by player"
        />
        {isLeaster && (
          <SeatGrid
            table={table}
            values={final.points_taken}
            label="Points taken"
          />
        )}
        <div className={styles.buttons}>
          {/* Only the host can deal the next hand (the server refuses anyone
              else), and not at a table closing after this hand. */}
          {table.closingAfterHand ? (
            <span className={styles.waitingHost}>Closing the table…</span>
          ) : isHost ? (
            <button
              className={`${ds.btn} ${ds.btnAccent} ${dealing.busy ? ds.btnBusy : ""}`}
              onClick={() => dealing.run(onNextHand, { hold: true })}
              disabled={dealing.busy}
              aria-busy={dealing.busy}
            >
              Next hand →
            </button>
          ) : (
            <span className={styles.waitingHost}>
              {table.host
                ? `Waiting for ${table.host} to deal the next hand…`
                : "Waiting for the host to deal the next hand…"}
            </span>
          )}
          <button className={`${ds.btn} ${ds.btnGhost}`} onClick={onShowScores}>
            Show scores
          </button>
          {isHost ? (
            confirmClose ? (
              <>
                <button
                  className={`${ds.btn} ${closing.busy ? ds.btnBusy : ""}`}
                  onClick={() => closing.run(onCloseTable, { hold: true })}
                  disabled={closing.busy}
                  aria-busy={closing.busy}
                >
                  Confirm close
                </button>
                <button
                  className={`${ds.btn} ${ds.btnGhost}`}
                  onClick={() => setConfirmClose(false)}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                className={`${ds.btn} ${ds.btnGhost}`}
                onClick={() => setConfirmClose(true)}
              >
                Close table
              </button>
            )
          ) : (
            <button className={`${ds.btn} ${ds.btnGhost}`} onClick={onLeave}>
              Leave
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
