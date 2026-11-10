import React, { useState } from "react";
import { ds } from "../../../../lib/ds";
import { useBusy } from "../../../../lib/hooks/useBusy";
import type { TableView } from "../../../../lib/types";
import styles from "./HandOverBar.module.css";

interface HandOverBarProps {
  table: TableView;
  /** Resolves false when the server refused the deal. */
  onNextHand: () => Promise<boolean>;
  onShowScores: () => void;
  isHost: boolean;
  onCloseTable: () => Promise<boolean>;
  onLeave: () => void;
  /** The last-trick button, set on mobile, where the status line that holds
   * it in play is gone between hands. */
  lastTrick?: React.ReactNode;
}

/** What everyone but the host waits on between hands. */
export function waitingForDeal(table: TableView): string {
  if (table.closingAfterHand) return "Closing the table…";
  return `Waiting for ${table.host || "the host"} to deal…`;
}

/** What to do between hands, in the hand's place under the table: the
 * result itself is on the ring, and nothing covers the chat. */
export default function HandOverBar({
  table,
  onNextHand,
  onShowScores,
  isHost,
  onCloseTable,
  onLeave,
  lastTrick,
}: HandOverBarProps) {
  const [confirmClose, setConfirmClose] = useState(false);
  // Both are held busy on success: the new deal (or the close) replaces
  // this bar when the server's state arrives.
  const dealing = useBusy();
  const closing = useBusy();

  return (
    <div className={styles.bar}>
      <div className={styles.primary}>
        {lastTrick && <span className={styles.lastTrick}>{lastTrick}</span>}
        {/* Only the host can deal the next hand (the server refuses anyone
            else), and not at a table closing after this hand. */}
        {isHost && !table.closingAfterHand ? (
          <button
            className={`${ds.btn} ${ds.btnAccent} ${dealing.busy ? ds.btnBusy : ""}`}
            onClick={() => dealing.run(onNextHand, { hold: true })}
            disabled={dealing.busy}
            aria-busy={dealing.busy}
          >
            Next hand →
          </button>
        ) : (
          <span className={styles.waitingHost}>{waitingForDeal(table)}</span>
        )}
      </div>
      <div className={styles.secondary}>
        {/* Hidden while the close is confirmed, to keep the buttons on one
            line on a phone: cancelling brings it back. */}
        {!confirmClose && (
          <button
            className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
            onClick={onShowScores}
          >
            Scores
          </button>
        )}
        {isHost ? (
          confirmClose ? (
            <>
              <button
                className={`${ds.btn} ${ds.btnSm} ${closing.busy ? ds.btnBusy : ""}`}
                onClick={() => closing.run(onCloseTable, { hold: true })}
                disabled={closing.busy}
                aria-busy={closing.busy}
              >
                Confirm close
              </button>
              <button
                className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                onClick={() => setConfirmClose(false)}
              >
                Cancel
              </button>
            </>
          ) : (
            <button
              className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
              onClick={() => setConfirmClose(true)}
            >
              Close table
            </button>
          )
        ) : (
          <button
            className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
            onClick={onLeave}
          >
            Leave
          </button>
        )}
      </div>
    </div>
  );
}
