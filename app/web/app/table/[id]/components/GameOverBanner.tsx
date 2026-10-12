import React, { useState } from "react";
import { ds } from "../../../../lib/ds";
import HandSummary, { SeatGrid } from "./HandSummary";
import type { FinalState, TableView } from "../../../../lib/types";
import styles from "./GameOverBanner.module.css";

interface GameOverBannerProps {
  final: FinalState;
  table: TableView;
  onRedeal: () => void;
  onShowScores: () => void;
  isHost: boolean;
  onCloseTable: () => void;
  onLeave: () => void;
}

export default function GameOverBanner({
  final,
  table,
  onRedeal,
  onShowScores,
  isHost,
  onCloseTable,
  onLeave,
}: GameOverBannerProps) {
  const [confirmClose, setConfirmClose] = useState(false);
  const isLeaster = final.mode === "leaster";

  return (
    <div className={styles.overlay}>
      <div className={styles.banner}>
        <div className={styles.header}>
          {isLeaster ? "Game Over · Leaster" : "Game Over"}
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
          {/* Only the host can redeal (the server refuses anyone else), and
              not at a table closing after this hand. */}
          {table.closingAfterHand ? (
            <span className={styles.waitingHost}>Closing the table…</span>
          ) : isHost ? (
            <button className={`${ds.btn} ${ds.btnAccent}`} onClick={onRedeal}>
              Redeal →
            </button>
          ) : (
            <span className={styles.waitingHost}>
              {table.host
                ? `Waiting for ${table.host} to redeal…`
                : "Waiting for the host to redeal…"}
            </span>
          )}
          <button className={`${ds.btn} ${ds.btnGhost}`} onClick={onShowScores}>
            Show scores
          </button>
          {isHost ? (
            confirmClose ? (
              <>
                <button className={ds.btn} onClick={onCloseTable}>
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
