import React, { useEffect, useRef, useState } from "react";
import { ds } from "../../../../lib/ds";
import type { CloseControls, LeaveControls } from "./TableHeader";
import styles from "./TableMenu.module.css";

interface TableMenuProps {
  roomName: string;
  rulesBadge: string | null;
  stakeBadge: string | null;
  onShowScores: () => void;
  leave: LeaveControls;
  /** Host only. */
  close: CloseControls | null;
  onClose: () => void;
}

const ICONS = {
  scores: "M4 16V9M10 16V4M16 16v-5",
  leave: "M8 4H4v12h4M12 6l4 4-4 4M16 10H8",
  close: "M7.5 7.5l5 5M12.5 7.5l-5 5",
};

function Icon({ d, ring }: { d: string; ring?: boolean }) {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      {ring && <circle cx="10" cy="10" r="7" />}
      <path d={d} />
    </svg>
  );
}

/** Mobile table menu: a bottom sheet holding what the action bar and the
 * header's Leave used to: Scores, Leave, and the host's Close table, each
 * of which can wait for the hand to end. */
export default function TableMenu({
  roomName,
  rulesBadge,
  stakeBadge,
  onShowScores,
  leave,
  close,
  onClose,
}: TableMenuProps) {
  const [open, setOpen] = useState<"leave" | "close" | null>(null);
  const sheetRef = useRef<HTMLDivElement>(null);
  // Read through a ref so the page's every re-render (each state message)
  // doesn't re-run the effect and pull focus back to the first row.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    sheetRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  let leaveRows: React.ReactNode;
  if (leave.canLeaveAfterHand && leave.leavingAfterHand) {
    leaveRows = (
      <div className={styles.row}>
        <Icon d={ICONS.leave} />
        <span className={styles.armed}>Leaving after this hand</span>
        <button
          type="button"
          className={styles.inline}
          onClick={() => leave.onLeaveAfterHand(false)}
        >
          Undo
        </button>
      </div>
    );
  } else if (open === "leave") {
    leaveRows = (
      <>
        <button
          type="button"
          className={styles.row}
          onClick={() => {
            setOpen(null);
            leave.onLeaveAfterHand(true);
          }}
        >
          <Icon d={ICONS.leave} />
          <span className={styles.grow}>Leave after this hand</span>
        </button>
        <button
          type="button"
          className={`${styles.row} ${styles.danger} ${styles.sub}`}
          onClick={leave.onLeave}
        >
          <span className={styles.grow}>Leave now</span>
          <span className={styles.note}>the AI plays your hand</span>
        </button>
      </>
    );
  } else {
    leaveRows = (
      <button
        type="button"
        className={styles.row}
        onClick={
          leave.canLeaveAfterHand ? () => setOpen("leave") : leave.onLeave
        }
      >
        <Icon d={ICONS.leave} />
        <span className={styles.grow}>Leave table</span>
      </button>
    );
  }

  let closeRows: React.ReactNode = null;
  if (close?.closingAfterHand) {
    closeRows = (
      <div className={`${styles.row} ${styles.danger}`}>
        <Icon d={ICONS.close} ring />
        <span className={styles.armed}>Table ends after this hand</span>
        <button
          type="button"
          className={styles.inline}
          onClick={() => close.onCloseAfterHand(false)}
        >
          Undo
        </button>
      </div>
    );
  } else if (close && open === "close") {
    closeRows = (
      <>
        {close.handInPlay && (
          <button
            type="button"
            className={`${styles.row} ${styles.danger}`}
            onClick={() => {
              setOpen(null);
              close.onCloseAfterHand(true);
            }}
          >
            <Icon d={ICONS.close} ring />
            <span className={styles.grow}>End after this hand</span>
          </button>
        )}
        <button
          type="button"
          className={`${styles.row} ${styles.danger} ${close.handInPlay ? styles.sub : ""}`}
          onClick={close.onCloseTable}
        >
          {!close.handInPlay && <Icon d={ICONS.close} ring />}
          <span className={styles.grow}>
            {close.handInPlay ? "Close now" : "Confirm close"}
          </span>
        </button>
      </>
    );
  } else if (close) {
    closeRows = (
      <button
        type="button"
        className={`${styles.row} ${styles.danger}`}
        onClick={() => setOpen("close")}
      >
        <Icon d={ICONS.close} ring />
        <span className={styles.grow}>Close table</span>
        <span className={styles.note}>for everyone</span>
      </button>
    );
  }

  return (
    <>
      <div className={styles.scrim} onClick={onClose} aria-hidden="true" />
      <div
        ref={sheetRef}
        className={styles.sheet}
        role="dialog"
        aria-modal="true"
        aria-label="Table menu"
      >
        <div className={styles.grab} aria-hidden="true" />
        <div className={styles.head}>
          <div className={styles.titleRow}>
            <span className={styles.title}>{roomName}</span>
            {stakeBadge && (
              <span className={`${ds.badge} ${ds.badgeAccent}`}>
                {stakeBadge}
              </span>
            )}
          </div>
          {rulesBadge && (
            <span className={`${ds.badge} ${ds.badgeQuiet} ${styles.rules}`}>
              {rulesBadge}
            </span>
          )}
        </div>
        <button
          type="button"
          className={styles.row}
          onClick={() => {
            onClose();
            onShowScores();
          }}
        >
          <Icon d={ICONS.scores} />
          <span className={styles.grow}>Scores</span>
        </button>
        <div className={styles.rule} />
        {leaveRows}
        {closeRows}
        <button
          type="button"
          className={`${styles.row} ${styles.cancel}`}
          onClick={onClose}
        >
          <span className={styles.grow}>Back to the table</span>
        </button>
      </div>
    </>
  );
}
