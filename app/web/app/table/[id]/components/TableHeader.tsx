import React, { useState } from "react";
import { Wordmark, MiniCardMark, ds } from "../../../../lib/ds";
import { useBusy } from "../../../../lib/hooks/useBusy";
import styles from "./TableHeader.module.css";

/** Leaving the table: mid-hand, Leave first asks when. */
export interface LeaveControls {
  /** Seated with a hand in play: Leave offers to wait for the hand's end. */
  canLeaveAfterHand: boolean;
  leavingAfterHand: boolean;
  onLeave: () => void;
  onLeaveAfterHand: (on: boolean) => void;
}

/** The host's Close table: mid-hand, it can wait for the hand's end. */
export interface CloseControls {
  /** A hand is being played, so the table can close once it ends. */
  handInPlay: boolean;
  closingAfterHand: boolean;
  onCloseAfterHand: (on: boolean) => void;
  /** Resolves false when the server refused; on success the table's close
   * takes everyone off the page. */
  onCloseTable: () => Promise<boolean>;
}

interface TableHeaderProps {
  roomName: string;
  rulesBadge: string | null;
  stakeBadge: string | null;
  handNumber: number;
  isMobile: boolean;
  leave: LeaveControls;
  /** Host only. */
  close: CloseControls | null;
  onShowScores: () => void;
  onShowCheatSheet: () => void;
  /** Mobile: chat and the menu live behind header buttons. */
  onShowChat: () => void;
  chatUnread: boolean;
  onShowMenu: () => void;
}

/** Drawn, not typed: a text "?" sits off-centre by however the font sets
 * its glyph. */
function HelpGlyph() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M5.9 5.9a2.1 2.1 0 1 1 3.1 1.85c-.6.33-1 .8-1 1.45v.5" />
      <circle cx="8" cy="11.5" r="0.8" />
    </svg>
  );
}

function HelpButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      className={styles.iconBtn}
      onClick={onClick}
      aria-label="Cheat sheet: card power and points"
      title="Cheat sheet"
    >
      <span className={styles.help}>
        <HelpGlyph />
      </span>
    </button>
  );
}

export default function TableHeader(props: TableHeaderProps) {
  if (props.isMobile) return <MobileHeader {...props} />;
  return <DesktopHeader {...props} />;
}

function MobileHeader({
  roomName,
  handNumber,
  leave,
  close,
  onShowCheatSheet,
  onShowChat,
  chatUnread,
  onShowMenu,
}: TableHeaderProps) {
  // Something set to happen when the hand ends is flagged on the menu.
  const armed = leave.leavingAfterHand || !!close?.closingAfterHand;
  return (
    <div className={styles.mob}>
      <div className={styles.mobLeft}>
        <MiniCardMark h={20} />
        <div className={styles.mobRoom}>{roomName}</div>
        <div className={styles.mobMeta}>H{handNumber}</div>
      </div>
      <div className={styles.mobRight}>
        <HelpButton onClick={onShowCheatSheet} />
        <button
          type="button"
          className={`${styles.iconBtn} ${chatUnread ? styles.unread : ""}`}
          onClick={onShowChat}
          aria-label={chatUnread ? "Chat, new messages" : "Chat"}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M3.5 4.5h13v9h-7l-4 3v-3h-2z" />
          </svg>
        </button>
        <button
          type="button"
          className={`${styles.iconBtn} ${armed ? styles.dot : ""}`}
          onClick={onShowMenu}
          aria-label="Table menu"
          aria-haspopup="dialog"
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M3.5 6h13M3.5 10h13M3.5 14h13" />
          </svg>
        </button>
      </div>
    </div>
  );
}

function DesktopHeader({
  roomName,
  rulesBadge,
  stakeBadge,
  handNumber,
  leave,
  close,
  onShowScores,
  onShowCheatSheet,
}: TableHeaderProps) {
  return (
    <header className={styles.desk}>
      <div className={styles.deskLeft}>
        <Wordmark size="sm" />
        <div className={styles.sep} />
        <div className={styles.room}>{roomName}</div>
        {rulesBadge && (
          <span className={`${ds.badge} ${ds.badgeQuiet}`}>{rulesBadge}</span>
        )}
        {stakeBadge && (
          <span className={`${ds.badge} ${ds.badgeAccent} ${styles.stake}`}>
            {stakeBadge}
          </span>
        )}
      </div>
      <div className={styles.deskRight}>
        <div className={styles.stat}>
          <span className={ds.overline}>Hand</span>
          <span className={styles.statNum}>{handNumber}</span>
        </div>
        <div className={styles.links}>
          <HelpButton onClick={onShowCheatSheet} />
          <button type="button" className={styles.link} onClick={onShowScores}>
            Scores
          </button>
          {close && <DesktopClose close={close} />}
          <DesktopLeave leave={leave} />
        </div>
      </div>
    </header>
  );
}

// Mid-hand, Leave first asks when: now forfeits the hand to the AI, after
// the hand waits for it to end. Once asked for, the wait can be undone.
function DesktopLeave({ leave }: { leave: LeaveControls }) {
  const [choosing, setChoosing] = useState(false);
  if (leave.canLeaveAfterHand && leave.leavingAfterHand) {
    return (
      <span className={styles.choiceRow}>
        <span className={styles.armed}>Leaving after this hand</span>
        <button
          type="button"
          className={styles.link}
          onClick={() => leave.onLeaveAfterHand(false)}
        >
          Undo
        </button>
      </span>
    );
  }
  if (leave.canLeaveAfterHand && choosing) {
    return (
      <span className={styles.choiceRow}>
        <button
          type="button"
          className={styles.link}
          onClick={() => {
            setChoosing(false);
            leave.onLeaveAfterHand(true);
          }}
        >
          After this hand
        </button>
        <button
          type="button"
          className={`${styles.link} ${styles.danger}`}
          onClick={leave.onLeave}
        >
          Leave now
        </button>
        <button
          type="button"
          className={styles.link}
          onClick={() => setChoosing(false)}
        >
          Cancel
        </button>
      </span>
    );
  }
  return (
    <button
      type="button"
      className={`${styles.link} ${styles.danger}`}
      onClick={
        leave.canLeaveAfterHand ? () => setChoosing(true) : leave.onLeave
      }
    >
      Leave
    </button>
  );
}

function DesktopClose({ close }: { close: CloseControls }) {
  const [confirming, setConfirming] = useState(false);
  const closing = useBusy();
  if (close.closingAfterHand) {
    return (
      <span className={styles.choiceRow}>
        <span className={styles.armed}>Table ends after this hand</span>
        <button
          type="button"
          className={styles.link}
          onClick={() => close.onCloseAfterHand(false)}
        >
          Undo
        </button>
      </span>
    );
  }
  if (confirming) {
    return (
      <span className={styles.choiceRow}>
        {close.handInPlay && (
          <button
            type="button"
            className={styles.link}
            onClick={() => {
              setConfirming(false);
              close.onCloseAfterHand(true);
            }}
          >
            End after this hand
          </button>
        )}
        <button
          type="button"
          className={`${styles.link} ${styles.danger} ${closing.busy ? ds.btnBusy : ""}`}
          onClick={() => closing.run(close.onCloseTable, { hold: true })}
          disabled={closing.busy}
          aria-busy={closing.busy}
        >
          {close.handInPlay ? "Close now" : "Confirm close"}
        </button>
        <button
          type="button"
          className={styles.link}
          onClick={() => setConfirming(false)}
        >
          Cancel
        </button>
      </span>
    );
  }
  return (
    <button
      type="button"
      className={`${styles.link} ${styles.danger}`}
      onClick={() => setConfirming(true)}
    >
      Close table
    </button>
  );
}
