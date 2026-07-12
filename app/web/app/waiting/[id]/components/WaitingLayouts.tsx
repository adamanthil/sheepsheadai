import React from "react";
import { ds, Wordmark } from "../../../../lib/ds";
import SeatCard from "./SeatCard";
import styles from "../page.module.css";

export interface WaitingLayoutProps {
  roomName: string;
  shortId: string;
  hostName: string | null;
  seatItems: Array<{ seat: number; name: string | null; isAI: boolean }>;
  humanCount: number;
  seatedCount: number;
  emptyCount: number;
  error: string | null;
  isHost: boolean;
  confirmClose: boolean;
  setConfirmClose: (v: boolean) => void;
  chooseSeat: (seat: number) => void;
  fillAI: () => void;
  startGame: () => void;
  closeTable: () => void;
  onLeave: () => void;
  chat: React.ReactNode;
  rules: React.ReactNode;
}

function leaveLinkFor(onLeave: () => void) {
  const leaveLink = (label: string, className: string) => (
    <a
      className={className}
      href="/"
      onClick={(e) => {
        e.preventDefault();
        onLeave();
      }}
    >
      {label}
    </a>
  );
  return leaveLink;
}

export function MobileWaitingLayout(props: WaitingLayoutProps) {
  const {
    roomName,
    shortId,
    hostName,
    seatItems,
    seatedCount,
    error,
    isHost,
    confirmClose,
    setConfirmClose,
    chooseSeat,
    fillAI,
    startGame,
    closeTable,
    chat,
    rules,
  } = props;
  const leaveLink = leaveLinkFor(props.onLeave);
  return (
        /* ---------- MOBILE ---------- */
        <div className={styles.mob}>
          <div className={styles.mobScroll}>
            <div className={styles.mobHead}>
              <div className={ds.overline} style={{ fontSize: 9 }}>
                Waiting Room · {shortId}
              </div>
              {leaveLink("Leave", styles.leaveLink)}
            </div>
            <div className={styles.mobTitleRow}>
              <h1 className={styles.mobTitle}>{roomName}</h1>
              <div className={styles.mobSeated}>
                <div className={styles.mobSeatedNum}>
                  {seatedCount}
                  <span className={styles.statOf}>/5</span>
                </div>
                <div
                  className={ds.overline}
                  style={{ fontSize: 9, marginTop: 2 }}
                >
                  seated
                </div>
              </div>
            </div>

            {error && <div className={styles.error}>{error}</div>}

            <div>
              <div
                className={ds.overline}
                style={{ fontSize: 9, marginBottom: 8 }}
              >
                Seats{hostName ? ` · Host: ${hostName}` : ""}
              </div>
              <div className={styles.mobSeatList}>
                {seatItems.map((s) => (
                  <SeatCard
                    key={s.seat}
                    seat={s}
                    variant="row"
                    onTake={chooseSeat}
                  />
                ))}
              </div>
            </div>

            <div>
              <div
                className={ds.overline}
                style={{ fontSize: 9, marginBottom: 8 }}
              >
                Game Mode
              </div>
              {rules}
            </div>

            <div>
              <div
                className={ds.overline}
                style={{ fontSize: 9, marginBottom: 8 }}
              >
                Table Chat
              </div>
              <div className={styles.mobChatWrap}>{chat}</div>
            </div>
          </div>

          <div className={styles.mobActionBar}>
            {isHost ? (
              <>
                <button
                  className={`${ds.btn} ${ds.btnAccent}`}
                  onClick={startGame}
                >
                  Deal cards →
                </button>
                <div className={styles.mobActionRow}>
                  <button
                    className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                    style={{ flex: 1 }}
                    onClick={fillAI}
                  >
                    Fill with AI
                  </button>
                  {confirmClose ? (
                    <>
                      <button
                        className={`${ds.btn} ${ds.btnSm}`}
                        style={{ flex: 1 }}
                        onClick={closeTable}
                      >
                        Confirm
                      </button>
                      <button
                        className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                        style={{ flex: 1 }}
                        onClick={() => setConfirmClose(false)}
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <button
                      className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.dangerLink}`}
                      style={{ flex: 1 }}
                      onClick={() => setConfirmClose(true)}
                    >
                      Close
                    </button>
                  )}
                </div>
              </>
            ) : (
              <div className={styles.waitHint} style={{ textAlign: "center" }}>
                Waiting for the host to deal…
              </div>
            )}
          </div>
        </div>
  );
}

export function DesktopWaitingLayout(props: WaitingLayoutProps) {
  const {
    roomName,
    shortId,
    hostName,
    seatItems,
    humanCount,
    emptyCount,
    error,
    isHost,
    confirmClose,
    setConfirmClose,
    chooseSeat,
    fillAI,
    startGame,
    closeTable,
    chat,
    rules,
  } = props;
  const leaveLink = leaveLinkFor(props.onLeave);
  return (
        /* ---------- DESKTOP ---------- */
        <>
          <div className={styles.chrome}>
            <Wordmark size="sm" />
            <div className={styles.chromeNav}>
              {leaveLink("Lobby ↗", ds.link)}
            </div>
          </div>

          <div className={styles.desk}>
            <div className={styles.header}>
              <div>
                <div className={ds.overline}>Waiting Room · {shortId}</div>
                <h1 className={styles.title}>{roomName}</h1>
              </div>
              <div className={styles.headerStats}>
                <div className={styles.stat}>
                  <div className={ds.overline} style={{ fontSize: 10 }}>
                    Players
                  </div>
                  <div className={styles.statNum}>
                    {humanCount}
                    <span className={styles.statOf}>/5</span>
                  </div>
                </div>
                {hostName && (
                  <>
                    <div className={styles.statRule} />
                    <div className={styles.stat}>
                      <div className={ds.overline} style={{ fontSize: 10 }}>
                        Host
                      </div>
                      <div className={styles.statName}>{hostName}</div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {error && <div className={styles.error}>{error}</div>}

            <div>
              <div className={ds.overline} style={{ marginBottom: 12 }}>
                The Table · 5 Seats
              </div>
              <div className={styles.seatsGrid}>
                {seatItems.map((s) => (
                  <SeatCard
                    key={s.seat}
                    seat={s}
                    variant="card"
                    onTake={chooseSeat}
                  />
                ))}
              </div>
            </div>

            <div className={styles.twoCol}>
              {rules}
              <div className={styles.chatWrap}>{chat}</div>
            </div>

            <div className={styles.actionBar}>
              <div className={styles.actionLeft}>
                {isHost && (
                  <button
                    className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                    onClick={fillAI}
                  >
                    Fill empty with AI
                  </button>
                )}
                {isHost &&
                  (confirmClose ? (
                    <div className={styles.confirmRow}>
                      <button
                        className={`${ds.btn} ${ds.btnSm}`}
                        onClick={closeTable}
                      >
                        Confirm close
                      </button>
                      <button
                        className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                        onClick={() => setConfirmClose(false)}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.dangerLink}`}
                      onClick={() => setConfirmClose(true)}
                    >
                      Close table
                    </button>
                  ))}
              </div>
              <div className={styles.actionRight}>
                {isHost ? (
                  <>
                    {emptyCount > 0 && (
                      <div className={styles.waitHint}>
                        Waiting on {emptyCount} more, or fill with AI
                      </div>
                    )}
                    <button
                      className={`${ds.btn} ${ds.btnAccent} ${ds.btnLg}`}
                      onClick={startGame}
                    >
                      Deal cards →
                    </button>
                  </>
                ) : (
                  <div className={styles.waitHint}>
                    Waiting for the host to deal…
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
  );
}
