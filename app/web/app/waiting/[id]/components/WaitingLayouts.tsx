import React, { useState } from "react";
import { MiniCardMark, TypingNote, ds, Wordmark } from "../../../../lib/ds";
import { useBusy } from "../../../../lib/hooks/useBusy";
import { SeatChip, SeatTile, type SeatInfo } from "./Seat";
import WaitingMenu from "./WaitingMenu";
import styles from "../page.module.css";

export interface WaitingLayoutProps {
  roomName: string;
  shortId: string;
  hostName: string | null;
  seatItems: SeatInfo[];
  humanCount: number;
  error: string | null;
  isHost: boolean;
  chooseSeat: (seat: number) => void;
  /** The seat the viewer asked to move into, while that is in flight. */
  takingSeat: number | null;
  /** Resolves false when the server refused. */
  removePlayer: (clientId: string) => Promise<boolean>;
  startGame: () => void;
  /** Deal cards is waiting on the server (or the table page). */
  starting: boolean;
  closeTable: () => void;
  closing: boolean;
  onLeave: () => void;
  chat: React.ReactNode;
  rules: React.ReactNode;
  /** Names of the others typing in the chat, shown to the host by Deal
   * cards. */
  typing: string[];
}

interface MobileWaitingLayoutProps extends WaitingLayoutProps {
  chatOpen: boolean;
  setChatOpen: (open: boolean) => void;
  chatUnread: boolean;
}

/** Which seat's player the host is about to remove, while it is still
 * removable (they may have left meanwhile). */
function useRemoving(seatItems: SeatInfo[]) {
  const [seat, setSeat] = useState<number | null>(null);
  const removing =
    seatItems.find((s) => s.seat === seat && s.removableId) ?? null;
  return [removing, (s: SeatInfo | null) => setSeat(s?.seat ?? null)] as const;
}

/** The host's confirmation before removing a player, under the seats. */
function RemoveConfirm({
  seat,
  onRemove,
  onCancel,
}: {
  seat: SeatInfo;
  onRemove: (clientId: string) => Promise<boolean>;
  onCancel: () => void;
}) {
  // Held on success: the confirmation goes once the seat is no longer
  // removable, when the server's update arrives.
  const removing = useBusy();
  return (
    <div
      className={styles.removeConfirm}
      role="group"
      aria-label="Confirm removal"
    >
      <span className={styles.removePrompt}>
        Remove {seat.name} from the table?
      </span>
      <button
        type="button"
        className={`${ds.btn} ${ds.btnSm} ${removing.busy ? ds.btnBusy : ""}`}
        onClick={() => {
          const target = seat.removableId;
          if (!target) return;
          void removing.run(
            async () => {
              const ok = await onRemove(target);
              if (!ok) onCancel();
              return ok;
            },
            { hold: true },
          );
        }}
        disabled={removing.busy}
        aria-busy={removing.busy}
      >
        Remove
      </button>
      <button
        type="button"
        className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
        onClick={onCancel}
      >
        Cancel
      </button>
    </div>
  );
}

function waitingForHost(hostName: string | null) {
  return hostName
    ? `Waiting for ${hostName} to deal…`
    : "Waiting for the host to deal…";
}

export function MobileWaitingLayout(props: MobileWaitingLayoutProps) {
  const {
    roomName,
    shortId,
    hostName,
    seatItems,
    error,
    isHost,
    chooseSeat,
    startGame,
    closeTable,
    onLeave,
    chat,
    rules,
    chatOpen,
    setChatOpen,
    chatUnread,
  } = props;
  const [menuOpen, setMenuOpen] = useState(false);
  const [removing, setRemoving] = useRemoving(seatItems);

  if (chatOpen) {
    return (
      <div className={styles.chatScreen}>
        <div className={styles.chatHead}>
          <button
            type="button"
            className={styles.back}
            onClick={() => setChatOpen(false)}
          >
            ← Back
          </button>
          <div className={styles.chatTitle}>{roomName}</div>
          <span className={styles.chatHeadSpacer} />
        </div>
        <div className={styles.chatBody}>{chat}</div>
      </div>
    );
  }

  return (
    <div className={styles.mob}>
      <div className={styles.mobHead}>
        <MiniCardMark h={20} />
        <h1 className={styles.mobRoom}>{roomName}</h1>
        <div className={styles.mobIcons}>
          <button
            type="button"
            className={`${styles.iconBtn} ${chatUnread ? styles.dot : ""}`}
            onClick={() => setChatOpen(true)}
            aria-label={chatUnread ? "Chat, new messages" : "Chat"}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M3.5 4.5h13v9h-7l-4 3v-3h-2z" />
            </svg>
          </button>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={() => setMenuOpen(true)}
            aria-label="Waiting room menu"
            aria-haspopup="dialog"
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M3.5 6h13M3.5 10h13M3.5 14h13" />
            </svg>
          </button>
        </div>
      </div>

      <div className={styles.mobBody}>
        {error && <div className={styles.error}>{error}</div>}
        <section>
          <div className={styles.sectionHead}>
            <span className={`${ds.overline} ${styles.sectionTitle}`}>
              Seats
            </span>
            {seatItems.some((s) => s.takeable && !s.isYou) && (
              <span className={styles.hint}>Tap an AI seat to sit there</span>
            )}
          </div>
          <div className={styles.seatStrip}>
            {seatItems.map((s) => (
              <SeatChip
                key={s.seat}
                seat={s}
                onTake={chooseSeat}
                taking={props.takingSeat === s.seat}
                onRemove={setRemoving}
              />
            ))}
          </div>
          {removing && (
            <RemoveConfirm
              seat={removing}
              onRemove={props.removePlayer}
              onCancel={() => setRemoving(null)}
            />
          )}
        </section>
        <section>
          <div className={styles.sectionHead}>
            <span className={`${ds.overline} ${styles.sectionTitle}`}>
              House rules
            </span>
          </div>
          {rules}
        </section>
      </div>

      <div className={styles.mobFoot}>
        {isHost && (
          <TypingNote names={props.typing} className={styles.dealTyping} />
        )}
        {isHost ? (
          <button
            type="button"
            className={`${ds.btn} ${ds.btnAccent} ${ds.btnLg} ${styles.deal} ${props.starting ? ds.btnBusy : ""}`}
            onClick={startGame}
            disabled={props.starting}
            aria-busy={props.starting}
          >
            Deal cards →
          </button>
        ) : (
          <div className={styles.waitHint}>{waitingForHost(hostName)}</div>
        )}
      </div>

      {menuOpen && (
        <WaitingMenu
          roomName={roomName}
          shortId={shortId}
          hostName={hostName}
          onLeave={onLeave}
          onCloseTable={isHost ? closeTable : null}
          closing={props.closing}
          onClose={() => setMenuOpen(false)}
        />
      )}
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
    error,
    isHost,
    chooseSeat,
    startGame,
    closeTable,
    onLeave,
    chat,
    rules,
  } = props;
  const [confirmClose, setConfirmClose] = useState(false);
  const [removing, setRemoving] = useRemoving(seatItems);
  return (
    <>
      <div className={styles.chrome}>
        <Wordmark size="sm" />
        {/* Not a navigation: onClick always preventDefault()s and delegates
            to onLeave(). The href only supplies link semantics (status bar,
            middle-click), so next/link would add a router it never uses. */}
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
        <a
          className={ds.link}
          href="/"
          onClick={(e) => {
            e.preventDefault();
            onLeave();
          }}
        >
          Lobby ↗
        </a>
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
          <div className={styles.seatTiles}>
            {seatItems.map((s) => (
              <SeatTile
                key={s.seat}
                seat={s}
                onTake={chooseSeat}
                taking={props.takingSeat === s.seat}
                onRemove={setRemoving}
              />
            ))}
          </div>
          {removing && (
            <RemoveConfirm
              seat={removing}
              onRemove={props.removePlayer}
              onCancel={() => setRemoving(null)}
            />
          )}
        </div>

        <div className={styles.twoCol}>
          {rules}
          <div className={styles.chatWrap}>{chat}</div>
        </div>

        <div className={styles.actionBar}>
          <div className={styles.actionLeft}>
            {isHost &&
              (confirmClose ? (
                <>
                  <button
                    className={`${ds.btn} ${ds.btnSm} ${props.closing ? ds.btnBusy : ""}`}
                    onClick={closeTable}
                    disabled={props.closing}
                    aria-busy={props.closing}
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
                  className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm} ${styles.dangerLink}`}
                  onClick={() => setConfirmClose(true)}
                >
                  Close table
                </button>
              ))}
          </div>
          {isHost && (
            <TypingNote names={props.typing} className={styles.dealTyping} />
          )}
          {isHost ? (
            <button
              className={`${ds.btn} ${ds.btnAccent} ${ds.btnLg} ${props.starting ? ds.btnBusy : ""}`}
              onClick={startGame}
              disabled={props.starting}
              aria-busy={props.starting}
            >
              Deal cards →
            </button>
          ) : (
            <div className={styles.waitHint}>{waitingForHost(hostName)}</div>
          )}
        </div>
      </div>
    </>
  );
}
