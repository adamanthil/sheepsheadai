"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import type { TableSummary } from "../../../lib/types";
import { apiFetch } from "../../../lib/api";
import { useTableSocket } from "../../../lib/hooks/useTableSocket";
import { STORAGE_KEYS } from "../../../lib/storage";
import styles from "./page.module.css";
import { ChatPanel } from "../../components/chat";
import { ds, Wordmark, useIsMobile } from "../../../lib/ds";
import SeatCard from "./components/SeatCard";
import RulesPanel from "./components/RulesPanel";

type TableInfo = TableSummary & {
  seats: Record<string, string | null>;
};

export default function WaitingRoom() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [clientId, setClientId] = useState<string>("");

  // Hydrate client_id from localStorage (Phase 2: no more ?client_id in URLs).
  useEffect(() => {
    if (typeof window === "undefined" || !params?.id) return;
    const stored = window.localStorage.getItem(
      STORAGE_KEYS.clientId(params.id),
    );
    if (stored) {
      setClientId(stored);
    } else {
      // No session for this table — send them back to the lobby.
      router.replace("/");
    }
  }, [params?.id, router]);

  const [table, setTable] = useState<TableInfo | null>(null);
  const [isHost, setIsHost] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [partnerMode, setPartnerMode] = useState<number>(1); // 1 = Called Ace (default), 0 = Jack of Diamonds
  const [scoringMode, setScoringMode] = useState<number>(1); // 1 = Double on the Bump (default), 0 = Symmetric
  const [callout, setCallout] = useState<string | null>(null);
  const [confirmClose, setConfirmClose] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/api/tables");
      const list: TableInfo[] = await res.json();
      const t = list.find((t) => t.id === params?.id);
      if (t) {
        setTable(t);
        // Initialize modes from table rules
        const rules = t.rules ?? {};
        const pMode = rules.partnerMode !== undefined ? rules.partnerMode : 1; // Default to Called Ace
        setPartnerMode(pMode);
        const sMode =
          rules.doubleOnTheBump !== undefined
            ? rules.doubleOnTheBump
              ? 1
              : 0
            : 1; // Default to Double on the Bump
        setScoringMode(sMode);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [params?.id]);

  // Shared socket hook: lobby updates, chat, auto-navigation on game start,
  // and reconnect-with-backoff all come from useTableSocket.
  const { lastState, chatMessages, sendChatMessage } = useTableSocket(
    params?.id,
    clientId,
    {
      onTableUpdate: (t, hostFlag) => {
        setTable(t as TableInfo);
        if (typeof hostFlag === "boolean") setIsHost(hostFlag);
        const rules = t.rules ?? {};
        if (rules.partnerMode !== undefined) setPartnerMode(rules.partnerMode);
        if (rules.doubleOnTheBump !== undefined)
          setScoringMode(rules.doubleOnTheBump ? 1 : 0);
        if (t.status === "playing") {
          router.push(`/table/${params?.id}`);
        }
      },
      onLobbyEvent: (msg) => {
        setCallout(msg);
        setTimeout(() => setCallout(null), 1800);
      },
      onTableClosed: () => setCallout("Table closed"),
    },
  );

  // A state message while in the waiting room means the game has started.
  useEffect(() => {
    if (lastState?.table?.status === "playing" && params?.id) {
      router.push(`/table/${params.id}`);
    }
  }, [lastState, params?.id, router]);

  async function chooseSeat(seat: number) {
    const res = await apiFetch(`/api/tables/${params?.id}/seat`, {
      method: "POST",
      body: JSON.stringify({ client_id: clientId, seat }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j?.detail || "Seat selection failed");
      return;
    }
    const t = await res.json();
    setTable(t);
  }

  async function fillAI() {
    const res = await apiFetch(`/api/tables/${params?.id}/fill_ai`, {
      method: "POST",
      body: JSON.stringify({ client_id: clientId }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j?.detail || "Failed to fill seats with AI");
      return;
    }
    const t = await res.json();
    setTable(t);
  }

  async function updatePartnerMode(newMode: number) {
    try {
      const res = await apiFetch(`/api/tables/${params?.id}/rules`, {
        method: "PATCH",
        body: JSON.stringify({
          client_id: clientId,
          rules: { partnerMode: newMode },
        }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        setError(j?.detail || "Failed to update partner mode");
        return;
      }
      setPartnerMode(newMode);
    } catch {
      setError("Failed to update partner mode");
    }
  }

  async function updateScoringMode(newMode: number) {
    try {
      const res = await apiFetch(`/api/tables/${params?.id}/rules`, {
        method: "PATCH",
        body: JSON.stringify({
          client_id: clientId,
          rules: { doubleOnTheBump: newMode === 1 },
        }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        setError(j?.detail || "Failed to update scoring mode");
        return;
      }
      setScoringMode(newMode);
    } catch {
      setError("Failed to update scoring mode");
    }
  }

  async function startGame() {
    const res = await apiFetch(`/api/tables/${params?.id}/start`, {
      method: "POST",
      body: JSON.stringify({ client_id: clientId }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j?.detail || "Start failed");
      return;
    }
    router.push(`/table/${params?.id}`);
  }

  async function closeTable() {
    if (!params?.id || !clientId) return;
    await apiFetch(`/api/tables/${params?.id}/close`, {
      method: "POST",
      body: JSON.stringify({ client_id: clientId }),
    });
  }

  const seatItems = useMemo(() => {
    if (!table)
      return [] as Array<{ seat: number; name: string | null; isAI: boolean }>;
    const out: Array<{ seat: number; name: string | null; isAI: boolean }> = [];
    for (let i = 1; i <= 5; i++)
      out.push({
        seat: i,
        name: table.seats[String(i)] || null,
        isAI: Boolean(table.seatIsAI?.[String(i)]),
      });
    return out;
  }, [table]);

  const isMobile = useIsMobile();
  const shortId = `#${String(params?.id || "")
    .slice(0, 4)
    .toUpperCase()}`;
  const roomName = table?.name || params?.id || "Table";
  const hostName = table?.host || null;
  const humanCount = seatItems.filter((s) => !!s.name && !s.isAI).length;
  const seatedCount = seatItems.filter((s) => !!s.name).length;
  const emptyCount = seatItems.filter((s) => !s.name).length;

  const chat = (
    <ChatPanel messages={chatMessages} onSendMessage={sendChatMessage} />
  );
  const rules = (
    <RulesPanel
      partnerMode={partnerMode}
      scoringMode={scoringMode}
      onPartnerMode={updatePartnerMode}
      onScoringMode={updateScoringMode}
      variant={isMobile ? "inline" : "panel"}
    />
  );

  const leaveLink = (label: string, className: string) => (
    <a
      className={className}
      href="/"
      onClick={(e) => {
        e.preventDefault();
        router.push("/");
      }}
    >
      {label}
    </a>
  );

  return (
    <div className={styles.root}>
      {callout && <div className={styles.callout}>{callout}</div>}

      {isMobile ? (
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
      ) : (
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
      )}
    </div>
  );
}
