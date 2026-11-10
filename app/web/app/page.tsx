"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { TableSummary } from "../lib/types";
import { tableRoute } from "../lib/tableRoute";
import { apiFetch, storeSessionToken } from "../lib/api";
import { STORAGE_KEYS } from "../lib/storage";
import { createAndJoinTable } from "../lib/createTable";
import { PlayingCard, ds } from "../lib/ds";
import { allPassModeLabel, partnerModeLabel } from "../lib/rules";
import { useIdentity } from "../lib/hooks/useIdentity";
import { useBusy } from "../lib/hooks/useBusy";
import Link from "next/link";
import SiteNav from "./components/SiteNav";
import AccountLine from "./components/home/AccountLine";
import MastheadBand from "./components/home/MastheadBand";
import styles from "./page.module.css";

/** How often the lobby refetches the table list while the tab is visible. */
const LOBBY_POLL_MS = 5000;

// Why a table sent the player back here (the ?notice= query).
const NOTICES: Record<string, string> = {
  removed: "The host removed you from that table.",
  idle: "That table closed after sitting idle.",
};

export default function HomePage() {
  const router = useRouter();
  const [tables, setTables] = useState<TableSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const {
    name,
    handleTableNameChange,
    displayPlaceholder,
    displayNameInput,
    setDisplayNameInput,
    displayName,
    playerId,
    persistTypedName,
    persistIdentityFromJoin,
  } = useIdentity(tables);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Why the player was sent back here from a table (useTableSocket).
  useEffect(() => {
    const notice = new URLSearchParams(window.location.search).get("notice");
    if (!notice || !(notice in NOTICES)) return;
    setError(NOTICES[notice]);
    window.history.replaceState(null, "", window.location.pathname);
  }, []);

  // `showLoading` only on the first fetch: the poll below must not flash the
  // list back to "Loading…" every few seconds.
  const refresh = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const res = await apiFetch("/api/tables");
      const data = await res.json();
      setTables(data);
    } catch {
      // Backend unreachable — show an empty lobby rather than throwing.
      setTables([]);
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  // The lobby has no socket, so it polls: seats fill, hands start and hosts
  // change the house rules while someone is looking at the list, and a
  // snapshot taken at mount goes stale within seconds. Paused while the tab
  // is hidden, and refetched immediately on return so the list is current
  // before it is read.
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;

    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };
    const start = () => {
      stop();
      timer = setInterval(() => void refresh(false), LOBBY_POLL_MS);
    };
    const onVisibility = () => {
      if (document.hidden) {
        stop();
      } else {
        void refresh(false);
        start();
      }
    };

    void refresh();
    start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  // Mobile Safari compatible click handler
  const handleCreateClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    create(displayName);
  };

  const create = async (resolvedDisplayName: string) => {
    // Clear any previous errors
    setError(null);
    setCreating(true);

    try {
      const tableId = await createAndJoinTable({
        name,
        displayName: resolvedDisplayName,
        typedName: displayNameInput.trim(),
        persistIdentityFromJoin,
      });
      // createAndJoinTable's localStorage writes are synchronous; navigate
      // directly.
      router.push(`/waiting/${tableId}`);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to create table. Please try again.";
      setError(errorMessage);
      // Only on failure: on success the button stays "Creating…" until the
      // waiting room replaces this page.
      setCreating(false);
    }
  };

  // Held on success: the waiting room replaces this page.
  const joining = useBusy();
  const [joinPressed, setJoinPressed] = useState<string | null>(null);
  const join = (tableId: string) => {
    setJoinPressed(tableId);
    return joining.run(() => joinTable(tableId), { hold: true });
  };

  const joinTable = async (tableId: string): Promise<boolean> => {
    const resolvedDisplayName = displayName;
    setError(null);
    // A dealt table, even between hands, is played on the table page; one
    // not in the list yet goes by way of the waiting room, which sends it on.
    const route = tableRoute(
      tableId,
      tables.find((t) => t.id === tableId)?.status,
    );

    try {
      // If we have a stored client id for this table, reuse it
      if (typeof window !== "undefined") {
        const existing = window.localStorage.getItem(
          STORAGE_KEYS.clientId(tableId),
        );
        if (existing) {
          await persistTypedName(playerId, displayNameInput.trim());
          router.push(route);
          return true;
        }
      }

      const res = await apiFetch(`/api/tables/${tableId}/join`, {
        method: "POST",
        body: JSON.stringify({ display_name: resolvedDisplayName }),
      });

      if (!res.ok) {
        const detail = (await res.json().catch(() => ({})))?.detail;
        throw new Error(
          detail === "removed_from_table"
            ? "The host removed you from that table."
            : `Failed to join table: ${res.status} ${res.statusText}`,
        );
      }

      const data = await res.json();
      storeSessionToken(data.session_token);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(
          STORAGE_KEYS.clientId(tableId),
          data.client_id,
        );
      }
      const typedName = displayNameInput.trim();
      await persistIdentityFromJoin(data, typedName);

      router.push(route);
      return true;
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to join table. Please try again.";
      setError(errorMessage);
      return false;
    }
  };

  // A table's "Rejoin" (after leaving it once a hand ended) lands here:
  // join it as soon as the player's name is known.
  const [rejoinId, setRejoinId] = useState<string | null>(null);
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("rejoin");
    if (!id) return;
    setRejoinId(id);
    window.history.replaceState(null, "", window.location.pathname);
  }, []);
  useEffect(() => {
    if (!rejoinId || !displayName) return;
    setRejoinId(null);
    void join(rejoinId);
    // join is redefined every render; the id is what triggers this.
  }, [rejoinId, displayName]);

  const botCount = (t: TableSummary) =>
    Object.entries(t.seats || {}).reduce((acc, [k, seatName]) => {
      const isAI =
        (t.seatIsAI as Record<string, boolean> | undefined)?.[k] ?? false;
      return acc + (seatName && isAI ? 1 : 0);
    }, 0);

  const humanFilledCount = (t: TableSummary) =>
    Object.entries(t.seats || {}).reduce((acc, [k, seatName]) => {
      const isAI =
        (t.seatIsAI as Record<string, boolean> | undefined)?.[k] ?? false;
      return acc + (seatName && !isAI ? 1 : 0);
    }, 0);

  return (
    <div className={styles.page}>
      <SiteNav />
      <div className={styles.grid}>
        {/* LEFT — hero + create */}
        <div className={styles.hero}>
          <div className={styles.band}>
            <MastheadBand />
          </div>
          {/* On a phone the band's cards sit alone over the wordmark. */}
          <div className={styles.heroCards} aria-hidden="true">
            <div style={{ transform: "rotate(-9deg) translateY(3px)" }}>
              <PlayingCard code="QC" w={34} />
            </div>
            <div style={{ zIndex: 1 }}>
              <PlayingCard code="JD" w={34} />
            </div>
            <div style={{ transform: "rotate(9deg) translateY(3px)" }}>
              <PlayingCard code="AH" w={34} />
            </div>
          </div>

          <div className={styles.wordmark}>
            <h1 className={styles.wordmarkTitle}>Sheepshead</h1>
            <div className={styles.wordmarkRow}>
              <em className={styles.wordmarkAI}>AI</em>
              <span className={styles.wordmarkRule} aria-hidden="true" />
            </div>
          </div>

          <p className={styles.lede}>
            <span className={styles.ledeLead}>
              A five‑handed trick‑taking game from Wisconsin.
            </span>
            Play live with friends, strangers, and a deep neural network.
          </p>

          {/* The outlined button and its time on desktop, a plain link on a
              phone: CSS picks one, so there is no viewport JS. */}
          <div className={styles.howTo}>
            <Link
              href="/how-to-play"
              className={`${ds.btn} ${ds.btnGhost} ${styles.howToBtn}`}
            >
              How to play →
            </Link>
            <span className={`${styles.createHint} ${styles.howToHint}`}>
              about 3 minutes
            </span>
            <Link
              href="/how-to-play"
              className={`${ds.link} ${styles.howToLink}`}
            >
              How to play
            </Link>
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <div className={styles.startBlock}>
            <div className={styles.formRow}>
              <div className={styles.field}>
                <label className={`${ds.overline} ${styles.fieldLabel}`}>
                  Your name
                </label>
                <input
                  className={ds.input}
                  value={displayNameInput}
                  placeholder={displayPlaceholder ?? "Your name"}
                  onChange={(e) => setDisplayNameInput(e.target.value)}
                  disabled={creating}
                  maxLength={32}
                />
              </div>
              <div className={styles.field}>
                <label className={`${ds.overline} ${styles.fieldLabel}`}>
                  Table name
                </label>
                <input
                  className={ds.input}
                  value={name}
                  placeholder="Table name"
                  onChange={(e) => handleTableNameChange(e.target.value)}
                  disabled={creating}
                  maxLength={48}
                />
              </div>
            </div>
            <div className={styles.createRow}>
              <button
                className={`${ds.btn} ${ds.btnAccent} ${ds.btnLg}`}
                onClick={handleCreateClick}
                disabled={creating || !name.trim() || !displayName.trim()}
                suppressHydrationWarning
                style={{ touchAction: "manipulation" }}
              >
                {creating ? "Creating…" : "Start a table →"}
              </button>
              <AccountLine />
            </div>
          </div>
        </div>

        {/* RIGHT — lobby */}
        <div className={styles.lobby}>
          <div className={`${ds.headRule} ${styles.lobbyHead}`}>
            <h2 className={styles.lobbyTitle}>The Lobby</h2>
            <div className={ds.overline}>
              {tables.length} {tables.length === 1 ? "table" : "tables"} · live
            </div>
          </div>

          <div className={styles.colHead}>
            <div className={ds.overline}>Name</div>
            <div className={ds.overline}>Players</div>
            <div className={`${ds.overline} ${styles.colHeadRight}`}>
              Action
            </div>
          </div>

          <div
            className={styles.rows}
            role={tables.length > 0 ? "list" : undefined}
            aria-label="Open tables"
          >
            {loading ? (
              <div className={styles.empty}>Loading…</div>
            ) : tables.length === 0 ? (
              <div className={styles.empty}>
                No open tables yet — start one.
              </div>
            ) : (
              tables.map((t) => {
                const humanFilled = humanFilledCount(t);
                const bots = botCount(t);
                const canJoin = humanFilled < 5;
                const playing = t.status === "playing";
                return (
                  <div key={t.id} className={styles.row} role="listitem">
                    <div>
                      <div className={styles.tableName}>{t.name}</div>
                      <div className={styles.tableRules}>
                        {/* Each mode is one unbreakable unit, so a narrow
                            column wraps between them -- carrying the
                            separator down -- rather than mid-mode. */}
                        <span className={styles.rulesPart}>
                          {partnerModeLabel(t.rules)}
                        </span>{" "}
                        <span className={styles.rulesPart}>
                          · {allPassModeLabel(t.rules)}
                        </span>
                      </div>
                      {(playing || bots > 0) && (
                        <div className={styles.tableSub}>
                          {[
                            playing ? "in play" : null,
                            bots > 0 ? `${bots} AI` : null,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </div>
                      )}
                    </div>
                    <div className={styles.players}>
                      {humanFilled}
                      <span className={styles.playersOf}>/5</span>
                    </div>
                    <div className={styles.rowAction}>
                      {canJoin ? (
                        <button
                          className={`${ds.btn} ${ds.btnSm} ${joining.busy && joinPressed === t.id ? ds.btnBusy : ""}`}
                          onClick={() => join(t.id)}
                          disabled={joining.busy}
                          aria-busy={joining.busy && joinPressed === t.id}
                        >
                          Join →
                        </button>
                      ) : (
                        <span className={`${ds.badge} ${ds.badgeQuiet}`}>
                          Full
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
