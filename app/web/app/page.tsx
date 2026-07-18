"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { TableSummary } from "../lib/types";
import { apiFetch, storeSessionToken } from "../lib/api";
import { STORAGE_KEYS } from "../lib/storage";
import { ds } from "../lib/ds";
import { useIdentity } from "../lib/hooks/useIdentity";
import MastheadBand from "./components/home/MastheadBand";
import Strapline from "./components/home/Strapline";
import styles from "./page.module.css";

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

  const refresh = async () => {
    setLoading(true);
    try {
      const res = await apiFetch("/api/tables");
      const data = await res.json();
      setTables(data);
    } catch {
      // Backend unreachable — show an empty lobby rather than throwing.
      setTables([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
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
      // Create table
      const res = await apiFetch("/api/tables", {
        method: "POST",
        body: JSON.stringify({
          name,
          fillWithAI: true,
          rules: { partnerMode: 1, doubleOnTheBump: true },
        }),
      });

      if (!res.ok) {
        throw new Error(
          `Failed to create table: ${res.status} ${res.statusText}`,
        );
      }

      const t = await res.json();

      // Auto-join the table; identity rides on the Authorization header.
      const res2 = await apiFetch(`/api/tables/${t.id}/join`, {
        method: "POST",
        body: JSON.stringify({ display_name: resolvedDisplayName }),
      });

      if (!res2.ok) {
        throw new Error(
          `Failed to join table: ${res2.status} ${res2.statusText}`,
        );
      }

      const joined = await res2.json();
      storeSessionToken(joined.session_token);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(
          STORAGE_KEYS.clientId(t.id),
          joined.client_id,
        );
      }
      const typedName = displayNameInput.trim();
      await persistIdentityFromJoin(joined, typedName);

      // localStorage writes above are synchronous; navigate directly.
      router.push(`/waiting/${t.id}`);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to create table. Please try again.";
      setError(errorMessage);
    } finally {
      setCreating(false);
    }
  };

  const join = async (tableId: string) => {
    const resolvedDisplayName = displayName;
    setError(null);

    try {
      // If we have a stored client id for this table, reuse it
      if (typeof window !== "undefined") {
        const existing = window.localStorage.getItem(
          STORAGE_KEYS.clientId(tableId),
        );
        if (existing) {
          await persistTypedName(playerId, displayNameInput.trim());
          router.push(`/waiting/${tableId}`);
          return;
        }
      }

      const res = await apiFetch(`/api/tables/${tableId}/join`, {
        method: "POST",
        body: JSON.stringify({ display_name: resolvedDisplayName }),
      });

      if (!res.ok) {
        throw new Error(
          `Failed to join table: ${res.status} ${res.statusText}`,
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

      router.push(`/waiting/${data.table.id}`);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to join table. Please try again.";
      setError(errorMessage);
    }
  };

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
      <div className={styles.grid}>
        {/* LEFT — hero + create */}
        <div className={styles.hero}>
          <MastheadBand />

          <div className={styles.wordmark}>
            <h1 className={styles.wordmarkTitle}>Sheepshead</h1>
            <div className={styles.wordmarkRow}>
              <em className={styles.wordmarkAI}>AI</em>
              <span className={styles.wordmarkRule} aria-hidden="true" />
              <Strapline />
            </div>
          </div>

          <p className={styles.lede}>
            A five‑handed, trick‑taking game from Wisconsin — played here with
            friends and a deep‑learning AI.
          </p>

          {error && <div className={styles.error}>{error}</div>}

          <div className={styles.startBlock}>
            <div className={`${ds.headRule} ${styles.startHead}`}>
              <div className={ds.overline}>Start a Table</div>
            </div>
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
                {creating ? "Creating…" : "Create table →"}
              </button>
              <span className={styles.createHint}>or join an open table</span>
            </div>
          </div>
        </div>

        {/* RIGHT — lobby */}
        <div className={styles.lobby}>
          <div className={`${ds.headRule} ${styles.lobbyHead}`}>
            <h2 className={styles.lobbyTitle}>The Lobby</h2>
            <div className={ds.overline}>{tables.length} tables · live</div>
          </div>

          <div className={styles.colHead}>
            <div className={ds.overline}>Name</div>
            <div className={ds.overline}>Players</div>
            <div className={`${ds.overline} ${styles.colHeadRight}`}>
              Action
            </div>
          </div>

          <div className={styles.rows}>
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
                  <div key={t.id} className={styles.row}>
                    <div>
                      <div className={styles.tableName}>{t.name}</div>
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
                          className={`${ds.btn} ${ds.btnSm}`}
                          onClick={() => join(t.id)}
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

          <div className={styles.lobbyFooter}>
            <a className={ds.link} href="/analyze" style={{ fontSize: 13 }}>
              Inspect AI model decisions ↗
            </a>
            <div className={styles.version}>v0.7 · Jun 2026</div>
          </div>
        </div>
      </div>
    </div>
  );
}
