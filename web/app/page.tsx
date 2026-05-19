"use client";

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { TableSummary } from '../lib/types';
import { API_BASE } from '../lib/apiBase';
import { STORAGE_KEYS } from '../lib/storage';
import styles from './page.module.css';

const SHIRE_TOWNS = [
  'Bywater',
  'Hobbiton',
  'Overhill',
  'Frogmorton',
  'Michel Delving',
  'Tuckborough',
  'Bree',
  'Bucklebury',
  'Deephallow',
  'Waymoot',
  'Nobottle',
  'Whitfurrows',
] as const;

const HOBBIT_NAMES = [
  'Bilbo',
  'Frodo',
  'Samwise',
  'Merry',
  'Pippin',
  'Otho',
  'Smeagol',
  'Deagol',
  'Lobelia',
  'Belladonna',
] as const;

const getRandomItem = <T,>(items: readonly T[]) => items[Math.floor(Math.random() * items.length)];

export default function HomePage() {
  const router = useRouter();
  const [tables, setTables] = useState<TableSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState<string>('');
  const [displayPlaceholder, setDisplayPlaceholder] = useState<string | null>(null);
  const [displayNameInput, setDisplayNameInput] = useState('');
  const displayName = displayNameInput.trim() || displayPlaceholder || '';
  const [hasCustomName, setHasCustomName] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playerId, setPlayerId] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/tables`);
      const data = await res.json();
      setTables(data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    setName(getRandomItem(SHIRE_TOWNS));
  }, []);

  useEffect(() => {
    if (!displayPlaceholder) {
      setDisplayPlaceholder(getRandomItem(HOBBIT_NAMES));
    }
  }, [displayPlaceholder]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    let cancelled = false;
    const storedId = window.localStorage.getItem(STORAGE_KEYS.playerId);
    const storedName = window.localStorage.getItem(STORAGE_KEYS.displayName) ?? '';
    if (storedName) {
      setDisplayNameInput(storedName);
    }
    if (!storedId) {
      setPlayerId(null);
      return;
    }
    setPlayerId(storedId);
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/players/${storedId}`);
        if (cancelled) return;
        if (res.status === 404) {
          window.localStorage.removeItem(STORAGE_KEYS.playerId);
          setPlayerId(null);
          return;
        }
        if (!res.ok) return;
        const data = await res.json();
        if (data.name) {
          window.localStorage.setItem(STORAGE_KEYS.displayName, data.name);
          setDisplayNameInput(prev => prev || data.name);
        }
      } catch {
        // Network failure — keep the stored id; the user can still play.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (hasCustomName || tables.length === 0) return;
    const usedNames = new Set(tables.map(t => t.name));
    if (!usedNames.has(name)) return;
    const available = SHIRE_TOWNS.filter(town => !usedNames.has(town));
    if (!available.length) return;
    setName(getRandomItem(available));
  }, [tables, hasCustomName, name]);

  const handleTableNameChange = (value: string) => {
    if (!hasCustomName) {
      setHasCustomName(true);
    }
    setName(value);
  };

  // Mobile Safari compatible click handler
  const handleCreateClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    create(displayName);
  };

  const persistTypedName = async (id: string | null, typedName: string) => {
    if (typeof window === 'undefined' || !id || !typedName) return;
    const previous = window.localStorage.getItem(STORAGE_KEYS.displayName);
    if (previous === typedName) return;
    try {
      await fetch(`${API_BASE}/api/players/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: typedName }),
      });
      window.localStorage.setItem(STORAGE_KEYS.displayName, typedName);
    } catch {
      // best-effort; live display name still works via display_name field
    }
  };

  const persistIdentityFromJoin = async (
    joined: { player_id?: string },
    typedName: string,
  ): Promise<string | null> => {
    if (typeof window === 'undefined') return playerId;
    let resultId = playerId;
    if (joined.player_id && joined.player_id !== playerId) {
      window.localStorage.setItem(STORAGE_KEYS.playerId, joined.player_id);
      setPlayerId(joined.player_id);
      resultId = joined.player_id;
    }
    await persistTypedName(resultId, typedName);
    return resultId;
  };

  const create = async (resolvedDisplayName: string) => {
    // Clear any previous errors
    setError(null);
    setCreating(true);

    try {
      // Create table
      const res = await fetch(`${API_BASE}/api/tables`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, fillWithAI: true, rules: { partnerMode: 1, doubleOnTheBump: true } }),
      });

      if (!res.ok) {
        throw new Error(`Failed to create table: ${res.status} ${res.statusText}`);
      }

      const t = await res.json();

      // Auto-join the table
      const res2 = await fetch(`${API_BASE}/api/tables/${t.id}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: resolvedDisplayName, player_id: playerId }),
      });

      if (!res2.ok) {
        throw new Error(`Failed to join table: ${res2.status} ${res2.statusText}`);
      }

      const joined = await res2.json();
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(STORAGE_KEYS.clientId(t.id), joined.client_id);
      }
      const typedName = displayNameInput.trim();
      await persistIdentityFromJoin(joined, typedName);

      // Use setTimeout to ensure state updates complete before navigation
      setTimeout(() => {
        router.push(`/waiting/${t.id}`);
      }, 100);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create table. Please try again.';
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
      if (typeof window !== 'undefined') {
        const existing = window.localStorage.getItem(STORAGE_KEYS.clientId(tableId));
        if (existing) {
          await persistTypedName(playerId, displayNameInput.trim());
          router.push(`/waiting/${tableId}`);
          return;
        }
      }

      const res = await fetch(`${API_BASE}/api/tables/${tableId}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: resolvedDisplayName, player_id: playerId }),
      });

      if (!res.ok) {
        throw new Error(`Failed to join table: ${res.status} ${res.statusText}`);
      }

      const data = await res.json();
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(STORAGE_KEYS.clientId(tableId), data.client_id);
      }
      const typedName = displayNameInput.trim();
      await persistIdentityFromJoin(data, typedName);

      // Use setTimeout for consistent navigation behavior
      setTimeout(() => {
        router.push(`/waiting/${data.table.id}`);
      }, 100);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to join table. Please try again.';
      setError(errorMessage);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.pageContent}>
        <h1>Sheepshead AI</h1>
        {error && (
          <div style={{
            background: 'rgba(255,50,50,0.1)',
            border: '1px solid rgba(255,50,50,0.3)',
            borderRadius: '8px',
            padding: '12px',
            marginBottom: '16px',
            color: 'white',
            fontSize: '14px'
          }}>
            {error}
          </div>
        )}
        <div className={styles.controls}>
          <input
            value={name}
            onChange={e => handleTableNameChange(e.target.value)}
            placeholder="Table name"
            disabled={creating}
          />
          <button
            onClick={handleCreateClick}
            disabled={creating || !name.trim() || !displayName.trim()}
            suppressHydrationWarning
            style={{
              opacity: creating ? 0.7 : 1,
              cursor: creating ? 'not-allowed' : 'pointer',
              WebkitTapHighlightColor: 'transparent',
              touchAction: 'manipulation'
            }}
          >
            {creating ? 'Creating...' : 'Create table'}
          </button>
          <span className={styles.nameRight}>
            Your name: <input
              value={displayNameInput}
              placeholder={displayPlaceholder ?? 'Your name'}
              onChange={e => setDisplayNameInput(e.target.value)}
              disabled={creating}
            />
          </span>
        </div>
        <div className={styles.lobby}>
          <h2>Lobby</h2>
          {loading ? (<div>Loading…</div>) : (
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.thLeft}>Name</th>
                  <th>Status</th>
                  <th>Players</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tables.map(t => {
                  const humanFilled = Object.entries(t.seats || {}).reduce((acc, [k, name]) => {
                    const isAI = (t.seatIsAI as any)?.[k] ?? false;
                    return acc + (name && !isAI ? 1 : 0);
                  }, 0);
                  const canJoin = humanFilled < 5;
                  return (
                    <tr key={t.id}>
                      <td>{t.name}</td>
                      <td>{t.status}</td>
                      <td>{humanFilled}/5</td>
                      <td>
                        <button onClick={() => join(t.id)} disabled={!canJoin}>Join</button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        <div style={{
          textAlign: 'center',
          marginTop: '3rem',
          paddingTop: '2rem',
          borderTop: '1px solid #e2e8f0'
        }}>
          <a
            href="/analyze"
            style={{
              color: '#64748b',
              fontSize: '0.875rem',
              textDecoration: 'none',
              transition: 'color 0.2s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.color = '#3b82f6'}
            onMouseOut={(e) => e.currentTarget.style.color = '#64748b'}
          >
            🧠 Analyze AI model decisions
          </a>
        </div>
      </div>
    </div>
  );
}


