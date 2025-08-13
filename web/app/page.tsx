"use client";

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { TableSummary } from '../lib/types';
import styles from './page.module.css';

// Using shared type from web/lib/types

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || (() => {
  if (typeof window === 'undefined') return 'http://localhost:9000';

  // Use the same hostname as the frontend, but with backend port
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  return `${protocol}//${hostname}:9000`;
})();

export default function HomePage() {
  const router = useRouter();
  const [tables, setTables] = useState<TableSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState('Quick Table');
  const [displayName, setDisplayName] = useState('Player');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [joinInfo, setJoinInfo] = useState<{ client_id: string; seat: number; table: TableSummary } | null>(null);

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

  // Mobile Safari compatible click handler
  const handleCreateClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    create();
  };

  const create = async () => {
    // Clear any previous errors
    setError(null);
    setCreating(true);

    try {
      console.log('Creating table...', { API_BASE, name, displayName }); // Debug log for mobile Safari

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
      console.log('Table created:', t.id); // Debug log

      // Auto-join the table
      const res2 = await fetch(`${API_BASE}/api/tables/${t.id}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: displayName }),
      });

      if (!res2.ok) {
        throw new Error(`Failed to join table: ${res2.status} ${res2.statusText}`);
      }

      const joined = await res2.json();
      console.log('Joined table, navigating...'); // Debug log

      // Use setTimeout to ensure state updates complete before navigation
      setTimeout(() => {
        router.push(`/waiting/${t.id}?client_id=${joined.client_id}`);
      }, 100);

    } catch (err) {
      console.error('Create table error:', err); // Debug log
      const errorMessage = err instanceof Error ? err.message : 'Failed to create table. Please try again.';
      setError(errorMessage);
    } finally {
      setCreating(false);
    }
  };

  const join = async (tableId: string) => {
    setError(null);

    try {
      console.log('Joining table:', tableId); // Debug log

      const res = await fetch(`${API_BASE}/api/tables/${tableId}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: displayName }),
      });

      if (!res.ok) {
        throw new Error(`Failed to join table: ${res.status} ${res.statusText}`);
      }

      const data = await res.json();
      setJoinInfo(data);

      console.log('Joined table, navigating...'); // Debug log

      // Use setTimeout for consistent navigation behavior
      setTimeout(() => {
        router.push(`/waiting/${data.table.id}?client_id=${data.client_id}`);
      }, 100);

    } catch (err) {
      console.error('Join table error:', err); // Debug log
      const errorMessage = err instanceof Error ? err.message : 'Failed to join table. Please try again.';
      setError(errorMessage);
    }
  };

  return (
    <div className={styles.page}>
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
          onChange={e => setName(e.target.value)}
          placeholder="Table name"
          disabled={creating}
        />
        <button
          onClick={handleCreateClick}
          disabled={creating || !name.trim() || !displayName.trim()}
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
            value={displayName}
            onChange={e => setDisplayName(e.target.value)}
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
                <th>Seats</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {tables.map(t => {
                const filled = Object.values(t.seats).filter(Boolean).length;
                return (
                  <tr key={t.id}>
                    <td>{t.name}</td>
                    <td>{t.status}</td>
                    <td>{filled}/5</td>
                    <td>
                      <button onClick={() => join(t.id)} disabled={t.status !== 'open'}>Join</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}


