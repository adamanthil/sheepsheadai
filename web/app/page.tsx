"use client";

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { TableSummary } from '../lib/types';
import styles from './page.module.css';

// Using shared type from web/lib/types

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:9000';

export default function HomePage() {
  const router = useRouter();
  const [tables, setTables] = useState<TableSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState('Quick Table');
  const [displayName, setDisplayName] = useState('Player');
  const [creating, setCreating] = useState(false);
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

  const create = async () => {
    setCreating(true);
    try {
      const res = await fetch(`${API_BASE}/api/tables`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, fillWithAI: true, rules: { partnerMode: 1, doubleOnTheBump: true } }),
      });
      const t = await res.json();
      // Auto-join and go to waiting room
      const res2 = await fetch(`${API_BASE}/api/tables/${t.id}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: displayName }),
      });
      const joined = await res2.json();
      router.push(`/waiting/${t.id}?client_id=${joined.client_id}`);
    } finally {
      setCreating(false);
    }
  };

  const join = async (tableId: string) => {
    const res = await fetch(`${API_BASE}/api/tables/${tableId}/join`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: displayName }),
    });
    const data = await res.json();
    setJoinInfo(data);
    router.push(`/waiting/${data.table.id}?client_id=${data.client_id}`);
  };

  return (
    <div className={styles.page}>
      <h1>Sheepshead AI</h1>
      <div className={styles.controls}>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="Table name" />
        <button onClick={create} disabled={creating}>Create table</button>
        <span className={styles.nameRight}>
          Your name: <input value={displayName} onChange={e => setDisplayName(e.target.value)} />
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


