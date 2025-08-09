"use client";

import { useEffect, useMemo, useState } from 'react';

type TableSummary = {
  id: string;
  name: string;
  status: 'open' | 'playing' | 'finished';
  rules: Record<string, any>;
  fillWithAI: boolean;
  seats: Record<string, string | null>;
  host: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:9000';

export default function HomePage() {
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
      window.location.href = `/waiting/${t.id}?client_id=${joined.client_id}`;
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
    window.location.href = `/waiting/${data.table.id}?client_id=${data.client_id}`;
  };

  return (
    <div style={{ padding: 24, maxWidth: 960, margin: '0 auto' }}>
      <h1>Sheepshead AI</h1>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="Table name" />
        <button onClick={create} disabled={creating}>Create table</button>
        <span style={{ marginLeft: 'auto' }}>
          Your name: <input value={displayName} onChange={e => setDisplayName(e.target.value)} />
        </span>
      </div>
      <div style={{ marginTop: 24 }}>
        <h2>Lobby</h2>
        {loading ? (<div>Loading…</div>) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Name</th>
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


