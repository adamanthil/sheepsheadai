"use client";

import { useEffect, useMemo, useState } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import type { TableSummary } from '../../../lib/types';
import styles from './page.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || (() => {
  if (typeof window === 'undefined') return 'http://localhost:9000';

  // Use the same hostname as the frontend, but with backend port
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  return `${protocol}//${hostname}:9000`;
})();

type TableInfo = TableSummary & {
  seats: Record<string, string | null>;
};

export default function WaitingRoom() {
  const params = useParams<{ id: string }>();
  const search = useSearchParams();
  const router = useRouter();
  const clientId = search.get('client_id') || '';

  const [table, setTable] = useState<TableInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tables`);
      const list: TableInfo[] = await res.json();
      const t = list.find(t => t.id === params?.id);
      if (t) setTable(t);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [params?.id]);

  async function chooseSeat(seat: number) {
    const res = await fetch(`${API_BASE}/api/tables/${params?.id}/seat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId, seat })
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j?.detail || 'Seat selection failed');
      return;
    }
    const t = await res.json();
    setTable(t);
  }

  async function fillAI() {
    const res = await fetch(`${API_BASE}/api/tables/${params?.id}/start_waiting`, { method: 'POST' });
    const t = await res.json();
    setTable(t);
  }

  async function startGame() {
    const res = await fetch(`${API_BASE}/api/tables/${params?.id}/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ client_id: clientId }) });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j?.detail || 'Start failed');
      return;
    }
    router.push(`/table/${params?.id}?client_id=${clientId}`);
  }

  const isHost = useMemo(() => {
    if (!table) return false;
    return Boolean((table as any).hostId);
  }, [table]);

  const seatItems = useMemo(() => {
    if (!table) return [] as Array<{ seat: number; name: string | null }>;
    const out: Array<{ seat: number; name: string | null }>= [];
    for (let i = 1; i <= 5; i++) out.push({ seat: i, name: table.seats[String(i)] || null });
    return out;
  }, [table]);

  return (
    <div className={styles.root}>
      <div className={styles.container}>
        <h2>Waiting Room · {table?.name || params?.id}</h2>
        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.seatsGrid}>
          {seatItems.map(({ seat, name }) => (
            <div key={seat} className={styles.seatCard}>
              <div style={{ opacity: 0.9, marginBottom: 6 }}>Seat {seat}</div>
              <div style={{ minHeight: 24, fontWeight: 600, fontSize: 16 }}>{name || '—'}</div>
              {!name && (
                <button className={`${styles.btn} ${styles.btnSmall}`} onClick={() => chooseSeat(seat)}>Take seat</button>
              )}
            </div>
          ))}
        </div>

        <div className={styles.actionsRow}>
          <button className={styles.btn} onClick={fillAI}>Fill empty seats with AI</button>
          {isHost && (<button className={`${styles.btn} ${styles.btnPrimary}`} onClick={startGame}>Start</button>)}
          <div className={styles.muted} style={{ marginLeft: 'auto' }}>Players: {seatItems.filter(s => !!s.name).length}/5</div>
        </div>

          <div className={styles.runTotals}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Running totals</div>
          {table?.runningBySeat ? (
            <div className={styles.runTotalsGrid}>
              {Array.from({ length: 5 }, (_, i) => i + 1).map(seat => (
                <div key={seat} style={{ textAlign: 'center', opacity: 0.95 }}>
                  <div style={{ fontSize: 13, opacity: 0.85 }}>Seat {seat}</div>
                  <div style={{ fontWeight: 700 }}>{table.runningBySeat![String(seat)] || 0}</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ opacity: 0.8 }}>No games completed yet</div>
          )}
        </div>
      </div>
    </div>
  );
}

