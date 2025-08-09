"use client";

import { useEffect, useMemo, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:9000';

type TableInfo = {
  id: string;
  name: string;
  status: 'open' | 'playing' | 'finished';
  seats: Record<string, string | null>; // seat -> name
  host: string | null;
  runningBySeat?: Record<string, number>;
};

export default function WaitingRoom() {
  const params = useParams<{ id: string }>();
  const search = useSearchParams();
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
    window.location.href = `/table/${params?.id}?client_id=${clientId}`;
  }

  const isHost = useMemo(() => {
    if (!table) return false;
    return (table as any).hostId ? true : false;
  }, [table]);

  const seatItems = useMemo(() => {
    if (!table) return [] as Array<{ seat: number; name: string | null }>;
    const out: Array<{ seat: number; name: string | null }>= [];
    for (let i = 1; i <= 5; i++) out.push({ seat: i, name: table.seats[String(i)] || null });
    return out;
  }, [table]);

  return (
    <div style={{ minHeight: '100vh', color: 'white', background: 'radial-gradient(1200px 700px at 50% -200px, #1f3b08 0%, #0b1d08 60%, #061106 100%)' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto', padding: 24 }}>
        <h2 style={{ marginTop: 0 }}>Waiting Room · {table?.name || params?.id}</h2>
        {error && <div style={{ padding: '8px 12px', background: 'rgba(255,50,50,0.1)', border: '1px solid rgba(255,50,50,0.3)', borderRadius: 8, marginBottom: 10 }}>{error}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
          {seatItems.map(({ seat, name }) => (
            <div key={seat} style={{ padding: 16, background: 'rgba(0,0,0,0.35)', borderRadius: 12, textAlign: 'center', border: '1px solid rgba(255,255,255,0.15)' }}>
              <div style={{ opacity: 0.9, marginBottom: 6 }}>Seat {seat}</div>
              <div style={{ minHeight: 24, fontWeight: 600, fontSize: 16 }}>{name || '—'}</div>
              {!name && (
                <button onClick={() => chooseSeat(seat)} style={{ marginTop: 8, padding: '6px 10px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.25)', background: 'transparent', color: 'white' }}>Take seat</button>
              )}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 12, marginTop: 16, alignItems: 'center' }}>
          <button onClick={fillAI} style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.25)', background: 'transparent', color: 'white' }}>Fill empty seats with AI</button>
          {isHost && (<button onClick={startGame} style={{ padding: '10px 14px', borderRadius: 12, border: '1px solid rgba(255,255,255,0.35)', background: 'linear-gradient(180deg, rgba(255,255,255,0.25), rgba(255,255,255,0.1))', color: 'white' }}>Start</button>)}
          <div style={{ marginLeft: 'auto', opacity: 0.85 }}>Players: {seatItems.filter(s => !!s.name).length}/5</div>
        </div>

        <div style={{ marginTop: 24, padding: 16, borderRadius: 12, background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.15)' }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Running totals</div>
          {table?.runningBySeat ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
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

