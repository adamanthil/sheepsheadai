"use client";

import { useEffect, useMemo, useRef, useState } from 'react';
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
  const [partnerMode, setPartnerMode] = useState<number>(1); // 1 = Called Ace (default), 0 = Jack of Diamonds
  const wsRef = useRef<WebSocket | null>(null);
  const [callout, setCallout] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tables`);
      const list: TableInfo[] = await res.json();
      const t = list.find(t => t.id === params?.id);
      if (t) {
        setTable(t);
        // Initialize partner mode from table rules
        const rules = (t as any).rules || {};
        const mode = rules.partnerMode !== undefined ? rules.partnerMode : 1; // Default to Called Ace
        setPartnerMode(mode);
      }
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [params?.id]);

  // Connect WS in waiting room to receive lobby updates and auto-navigate on start
  useEffect(() => {
    if (!params?.id || !clientId) return;
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.host.split(':')[0];
    const wsUrl = `${proto}://${host}:9000/ws/table/${params.id}?client_id=${clientId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data?.type === 'table_update') {
          const t = data.table as TableInfo;
          setTable(t);
          const rules = (t as any).rules || {};
          if (rules.partnerMode !== undefined) setPartnerMode(rules.partnerMode);
          if (t.status === 'playing') {
            router.push(`/table/${params.id}?client_id=${clientId}`);
          }
        } else if (data?.type === 'lobby_event') {
          setCallout(String(data.message || ''));
          setTimeout(() => setCallout(null), 1800);
        } else if (data?.type === 'state') {
          const t = data.table as TableInfo;
          // If we receive state while in waiting room, the game has started
          if (t?.status === 'playing') {
            router.push(`/table/${params.id}?client_id=${clientId}`);
          }
        }
      } catch (e) {
        console.error('WS message parse error', e);
      }
    };
    ws.onerror = (e) => {
      console.error('WS error', e);
    };
    return () => {
      try { ws.close(); } catch {}
      wsRef.current = null;
    };
  }, [params?.id, clientId]);

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
    const res = await fetch(`${API_BASE}/api/tables/${params?.id}/start_waiting`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId })
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      setError(j?.detail || 'Failed to fill seats with AI');
      return;
    }
    const t = await res.json();
    setTable(t);
  }

  async function updatePartnerMode(newMode: number) {
    try {
      const res = await fetch(`${API_BASE}/api/tables/${params?.id}/rules`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: clientId, rules: { partnerMode: newMode } })
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        setError(j?.detail || 'Failed to update partner mode');
        return;
      }
      setPartnerMode(newMode);
    } catch (e: any) {
      setError('Failed to update partner mode');
    }
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
    if (!table || !clientId) return false;
    return (table as any).hostId === clientId;
  }, [table, clientId]);

  const seatItems = useMemo(() => {
    if (!table) return [] as Array<{ seat: number; name: string | null; isAI: boolean }>;
    const out: Array<{ seat: number; name: string | null; isAI: boolean }>= [];
    for (let i = 1; i <= 5; i++) out.push({ seat: i, name: table.seats[String(i)] || null, isAI: Boolean((table as any).seatIsAI?.[String(i)]) });
    return out;
  }, [table]);

  return (
    <div className={styles.root}>
      <div className={styles.container}>
        {callout && (
          <div className={styles.callout}>{callout}</div>
        )}
        <h2>Waiting Room · {table?.name || params?.id}</h2>
        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.seatsGrid}>
          {seatItems.map(({ seat, name, isAI }) => (
            <div key={seat} className={styles.seatCard}>
              <div style={{ opacity: 0.9, marginBottom: 6 }}>Seat {seat}</div>
              <div style={{ minHeight: 24, fontWeight: 600, fontSize: 16 }}>
                {name ? (
                  <span className={styles.nameWithTag}>
                    <span>{name}</span>
                    {isAI && (
                      <span className={styles.aiTag}>AI</span>
                    )}
                  </span>
                ) : '—'}
              </div>
              {(!name || isAI) && (
                <button className={`${styles.btn} ${styles.btnSmall}`} onClick={() => chooseSeat(seat)}>Take seat</button>
              )}
            </div>
          ))}
        </div>

        <div className={styles.partnerModeSection}>
          <div className={styles.partnerModeLabel}>Partner Selection Mode</div>
          <div className={styles.partnerModeToggle}>
            <button
              className={`${styles.toggleOption} ${partnerMode === 1 ? styles.toggleActive : ''}`}
              onClick={() => updatePartnerMode(1)}
            >
              Called Ace
            </button>
            <button
              className={`${styles.toggleOption} ${partnerMode === 0 ? styles.toggleActive : ''}`}
              onClick={() => updatePartnerMode(0)}
            >
              Jack of Diamonds
            </button>
          </div>
          <div className={styles.partnerModeDescription}>
            {partnerMode === 1
              ? "The picker calls a fail Ace to find their partner"
              : "The player with the Jack of Diamonds is the partner"
            }
          </div>
        </div>

        <div className={styles.actionsRow}>
          {isHost && (
            <button className={styles.btn} onClick={fillAI}>Fill empty seats with AI</button>
          )}
          {isHost && (<button className={`${styles.btn} ${styles.btnPrimary}`} onClick={startGame}>Start</button>)}
          <div className={styles.muted} style={{ marginLeft: 'auto' }}>Players: {seatItems.filter(s => !!s.name).length}/5</div>
        </div>
      </div>
    </div>
  );
}

