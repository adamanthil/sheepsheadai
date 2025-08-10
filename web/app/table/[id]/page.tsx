"use client";
import styles from './page.module.css';
import cardStyles from './PlayingCard.module.css';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams, useParams } from 'next/navigation';
import type { TableStateMsg, GameOverMsg } from '../../../lib/types';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:9000';

// --- Card helpers (module-level so both components can use) ---
function parseCard(card: string): { rank: string; suit: 'C'|'S'|'H'|'D'|null } {
  if (!card || card === '__' || card === 'UNDER') return { rank: card, suit: null };
  if (card.startsWith('10')) return { rank: '10', suit: card.slice(2) as any };
  return { rank: card[0], suit: card[1] as any };
}

function suitSymbol(s: 'C'|'S'|'H'|'D'|null) {
  if (s === 'C') return '♣';
  if (s === 'S') return '♠';
  if (s === 'H') return '♥';
  if (s === 'D') return '♦';
  return '';
}

// Seat math + positions used by table layout and trick animations
export function relSeat(absSeat: number, me: number) {
  return (absSeat - me + 5) % 5;
}

export function spotStyle(rel: number): React.CSSProperties {
  const base: React.CSSProperties = { position: 'absolute', transform: 'translate(-50%, -50%)' };
  const map: Record<number, React.CSSProperties> = {
    0: { left: '50%', top: '84%' },
    1: { left: '17%', top: '66%' },
    2: { left: '27%', top: '20%' },
    3: { left: '73%', top: '20%' },
    4: { left: '83%', top: '66%' },
  };
  return { ...base, ...map[rel] };
}

// Using shared types from web/lib/types

export default function TablePage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const clientId = searchParams.get('client_id') || '';
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastState, setLastState] = useState<TableStateMsg | null>(null);
  const [actionLookup, setActionLookup] = useState<Record<string, string>>({}); // id -> label
  const startTried = useRef(false);
  const [showPrev, setShowPrev] = useState(false);
  const [animTrick, setAnimTrick] = useState<{ cards: string[]; winner: number } | null>(null);
  const animTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pauseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const handRowRef = useRef<HTMLDivElement | null>(null);
  const [handSize, setHandSize] = useState<{ w: number; h: number }>({ w: 72, h: 108 });
  const [centerSize, setCenterSize] = useState<{ w: number; h: number }>({ w: 92, h: 138 });
  const trickBoxRef = useRef<HTMLDivElement | null>(null);
  const [trickSize, setTrickSize] = useState<{ w: number; h: number }>({ w: 900, h: 400 });
  const handTopMargin = Math.max(32, Math.floor(centerSize.h * 0.2));
  const [showScores, setShowScores] = useState(false);
  const [callout, setCallout] = useState<{ kind: 'PICK' | 'CALL' | 'LEASTER' | 'ALONE'; message: string } | null>(null);

  useEffect(() => {
    (async () => {
      const res = await fetch(`${API_BASE}/api/actions`);
      const data = await res.json();
      setActionLookup(data.action_lookup);
    })();
  }, []);

  useEffect(() => {
    if (!params?.id || !clientId) return;
    const api = new URL(API_BASE);
    const wsProto = api.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${api.host}/ws/table/${params.id}?client_id=${encodeURIComponent(clientId)}`;
    const socket = new WebSocket(wsUrl);
    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as TableStateMsg | GameOverMsg;
      if (msg.type === 'state') {
        setLastState((prev: TableStateMsg | null) => {
          const prevWas = prev?.view?.was_trick_just_completed;
          const curWas = (msg as any).view?.was_trick_just_completed;
          // On trick completion: first pause showing previous trick, then animate, then hide
          if (curWas && !prevWas) {
            const last = (msg as any).view?.last_trick as string[] | null;
            const win = (msg as any).view?.last_trick_winner as number;
            if (last && win) {
              setShowPrev(true);
              // clear timers
              if (animTimerRef.current) clearTimeout(animTimerRef.current);
              if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
              // Start animation AFTER the viewing pause
              pauseTimerRef.current = setTimeout(() => {
                setAnimTrick({ cards: last, winner: win });
                animTimerRef.current = setTimeout(() => {
                  setAnimTrick(null);
                  setShowPrev(false);
                }, 1300);
              }, 2000);
            }
          }
          // Detect important phase callouts
          const prevPicker = prev?.view?.picker || 0;
          const curPicker = (msg as any).view?.picker || 0;
          const prevLeaster = !!prev?.view?.is_leaster;
          const curLeaster = !!(msg as any).view?.is_leaster;
          const prevCalled = prev?.view?.called_card || null;
          const curCalled = (msg as any).view?.called_card || null;
          const curCalledUnder = !!(msg as any).view?.called_under;
          const prevAlone = !!prev?.view?.alone;
          const curAlone = !!(msg as any).view?.alone;
          const playStarted = !!((msg as any).state?.[14] === 1);
          if (curPicker > 0 && prevPicker === 0) {
            const who = nameForSeat(curPicker, (msg as any).table);
            setCallout({ kind: 'PICK', message: `${who} picked` });
            setTimeout(() => setCallout(null), 1800);
          } else if (!prevLeaster && curLeaster) {
            setCallout({ kind: 'LEASTER', message: 'All passed · Leaster' });
            setTimeout(() => setCallout(null), 1800);
          } else if (curPicker > 0 && !playStarted && !prevAlone && curAlone) {
            const who = nameForSeat(curPicker, (msg as any).table);
            setCallout({ kind: 'ALONE', message: `${who} goes alone` });
            setTimeout(() => setCallout(null), 1800);
          } else if (curPicker > 0 && !playStarted && curCalled && prevCalled !== curCalled) {
            const who = nameForSeat(curPicker, (msg as any).table);
            const underTxt = curCalledUnder ? ' under' : '';
            setCallout({ kind: 'CALL', message: `${who} calls ${curCalled}${underTxt}` });
            setTimeout(() => setCallout(null), 1800);
          }
          return msg;
        });
      }
      // game_over message is informational; state messages keep coming
    };
    setWs(socket);
    return () => socket.close();
  }, [params?.id, clientId]);

  // Do not auto-start; host will start from waiting room

  const takeAction = async (actionId: number) => {
    // Prefer HTTP action to avoid WS fragility
    const cid = clientId;
    if (!cid) return;
    await fetch(`${API_BASE}/api/tables/${params.id}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: cid, action_id: actionId })
    }).catch(() => {});
  };

  const actionIdByString = useMemo(() => {
    const m: Record<string, number> = {};
    Object.entries(actionLookup).forEach(([id, label]) => {
      const n = Number(id);
      if (Number.isFinite(n)) m[label as string] = n;
    });
    return m;
  }, [actionLookup]);

  const validActionStrings = useMemo(() => {
    if (!lastState) return new Set<string>();
    const s = new Set<string>();
    for (const id of lastState.valid_actions) {
      const k = String(id);
      const label = actionLookup[k];
      if (label) s.add(label);
    }
    return s;
  }, [lastState, actionLookup]);

  const isYourTurn = lastState && lastState.actorSeat === lastState.yourSeat;

  function nameForSeat(seat: number | null, table: any): string {
    if (!seat) return '';
    return table?.seats?.[String(seat)] || `Seat ${seat}`;
  }

  // Picking round helpers (visualize PASS/PICK/PENDING)
  const lastPassed = useMemo(() => {
    if (!lastState) return 0;
    const v = lastState.state?.[2];
    const n = typeof v === 'number' ? Math.floor(v) : 0;
    return Number.isFinite(n) ? n : 0;
  }, [lastState]);

  // Clear timers on unmount
  useEffect(() => {
    return () => {
      if (animTimerRef.current) clearTimeout(animTimerRef.current);
      if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
    };
  }, []);

  const inPickDecision = !!lastState && !lastState.view.is_leaster && (lastState.view.picker === 0);
  const playStarted = useMemo(() => {
    if (!lastState) return false;
    const v = lastState.state?.[14];
    const n = typeof v === 'number' ? Math.floor(v) : 0;
    return n === 1;
  }, [lastState]);

  function pickStatusForSeat(absSeat: number): 'PASS' | 'PICK' | 'PENDING' | null {
    if (!lastState) return null;
    if (playStarted) return null; // persist badges only until play starts
    const picker = lastState.view.picker || 0;
    if (picker > 0) {
      if (absSeat === picker) return 'PICK';
      if (absSeat <= lastPassed) return 'PASS';
      return null;
    }
    if (!inPickDecision) return null;
    if (absSeat <= lastPassed) return 'PASS';
    const nextSeat = (lastPassed % 5) + 1;
    if (absSeat === nextSeat) return 'PENDING';
    return null;
  }

  const actionButtons = useMemo(() => {
    if (!lastState) return null;
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {lastState.valid_actions.map((aid: number) => (
          <button key={aid} onClick={() => takeAction(aid)}>
            {actionLookup[String(aid)] || `Action ${aid}`}
          </button>
        ))}
      </div>
    );
  }, [lastState, actionLookup]);

  async function redeal() {
    // Host-only: require client_id; backend will enforce
    const id = params?.id;
    if (!id || !clientId) return;
    await fetch(`${API_BASE}/api/tables/${id}/redeal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId })
    });
    // Immediately start a new hand after redeal
    await fetch(`${API_BASE}/api/tables/${id}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId })
    }).catch(() => {});
  }

  // --- Layout helpers ---


  function actionIdFor(label: string): number | null {
    const id = actionIdByString[label];
    return typeof id === 'number' ? id : null;
  }

  function handleCardClick(card: string) {
    if (!isYourTurn || !card) return;
    // Try PLAY first, then BURY, then UNDER
    const candidates = [`PLAY ${card}`, `BURY ${card}`, `UNDER ${card}`];
    for (const lbl of candidates) {
      if (validActionStrings.has(lbl)) {
        const id = actionIdFor(lbl);
        if (id) void takeAction(id);
        return;
      }
    }
  }

  function isCardClickable(card: string) {
    if (!isYourTurn) return false;
    return validActionStrings.has(`PLAY ${card}`) ||
           validActionStrings.has(`BURY ${card}`) ||
           validActionStrings.has(`UNDER ${card}`);
  }

  // --- Responsive sizing: fit hand to a single row ---
  useEffect(() => {
    function recalc() {
      const row = handRowRef.current;
      if (!row || !lastState) return;
      const count = Math.max(1, (lastState.view.hand as string[]).length || 6);
      const gap = 10;
      // Use viewport width to ensure single-row fitting across full width
      const vw = Math.max(640, Math.min(window.innerWidth * 0.94, 1600));
      const avail = vw - 40; // small padding
      let w = Math.floor((avail - (count - 1) * gap) / count);
      w = Math.max(64, Math.min(140, w));
      const h = Math.floor(w * 1.45);
      setHandSize({ w, h });
      // Center cards should be larger for emphasis but not overlapping
      const cw = Math.floor(w * 1.5);
      const ch = Math.floor(cw * 1.45);
      setCenterSize({ w: cw, h: ch });
    }
    recalc();
    window.addEventListener('resize', recalc);
    return () => window.removeEventListener('resize', recalc);
  }, [lastState]);

  // Measure trick box size for motion path math
  useEffect(() => {
    function recalcTrick() {
      const el = trickBoxRef.current;
      if (!el) return;
      setTrickSize({ w: el.clientWidth, h: el.clientHeight });
    }
    recalcTrick();
    window.addEventListener('resize', recalcTrick);
    return () => window.removeEventListener('resize', recalcTrick);
  }, []);

  return (
    <div className={styles.root}>
      <div className={styles.wrap}>
        <div className={styles.topRow}>
          <h2>Table {params?.id}</h2>
          <div style={{ opacity: 0.8 }}>Connection: {connected ? 'connected' : 'disconnected'}</div>
        </div>

        {!lastState ? (
          <div style={{ marginTop: 48, textAlign: 'center', opacity: 0.9 }}>Waiting for state…</div>
        ) : (
          <div className={styles.tableArea}>
              {/* Table area full width */}
              <div className={styles.tableFrame}>
                {/* Current trick in table layout */}
                <div className={styles.trickHeader}>Trick #{(lastState.view.current_trick_index || 0) + 1}</div>
                <div
                  id="trick-container"
                  ref={trickBoxRef}
                  className={styles.trickContainer}
                  style={{ ['--trickH' as any]: `${Math.max(280, Math.floor(centerSize.h * 1.7))}px`, ['--cardW' as any]: `${centerSize.w}px` }}
                >
                  {(() => {
                    const isPrev = !!showPrev;
                    const cards: string[] = isPrev ? (lastState.view.last_trick || []) : lastState.view.current_trick;
                    const winnerSeat = isPrev ? lastState.view.last_trick_winner : null;
                    return cards.map((c: string, idx: number) => {
                      const absSeat = idx + 1;
                      const r = relSeat(absSeat, lastState.yourSeat);
                      const highlight = isPrev && winnerSeat === absSeat;
                      const name = lastState.table?.seats?.[String(absSeat)] || `Seat ${absSeat}`;
                      return (
                        <div key={idx} style={{ ...spotStyle(r), width: centerSize.w }}>
                          <PlayingCard label={c || '__'} width={centerSize.w} height={centerSize.h} highlight={highlight} />
                          {r === 2 ? (
                            <div className={styles.nameLeftOfCard}>{name}</div>
                          ) : r === 3 ? (
                            <div className={styles.nameRightOfCard}>{name}</div>
                          ) : (
                            <div className={styles.nameBelow}>{name}</div>
                          )}
                          {(() => {
                            const status = pickStatusForSeat(absSeat);
                            if (!status) return null;
                            const color = status === 'PICK' ? 'rgba(34,197,94,0.25)'
                              : status === 'PASS' ? 'rgba(239,68,68,0.22)'
                              : 'rgba(234,179,8,0.22)';
                            const border = status === 'PICK' ? '1px solid rgba(34,197,94,0.5)'
                              : status === 'PASS' ? '1px solid rgba(239,68,68,0.45)'
                              : '1px solid rgba(234,179,8,0.45)';
                            const text = status;
                            return (
                              <div style={{ marginTop: 4, textAlign: 'center' }}>
                                <span style={{ display: 'inline-block', padding: '2px 6px', borderRadius: 999, background: color, border, fontSize: 10, letterSpacing: 0.3 }}>{text}</span>
                              </div>
                            );
                          })()}
                        </div>
                      );
                    });
                  })()}

                  {/* Previous trick info banner when toggled */}
                  {showPrev && !animTrick && lastState.view.last_trick ? (
                    <div className={styles.prevBanner}>
                      Previous trick · Winner: {nameForSeat(lastState.view.last_trick_winner, lastState.table)} · Points: {lastState.view.last_trick_points}
                    </div>
                  ) : null}

                  {/* Trick collect animation overlay */}
                  {animTrick && (
                    <CollectOverlay containerRef={trickBoxRef} yourSeat={lastState.yourSeat} winner={animTrick.winner} cards={animTrick.cards} centerSize={centerSize} />
                  )}
                  {callout && (
                    <div className={styles.callout}>
                      <div style={{ fontSize: 16, fontWeight: 800, letterSpacing: 0.3 }}>
                        {callout.message}
                      </div>
                    </div>
                  )}
                </div>

                {/* Persistent previous strip removed in favor of toggle */}

                {/* Your hand */}
                <div style={{ marginTop: handTopMargin }}>
                  <div className={styles.sectionTitle}>Your hand</div>
                  <div ref={handRowRef} className={styles.handRow}>
                    {lastState.view.hand.map((c: string, idx: number) => {
                      const clickable = isCardClickable(c);
                      return (
                        <div key={idx} onClick={() => clickable && handleCardClick(c)} style={{ cursor: clickable ? 'pointer' : 'default' }}>
                          <PlayingCard label={c} highlight={clickable} width={handSize.w} height={handSize.h} bigMarks />
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className={styles.muted} style={{ marginTop: 16, fontSize: 13 }}>
                  Picker: {nameForSeat(lastState.view.picker, lastState.table)} · Partner: {nameForSeat(lastState.view.partner, lastState.table)} · Alone: {String(lastState.view.alone)} · Called: {lastState.view.called_card}{lastState.view.called_under ? ' (under)' : ''}
                </div>

                {/* Final banner */}
                {lastState.view.is_done && lastState.view.final ? (
                  <div className={styles.finalBanner}>
                    {lastState.view.final.mode === 'leaster' ? (
                      <div>
                        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Game Over · Leaster</div>
                        <div style={{ opacity: 0.95 }}>Winner: <strong>{nameForSeat(lastState.view.final.winner, lastState.table)}</strong></div>
                        <div style={{ opacity: 0.95, marginTop: 8 }}>Scores by player</div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginTop: 6 }}>
                          {Array.from({ length: 5 }, (_, i) => i + 1).map((seat) => (
                            <div key={seat} style={{ textAlign: 'center' }}>
                              <div style={{ fontSize: 12, opacity: 0.85 }}>{nameForSeat(seat, lastState.table)}</div>
                              <div style={{ fontWeight: 700 }}>{(lastState.view.final?.scores && lastState.view.final.scores[seat - 1]) || 0}</div>
                            </div>
                          ))}
                        </div>
                        <div className={styles.muted} style={{ marginTop: 8 }}>Points taken by player</div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginTop: 6 }}>
                          {Array.from({ length: 5 }, (_, i) => i + 1).map((seat) => (
                            <div key={seat} style={{ textAlign: 'center' }}>
                              <div style={{ fontSize: 12, opacity: 0.85 }}>{nameForSeat(seat, lastState.table)}</div>
                              <div style={{ fontWeight: 700 }}>{(lastState.view.final?.points_taken && lastState.view.final.points_taken[seat - 1]) || 0}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Game Over</div>
                        <div className={styles.muted}>
                          Picker: <strong>{nameForSeat(lastState.view.final.picker, lastState.table)}</strong>
                          {" · "}
                          Partner: <strong>{nameForSeat(lastState.view.final.partner, lastState.table)}</strong>
                        </div>
                        <div className={styles.muted} style={{ marginTop: 4 }}>
                          Picker score: <strong>{lastState.view.final.picker_score}</strong> · Defenders score: <strong>{lastState.view.final.defender_score}</strong>
                        </div>
                        <div className={styles.muted} style={{ marginTop: 8 }}>Scores by player</div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginTop: 6 }}>
                          {Array.from({ length: 5 }, (_, i) => i + 1).map((seat) => (
                            <div key={seat} style={{ textAlign: 'center' }}>
                              <div style={{ fontSize: 12, opacity: 0.85 }}>{nameForSeat(seat, lastState.table)}</div>
                              <div style={{ fontWeight: 700 }}>{(lastState.view.final?.scores && lastState.view.final.scores[seat - 1]) || 0}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    <div style={{ marginTop: 8 }}>
                      <button onClick={() => redeal()} style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.25)', background: 'transparent', color: 'white' }}>Redeal</button>
                      <button onClick={() => setShowScores(true)} style={{ marginLeft: 8, padding: '8px 12px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.25)', background: 'transparent', color: 'white' }}>Show scores</button>
                    </div>
                  </div>
                ) : null}
              </div>
          </div>
        )}
      </div>

      {/* Bottom Action Bar */}
      {lastState && (
        <div className={styles.bottomBar}>
          <div className={styles.bottomBarInner}>
            <div className={styles.muted} style={{ fontSize: 13 }}>{nameForSeat(lastState.yourSeat, lastState.table)} {lastState.actorSeat === lastState.yourSeat ? '· Your turn' : ''}</div>
            {(lastState.view.last_trick && lastState.view.last_trick.length === 5) ? (
              <button onClick={() => setShowPrev(p => !p)}>{showPrev ? 'Hide previous trick' : 'Show previous trick'}</button>
            ) : null}
            {lastState.valid_actions.length ? actionButtons : (
              <div className={styles.dimmed}>
                Waiting for {nameForSeat(lastState.actorSeat, lastState.table) || `Seat ${lastState.actorSeat || ''}`}…
              </div>
            )}
            <div className={`${styles.pushRight} ${styles.dimmed}`} style={{ fontSize: 12 }}>
              Tricks: {lastState.view.trick_winners.filter((x: number) => x > 0).length}/6 · Points played so far: {lastState.view.trick_points.reduce((a: number, b: number) => a + (b || 0), 0)}
            </div>
            <button onClick={() => setShowScores(true)}>Scores</button>
          </div>
        </div>
      )}

      {/* Scores Overlay */}
      {lastState && showScores && (
        <ScoresOverlay onClose={() => setShowScores(false)} table={lastState.table} />
      )}
    </div>
  );
}

function PlayingCard({ label, small, highlight, width, height, bigMarks }: { label: string; small?: boolean; highlight?: boolean; width?: number; height?: number; bigMarks?: boolean }) {
  const blank = label === '__' || !label;
  const { rank, suit } = parseCard(label);
  const w = width ?? (small ? 48 : 76);
  const h = height ?? (small ? 64 : 108);
  const red = suit === 'H' || suit === 'D';
  const sizeClass = bigMarks ? 'big' : (small ? 'small' : 'normal');
  const classNames = [cardStyles.card, highlight ? cardStyles.highlight : '', blank ? cardStyles.blank : '', red ? cardStyles.redText : ''].filter(Boolean).join(' ');
  return (
    <div className={classNames} style={{ ['--w' as any]: `${w}px`, ['--h' as any]: `${h}px`, ['--pad' as any]: small ? '4px' : '8px' }}>
      <div className={`${cardStyles.rankTop} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>{rank}</div>
      <div className={`${cardStyles.suit} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>{suitSymbol(suit)}</div>
      <div className={`${cardStyles.rankBottom} ${cardStyles[sizeClass as 'small'|'normal'|'big']}`}>{rank}</div>
    </div>
  );
}

function ScoresOverlay({ onClose, table }: { onClose: () => void; table: any }) {
  const history: Array<any> = table?.resultsHistory || [];
  const seats = table?.seats || {};
  const rows = history.map((h, idx) => ({
    hand: h.hand || (idx + 1),
    bySeat: h.bySeat || {},
    sum: h.sum || 0,
  }));
  const initialOrder: string[] = (table?.initialSeatOrder || []).map((x: any) => String(x));
  const ids = table?.seatOccupants || {};
  const labelsById: Record<string, string> = {};
  for (let i = 1; i <= 5; i++) {
    const occ = String(ids[String(i)] || `seat-${i}`);
    labelsById[occ] = seats[String(i)] || `Seat ${i}`;
  }
  const columns: Array<{ id: string, label: string }> = (initialOrder.length === 5 ? initialOrder : Object.keys(labelsById)).map((id: string) => ({ id, label: labelsById[id] || id }));
  const scoreFor = (row: any, id: string) => {
    const entries = row.bySeat || {};
    for (const key of Object.keys(entries)) {
      const v = entries[key];
      if (v && String(v.id) === String(id)) return v.score || 0;
    }
    return 0;
  };
  const totalById: Record<string, number> = {};
  columns.forEach(c => { totalById[c.id] = 0; });
  rows.forEach(r => { columns.forEach(c => { totalById[c.id] += scoreFor(r, c.id); }); });
  const overallSum = Object.values(totalById).reduce((a, b) => a + (b || 0), 0);

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, background: 'rgba(0,0,0,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: 'min(1000px, 94vw)', maxHeight: '80vh', overflow: 'auto', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 12, color: 'white' }}>
        <div style={{ display: 'flex', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.15)' }}>
          <div style={{ fontWeight: 700 }}>Running totals</div>
          <div style={{ marginLeft: 'auto' }}>
            <button onClick={onClose} style={{ padding: '6px 10px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.25)', background: 'transparent', color: 'white' }}>Close</button>
          </div>
        </div>
        <div style={{ padding: 16 }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '6px 8px', borderBottom: '1px solid rgba(255,255,255,0.15)' }}>Hand</th>
                  {columns.map((c, idx) => (
                    <th key={idx} style={{ textAlign: 'right', padding: '6px 8px', borderBottom: '1px solid rgba(255,255,255,0.15)' }}>{c.label}</th>
                  ))}
                  <th style={{ textAlign: 'right', padding: '6px 8px', borderBottom: '1px solid rgba(255,255,255,0.15)' }}>Sum</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => (
                  <tr key={idx}>
                    <td style={{ padding: '6px 8px', opacity: 0.95 }}>{r.hand}</td>
                    {columns.map((c, i) => (
                      <td key={i} style={{ textAlign: 'right', padding: '6px 8px' }}>{scoreFor(r, c.id)}</td>
                    ))}
                    <td style={{ textAlign: 'right', padding: '6px 8px', fontWeight: 600 }}>{r.sum}</td>
                  </tr>
                ))}
                <tr>
                  <td style={{ padding: '8px 8px', borderTop: '1px solid rgba(255,255,255,0.2)', fontWeight: 700 }}>Total</td>
                  {columns.map((c, i) => (
                    <td key={i} style={{ textAlign: 'right', padding: '8px 8px', borderTop: '1px solid rgba(255,255,255,0.2)', fontWeight: 700 }}>{totalById[c.id]}</td>
                  ))}
                  <td style={{ textAlign: 'right', padding: '8px 8px', borderTop: '1px solid rgba(255,255,255,0.2)', fontWeight: 800 }}>{overallSum}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function CollectOverlay({ containerRef, yourSeat, winner, cards, centerSize }: { containerRef: React.MutableRefObject<HTMLDivElement | null>; yourSeat: number; winner: number; cards: string[]; centerSize: { w: number; h: number } }) {
  const cw = containerRef.current?.clientWidth ?? 1000;
  const ch = containerRef.current?.clientHeight ?? 400;

  const refs = useRef<Array<HTMLDivElement | null>>([]);
  if (refs.current.length !== cards.length) refs.current = Array(cards.length).fill(null);

  function pctNum(v: string | number | undefined): number {
    if (typeof v === 'number') return v;
    if (!v) return 0;
    return parseFloat(String(v).replace('%', ''));
  }

  function pToPx(pctLeft: string | number | undefined, pctTop: string | number | undefined) {
    const x = (pctNum(pctLeft) / 100) * cw;
    const y = (pctNum(pctTop) / 100) * ch;
    return { x, y };
  }

  useEffect(() => {
    const start = performance.now();
    const dur = 1100;

    function bezier(t: number, p0: number, c1: number, c2: number, p3: number) {
      const it = 1 - t;
      return it * it * it * p0 + 3 * it * it * t * c1 + 3 * it * t * t * c2 + t * t * t * p3;
    }
    function easeInOut(t: number) {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    const fromPts = cards.map((_, idx) => {
      const absSeat = idx + 1;
      const r = (absSeat - yourSeat + 5) % 5;
      const st = spotStyle(r);
      return pToPx(st.left as any, st.top as any);
    });
    const toPt = (() => {
      const wr = (winner - yourSeat + 5) % 5;
      const st = spotStyle(wr);
      return pToPx(st.left as any, st.top as any);
    })();

    const ctrl = fromPts.map(p0 => {
      const cx1 = (p0.x + cw / 2) / 2;
      const cy1 = Math.min(p0.y, toPt.y) - 0.15 * ch;
      const cx2 = (toPt.x + cw / 2) / 2;
      const cy2 = Math.min(p0.y, toPt.y) - 0.15 * ch;
      return { cx1, cy1, cx2, cy2 };
    });

    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      const et = easeInOut(t);
      cards.forEach((_, idx) => {
        const el = refs.current[idx];
        if (!el) return;
        const p0 = fromPts[idx];
        const c = ctrl[idx];
        const x = bezier(et, p0.x, c.cx1, c.cx2, toPt.x);
        const y = bezier(et, p0.y, c.cy1, c.cy2, toPt.y);
        const s = 1 - 0.3 * et;
        const o = 1 - 0.9 * et;
        el.style.left = `${Math.round(x)}px`;
        el.style.top = `${Math.round(y)}px`;
        el.style.transform = `translate(-50%, -50%) scale(${s})`;
        el.style.opacity = String(o);
      });
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [cards, yourSeat, winner, cw, ch]);

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 10 }}>
      {cards.map((c, idx) => (
        <div
          key={idx}
          ref={(el) => { refs.current[idx] = el; }}
          style={{ position: 'absolute', left: 0, top: 0, transform: 'translate(-50%, -50%)', opacity: 1 }}
        >
          <PlayingCard label={c || '__'} width={centerSize.w} height={centerSize.h} />
        </div>
      ))}
    </div>
  );
}


