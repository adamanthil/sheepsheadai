/* ============================================================
   WAITING ROOM
   ============================================================ */

const WAITING_SEATS = [
  { num: 1, name: 'Bilbo',   ai: false, host: false, ready: true },
  { num: 2, name: 'Pippin',  ai: false, host: true,  ready: true },
  { num: 3, name: null,      ai: false, empty: true },
  { num: 4, name: 'Gandalf', ai: true,  ready: true },
  { num: 5, name: 'Andrew',  ai: false, you: true,  ready: true },
];

const WAITING_CHAT = [
  { system: true, text: 'Pippin opened the table at Hobbiton.', time: '10:42 PM' },
  { system: true, text: 'Bilbo took seat 1.',  time: '10:43 PM' },
  { from: 'Bilbo', text: 'Going Jack of Diamonds tonight?', time: '10:43 PM' },
  { from: 'Pippin', text: 'Called Ace. Last week was rough on me.', time: '10:44 PM' },
  { system: true, text: 'Gandalf (AI) took seat 4.', time: '10:45 PM' },
  { system: true, text: 'Andrew joined and took seat 5.', time: '10:46 PM' },
];

function WaitingPage({ viewport, navigate, theme }) {
  const isDesktop = viewport === 'desktop';
  return (
    <div className={`theme-${theme}`} style={{
      width: '100%', height: '100%', background: 'var(--bg-page)',
      color: 'var(--ink)', fontFamily: 'var(--font-ui)', overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Mobile skips the app chrome — the room header right below carries
          identity, and the saved row matters at 390px. */}
      {isDesktop && <PageChrome current="waiting" theme={theme} onNav={navigate} />}
      {isDesktop ? <WaitingDesktop navigate={navigate} theme={theme} /> : <WaitingMobile navigate={navigate} theme={theme} />}
    </div>
  );
}

function WaitingDesktop({ navigate, theme }) {
  return (
    <div style={{
      flex: 1, minHeight: 0, padding: '32px 80px 24px',
      display: 'grid',
      gridTemplateRows: 'auto auto 1fr auto',
      gap: 22,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
        <div>
          <div className="ss-overline">Waiting Room · #A7F2</div>
          <h1 className="ss-display" style={{ fontSize: 72, margin: '4px 0 0', lineHeight: 1 }}>Hobbiton</h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
          <div style={{ textAlign: 'right' }}>
            <div className="ss-overline" style={{ fontSize: 10 }}>Players</div>
            <div className="ss-display ss-num" style={{ fontSize: 32, color: 'var(--ink)' }}>4<span style={{ color: 'var(--muted)' }}>/5</span></div>
          </div>
          <div style={{ width: 1, height: 48, background: 'var(--rule)' }} />
          <div style={{ textAlign: 'right' }}>
            <div className="ss-overline" style={{ fontSize: 10 }}>Host</div>
            <div className="ss-display" style={{ fontSize: 22 }}>Pippin</div>
          </div>
        </div>
      </div>

      {/* Seats row */}
      <div>
        <div className="ss-overline" style={{ marginBottom: 12 }}>The Table · 5 Seats</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          {WAITING_SEATS.map((s) => <SeatCard key={s.num} seat={s} />)}
        </div>
      </div>

      {/* Two-column: rules + chat */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 28, minHeight: 0 }}>
        <RulesPanel />
        <ChatPanel messages={WAITING_CHAT} />
      </div>

      {/* Action bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--rule)', paddingTop: 16 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button className="ss-btn ss-btn--ghost ss-btn--sm">Fill empty with AI</button>
          <button className="ss-btn ss-btn--ghost ss-btn--sm" style={{ color: 'var(--accent)', borderColor: 'transparent' }}>Close table</button>
        </div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', color: 'var(--muted)', fontSize: 14 }}>Waiting on 1 more, or fill with AI</div>
          <button className="ss-btn ss-btn--accent ss-btn--lg" onClick={() => navigate && navigate('table')}>Deal cards →</button>
        </div>
      </div>
    </div>
  );
}

function SeatCard({ seat }) {
  if (seat.empty) {
    return (
      <div style={{
        padding: 16, borderRadius: 4,
        border: '1px dashed var(--rule-strong)',
        display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center',
        background: 'transparent',
        minHeight: 168,
        justifyContent: 'space-between',
      }}>
        <div className="ss-overline" style={{ alignSelf: 'flex-start' }}>Seat {seat.num}</div>
        <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', color: 'var(--muted)', fontSize: 22, textAlign: 'center' }}>
          Empty
        </div>
        <button className="ss-btn ss-btn--sm" style={{ alignSelf: 'stretch' }}>Take this seat →</button>
      </div>
    );
  }
  return (
    <div className="ss-panel" style={{
      padding: 16, display: 'flex', flexDirection: 'column', gap: 10,
      minHeight: 168, justifyContent: 'space-between',
      background: seat.you ? 'var(--bg-page-deep)' : 'var(--bg-card)',
      borderColor: seat.you ? 'var(--ink)' : 'var(--rule)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="ss-overline">Seat {seat.num}</div>
        {seat.host && <span className="ss-badge ss-badge--ink">Host</span>}
        {seat.you && !seat.host && <span className="ss-badge">You</span>}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
        <SeatAvatar name={seat.name} isAI={seat.ai} size={44} />
        <div className="ss-display" style={{ fontSize: 20, color: 'var(--ink)' }}>{seat.name}</div>
        {seat.ai && <span className="ss-badge ss-badge--quiet">AI Bot</span>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--accent-2)' }}>
        <span style={{ width: 6, height: 6, background: 'var(--accent-2)', borderRadius: '50%' }} /> ready
      </div>
    </div>
  );
}

function RulesPanel() {
  return (
    <div className="ss-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 18, minHeight: 0 }}>
      <div>
        <div className="ss-overline" style={{ marginBottom: 4 }}>House Rules · Host decides</div>
        <div className="ss-display" style={{ fontSize: 24 }}>Game Mode</div>
      </div>

      <div>
        <div style={{ fontFamily: 'var(--font-ui)', fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--ink)' }}>Partner selection</div>
        <Segmented options={['Called Ace', 'Jack of Diamonds']} value={0} />
        <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 13, color: 'var(--muted)', marginTop: 8, lineHeight: 1.5 }}>
          The picker names a fail‑suit ace; whoever holds it is their secret partner until the card is played.
        </div>
      </div>

      <div>
        <div style={{ fontFamily: 'var(--font-ui)', fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--ink)' }}>Scoring</div>
        <Segmented options={['Double on Bump', 'Symmetric']} value={0} />
        <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 13, color: 'var(--muted)', marginTop: 8, lineHeight: 1.5 }}>
          The picking team loses double when they fail to take 60. Wins are scored normally.
        </div>
      </div>
    </div>
  );
}

function Segmented({ options, value }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${options.length}, 1fr)`, border: '1px solid var(--rule-strong)', borderRadius: 3, overflow: 'hidden' }}>
      {options.map((o, i) => (
        <button key={o} style={{
          padding: '10px 14px',
          border: 'none',
          borderLeft: i === 0 ? 'none' : '1px solid var(--rule-strong)',
          background: i === value ? 'var(--ink)' : 'transparent',
          color: i === value ? 'var(--bg-page)' : 'var(--ink-soft)',
          fontFamily: 'var(--font-ui)',
          fontSize: 13,
          fontWeight: i === value ? 600 : 400,
          cursor: 'pointer',
          textAlign: 'center',
        }}>{o}</button>
      ))}
    </div>
  );
}

function ChatPanel({ messages, compact }) {
  return (
    <div className="ss-panel" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{ padding: '14px 18px 8px', borderBottom: '1px solid var(--rule)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="ss-overline">Table Chat</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)' }}>{messages.length} msgs</div>
      </div>
      <div style={{ padding: '6px 18px', flex: 1, overflow: 'auto', minHeight: 0 }}>
        {messages.map((m, i) => <ChatRow key={i} {...m} />)}
      </div>
      <div style={{ padding: '10px 14px', borderTop: '1px solid var(--rule)', display: 'flex', gap: 8, alignItems: 'center' }}>
        <input placeholder="Say something…" style={{
          flex: 1, border: 'none', background: 'transparent', outline: 'none',
          fontFamily: 'var(--font-ui)', fontSize: 14, color: 'var(--ink)', padding: '4px 4px',
        }} />
        <button className="ss-btn ss-btn--sm">Send</button>
      </div>
    </div>
  );
}

// ─── Mobile ──────────────────────────────────────────────────────────
//
// Compress: tight header, seats in a single horizontal scrollable strip
// (no, on second thought 2×3 grid w/ inline cards), inline rules,
// collapsible chat, sticky bottom action.
function WaitingMobile({ navigate, theme }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
      <div style={{ flex: 1, overflow: 'auto', padding: '14px 16px 12px', display: 'flex', flexDirection: 'column', gap: 14 }}>

        {/* Header — compact, carries identity + exit since mobile has no chrome */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
          <div className="ss-overline" style={{ fontSize: 9 }}>Waiting Room · #A7F2</div>
          <a
            className="ss-link" href="#"
            onClick={(e) => { e.preventDefault(); navigate && navigate('home'); }}
            style={{ fontSize: 11, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase', textDecoration: 'none' }}
          >
            Leave
          </a>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 12, marginTop: -8 }}>
          <h1 className="ss-display" style={{ fontSize: 38, margin: 0, lineHeight: 1 }}>Hobbiton</h1>
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <div className="ss-display ss-num" style={{ fontSize: 22, lineHeight: 1 }}>4<span style={{ color: 'var(--muted)' }}>/5</span></div>
            <div className="ss-overline" style={{ fontSize: 9, marginTop: 2 }}>seated</div>
          </div>
        </div>

        {/* Seats — vertical list, single row each */}
        <div>
          <div className="ss-overline" style={{ fontSize: 9, marginBottom: 8 }}>Seats · Host: Pippin</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {WAITING_SEATS.map((s) => <SeatRow key={s.num} seat={s} />)}
          </div>
        </div>

        {/* Rules — inline, no panel chrome */}
        <div>
          <div className="ss-overline" style={{ fontSize: 9, marginBottom: 8 }}>Game Mode</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div>
              <div style={{ fontFamily: 'var(--font-ui)', fontSize: 11, fontWeight: 600, marginBottom: 4, color: 'var(--ink-soft)' }}>Partner</div>
              <Segmented options={['Called Ace', 'Jack of Diamonds']} value={0} />
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-ui)', fontSize: 11, fontWeight: 600, marginBottom: 4, color: 'var(--ink-soft)' }}>Scoring</div>
              <Segmented options={['Double on Bump', 'Symmetric']} value={0} />
            </div>
          </div>
        </div>

        {/* Chat preview (last 2) */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
            <div className="ss-overline" style={{ fontSize: 9 }}>Latest in chat</div>
            <a className="ss-link" href="#" style={{ fontSize: 11 }} onClick={(e) => e.preventDefault()}>open ↗</a>
          </div>
          <div style={{ padding: '4px 0' }}>
            {WAITING_CHAT.slice(-2).map((m, i) => <ChatRow key={i} {...m} />)}
          </div>
        </div>
      </div>

      {/* Sticky action bar */}
      <div style={{
        borderTop: '1px solid var(--rule)',
        padding: '10px 16px',
        background: 'var(--bg-page)',
        display: 'flex', flexDirection: 'column', gap: 8,
        flexShrink: 0,
      }}>
        <button className="ss-btn ss-btn--accent" style={{ padding: '12px 18px', fontSize: 14 }} onClick={() => navigate && navigate('table')}>Deal cards →</button>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="ss-btn ss-btn--ghost ss-btn--sm" style={{ flex: 1 }}>Fill with AI</button>
          <button className="ss-btn ss-btn--ghost ss-btn--sm" style={{ flex: 1, color: 'var(--accent)' }}>Close</button>
        </div>
      </div>
    </div>
  );
}

function SeatRow({ seat }) {
  if (seat.empty) {
    return (
      <button style={{
        display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px',
        border: '1px dashed var(--rule-strong)', borderRadius: 4,
        background: 'transparent', textAlign: 'left', cursor: 'pointer',
      }}>
        <div style={{ width: 36, height: 36, borderRadius: '50%', border: '1px dashed var(--rule-strong)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: 18 }}>+</div>
        <div style={{ flex: 1 }}>
          <div className="ss-overline" style={{ fontSize: 9 }}>Seat {seat.num}</div>
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 16, color: 'var(--muted)' }}>Take this seat</div>
        </div>
        <div style={{ fontFamily: 'var(--font-ui)', fontSize: 12, color: 'var(--ink)', letterSpacing: '0.1em' }}>→</div>
      </button>
    );
  }
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px',
      border: '1px solid ' + (seat.you ? 'var(--ink)' : 'var(--rule)'),
      borderRadius: 4,
      background: seat.you ? 'var(--bg-page-deep)' : 'var(--bg-card)',
    }}>
      <SeatAvatar name={seat.name} isAI={seat.ai} size={36} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <div className="ss-display" style={{ fontSize: 19, color: 'var(--ink)' }}>{seat.name}</div>
          <div className="ss-overline" style={{ fontSize: 9 }}>seat {seat.num}</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
          {seat.ai && <span className="ss-badge ss-badge--quiet" style={{ fontSize: 9 }}>AI bot</span>}
          {seat.host && <span className="ss-badge ss-badge--ink" style={{ fontSize: 9 }}>Host</span>}
          {seat.you && !seat.host && <span className="ss-badge" style={{ fontSize: 9 }}>You</span>}
          <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-ui)', fontSize: 10, color: 'var(--accent-2)' }}>
            <span style={{ width: 5, height: 5, background: 'var(--accent-2)', borderRadius: '50%' }} /> ready
          </span>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { WaitingPage, ChatPanel, Segmented });
