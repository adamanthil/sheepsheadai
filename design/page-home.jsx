/* ============================================================
   HOME / LOBBY PAGE
   ============================================================ */

const LOBBY_TABLES = [
  { name: 'Hobbiton',    host: 'Pippin',  players: 4, status: 'Waiting',  bots: 0 },
  { name: 'Bywater',     host: 'Merry',   players: 2, status: 'Waiting',  bots: 0 },
  { name: 'Bag End',     host: 'Bilbo',   players: 5, status: 'Playing',  bots: 0 },
  { name: 'Buckland',    host: 'Frodo',   players: 3, status: 'Waiting',  bots: 1 },
  { name: 'Tuckborough', host: 'Rosie',   players: 5, status: 'Playing',  bots: 2 },
  { name: 'Michel Delving', host: 'Sam',  players: 1, status: 'Waiting',  bots: 0 },
];

// The landing page is chromeless — the hero masthead IS the brand, so the
// app shell's wordmark row would be double branding here. Waiting and
// Table keep their own headers.
function HomePage({ viewport, navigate, theme }) {
  const isDesktop = viewport === 'desktop';
  return (
    <div className={`theme-${theme}`} style={{
      width: '100%', height: '100%', background: 'var(--bg-page)',
      color: 'var(--ink)', fontFamily: 'var(--font-ui)', overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      {isDesktop ? <HomeDesktop navigate={navigate} theme={theme} /> : <HomeMobile navigate={navigate} theme={theme} />}
    </div>
  );
}

// ─── Masthead band — newspaper nameplate shared by both viewports ────
// Thick-thin rule on top, volume line left, three small cards set INTO
// the band (Q♣ highest trump · J♦ the partner card · A♥ a fail ace),
// date right, hairline below. The cards live in the band's structure so
// they never collide with the wordmark.
function MastheadBand({ compact }) {
  const cardW = compact ? 24 : 32;
  return (
    <div>
      <div className="ss-head-rule" />
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center',
        gap: compact ? 10 : 24, padding: compact ? '9px 0 7px' : '12px 0 10px',
      }}>
        <div className="ss-overline" style={{ fontSize: compact ? 8.5 : 11, whiteSpace: 'nowrap' }}>
          {compact ? 'Vol. III' : 'Vol. III · Tavern Series'}
        </div>
        <div style={{ display: 'flex', gap: compact ? 4 : 6, alignItems: 'center' }} aria-hidden="true">
          <div style={{ transform: 'rotate(-7deg) translateY(1px)' }}><PlayingCard rank="Q" suit="C" w={cardW} /></div>
          <div style={{ transform: 'rotate(0deg) translateY(-1px)', zIndex: 1 }}><PlayingCard rank="J" suit="D" w={cardW} /></div>
          <div style={{ transform: 'rotate(7deg) translateY(1px)' }}><PlayingCard rank="A" suit="H" w={cardW} /></div>
        </div>
        <div className="ss-overline" style={{ fontSize: compact ? 8.5 : 11, textAlign: 'right', whiteSpace: 'nowrap' }}>
          {compact ? 'MMXXVI' : 'Wisconsin · MMXXVI'}
        </div>
      </div>
      <div style={{ borderBottom: '1px solid var(--rule-strong)' }} />
    </div>
  );
}

// Suit strapline — the rules of the game in one italic line.
function Strapline({ fontSize = 17 }) {
  return (
    <span style={{
      fontFamily: 'var(--font-display)', fontStyle: 'italic',
      fontSize, color: 'var(--ink-soft)', whiteSpace: 'nowrap',
    }}>
      <span style={{ color: 'var(--ink)' }}>♣</span> Queens high
      <span style={{ color: 'var(--muted)' }}> · </span>
      <span style={{ color: 'var(--card-red)' }}>♦</span> Diamonds are trump
      <span style={{ color: 'var(--muted)' }}> · </span>
      Take 61
    </span>
  );
}

function HomeDesktop({ navigate, theme }) {
  return (
    <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1.15fr 1fr', minHeight: 0 }}>
      {/* LEFT — Hero + create */}
      <div style={{ padding: '40px 56px 40px 80px', display: 'flex', flexDirection: 'column', gap: 26, borderRight: '1px solid var(--rule)', position: 'relative', overflow: 'hidden' }}>
        <MastheadBand />

        {/* Wordmark — full width, nothing crossing the letters. The AI
            sits on the second line, tied to the strapline by a rule. */}
        <div>
          <h1 className="ss-display" style={{ fontSize: 146, margin: 0, lineHeight: 0.92, letterSpacing: '-0.02em' }}>
            Sheepshead
          </h1>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 20, marginTop: 10 }}>
            <em className="ss-display" style={{ fontSize: 76, lineHeight: 0.9, color: 'var(--accent)', fontStyle: 'italic' }}>AI</em>
            <span style={{ flex: 1, height: 1, background: 'var(--rule-strong)', transform: 'translateY(-14px)' }} aria-hidden="true" />
            <Strapline fontSize={17} />
          </div>
        </div>

        <p className="ss-body" style={{ fontSize: 18, maxWidth: 470, color: 'var(--ink-soft)', margin: 0 }}>
          A five‑handed, trick‑taking game from Wisconsin — played here with friends and an opinionated deep‑learning AI.
        </p>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button className="ss-btn ss-btn--ghost ss-btn--sm" onClick={() => navigate && navigate('howto')}>How to play →</button>
          <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 14, color: 'var(--muted)' }}>about 4 minutes</span>
        </div>

        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="ss-head-rule" style={{ paddingTop: 12 }}>
            <div className="ss-overline">Start a Table</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28, maxWidth: 520 }}>
            <div>
              <label className="ss-overline" style={{ fontSize: 10 }}>Your name</label>
              <input className="ss-input" defaultValue="Andrew" />
            </div>
            <div>
              <label className="ss-overline" style={{ fontSize: 10 }}>Table name</label>
              <input className="ss-input" defaultValue="Hobbiton" />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8 }}>
            <button className="ss-btn ss-btn--accent ss-btn--lg" onClick={() => navigate && navigate('waiting')}>Create table →</button>
            <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', color: 'var(--muted)', fontSize: 14 }}>or join an open table</span>
          </div>
        </div>
      </div>

      {/* RIGHT — Lobby */}
      <div style={{ padding: '40px 80px 40px 48px', display: 'flex', flexDirection: 'column', gap: 22, minHeight: 0 }}>
        <div className="ss-head-rule" style={{ paddingTop: 14 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
            <h2 className="ss-display" style={{ fontSize: 40, margin: 0 }}>The Lobby</h2>
            <div className="ss-overline">{LOBBY_TABLES.length} tables · live</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 0.8fr 0.8fr auto', columnGap: 16, rowGap: 0, paddingBottom: 10, borderBottom: '1px solid var(--rule)' }}>
          <div className="ss-overline">Name</div>
          <div className="ss-overline">Host</div>
          <div className="ss-overline">Players</div>
          <div className="ss-overline" style={{ textAlign: 'right' }}>Action</div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {LOBBY_TABLES.map((t, i) => {
            const full = t.players >= 5;
            const playing = t.status === 'Playing';
            return (
              <div key={t.name} style={{ display: 'grid', gridTemplateColumns: '1.4fr 0.8fr 0.8fr auto', columnGap: 16, alignItems: 'center', padding: '16px 0', borderBottom: '1px solid var(--rule)' }}>
                <div>
                  <div className="ss-display" style={{ fontSize: 22, color: 'var(--ink)' }}>{t.name}</div>
                  {t.bots > 0 && (
                    <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 13, color: 'var(--muted)', marginTop: 2 }}>
                      {t.bots} AI {t.bots === 1 ? 'bot' : 'bots'}
                    </div>
                  )}
                </div>
                <div style={{ fontFamily: 'var(--font-ui)', fontSize: 14, color: 'var(--ink-soft)' }}>{t.host}</div>
                <div className="ss-num" style={{ fontSize: 18, color: 'var(--ink)' }}>
                  {t.players}<span style={{ color: 'var(--muted)' }}>/5</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  {playing ? (
                    <span className="ss-badge ss-badge--quiet">In play</span>
                  ) : full ? (
                    <span className="ss-badge ss-badge--quiet">Full</span>
                  ) : (
                    <button className="ss-btn ss-btn--sm" onClick={() => navigate && navigate('waiting')}>Join →</button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: 16 }}>
          <a className="ss-link" href="#analyze" onClick={(e) => { e.preventDefault(); navigate && navigate('analyze'); }} style={{ fontSize: 13 }}>
            Inspect AI model decisions ↗
          </a>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            v0.7 · MAY 2026
          </div>
        </div>
      </div>
    </div>
  );
}

function HomeMobile({ navigate, theme }) {
  return (
    <div style={{ flex: 1, padding: '14px 16px 14px', display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto' }}>

      {/* Masthead hero — band + one-line wordmark + strapline */}
      <div>
        <MastheadBand compact />
        <h1 className="ss-display" style={{ margin: '12px 0 0', lineHeight: 0.95, letterSpacing: '-0.015em', whiteSpace: 'nowrap' }}>
          <span style={{ fontSize: 52 }}>Sheepshead</span>
          <em style={{ fontSize: 30, color: 'var(--accent)', fontStyle: 'italic', marginLeft: 8 }}>AI</em>
        </h1>
        <div style={{ marginTop: 6, display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 10 }}>
          <Strapline fontSize={12.5} />
          <a className="ss-link" href="#" style={{ fontSize: 11, whiteSpace: 'nowrap' }} onClick={(e) => e.preventDefault()}>How to play →</a>
        </div>
      </div>

      <div style={{ height: 1, background: 'var(--rule)' }} />

      {/* Compact start-a-table — two inputs side-by-side */}
      <div>
        <div className="ss-overline" style={{ marginBottom: 8, fontSize: 9 }}>Start a table</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 10 }}>
          <div>
            <label className="ss-overline" style={{ fontSize: 9 }}>Your name</label>
            <input className="ss-input" defaultValue="Andrew" style={{ fontSize: 18, paddingBottom: 4 }} />
          </div>
          <div>
            <label className="ss-overline" style={{ fontSize: 9 }}>Table name</label>
            <input className="ss-input" defaultValue="Hobbiton" style={{ fontSize: 18, paddingBottom: 4 }} />
          </div>
        </div>
        <button className="ss-btn ss-btn--accent" style={{ width: '100%', padding: '12px 18px', fontSize: 14 }} onClick={() => navigate && navigate('waiting')}>Create table →</button>
      </div>

      <div style={{ height: 1, background: 'var(--rule)' }} />

      {/* Lobby — tight rows */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10, gap: 12 }}>
          <h2 className="ss-display" style={{ fontSize: 24, margin: 0, whiteSpace: 'nowrap' }}>The Lobby</h2>
          <div className="ss-overline" style={{ fontSize: 9, whiteSpace: 'nowrap' }}>{LOBBY_TABLES.length} tables · live</div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {LOBBY_TABLES.map((t, i) => {
            const full = t.players >= 5;
            const playing = t.status === 'Playing';
            return (
              <div key={t.name} style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', alignItems: 'center', columnGap: 12, padding: '10px 0', borderBottom: i < LOBBY_TABLES.length - 1 ? '1px solid var(--rule)' : 'none' }}>
                <div style={{ minWidth: 0 }}>
                  <div className="ss-display" style={{ fontSize: 17, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.name}</div>
                  <div style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--muted)' }}>
                    {t.host}{t.bots > 0 ? ` · ${t.bots} AI` : ''}
                  </div>
                </div>
                <div className="ss-num" style={{ fontSize: 14, color: 'var(--ink-soft)' }}>
                  {t.players}<span style={{ color: 'var(--muted)' }}>/5</span>
                </div>
                <div>
                  {playing ? <span className="ss-badge ss-badge--quiet" style={{ fontSize: 9 }}>In play</span>
                   : full ? <span className="ss-badge ss-badge--quiet" style={{ fontSize: 9 }}>Full</span>
                   : <button className="ss-btn ss-btn--sm" style={{ padding: '5px 10px', fontSize: 11 }} onClick={() => navigate && navigate('waiting')}>Join →</button>}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 4 }}>
        <a className="ss-link" href="#analyze" style={{ fontSize: 11 }} onClick={(e) => { e.preventDefault(); navigate && navigate('analyze'); }}>Inspect AI model ↗</a>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>v0.7</div>
      </div>
    </div>
  );
}

// ─── Decorative three-card flourish for hero ──────────────────────
function DecorativeCards({ theme }) {
  // Q♣, J♦, A♥ — the highest trump, the partner card (JD), and a fail ace
  return (
    <div style={{ position: 'absolute', right: 32, top: 80, opacity: 0.95, pointerEvents: 'none' }} aria-hidden="true">
      <div style={{ position: 'relative', width: 280, height: 240 }}>
        <div style={{ position: 'absolute', left: 0, top: 36, transform: 'rotate(-14deg)' }}>
          <PlayingCard rank="Q" suit="C" w={108} />
        </div>
        <div style={{ position: 'absolute', left: 86, top: 0, transform: 'rotate(-2deg)', zIndex: 1 }}>
          <PlayingCard rank="J" suit="D" w={108} />
        </div>
        <div style={{ position: 'absolute', left: 172, top: 36, transform: 'rotate(10deg)' }}>
          <PlayingCard rank="A" suit="H" w={108} />
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { HomePage, DecorativeCards });
