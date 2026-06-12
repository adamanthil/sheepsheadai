/* ============================================================
   HERO TUNING — keep the ORIGINAL layout, explore two things:
     1. Card flourish placement (lower, so "a/d" of Sheepshead read)
     2. Wordmark font + text treatments
   The shared scaffold below is a faithful copy of the real
   HomeDesktop left panel so every option reads in context.
   ============================================================ */

function Lede() {
  return (
    <p className="ss-body" style={{ fontSize: 19, maxWidth: 460, color: 'var(--ink-soft)', margin: 0 }}>
      A five‑handed, trick‑taking game from Wisconsin — played here with friends and an opinionated deep‑learning AI. Queens are highest. Diamonds are trump. Take more than sixty.
    </p>
  );
}

function HowToRow() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
      <button className="ss-btn ss-btn--ghost ss-btn--sm">How to play →</button>
      <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 14, color: 'var(--muted)' }}>about 4 minutes</span>
    </div>
  );
}

function StartTable() {
  return (
    <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div className="ss-overline">Start a Table</div>
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
        <button className="ss-btn ss-btn--accent ss-btn--lg">Create table →</button>
        <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', color: 'var(--muted)', fontSize: 14 }}>or join an open table</span>
      </div>
    </div>
  );
}

/* The card flourish — parametric so we can slide it around.
   `layout` controls how the three cards are arranged. */
function CardFlourish({ top = 80, right = 32, w = 108, layout = 'fan' }) {
  const arrangements = {
    // original fan: Q low-left, J high-center, A low-right
    fan: [
      { card: ['Q', 'C'], left: 0,   top: 36, rot: -14, z: 0 },
      { card: ['J', 'D'], left: 86,  top: 0,  rot: -2,  z: 1 },
      { card: ['A', 'H'], left: 172, top: 36, rot: 10,  z: 0 },
    ],
    // gentle cascade stepping downward to the right (stays clear of letters)
    cascade: [
      { card: ['Q', 'C'], left: 0,   top: 0,  rot: -8, z: 0 },
      { card: ['J', 'D'], left: 78,  top: 30, rot: 3,  z: 1 },
      { card: ['A', 'H'], left: 156, top: 64, rot: 13, z: 0 },
    ],
  };
  const cards = arrangements[layout] || arrangements.fan;
  return (
    <div style={{ position: 'absolute', right, top, opacity: 0.95, pointerEvents: 'none' }} aria-hidden="true">
      <div style={{ position: 'relative', width: w * 2.6, height: w * 2.2 }}>
        {cards.map((c, i) => (
          <div key={i} style={{ position: 'absolute', left: c.left, top: c.top, transform: `rotate(${c.rot}deg)`, zIndex: c.z }}>
            <PlayingCard rank={c.card[0]} suit={c.card[1]} w={w} />
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── WORDMARK treatments ─────────────────────────────────────
   Each returns the <h1> for the hero. They share the overline
   above and the rest of the panel below. */
function Wordmark({ variant }) {
  const ai = (extra = {}) => (
    <em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)', ...extra }}>AI</em>
  );

  switch (variant) {
    /* Current — Instrument Serif, stacked, italic AI accent */
    case 'og':
      return (
        <h1 className="ss-display" style={{ fontSize: 124, margin: 0, lineHeight: 0.92 }}>
          Sheepshead<br/>{ai()}
        </h1>
      );

    /* Inline AI — one line for the name, AI as a smaller trailing accent.
       Frees the top-right so the flourish never crosses letters. */
    case 'inline':
      return (
        <h1 className="ss-display" style={{ fontSize: 92, margin: 0, lineHeight: 0.96, letterSpacing: '-0.015em' }}>
          Sheepshead {ai({ fontSize: 52, verticalAlign: '0.18em' })}
        </h1>
      );

    /* Classic American broadsheet — Libre Caslon Display */
    case 'caslon':
      return (
        <h1 style={{ fontFamily: "'Libre Caslon Display', Georgia, serif", fontSize: 116, margin: 0, lineHeight: 0.94, color: 'var(--ink)', fontWeight: 400 }}>
          Sheepshead<br/><span style={{ fontStyle: 'italic', color: 'var(--accent)' }}>AI</span>
        </h1>
      );

    /* Didone masthead — Bodoni Moda, all-caps, letterspaced */
    case 'didone':
      return (
        <h1 style={{ fontFamily: "'Bodoni Moda', 'Didot', serif", margin: 0 }}>
          <span style={{ display: 'block', fontSize: 74, lineHeight: 0.96, letterSpacing: '0.04em', fontWeight: 600, textTransform: 'uppercase', color: 'var(--ink)' }}>Sheepshead</span>
          <span style={{ display: 'block', fontSize: 30, letterSpacing: '0.5em', textTransform: 'uppercase', color: 'var(--accent)', fontStyle: 'italic', marginTop: 10, paddingLeft: '0.5em' }}>Artificial Intelligence</span>
        </h1>
      );

    /* Tavern slab — Zilla Slab, warm + sturdy */
    case 'slab':
      return (
        <h1 style={{ fontFamily: "'Zilla Slab', serif", fontSize: 104, margin: 0, lineHeight: 0.92, fontWeight: 700, letterSpacing: '-0.01em', color: 'var(--ink)' }}>
          Sheepshead<br/><span style={{ color: 'var(--accent)', fontWeight: 500, fontStyle: 'italic' }}>AI</span>
        </h1>
      );

    /* Mono tag — serif name + "AI" set as a small monospaced product tag */
    case 'monotag':
      return (
        <h1 className="ss-display" style={{ fontSize: 112, margin: 0, lineHeight: 0.9, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 14 }}>
          <span>Sheepshead</span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 500, letterSpacing: '0.18em',
            textTransform: 'uppercase', color: 'var(--accent)', border: '1.5px solid var(--accent)',
            padding: '6px 12px 4px', borderRadius: 3, lineHeight: 1,
          }}>AI Engine</span>
        </h1>
      );

    default:
      return null;
  }
}

/* Full hero panel: overline + chosen wordmark + flourish + rest. */
function HeroPanel({ theme = 'heirloom', wordmark = 'og', cardTop = 140, cardRight = 32, cardW = 108, cardLayout = 'fan', showCards = true }) {
  return (
    <div className={`theme-${theme}`} style={{
      width: 770, height: 860, background: 'var(--bg-page)', color: 'var(--ink)',
      fontFamily: 'var(--font-ui)', overflow: 'hidden', position: 'relative',
      borderRight: '1px solid var(--rule)',
      padding: '56px 56px 40px 80px', display: 'flex', flexDirection: 'column', gap: 32,
    }}>
      {showCards && <CardFlourish top={cardTop} right={cardRight} w={cardW} layout={cardLayout} />}
      <div className="ss-overline">Vol. III · Wisconsin Tavern Series</div>
      <Wordmark variant={wordmark} />
      <Lede />
      <HowToRow />
      <StartTable />
    </div>
  );
}

Object.assign(window, { HeroPanel, CardFlourish, Wordmark });
