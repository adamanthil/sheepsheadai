/* ============================================================
   LANDMARK / HERO — variations
   Each Hero is the LEFT panel of the home page (770×860),
   keeping the overall page rhythm but exploring how the
   "Sheepshead AI" wordmark coexists with the card flourish.
   ============================================================ */

// Shared shell so every variant feels like the same product
function HeroShell({ children, theme = 'broadsheet' }) {
  return (
    <div className={`theme-${theme}`} style={{
      width: 770, height: 860, background: 'var(--bg-page)', color: 'var(--ink)',
      fontFamily: 'var(--font-ui)', overflow: 'hidden', position: 'relative',
      borderRight: '1px solid var(--rule)',
    }}>
      {children}
    </div>
  );
}

function HeroBody({ children, padding = '56px 56px 40px 80px', gap = 32, style }) {
  return (
    <div style={{
      width: '100%', height: '100%', padding, display: 'flex',
      flexDirection: 'column', gap, position: 'relative', ...style,
    }}>{children}</div>
  );
}

function StartTable({ accent }) {
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
        <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', color: 'var(--muted)', fontSize: 14 }}>
          or join an open table
        </span>
      </div>
    </div>
  );
}

function Lede({ size = 19 }) {
  return (
    <p className="ss-body" style={{ fontSize: size, maxWidth: 460, color: 'var(--ink-soft)', margin: 0 }}>
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

/* ──────────────────────────────────────────────────────────────
   A — CURRENT (for reference)
   Cards layered over the right edge of "Sheepshead".
   ──────────────────────────────────────────────────────────── */
function HeroA_Current() {
  return (
    <HeroShell>
      <HeroBody>
        <div style={{ position: 'absolute', right: 32, top: 80, pointerEvents: 'none' }} aria-hidden="true">
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
        <div className="ss-overline">Vol. III · Wisconsin Tavern Series</div>
        <h1 className="ss-display" style={{ fontSize: 124, margin: 0, lineHeight: 0.92 }}>
          Sheepshead<br/><em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)' }}>AI</em>
        </h1>
        <Lede />
        <HowToRow />
        <StartTable />
      </HeroBody>
    </HeroShell>
  );
}

/* ──────────────────────────────────────────────────────────────
   B — MASTHEAD BAND
   Newspaper-style nameplate row across the top: rule line,
   volume info on the left, three tiny cards centered, date right.
   Below it: the wordmark gets the full width, totally unobstructed.
   ──────────────────────────────────────────────────────────── */
function HeroB_Masthead() {
  return (
    <HeroShell>
      <HeroBody padding="32px 56px 40px 80px" gap={28}>
        {/* Masthead band */}
        <div style={{
          borderTop: '2px solid var(--ink)',
          borderBottom: '1px solid var(--rule-strong)',
          padding: '10px 0',
          display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center',
          gap: 24,
        }}>
          <div className="ss-overline" style={{ textAlign: 'left' }}>Vol. III · No. 7</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }} aria-hidden="true">
            <PlayingCard rank="Q" suit="C" w={32} />
            <PlayingCard rank="J" suit="D" w={32} />
            <PlayingCard rank="A" suit="H" w={32} />
          </div>
          <div className="ss-overline" style={{ textAlign: 'right' }}>Wis. · May MMXXVI</div>
        </div>

        <h1 className="ss-display" style={{ fontSize: 156, margin: '8px 0 0', lineHeight: 0.88, letterSpacing: '-0.02em' }}>
          Sheepshead<br/>
          <em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)' }}>AI</em>
        </h1>
        <Lede />
        <HowToRow />
        <StartTable />
      </HeroBody>
    </HeroShell>
  );
}

/* ──────────────────────────────────────────────────────────────
   C — MONOGRAM DROP-CARD
   A single oversized Q♣ (the boss card in sheepshead) sits to
   the LEFT of the wordmark as a fleuron / drop cap. Wordmark
   sits clean to its right, fully readable.
   ──────────────────────────────────────────────────────────── */
function HeroC_DropCard() {
  return (
    <HeroShell>
      <HeroBody>
        <div className="ss-overline">Vol. III · Wisconsin Tavern Series</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 28, alignItems: 'center' }}>
          <div style={{ transform: 'rotate(-6deg)', filter: 'drop-shadow(0 14px 22px rgba(0,0,0,.18))' }} aria-hidden="true">
            <PlayingCard rank="Q" suit="C" w={168} />
          </div>
          <h1 className="ss-display" style={{ fontSize: 96, margin: 0, lineHeight: 0.9, letterSpacing: '-0.015em' }}>
            Sheep<br/>shead<br/>
            <em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)' }}>AI</em>
          </h1>
        </div>
        <Lede />
        <HowToRow />
        <StartTable />
      </HeroBody>
    </HeroShell>
  );
}

/* ──────────────────────────────────────────────────────────────
   D — CORNER FLOURISH
   Wordmark gets the full width up top. Three cards tucked into
   the bottom-right of the hero panel as a quiet corner ornament,
   tilted away from the form.
   ──────────────────────────────────────────────────────────── */
function HeroD_Corner() {
  return (
    <HeroShell>
      <HeroBody>
        <div className="ss-overline">Vol. III · Wisconsin Tavern Series</div>
        <h1 className="ss-display" style={{ fontSize: 156, margin: 0, lineHeight: 0.88, letterSpacing: '-0.02em' }}>
          Sheepshead<br/>
          <em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)' }}>AI</em>
        </h1>
        <Lede />
        <HowToRow />
        <StartTable />

        {/* corner flourish */}
        <div style={{ position: 'absolute', right: -28, bottom: -24, pointerEvents: 'none', opacity: 0.92 }} aria-hidden="true">
          <div style={{ position: 'relative', width: 260, height: 220 }}>
            <div style={{ position: 'absolute', right: 130, bottom: 12, transform: 'rotate(-22deg)' }}>
              <PlayingCard rank="A" suit="H" w={96} />
            </div>
            <div style={{ position: 'absolute', right: 60, bottom: 0, transform: 'rotate(-6deg)', zIndex: 1 }}>
              <PlayingCard rank="J" suit="D" w={96} />
            </div>
            <div style={{ position: 'absolute', right: -10, bottom: 18, transform: 'rotate(14deg)' }}>
              <PlayingCard rank="Q" suit="C" w={96} />
            </div>
          </div>
        </div>
      </HeroBody>
    </HeroShell>
  );
}

/* ──────────────────────────────────────────────────────────────
   E — TYPE ONLY (broadsheet)
   No cards in the landmark — pure editorial. Suit ornaments
   bracket the wordmark; rules above and below.
   ──────────────────────────────────────────────────────────── */
function HeroE_TypeOnly() {
  return (
    <HeroShell>
      <HeroBody gap={26}>
        <div style={{ borderTop: '2px solid var(--ink)', paddingTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <div className="ss-overline">Vol. III · Wisconsin Tavern Series</div>
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 14, color: 'var(--muted)' }}>est. 2026</div>
        </div>

        <div style={{ position: 'relative' }}>
          {/* Giant ghost club behind the wordmark */}
          <div aria-hidden="true" style={{
            position: 'absolute', right: 8, top: -28,
            fontFamily: 'var(--font-display)', fontSize: 360, lineHeight: 1,
            color: 'var(--accent)', opacity: 0.08, pointerEvents: 'none',
            transform: 'rotate(-6deg)',
          }}>♣</div>
          <h1 className="ss-display" style={{ fontSize: 148, margin: 0, lineHeight: 0.88, letterSpacing: '-0.02em', position: 'relative' }}>
            Sheepshead<br/>
            <em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)' }}>AI</em>
          </h1>
        </div>

        <div style={{ borderBottom: '1px solid var(--rule)', paddingBottom: 16, display: 'flex', gap: 14, alignItems: 'center', color: 'var(--muted)', fontFamily: 'var(--font-display)', fontSize: 18, fontStyle: 'italic' }}>
          <span>♣ Queens</span><span>·</span><span>♦ Diamonds</span><span>·</span><span style={{ color: 'var(--accent-2)' }}>♥ Take 61</span>
        </div>

        <Lede />
        <HowToRow />
        <StartTable />
      </HeroBody>
    </HeroShell>
  );
}

/* ──────────────────────────────────────────────────────────────
   F — SIDE RAIL
   A vertical column of two cards pinned to the right edge,
   touching but never crossing the wordmark.
   ──────────────────────────────────────────────────────────── */
function HeroF_SideRail() {
  return (
    <HeroShell>
      <HeroBody style={{ paddingRight: 180 }}>
        <div className="ss-overline">Vol. III · Wisconsin Tavern Series</div>
        <h1 className="ss-display" style={{ fontSize: 124, margin: 0, lineHeight: 0.92, letterSpacing: '-0.015em' }}>
          Sheepshead<br/>
          <em style={{ color: 'var(--accent)', fontStyle: 'italic', fontFamily: 'var(--font-display)' }}>AI</em>
        </h1>
        <Lede />
        <HowToRow />
        <StartTable />

        {/* Right rail */}
        <div style={{ position: 'absolute', right: 36, top: 56, bottom: 56, width: 120, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 18, pointerEvents: 'none' }} aria-hidden="true">
          <div style={{ transform: 'rotate(-3deg)' }}>
            <PlayingCard rank="Q" suit="C" w={112} />
          </div>
          <div style={{ transform: 'rotate(4deg)', marginTop: -36 }}>
            <PlayingCard rank="J" suit="D" w={112} />
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 13, color: 'var(--muted)', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
            Queens are highest · Diamonds are trump
          </div>
        </div>
      </HeroBody>
    </HeroShell>
  );
}

Object.assign(window, {
  HeroA_Current, HeroB_Masthead, HeroC_DropCard, HeroD_Corner, HeroE_TypeOnly, HeroF_SideRail,
});
