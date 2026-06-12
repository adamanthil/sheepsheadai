/* ============================================================
   Shared components: PlayingCard, Wordmark, Seat, Chrome bits
   ============================================================ */

const SUITS = { S: '♠', H: '♥', D: '♦', C: '♣' };
const SUIT_NAMES = { S: 'Spades', H: 'Hearts', D: 'Diamonds', C: 'Clubs' };
const RED = new Set(['H', 'D']);

// Card pip layouts for "classic" style center — for face cards we show 1 big pip,
// for number cards we could too. Keep it simple: 1 centered pip + corners.
function PlayingCard({ rank, suit, faceDown, playable, dim, w = 64, style = 'classic', ariaHidden, special }) {
  const h = Math.round(w * 1.45);
  if (special === 'UNDER') {
    return (
      <div className="pc pc--inset" style={{ width: w, height: h, '--pc-w': w + 'px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-display)', fontStyle: 'italic', color: 'var(--muted)' }}>
        <span style={{ fontSize: w * 0.18 }}>under</span>
      </div>
    );
  }
  if (faceDown) {
    return <div className="pc pc--back" style={{ width: w, height: h, '--pc-w': w + 'px' }} aria-hidden={ariaHidden} />;
  }
  const isRed = RED.has(suit);
  const cls = ['pc', style === 'modern' ? 'pc--modern' : '', isRed ? 'pc--red' : '', playable ? 'pc--playable' : '', dim ? 'pc--dim' : ''].filter(Boolean).join(' ');
  const sym = SUITS[suit] || suit;
  return (
    <div className={cls} style={{ width: w, height: h, '--pc-w': w + 'px' }} aria-hidden={ariaHidden}>
      <div className="pc__corner">
        <div className="pc__rank">{rank}</div>
        <div className="pc__suit-sm">{sym}</div>
      </div>
      <div className="pc__center">{sym}</div>
      <div className="pc__corner pc__corner--br">
        <div className="pc__rank">{rank}</div>
        <div className="pc__suit-sm">{sym}</div>
      </div>
    </div>
  );
}

// Parse "QC" / "10H" / "AD" / "__" → { rank, suit }
function parseCard(code) {
  if (!code || code === '__') return { faceDown: true };
  if (code === 'UNDER') return { special: 'UNDER' };
  const m = code.match(/^(10|[AKQJ98765432])([SHDC])$/);
  if (!m) return { faceDown: true };
  return { rank: m[1], suit: m[2] };
}

function C(code, props = {}) {
  return <PlayingCard {...parseCard(code)} {...props} />;
}

// ─── Brand mark — a miniature Q♣ playing card ──────────────
// (Queen of Clubs is the highest trump in Sheepshead — the crown of the
// deck.) Rendering it as an actual tiny card, slightly tilted, keeps the
// brand mark cohesive with the cards on the table.
function MiniCardMark({ h = 24, rot = -8 }) {
  const w = Math.max(12, Math.round(h / 1.45));
  return (
    <span aria-hidden="true" style={{
      width: w, height: h, flexShrink: 0,
      background: 'var(--card-paper)',
      border: '1px solid var(--card-edge)',
      borderRadius: Math.max(2, Math.round(h * 0.09)),
      boxShadow: 'var(--shadow-1)',
      transform: `rotate(${rot}deg)`,
      display: 'inline-flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      lineHeight: 1, fontFamily: 'var(--font-display)',
      color: 'var(--card-black)',
    }}>
      <span style={{ fontSize: Math.round(h * 0.34), lineHeight: 1 }}>Q</span>
      <span style={{ fontSize: Math.round(h * 0.3), lineHeight: 1, marginTop: 1 }}>♣</span>
    </span>
  );
}

// ─── Wordmark ──────────────────────────────────────────────
function Wordmark({ size = 'md', stacked = false, mark = true }) {
  const sizes = {
    sm: { wm: 22, ai: 11, mark: 24, gap: 8 },
    md: { wm: 30, ai: 14, mark: 32, gap: 10 },
    lg: { wm: 64, ai: 26, mark: 64, gap: 16 },
    xl: { wm: 120, ai: 44, mark: 112, gap: 26 },
  };
  const s = sizes[size] || sizes.md;
  return (
    <div style={{ display: 'inline-flex', alignItems: stacked ? 'flex-start' : 'center', gap: s.gap, flexDirection: stacked ? 'column' : 'row' }}>
      {mark && <MiniCardMark h={s.mark} />}
      <div style={{ display: 'inline-flex', alignItems: 'baseline', gap: s.gap * 0.5 }}>
        <span className="ss-display" style={{ fontSize: s.wm, color: 'var(--ink)' }}>Sheepshead</span>
        <span style={{ fontFamily: 'var(--font-ui)', fontSize: s.ai, letterSpacing: '0.22em', color: 'var(--muted)', fontWeight: 500 }}>AI</span>
      </div>
    </div>
  );
}

// ─── Top page chrome (a small navbar with wordmark + nav) ───
function PageChrome({ current, theme, onNav, dense }) {
  const link = (key, label) => (
    <a href="#" onClick={(e) => { e.preventDefault(); onNav && onNav(key); }}
       style={{
         fontFamily: 'var(--font-ui)', fontSize: dense ? 10 : 12, letterSpacing: '0.14em', textTransform: 'uppercase',
         color: current === key ? 'var(--ink)' : 'var(--muted)',
         textDecoration: 'none',
         padding: '6px 0',
         borderBottom: current === key ? '1px solid var(--ink)' : '1px solid transparent',
       }}>
      {label}
    </a>
  );
  if (dense) {
    // Mobile: stack into two rows for breathing room
    return (
      <div style={{ borderBottom: '1px solid var(--rule)', padding: '12px 16px 10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <Wordmark size="sm" />
          <div style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--muted)' }}>
            Andrew · <span style={{ color: 'var(--ink)' }}>●</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 16 }}>
          {link('home', 'Lobby')}
          {link('waiting', 'Waiting')}
          {link('table', 'Table')}
          {link('analyze', 'Analyze')}
        </div>
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 56px', borderBottom: '1px solid var(--rule)' }}>
      <Wordmark size="sm" />
      <div style={{ display: 'flex', gap: 24 }}>
        {link('home', 'Lobby')}
        {link('waiting', 'Waiting')}
        {link('table', 'Table')}
        {link('analyze', 'Analyze')}
      </div>
      <div style={{ fontFamily: 'var(--font-ui)', fontSize: 12, color: 'var(--muted)', letterSpacing: '0.08em' }}>
        Andrew · <span style={{ color: 'var(--ink)' }}>online</span>
      </div>
    </div>
  );
}

// ─── Seat avatar (initial in a circle, with optional AI mark) ───
//
// `tone` controls the disc fill so we can convey role at a glance:
//   default | picker (accent) | partner (gold) | you (paper, ink ring)
// `accent` is a legacy boolean — true ⇒ tone='picker'.
function SeatAvatar({ name, isAI, size = 44, accent, tone = 'default' }) {
  if (accent && tone === 'default') tone = 'picker';
  const initial = (name || '?').slice(0, 1).toUpperCase();
  const styles = {
    default: { bg: 'var(--bg-page-deep)', border: 'var(--rule-strong)', fg: 'var(--ink)' },
    picker:  { bg: 'var(--accent)',        border: 'var(--accent)',      fg: 'var(--card-paper)' },
    partner: { bg: 'var(--gold)',          border: 'var(--gold)',        fg: 'var(--card-paper)' },
    you:     { bg: 'var(--card-paper)',    border: 'var(--ink)',         fg: 'var(--ink)' },
  };
  const s = styles[tone] || styles.default;
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%',
      background: s.bg,
      border: '1px solid ' + s.border,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-display)', fontSize: size * 0.5,
      color: s.fg,
      position: 'relative',
      flexShrink: 0,
    }}>
      {initial}
      {isAI && (
        <span style={{
          position: 'absolute', bottom: -3, right: -4,
          fontFamily: 'var(--font-ui)', fontSize: 9, fontWeight: 600, letterSpacing: '0.12em',
          background: 'var(--ink)', color: 'var(--bg-page)',
          padding: '1px 4px', borderRadius: 2,
        }}>AI</span>
      )}
    </div>
  );
}

// Legacy alias — earlier phase-stage code used <AITag /> inline next to
// names. Keep it as a no-op so call sites don't break, but render nothing
// (the disc-corner badge is back to being the AI marker of record).
function AITag() { return null; }

// ─── Chat bubble row ───
function ChatRow({ from, text, time, system }) {
  if (system) {
    return (
      <div style={{ padding: '6px 0' }}>
        <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 14, color: 'var(--muted)' }}>{text}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)', opacity: 0.7, marginTop: 2 }}>{time}</div>
      </div>
    );
  }
  return (
    <div style={{ padding: '6px 0' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <div style={{ fontFamily: 'var(--font-ui)', fontWeight: 600, fontSize: 13, color: 'var(--ink)' }}>{from}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)' }}>{time}</div>
      </div>
      <div style={{ fontFamily: 'var(--font-ui)', fontSize: 14, color: 'var(--ink-soft)', lineHeight: 1.5 }}>{text}</div>
    </div>
  );
}

// ─── Small key/value caption (used in "stat" displays) ───
function Stat({ label, value, valueClass = 'ss-display', size = 28 }) {
  return (
    <div>
      <div className="ss-overline" style={{ marginBottom: 4 }}>{label}</div>
      <div className={valueClass} style={{ fontSize: size, color: 'var(--ink)' }}>{value}</div>
    </div>
  );
}

Object.assign(window, { PlayingCard, parseCard, C, Wordmark, MiniCardMark, PageChrome, SeatAvatar, AITag, ChatRow, Stat, SUITS, SUIT_NAMES });
