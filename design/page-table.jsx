/* ============================================================
   GAME TABLE PAGE

   The table has several "moments":
     trick — mid-trick play (default; Andrew is last to play)
     pick  — pick/pass decision (blind in center)
     bury  — picked the blind, now choose 2 cards to bury
     call  — choose called Ace (partner), or go Alone
     under — Andrew has marked 8♠ as "under" — face-down in hand,
             still playable. Demonstrates the flipped-card affordance.

   Trevor is shown as a REVEALED PARTNER (gold) instead of "Partner?".
   The mystery-partner case is conveyed by absence — non-picker seats
   simply show no role badge when the partner is still secret.
   ============================================================ */

const TABLE_PLAYERS = [
  // seat 0 = bottom (you), then clockwise
  { seat: 0, name: 'Andrew',  ai: false, you: true,  role: null,      played: null, position: 'bc' },
  { seat: 1, name: 'Dan',     ai: true,  you: false, role: 'PASS',    played: '10C', position: 'ml' },
  { seat: 2, name: 'Kyle',    ai: false, you: false, role: 'PICKER',  played: 'JS',  position: 'tl', leadFor: true },
  { seat: 3, name: 'John',    ai: false, you: false, role: 'PASS',    played: '7S',  position: 'tr' },
  { seat: 4, name: 'Trevor',  ai: true,  you: false, role: 'PARTNER', played: '9C',  position: 'mr' },
];

// Andrew's hand. Trumps: Q♦ 10♦ (diamonds are trump in Sheepshead;
// Q's and J's too). Kyle led J♠ which IS trump (jack), so Andrew
// must follow trump if he can — so only Q♦ and 10♦ are playable.
const ANDREW_HAND = [
  { code: 'QD',  playable: true  },
  { code: '10D', playable: true  },
  { code: 'KC',  playable: false },
  { code: '8C',  playable: false },
  { code: '8S',  playable: false },
  { code: '8H',  playable: false },
];

// 8-card hand (after picking blind, before burying)
const ANDREW_HAND_8 = [
  { code: 'QD',  playable: true },
  { code: '10D', playable: true },
  { code: 'KC',  playable: true },
  { code: '8C',  playable: true, staged: true },  // staged to bury
  { code: '8S',  playable: true },
  { code: '8H',  playable: true, staged: true },  // staged to bury
  { code: 'AS',  playable: true },
  { code: '7H',  playable: true },
];

const SCORES = [
  { name: 'Bilbo',   pts: +6 },
  { name: 'Pippin',  pts: -2 },
  { name: 'Andrew',  pts: +4, you: true },
  { name: 'Gandalf', pts: -4, ai: true },
  { name: 'Frodo',   pts: -4 },
];

const PHASE_LABEL = {
  trick: 'Play a card',
  pick:  'Pick or pass',
  bury:  'Bury 2 cards',
  call:  'Call your partner',
  under: 'Play (with under)',
};

function TablePage({ viewport, navigate, theme, cardStyle, phase = 'trick', mobileScreen = 'table', handLayout = 'even' }) {
  const isDesktop = viewport === 'desktop';
  return (
    <div className={`theme-${theme}`} style={{
      width: '100%', height: '100%', background: 'var(--bg-page)',
      color: 'var(--ink)', fontFamily: 'var(--font-ui)', overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      {isDesktop
        ? <TableDesktop navigate={navigate} theme={theme} cardStyle={cardStyle} phase={phase} handLayout={handLayout} />
        : <TableMobile  navigate={navigate} theme={theme} cardStyle={cardStyle} phase={phase} mobileScreen={mobileScreen} handLayout={handLayout} />}
    </div>
  );
}

// ─── Desktop ────────────────────────────────────────────────────────
function TableDesktop({ navigate, theme, cardStyle, phase, handLayout }) {
  return (
    <>
      <TableHeader navigate={navigate} phase={phase} />

      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 280px', minHeight: 0 }}>
        {/* Main play column */}
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, padding: '24px 32px 0 56px' }}>
          <StageForPhase phase={phase} cardStyle={cardStyle} />
          <HandForPhase  phase={phase} cardStyle={cardStyle} viewport="desktop" layout={handLayout} />
          <ActionForPhase phase={phase} navigate={navigate} viewport="desktop" />
        </div>

        {/* Right rail — scoreboard + history + chat */}
        <div style={{ borderLeft: '1px solid var(--rule)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <Scoreboard />
          <HandHistory />
          <ChatPanelMini messages={[
            { from: 'Pippin', text: 'Don\'t bury all the trump again 🙃', time: '11:02 PM' },
            { from: 'Bilbo',  text: 'No comment.', time: '11:02 PM' },
            { system: true, text: 'Kyle picked. Called the Ace of Hearts.', time: '11:03 PM' },
          ]} />
        </div>
      </div>
    </>
  );
}

function TableHeader({ navigate, phase }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '16px 32px 14px 56px', borderBottom: '1px solid var(--rule)',
      background: 'var(--bg-page)',
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 18 }}>
        <Wordmark size="sm" />
        <div style={{ width: 1, height: 20, background: 'var(--rule)' }} />
        <div className="ss-display" style={{ fontSize: 22 }}>Hobbiton</div>
        <span className="ss-badge ss-badge--quiet">Called Ace · Double on Bump</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 28, fontFamily: 'var(--font-ui)' }}>
        <div><span className="ss-overline">Hand</span> <span className="ss-num" style={{ fontSize: 18, color: 'var(--ink)' }}>7</span></div>
        <div><span className="ss-overline">Phase</span> <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 16, color: 'var(--ink)' }}>{PHASE_LABEL[phase]}</span></div>
        <div style={{ display: 'flex', gap: 10 }}>
          <a className="ss-link" href="#" onClick={(e) => e.preventDefault()} style={{ fontSize: 12 }}>Scores</a>
          <a className="ss-link" href="#" onClick={(e) => e.preventDefault()} style={{ fontSize: 12 }}>Chat</a>
          <a className="ss-link" href="#" onClick={(e) => { e.preventDefault(); navigate && navigate('home'); }} style={{ fontSize: 12, color: 'var(--accent)' }}>Leave</a>
        </div>
      </div>
    </div>
  );
}

// ─── Stage router ────────────────────────────────────────────────────
function StageForPhase({ phase, cardStyle }) {
  if (phase === 'pick') return <PickStageDesktop players={TABLE_PLAYERS} />;
  if (phase === 'bury') return <BuryStageDesktop players={TABLE_PLAYERS} chosen={2} />;
  if (phase === 'call') return <CallStageDesktop players={TABLE_PLAYERS} selected="AC" />;
  // trick + under both show the trick stage; under just modifies the hand
  return <PlayStage cardStyle={cardStyle} />;
}

// ─── The trick stage — 4 player slots clustered around a central pile ──
function PlayStage({ cardStyle }) {
  return (
    <div style={{
      position: 'relative',
      height: 480,
      flexShrink: 0,
      display: 'flex',
      justifyContent: 'center',
      background: 'radial-gradient(ellipse 50% 60% at 50% 52%, color-mix(in oklab, var(--accent-2) 10%, transparent) 0%, transparent 65%)',
    }}>
      <div style={{ position: 'relative', width: '100%', maxWidth: 560, height: '100%' }}>
        <svg viewBox="0 0 560 480" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} aria-hidden="true">
          <ellipse cx="280" cy="260" rx="260" ry="200" fill="none" stroke="var(--rule)" strokeDasharray="2 5" strokeWidth="1" />
        </svg>

        <div style={{
          position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)',
          textAlign: 'center', zIndex: 2,
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
        }}>
          <div className="ss-overline" style={{ fontSize: 10 }}>Trick 3 of 6</div>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            background: 'var(--bg-page)', padding: '6px 14px',
            border: '1px solid var(--accent-2)', borderRadius: 999,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', background: 'var(--accent-2)',
              animation: 'ssPulse 1.6s ease-in-out infinite',
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.18em',
              textTransform: 'uppercase', color: 'var(--ink)',
            }}>
              your turn
            </span>
          </div>
        </div>

        {TABLE_PLAYERS.filter(p => !p.you).map((p) => (
          <PlayerSlot key={p.seat} player={p} cardStyle={cardStyle} />
        ))}
      </div>
      <style>{`@keyframes ssPulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }`}</style>
    </div>
  );
}

function PlayerSlot({ player, cardStyle }) {
  const POSITIONS = {
    tl: { style: { left:  0, top:  10 }, cardSide: 'right' },
    tr: { style: { right: 0, top:  10 }, cardSide: 'left'  },
    ml: { style: { left:  0, top: 280 }, cardSide: 'right' },
    mr: { style: { right: 0, top: 280 }, cardSide: 'left'  },
  };
  const pos = POSITIONS[player.position];
  const isLead = player.leadFor;
  const cardOnRight = pos.cardSide === 'right';

  const card = (
    <div style={{ position: 'relative' }}>
      {player.played
        ? <PlayingCard {...parseCard(player.played)} w={104} style={cardStyle} />
        : <div style={{ width: 104, height: 151, border: '1px dashed var(--rule)', borderRadius: 4 }} />}
      {isLead && (
        <div style={{
          position: 'absolute', top: -10, [cardOnRight ? 'left' : 'right']: -6,
          fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.2em',
          textTransform: 'uppercase', color: 'var(--gold, var(--accent))',
          background: 'var(--bg-page)', padding: '2px 6px',
          border: '1px solid var(--gold, var(--accent))', borderRadius: 2,
        }}>
          Led
        </div>
      )}
    </div>
  );

  return (
    <div style={{
      position: 'absolute',
      ...pos.style,
      display: 'flex',
      flexDirection: cardOnRight ? 'row' : 'row-reverse',
      alignItems: 'center', gap: 14,
    }}>
      <PlayerChip player={player} cardOnRight={cardOnRight} />
      {card}
    </div>
  );
}

// ─── Hand strip router ──────────────────────────────────────────────
function HandForPhase({ phase, cardStyle, viewport, layout }) {
  const isDesktop = viewport === 'desktop';
  const common = { cardStyle, isDesktop, layout };
  if (phase === 'pick')  return <HandStrip cards={ANDREW_HAND} mode="idle"  meta="Your starting hand · waiting on the blind" {...common} />;
  if (phase === 'bury')  return <HandStrip cards={ANDREW_HAND_8} mode="bury" meta="Picked the blind · choose 2 to bury" {...common} />;
  if (phase === 'call')  return <HandStrip cards={ANDREW_HAND} mode="idle"  meta="Buried · now call your partner" {...common} />;
  if (phase === 'under') return <HandStrip cards={ANDREW_HAND} mode="under" meta="8♠ is your under — face-down but still in hand" {...common} />;
  // trick
  return <HandStrip cards={ANDREW_HAND} mode="play" meta="You must follow trump. Two cards are eligible." {...common} />;
}

// ─── HandStrip — three layouts that all fit 6 OR 8 cards.
//
//   even  — flush, gapped, no overlap. Mobile cards shrink so 8 fits.
//   fan   — soft overlap (~30%). Top-left corner of every card visible.
//   tight — heavy overlap (~50%). Most cards show only rank+suit corner.
//
// In overlap modes, z-index increases strictly left-to-right so each
// card's top-left corner stays exposed — including the corners of
// cards sitting next to a lifted (playable / staged / under) card.
// The lifted card's right-edge ring is covered by its right neighbor,
// but the lift + top/left/bottom ring is enough affordance and is
// less important than keeping every card's designation readable.
function HandStrip({ cards, mode, meta, cardStyle, isDesktop, layout = 'even' }) {
  const count = cards.length;

  // Per-layout sizing. Mobile 'even' uses smaller cards so an 8-card hand
  // fits without horizontal scrolling on a 390px viewport.
  const SIZES = {
    even:  { wDesk: 92, wMob: 44, gapDesk: 10, gapMob: 3, overlap: 0 },
    fan:   { wDesk: 96, wMob: 60, gapDesk: 0,  gapMob: 0, overlap: 0.32 },
    tight: { wDesk: 100, wMob: 64, gapDesk: 0, gapMob: 0, overlap: 0.55 },
  };
  const s = SIZES[layout] || SIZES.even;
  const w = isDesktop ? s.wDesk : s.wMob;
  const gap = isDesktop ? s.gapDesk : s.gapMob;
  const overlapPx = Math.round(w * s.overlap);
  const step = layout === 'even' ? gap : -overlapPx;

  return (
    <div style={isDesktop ? {
      flex: 1, minHeight: 0,
      display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
      paddingTop: 12, paddingBottom: 8,
    } : {
      padding: '10px 12px 8px', flexShrink: 0,
    }}>
      <div style={{
        display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
        marginBottom: isDesktop ? 14 : 8,
        padding: isDesktop ? 0 : '0 4px',
        gap: 8,
      }}>
        <div style={{ minWidth: 0 }}>
          <div className="ss-overline" style={{ fontSize: isDesktop ? 11 : 9 }}>
            Your hand · {count} cards
          </div>
          <div style={{
            fontFamily: 'var(--font-display)', fontStyle: 'italic',
            fontSize: isDesktop ? 16 : 12, color: 'var(--muted)', marginTop: 2,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {meta}
          </div>
        </div>
        <HandModeStatus mode={mode} cards={cards} />
      </div>
      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        // Reserve room above for lifted cards (bury -12px) + label tag (16px below)
        paddingTop: 18, paddingBottom: 20,
      }}>
        {cards.map((c, i) => {
          return (
            <div key={c.code + i} style={{
              marginLeft: i === 0 ? 0 : step,
              position: 'relative',
              // Strict left-to-right stacking — keeps every card's top-left
              // designation visible even when the card to its left is lifted.
              zIndex: i,
            }}>
              <HandCard card={c} mode={mode} w={w} cardStyle={cardStyle} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function HandModeStatus({ mode, cards }) {
  if (mode === 'play') return <span className="ss-badge ss-badge--accent2" style={{ fontSize: 10 }}>● Your turn</span>;
  if (mode === 'bury') {
    const n = cards.filter(c => c.staged).length;
    return (
      <span className="ss-badge ss-badge--accent" style={{ fontSize: 10 }}>
        {n} / 2 chosen
      </span>
    );
  }
  if (mode === 'under') return <span className="ss-badge" style={{ fontSize: 10 }}>● Your turn · under in play</span>;
  if (mode === 'idle')  return <span className="ss-badge ss-badge--quiet" style={{ fontSize: 10 }}>Waiting on you</span>;
  return null;
}

// A single hand card that renders one of:
//   play  → highlight playable cards (accent ring + lift); rest sit flat
//   bury  → "staged" cards float up + get a small "bury" tag below
//   under → the under card is shown face-down, ringed accent (playable)
//   idle  → all flat, neutral
// All cards keep full opacity — the lift + ring is the only "playable"
// affordance. Dimming via transparency was hurting hand-read during the
// pick/pass decision, where you need every card legible.
function HandCard({ card, mode, w, cardStyle }) {
  const h = Math.round(w * 1.45);

  // UNDER MODE: 8♠ is face-down and itself playable (the user can play the
  // hidden card during a trick). Everything else uses normal play rules.
  if (mode === 'under') {
    const isUnder = card.code === '8S';
    if (isUnder) {
      return (
        <div style={{ position: 'relative' }}>
          <div style={{
            transform: 'translateY(-6px)',
            boxShadow: '0 0 0 2px var(--accent-2), var(--shadow-2)',
            borderRadius: 6,
          }}>
            <PlayingCard faceDown w={w} />
          </div>
          <div style={{
            position: 'absolute', bottom: -16, left: 0, right: 0,
            textAlign: 'center',
            fontFamily: 'var(--font-mono)', fontSize: 9,
            letterSpacing: '0.2em', textTransform: 'uppercase',
            color: 'var(--accent-2)',
          }}>
            under
          </div>
        </div>
      );
    }
    // Other cards: enforce trump-follow rule like trick mode (full opacity)
    return <PlayingCard {...parseCard(card.code)} w={w} playable={card.playable} style={cardStyle} />;
  }

  if (mode === 'bury') {
    if (card.staged) {
      // staged-to-bury: card flips to back & floats up, with "bury" tag
      return (
        <div style={{ position: 'relative' }}>
          <div style={{
            transform: 'translateY(-12px) rotate(-2deg)',
            boxShadow: '0 0 0 2px var(--accent), var(--shadow-2)',
            borderRadius: 6,
            transition: 'transform .2s ease',
          }}>
            <PlayingCard faceDown w={w} />
          </div>
          <div style={{
            position: 'absolute', bottom: -16, left: 0, right: 0,
            textAlign: 'center',
            fontFamily: 'var(--font-mono)', fontSize: 9,
            letterSpacing: '0.2em', textTransform: 'uppercase',
            color: 'var(--accent)',
          }}>
            bury
          </div>
        </div>
      );
    }
    // unstaged: tappable, all cards bury-eligible during bury phase
    return (
      <div style={{ cursor: 'pointer' }}>
        <PlayingCard {...parseCard(card.code)} w={w} style={cardStyle} />
      </div>
    );
  }

  if (mode === 'play') {
    return <PlayingCard {...parseCard(card.code)} w={w} playable={card.playable} style={cardStyle} />;
  }

  // idle (pick / call phases — cards visible but not actionable).
  // No dimming — full opacity, since during pick/pass the user is
  // reading their hand to decide. Lack of green ring / lift is enough
  // to signal "not your turn to play this card."
  return <PlayingCard {...parseCard(card.code)} w={w} style={cardStyle} />;
}

// ─── Action bar router ──────────────────────────────────────────────
function ActionForPhase({ phase, navigate, viewport }) {
  const isDesktop = viewport === 'desktop';
  const Wrap = isDesktop ? ActionBarDesktopShell : ActionBarMobileShell;
  if (phase === 'pick') {
    return (
      <Wrap helper="2 face-down cards become yours — or pass the buck." navigate={navigate}>
        <button className="ss-btn ss-btn--accent">Pick the blind</button>
        <button className="ss-btn ss-btn--ghost">Pass</button>
      </Wrap>
    );
  }
  if (phase === 'bury') {
    return (
      <Wrap helper="2 of 2 chosen — confirm or swap." navigate={navigate}>
        <button className="ss-btn ss-btn--ghost ss-btn--sm">Clear</button>
        <button className="ss-btn ss-btn--accent">Confirm bury</button>
      </Wrap>
    );
  }
  if (phase === 'call') {
    return (
      <Wrap helper="A♣ chosen — whoever holds it is your partner." navigate={navigate}>
        <button className="ss-btn ss-btn--ghost ss-btn--sm">Go alone instead</button>
        <button className="ss-btn ss-btn--accent">Call A♣</button>
      </Wrap>
    );
  }
  if (phase === 'under') {
    return (
      <Wrap helper="tap a highlighted card · or tap your under to play face-down" navigate={navigate}>
        <button className="ss-btn ss-btn--ghost ss-btn--sm">Show last trick</button>
      </Wrap>
    );
  }
  // trick
  return (
    <Wrap helper="tap a highlighted card to play" navigate={navigate}>
      <button className="ss-btn ss-btn--ghost ss-btn--sm">Show last trick</button>
      <button className="ss-btn ss-btn--ghost ss-btn--sm" onClick={() => navigate && navigate('waiting')}>Redeal</button>
    </Wrap>
  );
}

function ActionBarDesktopShell({ helper, children }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      borderTop: '1px solid var(--rule)', padding: '14px 0', marginTop: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <SeatAvatar name="Andrew" size={32} tone="you" />
        <div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <div className="ss-display" style={{ fontSize: 20 }}>Andrew</div>
            <span className="ss-badge" style={{ fontSize: 9 }}>You</span>
          </div>
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 13, color: 'var(--muted)' }}>seat 5 · your decision</div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 14, color: 'var(--muted)', marginRight: 8 }}>
          {helper}
        </span>
        {children}
      </div>
    </div>
  );
}

function ActionBarMobileShell({ helper, children, navigate }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 8,
      borderTop: '1px solid var(--rule)', padding: '10px 16px 12px',
      background: 'var(--bg-page)',
      flexShrink: 0,
    }}>
      <div style={{
        fontFamily: 'var(--font-display)', fontStyle: 'italic',
        fontSize: 12, color: 'var(--muted)', textAlign: 'center',
      }}>
        {helper}
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
        {children}
      </div>
    </div>
  );
}

// ─── Right rail ──────────────────────────────────────────────────────
function Scoreboard() {
  return (
    <div style={{ padding: '20px 22px', borderBottom: '1px solid var(--rule)' }}>
      <div className="ss-overline" style={{ marginBottom: 10 }}>Running Scores · 6 hands</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {SCORES.map((s) => (
          <div key={s.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 16, color: s.you ? 'var(--ink)' : 'var(--ink-soft)', fontWeight: s.you ? 500 : 400 }}>{s.name}</div>
              {s.ai && <AITag />}
              {s.you && <span className="ss-badge" style={{ fontSize: 9 }}>You</span>}
            </div>
            <div className="ss-num" style={{
              fontSize: 16,
              color: s.pts > 0 ? 'var(--accent-2)' : s.pts < 0 ? 'var(--accent)' : 'var(--muted)',
              fontWeight: 500,
            }}>
              {s.pts > 0 ? '+' : ''}{s.pts}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function HandHistory() {
  const events = [
    { phase: 'Pick',    text: 'Dan, Trevor passed.', muted: true },
    { phase: 'Pick',    text: 'Kyle picked the blind.', strong: true },
    { phase: 'Bury',    text: 'Kyle buried 2 cards.' },
    { phase: 'Call',    text: 'Kyle called the Ace of Hearts.', red: true },
    { phase: 'Trick 1', text: 'Won by Kyle · 14 pts' },
    { phase: 'Trick 2', text: 'Won by John · 12 pts' },
  ];
  return (
    <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--rule)', flexShrink: 0 }}>
      <div className="ss-overline" style={{ marginBottom: 10 }}>This Hand</div>
      <div style={{ display: 'grid', gridTemplateColumns: '58px 1fr', columnGap: 10, rowGap: 6, fontFamily: 'var(--font-ui)', fontSize: 12 }}>
        {events.map((e, i) => (
          <React.Fragment key={i}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.1em', paddingTop: 1 }}>{e.phase}</span>
            <span style={{
              color: e.strong ? 'var(--ink)' : e.red ? 'var(--card-red)' : e.muted ? 'var(--muted)' : 'var(--ink-soft)',
              fontWeight: e.strong ? 500 : 400,
              lineHeight: 1.35,
            }}>{e.text}</span>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function ChatPanelMini({ messages }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{ padding: '14px 22px 6px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="ss-overline">Chat</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)' }}>{messages.length}</div>
      </div>
      <div style={{ padding: '4px 22px', flex: 1, overflow: 'auto' }}>
        {messages.map((m, i) => <ChatRow key={i} {...m} />)}
      </div>
      <div style={{ padding: '10px 16px', borderTop: '1px solid var(--rule)', display: 'flex', gap: 6, alignItems: 'center' }}>
        <input placeholder="Message…" style={{
          flex: 1, border: 'none', background: 'transparent', outline: 'none',
          fontFamily: 'var(--font-ui)', fontSize: 13, color: 'var(--ink)', padding: '4px 2px',
        }} />
        <button className="ss-btn ss-btn--sm">→</button>
      </div>
    </div>
  );
}

// ─── Mobile ──────────────────────────────────────────────────────────
function TableMobile({ navigate, theme, cardStyle, phase, mobileScreen, handLayout }) {
  if (mobileScreen === 'log') {
    return <MobileLogChatScreen navigate={navigate} />;
  }
  return (
    <>
      <MobileHeader navigate={navigate} phase={phase} />
      <MobileTrickStage cardStyle={cardStyle} phase={phase} />
      <MobileEventRibbon />
      <HandForPhase phase={phase} cardStyle={cardStyle} viewport="mobile" layout={handLayout} />
      <ActionForPhase phase={phase} navigate={navigate} viewport="mobile" />
    </>
  );
}

function MobileHeader({ navigate, phase }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '12px 16px 11px', borderBottom: '1px solid var(--rule)',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
        <MiniCardMark h={20} />
        <div className="ss-display" style={{ fontSize: 20, lineHeight: 1 }}>Hobbiton</div>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)',
          letterSpacing: '0.14em', textTransform: 'uppercase', whiteSpace: 'nowrap',
        }}>
          H7 · {PHASE_LABEL[phase]}
        </div>
      </div>
      <a
        className="ss-link"
        href="#"
        onClick={(e) => { e.preventDefault(); navigate && navigate('home'); }}
        style={{ fontSize: 11, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase' }}
      >
        Leave
      </a>
    </div>
  );
}

function MobileTrickStage({ cardStyle, phase }) {
  // Phase-specific overlays are shown over the same 2×2 ring layout. For
  // pick/bury/call we replace played-card slots with phase chits and add
  // a central content block; for trick/under we use the played cards.
  const showPlayed = phase === 'trick' || phase === 'under';
  const Kyle   = TABLE_PLAYERS[2];
  const John   = TABLE_PLAYERS[3];
  const Dan    = TABLE_PLAYERS[1];
  const Trevor = TABLE_PLAYERS[4];

  return (
    <div style={{
      flex: 1, minHeight: 0,
      position: 'relative',
      padding: '14px 10px',
      background: 'radial-gradient(ellipse 55% 65% at 50% 50%, color-mix(in oklab, var(--accent-2) 10%, transparent), transparent 70%)',
    }}>
      <svg
        style={{ position: 'absolute', inset: 14, width: 'calc(100% - 28px)', height: 'calc(100% - 28px)', pointerEvents: 'none' }}
        preserveAspectRatio="none" viewBox="0 0 100 100"
        aria-hidden="true"
      >
        <ellipse cx="50" cy="50" rx="48" ry="46" fill="none" stroke="var(--rule)" strokeDasharray="0.6 1.4" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
      </svg>

      <div style={{
        position: 'relative',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: '1fr 1fr',
        height: '100%',
      }}>
        <MobileSeatCell player={Kyle}   corner="tl" cardStyle={cardStyle} phase={phase} showPlayed={showPlayed} />
        <MobileSeatCell player={John}   corner="tr" cardStyle={cardStyle} phase={phase} showPlayed={showPlayed} />
        <MobileSeatCell player={Dan}    corner="bl" cardStyle={cardStyle} phase={phase} showPlayed={showPlayed} />
        <MobileSeatCell player={Trevor} corner="br" cardStyle={cardStyle} phase={phase} showPlayed={showPlayed} />
      </div>

      <MobileStageCenterOverlay phase={phase} />

      <style>{`@keyframes ssPulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }`}</style>
    </div>
  );
}

// Reusable "Alone" card-shaped panel — used in the mobile call overlay
// to match the desktop treatment. `selected` matches a card's playable
// ring + lift.
function AlonePanel({ w = 72, selected = false }) {
  const h = Math.round(w * 1.45);
  return (
    <div style={{
      width: w, height: h,
      border: '1px solid ' + (selected ? 'var(--accent-2)' : 'var(--rule-strong)'),
      background: 'var(--bg-card)',
      borderRadius: 6,
      boxShadow: selected ? '0 0 0 2px var(--accent-2), var(--shadow-2)' : 'var(--shadow-1)',
      transform: selected ? 'translateY(-6px)' : 'none',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: 6, textAlign: 'center',
    }}>
      <div className="ss-display" style={{
        fontSize: Math.round(w * 0.28), lineHeight: 1,
      }}>
        Alone
      </div>
      <div style={{
        fontFamily: 'var(--font-display)', fontStyle: 'italic',
        fontSize: Math.max(8, Math.round(w * 0.13)),
        color: 'var(--muted)', marginTop: 4, lineHeight: 1.25,
      }}>
        vs J♦ rule
      </div>
    </div>
  );
}

function MobileStageCenterOverlay({ phase }) {
  // The center overlay differs by phase. For pick/bury/call we show the
  // phase artifact (blind, bury slots, call options) right in the middle.
  // For trick/under we show the "your turn" pill.
  if (phase === 'pick') {
    return (
      <CenterOverlay>
        <BlindStack w={56} gap={6} />
        <div className="ss-overline" style={{ fontSize: 8, marginTop: 6 }}>The blind</div>
      </CenterOverlay>
    );
  }
  if (phase === 'bury') {
    return (
      <CenterOverlay>
        <div style={{ display: 'flex', gap: 6 }}>
          <PlayingCard faceDown w={48} />
          <div style={{
            width: 48, height: Math.round(48 * 1.45),
            border: '1px dashed var(--accent)', borderRadius: 4,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'color-mix(in oklab, var(--accent) 6%, transparent)',
          }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--accent)', letterSpacing: '0.18em' }}>2/2</span>
          </div>
        </div>
        <div className="ss-overline" style={{ fontSize: 8, marginTop: 6 }}>Burying</div>
      </CenterOverlay>
    );
  }
  if (phase === 'call') {
    // Three non-trump Aces plus an "Alone" card-shaped option. Same
    // treatment as desktop — the Alone panel matches a card's footprint
    // so it reads as a fourth choice rather than a button. Selected
    // option gets the playable ring + lift.
    return (
      <CenterOverlay>
        <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end' }}>
          <PlayingCard {...parseCard('AS')} w={68} />
          <PlayingCard {...parseCard('AH')} w={68} />
          <PlayingCard {...parseCard('AC')} w={68} playable />
          <AlonePanel w={68} />
        </div>
        <div className="ss-overline" style={{ fontSize: 8, marginTop: 10 }}>Call partner</div>
      </CenterOverlay>
    );
  }
  // trick + under: the "your turn" pill
  return (
    <CenterOverlay>
      <div className="ss-overline" style={{ fontSize: 8 }}>Trick 3/6</div>
      <div style={{
        marginTop: 4,
        display: 'inline-flex', alignItems: 'center', gap: 6,
        background: 'var(--bg-page)', padding: '4px 10px',
        border: '1px solid var(--accent-2)', borderRadius: 999,
      }}>
        <span style={{
          width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-2)',
          animation: 'ssPulse 1.6s ease-in-out infinite',
        }} />
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.18em',
          textTransform: 'uppercase', color: 'var(--ink)',
        }}>
          your turn
        </span>
      </div>
    </CenterOverlay>
  );
}

function CenterOverlay({ children }) {
  return (
    <div style={{
      position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)',
      zIndex: 3,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      pointerEvents: 'none',
    }}>
      {children}
    </div>
  );
}

function MobileSeatCell({ player, corner, cardStyle, phase, showPlayed }) {
  const isLeft = corner === 'tl' || corner === 'bl';
  const isTop  = corner === 'tl' || corner === 'tr';
  const role = player.role;
  const cardOnRight = isLeft;
  const tone = role === 'PICKER' ? 'picker' : role === 'PARTNER' ? 'partner' : 'default';

  const chip = (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
      minWidth: 48,
    }}>
      <SeatAvatar name={player.name} isAI={player.ai} tone={tone} size={28} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <div className="ss-display" style={{ fontSize: 13, lineHeight: 1, color: 'var(--ink)', textAlign: 'center' }}>
          {player.name}
        </div>
        {player.ai && <AITag size={8} />}
      </div>
      <RoleBadge role={role} />
    </div>
  );

  let cardContent;
  if (!showPlayed && (phase === 'pick' || phase === 'bury' || phase === 'call')) {
    // Phase-specific small chits replace played cards
    const text = phase === 'pick' ? 'PASSED' : 'WAITING';
    cardContent = (
      <div style={{
        width: 78, height: 113,
        border: '1px dashed var(--rule-strong)',
        borderRadius: 4,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: phase === 'pick' ? 'color-mix(in oklab, var(--bg-page) 60%, transparent)' : 'transparent',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 9,
          letterSpacing: '0.2em', textTransform: 'uppercase',
          color: 'var(--muted)',
        }}>
          {text}
        </span>
      </div>
    );
  } else {
    cardContent = player.played
      ? <PlayingCard {...parseCard(player.played)} w={78} style={cardStyle} />
      : <div style={{ width: 78, height: 113, border: '1px dashed var(--rule)', borderRadius: 4 }} />;
  }

  const card = (
    <div style={{ position: 'relative' }}>
      {cardContent}
      {player.leadFor && showPlayed && (
        <div style={{
          position: 'absolute',
          [isTop ? 'top' : 'bottom']: -8,
          [cardOnRight ? 'left' : 'right']: -4,
          fontFamily: 'var(--font-mono)', fontSize: 8,
          letterSpacing: '0.18em', textTransform: 'uppercase',
          color: 'var(--gold, var(--accent))',
          background: 'var(--bg-page)', padding: '1px 5px',
          border: '1px solid var(--gold, var(--accent))', borderRadius: 2,
          whiteSpace: 'nowrap',
        }}>
          Led
        </div>
      )}
    </div>
  );

  return (
    <div style={{
      display: 'flex',
      justifyContent: isLeft ? 'flex-end' : 'flex-start',
      alignItems:    isTop  ? 'flex-end' : 'flex-start',
      paddingTop:    isTop  ? 6  : 26,
      paddingBottom: isTop  ? 26 : 6,
      paddingLeft:   isLeft ? 6  : 22,
      paddingRight:  isLeft ? 22 : 6,
    }}>
      <div style={{
        display: 'flex',
        flexDirection: cardOnRight ? 'row' : 'row-reverse',
        alignItems: 'center', gap: 8,
      }}>
        {chip}
        {card}
      </div>
    </div>
  );
}

function RoleBadge({ role }) {
  const common = { fontSize: 8, padding: '1px 5px' };
  if (role === 'PICKER')  return <span className="ss-badge ss-badge--accent" style={common}>Picker</span>;
  if (role === 'PARTNER') return <span className="ss-badge ss-badge--gold" style={common}>Partner</span>;
  if (role === 'PASS')    return <span className="ss-badge ss-badge--quiet" style={common}>Pass</span>;
  return null;
}

// Single-line event ribbon — latest event + jumpoff into full log/chat.
function MobileEventRibbon() {
  return (
    <a
      href="#table-log"
      onClick={(e) => {
        // Let the prototype shell handle the route via hashchange
      }}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '9px 16px',
        borderTop: '1px solid var(--rule)',
        borderBottom: '1px solid var(--rule)',
        background: 'color-mix(in oklab, var(--bg-page) 60%, var(--bg-card))',
        textDecoration: 'none', color: 'inherit',
        flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, minWidth: 0 }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 9,
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: 'var(--muted)',
        }}>
          Last
        </span>
        <span style={{
          fontFamily: 'var(--font-display)', fontStyle: 'italic',
          fontSize: 13, color: 'var(--ink-soft)',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>
          Trick 2 to John · <span style={{ color: 'var(--ink)' }}>+12</span> picking team
        </span>
      </div>
      <span style={{
        fontFamily: 'var(--font-ui)', fontSize: 11,
        color: 'var(--accent)', letterSpacing: '0.06em',
        whiteSpace: 'nowrap', marginLeft: 10,
      }}>
        Log &amp; Chat ↗
      </span>
    </a>
  );
}

// ─── Mobile Log & Chat (full-screen) ────────────────────────────────
//
// Mirrors the desktop right rail's three sections (scoreboard / hand
// history / chat) but stacks them and gives the chat composer the
// bottom-anchored position that mobile users expect. A tabs strip at
// top lets you scroll-jump or filter, and a back arrow takes you home
// to the table.

const LOG_EVENTS = [
  { phase: 'Pick',    text: 'Dan passed.', muted: true,  time: '11:00 PM' },
  { phase: 'Pick',    text: 'Trevor passed.', muted: true, time: '11:00 PM' },
  { phase: 'Pick',    text: 'Kyle picked the blind.', strong: true, time: '11:01 PM' },
  { phase: 'Bury',    text: 'Kyle buried 2 cards.', time: '11:01 PM' },
  { phase: 'Call',    text: 'Kyle called the Ace of Hearts.', red: true, time: '11:01 PM' },
  { phase: 'Trick 1', text: 'Andrew led K♦. Kyle won with Q♣ · 14 pts', time: '11:01 PM' },
  { phase: 'Trick 2', text: 'Kyle led 9♥. John won with A♥ · 12 pts', time: '11:02 PM' },
  { phase: 'Trick 3', text: 'Kyle led J♠. Awaiting Andrew…', muted: true, time: '11:03 PM' },
];

const CHAT_MESSAGES = [
  { from: 'Pippin', text: 'Don\'t bury all the trump again 🙃', time: '11:02 PM' },
  { from: 'Bilbo',  text: 'No comment.', time: '11:02 PM' },
  { system: true, text: 'Kyle picked. Called the Ace of Hearts.', time: '11:03 PM' },
  { from: 'Andrew', text: 'About to trump in. Sorry Trevor.', time: '11:03 PM', mine: true },
  { from: 'Trevor', text: 'rude.', time: '11:03 PM' },
];

function MobileLogChatScreen({ navigate }) {
  const [tab, setTab] = React.useState('all');
  return (
    <>
      <MobileLogHeader />
      <MobileLogTabs tab={tab} onTab={setTab} />
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
        {(tab === 'all' || tab === 'scores') && <MobileLogScores />}
        {(tab === 'all' || tab === 'log')    && <MobileLogTimeline />}
        {(tab === 'all' || tab === 'chat')   && <MobileLogChat />}
      </div>
      <MobileLogComposer />
    </>
  );
}

function MobileLogHeader() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '12px 16px 11px', borderBottom: '1px solid var(--rule)',
      flexShrink: 0,
    }}>
      <a
        href="#table"
        style={{
          textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6,
          fontFamily: 'var(--font-ui)', fontSize: 11, letterSpacing: '0.08em',
          textTransform: 'uppercase', color: 'var(--accent)',
        }}
      >
        ← Table
      </a>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <div className="ss-display" style={{ fontSize: 18, lineHeight: 1 }}>Log &amp; Chat</div>
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)',
        letterSpacing: '0.14em', textTransform: 'uppercase',
      }}>
        Hand 7
      </div>
    </div>
  );
}

function MobileLogTabs({ tab, onTab }) {
  const tabs = [
    { key: 'all',    label: 'All' },
    { key: 'scores', label: 'Scores' },
    { key: 'log',    label: 'This hand' },
    { key: 'chat',   label: 'Chat' },
  ];
  return (
    <div style={{
      display: 'flex',
      borderBottom: '1px solid var(--rule)',
      flexShrink: 0,
    }}>
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onTab(t.key)}
          style={{
            flex: 1, padding: '10px 6px',
            background: 'transparent', border: 'none',
            borderBottom: tab === t.key ? '2px solid var(--ink)' : '2px solid transparent',
            fontFamily: 'var(--font-ui)', fontSize: 11,
            letterSpacing: '0.1em', textTransform: 'uppercase',
            color: tab === t.key ? 'var(--ink)' : 'var(--muted)',
            cursor: 'pointer',
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

function MobileLogScores() {
  return (
    <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--rule)' }}>
      <div className="ss-overline" style={{ marginBottom: 10, fontSize: 9 }}>Running scores · 6 hands</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {SCORES.map((s) => (
          <div key={s.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                fontFamily: 'var(--font-display)', fontSize: 17,
                color: s.you ? 'var(--ink)' : 'var(--ink-soft)',
                fontWeight: s.you ? 500 : 400,
              }}>{s.name}</div>
              {s.ai && <AITag />}
              {s.you && <span className="ss-badge" style={{ fontSize: 9 }}>You</span>}
            </div>
            <div className="ss-num" style={{
              fontSize: 16,
              color: s.pts > 0 ? 'var(--accent-2)' : s.pts < 0 ? 'var(--accent)' : 'var(--muted)',
              fontWeight: 500,
            }}>
              {s.pts > 0 ? '+' : ''}{s.pts}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MobileLogTimeline() {
  return (
    <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--rule)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <div className="ss-overline" style={{ fontSize: 9 }}>This hand</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)' }}>
          {LOG_EVENTS.length} events
        </div>
      </div>
      <div style={{
        display: 'grid', gridTemplateColumns: '64px 1fr auto',
        columnGap: 10, rowGap: 10,
        fontFamily: 'var(--font-ui)', fontSize: 12,
      }}>
        {LOG_EVENTS.map((e, i) => (
          <React.Fragment key={i}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 9,
              color: 'var(--muted)', textTransform: 'uppercase',
              letterSpacing: '0.1em', paddingTop: 2,
            }}>
              {e.phase}
            </span>
            <span style={{
              color: e.strong ? 'var(--ink)' : e.red ? 'var(--card-red)' : e.muted ? 'var(--muted)' : 'var(--ink-soft)',
              fontWeight: e.strong ? 500 : 400,
              lineHeight: 1.35,
            }}>
              {e.text}
            </span>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 9,
              color: 'var(--muted)', paddingTop: 2,
            }}>
              {e.time}
            </span>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function MobileLogChat() {
  return (
    <div style={{ padding: '14px 18px 18px', flex: 1, minHeight: 0 }}>
      <div className="ss-overline" style={{ marginBottom: 12, fontSize: 9 }}>Chat</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {CHAT_MESSAGES.map((m, i) => (
          <MobileChatBubble key={i} {...m} />
        ))}
      </div>
    </div>
  );
}

function MobileChatBubble({ from, text, time, system, mine }) {
  if (system) {
    return (
      <div style={{
        alignSelf: 'center', textAlign: 'center', padding: '6px 0',
        maxWidth: 280,
      }}>
        <div style={{
          fontFamily: 'var(--font-display)', fontStyle: 'italic',
          fontSize: 13, color: 'var(--muted)', lineHeight: 1.35,
        }}>
          {text}
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)',
          opacity: 0.6, marginTop: 2, letterSpacing: '0.08em',
        }}>
          {time}
        </div>
      </div>
    );
  }
  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      alignSelf: mine ? 'flex-end' : 'flex-start',
      maxWidth: '78%',
      marginBottom: 4,
    }}>
      {!mine && (
        <div style={{
          fontFamily: 'var(--font-ui)', fontWeight: 600, fontSize: 11,
          color: 'var(--ink)', marginBottom: 2, paddingLeft: 2,
        }}>
          {from}
        </div>
      )}
      <div style={{
        padding: '8px 11px',
        background: mine ? 'var(--ink)' : 'var(--bg-card)',
        color: mine ? 'var(--bg-page)' : 'var(--ink-soft)',
        border: mine ? '1px solid var(--ink)' : '1px solid var(--rule)',
        borderRadius: 10,
        borderBottomRightRadius: mine ? 2 : 10,
        borderBottomLeftRadius:  mine ? 10 : 2,
        fontFamily: 'var(--font-ui)', fontSize: 13, lineHeight: 1.4,
      }}>
        {text}
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)',
        marginTop: 3, paddingLeft: mine ? 0 : 2, paddingRight: mine ? 4 : 0,
        textAlign: mine ? 'right' : 'left',
        letterSpacing: '0.08em',
      }}>
        {time}
      </div>
    </div>
  );
}

function MobileLogComposer() {
  return (
    <div style={{
      display: 'flex', gap: 8, alignItems: 'center',
      padding: '10px 14px',
      borderTop: '1px solid var(--rule)',
      background: 'var(--bg-page)',
      flexShrink: 0,
    }}>
      <input
        placeholder="Message the table…"
        style={{
          flex: 1, border: '1px solid var(--rule)',
          background: 'var(--bg-card)',
          borderRadius: 999, padding: '8px 14px',
          fontFamily: 'var(--font-ui)', fontSize: 13, color: 'var(--ink)',
          outline: 'none',
        }}
      />
      <button className="ss-btn" style={{ borderRadius: 999, padding: '8px 14px' }}>Send</button>
    </div>
  );
}

Object.assign(window, { TablePage });
