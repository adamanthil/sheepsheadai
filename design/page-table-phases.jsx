/* ============================================================
   GAME TABLE — phase-specific stage variants

   The table has more "moments" than just mid-trick play. Each phase
   re-skins the central stage, the hand strip's mode, and the action
   bar's prompts. This file exports those variants; page-table.jsx is
   the shell that picks between them.

   Phases:
     pick   — Andrew is last to pick/pass. Blind sits in the center.
     bury   — Andrew picked. Choose 2 cards to bury face-down.
     call   — Andrew choose a non-trump Ace as partner, or go Alone.
     under  — Mid-trick play with Andrew's 8♠ marked as "under"
              (face-down in hand, still playable as a hidden card).
   ============================================================ */

// ─── Pick / Pass ─────────────────────────────────────────────
//
// The center shows the BLIND: two face-down cards waiting to be picked.
// Andrew's a soft pendulum between PICK and PASS; the action bar carries
// the two big buttons. Others around the table all show "Passed" so the
// player understands "the buck has come to you."

function PickStageDesktop({ players }) {
  // Re-use the same 4-seat ring as the trick stage but swap played cards
  // for "Passed" chits and put the blind in the middle.
  return (
    <StageRing>
      {/* Center: the blind */}
      <StageCenter>
        <BlindStack w={104} gap={10} />
        <div style={{ marginTop: 14, textAlign: 'center' }}>
          <div className="ss-overline" style={{ fontSize: 10 }}>The blind</div>
          <div style={{
            fontFamily: 'var(--font-display)', fontStyle: 'italic',
            fontSize: 14, color: 'var(--muted)', marginTop: 4,
          }}>
            two cards face-down
          </div>
        </div>
      </StageCenter>

      {players.filter(p => !p.you).map((p) => (
        <RingSeat key={p.seat} player={p} position={p.position} cardW={104}>
          <PassedChit />
        </RingSeat>
      ))}
    </StageRing>
  );
}

function BlindStack({ w = 104, gap = 8 }) {
  const h = Math.round(w * 1.45);
  return (
    <div style={{
      position: 'relative', width: w * 2 + gap, height: h,
      display: 'flex', justifyContent: 'center',
    }}>
      <div style={{ position: 'absolute', left: 0,                transform: 'rotate(-4deg)' }}><PlayingCard faceDown w={w} /></div>
      <div style={{ position: 'absolute', left: w + gap - 8,      transform: 'rotate(3deg)'  }}><PlayingCard faceDown w={w} /></div>
    </div>
  );
}

function PassedChit({ size = 'md' }) {
  const small = size === 'sm';
  return (
    <div style={{
      width: small ? 78 : 104, height: small ? 113 : 151,
      border: '1px dashed var(--rule-strong)',
      borderRadius: 4,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'color-mix(in oklab, var(--bg-page) 60%, transparent)',
    }}>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: small ? 9 : 11,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: 'var(--muted)',
      }}>
        passed
      </span>
    </div>
  );
}

// ─── Bury (after picking) ───────────────────────────────────
//
// Center: two slots showing what Andrew is burying (one filled w/ a
// chosen card face-down, one waiting). The hand strip enters BURY mode
// where 2 cards can be flagged for burial.

function BuryStageDesktop({ players, chosen = 1 }) {
  return (
    <StageRing>
      <StageCenter>
        <div style={{ display: 'flex', gap: 14 }}>
          {[0, 1].map((i) => (
            <div key={i} style={{ position: 'relative' }}>
              {i < chosen
                ? <PlayingCard faceDown w={104} />
                : <div style={{
                    width: 104, height: 151,
                    border: '1px dashed var(--accent)',
                    borderRadius: 4,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'color-mix(in oklab, var(--accent) 6%, transparent)',
                  }}>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontSize: 10,
                      letterSpacing: '0.2em', textTransform: 'uppercase',
                      color: 'var(--accent)',
                    }}>
                      slot {i + 1}
                    </span>
                  </div>}
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14, textAlign: 'center' }}>
          <div className="ss-overline" style={{ fontSize: 10 }}>Burying</div>
          <div style={{
            fontFamily: 'var(--font-display)', fontStyle: 'italic',
            fontSize: 14, color: 'var(--muted)', marginTop: 4,
          }}>
            {chosen} of 2 chosen · tap a hand card to bury
          </div>
        </div>
      </StageCenter>

      {players.filter(p => !p.you).map((p) => (
        <RingSeat key={p.seat} player={p} position={p.position} cardW={104}>
          <WaitingChit text="Waiting" />
        </RingSeat>
      ))}
    </StageRing>
  );
}

function WaitingChit({ text = 'Waiting' }) {
  return (
    <div style={{
      width: 104, height: 151,
      border: '1px dashed var(--rule)',
      borderRadius: 4,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10,
        letterSpacing: '0.22em', textTransform: 'uppercase',
        color: 'var(--muted)',
      }}>
        {text}
      </span>
    </div>
  );
}

// ─── Call (partner) ─────────────────────────────────────────
//
// Center: three non-trump aces side-by-side as selectable cards, plus an
// "Alone" option as a card-shaped panel on the right. The currently-
// selected option gets the accent ring (same treatment as a playable
// card in the trick phase — consistent affordance).

const CALL_OPTIONS = [
  { kind: 'card', code: 'AS', label: 'Ace of Spades' },
  { kind: 'card', code: 'AH', label: 'Ace of Hearts' },
  { kind: 'card', code: 'AC', label: 'Ace of Clubs' },
  { kind: 'alone' },
];

function CallStageDesktop({ players, selected = 'AC' }) {
  return (
    <StageRing>
      <StageCenter wide>
        <div className="ss-overline" style={{ fontSize: 10, marginBottom: 12, textAlign: 'center' }}>
          Choose your partner
        </div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'flex-end' }}>
          {CALL_OPTIONS.map((o, i) => {
            const isSel = (o.kind === 'card' && selected === o.code) || (o.kind === 'alone' && selected === 'ALONE');
            if (o.kind === 'card') {
              return (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                  <PlayingCard {...parseCard(o.code)} w={96} playable={isSel} />
                  <div style={{
                    fontFamily: 'var(--font-display)', fontStyle: 'italic',
                    fontSize: 12, color: isSel ? 'var(--ink)' : 'var(--muted)',
                  }}>
                    {o.label}
                  </div>
                </div>
              );
            }
            return (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <div style={{
                  width: 96, height: Math.round(96 * 1.45),
                  border: '1px solid ' + (isSel ? 'var(--accent-2)' : 'var(--rule-strong)'),
                  background: 'var(--bg-card)',
                  borderRadius: 6,
                  boxShadow: isSel ? '0 0 0 2px var(--accent-2), var(--shadow-2)' : 'var(--shadow-1)',
                  transform: isSel ? 'translateY(-6px)' : 'none',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                  padding: 8, textAlign: 'center',
                }}>
                  <div className="ss-display" style={{ fontSize: 26, lineHeight: 1 }}>Alone</div>
                  <div style={{
                    fontFamily: 'var(--font-display)', fontStyle: 'italic',
                    fontSize: 11, color: 'var(--muted)', marginTop: 6, lineHeight: 1.3,
                  }}>
                    vs J♦ rule
                  </div>
                </div>
                <div style={{
                  fontFamily: 'var(--font-display)', fontStyle: 'italic',
                  fontSize: 12, color: isSel ? 'var(--ink)' : 'var(--muted)',
                }}>
                  Go alone
                </div>
              </div>
            );
          })}
        </div>
      </StageCenter>

      {players.filter(p => !p.you).map((p) => (
        <RingSeat key={p.seat} player={p} position={p.position} cardW={104}>
          <WaitingChit text="Waiting" />
        </RingSeat>
      ))}
    </StageRing>
  );
}

// ─── Shared stage primitives ────────────────────────────────
//
// The trick stage in page-table.jsx hand-rolls its layout for historical
// reasons; the variant stages above use these helpers to stay
// consistent. We don't refactor the trick stage to use these because
// the played-card seat layout is subtly different (chip OUTSIDE, card
// INSIDE leans toward middle); the phase stages need the same chip
// position but the inner content varies.

function StageRing({ children }) {
  return (
    <div style={{
      position: 'relative', height: 480, flexShrink: 0,
      display: 'flex', justifyContent: 'center',
      background: 'radial-gradient(ellipse 50% 60% at 50% 52%, color-mix(in oklab, var(--accent-2) 10%, transparent) 0%, transparent 65%)',
    }}>
      <div style={{ position: 'relative', width: '100%', maxWidth: 560, height: '100%' }}>
        <svg viewBox="0 0 560 480" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} aria-hidden="true">
          <ellipse cx="280" cy="260" rx="260" ry="200" fill="none" stroke="var(--rule)" strokeDasharray="2 5" strokeWidth="1" />
        </svg>
        {children}
      </div>
    </div>
  );
}

function StageCenter({ children, wide }) {
  return (
    <div style={{
      position: 'absolute', left: '50%', top: '50%',
      transform: 'translate(-50%, -50%)',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      zIndex: 2,
      maxWidth: wide ? 480 : 280,
    }}>
      {children}
    </div>
  );
}

function RingSeat({ player, position, cardW = 104, children }) {
  const POSITIONS = {
    tl: { style: { left:  0, top:  10 }, cardSide: 'right' },
    tr: { style: { right: 0, top:  10 }, cardSide: 'left'  },
    ml: { style: { left:  0, top: 280 }, cardSide: 'right' },
    mr: { style: { right: 0, top: 280 }, cardSide: 'left'  },
  };
  const pos = POSITIONS[position];
  const cardOnRight = pos.cardSide === 'right';

  return (
    <div style={{
      position: 'absolute',
      ...pos.style,
      display: 'flex',
      flexDirection: cardOnRight ? 'row' : 'row-reverse',
      alignItems: 'center', gap: 14,
    }}>
      <PlayerChip player={player} cardOnRight={cardOnRight} />
      <div>{children}</div>
    </div>
  );
}

// Shared chip used by both the trick stage seats and the phase stages.
// page-table.jsx still has its own inline copy (we keep both for the
// asymmetry above); this one is used by phase variants.
function PlayerChip({ player, cardOnRight }) {
  const role = player.role;
  const tone = role === 'PICKER' ? 'picker'
            : role === 'PARTNER' ? 'partner'
            : 'default';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 110 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        flexDirection: cardOnRight ? 'row' : 'row-reverse',
      }}>
        <SeatAvatar name={player.name} isAI={player.ai} tone={tone} size={36} />
        <div style={{
          textAlign: cardOnRight ? 'left' : 'right',
          display: 'flex', flexDirection: 'column', gap: 3,
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            flexDirection: cardOnRight ? 'row' : 'row-reverse',
          }}>
            <div className="ss-display" style={{ fontSize: 19, lineHeight: 1, color: 'var(--ink)' }}>{player.name}</div>
            {player.ai && <AITag />}
          </div>
          <div className="ss-overline" style={{ fontSize: 9 }}>Seat {player.seat}</div>
        </div>
      </div>
      <div style={{
        display: 'flex', gap: 4, flexWrap: 'wrap',
        justifyContent: cardOnRight ? 'flex-start' : 'flex-end',
      }}>
        <RoleBadgeDesktop role={role} />
      </div>
    </div>
  );
}

function RoleBadgeDesktop({ role }) {
  if (role === 'PICKER')  return <span className="ss-badge ss-badge--accent">Picker</span>;
  if (role === 'PARTNER') return <span className="ss-badge ss-badge--gold">Partner</span>;
  if (role === 'PASS')    return <span className="ss-badge ss-badge--quiet">Pass</span>;
  return null;
}

Object.assign(window, {
  PickStageDesktop, BuryStageDesktop, CallStageDesktop,
  BlindStack, PassedChit, WaitingChit,
  StageRing, StageCenter, RingSeat, PlayerChip, RoleBadgeDesktop,
  CALL_OPTIONS,
});
