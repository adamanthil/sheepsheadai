# Broadsheet UI Migration — Implementation Plan

Status: in progress · June 2026 · Owner: Andrew

## 1. Context

The production web app (`web/`: Next.js 16 App Router, React 19, TS strict, pure CSS Modules, no UI/state libraries) ships a dark "green felt" theme with system fonts. A complete static prototype of the new broadsheet/newspaper design lives in `design/` (`tokens.css` + JSX pages for Home, Waiting, Table at desktop 1440×900 and mobile 390×780, viewable via `Sheepshead AI Prototype.html`).

This plan migrates the broadsheet design into production **page by page (Home → Waiting → Table)**, keeping the working socket/API layer (`useTableSocket`, REST endpoints, chat protocol) intact, and folds in the refactors that make the UI debuggable and human-readable: explicit phase routing, one card component, a small shared design-system layer, and breaking up the ~1060-line `table/[id]/page.module.css`.

**Locked decisions:** broadsheet theme only (token *names* kept so a second theme could be added later; no switcher UI) · classic card style + fan hand (~30% overlap) only — drop modern/even/tight variants · `/analyze` out of scope (inherits global tokens/fonts harmlessly, no dedicated work) · replace in place; app stays shippable after each phase.

## 2. Foundation decisions

### 2.1 Tokens → `web/app/globals.css`
New global stylesheet imported once in `web/app/layout.tsx` (the only place Next allows global CSS):
- All custom properties from `design/tokens.css` with **broadsheet values promoted onto `:root`** (heirloom values deleted; every token *name* survives: `--bg-page`, `--ink`, `--ink-soft`, `--muted`, `--rule`, `--rule-strong`, `--accent`, `--accent-2`, `--gold`, `--card-*`, `--shadow-*`).
- `--font-display/--font-ui/--font-mono` mapped from next/font CSS variables.
- Minimal base only: `body` reset (margin, bg, color, font), shared keyframes (pulse), tap-highlight/touch-callout resets currently inlined in `layout.tsx`. **No global utility classes** — see 2.3.

### 2.2 Fonts — next/font (self-hosted, no FOUC, offline-friendly)
In `layout.tsx` via `next/font/google`, each with `variable:` attached to `<html>`:
- `Instrument_Serif` (weight 400, normal+italic) → `--font-display`. Skip Newsreader (only a fallback in the prototype).
- `Geist` → `--font-ui` (fallback chain keeps `system-ui` so a missing manifest entry never breaks the build).
- `JetBrains_Mono` → `--font-mono`.
Remove the inline `<body style>` from `layout.tsx`.

### 2.3 Shared design-system layer — `web/lib/ds/`
CSS Modules remain the mechanism. The prototype's `ss-*` classes become **one shared module + a few real components**:

```
web/lib/ds/
  ds.module.css            type & surface primitives from tokens.css:
                           display, overline, eyebrow, num, body, panel, panelInset,
                           divider, headRule, btn/btnGhost/btnAccent/btnLg/btnSm,
                           input, badge + variants (ink/accent/accent2/gold/quiet), link
  PlayingCard.tsx(+css)    THE one card component (replaces lib/components/PlayingCard);
                           API: <PlayingCard code="QC" w={96} playable dim />;
                           code '__' → faceDown, 'UNDER' → italic inset card;
                           width-driven via --pc-w, h = round(w*1.45); classic style only
  cardUtils.ts             parseCard/suitSymbol moved from lib/components, extended for '__'/'UNDER'
  CardText.tsx             moved from lib/components; colors → var(--card-red)/var(--ink)
  SeatAvatar.tsx           initial disc; tones default/picker/partner/you; AI corner badge
  Wordmark.tsx             "Sheepshead AI" lockup, sizes sm/md/lg/xl
  MiniCardMark.tsx         fanned Q♣ J♦ A♥ brand mark
  useMediaQuery.ts         SSR-safe useIsMobile() (matchMedia)
  index.ts
```
Pages compose classes directly (`${ds.btn} ${ds.btnAccent}`) — boring and explicit; React components only where there's real markup/logic.

## 3. Prototype data → server data mapping

| Prototype concept | Server source (`TableStateMsg`) | Status |
|---|---|---|
| Card codes `'QC'/'10D'/'__'` | identical strings in `view.hand`, `view.current_trick`, `view.last_trick` | direct |
| `'UNDER'` special card | engine supports `UNDER <card>` / `PLAY UNDER` actions; current UI already handles both | supported; verify `'UNDER'` token appears in `view.hand` at runtime |
| Seat positions bc/ml/tl/tr/mr | `relSeat(absSeat, yourSeat)` (`utils/seatMath.ts`): rel 0→bc(you), 1→ml, 2→tl, 3→tr, 4→mr | direct |
| name/ai/you | `table.seats`, `table.seatIsAI`, `yourSeat` | direct |
| Role badges PICKER/PARTNER/PASS | existing `getPlayerStatus` (`table/[id]/page.tsx`) — extract to pure function; PENDING renders as pulsing "deciding" chit | direct |
| Hidden partner | `view.partner === 0` → no badge | direct |
| Phase routing | derive explicitly (§4): pick = `!is_leaster && picker===0`; play = existing `playStarted` memo; interlude = `picker>0 && !playStarted`; done = `view.is_done`. Your interlude mode from `valid_actions` labels (BURY… / CALL…/ALONE/JD PARTNER / UNDER…) | direct |
| Bury slots "n of 2" | chosen = `8 − view.hand.length`; one `BURY <card>` action at a time (no client staging/confirm in v1) | simplified |
| Call options | built from `valid_actions`, NOT hardcoded — engine also allows called 10s and `CALL Ax UNDER`; render those as card + small "under" badge | supported |
| "Your turn" pill / Trick n of 6 | `actorSeat===yourSeat`; `view.current_trick_index` | direct |
| "Led" tag on leader's card | not in view | **omit v1** |
| Running scores rail | `table.runningBySeat` + `table.seats`; hand # = `resultsHistory.length + 1` | supported |
| Hand-history timeline panel | no structured event log; server emits system chat lines that carry the narrative | **omit panel v1**; flag as future server work |
| Mobile event ribbon "Last: …" | latest callout / system chat message (client-side) | approximation |
| Rules badge ("Called Ace · Double on Bump") | `table.rules.partnerMode/doubleOnTheBump`; called-ace via `view.called_card_display`/`called_under` | supported |
| Waiting: host/you per seat; ready dots | `table.host`, `seatOccupants` — verify host/you seat attribution; **drop ready dots** (seated = ready) | partial |
| Lobby host column / bot count | `TableSummary.host` (verify name vs id; omit column if id), bots = count of `seatIsAI` | mostly |
| "How to play" page | doesn't exist | omit v1, leave no dead link |
| Game-over banner / scores overlay / redeal / trick-collect animation / callouts | exist in production, absent from prototype | **keep behavior, restyle with tokens** |

## 4. Target architecture (`web/` after migration)

```
web/app/
  layout.tsx                   next/font + imports globals.css
  globals.css                  broadsheet tokens on :root + base + keyframes
  page.tsx                     home — create/join/identity logic kept verbatim, new view
  page.module.css              rewritten
  components/
    home/MastheadBand.tsx      thick-thin rule · "The Card Room" · 3 cards · date
    home/Strapline.tsx
    chat/ChatPanel.tsx(+css)   same props/API, broadsheet restyle
  waiting/[id]/
    page.tsx                   handlers (chooseSeat/fillAI/rules/start/close) kept, new view
    page.module.css            rewritten
    components/SeatCard.tsx     desktop card + mobile row variants
    components/RulesPanel.tsx   segmented controls bound to partnerMode/scoringMode
  table/[id]/
    page.tsx                   thin orchestrator
    page.module.css            page shell only
    lib/phase.ts               derivePhase(), getSeatRole(), playStarted() — pure, explicit
    lib/seatLayout.ts          rel-seat → anchor table (single source for Stage + CollectOverlay)
    hooks/                     useTableSocket/useTrickAnimation/useCallout unchanged;
                               + useHandLayout.ts (container width → fan card width)
    components/
      TableHeader.tsx(+css)    wordmark · table name · rules badge · hand/phase · links
      Stage.tsx(+css)          ellipse ring (desktop) / 2×2 grid (mobile), center router
      RingSeat.tsx             PlayerChip (avatar+name+role badge) + played card / chit
      StageCenter.tsx          per-phase center: TurnPill | BlindStack | BurySlots | CallOptions
      PlayerHand.tsx(+css)     fan layout; play/bury/under affordances
      ActionBar.tsx(+css)      desktop row / mobile sticky; absorbs BottomActionBar logic
      RightRail.tsx(+css)      scoreboard + ChatPanel (desktop)
      MobileLogScreen.tsx(+css) tabs Scores | Chat, state-toggled, bottom composer
      GameOverBanner.tsx       kept, token restyle
      ScoresOverlay.tsx        kept, token restyle
      CollectOverlay.tsx       kept, anchors from seatLayout.ts
  analyze/                     untouched
web/lib/
  ds/                          §2.3
  types.ts, apiBase.ts, storage.ts   unchanged
```

**Not migrated from `design/`:** `tweaks-panel.jsx`, `design-canvas.jsx`, `app.jsx`, `hero-tune.jsx`, `landmark-variants.jsx`, `PageChrome` (home is chromeless; waiting/table have bespoke headers), theme/viewport switchers, heirloom theme, modern/even/tight variants. `design/` stays in the repo as reference.

## 5. Refactors included (the maintainability work)

1. **Explicit phase routing** — `table/[id]/lib/phase.ts`: `TablePhase = 'pick' | 'interlude' | 'play' | 'done'`, `YourInterludeMode = 'bury' | 'call' | 'under' | 'waiting'`; pure functions extracted from the proven memos in `page.tsx`. Components consume these instead of re-checking state shape.
2. **One card component** — `lib/ds/PlayingCard` replaces `lib/components/PlayingCard` everywhere; old `label/width/height/small/bigMarks/highlight` API deleted in Phase 4.
3. **CSS split** — 1060-line `table/[id]/page.module.css` → per-component modules + small shell; `app/styles/ui.module.css` deleted; badges/buttons come from `ds.module.css`.
4. **Simpler responsive machinery** — delete the 217-line `useResponsive`; stage card width via CSS `clamp()`; `useIsMobile()` for layout switch; `useHandLayout(count)` for fan fit. `trickBoxRef` survives only for CollectOverlay pixel math.
5. **Seat geometry single-sourced** — `seatLayout.ts` used by both Stage rendering and CollectOverlay (today `spotStyle` percentages + CSS `spotR0..4` duplicate it).
6. **Dead code removal** — §7 Phase 4.

## 6. Phased steps

Common verification per phase: `cd web && npm run typecheck && npm run lint`; run the python server (port 9000) + `npm run dev`; visual comparison against the prototype (serve `design/` with `python3 -m http.server`, headless-Chrome screenshots at 1440×900 and 390×780 — workflow already proven).

### Phase 0 — Foundation (additive; zero page rewrites)
`app/globals.css` (new), `app/layout.tsx` (fonts + import, drop inline styles), `lib/ds/*` (new; old `lib/components/*` untouched). Verify: all routes load; only intended global change is body font/background; `/analyze` unharmed.

### Phase 1 — Home (`/`)
`app/page.tsx` (render only — keep create/join/identity logic verbatim), `page.module.css` rewrite, `app/components/home/*`. Desktop masthead/hero/form/lobby; mobile stacked; fluid 390–1440. Verify: create→waiting, join, name persistence, screenshots, resize sweep.

### Phase 2 — Waiting (`/waiting/[id]`)
`waiting/[id]/page.tsx` (render only), CSS rewrite, `SeatCard`/`RulesPanel`, ChatPanel restyle. Header/seat grid/rules/action bar/chat. Verify: two-browser session, rules round-trip, fill AI, start.

### Phase 3 — Table (`/table/[id]`) — sub-phased; ships when complete
- **3a Logic extraction (zero visual change):** `phase.ts` + `seatLayout.ts`; refactor consumers.
- **3b Desktop:** TableHeader, Stage/RingSeat/StageCenter, PlayerHand, ActionBar, RightRail; overlay/banner restyle; CollectOverlay re-anchored.
- **3c Mobile:** 2×2 grid, per-phase center, event ribbon, sticky ActionBar, MobileLogScreen.
- **3d Removal:** delete TrickArea/TrickCard/BottomActionBar/useResponsive/`spotStyle` + dead CSS.

### Phase 4 — Consolidation
Delete `web/lib/components/`, `app/styles/ui.module.css`; grep-sweep hardcoded hex; remove unused classes; `npm run build`; full visual sweep.

## 7. Risks & open questions

1. **Fixed artboards → fluid layouts:** 768–1100px undesigned. Stage `max-width` centers; right rail collapses below ~1100px; stage height `min(52dvh, 480px)`.
2. **Your-played-card slot** absent from prototype — bottom-center in-ring slot, reviewed in 3b.
3. **CollectOverlay geometry** must match new seat anchors; desktop ring vs mobile grid need separate anchor tables.
4. **Undesigned-but-real server features:** called 10s, `CALL Ax UNDER`, JD-partner mode (no call stage), leaster styling — options come from `valid_actions` so no dead-ends.
5. **Fan-hand hit targets:** ~30% exposure per card; strict left-to-right z-order, wrapper as click target.
6. **Runtime checks needed:** host/you attribution, lobby host name vs id, `current_trick_index` base, `'UNDER'` in `view.hand`.
7. **`color-mix(in oklab, …)`** (card backs/chits) needs 2023+ browsers.
8. **Deferred (future server work):** structured hand-history timeline, per-seat ready state, "Led" marker, How-to-play page.
