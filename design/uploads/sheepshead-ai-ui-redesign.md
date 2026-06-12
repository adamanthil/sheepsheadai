# Sheepshead AI — UI Design Brief

## Project Overview

**Sheepshead AI** is a multiplayer, browser-based implementation of Sheepshead — a 5-player trick-taking card game from Wisconsin with a 32-card deck. The application pairs human players with an AI opponent system powered by a deep-learning PPO model. Players create or join named tables, configure game rules in a lobby, play hands in real-time via WebSocket, and can optionally inspect raw AI decision-making in a separate analysis view.

The backend is a FastAPI server on port 9000 with a PostgreSQL database. The frontend is a Next.js 15 / React 19 app with no external UI library — all styling is hand-written CSS Modules. Communication is via REST + WebSocket.

We would like to completely redesign the UI of the application to be prettier and more user-friendly on both desktop and mobile. What follows below is a detailed breakdown of the current UI and some identified areas for improvement.

---

## Game Rules Summary (for context)

- **5 players**, 32-card deck (standard 52 minus all 6s and below).
- Each player gets **6 cards**; a **2-card blind** sits face-down.
- **Trump suit**: all Queens and Jacks (by rank), then all Diamonds.
- **Point values**: Aces=11, 10s=10, Kings=4, Queens=3, Jacks=2. Total = 120 points.
- **Roles are dynamic per hand**: one player may "pick" the blind, buries 2 cards face-down, and either calls a partner (who holds a specific card) or plays "Alone" (1v4).
- **Picker's team** needs >60 points to win; ties go to the defenders (3 players).
- **Leaster variant**: if nobody picks, all play for themselves and the player taking the fewest points wins.
- **Two partnership modes**: Jack of Diamonds (secret partner has JD) or Called Ace (picker calls a fail-suit ace; partner is whoever holds it).

---

## Identity & Persistence

The app uses `localStorage` for lightweight session persistence with no login flow:

| Key | Value |
|---|---|
| `sheepshead_player_id` | Persistent UUID for the player across sessions |
| `sheepshead_display_name` | Last-used display name |
| `sheepshead_client_id_<tableId>` | Per-table session token enabling reconnection |

If a player navigates directly to `/waiting/:id` or `/table/:id` without a stored `clientId`, they are silently redirected to `/`.

---

## Pages & User Stories

---

### Page 1: `/` — Home / Lobby

**Goal:** Let players identify themselves, find active tables, and start or join a game.

#### User Stories

1. **As a new player**, I open the app and see a lobby listing all open tables.
2. **As a new player**, I am given a randomly suggested display name (a Hobbit name: Bilbo, Frodo, Gandalf…) which I can keep or overwrite.
3. **As a new player**, I am given a randomly suggested table name (a Shire town: Bywater, Hobbiton…) which I can keep or overwrite.
4. **As a returning player**, my display name and player UUID are restored from `localStorage` automatically.
5. **As a player**, I can type a display name and create a new table, which immediately takes me to the Waiting Room.
6. **As a player**, I can join an existing table from the lobby list (button is disabled if the table is full — 5 human seats taken).
7. **As a curious visitor**, I can navigate to the AI analysis page via a footer link.

#### UI Elements

- Dark green radial gradient background (forest/card table aesthetic)
- Heading: `Sheepshead AI`
- Error banner (red-tinted box, conditionally shown)
- **Controls row:**
  - Table name text input (with random Shire-town placeholder)
  - "Create table" button (disabled while creating)
  - Display name text input (with random Hobbit-name placeholder)
- **Lobby table:**
  - Columns: Name / Status / Players / Join
  - Status column is hidden on mobile
  - Join button is disabled when human count = 5
- **Footer:** `🧠 Analyze AI model decisions` link → `/analyze`

---

### Page 2: `/waiting/[id]` — Waiting Room

**Goal:** Let players choose seats, configure game rules, and wait for the host to start. Connected via WebSocket in real-time.

#### User Stories

1. **As a player**, I arrive in the waiting room after creating or joining a table and see all 5 seat slots.
2. **As a player**, I can click an empty seat slot to claim it.
3. **As a player**, I can see which seats are taken by humans vs AI bots, and which player is in which seat.
4. **As the host**, I can fill all empty seats with AI bots so the game can start with fewer than 5 humans.
5. **As the host**, I can configure **Partner Mode**: Jack of Diamonds or Called Ace.
6. **As the host**, I can configure **Scoring Mode**: Double on Bump or Symmetric.
7. **As the host**, I can start the game, which immediately navigates all connected players to `/table/:id`.
8. **As the host**, I can close the table (two-step confirmation), which disconnects all players and redirects everyone to the lobby.
9. **As a player**, I receive real-time toast notifications (e.g., "Player X joined") via `lobby_event` WebSocket messages.
10. **As a player**, if the host starts the game while I'm watching, I am automatically navigated to the table.
11. **As a player**, I can chat with others in the waiting room via the Chat Panel.

#### UI Elements

- **5 seat cards** in a responsive grid (2-col mobile → 3-col tablet → 5-col desktop)
  - Each card shows seat number, player name + AI badge, "Take seat" button
- **Player count badge:** `X/5`
- **Game Mode section** (host-editable toggles):
  - Partner Mode: JD / Called Ace
  - Scoring: Double on Bump / Symmetric
- **Actions row** (host-only):
  - "Fill empty seats with AI" button
  - "Start" button
  - "Close table" button (two-step confirmation)
- **Chat Panel** at bottom (shared component)
- Toast **callout** notification overlay (top-center, auto-dismisses at 1.8s)

---

### Page 3: `/table/[id]` — Active Game Table

**Goal:** Play Sheepshead in real time. The player sees their hand, the trick area, other players' statuses, and takes actions. Connected via WebSocket.

#### User Stories

1. **As a player**, I see the current trick in the center of the table, with cards positioned around a virtual table (my position at the bottom, others arranged clockwise).
2. **As a player**, I see my full hand of cards at the bottom of the screen.
3. **As a player on my turn**, I can click a highlighted card to play it (or bury it, or go under trump).
4. **As a player**, I can see action buttons for non-card actions: **PICK**, **PASS**, **ALONE**, **CALL <suit>** ace, etc.
5. **As a player**, I see status badges on each seat indicating: PASS, PICK, PICKER, PARTNER, PENDING.
6. **As a player**, when a trick completes, I see the winner's cards briefly before they animate away ("collect" animation, ~3.3s total).
7. **As a player**, I can toggle "Show prev" to display the previous trick's cards and who won them.
8. **As a player**, when a hand ends, I see a **Game Over Banner** summarizing results (teams, scores, points taken, whether it was a Leaster).
9. **As a player**, I can click **Redeal** to start a new hand at the same table.
10. **As the host**, I can close the table from the bottom action bar (two-step confirm).
11. **As any player**, I can open a **Scores Overlay** to see the running score history across all hands played at this table.
12. **As a player**, I can chat with others during the game via the Chat Panel.
13. **As a mobile player**, I get a responsive layout with appropriately sized cards, a collapsible bottom utility bar, and a locked viewport (no pinch-zoom).

#### UI Architecture

The page is composed of:

| Component | Purpose |
|---|---|
| `TrickArea` | Center: 5-position trick layout with card positions, player name labels, callout overlays for pick/leaster/alone announcements |
| `TrickCard` | Individual card in trick + player name label + status badge |
| `CollectOverlay` | rAF-driven animation: cards fly from their positions toward the winner's spot |
| `PlayerHand` | Bottom row: player's hand cards, clickable/highlighted when playable |
| `BottomActionBar` | Sticky bottom bar: player name, turn indicator, action buttons, scores, close table |
| `GameOverBanner` | Inline overlay when hand ends: results summary + Redeal + Show Scores |
| `ScoresOverlay` | Full-screen modal: running scores table per hand |
| `ChatPanel` | Shared chat UI: scrollable history + send form |

#### Trick Area Seat Layout

```
   [2: top-left]   [3: top-right]
[1: mid-left]       [4: mid-right]
         [0: bottom-center (you)]
```

(Seat positions are rotated relative to the player's own seat using circular modular arithmetic.)

#### State Machine (WebSocket messages → UI changes)

| WS Event | UI Response |
|---|---|
| `state` | Update full game state; detect phase changes (pick announced, trick complete, leaster, alone call); trigger trick collect animation |
| `table_update` | Patch table metadata (seats, status) |
| `table_closed` | Hard redirect to `/` |
| `lobby_event` | Show toast callout |
| `chat:init` | Populate full chat history |
| `chat:append` | Append new chat message |

---

### Page 4: `/analyze` — AI Model Analysis

**Goal:** Allow developers/researchers to simulate a full game and inspect every AI decision in detail, including the model's internal state vector, action probabilities, value estimates, and reward trace.

#### User Stories

1. **As a developer**, I can configure a simulation with a seed, partner mode, and deterministic/stochastic toggle, then run a full game instantly.
2. **As a researcher**, I can inspect every action in the game trace: what action the model chose, why (probabilities), and what it estimated the game was worth (value function).
3. **As a researcher**, I can see how the model's discounted return estimate (`Gₜ`) evolved over the game.
4. **As a researcher**, I can inspect the raw 292-element state vector for any decision step, with human-readable labels for each index (hand one-hots, trick cards, game header fields).
5. **As a researcher**, I can see per-seat point predictions vs actual outcome at each step.
6. **As a researcher**, I can see the model's trump tracking probability — how confident it is that it has seen or not seen each trump card.
7. **As a researcher**, I can adjust a **shaping weight** slider to re-blend step rewards between base reward and head rewards, and the timeline recomputes live.

#### UI Elements

- **Controls Panel** (`⚙️ Simulation Settings`):
  - Partner Mode `<select>` (JD / Called Ace)
  - Random Seed `<input>` (integer, optional)
  - Deterministic toggle checkbox
  - Shaping weight `<input type="range">` 0–100%
  - "🎮 Simulate Game" button
- **Results Panel** (after simulation):
  - Meta chips (steps, partner mode, deterministic/seed info, picker/partner)
  - `GameSummary`: hands dealt, blind, bury, picker points, final scores
  - `ActionTimeline`: chronological list of every action in the game
    - Phase dividers (Pick → Partner → Bury → Play) and trick dividers
    - Each `ActionRow` collapsed by default: step number, seat badge (yellow=picker, blue=partner), action text, phase, metric chips (V / Gₜ / r / Win% / E[Ret])
    - Color-coded value estimate: red (low) → green (high)
    - Expanded `ActionRow`: `ActionDetails` (hand cards, trick state, probability bars) + `ActionInsights` (points prediction vs actual, trump tracking, secret partner confidence) + `ActionStateVector` (full 292-element indexed table)

---

## Shared Components

### `PlayingCard`
A CSS-Grid card rendered with rank/suit in three rows (top-left corner, center large suit, bottom-right corner). Props: `label`, `small`, `highlight` (green glow for playable), `width`, `height`, `bigMarks`. Special labels: `"__"` = face-down blank, `"UNDER"` = special token.

### `CardText`
Inline text renderer that auto-formats card codes in action strings (e.g. `PLAY QC` → `PLAY Q♣` with red/black coloring via regex).

### `ChatPanel`
Scrollable chat with auto-scroll-to-bottom, timestamp display, system vs player message styling, slide-in animation, and a send form (max 500 chars, 16px font to prevent iOS zoom).

---

## Navigation Flow

```
/ (Lobby)
├─ Create Table ─────────────────────────────┐
│   POST /api/tables → POST /api/tables/:id/join │
│   ─────────────────────────────────────────┘
│                                            ▼
├─ Join Table ──────────────────────── /waiting/:id (Waiting Room)
│   POST /api/tables/:id/join                │
│                                       WS: status='playing' ─► /table/:id (Game Table)
│                                       WS: table_closed ──────► /
│
└─ /analyze (AI Analysis, standalone, no navigation out)
```

---

## Technical Constraints & Notes

- **No external UI component library** — all styles are hand-written CSS Modules
- **No global CSS imports** — layout styles are inline in `layout.tsx`
- **Viewport locked** to 1× scale (`maximum-scale=1, user-scalable=no`) for mobile card interaction
- **iOS mitigations**: `WebkitTouchCallout: none`, `WebkitTapHighlightColor: transparent`
- **WebSocket reconnect**: not implemented — page refresh or redirect on disconnect
- **API Base**: `window.location.hostname:9000` by default; overridable via `NEXT_PUBLIC_API_BASE`
- **Card aspect ratio**: 1.45 (width:height), used for all responsive sizing calculations
- **Responsive breakpoints**: mobile < 480px, tablet < 768px, desktop ≥ 768px
- **Trick animation timing**: 2000ms preview → 1100ms collect animation → 200ms clear
- **Toast duration**: 1800ms for most `lobby_event` callouts
- The analyze page is completely decoupled from the live game system — it calls a separate `/api/analyze/simulate` endpoint

---

## Current Pain Points / Design Opportunities

Based on the code analysis, the following areas have the most potential for improvement in a redesign:

1. **No design system** — every component reinvents spacing, color, typography, and interaction from scratch in isolated CSS Modules, leading to visual inconsistency.
2. **Home page is purely functional** — no branding, no game explanation, no onboarding for first-time players unfamiliar with Sheepshead.
3. **Waiting room is sparse** — seat slots are plain cards with minimal visual interest; rules configuration is toggle-button only with no explanation of what the modes mean.
4. **Trick area positioning is CSS-class based** — `spotR0`–`spotR4` are hardcoded percentage positions, making it fragile on unusual viewport sizes.
5. **Mobile layout is a compromise** — the game was clearly designed desktop-first; the bottom action bar collapses on mobile but the overall trick/hand layout does not fundamentally change.
6. **Game Over state lacks ceremony** — the banner is an inline overlay in the trick area with minimal visual weight for what should be a satisfying moment.
7. **Scores Overlay is a plain HTML table** — with no charts, no cumulative trend line, no "biggest hand" callouts.
8. **Chat is below the fold on both waiting and table pages** — players may not discover it.
9. **No loading/error UX for WebSocket disconnects** — a lost connection just freezes the game silently.
10. **The analyze page is very developer-facing** — it has no introductory copy explaining what any of the metrics mean to someone unfamiliar with RL terminology.
