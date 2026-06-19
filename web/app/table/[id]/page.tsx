"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";

import { STORAGE_KEYS } from "../../../lib/storage";
import { useIsMobile, useMediaQuery, parseCard } from "../../../lib/ds";
import styles from "./page.module.css";
import { nameForSeat, isAiSeat } from "./utils/seatMath";
import { relSeat } from "./lib/seatLayout";
import {
  derivePhase,
  getSeatRole,
  playStarted as playStartedFn,
  interludeMode,
} from "./lib/phase";
import { useTableSocket, useTrickAnimation, useCallout } from "./hooks";
import GameOverBanner from "./components/GameOverBanner";
import ScoresOverlay from "./components/ScoresOverlay";
import TableHeader from "./components/TableHeader";
import Stage, { type SeatView, type CallOption } from "./components/Stage";
import PlayerHand from "./components/PlayerHand";
import ActionBar from "./components/ActionBar";
import RightRail from "./components/RightRail";
import MobileLogScreen from "./components/MobileLogScreen";

const TOTAL_TRICKS = 6;

const PHASE_LABEL = {
  pick: "Pick or pass",
  bury: "Bury 2 cards",
  call: "Call partner",
  setup: "Setting up",
  play: "Play a card",
  done: "Hand over",
} as const;

const HELPER = {
  pick: "Pick the blind, or pass the buck.",
  bury: "Tap two cards to bury.",
  call: "Choose your partner ace, or go alone.",
  play: "Tap a highlighted card to play.",
  setup: "Setting up the hand…",
  done: "Hand complete.",
} as const;

function rulesBadgeText(
  rules: Record<string, unknown> | undefined,
): string | null {
  if (!rules) return null;
  const partner = rules.partnerMode === 0 ? "Jack of Diamonds" : "Called Ace";
  const scoring = rules.doubleOnTheBump ? "Double on Bump" : "Symmetric";
  return `${partner} · ${scoring}`;
}

const SUIT_NAME: Record<string, string> = {
  C: "Clubs",
  S: "Spades",
  H: "Hearts",
  D: "Diamonds",
};

function buildCallOptions(
  validActions: number[],
  actionLookup: Record<string, string>,
): CallOption[] {
  const out: CallOption[] = [];
  for (const aid of validActions) {
    const label = actionLookup[String(aid)];
    if (!label) continue;
    if (label.startsWith("CALL ")) {
      const rest = label.slice(5); // e.g. "AC" or "AC UNDER"
      const [code, ...mods] = rest.split(" ");
      const { rank, suit } = parseCard(code);
      const under = mods.includes("UNDER");
      const display = suit
        ? `${rank === "A" ? "Ace" : rank} of ${SUIT_NAME[suit] ?? suit}${under ? " (under)" : ""}`
        : label;
      out.push({ actionId: aid, label, code, display });
    } else if (label === "ALONE") {
      out.push({ actionId: aid, label, code: null, display: "Go alone" });
    } else if (label.startsWith("JD")) {
      out.push({
        actionId: aid,
        label,
        code: null,
        display: "Jack of Diamonds",
      });
    }
  }
  return out;
}

export default function TablePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const isMobile = useIsMobile();
  // Scale table + hand cards up on large desktops so the play area fills the
  // extra space instead of leaving a big gap above the hand. Gated on height
  // too, so short/wide screens (where vertical room is the constraint) stay at
  // the base size. Matches the .deskStage height breakpoints in Stage.module.css.
  const isWideDesk = useMediaQuery(
    "(min-width: 1600px) and (min-height: 900px)",
  );
  const isXWideDesk = useMediaQuery(
    "(min-width: 1920px) and (min-height: 1024px)",
  );
  const uiScale = isXWideDesk ? 1.3 : isWideDesk ? 1.15 : 1;
  const trickBoxRef = useRef<HTMLDivElement>(null);
  const [clientId, setClientId] = useState<string>("");

  useEffect(() => {
    if (typeof window === "undefined" || !params?.id) return;
    const stored = window.localStorage.getItem(
      STORAGE_KEYS.clientId(params.id),
    );
    if (stored) setClientId(stored);
    else router.replace("/");
  }, [params?.id, router]);

  const { showPrev, animTrick, triggerCollect, setShowPrev } =
    useTrickAnimation();
  const { callout, showCallout } = useCallout();

  const {
    connected,
    lastState,
    actionLookup,
    chatMessages,
    takeAction,
    closeTable,
    redeal,
    sendChatMessage,
  } = useTableSocket(params?.id, clientId, {
    onTrickComplete: triggerCollect,
    onPickerAnnounced: (name) => showCallout("PICK", `${name} picked`),
    onLeaster: () => showCallout("LEASTER", "All passed · Leaster"),
    onAlone: (name) => showCallout("ALONE", `${name} goes alone`),
    onCall: (name, cardDisplay, under) =>
      showCallout(
        "CALL",
        `${name} calls ${cardDisplay}${under ? " under" : ""}`,
      ),
    onTableClosed: () => showCallout("PICK", "Table closed", 1200),
    onLobbyEvent: (msg) => showCallout("PICK", msg),
  });

  const [showScores, setShowScores] = useState(false);
  const [confirmClose, setConfirmClose] = useState(false);
  const [showMobileLog, setShowMobileLog] = useState(false);

  const isYourTurn = lastState?.actorSeat === lastState?.yourSeat;
  const isHost = lastState?.isHost ?? false;

  const validActionStrings = useMemo(() => {
    const s = new Set<string>();
    if (!lastState) return s;
    for (const id of lastState.valid_actions) {
      const label = actionLookup[String(id)];
      if (label) s.add(label);
    }
    return s;
  }, [lastState, actionLookup]);

  const actionIdByString = useMemo(() => {
    const m: Record<string, number> = {};
    Object.entries(actionLookup).forEach(([id, label]) => {
      const n = Number(id);
      if (Number.isFinite(n)) m[label] = n;
    });
    return m;
  }, [actionLookup]);

  const handleCardClick = useCallback(
    (card: string) => {
      if (!isYourTurn || !card) return;
      for (const lbl of [`PLAY ${card}`, `BURY ${card}`, `UNDER ${card}`]) {
        if (validActionStrings.has(lbl)) {
          const id = actionIdByString[lbl];
          if (id !== undefined) {
            void takeAction(id);
            return;
          }
        }
      }
    },
    [isYourTurn, validActionStrings, actionIdByString, takeAction],
  );

  if (!lastState) {
    return (
      <div className={styles.root}>
        <div className={styles.waiting}>Waiting for state…</div>
      </div>
    );
  }

  const view = lastState.view;
  const table = lastState.table;
  const yourSeat = lastState.yourSeat;
  const started = playStartedFn(lastState);
  const { phase, isLeaster } = derivePhase(lastState);
  const yourMode = interludeMode(validActionStrings);

  // The "kind" used for labels/helper text.
  const kind: keyof typeof PHASE_LABEL =
    phase === "pick"
      ? "pick"
      : phase === "play"
        ? "play"
        : phase === "done"
          ? "done"
          : yourMode === "bury"
            ? "bury"
            : yourMode === "call"
              ? "call"
              : "setup";

  const seats: SeatView[] = [1, 2, 3, 4, 5].map((absSeat) => ({
    absSeat,
    rel: relSeat(absSeat, yourSeat),
    name: nameForSeat(absSeat, table),
    isAI: isAiSeat(absSeat, table),
    role: getSeatRole(lastState, absSeat, started),
    you: absSeat === yourSeat,
  }));

  const displayCards =
    showPrev && view.last_trick ? view.last_trick : view.current_trick;
  const winnerSeat = showPrev ? view.last_trick_winner : null;
  const callOptions = buildCallOptions(lastState.valid_actions, actionLookup);
  const handNumber = (table.resultsHistory?.length ?? 0) + 1;
  const rulesBadge = rulesBadgeText(table.rules);
  const hasLastTrick = view.last_trick?.length === 5;
  const prevText =
    showPrev && hasLastTrick
      ? `Trick to ${nameForSeat(view.last_trick_winner, table)} · ${view.last_trick_points ?? 0} pts`
      : null;

  const lastMessage = chatMessages.length
    ? chatMessages[chatMessages.length - 1].body
    : "Hand in progress";

  const stage = (
    <Stage
      seats={seats}
      yourSeat={yourSeat}
      phase={phase}
      isLeaster={isLeaster}
      yourMode={yourMode}
      isYourTurn={isYourTurn}
      handLen={view.hand.length}
      trickIndex={view.current_trick_index}
      totalTricks={TOTAL_TRICKS}
      displayCards={displayCards}
      winnerSeat={winnerSeat}
      showPrev={showPrev}
      prevText={prevText}
      callOptions={callOptions}
      selectedCall={null}
      onAction={takeAction}
      pickActionId={
        validActionStrings.has("PICK") ? actionIdByString["PICK"] : null
      }
      passActionId={
        validActionStrings.has("PASS") ? actionIdByString["PASS"] : null
      }
      isMobile={isMobile}
      uiScale={uiScale}
      trickBoxRef={trickBoxRef}
      animTrick={animTrick}
      callout={callout?.message ?? null}
    />
  );

  const hand = (
    <PlayerHand
      hand={view.hand}
      isYourTurn={isYourTurn}
      phase={phase}
      yourMode={yourMode}
      validActionStrings={validActionStrings}
      onCardClick={handleCardClick}
      isMobile={isMobile}
      uiScale={uiScale}
    />
  );

  const actionBar = (
    <ActionBar
      yourName={nameForSeat(yourSeat, table)}
      yourSeat={yourSeat}
      isYourTurn={isYourTurn}
      actorName={nameForSeat(lastState.actorSeat, table)}
      helper={HELPER[kind]}
      validActions={lastState.valid_actions}
      actionLookup={actionLookup}
      onTakeAction={takeAction}
      hasLastTrick={hasLastTrick}
      showPrev={showPrev}
      onTogglePrev={() => setShowPrev(!showPrev)}
      onShowScores={() => setShowScores(true)}
      isHost={isHost}
      confirmClose={confirmClose}
      onConfirmClose={setConfirmClose}
      onCloseTable={closeTable}
      isMobile={isMobile}
    />
  );

  const overlays = (
    <>
      {view.is_done && view.final && (
        <GameOverBanner
          final={view.final}
          table={table}
          onRedeal={redeal}
          onShowScores={() => setShowScores(true)}
          isHost={isHost}
          onCloseTable={closeTable}
          onLeave={() => router.push("/")}
        />
      )}
      {showScores && (
        <ScoresOverlay table={table} onClose={() => setShowScores(false)} />
      )}
    </>
  );

  // ---------- Mobile ----------
  if (isMobile) {
    if (showMobileLog) {
      return (
        <div className={styles.root}>
          <MobileLogScreen
            table={table}
            yourSeat={yourSeat}
            chatMessages={chatMessages}
            onSendMessage={sendChatMessage}
            onClose={() => setShowMobileLog(false)}
          />
          {overlays}
        </div>
      );
    }
    return (
      <div className={styles.root}>
        <TableHeader
          roomName={table.name}
          rulesBadge={rulesBadge}
          handNumber={handNumber}
          phaseLabel={PHASE_LABEL[kind]}
          connected={connected}
          isMobile
          onLeave={() => router.push("/")}
          onShowScores={() => setShowScores(true)}
        />
        <div className={styles.mobBody}>
          {stage}
          <div className={styles.ribbon} onClick={() => setShowMobileLog(true)}>
            <span className={styles.ribbonTag}>Last</span>
            <span className={styles.ribbonLast}>{lastMessage}</span>
            <span className={styles.ribbonLink}>Log &amp; Chat ↗</span>
          </div>
          {hand}
          {actionBar}
        </div>
        {overlays}
      </div>
    );
  }

  // ---------- Desktop ----------
  return (
    <div className={styles.root}>
      <TableHeader
        roomName={table.name}
        rulesBadge={rulesBadge}
        handNumber={handNumber}
        phaseLabel={PHASE_LABEL[kind]}
        connected={connected}
        isMobile={false}
        onLeave={() => router.push("/")}
        onShowScores={() => setShowScores(true)}
      />
      <div className={styles.deskMain}>
        <div className={styles.deskPlay}>
          <span
            className={`${styles.connDot} ${connected ? styles.connOk : styles.connBad}`}
            role="status"
            aria-label={connected ? "Connected" : "Disconnected"}
          />
          {stage}
          {hand}
          {actionBar}
        </div>
        <RightRail
          table={table}
          yourSeat={yourSeat}
          chatMessages={chatMessages}
          onSendMessage={sendChatMessage}
        />
      </div>
      {overlays}
    </div>
  );
}
