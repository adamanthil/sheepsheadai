"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { STORAGE_KEYS } from "../../../lib/storage";
import { useCountdown } from "../../../lib/hooks/useCountdown";
import { useChatMutes } from "../../../lib/hooks/useChatMutes";
import { useChatUnread } from "../../../lib/hooks/useChatUnread";
import { ds, useIsMobile, useMediaQuery } from "../../../lib/ds";
import styles from "./page.module.css";
import { isAiSeat, nameForSeat } from "./utils/seatMath";
import { RemovePlayerButton } from "../../components/RemovePlayerButton";
import type { ChatMessage } from "../../../lib/types";
import {
  derivePhase,
  playStarted as playStartedFn,
  interludeMode,
  getYourRole,
} from "./lib/phase";
import {
  rulesBadgeText,
  stakeBadgeText,
  buildCallOptions,
  computeIsYourTurn,
  computeIsHost,
  computeValidActionStrings,
  computeActionIdByString,
  computeKind,
  buildSeats,
  DEALER_SEAT,
  computeDisplayCards,
  computeWinnerSeat,
  computeHandNumber,
  computeHasLastTrick,
  computeContext,
  resultLine,
} from "./lib/tableSelectors";
import {
  useTableSocket,
  useTrickAnimation,
  useCallout,
  useStagedActions,
} from "./hooks";
import HandOverBar from "./components/HandOverBar";
import ScoresOverlay from "./components/ScoresOverlay";
import CheatSheet from "../../how-to-play/CheatSheet";
import TableHeader from "./components/TableHeader";
import TableMenu from "./components/TableMenu";
import Stage from "./components/Stage";
import PlayerHand from "./components/PlayerHand";
import TableStatus from "./components/TableStatus";
import { LastTrickButton, showLastTrick } from "./components/stage/chrome";
import SpectatorBar from "./components/SpectatorBar";
import RightRail from "./components/RightRail";
import { ChatPanel } from "../../components/chat";
import MobileLogScreen from "./components/MobileLogScreen";

const TOTAL_TRICKS = 6;

// The turn line's word for what your turn asks of you.
const TURN_LABEL = {
  pick: "Your pick",
  bury: "Bury 2 cards",
  call: "Your call",
  under: "Choose the under",
  play: "Your play",
  setup: "Your move",
  done: "Your move",
} as const;

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

  const {
    showPrev,
    lastTrickOpen,
    animTrick,
    triggerCollect,
    setLastTrickOpen,
  } = useTrickAnimation();
  const { callout, showCallout } = useCallout();

  const {
    connected,
    connectionState,
    departure,
    lastState,
    turnDeadline,
    turnSeconds,
    actionLookup,
    chatMessages,
    takeAction,
    closeTable,
    setCloseAfterHand,
    setLeaveAfterHand,
    nextHand,
    takeSeat,
    kickPlayer,
    sendChatMessage,
  } = useTableSocket(params?.id, clientId, {
    onTrickComplete: triggerCollect,
    onPickerAnnounced: (name) => showCallout("PICK", `${name} picked`),
    onLeaster: () => showCallout("LEASTER", "All passed · Leaster"),
    onDoubler: (multiplier, capped) =>
      showCallout(
        "DOUBLER",
        capped
          ? `All passed · Already at the ${multiplier}× maximum · Redealing`
          : `All passed · Redealing at ${multiplier}×`,
        2600,
      ),
    onAlone: (name) => showCallout("ALONE", `${name} goes alone`),
    onCall: (name, cardDisplay, under) =>
      showCallout(
        "CALL",
        `${name} calls ${cardDisplay}${under ? " under" : ""}`,
      ),
    onTableClosed: () => showCallout("PICK", "Table closed", 1200),
    onLobbyEvent: (msg) => showCallout("PICK", msg),
    onError: (msg) => showCallout("PICK", msg, 2500),
  });

  const secondsLeft = useCountdown(turnDeadline);
  const mutes = useChatMutes(params?.id);
  const [showScores, setShowScores] = useState(false);
  const [showMobileLog, setShowMobileLog] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [showCheatSheet, setShowCheatSheet] = useState(false);

  // The after-hand choices (leave, or as host close) show the moment they
  // are made, ahead of the server's table_update; a refusal puts the
  // server's value back, with the reason shown. A choice is dropped once
  // the server agrees, so later changes from the server show as they come.
  const serverLeaving =
    !!lastState?.yourSeat &&
    !!lastState.table.seatLeavingAfterHand[String(lastState.yourSeat)];
  const serverClosing = !!lastState?.table.closingAfterHand;
  const [leaveChoice, setLeaveChoice] = useState<boolean | null>(null);
  const [closeChoice, setCloseChoice] = useState<boolean | null>(null);
  useEffect(() => {
    setLeaveChoice((c) => (c === serverLeaving ? null : c));
  }, [serverLeaving]);
  useEffect(() => {
    setCloseChoice((c) => (c === serverClosing ? null : c));
  }, [serverClosing]);
  const chooseAfterHand =
    (
      setChoice: (on: boolean | null) => void,
      request: (on: boolean) => Promise<boolean>,
    ) =>
    (on: boolean) => {
      setChoice(on);
      void request(on).then((ok) => {
        if (!ok) setChoice(null);
      });
    };

  const isYourTurn = !!lastState && computeIsYourTurn(lastState);
  const isHost = lastState ? computeIsHost(lastState) : false;

  const validActionStrings = useMemo(
    () => computeValidActionStrings(lastState, actionLookup),
    [lastState, actionLookup],
  );

  const actionIdByString = useMemo(
    () => computeActionIdByString(actionLookup),
    [actionLookup],
  );

  // Bury/under staging: card taps only stage a local selection (shown face-up
  // in the center, tappable to put back); nothing is sent to the server until
  // the player confirms. PLAY stays a direct send.
  const { staging, stagedCards, stageCard } = useStagedActions({
    serverBury: lastState?.view.bury ?? [],
    validActionStrings,
    actionIdByString,
    takeAction,
  });

  // The last trick leaves one card and no choice: play it for the player
  // after a beat, exactly as if they had tapped it (it is their move, sent
  // from here, not an AI move). The beat starts once the previous trick has
  // been shown and collected, so the table can still be read.
  const trickOnShow = showPrev || animTrick !== null;
  // Between hands the result is on the table once the last trick has been
  // shown and collected; the player can still open the last trick, which
  // shows in the result's place without leaving the between-hands layout.
  const final = (lastState?.view.is_done && lastState.view.final) || null;
  const betweenHands =
    !!final && animTrick === null && (!showPrev || lastTrickOpen);

  // On a phone the chat is on screen between hands, so it is read there too.
  const chatUnread = useChatUnread(
    mutes.visible(chatMessages),
    clientId,
    showMobileLog || (isMobile && betweenHands),
  );
  const lastTrickPlay =
    isYourTurn &&
    !trickOnShow &&
    lastState?.view.current_trick_index === TOTAL_TRICKS - 1
      ? [...validActionStrings].filter((a) => a.startsWith("PLAY "))
      : [];
  const autoPlayId =
    lastTrickPlay.length === 1 && validActionStrings.size === 1
      ? actionIdByString[lastTrickPlay[0]]
      : undefined;
  // One card play in flight at a time: a tap on the last card and the
  // auto-play racing it would otherwise both post, and the loser is
  // refused as not_your_turn.
  const playPending = useRef(false);
  const playCard = useCallback(
    async (actionId: number) => {
      if (playPending.current) return;
      playPending.current = true;
      try {
        await takeAction(actionId);
      } finally {
        playPending.current = false;
      }
    },
    [takeAction],
  );
  useEffect(() => {
    if (autoPlayId === undefined) return;
    const timer = setTimeout(() => void playCard(autoPlayId), 400);
    return () => clearTimeout(timer);
  }, [autoPlayId, playCard]);

  const handleCardClick = (card: string) => {
    if (!isYourTurn || !card) return;
    if (stageCard(card)) return;
    if (validActionStrings.has(`PLAY ${card}`)) {
      const id = actionIdByString[`PLAY ${card}`];
      if (id !== undefined) void playCard(id);
    }
  };

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
  // Spectators (yourSeat null) watch the ring from seat 1's point of view.
  const viewSeat = yourSeat ?? 1;
  const started = playStartedFn(lastState);
  const { phase } = derivePhase(lastState);
  const yourMode = interludeMode(validActionStrings);

  // What your turn asks of you, for the turn line.
  const kind = computeKind(phase, yourMode);

  const seats = buildSeats(lastState, table, yourSeat, started);
  const yourRole = getYourRole(lastState);

  const displayCards = computeDisplayCards(showPrev, view);
  const winnerSeat = computeWinnerSeat(showPrev, view);
  const callOptions = buildCallOptions(lastState.valid_actions, actionLookup);
  const handNumber = computeHandNumber(table);
  const rulesBadge = rulesBadgeText(table.rules);
  const stakeBadge = stakeBadgeText(table.scoreMultiplier);
  const hasLastTrick = computeHasLastTrick(view);
  const handInPlay = table.status === "playing" && !view.is_done;
  const context = computeContext(lastState, phase, lastTrickOpen);

  const shownChat = mutes.visible(chatMessages);

  // Host-only Remove for another human, offered on scoreboard rows and on
  // chat authors (the only place an unseated spectator shows up).
  const removeControl = (target: string | null | undefined, name: string) =>
    isHost && target && target !== clientId ? (
      <RemovePlayerButton name={name} onRemove={() => kickPlayer(target)} />
    ) : null;
  const seatControls = (seat: number) =>
    isAiSeat(seat, table)
      ? null
      : removeControl(
          table.seatOccupants[String(seat)],
          nameForSeat(seat, table),
        );
  // Anyone can mute another player's chat for themselves; the host can
  // also remove them.
  const authorActions = (msg: ChatMessage) => {
    const author = msg.author_id;
    if (!author || author === clientId) return null;
    const name = msg.author ?? "this player";
    return (
      <>
        <button
          type="button"
          className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
          onClick={() => mutes.mute(author, name)}
        >
          Mute
        </button>
        {removeControl(author, name)}
      </>
    );
  };
  const chat = {
    messages: shownChat,
    onSendMessage: sendChatMessage,
    authorActions,
    muted: mutes.muted,
    onUnmute: mutes.unmute,
  };

  const stage = (
    <Stage
      seats={seats}
      yourSeat={viewSeat}
      phase={phase}
      yourMode={yourMode}
      trickIndex={view.current_trick_index}
      totalTricks={TOTAL_TRICKS}
      displayCards={displayCards}
      winnerSeat={winnerSeat}
      showPrev={showPrev}
      lastTrickOpen={lastTrickOpen}
      hasLastTrick={hasLastTrick}
      onTogglePrev={() => setLastTrickOpen(!lastTrickOpen)}
      callOptions={callOptions}
      selectedCall={null}
      onAction={takeAction}
      staging={staging}
      calledCardDisplay={view.called_card_display}
      yourRole={yourRole}
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
      result={betweenHands && !showPrev ? final : null}
    />
  );

  // A spectator has no hand. On desktop the stage is fixed-height and the
  // hand fills the rest of the column, so a spacer takes its place to keep
  // the SpectatorBar at the bottom; on mobile the stage absorbs the space.
  const hand =
    yourSeat === null ? (
      !isMobile && <div className={styles.spectatorSpacer} aria-hidden="true" />
    ) : (
      <PlayerHand
        hand={view.hand}
        hasUnder={!!view.under_card}
        isYourTurn={isYourTurn}
        validActionStrings={validActionStrings}
        onCardClick={handleCardClick}
        stagedCards={stagedCards}
        isMobile={isMobile}
        uiScale={uiScale}
      />
    );

  const actorSeat = lastState.actorSeat;
  // On mobile the last-trick button rides on the line under the table: the
  // status line in play, the hand-over bar between hands.
  const mobLastTrick = isMobile &&
    showLastTrick({ hasLastTrick, phase, animTrick }) && (
      <LastTrickButton
        open={lastTrickOpen}
        onToggle={() => setLastTrickOpen(!lastTrickOpen)}
      />
    );
  const status = (
    <TableStatus
      context={context}
      clock={
        turnDeadline !== null && turnSeconds !== null
          ? { deadline: turnDeadline, totalSeconds: turnSeconds }
          : null
      }
      secondsLeft={secondsLeft}
      isMobile={isMobile}
      lastTrick={mobLastTrick}
      turn={
        yourSeat === null
          ? null
          : {
              isYourTurn,
              yourLabel: TURN_LABEL[kind],
              actorName:
                actorSeat && !isYourTurn ? nameForSeat(actorSeat, table) : null,
              role: yourRole,
              isDealer: yourSeat === DEALER_SEAT,
            }
      }
    />
  );

  // Between hands, the next step takes the hand's place.
  const handOver = betweenHands && !departure && (
    <HandOverBar
      table={table}
      onNextHand={nextHand}
      onShowScores={() => setShowScores(true)}
      isHost={isHost}
      onCloseTable={closeTable}
      onLeave={() => router.push("/")}
      lastTrick={mobLastTrick}
    />
  );

  const spectatorBar = yourSeat === null && (
    <SpectatorBar
      table={table}
      actorName={nameForSeat(actorSeat, table)}
      onTakeSeat={takeSeat}
      isMobile={isMobile}
      secondsLeft={secondsLeft}
      homeOccupant={lastState.homeOccupant}
      queuedOccupant={lastState.queuedOccupant}
    />
  );

  const overlays = (
    <>
      {!departure &&
        (connectionState === "reconnecting" ||
          connectionState === "failed") && (
          <div
            role="alert"
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              zIndex: 1000,
              textAlign: "center",
              padding: "6px 12px",
              fontSize: 13,
              fontWeight: 600,
              color: "#fff",
              background: connectionState === "failed" ? "#7f1d1d" : "#92400e",
            }}
          >
            {connectionState === "failed"
              ? "Connection rejected — rejoin from the lobby"
              : "Connection lost — reconnecting…"}
          </div>
        )}
      {departure && (
        <ScoresOverlay
          table={table}
          title={
            departure === "left"
              ? `You left after hand ${table.resultsHistory.length}`
              : "The host ended the table"
          }
          lastHand={(view.is_done && view.final) || undefined}
          onClose={() => router.push("/")}
          actions={
            <>
              {departure === "left" && (
                <button
                  className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
                  onClick={() =>
                    router.push(`/?rejoin=${encodeURIComponent(table.id)}`)
                  }
                >
                  Rejoin
                </button>
              )}
              <button
                className={`${ds.btn} ${ds.btnSm}`}
                onClick={() => router.push("/")}
              >
                Lobby
              </button>
            </>
          }
        />
      )}
      {showScores && (
        <ScoresOverlay table={table} onClose={() => setShowScores(false)} />
      )}
      {showCheatSheet && (
        <CheatSheet onClose={() => setShowCheatSheet(false)} />
      )}
    </>
  );

  const leave = {
    canLeaveAfterHand: handInPlay && yourSeat !== null && !departure,
    leavingAfterHand: leaveChoice ?? serverLeaving,
    onLeave: () => router.push("/"),
    onLeaveAfterHand: chooseAfterHand(setLeaveChoice, setLeaveAfterHand),
  };
  const close = isHost
    ? {
        handInPlay,
        closingAfterHand: closeChoice ?? serverClosing,
        onCloseAfterHand: chooseAfterHand(setCloseChoice, setCloseAfterHand),
        onCloseTable: closeTable,
      }
    : null;

  const header = (
    <TableHeader
      roomName={table.name}
      rulesBadge={rulesBadge}
      stakeBadge={stakeBadge}
      handNumber={handNumber}
      isMobile={isMobile}
      leave={leave}
      close={close}
      onShowScores={() => setShowScores(true)}
      onShowCheatSheet={() => setShowCheatSheet(true)}
      onShowChat={() => setShowMobileLog(true)}
      chatUnread={chatUnread}
      onShowMenu={() => setShowMenu(true)}
    />
  );

  // ---------- Mobile ----------
  if (isMobile) {
    if (showMobileLog) {
      return (
        <div className={styles.root}>
          <MobileLogScreen
            table={table}
            yourSeat={yourSeat}
            chat={chat}
            seatControls={seatControls}
            onClose={() => setShowMobileLog(false)}
          />
          {overlays}
        </div>
      );
    }
    return (
      <div className={styles.root}>
        {header}
        {handOver ? (
          // Between hands the chat takes the hand's place, open: the time
          // between hands is when the table talks.
          <div className={`${styles.mobBody} ${styles.mobBetween}`}>
            <div
              className={`${styles.mobBetweenStage} ${yourSeat === null ? styles.mobBetweenWatching : ""}`}
            >
              {stage}
            </div>
            {final && (
              <div className={styles.mobResultLine}>
                {resultLine(final, table, yourSeat)}
              </div>
            )}
            {/* No status line: the result on the table says what it said
                of the hand in play, and the phone needs the room. */}
            <div className={styles.mobHandOver}>{handOver}</div>
            {spectatorBar}
            <div className={styles.mobChat}>
              <ChatPanel {...chat} />
            </div>
          </div>
        ) : (
          <div className={styles.mobBody}>
            {stage}
            {status}
            {hand}
            {spectatorBar}
          </div>
        )}
        {showMenu && (
          <TableMenu
            roomName={table.name}
            rulesBadge={rulesBadge}
            stakeBadge={stakeBadge}
            onShowScores={() => setShowScores(true)}
            leave={leave}
            close={close}
            onClose={() => setShowMenu(false)}
          />
        )}
        {overlays}
      </div>
    );
  }

  // ---------- Desktop ----------
  return (
    <div className={styles.root}>
      {header}
      <div className={styles.deskMain}>
        <div className={styles.deskPlay}>
          <span
            className={`${styles.connDot} ${connected ? styles.connOk : styles.connBad}`}
            role="status"
            aria-label={connected ? "Connected" : "Disconnected"}
          />
          {stage}
          {/* Between hands the result on the table says what the status
              line said of the hand in play. */}
          {handOver ? (
            <div className={styles.deskHandOver}>{handOver}</div>
          ) : (
            <>
              {status}
              {hand}
            </>
          )}
          {spectatorBar}
        </div>
        <RightRail
          table={table}
          yourSeat={yourSeat}
          chat={chat}
          seatControls={seatControls}
        />
      </div>
      {overlays}
    </div>
  );
}
