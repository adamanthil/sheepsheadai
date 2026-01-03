"use client";

import { useMemo, useState, useCallback } from 'react';
import { useSearchParams, useParams } from 'next/navigation';

import styles from './page.module.css';
import { nameForSeat } from './utils';
import { useTableSocket, useResponsive, useTrickAnimation, useCallout } from './hooks';
import {
  TrickArea,
  PlayerHand,
  BottomActionBar,
  GameOverBanner,
  ScoresOverlay,
} from './components';
import { ChatPanel } from '../../components/chat';

type PlayerStatus = 'PASS' | 'PICK' | 'PICKER' | 'PENDING' | 'PARTNER' | null;

export default function TablePage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const clientId = searchParams.get('client_id') || '';

  // Custom hooks for grouped state management
  const { showPrev, animTrick, triggerCollect, setShowPrev } = useTrickAnimation();
  const { callout, showCallout } = useCallout();

  const { connected, lastState, actionLookup, chatMessages, takeAction, closeTable, redeal, sendChatMessage } = useTableSocket(
    params?.id,
    clientId,
    {
      onTrickComplete: triggerCollect,
      onPickerAnnounced: (name) => showCallout('PICK', `${name} picked`),
      onLeaster: () => showCallout('LEASTER', 'All passed · Leaster'),
      onAlone: (name) => showCallout('ALONE', `${name} goes alone`),
      onCall: (name, cardDisplay, under) =>
        showCallout('CALL', `${name} calls ${cardDisplay}${under ? ' under' : ''}`),
      onTableClosed: () => showCallout('PICK', 'Table closed', 1200),
      onLobbyEvent: (msg) => showCallout('PICK', msg),
    }
  );

  // Get hand card count for responsive sizing
  const cardCount = lastState?.view?.hand?.length || 6;
  const {
    isMobile,
    handSize,
    centerSize,
    handTopMargin,
    handRowRef,
    trickBoxRef,
  } = useResponsive(cardCount);

  // Local UI state (minimal)
  const [showScores, setShowScores] = useState(false);
  const [confirmClose, setConfirmClose] = useState(false);

  // Derived state
  const isYourTurn = useMemo(
    () => lastState?.actorSeat === lastState?.yourSeat,
    [lastState?.actorSeat, lastState?.yourSeat]
  );

  const isHost = useMemo(() => {
    if (!lastState || !clientId) return false;
    const hostId = lastState.table?.hostId || lastState.table?.host_id;
    return !!hostId && String(hostId) === String(clientId);
  }, [lastState, clientId]);

  const inPickDecision = useMemo(
    () => !!lastState && !lastState.view.is_leaster && lastState.view.picker === 0,
    [lastState]
  );

  const playStarted = useMemo(() => {
    if (!lastState) return false;
    if (lastState.state?.play_started === 1) return true;
    if (lastState.view.current_trick_index > 0) return true;
    const picker = lastState.view.picker || 0;
    if (picker > 0) {
      const ct = lastState.view.current_trick as string[] | undefined;
      if (ct && ct.some((c: string) => c !== '')) return true;
    }
    return false;
  }, [lastState]);

  // Build set of valid action strings for card clicking
  const validActionStrings = useMemo(() => {
    if (!lastState) return new Set<string>();
    const s = new Set<string>();
    for (const id of lastState.valid_actions) {
      const label = actionLookup[String(id)];
      if (label) s.add(label);
    }
    return s;
  }, [lastState, actionLookup]);

  // Map action label to ID
  const actionIdByString = useMemo(() => {
    const m: Record<string, number> = {};
    Object.entries(actionLookup).forEach(([id, label]) => {
      const n = Number(id);
      if (Number.isFinite(n)) m[label] = n;
    });
    return m;
  }, [actionLookup]);

  // Get displayed status for a seat
  const getPlayerStatus = useCallback(
    (absSeat: number): PlayerStatus => {
      if (!lastState) return null;

      const picker = lastState.view.picker || 0;
      const partner = lastState.view.partner || 0;
      const actorSeat = lastState.actorSeat;

      if (playStarted) {
        if (absSeat === picker) return 'PICKER';
        if (absSeat === partner) return 'PARTNER';
        return null;
      }

      if (picker > 0) {
        if (absSeat === picker) return 'PICK';
        if (absSeat < picker) return 'PASS';
        return null;
      }

      if (!inPickDecision) return null;
      if (actorSeat && absSeat === actorSeat) return 'PENDING';
      if (actorSeat && absSeat < actorSeat) return 'PASS';
      return null;
    },
    [lastState, playStarted, inPickDecision]
  );

  // Handle card click (play/bury/under)
  const handleCardClick = useCallback(
    (card: string) => {
      if (!isYourTurn || !card) return;
      const candidates = [`PLAY ${card}`, `BURY ${card}`, `UNDER ${card}`];
      for (const lbl of candidates) {
        if (validActionStrings.has(lbl)) {
          const id = actionIdByString[lbl];
          if (id !== undefined) {
            void takeAction(id);
            return;
          }
        }
      }
    },
    [isYourTurn, validActionStrings, actionIdByString, takeAction]
  );


  return (
    <div className={styles.root}>
      <div className={styles.wrap}>
        <div className={styles.topRow}>
          <div className={styles.connectionStatus}>
            Connection: {connected ? 'connected' : 'disconnected'}
          </div>
        </div>

        {!lastState ? (
          <div className={styles.waitingMessage}>Waiting for state…</div>
        ) : (
          <div className={styles.tableArea}>
            <div className={styles.tableFrame}>
              <TrickArea
                cards={lastState.view.current_trick}
                yourSeat={lastState.yourSeat}
                table={lastState.table}
                showPrev={showPrev}
                lastTrick={lastState.view.last_trick}
                lastTrickWinner={lastState.view.last_trick_winner}
                lastTrickPoints={lastState.view.last_trick_points}
                animTrick={animTrick}
                callout={callout}
                centerSize={centerSize}
                trickBoxRef={trickBoxRef}
                getPlayerStatus={getPlayerStatus}
              />

              <PlayerHand
                hand={lastState.view.hand}
                handSize={handSize}
                handTopMargin={handTopMargin}
                handRowRef={handRowRef}
                isYourTurn={isYourTurn}
                validActionStrings={validActionStrings}
                onCardClick={handleCardClick}
                userStatus={getPlayerStatus(lastState.yourSeat)}
              />

              {lastState.view.is_done && lastState.view.final && (
                <GameOverBanner
                  final={lastState.view.final}
                  table={lastState.table}
                  onRedeal={redeal}
                  onShowScores={() => setShowScores(true)}
                />
              )}
            </div>

            <div className={styles.chatSection}>
              <ChatPanel messages={chatMessages} onSendMessage={sendChatMessage} />
            </div>
          </div>
        )}
      </div>

      {lastState && (
        <BottomActionBar
          yourSeat={lastState.yourSeat}
          actorSeat={lastState.actorSeat}
          table={lastState.table}
          isMobile={isMobile}
          isYourTurn={isYourTurn}
          validActions={lastState.valid_actions}
          actionLookup={actionLookup}
          showPrev={showPrev}
          hasLastTrick={!!(lastState.view.last_trick?.length === 5)}
          isHost={isHost}
          confirmClose={confirmClose}
          onTogglePrev={() => setShowPrev(!showPrev)}
          onShowScores={() => setShowScores(true)}
          onCloseTable={closeTable}
          onConfirmClose={setConfirmClose}
          onTakeAction={takeAction}
        />
      )}

      {lastState && showScores && (
        <ScoresOverlay onClose={() => setShowScores(false)} table={lastState.table} />
      )}
    </div>
  );
}
