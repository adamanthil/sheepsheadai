import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { ChatMessage, TableStateMsg, TableView } from "../types";
import { apiFetch, wsSubprotocols, wsUrl } from "../api";
import { STORAGE_KEYS } from "../storage";
import { parseWsMessage } from "../wsMessages";
import { diffCallouts } from "../callouts";
import { applyPendingMove, isPredicted, type PendingMove } from "../optimistic";
import { applyTyping, dropStale, type Typist } from "../typing";

export interface TableSocketCallbacks {
  /** A trick was taken. ``shownForMs`` is how long it has already been on
   * the table complete: the player's own last card was drawn before the
   * server answered. 0 when the server's state is the first to show it. */
  onTrickComplete?: (
    cards: string[],
    winner: number,
    shownForMs: number,
  ) => void;
  onPickerAnnounced?: (pickerName: string) => void;
  onLeaster?: () => void;
  onDoubler?: (multiplier: number, capped: boolean) => void;
  onAlone?: (pickerName: string) => void;
  onCall?: (pickerName: string, cardDisplay: string, under: boolean) => void;
  onTableClosed?: () => void;
  onLobbyEvent?: (message: string) => void;
  /** Fired on table_update messages (waiting room seat/rules changes). */
  onTableUpdate?: (table: TableView, isHost?: boolean) => void;
  /** A REST action or the socket failed in a way the user should see. */
  onError?: (message: string) => void;
}

/** How the player came to be off the table while still on its page: they
 * left once the hand ended, or the host closed the table after it. */
export type Departure = "left" | "closed";

export type ConnectionState =
  "connecting" | "connected" | "reconnecting" | "failed";

/** Close codes the server uses for auth/authorization failures — retrying
 * cannot help, so the client must not hammer the server. */
const TERMINAL_WS_CODES = new Set([4401, 4403, 4404, 4410, 4429]);

/** Close code for a table the server no longer has (closed or expired). */
const TABLE_GONE_WS_CODE = 4404;

export interface UseTableSocketReturn {
  connected: boolean;
  connectionState: ConnectionState;
  /** Set once the player is off the table but kept on its final scores. */
  departure: Departure | null;
  /** The table as drawn: the server's last state, with this player's own
   * move laid over it from the moment it is sent until the server shows it
   * (or refuses it). */
  lastState: TableStateMsg | null;
  /** Local-clock ms when the acting human's turn runs out, or null. */
  turnDeadline: number | null;
  /** Full length in seconds of the running turn clock, or null. */
  turnSeconds: number | null;
  actionLookup: Record<string, string>;
  chatMessages: ChatMessage[];
  /** The others typing in the chat now, in the order they started. */
  typists: Typist[];
  /** POST one action, shown at once where it can be predicted. Resolves
   * true when the server accepted it. */
  takeAction: (actionId: number) => Promise<boolean>;
  // The table requests below resolve true when the server accepted them;
  // a refusal or network error is reported through onError and resolves
  // false.
  closeTable: () => Promise<boolean>;
  /** Host only: close the table once the hand in play ends, or not. */
  setCloseAfterHand: (on: boolean) => Promise<boolean>;
  /** Leave the table once the hand in play ends, or not. */
  setLeaveAfterHand: (on: boolean) => Promise<boolean>;
  /** Leave the table now: the AI takes the seat at once. */
  leaveTable: () => Promise<boolean>;
  nextHand: () => Promise<boolean>;
  /** Spectator only: take over the AI at ``seat``. */
  takeSeat: (seat: number) => Promise<boolean>;
  /** Host only: remove a player (by client id) from the table. */
  kickPlayer: (targetClientId: string) => Promise<boolean>;
  sendChatMessage: (message: string) => void;
  /** Tell the others this player is typing in the chat, or stopped.
   * Returns whether it went out (the socket may be down). */
  sendTyping: (typing: boolean) => boolean;
}

export function useTableSocket(
  tableId: string | undefined,
  clientId: string,
  callbacks?: TableSocketCallbacks,
): UseTableSocketReturn {
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const connected = connectionState === "connected";
  const [lastState, setLastState] = useState<TableStateMsg | null>(null);
  const [departure, setDeparture] = useState<Departure | null>(null);
  // Kept on the local clock: the server sends seconds left, not a
  // timestamp, so a skewed device clock can't distort the countdown.
  const [turnDeadline, setTurnDeadline] = useState<number | null>(null);
  const [turnSeconds, setTurnSeconds] = useState<number | null>(null);
  const [actionLookup, setActionLookup] = useState<Record<string, string>>({});
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [typists, setTypists] = useState<Typist[]>([]);
  const [pending, setPending] = useState<PendingMove | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;
  // Read by takeAction when a move is sent, to predict it from the state
  // and labels on screen at that moment.
  const lastStateRef = useRef(lastState);
  lastStateRef.current = lastState;
  const actionLookupRef = useRef(actionLookup);
  actionLookupRef.current = actionLookup;
  // When the player's own predicted card completed the trick on screen: the
  // move it was predicted from, and when. Matched against the state that
  // confirms that move (the next moveSeq), so a stale one never applies.
  const completedTrickRef = useRef<{ baseSeq: number; at: number } | null>(
    null,
  );

  // Fetch action lookup on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await apiFetch("/api/actions");
        const data = await res.json();
        setActionLookup(data.action_lookup);
      } catch (err) {
        console.warn("action lookup fetch failed", err);
        callbacksRef.current?.onError?.("Backend unreachable");
      }
    })();
  }, []);

  // WebSocket connection with automatic reconnect. The server sends the full
  // table state on every connect, so a successful reconnect resyncs for free.
  useEffect(() => {
    if (!tableId || !clientId) return;

    let disposed = false;
    // Set on a terminal close (kicked, refused): no reconnect of any kind.
    let gaveUp = false;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (disposed) return;
      // client_id and session token are carried in the Sec-WebSocket-Protocol
      // header so they do not appear in URL access logs.
      const socket = new WebSocket(wsUrl(tableId), wsSubprotocols(clientId));
      wsRef.current = socket;

      socket.onopen = () => {
        attempt = 0;
        setConnectionState("connected");
      };
      socket.onclose = (e) => {
        // A socket reconnectNow replaced while it was still closing: the
        // new one owns the connection, so this close schedules nothing.
        if (wsRef.current !== socket) return;
        wsRef.current = null;
        // Nobody's repeats reach a closed socket; the reconnect starts
        // from nobody typing rather than letting them go stale.
        setTypists([]);
        if (disposed) return;
        if (e.code === TABLE_GONE_WS_CODE) {
          // The table closed while we were away: forget it so the lobby
          // doesn't offer to resume it, and leave its dead page.
          window.localStorage.removeItem(STORAGE_KEYS.clientId(tableId));
          window.location.replace("/");
          return;
        }
        if (TERMINAL_WS_CODES.has(e.code)) {
          gaveUp = true;
          setConnectionState("failed");
          return;
        }
        setConnectionState("reconnecting");
        // Exponential backoff, 0.5s → 8s, with jitter.
        const delay =
          Math.min(8000, 500 * 2 ** attempt) * (0.5 + Math.random() * 0.5);
        attempt += 1;
        timer = setTimeout(connect, delay);
      };

      // Off the table for good, but kept on its final scores: no reconnect,
      // and a rejoin is a fresh join (the server forgot this connection).
      const departed = (how: Departure) => {
        disposed = true;
        window.localStorage.removeItem(STORAGE_KEYS.clientId(tableId));
        socket.close();
        setDeparture(how);
      };

      socket.onmessage = (ev) => {
        const data = parseWsMessage(ev.data);
        if (!data) return;

        if (data.type === "state") {
          const msg: TableStateMsg = data;
          setTurnDeadline(
            msg.turnSecondsLeft != null
              ? Date.now() + msg.turnSecondsLeft * 1000
              : null,
          );
          setTurnSeconds(msg.turnSeconds ?? null);

          setLastState((prev: TableStateMsg | null) => {
            const cbs = callbacksRef.current;
            const { trickComplete, phase } = diffCallouts(prev, msg);

            if (trickComplete) {
              const shown = completedTrickRef.current;
              const shownForMs =
                shown && msg.moveSeq === shown.baseSeq + 1
                  ? performance.now() - shown.at
                  : 0;
              cbs?.onTrickComplete?.(
                trickComplete.cards,
                trickComplete.winner,
                shownForMs,
              );
            }

            if (phase?.kind === "picker") {
              cbs?.onPickerAnnounced?.(phase.pickerName);
            } else if (phase?.kind === "leaster") {
              cbs?.onLeaster?.();
            } else if (phase?.kind === "doubler") {
              cbs?.onDoubler?.(phase.multiplier, phase.capped);
            } else if (phase?.kind === "alone") {
              cbs?.onAlone?.(phase.pickerName);
            } else if (phase?.kind === "call") {
              cbs?.onCall?.(phase.pickerName, phase.cardDisplay, phase.under);
            }

            return msg;
          });
          // A state from any other move than the one we predicted from
          // replaces the prediction: it holds our move, or a refusal or a
          // timeout move is on its way.
          setPending((p) => (p && msg.moveSeq !== p.baseSeq ? null : p));
        } else if (data.type === "table_closed") {
          // Ended at the host's word after a hand: stay on the final scores.
          if (data.reason === "host_closed_after_hand") {
            departed("closed");
            return;
          }
          callbacksRef.current?.onTableClosed?.();
          socket.close();
          // A table the server closed for sitting idle says so on arrival.
          const idle =
            data.reason === "lobby_expired" || data.reason === "hand_idle";
          window.location.href = idle ? "/?notice=idle" : "/";
        } else if (data.type === "lobby_event") {
          if (data.message) {
            callbacksRef.current?.onLobbyEvent?.(data.message);
          }
        } else if (data.type === "table_update") {
          const tbl = data.table;
          callbacksRef.current?.onTableUpdate?.(tbl, data.isHost);
          // isHost rides along so a host handoff shows up without waiting
          // for the next state message.
          setLastState((prev: TableStateMsg | null) =>
            prev
              ? { ...prev, table: tbl, isHost: data.isHost ?? prev.isHost }
              : prev,
          );
        } else if (data.type === "chat:init") {
          // Initialize chat with full history
          setChatMessages(data.messages);
        } else if (data.type === "chat:append") {
          setChatMessages((prev) => [...prev, data.message]);
          // A typist whose message arrives has stopped typing it.
          const author = data.message.author_id;
          if (author) {
            setTypists((prev) =>
              applyTyping(prev, author, "", false, Date.now()),
            );
          }
        } else if (data.type === "chat:typing") {
          setTypists((prev) =>
            applyTyping(
              prev,
              data.client_id,
              data.name,
              data.typing,
              Date.now(),
            ),
          );
        } else if (data.type === "kicked") {
          // Forget this table's connection so the lobby's Join doesn't try
          // to resume it, and say why on the home page.
          window.localStorage.removeItem(STORAGE_KEYS.clientId(tableId));
          socket.close();
          window.location.href = "/?notice=removed";
        } else if (data.type === "left") {
          // This player left (from this tab or another): every tab goes
          // home, and a rejoin is a fresh join.
          disposed = true;
          window.localStorage.removeItem(STORAGE_KEYS.clientId(tableId));
          socket.close();
          window.location.replace("/");
        } else if (data.type === "left_after_hand") {
          // Carries the table with the hand just finished, which this
          // player is gone before the broadcast of.
          const tbl = data.table;
          setLastState((prev: TableStateMsg | null) =>
            prev ? { ...prev, table: tbl } : prev,
          );
          departed("left");
        } else if (data.type === "turn_timer") {
          setTurnDeadline(Date.now() + data.secondsLeft * 1000);
          setTurnSeconds(data.turnSeconds ?? data.secondsLeft);
        } else if (data.type === "server_restart") {
          // Deploy in progress: the socket is about to drop; the reconnect
          // loop keeps retrying until the server is back.
          callbacksRef.current?.onError?.("Server restarting…");
        }
      };
    };

    // Backgrounding a phone drops the socket, and the backoff may have
    // grown while the page was frozen. Coming back to the page (or back
    // online), reconnect at once rather than waiting out the backoff, so
    // the seat is reclaimed before the player can act on stale state.
    const reconnectNow = () => {
      if (disposed || gaveUp || document.visibilityState !== "visible") return;
      const ws = wsRef.current;
      if (
        ws &&
        (ws.readyState === WebSocket.OPEN ||
          ws.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }
      if (timer) clearTimeout(timer);
      timer = null;
      attempt = 0;
      connect();
    };
    document.addEventListener("visibilitychange", reconnectNow);
    window.addEventListener("online", reconnectNow);
    window.addEventListener("pageshow", reconnectNow);

    connect();
    return () => {
      disposed = true;
      document.removeEventListener("visibilitychange", reconnectNow);
      window.removeEventListener("online", reconnectNow);
      window.removeEventListener("pageshow", reconnectNow);
      if (timer) clearTimeout(timer);
      try {
        wsRef.current?.close();
      } catch {}
      wsRef.current = null;
    };
  }, [tableId, clientId]);

  const takeAction = useCallback(
    async (actionId: number) => {
      if (!clientId || !tableId) return false;
      const base = lastStateRef.current;
      const label = actionLookupRef.current[String(actionId)];
      const move: PendingMove | null =
        base?.moveSeq !== undefined && label && isPredicted(label)
          ? { label, baseSeq: base.moveSeq }
          : null;
      if (move) setPending(move);
      const completesTrick =
        base &&
        move &&
        label.startsWith("PLAY ") &&
        applyPendingMove(base, label).view.current_trick.every((c) => c !== "");
      completedTrickRef.current = completesTrick
        ? { baseSeq: move.baseSeq, at: performance.now() }
        : null;
      // Refused or lost: back to what the server last showed. An accepted
      // move stays drawn until the state holding it arrives.
      const withdraw = () => {
        if (move) setPending((p) => (p === move ? null : p));
        completedTrickRef.current = null;
      };
      try {
        const res = await apiFetch(`/api/tables/${tableId}/action`, {
          method: "POST",
          body: JSON.stringify({ client_id: clientId, action_id: actionId }),
        });
        if (!res.ok) {
          withdraw();
          const j = await res.json().catch(() => ({}));
          callbacksRef.current?.onError?.(
            `Action failed: ${j?.detail || res.status}`,
          );
          return false;
        }
        return true;
      } catch (err) {
        withdraw();
        console.warn("action POST failed", err);
        callbacksRef.current?.onError?.("Action failed: network error");
        return false;
      }
    },
    [tableId, clientId],
  );

  /** POST a table request; false (with the reason shown) when it failed.
   * ``what`` names it in the error, e.g. "Close failed: not_host". */
  const post = useCallback(
    async (path: string, body: object, what: string): Promise<boolean> => {
      if (!tableId || !clientId) return false;
      try {
        const res = await apiFetch(`/api/tables/${tableId}/${path}`, {
          method: "POST",
          body: JSON.stringify({ client_id: clientId, ...body }),
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          callbacksRef.current?.onError?.(
            `${what}: ${j?.detail || res.status}`,
          );
          return false;
        }
        return true;
      } catch (err) {
        console.warn(`${path} POST failed`, err);
        callbacksRef.current?.onError?.(`${what}: network error`);
        return false;
      }
    },
    [tableId, clientId],
  );

  const closeTable = useCallback(
    () => post("close", {}, "Close failed"),
    [post],
  );

  /** Leave now: the AI takes the seat at once, instead of holding it for
   * a return as it does when the connection merely drops. */
  const leaveTable = useCallback(async () => {
    const ok = await post("leave", {}, "Leave failed");
    if (ok && tableId) {
      window.localStorage.removeItem(STORAGE_KEYS.clientId(tableId));
    }
    return ok;
  }, [post, tableId]);

  const setLeaveAfterHand = useCallback(
    (on: boolean) => post("leave_after_hand", { on }, "Leave failed"),
    [post],
  );

  const setCloseAfterHand = useCallback(
    (on: boolean) => post("close_after_hand", { on }, "Close failed"),
    [post],
  );

  const nextHand = useCallback(
    () => post("next-hand", {}, "Next hand failed"),
    [post],
  );

  const takeSeat = useCallback(
    (seat: number) => post("seat", { seat }, "Couldn't take seat"),
    [post],
  );

  const kickPlayer = useCallback(
    (targetClientId: string) =>
      post(
        "kick",
        { target_client_id: targetClientId },
        "Couldn't remove player",
      ),
    [post],
  );

  // Drop a typist when their repeats stop arriving.
  useEffect(() => {
    if (typists.length === 0) return;
    const next = Math.min(...typists.map((t) => t.until));
    const timer = setTimeout(
      () => setTypists((prev) => dropStale(prev, Date.now())),
      Math.max(0, next - Date.now()) + 50,
    );
    return () => clearTimeout(timer);
  }, [typists]);

  const sendTyping = useCallback((typing: boolean) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    try {
      ws.send(JSON.stringify({ type: "chat:typing", typing }));
      return true;
    } catch (err) {
      console.warn("Failed to send typing:", err);
      return false;
    }
  }, []);

  const sendChatMessage = useCallback((message: string) => {
    if (!wsRef.current || !message.trim()) return;
    try {
      wsRef.current.send(
        JSON.stringify({
          type: "chat:send",
          message: message.trim(),
        }),
      );
    } catch (err) {
      console.error("Failed to send chat message:", err);
    }
  }, []);

  // The prediction is drawn only over the state it was made from.
  const predicting = pending !== null && lastState?.moveSeq === pending.baseSeq;
  const shownState = useMemo(
    () =>
      predicting && lastState && pending
        ? applyPendingMove(lastState, pending.label)
        : lastState,
    [predicting, lastState, pending],
  );

  return {
    connected,
    connectionState,
    departure,
    lastState: shownState,
    // The clock stops with the move made.
    turnDeadline: predicting ? null : turnDeadline,
    turnSeconds,
    actionLookup,
    chatMessages,
    takeAction,
    closeTable,
    setCloseAfterHand,
    setLeaveAfterHand,
    leaveTable,
    nextHand,
    takeSeat,
    kickPlayer,
    sendChatMessage,
    sendTyping,
    typists,
  };
}
