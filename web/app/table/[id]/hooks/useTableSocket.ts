import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import type { TableStateMsg } from "../../../../lib/types";
import { apiFetch, wsSubprotocols, wsUrl } from "../../../../lib/api";

export interface TableSocketCallbacks {
  onTrickComplete?: (cards: string[], winner: number) => void;
  onPickerAnnounced?: (pickerName: string) => void;
  onLeaster?: () => void;
  onAlone?: (pickerName: string) => void;
  onCall?: (pickerName: string, cardDisplay: string, under: boolean) => void;
  onTableClosed?: () => void;
  onLobbyEvent?: (message: string) => void;
  /** A REST action or the socket failed in a way the user should see. */
  onError?: (message: string) => void;
}

export type ConnectionState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "failed";

/** Close codes the server uses for auth/authorization failures — retrying
 * cannot help, so the client must not hammer the server. */
const TERMINAL_WS_CODES = new Set([4401, 4403, 4404, 4429]);

export interface ChatMessage {
  id: string;
  table_id: string;
  type: "player" | "system";
  author: string | null;
  body: string;
  timestamp: number;
}

export interface UseTableSocketReturn {
  connected: boolean;
  connectionState: ConnectionState;
  lastState: TableStateMsg | null;
  actionLookup: Record<string, string>;
  chatMessages: ChatMessage[];
  takeAction: (actionId: number) => Promise<void>;
  closeTable: () => Promise<void>;
  redeal: () => Promise<void>;
  sendChatMessage: (message: string) => void;
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
  const [actionLookup, setActionLookup] = useState<Record<string, string>>({});
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

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
        if (wsRef.current === socket) wsRef.current = null;
        if (disposed) return;
        if (TERMINAL_WS_CODES.has(e.code)) {
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

      socket.onmessage = (ev) => {
        const data = JSON.parse(ev.data);

      if (data.type === "state") {
        const msg = data as TableStateMsg;

        setLastState((prev: TableStateMsg | null) => {
          const cbs = callbacksRef.current;

          // Detect trick completion
          const prevWas = prev?.view?.was_trick_just_completed;
          const curWas = msg.view?.was_trick_just_completed;
          if (curWas && !prevWas && cbs?.onTrickComplete) {
            const last = msg.view?.last_trick as string[] | null;
            const win = msg.view?.last_trick_winner as number;
            if (last && win) {
              cbs.onTrickComplete(last, win);
            }
          }

          // Detect game phase callouts
          const prevPicker = prev?.view?.picker || 0;
          const curPicker = msg.view?.picker || 0;
          const prevLeaster = !!prev?.view?.is_leaster;
          const curLeaster = !!msg.view?.is_leaster;
          const prevCalled = prev?.view?.called_card || null;
          const curCalled = msg.view?.called_card || null;
          const curCalledDisplay = msg.view?.called_card_display || curCalled;
          const curCalledUnder = !!msg.view?.called_under;
          const prevAlone = !!prev?.view?.alone;
          const curAlone = !!msg.view?.alone;

          const playStarted =
            !!(msg.state?.play_started === 1) ||
            msg.view?.current_trick_index > 0 ||
            msg.view?.current_trick?.some((c: string) => c !== "");

          const getPickerName = () => {
            const seat = msg.view?.picker;
            return msg.table?.seats?.[String(seat)] || `Seat ${seat}`;
          };

          if (curPicker > 0 && prevPicker === 0 && cbs?.onPickerAnnounced) {
            cbs.onPickerAnnounced(getPickerName());
          } else if (!prevLeaster && curLeaster && cbs?.onLeaster) {
            cbs.onLeaster();
          } else if (
            curPicker > 0 &&
            !playStarted &&
            !prevAlone &&
            curAlone &&
            cbs?.onAlone
          ) {
            cbs.onAlone(getPickerName());
          } else if (
            curPicker > 0 &&
            !playStarted &&
            curCalled &&
            prevCalled !== curCalled &&
            cbs?.onCall
          ) {
            cbs.onCall(
              getPickerName(),
              curCalledDisplay ?? curCalled ?? "",
              curCalledUnder,
            );
          }

          return msg;
        });
      } else if (data?.type === "table_closed") {
        callbacksRef.current?.onTableClosed?.();
        socket.close();
        window.location.href = "/";
      } else if (data?.type === "lobby_event") {
        const message = String(data.message || "");
        if (message) {
          callbacksRef.current?.onLobbyEvent?.(message);
        }
      } else if (data?.type === "table_update") {
        const tbl = data.table;
        if (tbl) {
          setLastState((prev: TableStateMsg | null) =>
            prev ? { ...prev, table: tbl } : prev,
          );
        }
      } else if (data?.type === "chat:init") {
        // Initialize chat with full history
        const messages = (data.messages || []) as ChatMessage[];
        setChatMessages(messages);
      } else if (data?.type === "chat:append") {
        // Append new message to chat
        const message = data.message as ChatMessage;
        if (message) {
          setChatMessages((prev) => [...prev, message]);
        }
      } else if (data?.type === "server_restart") {
        // Deploy in progress: the socket is about to drop; the reconnect
        // loop keeps retrying until the server is back.
        callbacksRef.current?.onError?.("Server restarting…");
      }
      };
    };

    connect();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      try {
        wsRef.current?.close();
      } catch {}
      wsRef.current = null;
    };
  }, [tableId, clientId]);

  const takeAction = useCallback(
    async (actionId: number) => {
      if (!clientId || !tableId) return;
      try {
        const res = await apiFetch(`/api/tables/${tableId}/action`, {
          method: "POST",
          body: JSON.stringify({ client_id: clientId, action_id: actionId }),
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          callbacksRef.current?.onError?.(
            `Action failed: ${j?.detail || res.status}`,
          );
        }
      } catch (err) {
        console.warn("action POST failed", err);
        callbacksRef.current?.onError?.("Action failed: network error");
      }
    },
    [tableId, clientId],
  );

  const closeTable = useCallback(async () => {
    if (!tableId || !clientId) return;
    try {
      await apiFetch(`/api/tables/${tableId}/close`, {
        method: "POST",
        body: JSON.stringify({ client_id: clientId }),
      });
    } catch (err) {
      console.warn("close POST failed", err);
      callbacksRef.current?.onError?.("Close failed: network error");
    }
  }, [tableId, clientId]);

  const redeal = useCallback(async () => {
    if (!tableId || !clientId) return;
    try {
      await apiFetch(`/api/tables/${tableId}/redeal`, {
        method: "POST",
        body: JSON.stringify({ client_id: clientId }),
      });
      await apiFetch(`/api/tables/${tableId}/start`, {
        method: "POST",
        body: JSON.stringify({ client_id: clientId }),
      });
    } catch (err) {
      console.warn("redeal POST failed", err);
      callbacksRef.current?.onError?.("Redeal failed: network error");
    }
  }, [tableId, clientId]);

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

  return {
    connected,
    connectionState,
    lastState,
    actionLookup,
    chatMessages,
    takeAction,
    closeTable,
    redeal,
    sendChatMessage,
  };
}
