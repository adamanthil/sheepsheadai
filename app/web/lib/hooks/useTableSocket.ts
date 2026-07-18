import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import type { ChatMessage, TableStateMsg, TableView } from "../types";
import { apiFetch, wsSubprotocols, wsUrl } from "../api";
import { parseWsMessage } from "../wsMessages";
import { diffCallouts } from "../callouts";

export interface TableSocketCallbacks {
  onTrickComplete?: (cards: string[], winner: number) => void;
  onPickerAnnounced?: (pickerName: string) => void;
  onLeaster?: () => void;
  onAlone?: (pickerName: string) => void;
  onCall?: (pickerName: string, cardDisplay: string, under: boolean) => void;
  onTableClosed?: () => void;
  onLobbyEvent?: (message: string) => void;
  /** Fired on table_update messages (waiting room seat/rules changes). */
  onTableUpdate?: (table: TableView, isHost?: boolean) => void;
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
        const data = parseWsMessage(ev.data);
        if (!data) return;

      if (data.type === "state") {
        const msg: TableStateMsg = data;

        setLastState((prev: TableStateMsg | null) => {
          const cbs = callbacksRef.current;
          const { trickComplete, phase } = diffCallouts(prev, msg);

          if (trickComplete) {
            cbs?.onTrickComplete?.(trickComplete.cards, trickComplete.winner);
          }

          if (phase?.kind === "picker") {
            cbs?.onPickerAnnounced?.(phase.pickerName);
          } else if (phase?.kind === "leaster") {
            cbs?.onLeaster?.();
          } else if (phase?.kind === "alone") {
            cbs?.onAlone?.(phase.pickerName);
          } else if (phase?.kind === "call") {
            cbs?.onCall?.(phase.pickerName, phase.cardDisplay, phase.under);
          }

          return msg;
        });
      } else if (data.type === "table_closed") {
        callbacksRef.current?.onTableClosed?.();
        socket.close();
        window.location.href = "/";
      } else if (data.type === "lobby_event") {
        if (data.message) {
          callbacksRef.current?.onLobbyEvent?.(data.message);
        }
      } else if (data.type === "table_update") {
        const tbl = data.table;
        callbacksRef.current?.onTableUpdate?.(tbl, data.isHost);
        setLastState((prev: TableStateMsg | null) =>
          prev ? { ...prev, table: tbl } : prev,
        );
      } else if (data.type === "chat:init") {
        // Initialize chat with full history
        setChatMessages(data.messages);
      } else if (data.type === "chat:append") {
        setChatMessages((prev) => [...prev, data.message]);
      } else if (data.type === "server_restart") {
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
