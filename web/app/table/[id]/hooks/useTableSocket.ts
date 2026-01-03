import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import type { TableStateMsg } from '../../../../lib/types';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || (() => {
  if (typeof window === 'undefined') return 'http://localhost:9000';
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  return `${protocol}//${hostname}:9000`;
})();

export interface TableSocketCallbacks {
  onTrickComplete?: (cards: string[], winner: number) => void;
  onPickerAnnounced?: (pickerName: string) => void;
  onLeaster?: () => void;
  onAlone?: (pickerName: string) => void;
  onCall?: (pickerName: string, cardDisplay: string, under: boolean) => void;
  onTableClosed?: () => void;
  onLobbyEvent?: (message: string) => void;
}

export interface ChatMessage {
  id: string;
  table_id: string;
  type: 'player' | 'system';
  author: string | null;
  body: string;
  timestamp: number;
}

export interface UseTableSocketReturn {
  connected: boolean;
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
  callbacks?: TableSocketCallbacks
): UseTableSocketReturn {
  const [connected, setConnected] = useState(false);
  const [lastState, setLastState] = useState<TableStateMsg | null>(null);
  const [actionLookup, setActionLookup] = useState<Record<string, string>>({});
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  // Fetch action lookup on mount
  useEffect(() => {
    (async () => {
      const res = await fetch(`${API_BASE}/api/actions`);
      const data = await res.json();
      setActionLookup(data.action_lookup);
    })();
  }, []);

  // WebSocket connection
  useEffect(() => {
    if (!tableId || !clientId) return;

    const api = new URL(API_BASE);
    const wsProto = api.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${api.host}/ws/table/${tableId}?client_id=${encodeURIComponent(clientId)}`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);

    socket.onmessage = (ev) => {
      const data = JSON.parse(ev.data);

      if (data.type === 'state') {
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

          const playStarted = !!(msg.state?.play_started === 1) ||
            (msg.view?.current_trick_index > 0) ||
            (msg.view?.current_trick?.some((c: string) => c !== ''));

          const getPickerName = () => {
            const seat = msg.view?.picker;
            return msg.table?.seats?.[String(seat)] || `Seat ${seat}`;
          };

          if (curPicker > 0 && prevPicker === 0 && cbs?.onPickerAnnounced) {
            cbs.onPickerAnnounced(getPickerName());
          } else if (!prevLeaster && curLeaster && cbs?.onLeaster) {
            cbs.onLeaster();
          } else if (curPicker > 0 && !playStarted && !prevAlone && curAlone && cbs?.onAlone) {
            cbs.onAlone(getPickerName());
          } else if (curPicker > 0 && !playStarted && curCalled && prevCalled !== curCalled && cbs?.onCall) {
            cbs.onCall(getPickerName(), curCalledDisplay, curCalledUnder);
          }

          return msg;
        });
      } else if (data?.type === 'table_closed') {
        callbacksRef.current?.onTableClosed?.();
        socket.close();
        window.location.href = '/';
      } else if (data?.type === 'lobby_event') {
        const message = String(data.message || '');
        if (message) {
          callbacksRef.current?.onLobbyEvent?.(message);
        }
      } else if (data?.type === 'table_update') {
        const tbl = data.table;
        if (tbl) {
          setLastState((prev: TableStateMsg | null) =>
            prev ? { ...prev, table: tbl } : prev
          );
        }
      } else if (data?.type === 'chat:init') {
        // Initialize chat with full history
        const messages = (data.messages || []) as ChatMessage[];
        setChatMessages(messages);
      } else if (data?.type === 'chat:append') {
        // Append new message to chat
        const message = data.message as ChatMessage;
        if (message) {
          setChatMessages((prev) => [...prev, message]);
        }
      }
    };

    wsRef.current = socket;
    return () => socket.close();
  }, [tableId, clientId]);

  const takeAction = useCallback(async (actionId: number) => {
    if (!clientId || !tableId) return;
    await fetch(`${API_BASE}/api/tables/${tableId}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId, action_id: actionId }),
    }).catch(() => {});
  }, [tableId, clientId]);

  const closeTable = useCallback(async () => {
    if (!tableId || !clientId) return;
    await fetch(`${API_BASE}/api/tables/${tableId}/close`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId }),
    }).catch(() => {});
  }, [tableId, clientId]);

  const redeal = useCallback(async () => {
    if (!tableId || !clientId) return;
    await fetch(`${API_BASE}/api/tables/${tableId}/redeal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId }),
    });
    await fetch(`${API_BASE}/api/tables/${tableId}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId }),
    }).catch(() => {});
  }, [tableId, clientId]);

  const sendChatMessage = useCallback((message: string) => {
    if (!wsRef.current || !message.trim()) return;
    try {
      wsRef.current.send(JSON.stringify({
        type: 'chat:send',
        message: message.trim(),
      }));
    } catch (err) {
      console.error('Failed to send chat message:', err);
    }
  }, []);

  return {
    connected,
    lastState,
    actionLookup,
    chatMessages,
    takeAction,
    closeTable,
    redeal,
    sendChatMessage,
  };
}

