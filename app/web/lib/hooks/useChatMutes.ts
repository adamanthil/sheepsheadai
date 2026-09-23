import { useCallback, useEffect, useState } from "react";
import type { ChatMessage } from "../types";
import { STORAGE_KEYS } from "../storage";

export interface ChatMutes {
  /** Muted authors at this table: client id -> display name. */
  muted: Record<string, string>;
  mute: (clientId: string, name: string) => void;
  unmute: (clientId: string) => void;
  /** ``messages`` without the muted authors' player messages. */
  visible: (messages: ChatMessage[]) => ChatMessage[];
}

/** Chat mutes for one viewer at one table. Client-side only: muting hides
 * someone's messages from you, nobody else. Remembered per table in
 * localStorage when it is available. */
export function useChatMutes(tableId: string | undefined): ChatMutes {
  const [muted, setMuted] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!tableId) return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEYS.chatMutes(tableId));
      setMuted(raw ? (JSON.parse(raw) as Record<string, string>) : {});
    } catch {
      setMuted({});
    }
  }, [tableId]);

  const save = useCallback(
    (next: Record<string, string>) => {
      setMuted(next);
      if (!tableId) return;
      try {
        window.localStorage.setItem(
          STORAGE_KEYS.chatMutes(tableId),
          JSON.stringify(next),
        );
      } catch {
        // Storage unavailable: the mute still holds for this page view.
      }
    },
    [tableId],
  );

  const mute = useCallback(
    (clientId: string, name: string) => save({ ...muted, [clientId]: name }),
    [muted, save],
  );
  const unmute = useCallback(
    (clientId: string) => {
      const next = { ...muted };
      delete next[clientId];
      save(next);
    },
    [muted, save],
  );
  const visible = useCallback(
    (messages: ChatMessage[]) =>
      messages.filter((m) => !(m.author_id && m.author_id in muted)),
    [muted],
  );

  return { muted, mute, unmute, visible };
}
