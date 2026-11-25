import { useCallback, useEffect, useState } from "react";
import { STORAGE_KEYS, sessionStore } from "../storage";

/** The unsent chat message at one table. Kept in sessionStorage, when it
 * is available, so it outlives the page: the waiting room becomes the
 * table mid-message, and a phone's between-hands chat unmounts when the
 * next hand starts. Per tab, and gone when the tab closes. */
export function useChatDraft(
  tableId: string | undefined,
): [string, (draft: string) => void] {
  const [draft, setDraft] = useState("");

  useEffect(() => {
    if (!tableId) return;
    try {
      setDraft(sessionStore.getItem(STORAGE_KEYS.chatDraft(tableId)) ?? "");
    } catch {
      setDraft("");
    }
  }, [tableId]);

  const save = useCallback(
    (next: string) => {
      setDraft(next);
      if (!tableId) return;
      try {
        const key = STORAGE_KEYS.chatDraft(tableId);
        if (next) sessionStore.setItem(key, next);
        else sessionStore.removeItem(key);
      } catch {
        // Storage unavailable: the draft still holds for this page view.
      }
    },
    [tableId],
  );

  return [draft, save];
}
