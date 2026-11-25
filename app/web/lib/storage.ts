export const STORAGE_KEYS = {
  clientId: (tableId: string) => `sheepshead_client_id_${tableId}`,
  playerId: "sheepshead_player_id",
  displayName: "sheepshead_display_name",
  sessionToken: "sheepshead_session_token",
  chatMutes: (tableId: string) => `sheepshead_chat_mutes_${tableId}`,
  chatDraft: (tableId: string) => `sheepshead_chat_draft_${tableId}`,
} as const;

/** The part of Storage the app uses, plus the keys it holds. */
export interface Store {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  keys(): string[];
}

/**
 * Web storage, or memory when the browser has none: some in-app browsers
 * (WebViews with DOM storage off) give `null`, and blocked storage throws
 * a SecurityError on access. Memory keeps the page working for the visit,
 * as a fresh guest on every load. Which one is decided at first use. On
 * the server, reads are null and writes do nothing.
 */
function browserStore(kind: "localStorage" | "sessionStorage"): Store {
  let backing: Storage | Map<string, string> | undefined;
  const resolve = (): Storage | Map<string, string> | null => {
    if (typeof window === "undefined") return null;
    if (backing === undefined) {
      let storage: Storage | null = null;
      try {
        storage = window[kind];
      } catch (err) {
        if (!(err instanceof DOMException)) throw err;
      }
      backing = storage ?? new Map();
    }
    return backing;
  };
  return {
    getItem(key) {
      const b = resolve();
      if (b instanceof Map) return b.get(key) ?? null;
      return b ? b.getItem(key) : null;
    },
    setItem(key, value) {
      const b = resolve();
      if (b instanceof Map) b.set(key, value);
      else b?.setItem(key, value);
    },
    removeItem(key) {
      const b = resolve();
      if (b instanceof Map) b.delete(key);
      else b?.removeItem(key);
    },
    keys() {
      const b = resolve();
      if (b instanceof Map) return [...b.keys()];
      return b ? Object.keys(b) : [];
    },
  };
}

/** Kept across visits: identity, seats, chat mutes. */
export const localStore = browserStore("localStorage");
/** Kept per tab: chat drafts. */
export const sessionStore = browserStore("sessionStorage");
