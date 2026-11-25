import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { STORAGE_KEYS } from "./storage";

// The stores pick their backing at first use, so each case loads a fresh
// copy of the module against the window it sets up.
async function freshStore() {
  vi.resetModules();
  return (await import("./storage")).localStore;
}

/** Like a browser's Storage: Object.keys() lists the stored keys only. */
function fakeStorage(): Storage {
  const methods = {
    getItem(this: Record<string, string>, k: string) {
      return Object.hasOwn(this, k) ? this[k] : null;
    },
    setItem(this: Record<string, string>, k: string, v: string) {
      this[k] = v;
    },
    removeItem(this: Record<string, string>, k: string) {
      delete this[k];
    },
  };
  return Object.create(methods) as Storage;
}

describe("localStore", () => {
  beforeEach(() => vi.unstubAllGlobals());
  afterEach(() => vi.unstubAllGlobals());

  it("uses web storage when the browser has it", async () => {
    const storage = fakeStorage();
    vi.stubGlobal("window", { localStorage: storage });
    const store = await freshStore();
    store.setItem(STORAGE_KEYS.playerId, "p1");
    expect(storage.getItem(STORAGE_KEYS.playerId)).toBe("p1");
    expect(store.keys()).toEqual([STORAGE_KEYS.playerId]);
  });

  it("falls back to memory when storage is null", async () => {
    // As in WebViews with DOM storage turned off.
    vi.stubGlobal("window", { localStorage: null });
    const store = await freshStore();
    expect(store.getItem(STORAGE_KEYS.sessionToken)).toBeNull();
    store.setItem(STORAGE_KEYS.sessionToken, "tok");
    expect(store.getItem(STORAGE_KEYS.sessionToken)).toBe("tok");
    expect(store.keys()).toEqual([STORAGE_KEYS.sessionToken]);
    store.removeItem(STORAGE_KEYS.sessionToken);
    expect(store.getItem(STORAGE_KEYS.sessionToken)).toBeNull();
  });

  it("falls back to memory when access is blocked", async () => {
    vi.stubGlobal("window", {
      get localStorage(): Storage {
        throw new DOMException("denied", "SecurityError");
      },
    });
    const store = await freshStore();
    store.setItem(STORAGE_KEYS.displayName, "Ann");
    expect(store.getItem(STORAGE_KEYS.displayName)).toBe("Ann");
  });

  it("reads nothing and keeps nothing on the server", async () => {
    vi.stubGlobal("window", undefined);
    const store = await freshStore();
    store.setItem(STORAGE_KEYS.playerId, "p1");
    expect(store.getItem(STORAGE_KEYS.playerId)).toBeNull();
    expect(store.keys()).toEqual([]);
  });
});
