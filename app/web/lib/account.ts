/** Optional accounts: the client side of /api/account.
 *
 * An account is the same player identity as a guest's, reachable from any
 * device with a password. Signing in swaps this device's stored identity
 * (session token + player id) for the account's; signing out drops it, so
 * the next table join mints a fresh guest. */
import { useCallback, useEffect, useState } from "react";
import { apiFetch, getSessionToken } from "./api";
import { STORAGE_KEYS } from "./storage";
import type { AccountMe, AccountSession } from "./types";

/** A failed account call, carrying the server's `detail` code. */
export class AccountError extends Error {
  constructor(
    readonly code: string,
    readonly status: number,
  ) {
    super(code);
  }
}

const MESSAGES: Record<string, string> = {
  already_registered: "This player already has an account.",
  username_taken: "That username is taken.",
  email_taken: "An account already uses that email. Sign in instead?",
  invalid_credentials: "That email/username and password don't match.",
  invalid_or_expired_token: "This link has expired or was already used.",
  already_verified: "Your email is already confirmed.",
  rate_limited: "Too many attempts. Please wait a little and try again.",
  network: "Couldn't reach the server. Check your connection.",
};

export function accountErrorMessage(err: unknown): string {
  if (err instanceof AccountError) {
    return MESSAGES[err.code] ?? "Something went wrong. Please try again.";
  }
  return MESSAGES.network;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await apiFetch(path, init);
  } catch {
    throw new AccountError("network", 0);
  }
  if (res.status === 429) throw new AccountError("rate_limited", 429);
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    // FastAPI validation errors carry a list; surface its first message.
    const detail = data?.detail;
    const code =
      typeof detail === "string"
        ? detail
        : (detail?.[0]?.msg ?? "unknown_error");
    throw new AccountError(code, res.status);
  }
  return data as T;
}

const post = <T>(path: string, body?: unknown) =>
  call<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });

/** Adopt the identity an account call returned. Per-table client ids
 * belong to the previous player (the socket would refuse them), so they
 * are dropped when the player changes; rejoining a table is one click. */
export function switchIdentity(session: AccountSession) {
  const ls = window.localStorage;
  if (session.session_token) {
    ls.setItem(STORAGE_KEYS.sessionToken, session.session_token);
  }
  if (ls.getItem(STORAGE_KEYS.playerId) !== session.player_id) {
    forgetTableSeats();
    ls.setItem(STORAGE_KEYS.playerId, session.player_id);
  }
  if (session.name) ls.setItem(STORAGE_KEYS.displayName, session.name);
}

function forgetTableSeats() {
  const ls = window.localStorage;
  const prefix = STORAGE_KEYS.clientId("");
  for (const key of Object.keys(ls)) {
    if (key.startsWith(prefix)) ls.removeItem(key);
  }
}

export const register = (username: string, email: string, password: string) =>
  post<AccountSession>("/api/account/register", { username, email, password });

export const login = (loginName: string, password: string) =>
  post<AccountSession>("/api/account/login", { login: loginName, password });

/** Sign out: end the session server-side, then forget this device's
 * identity. The typed display name stays for the next guest session. */
export async function logout() {
  try {
    await post("/api/account/logout");
  } catch {
    // Already invalid server-side; forgetting it locally is what matters.
  }
  const ls = window.localStorage;
  ls.removeItem(STORAGE_KEYS.sessionToken);
  ls.removeItem(STORAGE_KEYS.playerId);
  forgetTableSeats();
}

export const verifyEmail = (token: string) =>
  post("/api/account/verify", { token });

export const resendVerification = () =>
  post("/api/account/resend-verification");

export const forgotPassword = (email: string) =>
  post("/api/account/forgot", { email });

export const resetPassword = (token: string, password: string) =>
  post<AccountSession>("/api/account/reset", { token, password });

export async function usernameAvailable(username: string): Promise<boolean> {
  const data = await call<{ available: boolean }>(
    `/api/account/username-available?u=${encodeURIComponent(username)}`,
  );
  return data.available;
}

/** The emailed token, read from the URL fragment (never sent to a
 * server), which is then cleared so it doesn't linger in history. */
export function takeFragmentToken(): string | null {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const token = params.get("token");
  if (token) {
    window.history.replaceState(
      null,
      "",
      window.location.pathname + window.location.search,
    );
  }
  return token;
}

export const USERNAME_RE = /^[A-Za-z0-9_-]{3,20}$/;

export interface UseAccount {
  /** undefined while loading; null me when this device has no identity. */
  me: AccountMe | null | undefined;
  refresh: () => Promise<void>;
}

/** Who this device is: a guest, an account (verified or not), or nobody
 * yet (no session token). */
export function useAccount(): UseAccount {
  const [me, setMe] = useState<AccountMe | null | undefined>(undefined);
  const refresh = useCallback(async () => {
    if (!getSessionToken()) {
      setMe(null);
      return;
    }
    try {
      setMe(await call<AccountMe>("/api/account/me"));
    } catch {
      setMe(null);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { me, refresh };
}
