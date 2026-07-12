import { API_BASE } from "./apiBase";
import { STORAGE_KEYS } from "./storage";

export function getSessionToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(STORAGE_KEYS.sessionToken);
}

/** Store the token returned by a join response (present only when the
 * server minted a fresh identity). */
export function storeSessionToken(token: string | null | undefined) {
  if (typeof window === "undefined" || !token) return;
  window.localStorage.setItem(STORAGE_KEYS.sessionToken, token);
}

/** fetch against the API with the session token attached. Sets a JSON
 * Content-Type when a body is present. */
export function apiFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getSessionToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}

export function wsUrl(tableId: string): string {
  // Empty API_BASE means same-origin: derive the ws host from the page URL.
  const base = API_BASE ? new URL(API_BASE) : window.location;
  const wsProto = base.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${base.host}/ws/table/${tableId}`;
}

/** Subprotocol entries carry the client id and session token so neither
 * appears in URL access logs. token_urlsafe output is valid in this header. */
export function wsSubprotocols(clientId: string): string[] {
  const protos = [`sheepshead.client.${clientId}`];
  const token = getSessionToken();
  if (token) protos.push(`sheepshead.token.${token}`);
  return protos;
}
