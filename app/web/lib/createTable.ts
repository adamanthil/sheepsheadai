import { apiFetch, storeSessionToken } from "./api";
import { STORAGE_KEYS } from "./storage";
import type { UseIdentityReturn } from "./hooks/useIdentity";

// Why the server refused to open a table, in the player's terms.
const CREATE_REFUSED: Record<string, string> = {
  table_limit_per_player:
    "You already have 2 open tables. Close one to start another.",
  table_limit_per_ip:
    "Too many tables are open from your network right now. Try again once one closes.",
  table_limit_reached: "The server is full right now. Try again in a bit.",
};

/**
 * Opens a table with the default house rules and seats its creator as host,
 * storing the session the waiting room picks up. Returns the table id, or
 * throws an Error whose message is fit to show the player.
 */
export async function createAndJoinTable({
  name,
  displayName,
  typedName,
  persistIdentityFromJoin,
}: {
  name: string;
  /** The name to sit down under (typed, or the placeholder). */
  displayName: string;
  /** What the player typed, if anything; saved to their profile. */
  typedName: string;
  persistIdentityFromJoin: UseIdentityReturn["persistIdentityFromJoin"];
}): Promise<string> {
  const res = await apiFetch("/api/tables", {
    method: "POST",
    body: JSON.stringify({
      name,
      fillWithAI: true,
      rules: {
        partnerMode: 1,
        allPassMode: "leasters",
        doubleOnTheBump: true,
      },
    }),
  });

  if (!res.ok) {
    const detail = (await res.json().catch(() => ({})))?.detail;
    throw new Error(
      CREATE_REFUSED[detail as string] ??
        `Failed to create table: ${res.status} ${res.statusText}`,
    );
  }

  const t = await res.json();

  // Auto-join the table; identity rides on the Authorization header.
  const res2 = await apiFetch(`/api/tables/${t.id}/join`, {
    method: "POST",
    // The one-time key from the create response makes this join host.
    body: JSON.stringify({
      display_name: displayName,
      host_key: t.host_key,
    }),
  });

  if (!res2.ok) {
    throw new Error(`Failed to join table: ${res2.status} ${res2.statusText}`);
  }

  const joined = await res2.json();
  storeSessionToken(joined.session_token);
  window.localStorage.setItem(STORAGE_KEYS.clientId(t.id), joined.client_id);
  await persistIdentityFromJoin(joined, typedName);
  return t.id;
}
