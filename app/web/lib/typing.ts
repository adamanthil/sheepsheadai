/** Who else is typing in the table's chat, as the server relays it.
 *
 * A typist's client repeats "typing" every TYPING_REPEAT_MS while they
 * type; one not heard from for TYPING_STALE_MS is dropped, so a lost
 * "stopped" (a dropped connection, a closed tab) never leaves anyone
 * typing for good. */

/** How often a typing client repeats that it is typing. */
export const TYPING_REPEAT_MS = 3000;
/** How long a typist is shown without hearing from them again. */
export const TYPING_STALE_MS = 6000;
/** How long after the last keystroke a client says it stopped. */
export const TYPING_IDLE_MS = 5000;

export interface Typist {
  clientId: string;
  name: string;
  /** Local-clock ms after which they are dropped unless heard from. */
  until: number;
}

/** The list after a typing (or stopped) relay for ``clientId``. Keeps the
 * order they started in, so names don't swap places as repeats arrive. */
export function applyTyping(
  typists: Typist[],
  clientId: string,
  name: string,
  typing: boolean,
  now: number,
): Typist[] {
  const rest = typists.filter((t) => t.clientId !== clientId);
  if (!typing) return rest.length === typists.length ? typists : rest;
  const until = now + TYPING_STALE_MS;
  const i = typists.findIndex((t) => t.clientId === clientId);
  if (i === -1) return [...typists, { clientId, name, until }];
  return typists.map((t, j) => (j === i ? { clientId, name, until } : t));
}

/** The list without anyone not heard from in time. */
export function dropStale(typists: Typist[], now: number): Typist[] {
  const live = typists.filter((t) => t.until > now);
  return live.length === typists.length ? typists : live;
}
