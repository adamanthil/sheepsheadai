/** Re-reading the hand history for reviews that have arrived since. */
import { fetchHands } from "./account";
import type { HandHistory, HandHistoryRow } from "./types";

type FetchPage = (before: string | null) => Promise<HandHistory>;

/** Where the AI reviews of ``gameIds`` stand now. The history only reads a
 * page at a time, newest first, so this walks it from the top until every
 * hand asked for has been passed (or the history ends), however many
 * pages down the oldest of them is. */
export async function latestReviews(
  gameIds: Iterable<string>,
  fetchPage: FetchPage = fetchHands,
): Promise<Map<string, HandHistoryRow["review"]>> {
  const wanted = new Set(gameIds);
  const found = new Map<string, HandHistoryRow["review"]>();
  let cursor: string | null = null;
  while (found.size < wanted.size) {
    const page = await fetchPage(cursor);
    for (const hand of page.hands) {
      if (wanted.has(hand.game_id)) found.set(hand.game_id, hand.review);
    }
    cursor = page.next_cursor ?? null;
    if (!cursor) break;
  }
  return found;
}
