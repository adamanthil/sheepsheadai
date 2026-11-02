import { useState } from "react";
import type { ChatMessage } from "../types";

/**
 * Whether another player has typed in chat since it was last open: the dot
 * on a mobile chat button. Counts only what other players type, never the
 * game log. The first batch is the history the table had when we arrived,
 * and while the chat is open everything in it is read.
 */
export function useChatUnread(
  messages: ChatMessage[],
  clientId: string,
  open: boolean,
): boolean {
  const typed = messages.filter(
    (m) => m.type === "player" && m.author_id !== clientId,
  ).length;
  const [seen, setSeen] = useState<number | null>(null);
  if (messages.length > 0 && (seen === null || (open && seen !== typed))) {
    setSeen(typed);
  }
  return seen !== null && typed > seen;
}
