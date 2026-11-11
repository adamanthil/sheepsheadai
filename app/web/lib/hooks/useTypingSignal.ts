import { useCallback, useEffect, useRef } from "react";
import { TYPING_IDLE_MS, TYPING_REPEAT_MS } from "../typing";

/** Says when this player is typing in the chat, for the others' typing
 * indicator. Returns the function to call with the draft on every edit:
 * a non-empty draft says "typing" (repeated while edits keep coming), an
 * empty one (cleared, or sent) says "stopped", and so does a pause of
 * TYPING_IDLE_MS or leaving the page. A draft restored on load says
 * nothing until it is edited. */
export function useTypingSignal(
  sendTyping: (typing: boolean) => boolean,
): (draft: string) => void {
  const sentAt = useRef<number | null>(null);
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stop = useCallback(() => {
    if (idleTimer.current) clearTimeout(idleTimer.current);
    idleTimer.current = null;
    if (sentAt.current === null) return;
    sentAt.current = null;
    sendTyping(false);
  }, [sendTyping]);

  useEffect(() => stop, [stop]);

  return useCallback(
    (draft: string) => {
      if (!draft.trim()) {
        stop();
        return;
      }
      const now = Date.now();
      if (sentAt.current === null || now - sentAt.current >= TYPING_REPEAT_MS) {
        // Not counted unless it went out, so the next edit tries again.
        if (sendTyping(true)) sentAt.current = now;
      }
      if (idleTimer.current) clearTimeout(idleTimer.current);
      idleTimer.current = setTimeout(stop, TYPING_IDLE_MS);
    },
    [sendTyping, stop],
  );
}
