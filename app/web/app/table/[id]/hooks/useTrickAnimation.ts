import { useState, useRef, useCallback, useEffect } from "react";

export interface AnimTrick {
  cards: string[];
  winner: number;
}

export interface UseTrickAnimationReturn {
  /** The last trick is on the table: held there after it is taken, or
   * opened by the player. */
  showPrev: boolean;
  /** The player opened the last trick with its button. */
  lastTrickOpen: boolean;
  animTrick: AnimTrick | null;
  triggerCollect: (
    cards: string[],
    winner: number,
    shownForMs?: number,
  ) => void;
  setLastTrickOpen: (open: boolean) => void;
}

// A finished trick stays on the table this long before it is collected,
// counted from when it was complete on screen: the player's own last card
// shows before the server confirms the trick, and that wait counts toward
// the look. Once the server names the winner it still gets a short beat.
// The collect animation follows. TRICK_COLLECT_SECONDS on the server (the
// pause before the next AI lead) covers the full look plus the collect.
const TRICK_LOOK_MS = 2000;
const WINNER_BEAT_MS = 600;
const COLLECT_MS = 1300;

export function useTrickAnimation(): UseTrickAnimationReturn {
  const [holding, setHolding] = useState(false);
  const [lastTrickOpen, setLastTrickOpen] = useState(false);
  const [animTrick, setAnimTrick] = useState<AnimTrick | null>(null);
  const animTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pauseTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (animTimerRef.current) clearTimeout(animTimerRef.current);
      if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
    };
  }, []);

  const triggerCollect = useCallback(
    (cards: string[], winner: number, shownForMs = 0) => {
      // Clear existing timers
      if (animTimerRef.current) clearTimeout(animTimerRef.current);
      if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);

      // First hold the trick just taken on the table
      setLastTrickOpen(false);
      setHolding(true);

      // After pause, start collection animation
      pauseTimerRef.current = setTimeout(
        () => {
          setAnimTrick({ cards, winner });

          // After animation completes, hide everything
          animTimerRef.current = setTimeout(() => {
            setAnimTrick(null);
            setHolding(false);
          }, COLLECT_MS);
        },
        Math.max(WINNER_BEAT_MS, TRICK_LOOK_MS - shownForMs),
      );
    },
    [],
  );

  return {
    showPrev: holding || lastTrickOpen,
    lastTrickOpen,
    animTrick,
    triggerCollect,
    setLastTrickOpen,
  };
}
