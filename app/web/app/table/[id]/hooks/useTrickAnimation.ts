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
  triggerCollect: (cards: string[], winner: number) => void;
  setLastTrickOpen: (open: boolean) => void;
}

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

  const triggerCollect = useCallback((cards: string[], winner: number) => {
    // Clear existing timers
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);

    // First hold the trick just taken on the table
    setLastTrickOpen(false);
    setHolding(true);

    // After pause, start collection animation
    pauseTimerRef.current = setTimeout(() => {
      setAnimTrick({ cards, winner });

      // After animation completes, hide everything
      animTimerRef.current = setTimeout(() => {
        setAnimTrick(null);
        setHolding(false);
      }, 1300);
    }, 2000);
  }, []);

  return {
    showPrev: holding || lastTrickOpen,
    lastTrickOpen,
    animTrick,
    triggerCollect,
    setLastTrickOpen,
  };
}
