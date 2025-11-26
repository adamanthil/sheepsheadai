import { useState, useRef, useCallback, useEffect } from 'react';

export interface AnimTrick {
  cards: string[];
  winner: number;
}

export interface UseTrickAnimationReturn {
  showPrev: boolean;
  animTrick: AnimTrick | null;
  triggerCollect: (cards: string[], winner: number) => void;
  setShowPrev: (show: boolean) => void;
}

export function useTrickAnimation(): UseTrickAnimationReturn {
  const [showPrev, setShowPrev] = useState(false);
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

    // First show the previous trick
    setShowPrev(true);

    // After pause, start collection animation
    pauseTimerRef.current = setTimeout(() => {
      setAnimTrick({ cards, winner });

      // After animation completes, hide everything
      animTimerRef.current = setTimeout(() => {
        setAnimTrick(null);
        setShowPrev(false);
      }, 1300);
    }, 2000);
  }, []);

  return { showPrev, animTrick, triggerCollect, setShowPrev };
}

