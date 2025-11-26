import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { BREAKPOINTS, CARD_ASPECT_RATIO } from '../utils/breakpoints';

export interface CardSize {
  w: number;
  h: number;
}

export interface UseResponsiveReturn {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  handSize: CardSize;
  centerSize: CardSize;
  trickSize: CardSize;
  handTopMargin: number;
  handRowRef: React.RefObject<HTMLDivElement>;
  trickBoxRef: React.RefObject<HTMLDivElement>;
}

// Get initial viewport safely (SSR-compatible)
function getInitialViewport() {
  if (typeof window === 'undefined') {
    return { width: 1024, height: 768 };
  }
  return { width: window.innerWidth, height: window.innerHeight };
}

// Calculate hand card size - pure function, no ref dependency
function calculateHandSize(vw: number, cardCount: number): { w: number; h: number; visibleWidth: number } {
  const count = Math.max(1, cardCount);

  let minVisibleWidth: number;
  let maxCardWidth: number;
  let minCardWidth: number;
  let availableWidth: number;

  if (vw < BREAKPOINTS.mobile) {
    minVisibleWidth = 32;
    maxCardWidth = 110;
    minCardWidth = 60;
    availableWidth = vw * 0.92;
  } else if (vw < BREAKPOINTS.tablet) {
    minVisibleWidth = 32;
    maxCardWidth = 140;
    minCardWidth = 80;
    availableWidth = vw * 0.94;
  } else {
    minVisibleWidth = 40;
    maxCardWidth = 160;
    minCardWidth = 100;
    availableWidth = Math.min(vw * 0.94, 1600);
  }

  const padding = 32;
  const maxTotalWidth = availableWidth - padding;

  let w = maxCardWidth;
  let visibleWidth = minVisibleWidth;
  let totalNeeded = w + (count - 1) * visibleWidth;

  if (totalNeeded > maxTotalWidth) {
    w = Math.floor(maxTotalWidth - (count - 1) * visibleWidth);
    w = Math.max(minCardWidth, w);
    totalNeeded = w + (count - 1) * visibleWidth;
    if (totalNeeded > maxTotalWidth) {
      visibleWidth = Math.max(24, Math.floor((maxTotalWidth - w) / (count - 1)));
    }
  }

  const h = Math.floor(w * CARD_ASPECT_RATIO);
  return { w, h, visibleWidth };
}

export function useResponsive(cardCount: number = 6): UseResponsiveReturn {
  const handRowRef = useRef<HTMLDivElement>(null);
  const trickBoxRef = useRef<HTMLDivElement>(null);

  // Initialize with actual window size if available
  const [viewport, setViewport] = useState(getInitialViewport);
  const [trickBoxSize, setTrickBoxSize] = useState<CardSize>({ w: 0, h: 0 });

  // Derived breakpoint flags
  const isMobile = viewport.width < BREAKPOINTS.mobile;
  const isTablet = viewport.width >= BREAKPOINTS.mobile && viewport.width < BREAKPOINTS.tablet;
  const isDesktop = viewport.width >= BREAKPOINTS.tablet;

  // Calculate hand size directly from viewport - no ref needed for the calculation itself
  const handSize = useMemo<CardSize>(() => {
    const { w, h } = calculateHandSize(viewport.width, cardCount);
    return { w, h };
  }, [viewport.width, cardCount]);

  // Measure trick box
  const measureTrickBox = useCallback(() => {
    const el = trickBoxRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setTrickBoxSize((prev) => {
      if (prev.w !== rect.width || prev.h !== rect.height) {
        return { w: rect.width, h: rect.height };
      }
      return prev;
    });
  }, []);

  // Update CSS variables on the hand row element
  const updateHandCssVars = useCallback(() => {
    const row = handRowRef.current;
    if (!row) return;

    const { w, h, visibleWidth } = calculateHandSize(viewport.width, cardCount);
    row.style.setProperty('--cardWidth', `${w}px`);
    row.style.setProperty('--visibleWidth', `${visibleWidth}px`);
    row.style.setProperty('--h', `${h}px`);
  }, [viewport.width, cardCount]);

  // Viewport resize listener
  useEffect(() => {
    function handleResize() {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Trick box measurement with ResizeObserver
  useEffect(() => {
    const el = trickBoxRef.current;

    // Try to measure immediately
    measureTrickBox();

    // If element not ready, try again shortly
    if (!el) {
      const timer = setTimeout(measureTrickBox, 100);
      return () => clearTimeout(timer);
    }

    // Use ResizeObserver for container size changes
    const resizeObserver = new ResizeObserver(() => {
      measureTrickBox();
    });
    resizeObserver.observe(el);

    window.addEventListener('resize', measureTrickBox);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', measureTrickBox);
    };
  }, [measureTrickBox]);

  // Update CSS variables when hand size changes or ref becomes available
  useEffect(() => {
    updateHandCssVars();

    // Also try after a short delay for initial mount
    const timer = setTimeout(updateHandCssVars, 50);

    window.addEventListener('resize', updateHandCssVars);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', updateHandCssVars);
    };
  }, [updateHandCssVars]);

  // Calculate center card size - sized to fit within positioned spots
  const centerSize = useMemo<CardSize>(() => {
    // Use viewport width as fallback if trick box not measured yet
    const containerW = trickBoxSize.w > 0 ? trickBoxSize.w : viewport.width * 0.96;

    let cw: number;
    if (isMobile) {
      // Mobile: smaller cards
      cw = Math.floor(Math.min(90, Math.max(65, containerW * 0.16)));
    } else if (isTablet) {
      // Tablet: medium cards
      cw = Math.floor(Math.min(110, Math.max(80, containerW * 0.12)));
    } else {
      // Desktop: cards sized to fit within the 20%-80% positioning range
      // With positions at 20% and 80%, we need cards to fit with margin
      // Max card width ~10% of container to stay safely within bounds
      const maxFromContainer = Math.floor(containerW * 0.10);
      cw = Math.floor(Math.min(130, Math.max(95, maxFromContainer)));
    }

    return { w: cw, h: Math.floor(cw * CARD_ASPECT_RATIO) };
  }, [isMobile, isTablet, trickBoxSize.w, viewport.width]);

  // Calculate hand top margin
  const handTopMargin = useMemo(() => {
    if (isMobile) {
      return Math.max(54, Math.floor(centerSize.h * 0.25));
    } else if (isTablet) {
      return Math.max(36, Math.floor(centerSize.h * 0.22));
    }
    return Math.max(32, Math.floor(centerSize.h * 0.2));
  }, [isMobile, isTablet, centerSize.h]);

  return {
    isMobile,
    isTablet,
    isDesktop,
    handSize,
    centerSize,
    trickSize: trickBoxSize,
    handTopMargin,
    handRowRef,
    trickBoxRef,
  };
}
