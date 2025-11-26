import { useState, useEffect, useRef, useMemo } from 'react';
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

export function useResponsive(cardCount: number = 6): UseResponsiveReturn {
  const handRowRef = useRef<HTMLDivElement>(null);
  const trickBoxRef = useRef<HTMLDivElement>(null);

  const [viewport, setViewport] = useState({ width: 1024, height: 768 });
  const [trickBoxSize, setTrickBoxSize] = useState({ w: 900, h: 400 });
  const [handSize, setHandSize] = useState<CardSize>({ w: 72, h: 108 });

  // Derived breakpoint flags
  const isMobile = viewport.width < BREAKPOINTS.mobile;
  const isTablet = viewport.width >= BREAKPOINTS.mobile && viewport.width < BREAKPOINTS.tablet;
  const isDesktop = viewport.width >= BREAKPOINTS.tablet;

  // Single resize effect for viewport
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

  // Measure trick box size
  useEffect(() => {
    function measureTrickBox() {
      const el = trickBoxRef.current;
      if (!el) return;
      setTrickBoxSize({ w: el.clientWidth, h: el.clientHeight });
    }
    measureTrickBox();
    window.addEventListener('resize', measureTrickBox);
    return () => window.removeEventListener('resize', measureTrickBox);
  }, []);

  // Calculate hand card sizing with overlap
  useEffect(() => {
    const row = handRowRef.current;
    if (!row) return;

    function recalcHand() {
      const count = Math.max(1, cardCount);
      const vw = viewport.width;

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
      setHandSize({ w, h });

      // Set CSS variables for overlap layout
      if (row) {
        row.style.setProperty('--cardWidth', `${w}px`);
        row.style.setProperty('--visibleWidth', `${visibleWidth}px`);
        row.style.setProperty('--h', `${h}px`);
      }
    }

    recalcHand();
  }, [cardCount, viewport.width]);

  // Calculate center card size
  const centerSize = useMemo<CardSize>(() => {
    const containerW = trickBoxSize.w || viewport.width;

    let cw: number;
    if (isMobile) {
      cw = Math.floor(Math.min(120, Math.max(84, containerW * 0.22)));
    } else if (isTablet) {
      cw = Math.floor(Math.min(140, Math.max(96, containerW * 0.18)));
    } else {
      cw = Math.floor(Math.min(172, Math.max(112, containerW * 0.12)));
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

