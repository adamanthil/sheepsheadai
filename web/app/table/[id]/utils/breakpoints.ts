// Centralized breakpoint constants
export const BREAKPOINTS = {
  mobile: 480,
  tablet: 768,
  desktop: 1024,
} as const;

export type BreakpointKey = keyof typeof BREAKPOINTS;

// Helper to check current breakpoint
export function getBreakpoint(width: number): BreakpointKey {
  if (width < BREAKPOINTS.mobile) return 'mobile';
  if (width < BREAKPOINTS.tablet) return 'tablet';
  return 'desktop';
}

// Card aspect ratio constant
export const CARD_ASPECT_RATIO = 1.45;

