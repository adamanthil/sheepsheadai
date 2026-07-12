import { useEffect, useState } from "react";

/** SSR-safe media-query hook. Returns false on the server / first paint. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mql = window.matchMedia(query);
    const update = () => setMatches(mql.matches);
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, [query]);

  return matches;
}

/** Layout switch used across pages: mobile is <= 768px wide. */
export function useIsMobile(): boolean {
  return useMediaQuery("(max-width: 768px)");
}
