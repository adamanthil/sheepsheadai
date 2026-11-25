import { useState } from "react";

/** Which of a page's views is showing, kept in the address (?view=) so
 * that a link, or the way back from another page, can open one of them.
 * ``views`` lists them, the default first; ``href`` is the page's address
 * on each. Only for components that render in the browser. */
export function useLinkedView<T extends string>(
  views: readonly T[],
  href: (view: T) => string,
): [T, (view: T) => void] {
  const [view, setView] = useState<T>(() => {
    const asked = new URLSearchParams(window.location.search).get("view");
    return views.find((v) => v === asked) ?? views[0];
  });
  const show = (next: T) => {
    setView(next);
    window.history.replaceState(null, "", href(next));
  };
  return [view, show];
}
