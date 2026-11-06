"use client";

import React from "react";
import Script from "next/script";
import { usePathname } from "next/navigation";

// GoatCounter (https://sheepsheadai.goatcounter.com): cookieless page
// counts. The counting script is served from our own origin:
// public/vendor/goatcounter/count.v4.js is https://gc.zgo.at/count.v4.js
// copied verbatim (ISC). To update it, download over that file, read the
// diff, and commit it on its own. The endpoint must also be in the CSP's
// connect-src (next.config.mjs).
export const GOATCOUNTER_ENDPOINT =
  "https://sheepsheadai.goatcounter.com/count";

/** The only host that counts. The repository is public: a clone, a fork
 * deployed elsewhere, a staging box or a dev server must not post page
 * views into this account. (count.js alone skips only localhost and
 * private IPs.) */
export const COUNTED_HOSTNAME = "sheepshead.ai";

const subscribeNever = () => () => {};
const onCountedHost = () => window.location.hostname === COUNTED_HOSTNAME;
const onServer = () => false;

type GoatCounter = {
  count?: (vars: { path: string; referrer?: string }) => void;
};

declare global {
  interface Window {
    goatcounter?: GoatCounter;
  }
}

/** Ids name one table (and a link to it is an invitation), so they are
 * folded out: GoatCounter sees one path per page, not one per table. */
export function countedPath(pathname: string): string {
  return pathname.replace(/^\/(table|waiting)\/[^/]+/, "/$1/[id]");
}

/** Counts each page view, client-side navigations included. count.js's
 * own on-load count is off (no_onload): it would see only the first page
 * of the session, and with the raw path. Off the production host the
 * script is never loaded, so count() finds no window.goatcounter. */
export default function Analytics() {
  const enabled = React.useSyncExternalStore(
    subscribeNever,
    onCountedHost,
    onServer,
  );
  const pathname = usePathname();
  const latest = React.useRef(pathname);
  // The last path counted. Route changes and the script's load race on the
  // first page, and whichever comes second must not count it again.
  const counted = React.useRef<string | null>(null);

  const count = React.useCallback(() => {
    const gc = window.goatcounter;
    const path = latest.current;
    if (!gc?.count || counted.current === path) return;
    // document.referrer stays the external referrer for the whole session
    // in an SPA; send it with the landing page only ("" omits it).
    const first = counted.current === null;
    counted.current = path;
    gc.count({ path: countedPath(path), ...(first ? {} : { referrer: "" }) });
  }, []);

  React.useEffect(() => {
    latest.current = pathname;
    count();
  }, [pathname, count]);

  if (!enabled) return null;
  return (
    <Script
      src="/vendor/goatcounter/count.v4.js"
      strategy="afterInteractive"
      data-goatcounter={GOATCOUNTER_ENDPOINT}
      data-goatcounter-settings='{"no_onload": true}'
      onLoad={count}
    />
  );
}
