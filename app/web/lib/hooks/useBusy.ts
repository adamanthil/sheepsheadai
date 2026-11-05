import { useCallback, useEffect, useRef, useState } from "react";

/** How long a held button may stay busy with nothing moving on, before it
 * gives the player their button back. */
const HOLD_LIMIT_MS = 15_000;

export interface Busy {
  /** True from the press until the action settles (see ``run``). */
  busy: boolean;
  /**
   * Run ``action`` with the button busy; a press while busy is ignored.
   * ``action`` resolves false when it failed (and said so), true or nothing
   * when it succeeded.
   *
   * With ``hold``, a success keeps the button busy: for actions whose
   * result is a navigation or a state the server sends after answering
   * (the new deal, the seat taken), where the button goes away on its own
   * and must not flash back to idle in between. It clears after
   * HOLD_LIMIT_MS in case nothing comes.
   */
  run: (
    action: () => Promise<boolean | void>,
    opts?: { hold?: boolean },
  ) => Promise<void>;
}

/** Busy state for one button that waits on the server; style it with
 * ``ds.btnBusy``. */
export function useBusy(): Busy {
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (holdTimer.current) clearTimeout(holdTimer.current);
    },
    [],
  );

  const run = useCallback<Busy["run"]>(async (action, opts) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    const release = () => {
      inFlight.current = false;
      setBusy(false);
    };
    let ok = false;
    try {
      ok = (await action()) !== false;
    } finally {
      if (ok && opts?.hold) {
        holdTimer.current = setTimeout(release, HOLD_LIMIT_MS);
      } else {
        release();
      }
    }
  }, []);

  return { busy, run };
}
