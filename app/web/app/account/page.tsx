"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import { HttpStatusError } from "../../lib/apiError";
import { reportCaught } from "../../lib/reportCaught";
import { useAccount } from "../../lib/account";
import type {
  AccountRunningScore,
  AccountStats,
  StatsMode,
} from "../../lib/types";
import AccountShell from "./AccountShell";
import AuthForms from "./components/AuthForms";
import ResendVerification from "./components/ResendVerification";
import SignOut from "./components/SignOut";
import StatsPanel, { STATS_MODES } from "./components/StatsPanel";
import styles from "./account.module.css";

export default function AccountPage() {
  const { me, refresh } = useAccount();
  const account = me?.account ?? null;

  let body: React.ReactNode;
  if (me === undefined) {
    body = <p className={styles.note}>Loading…</p>;
  } else if (!account) {
    body = <AuthForms me={me} onSignedIn={refresh} />;
  } else if (!account.email_verified) {
    body = <Unverified email={account.email} onSignOut={refresh} />;
  } else {
    body = <Verified username={account.username} />;
  }
  const verified = !!account?.email_verified;

  return (
    <AccountShell
      title={account ? `@${account.username}` : "Your account"}
      tab="overview"
      account={account}
      wide={verified}
    >
      {body}
    </AccountShell>
  );
}

function Unverified({
  email,
  onSignOut,
}: {
  email: string;
  onSignOut: () => Promise<void>;
}) {
  return (
    <div className={styles.form}>
      <p className={styles.lede}>
        Almost there. We sent a confirmation link to <strong>{email}</strong>.
        Confirm it to unlock your stats, the leaderboard, and your account badge
        at the table. You can keep playing in the meantime.
      </p>
      <p className={styles.note}>
        Unconfirmed accounts are removed after 7 days (your hands stay).
      </p>
      <div className={styles.actions}>
        <ResendVerification />
        <SignOut onSignOut={onSignOut} />
      </div>
    </div>
  );
}

const DEFAULT_MODE: StatsMode = "all";

/** The mode in the URL. Only read in the browser: the stats render once
 * the account has loaded there, never on the server. */
function modeFromUrl(): StatsMode {
  const m = new URLSearchParams(window.location.search).get("mode");
  return STATS_MODES.find((x) => x.key === m)?.key ?? DEFAULT_MODE;
}

/** Data as loaded for one mode, so a switch can keep showing the last
 * mode's numbers until the new ones arrive. */
type ForMode<T> = { mode: StatsMode; data: T };

function Verified({ username }: { username: string }) {
  const [mode, setMode] = useState<StatsMode>(modeFromUrl);
  const [stats, setStats] = useState<ForMode<AccountStats> | null>(null);
  const [running, setRunning] = useState<ForMode<AccountRunningScore> | null>(
    null,
  );
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const load = async <T,>(path: string, set: (got: ForMode<T>) => void) => {
      try {
        const res = await apiFetch(`${path}?mode=${mode}`);
        if (!res.ok) throw new HttpStatusError(res.status);
        const data = (await res.json()) as T;
        if (!cancelled) {
          set({ mode, data });
          setFailed(false);
        }
      } catch (err) {
        reportCaught(err, "account stats");
        if (!cancelled) setFailed(true);
      }
    };
    // Separately, so the figures show while a long history loads.
    void load<AccountStats>("/api/account/stats", setStats);
    void load<AccountRunningScore>("/api/account/running-score", setRunning);
    return () => {
      cancelled = true;
    };
  }, [username, mode]);

  const chooseMode = (next: StatsMode) => {
    setMode(next);
    window.history.replaceState(
      null,
      "",
      next === DEFAULT_MODE ? window.location.pathname : `?mode=${next}`,
    );
  };

  return (
    <div className={styles.form}>
      {failed && <p className={styles.error}>Couldn&rsquo;t load stats.</p>}
      {stats && (
        <StatsPanel
          mode={mode}
          onMode={chooseMode}
          stats={stats.data}
          statsMode={stats.mode}
          running={running?.data ?? null}
          // A switch keeps the last mode's numbers, dimmed, until the new
          // mode's arrive; a first load shows the chart's own "Loading…".
          busy={stats.mode !== mode || (!!running && running.mode !== mode)}
        />
      )}
      <div className={styles.actions}>
        <Link
          href="/account/history"
          className={`${ds.btn} ${ds.btnSm} ${styles.buttonLink}`}
        >
          See your hand history →
        </Link>
        <Link href="/account/settings" className={ds.link}>
          Settings
        </Link>
      </div>
    </div>
  );
}
