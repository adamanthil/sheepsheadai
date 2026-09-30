"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import {
  accountErrorMessage,
  resendVerification,
  useAccount,
} from "../../lib/account";
import type { AccountStats } from "../../lib/types";
import AccountShell from "./AccountShell";
import AuthForms from "./components/AuthForms";
import SignOut from "./components/SignOut";
import StatsPanel from "./components/StatsPanel";
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
    body = <Verified username={account.username} onSignOut={refresh} />;
  }

  return (
    <AccountShell
      title={account ? `@${account.username}` : "Your account"}
      tab="overview"
      signedIn={!!account}
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
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
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
      {status && <p className={styles.note}>{status}</p>}
      <div className={styles.actions}>
        <button
          type="button"
          className={`${ds.btn} ${ds.btnSm}`}
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              await resendVerification();
              setStatus("Sent. Check your inbox (and spam folder).");
            } catch (err) {
              setStatus(accountErrorMessage(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          Resend the email
        </button>
        <SignOut onSignOut={onSignOut} />
      </div>
    </div>
  );
}

function Verified({
  username,
  onSignOut,
}: {
  username: string;
  onSignOut: () => Promise<void>;
}) {
  const [stats, setStats] = useState<AccountStats | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch("/api/account/stats");
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as AccountStats;
        if (!cancelled) setStats(data);
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [username]);

  return (
    <div className={styles.form}>
      {failed && <p className={styles.error}>Couldn&rsquo;t load stats.</p>}
      {stats && <StatsPanel stats={stats} />}
      <div className={styles.actions}>
        <Link
          href="/leaderboard"
          className={`${ds.btn} ${ds.btnSm} ${styles.buttonLink}`}
        >
          Leaderboard →
        </Link>
        <SignOut onSignOut={onSignOut} />
      </div>
    </div>
  );
}
