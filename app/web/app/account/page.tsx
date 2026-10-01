"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import { apiFetch } from "../../lib/api";
import { useAccount } from "../../lib/account";
import type { AccountStats } from "../../lib/types";
import AccountShell from "./AccountShell";
import AuthForms from "./components/AuthForms";
import ResendVerification from "./components/ResendVerification";
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
    body = <Verified username={account.username} />;
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

function Verified({ username }: { username: string }) {
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
