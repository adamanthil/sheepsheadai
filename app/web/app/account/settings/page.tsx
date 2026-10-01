"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ds } from "../../../lib/ds";
import {
  AccountError,
  accountErrorMessage,
  changePassword,
  useAccount,
} from "../../../lib/account";
import type { AccountPublic } from "../../../lib/types";
import AccountShell from "../AccountShell";
import Field from "../components/Field";
import ResendVerification from "../components/ResendVerification";
import SignOut from "../components/SignOut";
import styles from "../account.module.css";

/** Account details, password change, and sign out. Open to unconfirmed
 * accounts too; signed-out visitors go back to the sign-in page. */
export default function SettingsPage() {
  const router = useRouter();
  const { me, refresh } = useAccount();
  const account = me?.account ?? null;

  useEffect(() => {
    if (me !== undefined && !account) router.replace("/account");
  }, [me, account, router]);

  return (
    <AccountShell
      title={account ? `@${account.username}` : "Your account"}
      tab="settings"
      signedIn={!!account}
    >
      {account ? (
        <div className={styles.sections}>
          <Details account={account} />
          <ChangePassword />
          <div className={styles.actions}>
            <SignOut onSignOut={refresh} />
          </div>
        </div>
      ) : (
        <p className={styles.note}>Loading…</p>
      )}
    </AccountShell>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section aria-label={title} className={styles.section}>
      <div className={`${ds.headRule} ${styles.statsHead}`}>
        <span className={ds.overline}>{title}</span>
      </div>
      {children}
    </section>
  );
}

function Details({ account }: { account: AccountPublic }) {
  return (
    <Section title="Account">
      <dl className={styles.details}>
        <div>
          <dt className={styles.label}>Username</dt>
          <dd>@{account.username}</dd>
        </div>
        <div>
          <dt className={styles.label}>Email</dt>
          <dd>
            {account.email}{" "}
            <span className={styles.note}>
              {account.email_verified ? "· confirmed" : "· not confirmed yet"}
            </span>
          </dd>
        </div>
      </dl>
      {!account.email_verified && (
        <div className={styles.actions}>
          <ResendVerification />
        </div>
      )}
    </Section>
  );
}

function ChangePassword() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const mismatch = confirm.length > 0 && confirm !== next;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setDone(false);
    try {
      await changePassword(current, next);
      setCurrent("");
      setNext("");
      setConfirm("");
      setDone(true);
    } catch (err) {
      setError(
        err instanceof AccountError && err.code === "invalid_credentials"
          ? "Your current password isn't right."
          : accountErrorMessage(err),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Change password">
      <form className={styles.form} onSubmit={submit}>
        <Field label="Current password">
          <input
            className={ds.input}
            type="password"
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
            autoComplete="current-password"
            required
          />
        </Field>
        <Field label="New password" hint="At least 8 characters.">
          <input
            className={ds.input}
            type="password"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            maxLength={128}
            required
          />
        </Field>
        <Field
          label="Confirm new password"
          hint={mismatch ? "Doesn't match the new password." : undefined}
        >
          <input
            className={ds.input}
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            required
          />
        </Field>
        {error && <p className={styles.error}>{error}</p>}
        {done && (
          <p className={styles.note}>
            Password changed. Your other devices are signed out.
          </p>
        )}
        <div className={styles.actions}>
          <button
            className={`${ds.btn} ${ds.btnAccent}`}
            disabled={busy || !current || next.length < 8 || confirm !== next}
          >
            {busy ? "Changing…" : "Change password"}
          </button>
        </div>
      </form>
    </Section>
  );
}
