"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ds } from "../../../lib/ds";
import {
  accountErrorMessage,
  resetPassword,
  switchIdentity,
  takeFragmentToken,
} from "../../../lib/account";
import PageShell from "../../components/PageShell";
import styles from "../account.module.css";

/** Landing page for the emailed reset link (/account/reset#token=…). */
export default function ResetPage() {
  // undefined until the fragment is read on the client.
  const [token, setToken] = useState<string | null | undefined>(undefined);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    setToken(takeFragmentToken());
  }, []);

  const mismatch = confirm.length > 0 && confirm !== password;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setBusy(true);
    setError(null);
    try {
      // The reset signs out every other session and signs this device in.
      switchIdentity(await resetPassword(token, password));
      setDone(true);
    } catch (err) {
      setError(accountErrorMessage(err));
      setBusy(false);
    }
  };

  let body: React.ReactNode;
  if (token === undefined) {
    body = null;
  } else if (done) {
    body = (
      <>
        <p className={styles.lede}>
          Password changed. You&rsquo;re signed in here and signed out
          everywhere else.
        </p>
        <div className={styles.actions}>
          <Link
            href="/account"
            className={`${ds.btn} ${ds.btnSm} ${styles.buttonLink}`}
          >
            Your account →
          </Link>
        </div>
      </>
    );
  } else if (!token) {
    body = (
      <p className={styles.error}>
        This link is missing its token. Try the link in the email, or{" "}
        <Link href="/account" className={ds.link}>
          request a new one
        </Link>
        .
      </p>
    );
  } else {
    body = (
      <form className={styles.form} onSubmit={submit}>
        <label className={styles.field}>
          <span className={styles.label}>New password</span>
          <input
            className={ds.input}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            maxLength={128}
            required
          />
          <span className={styles.hint}>At least 8 characters.</span>
        </label>
        <label className={styles.field}>
          <span className={styles.label}>Again</span>
          <input
            className={ds.input}
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            required
          />
          {mismatch && (
            <span className={styles.error}>Passwords don&rsquo;t match.</span>
          )}
        </label>
        {error && <p className={styles.error}>{error}</p>}
        <div className={styles.actions}>
          <button
            className={`${ds.btn} ${ds.btnAccent}`}
            disabled={busy || password.length < 8 || confirm !== password}
          >
            {busy ? "Saving…" : "Set new password →"}
          </button>
        </div>
      </form>
    );
  }

  return <PageShell title="New password">{body}</PageShell>;
}
