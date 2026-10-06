"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ds } from "../../../lib/ds";
import {
  accountErrorMessage,
  takeFragmentToken,
  verifyEmail,
} from "../../../lib/account";
import PageShell from "../../components/PageShell";
import styles from "../account.module.css";

/** Landing page for the emailed confirmation link (/account/verify#token=…). */
export default function VerifyPage() {
  const [status, setStatus] = useState<"working" | "done" | string>("working");

  // Taking the token clears the fragment, so a second effect run (Strict
  // Mode) would read nothing and report a missing token.
  const taken = useRef(false);
  useEffect(() => {
    if (taken.current) return;
    taken.current = true;
    const token = takeFragmentToken();
    if (!token) {
      setStatus("This link is missing its token. Try the link in the email.");
      return;
    }
    verifyEmail(token).then(
      () => setStatus("done"),
      (err) => setStatus(accountErrorMessage(err)),
    );
  }, []);

  return (
    <PageShell title="Confirm email">
      <div className={styles.form}>
        {status === "working" && <p className={styles.note}>Confirming…</p>}
        {status === "done" && (
          <p className={styles.lede}>
            Your email is confirmed. Your stats are unlocked and your account
            badge now shows at the table.
          </p>
        )}
        {status !== "working" && status !== "done" && (
          <p className={styles.error}>{status}</p>
        )}
        <div className={styles.actions}>
          <Link
            href="/account"
            className={`${ds.btn} ${ds.btnSm} ${styles.buttonLink}`}
          >
            Your account →
          </Link>
          <Link href="/" className={ds.link}>
            Play
          </Link>
        </div>
      </div>
    </PageShell>
  );
}
