"use client";

import React from "react";
import Link from "next/link";
import { ds } from "../../../lib/ds";
import { useAccount } from "../../../lib/account";
import styles from "./AccountLine.module.css";

/** One line under the create form: who this device plays as, and the way
 * into the optional account. Guests are never nagged beyond this line. */
export default function AccountLine() {
  const { me } = useAccount();
  if (me === undefined) return <div className={styles.line} />;
  const account = me?.account;

  if (!account) {
    return (
      <div className={styles.line}>
        Playing as a guest.{" "}
        <Link href="/account" className={ds.link}>
          Sign in or create an account
        </Link>{" "}
        to keep your name and stats on any device.
      </div>
    );
  }
  return (
    <div className={styles.line}>
      Signed in as{" "}
      <Link href="/account" className={ds.link}>
        @{account.username}
      </Link>
      {!account.email_verified && <> · confirm your email to unlock stats</>}
    </div>
  );
}
