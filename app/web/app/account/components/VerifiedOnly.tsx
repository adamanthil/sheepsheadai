import React from "react";
import Link from "next/link";
import { ds } from "../../../lib/ds";
import type { AccountPublic } from "../../../lib/types";
import styles from "../account.module.css";

/** The body of a page that needs a confirmed email: its children for an
 * account that has one, and otherwise word that it is loading or that the
 * email is still to be confirmed. */
export default function VerifiedOnly({
  account,
  to,
  children,
}: {
  /** As useRequiredAccount gives it: null while loading. */
  account: AccountPublic | null;
  /** What confirming unlocks, after "Confirm your email to". */
  to: string;
  children: React.ReactNode;
}) {
  if (!account) return <p className={styles.note}>Loading…</p>;
  if (!account.email_verified) {
    return (
      <p className={styles.note}>
        Confirm your email to {to}.{" "}
        <Link href="/account" className={ds.link}>
          Back to your account
        </Link>
      </p>
    );
  }
  return children;
}
