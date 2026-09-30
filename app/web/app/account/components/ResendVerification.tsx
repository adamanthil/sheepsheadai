"use client";

import React, { useState } from "react";
import { ds } from "../../../lib/ds";
import { accountErrorMessage, resendVerification } from "../../../lib/account";
import styles from "../account.module.css";

/** Re-send the confirmation email, reporting how it went beside the
 * button. */
export default function ResendVerification() {
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <>
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
      {status && <span className={styles.note}>{status}</span>}
    </>
  );
}
