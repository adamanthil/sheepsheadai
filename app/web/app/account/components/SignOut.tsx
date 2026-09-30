"use client";

import React, { useState } from "react";
import { ds } from "../../../lib/ds";
import { logout } from "../../../lib/account";

export default function SignOut({
  onSignOut,
}: {
  onSignOut: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <button
      type="button"
      className={`${ds.btn} ${ds.btnGhost} ${ds.btnSm}`}
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        await logout();
        await onSignOut();
      }}
    >
      Sign out
    </button>
  );
}
