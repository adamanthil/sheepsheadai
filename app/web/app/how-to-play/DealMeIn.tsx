"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { createAndJoinTable } from "../../lib/createTable";
import { useIdentity } from "../../lib/hooks/useIdentity";
import type { TableSummary } from "../../lib/types";
import styles from "./HowToPlay.module.css";

// The lesson has no lobby list to steer the default table name around.
const NO_TABLES: TableSummary[] = [];

/** The lesson's closing call to action: opens a table with AI players under
 * the player's saved (or placeholder) name and goes to its waiting room,
 * exactly as the lobby's "Create table" does with untouched fields. */
export default function DealMeIn({
  text,
  x,
  y,
}: {
  text: string;
  x: number;
  y: number;
}) {
  const router = useRouter();
  const { name, displayName, displayNameInput, persistIdentityFromJoin } =
    useIdentity(NO_TABLES);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const deal = async () => {
    setError(null);
    setCreating(true);
    try {
      const tableId = await createAndJoinTable({
        name,
        displayName,
        typedName: displayNameInput.trim(),
        persistIdentityFromJoin,
      });
      router.push(`/waiting/${tableId}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to create table. Please try again.",
      );
      setCreating(false);
    }
  };

  return (
    <div
      className={styles.cta}
      style={{ "--x": x, "--y": y } as React.CSSProperties}
    >
      <button
        type="button"
        className={styles.ctaBtn}
        onClick={deal}
        disabled={creating || !name || !displayName}
      >
        {creating ? "Dealing…" : text} <span aria-hidden="true">→</span>
      </button>
      {error && <div className={styles.ctaError}>{error}</div>}
    </div>
  );
}
