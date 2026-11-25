import React from "react";
import Link from "next/link";
import PageShell from "../components/PageShell";
import type { AccountPublic } from "../../lib/types";
import BetaTag from "./components/BetaTag";
import styles from "./account.module.css";

export type AccountTab =
  "overview" | "history" | "review" | "settings" | "admin";

const TABS: {
  key: AccountTab;
  label: string;
  href: string;
  beta?: boolean;
  adminOnly?: boolean;
}[] = [
  { key: "overview", label: "Overview", href: "/account" },
  { key: "history", label: "Hand history", href: "/account/history" },
  { key: "review", label: "AI review", href: "/account/review", beta: true },
  { key: "settings", label: "Settings", href: "/account/settings" },
  { key: "admin", label: "Admin", href: "/account/admin", adminOnly: true },
];

/** The account area: the page shell, titled with the account's name,
 * plus, for a signed-in account, the tabs between its pages.
 * Administrators get one more. */
export default function AccountShell({
  tab,
  account,
  wide = false,
  children,
}: {
  tab: AccountTab;
  /** The signed-in account, or null for a guest (no tabs). */
  account: AccountPublic | null;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <PageShell
      title={account ? `@${account.username}` : "Your account"}
      wide={wide}
    >
      {account && (
        <nav className={styles.tabs} aria-label="Account">
          {TABS.filter((t) => !t.adminOnly || account.is_admin).map((t) => (
            <Link
              key={t.key}
              href={t.href}
              aria-current={t.key === tab ? "page" : undefined}
              className={`${styles.tab} ${styles.tabLink} ${t.key === tab ? styles.tabActive : ""}`}
            >
              {t.label}
              {t.beta && <BetaTag />}
            </Link>
          ))}
        </nav>
      )}
      {children}
    </PageShell>
  );
}
