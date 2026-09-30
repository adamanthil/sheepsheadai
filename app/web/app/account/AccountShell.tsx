import React from "react";
import Link from "next/link";
import PageShell from "../components/PageShell";
import styles from "./account.module.css";

export type AccountTab = "overview" | "history" | "settings";

const TABS: { key: AccountTab; label: string; href: string }[] = [
  { key: "overview", label: "Overview", href: "/account" },
  { key: "history", label: "Hand history", href: "/account/history" },
  { key: "settings", label: "Settings", href: "/account/settings" },
];

/** The account area: the page shell plus, for a signed-in account, the
 * tabs between its pages. */
export default function AccountShell({
  title,
  tab,
  signedIn,
  wide = false,
  children,
}: {
  title: string;
  tab: AccountTab;
  signedIn: boolean;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <PageShell title={title} wide={wide}>
      {signedIn && (
        <nav className={styles.tabs} aria-label="Account">
          {TABS.map((t) => (
            <Link
              key={t.key}
              href={t.href}
              aria-current={t.key === tab ? "page" : undefined}
              className={`${styles.tab} ${styles.tabLink} ${t.key === tab ? styles.tabActive : ""}`}
            >
              {t.label}
            </Link>
          ))}
        </nav>
      )}
      {children}
    </PageShell>
  );
}
