"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAccount } from "../../lib/account";
import styles from "./SiteNav.module.css";

/** Where the Donate link points until the Patreon page exists. */
const DONATE_HREF = "/about#support";

const SECTIONS: {
  href: string;
  label: string;
  short: string;
  exact?: boolean;
}[] = [
  { href: "/", label: "Lobby", short: "Lobby", exact: true },
  { href: "/leaderboard", label: "Leaderboard", short: "Leaderboard" },
  { href: "/about", label: "About", short: "About" },
  // ↗: the analysis tools leave the broadsheet for their own dashboard look.
  { href: "/analyze", label: "The AI ↗", short: "AI ↗" },
];

function isActive(pathname: string, href: string, exact = false): boolean {
  if (exact) return pathname === href;
  return pathname === href || pathname.startsWith(`${href}/`);
}

/** The skybox strip: one full-width row over every broadsheet page, with
 * the site's sections on the left and Donate + the account on the right.
 * The table and waiting pages keep their own immersive chrome instead. */
export default function SiteNav() {
  const pathname = usePathname();
  const { me } = useAccount();
  const account = me?.account;
  const accountActive = isActive(pathname, "/account");

  return (
    <nav className={styles.strip} aria-label="Site">
      <div className={styles.inner}>
        <div className={styles.sections}>
          {SECTIONS.map((s) => {
            const active = isActive(pathname, s.href, s.exact);
            return (
              <Link
                key={s.href}
                href={s.href}
                aria-current={active ? "page" : undefined}
                className={`${styles.item} ${active ? styles.itemActive : ""}`}
              >
                <span className={styles.longLabel}>{s.label}</span>
                <span className={styles.shortLabel}>{s.short}</span>
              </Link>
            );
          })}
        </div>
        <div className={styles.actions}>
          <a href={DONATE_HREF} className={styles.donate} aria-label="Donate">
            <span className={styles.heart} aria-hidden="true">
              ♥
            </span>
            <span className={styles.longLabel}>Donate</span>
          </a>
          <span className={styles.divider} aria-hidden="true" />
          {/* Reserve the slot while /me loads so the strip doesn't jump. */}
          {me === undefined ? (
            <span className={styles.accountPending} />
          ) : account ? (
            <Link
              href="/account"
              aria-current={accountActive ? "page" : undefined}
              aria-label={`Account: @${account.username}`}
              className={`${styles.account} ${accountActive ? styles.accountActive : ""}`}
            >
              <span className={styles.avatar} aria-hidden="true">
                {account.username.charAt(0).toUpperCase()}
              </span>
              <span className={styles.longLabel}>@{account.username}</span>
            </Link>
          ) : (
            <Link
              href="/account"
              aria-current={accountActive ? "page" : undefined}
              className={`${styles.item} ${accountActive ? styles.itemActive : ""}`}
            >
              Sign in
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
