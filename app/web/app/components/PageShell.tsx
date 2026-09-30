import React from "react";
import Link from "next/link";
import { ds } from "../../lib/ds";
import MastheadBand from "./home/MastheadBand";
import styles from "./PageShell.module.css";

/** Single-column shell for the account and leaderboard pages: the
 * masthead band, a way home, and a display-face title. */
export default function PageShell({
  title,
  children,
  wide = false,
}: {
  title: string;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={styles.page}>
      <div className={`${styles.inner} ${wide ? styles.wide : ""}`}>
        <MastheadBand compact />
        <nav className={styles.nav} aria-label="Site">
          <Link href="/" className={ds.link}>
            ← Sheepshead
          </Link>
          <span className={styles.navLinks}>
            <Link href="/leaderboard" className={ds.link}>
              Leaderboard
            </Link>
            <Link href="/account" className={ds.link}>
              Account
            </Link>
          </span>
        </nav>
        <h1 className={styles.title}>{title}</h1>
        {children}
      </div>
    </div>
  );
}
