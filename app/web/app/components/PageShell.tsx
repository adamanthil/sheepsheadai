import React from "react";
import MastheadBand from "./home/MastheadBand";
import SiteNav from "./SiteNav";
import styles from "./PageShell.module.css";

/** Single-column shell for the account, leaderboard and about pages: the
 * site strip, the masthead band, and a display-face title. */
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
      <SiteNav />
      <div className={`${styles.inner} ${wide ? styles.wide : ""}`}>
        <MastheadBand compact />
        <h1 className={styles.title}>{title}</h1>
        {children}
      </div>
    </div>
  );
}
