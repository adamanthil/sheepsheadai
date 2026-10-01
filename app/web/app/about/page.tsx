"use client";

import React from "react";
import { ds } from "../../lib/ds";
import PageShell from "../components/PageShell";
import styles from "./about.module.css";

/** The Patreon page, once it exists; until then the button says so. */
const PATREON_URL: string | null = null;

/** About the site. Only the Support section is written so far; the site
 * strip's Donate link lands on its #support anchor. */
export default function AboutPage() {
  return (
    <PageShell title="About" wide>
      <section id="support" className={styles.section}>
        <div className={ds.headRule} />
        <div className={`${ds.overline} ${styles.sectionHead}`}>Support</div>
        <div className={styles.support}>
          <p className={styles.supportText}>
            Sheepshead AI is free to play and open source. Hosting costs real
            money. Help support the project if you're so inclined!
          </p>
          {PATREON_URL ? (
            <a
              href={PATREON_URL}
              className={styles.supportButton}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className={styles.heart} aria-hidden="true">
                ♥
              </span>
              Support on Patreon ↗
            </a>
          ) : (
            <span
              className={`${styles.supportButton} ${styles.supportPending}`}
            >
              <span className={styles.heart} aria-hidden="true">
                ♥
              </span>
              Patreon page coming soon
            </span>
          )}
        </div>
      </section>
    </PageShell>
  );
}
