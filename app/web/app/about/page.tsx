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
        <p>
          Sheepshead AI is a passion project created by Andrew Bender, a
          software engineer from Milwaukee. It was built primarily as a showcase
          for a reinforcement learning project to train a smart and flexible
          neural network that is an expert Sheepshead player. You can read about
          the details of the project here: (insert blog link)
        </p>
        <p>
          I hope you enjoy playing Sheepshead on the site. I'm curious how well
          the AI agent's skill compares to real people, and the only way to find
          that out is with real people playing games here! If you like it,
          please share it with your friends.
        </p>
        <p>Feel free to send any comments to feedback@sheepshead.ai.</p>
        <div className={`${ds.overline} ${styles.sectionHead}`}>Support</div>
        <div className={styles.support}>
          <p className={styles.supportText}>
            Sheepshead AI is free to play and open source. Hosting costs real
            money. If you're so inclined, please support the project and ensure
            it stays online!
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
