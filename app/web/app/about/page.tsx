"use client";

import React from "react";
import { ds } from "../../lib/ds";
import { PATREON_URL } from "../../lib/links";
import PageShell from "../components/PageShell";
import styles from "./about.module.css";

export default function AboutPage() {
  return (
    <PageShell title="About" wide>
      <section id="support" className={styles.section}>
        <div className={ds.headRule} />
        <p>
          Sheepshead AI is a passion project created by Andrew Bender, a
          software engineer from Milwaukee. It was built as a showcase for a
          reinforcement learning project to train a neural network to be an
          expert Sheepshead player. The AI was trained by playing copies of
          itself with only the rules of the game. All strategy, such as leading
          trump as the partner and called suit as a defender, emerged entirely
          because it is the best way to win. You can read about the details of
          the project here: (Coming soon)
        </p>
        <p>
          I hope you enjoy playing Sheepshead on the site. One of the goals is
          to learn how the AI's skill compares to real people, and the only way
          to find that out is with real people playing games here! If you like
          it, please share it with your friends.
        </p>
        <p>Feel free to send any comments to feedback@sheepshead.ai.</p>
        <div className={`${ds.overline} ${styles.sectionHead}`}>Support</div>
        <div className={styles.support}>
          <p className={styles.supportText}>
            Sheepshead AI is free to play and open source. Hosting costs real
            money. If you're so inclined, please support the project and ensure
            it stays online!
          </p>
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
        </div>
      </section>
    </PageShell>
  );
}
