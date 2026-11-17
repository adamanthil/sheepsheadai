"use client";

import React from "react";
import { ds } from "../../lib/ds";
import { PATREON_URL } from "../../lib/links";
import PageShell from "../components/PageShell";
import styles from "./about.module.css";

const ARTICLE_URL =
  "https://bender.io/2026/11/17/building-an-expert-sheepshead-ai-with-reinforcement-learning/";
const FEEDBACK_EMAIL = "feedback@sheepshead.ai";

export default function AboutPage() {
  return (
    <PageShell title="About" wide>
      <div className={ds.headRule} />

      <section className={styles.section}>
        <p>
          Sheepshead AI is a passion project created by Andrew Bender, a
          software engineer from Milwaukee.
        </p>
        <p>
          It was built as a showcase for a reinforcement learning project to
          train a neural network to be an expert Sheepshead player. The AI was
          trained by playing copies of itself with only the rules of the game.
          All strategy, such as leading trump as the partner and called suit as
          a defender, emerged entirely because it is the best way to win. You
          can read about the details of the project here:
        </p>
        <a
          href={ARTICLE_URL}
          className={styles.article}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className={`${ds.overline} ${styles.articleKicker}`}>
            How the system was built
          </span>
          <span className={styles.articleTitle}>
            Building an Expert Sheepshead AI with Reinforcement Learning
          </span>
          <span className={styles.articleCta}>
            Read the article <span aria-hidden="true">↗</span>
          </span>
        </a>
        <p>
          I hope you enjoy playing Sheepshead on the site. One of the goals is
          for the AI to be good enough that it&apos;s genuinely fun to play 5
          handed solo. I also hope to learn how the AI&apos;s skill compares to
          real people, and the only way to find that out is with real people
          playing games here. If you like it, please share it with your friends!
        </p>
        <p>
          Feel free to send any comments to{" "}
          <a href={`mailto:${FEEDBACK_EMAIL}`}>{FEEDBACK_EMAIL}</a>.
        </p>
      </section>

      <section id="support" className={styles.section}>
        <h2 className={`${ds.overline} ${styles.sectionHead}`}>Support</h2>
        <div className={styles.support}>
          <p>
            Sheepshead AI is free to play and open source. Hosting costs real
            money. If you&apos;re so inclined, please support the project and
            ensure it stays online!
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
