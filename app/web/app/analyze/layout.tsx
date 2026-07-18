"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import CardEmbeddingsModal from "./CardEmbeddingsModal";
import styles from "./layout.module.css";

const TABS = [
  { href: "/analyze", label: "Game Analysis" },
  { href: "/analyze/pick", label: "Pick Decisions" },
];

/** Shared shell for the analysis tools: scoped palette, masthead, and the
 * tab nav that switches between the full-game and pick-scenario views. */
export default function AnalyzeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [embeddingsOpen, setEmbeddingsOpen] = useState(false);

  return (
    <div className={styles.shell}>
      <header className={styles.masthead}>
        <div className={styles.mastheadInner}>
          <div className={styles.mastheadTop}>
            <Link href="/" className={styles.homeLink}>
              ← Sheepshead
            </Link>
            <h1 className={styles.title}>AI Model Analysis</h1>
            {/* Model-level view, independent of any simulated game. */}
            <span className={styles.mastheadActions}>
              <button
                type="button"
                className={styles.embeddingsButton}
                onClick={() => setEmbeddingsOpen(true)}
              >
                Card embeddings
              </button>
            </span>
          </div>
          <nav className={styles.tabs} aria-label="Analysis tools">
            {TABS.map((tab) => (
              <Link
                key={tab.href}
                href={tab.href}
                className={
                  pathname === tab.href
                    ? `${styles.tab} ${styles.tabActive}`
                    : styles.tab
                }
                aria-current={pathname === tab.href ? "page" : undefined}
              >
                {tab.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>
      <div className={styles.content}>{children}</div>

      <CardEmbeddingsModal
        open={embeddingsOpen}
        onClose={() => setEmbeddingsOpen(false)}
      />
    </div>
  );
}
