"use client";

import SiteNav from "../components/SiteNav";
import { ds } from "../../lib/ds";
import {
  ReferenceStage,
  ReferenceTabs,
  useReferencePage,
} from "../how-to-play/QuickReference";
import styles from "./page.module.css";

/** The table's quick reference on a page of its own, to keep open on a
 * phone beside a game played in person. No masthead, and the Power /
 * Points switch shares the header row, so the cards start as high on the
 * screen as the site strip allows. */
export default function HelpPage() {
  const [page, setPage] = useReferencePage();
  return (
    <div className={styles.page}>
      <SiteNav />
      <main className={styles.inner}>
        <header className={styles.head}>
          <div>
            <p className={`${ds.overline} ${styles.overline}`}>
              Quick reference
            </p>
            <h1 className={styles.title}>Sheepshead</h1>
          </div>
          <ReferenceTabs page={page} onPage={setPage} className={styles.tabs} />
        </header>
        <ReferenceStage page={page} onPage={setPage} />
      </main>
    </div>
  );
}
