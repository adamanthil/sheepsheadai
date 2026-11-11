"use client";

import PageShell from "../components/PageShell";
import { QuickReferenceBody } from "../how-to-play/QuickReference";

/** The table's quick reference on a page of its own, to keep open beside a
 * game played elsewhere or to share. */
export default function HelpPage() {
  return (
    <PageShell title="Sheepshead Reference" wide>
      <QuickReferenceBody />
    </PageShell>
  );
}
