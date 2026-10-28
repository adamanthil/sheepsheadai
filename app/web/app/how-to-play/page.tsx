"use client";

import PageShell from "../components/PageShell";
import HowToPlay from "./HowToPlay";

/** The lesson as a full page, for direct visits and shared links; links
 * inside the app open it as a modal instead (app/@modal/(.)how-to-play). */
export default function HowToPlayPage() {
  return (
    <PageShell title="How to play" wide>
      <HowToPlay />
    </PageShell>
  );
}
