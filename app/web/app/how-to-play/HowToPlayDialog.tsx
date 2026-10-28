"use client";

import React from "react";
import { useRouter } from "next/navigation";
import HowToPlay from "./HowToPlay";
import ModalSheet from "./ModalSheet";

/** The lesson as a modal over whatever page linked to it. Rendered by the
 * intercepting route, so the URL reads /how-to-play while it is open and
 * closing it is a step back in history. */
export default function HowToPlayDialog() {
  const router = useRouter();
  return (
    <ModalSheet
      overline="The rules in three minutes"
      title="How to play"
      onClose={() => router.back()}
    >
      <HowToPlay />
    </ModalSheet>
  );
}
