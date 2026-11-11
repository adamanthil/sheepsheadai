import React from "react";
import styles from "./TypingDots.module.css";

export function TypingDots() {
  return (
    <span className={styles.dots} aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
}

/** The dots in a speech bubble, for the corner of a seat's avatar. */
export function TypingBubble({ name }: { name?: string }) {
  return (
    <span
      className={styles.bubble}
      role="img"
      aria-label={name ? `${name} is typing` : "Typing"}
    >
      <TypingDots />
    </span>
  );
}

/** Who is typing, in words: "Kyle is typing", "Kyle and Ann are typing",
 * "3 people are typing". Empty when nobody is. */
export function typingText(names: string[]): string {
  if (names.length === 0) return "";
  if (names.length === 1) return `${names[0]} is typing`;
  if (names.length === 2) return `${names[0]} and ${names[1]} are typing`;
  return `${names.length} people are typing`;
}

/** The dots and who is typing, as a line; nothing when nobody is. */
export function TypingNote({
  names,
  className,
}: {
  names: string[];
  className?: string;
}) {
  if (names.length === 0) return null;
  return (
    <span
      className={`${styles.note} ${className ?? ""}`}
      role="status"
      aria-live="polite"
    >
      <TypingDots />
      {typingText(names)}
    </span>
  );
}
