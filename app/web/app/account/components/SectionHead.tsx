import React from "react";
import styles from "./SectionHead.module.css";

/** The head of a section in the AI review and admin pages: a double rule
 * with a small-caps title on the left and a quieter aside on the right.
 * The space under it is --head-gap, for a caller's class to set. */
export default function SectionHead({
  title,
  aside,
  className,
}: {
  title: React.ReactNode;
  aside?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`${styles.head} ${className ?? ""}`}>
      <span>{title}</span>
      {aside != null && <span>{aside}</span>}
    </div>
  );
}
