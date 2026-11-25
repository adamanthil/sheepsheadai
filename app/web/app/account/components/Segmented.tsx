import React from "react";
import styles from "./Segmented.module.css";

/** A row of buttons, one pressed: a switch between a page's views, or a
 * small filter. */
export default function Segmented<T extends string | number>({
  label,
  options,
  value,
  onChange,
  size = "regular",
  className,
}: {
  /** What the choice is between, for screen readers. */
  label: string;
  options: { value: T; label: React.ReactNode }[];
  value: T;
  onChange: (value: T) => void;
  /** dense: a filter tucked beside a chart. regular: a filter over a
   * list. large: the switch between a page's views. */
  size?: "dense" | "regular" | "large";
  className?: string;
}) {
  return (
    <div
      className={`${styles.seg} ${styles[size]} ${className ?? ""}`}
      role="group"
      aria-label={label}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
