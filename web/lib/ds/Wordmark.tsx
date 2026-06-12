import React from 'react';
import MiniCardMark from './MiniCardMark';

type WordmarkSize = 'sm' | 'md' | 'lg' | 'xl';

const SIZES: Record<WordmarkSize, { wm: number; ai: number; mark: number; gap: number }> = {
  sm: { wm: 22, ai: 11, mark: 24, gap: 8 },
  md: { wm: 30, ai: 14, mark: 32, gap: 10 },
  lg: { wm: 64, ai: 26, mark: 64, gap: 16 },
  xl: { wm: 120, ai: 44, mark: 112, gap: 26 },
};

interface WordmarkProps {
  size?: WordmarkSize;
  mark?: boolean;
}

/** "Sheepshead AI" lockup with the optional fanned-card brand mark. */
export default function Wordmark({ size = 'md', mark = true }: WordmarkProps) {
  const s = SIZES[size] ?? SIZES.md;
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: s.gap }}>
      {mark && <MiniCardMark h={s.mark} />}
      <div style={{ display: 'inline-flex', alignItems: 'baseline', gap: s.gap * 0.5 }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: s.wm, color: 'var(--ink)' }}>Sheepshead</span>
        <span style={{ fontFamily: 'var(--font-ui)', fontSize: s.ai, letterSpacing: '0.22em', color: 'var(--muted)', fontWeight: 500 }}>AI</span>
      </div>
    </div>
  );
}
