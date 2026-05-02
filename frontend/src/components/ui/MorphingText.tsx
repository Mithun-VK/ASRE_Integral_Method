import React, { useEffect, useState, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

interface MorphingTextProps {
  words: string[];
  interval?: number;
  className?: string;
  pauseOnHidden?: boolean;
}

const SPRING = [0.16, 1, 0.3, 1] as const;

export const MorphingText: React.FC<MorphingTextProps> = ({
  words,
  interval = 3000,
  className,
  pauseOnHidden = true,
}) => {
  const [index, setIndex] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isPaused = useRef(false);

  const safeWords = Array.isArray(words) && words.length > 0 ? words : [''];

  // ── Interval ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (safeWords.length <= 1) return;
    intervalRef.current = setInterval(() => {
      if (!isPaused.current) {
        setIndex((prev) => (prev + 1) % safeWords.length);
      }
    }, interval);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [safeWords.length, interval]);

  // ── Pause on tab hidden ───────────────────────────────────────────────────
  useEffect(() => {
    if (!pauseOnHidden) return;
    const handler = () => { isPaused.current = document.hidden; };
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, [pauseOnHidden]);

  // ── Reduced motion ────────────────────────────────────────────────────────
  const prefersReduced =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (prefersReduced || safeWords.length <= 1) {
    return <span className={className}>{safeWords[0]}</span>;
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    /*
     * `inline-block` with `relative` positioning.
     *
     * The invisible spacer (aria-hidden) is `display: block` and sets
     * the container's natural width to the CURRENT word — not the longest.
     * This means the container shrinks and grows with each word swap,
     * and "Advisors" (the sibling) dynamically repositions alongside it.
     *
     * The animated word sits `absolute inset-0` so it doesn't
     * affect layout — only the spacer drives the width.
     */
    <span
      className="relative inline-block"
      aria-live="polite"
      aria-atomic="true"
    >
      {/* Width driver — current word, invisible, in normal flow */}
      <motion.span
        key={`spacer-${safeWords[index]}`}
        className={className}
        aria-hidden="true"
        // Animate width transition so it smoothly expands/contracts
        animate={{ opacity: 0 }}
        style={{ visibility: 'hidden', display: 'inline-block', pointerEvents: 'none' }}
        transition={{ duration: 0 }}
      >
        {safeWords[index]}
      </motion.span>

      {/* Animated word — floats above the spacer */}
      <span
        className="absolute inset-0 flex items-center justify-center"
        aria-hidden="true"
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={safeWords[index]}
            className={className}
            initial={{ opacity: 0, y: 16, filter: 'blur(8px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            exit={{ opacity: 0, y: -14, filter: 'blur(6px)' }}
            transition={{ duration: 0.42, ease: SPRING }}
            style={{ display: 'inline-block', whiteSpace: 'nowrap' }}
          >
            {safeWords[index]}
          </motion.span>
        </AnimatePresence>
      </span>

      {/* SR-only live region */}
      <span className="sr-only">{safeWords[index]}</span>
    </span>
  );
};
