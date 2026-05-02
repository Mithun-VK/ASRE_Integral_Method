import React from 'react';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { useScrollProgress } from '../../hooks/useScrollProgress';

/**
 * Thin 2px progress bar fixed at top of viewport.
 * Uses CSS scroll-driven animation where supported,
 * falls back to JS-based scroll tracking.
 */
export const ScrollProgressBar: React.FC = () => {
  const reduced = useReducedMotion();
  const progress = useScrollProgress();

  if (reduced) return null;

  // Check if browser supports scroll-driven animations
  const supportsScrollTimeline =
    typeof CSS !== 'undefined' &&
    CSS.supports('animation-timeline', 'scroll()');

  if (supportsScrollTimeline) {
    return (
      <div
        className="scroll-progress-bar"
        aria-hidden="true"
      />
    );
  }

  // JS fallback
  return (
    <div
      className="fixed top-0 left-0 right-0 h-[2px] z-[9999] pointer-events-none"
      aria-hidden="true"
    >
      <div
        className="h-full bg-gradient-to-r from-ark-red to-red-400"
        style={{
          transform: `scaleX(${progress})`,
          transformOrigin: 'left center',
          transition: 'transform 0.1s linear',
        }}
      />
    </div>
  );
};
