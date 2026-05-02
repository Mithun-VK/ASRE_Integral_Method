import { useState, useEffect } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

/**
 * Respects the user's OS-level "reduce motion" preference.
 * Returns `true` when animations should be suppressed.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia(QUERY).matches;
  });

  useEffect(() => {
    const mql = window.matchMedia(QUERY);
    const handler = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  return reduced;
}

/**
 * Utility: returns `true` when it's safe to animate.
 * Use this to guard imperative GSAP calls.
 */
export function shouldAnimate(): boolean {
  if (typeof window === 'undefined') return false;
  return !window.matchMedia(QUERY).matches;
}
