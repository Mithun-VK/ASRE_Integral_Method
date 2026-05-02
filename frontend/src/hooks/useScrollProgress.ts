import { useState, useEffect, useRef } from 'react';

/**
 * Returns page scroll progress as a 0-1 value.
 * Throttled to ~60fps using rAF to avoid layout thrashing.
 */
export function useScrollProgress(): number {
  const [progress, setProgress] = useState(0);
  const rafId = useRef(0);

  useEffect(() => {
    let ticking = false;

    const update = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(docHeight > 0 ? Math.min(scrollTop / docHeight, 1) : 0);
      ticking = false;
    };

    const onScroll = () => {
      if (!ticking) {
        ticking = true;
        rafId.current = requestAnimationFrame(update);
      }
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    update(); // initial

    return () => {
      window.removeEventListener('scroll', onScroll);
      cancelAnimationFrame(rafId.current);
    };
  }, []);

  return progress;
}
