import React, { useEffect, useRef, useState } from 'react';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { useIsMobile } from '../../hooks/useIsMobile';

interface AnimatedCounterProps {
  target: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  className?: string;
}

/**
 * Counts up from 0 to `target` when the component enters the viewport.
 * Uses rAF with ease-out cubic for smooth animation.
 * Shows final value immediately if reduced-motion or already counted.
 */
export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  target,
  duration = 2000,
  prefix = '',
  suffix = '',
  decimals = 0,
  className,
}) => {
  const [display, setDisplay] = useState(0);
  const [ref, inView] = useInView<HTMLSpanElement>({ threshold: 0.3 });
  const reduced = useReducedMotion();
  const isMobile = useIsMobile();
  const hasAnimated = useRef(false);

  const actualDuration = isMobile ? duration * 0.6 : duration;

  useEffect(() => {
    if (reduced) { setDisplay(target); return; }
    if (!inView || hasAnimated.current) return;
    hasAnimated.current = true;

    const startTime = performance.now();
    let rafId: number;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / actualDuration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(target * eased);

      if (progress < 1) {
        rafId = requestAnimationFrame(tick);
      }
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [inView, target, actualDuration, reduced]);

  const formatted = new Intl.NumberFormat('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(display);

  return (
    <span ref={ref} className={className}>
      {prefix}{formatted}{suffix}
    </span>
  );
};
