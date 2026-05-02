import React, { useState, useEffect, useRef } from 'react';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { useIsMobile } from '../../hooks/useIsMobile';

interface TypewriterTextProps {
  text: string;
  speed?: number;
  className?: string;
  startOnView?: boolean;
}

/**
 * Types out text character by character with a blinking cursor.
 * Starts when the component enters the viewport (or immediately).
 * Shows full text instantly when reduced-motion is preferred.
 */
export const TypewriterText: React.FC<TypewriterTextProps> = ({
  text,
  speed = 40,
  className,
  startOnView = true,
}) => {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);
  const [ref, inView] = useInView<HTMLSpanElement>({ threshold: 0.3 });
  const reduced = useReducedMotion();
  const isMobile = useIsMobile();
  const started = useRef(false);

  const actualSpeed = isMobile ? speed * 0.5 : speed; // faster on mobile

  useEffect(() => {
    if (reduced) { setDisplayed(text); setDone(true); return; }
    if (startOnView && !inView) return;
    if (started.current) return;
    started.current = true;

    let i = 0;
    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(interval);
        setDone(true);
      }
    }, actualSpeed);

    return () => clearInterval(interval);
  }, [text, actualSpeed, inView, startOnView, reduced]);

  return (
    <span ref={ref} className={className}>
      {displayed}
      {!done && (
        <span className="inline-block w-[2px] h-[1em] bg-current ml-[1px] animate-pulse align-text-bottom" />
      )}
      {done && (
        <span className="inline-block w-[2px] h-[1em] bg-current ml-[1px] opacity-0 animate-[pulse_1s_ease-in-out_infinite] align-text-bottom" />
      )}
    </span>
  );
};
