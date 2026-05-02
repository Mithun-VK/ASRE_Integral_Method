import { useEffect, useState, useCallback, type RefObject } from 'react';

interface MousePosition {
  x: number;
  y: number;
  /** -1 to 1 range relative to element center */
  normalizedX: number;
  /** -1 to 1 range relative to element center */
  normalizedY: number;
}

const INITIAL: MousePosition = { x: 0, y: 0, normalizedX: 0, normalizedY: 0 };

function isTouchDevice(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(hover: none)').matches;
}

/**
 * Tracks mouse position relative to a target element.
 * Returns normalized coordinates (-1 to 1) for tilt/magnetic effects.
 * Automatically disabled on touch devices.
 */
export function useMousePosition<T extends HTMLElement>(
  ref: RefObject<T | null>
): MousePosition {
  const [pos, setPos] = useState<MousePosition>(INITIAL);

  const handleMove = useCallback((e: MouseEvent) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const normalizedX = (x / rect.width) * 2 - 1;
    const normalizedY = (y / rect.height) * 2 - 1;
    setPos({ x, y, normalizedX, normalizedY });
  }, [ref]);

  const handleLeave = useCallback(() => {
    setPos(INITIAL);
  }, []);

  useEffect(() => {
    if (isTouchDevice()) return;
    const el = ref.current;
    if (!el) return;

    el.addEventListener('mousemove', handleMove);
    el.addEventListener('mouseleave', handleLeave);
    return () => {
      el.removeEventListener('mousemove', handleMove);
      el.removeEventListener('mouseleave', handleLeave);
    };
  }, [ref, handleMove, handleLeave]);

  return pos;
}
