import React, { useRef, useState, useCallback } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';
import clsx from 'clsx';
import { useIsMobile } from '../../hooks/useIsMobile';
import { useReducedMotion } from '../../hooks/useReducedMotion';

interface HolographicCardProps {
  children: React.ReactNode;
  className?: string;
  tiltIntensity?: number;
  glowColor?: string;
  borderSpin?: boolean;
}

export const HolographicCard: React.FC<HolographicCardProps> = ({
  children,
  className,
  tiltIntensity = 10,
  glowColor = 'rgba(220, 38, 38, 0.18)',
  borderSpin = true,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const isMobile = useIsMobile();
  const reduced = useReducedMotion();

  // Spring-smoothed raw mouse offsets (0 when not hovering)
  const rawX = useSpring(0, { stiffness: 180, damping: 22 });
  const rawY = useSpring(0, { stiffness: 180, damping: 22 });

  const rotateX = useTransform(rawY, [-0.5, 0.5], [tiltIntensity, -tiltIntensity]);
  const rotateY = useTransform(rawX, [-0.5, 0.5], [-tiltIntensity, tiltIntensity]);

  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [hovering, setHovering] = useState(false);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    rawX.set(x);
    rawY.set(y);
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, [rawX, rawY]);

  const onMouseEnter = useCallback(() => setHovering(true), []);

  const onMouseLeave = useCallback(() => {
    rawX.set(0);
    rawY.set(0);
    setHovering(false);
  }, [rawX, rawY]);

  // Static fallback for mobile / reduced motion
  if (isMobile || reduced) {
    return (
      <div className={clsx('relative rounded-2xl', className)}>
        {children}
      </div>
    );
  }

  return (
    <motion.div
      ref={cardRef}
      className={clsx(
        'relative rounded-2xl',
        borderSpin && 'holographic-border',
        className,
      )}
      style={{
        rotateX,
        rotateY,
        transformStyle: 'preserve-3d',
        transformPerspective: 900,
      }}
      onMouseMove={onMouseMove}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      // Lift on hover
      whileHover={{ scale: 1.02, zIndex: 2 }}
      transition={{ scale: { duration: 0.25, ease: [0.16, 1, 0.3, 1] as const } }}
    >
      {/* Mouse-follow spotlight glow */}
      <motion.div
        className="absolute inset-0 rounded-2xl pointer-events-none"
        animate={{ opacity: hovering ? 1 : 0 }}
        transition={{ duration: 0.25 }}
        style={{
          background: hovering
            ? `radial-gradient(320px circle at ${mousePos.x}px ${mousePos.y}px, ${glowColor}, transparent 65%)`
            : 'none',
        }}
        aria-hidden="true"
      />

      {/* Holographic shimmer layer — follows mouse X as shine angle */}
      <motion.div
        className="absolute inset-0 rounded-2xl pointer-events-none overflow-hidden"
        animate={{ opacity: hovering ? 0.5 : 0 }}
        transition={{ duration: 0.3 }}
        aria-hidden="true"
      >
        <div
          className="absolute inset-0"
          style={{
            background: `linear-gradient(
              ${105 + (mousePos.x / (cardRef.current?.offsetWidth || 1)) * 30}deg,
              transparent 30%,
              rgba(255,255,255,0.06) 50%,
              transparent 70%
            )`,
          }}
        />
      </motion.div>

      <div className="relative z-10" style={{ transform: 'translateZ(20px)' }}>
        {children}
      </div>
    </motion.div>
  );
};
