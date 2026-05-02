import React, { useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { twMerge } from 'tailwind-merge';
import clsx from 'clsx';
import { useReducedMotion } from '../../hooks/useReducedMotion';

interface CTAButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'gold';
  size?: 'sm' | 'md' | 'lg';
  href?: string;
  onClick?: () => void;
  className?: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  loading?: boolean;
  type?: 'button' | 'submit';
  id?: string;
}

const variants = {
  primary: 'bg-ark-red text-white shadow-lg shadow-ark-red/25',
  secondary: 'bg-white/[0.06] text-white border border-white/10',
  ghost: 'bg-transparent text-gray-300',
  gold: 'bg-ark-gold text-black font-semibold shadow-lg shadow-ark-gold/25',
};

const hoverVariants = {
  primary: 'hover:bg-red-700 hover:shadow-ark-red/40',
  secondary: 'hover:bg-white/[0.1] hover:border-ark-red/40',
  ghost: 'hover:bg-white/[0.05] hover:text-white',
  gold: 'hover:bg-yellow-600 hover:shadow-ark-gold/40',
};

const sizes = {
  sm: 'px-4 py-2 text-sm rounded-lg',
  md: 'px-5 py-2.5 text-sm sm:text-base rounded-xl',
  lg: 'px-6 py-3.5 sm:px-8 sm:py-4 text-base sm:text-lg rounded-xl',
};

interface Ripple {
  id: number;
  x: number;
  y: number;
}

const SPRING = [0.16, 1, 0.3, 1] as const;

export const CTAButton: React.FC<CTAButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  href,
  onClick,
  className,
  icon,
  disabled = false,
  loading = false,
  type = 'button',
  id,
}) => {
  const reduced = useReducedMotion();
  const btnRef = useRef<HTMLButtonElement>(null);
  const [ripples, setRipples] = React.useState<Ripple[]>([]);

  const addRipple = useCallback((e: React.MouseEvent) => {
    if (reduced || disabled) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const id = Date.now();
    setRipples(prev => [...prev, { id, x, y }]);
    setTimeout(() => setRipples(prev => prev.filter(r => r.id !== id)), 650);
  }, [reduced, disabled]);

  const baseClasses = clsx(
    'relative inline-flex items-center justify-center gap-2 font-semibold overflow-hidden',
    '[touch-action:manipulation]',
    'focus:outline-none focus:ring-2 focus:ring-ark-red/50 focus:ring-offset-2 focus:ring-offset-ark-bg-primary',
    'disabled:opacity-50 disabled:cursor-not-allowed',
    'transition-colors duration-200',
    variants[variant],
    hoverVariants[variant],
    sizes[size],
  );

  const merged = twMerge(baseClasses, className);

  const motionProps = reduced ? {} : {
    whileHover: disabled ? {} : { y: -2, scale: 1.01 },
    whileTap: disabled ? {} : { scale: 0.96, y: 0 },
    transition: { duration: 0.18, ease: SPRING },
  };

  // Shimmer sweep only for primary
  const shimmer = variant === 'primary' && !reduced && !disabled && (
    <span
      className="absolute inset-0 pointer-events-none"
      style={{
        background: 'linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.10) 50%, transparent 60%)',
        backgroundSize: '200% 100%',
        animation: 'shimmerSweep 3s ease-in-out infinite',
      }}
      aria-hidden="true"
    />
  );

  const content = (
    <>
      {shimmer}

      {/* Ripple container */}
      <AnimatePresence>
        {ripples.map(r => (
          <motion.span
            key={r.id}
            className="absolute rounded-full bg-white/20 pointer-events-none"
            style={{ left: r.x - 30, top: r.y - 30, width: 60, height: 60 }}
            initial={{ scale: 0, opacity: 0.5 }}
            animate={{ scale: 4, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        ))}
      </AnimatePresence>

      {/* Loading spinner */}
      {loading ? (
        <span className="flex items-center justify-center gap-2">
          <span className="brand-spinner" aria-hidden="true" />
          <span>Loading…</span>
        </span>
      ) : (
        <>
          {icon && <span className="flex-shrink-0 relative z-10">{icon}</span>}
          <span className="relative z-10">{children}</span>
        </>
      )}
    </>
  );

  if (href) {
    return (
      <motion.a
        href={href}
        className={merged}
        id={id}
        target={href.startsWith('http') ? '_blank' : undefined}
        rel={href.startsWith('http') ? 'noopener noreferrer' : undefined}
        {...motionProps}
        onClick={addRipple as any}
      >
        {content}
      </motion.a>
    );
  }

  return (
    <motion.button
      ref={btnRef}
      type={type}
      onClick={(e) => { addRipple(e); onClick?.(); }}
      className={merged}
      disabled={disabled || loading}
      id={id}
      {...motionProps}
    >
      {content}
    </motion.button>
  );
};
