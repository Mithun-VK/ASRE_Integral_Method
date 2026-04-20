import React from 'react';
import { twMerge } from 'tailwind-merge';
import clsx from 'clsx';

interface CTAButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'gold';
  size?: 'sm' | 'md' | 'lg';
  href?: string;
  onClick?: () => void;
  className?: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  type?: 'button' | 'submit';
  id?: string;
}

const variants = {
  primary: 'bg-ark-red hover:bg-red-700 text-white shadow-lg shadow-ark-red/25 hover:shadow-ark-red/40',
  secondary: 'bg-white/[0.06] hover:bg-white/[0.1] text-white border border-white/10 hover:border-ark-red/40',
  ghost: 'bg-transparent hover:bg-white/[0.05] text-gray-300 hover:text-white',
  gold: 'bg-ark-gold hover:bg-yellow-600 text-black font-semibold shadow-lg shadow-ark-gold/25 hover:shadow-ark-gold/40',
};

const sizes = {
  sm: 'px-4 py-2 text-sm rounded-lg',
  md: 'px-5 py-2.5 text-sm sm:text-base rounded-xl',
  lg: 'px-6 py-3.5 sm:px-8 sm:py-4 text-base sm:text-lg rounded-xl',
};

export const CTAButton: React.FC<CTAButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  href,
  onClick,
  className,
  icon,
  disabled = false,
  type = 'button',
  id,
}) => {
  const baseClasses = clsx(
    'inline-flex items-center justify-center gap-2 font-semibold',
    'transition-all duration-300 ease-out',
    // Remove 300ms tap delay on mobile
    '[touch-action:manipulation]',
    'hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.98]',
    'focus:outline-none focus:ring-2 focus:ring-ark-red/50 focus:ring-offset-2 focus:ring-offset-ark-bg-primary',
    'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0',
    variants[variant],
    sizes[size],
  );

  const merged = twMerge(baseClasses, className);

  if (href) {
    return (
      <a href={href} className={merged} id={id} target={href.startsWith('http') ? '_blank' : undefined} rel={href.startsWith('http') ? 'noopener noreferrer' : undefined}>
        {icon && <span className="flex-shrink-0">{icon}</span>}
        {children}
      </a>
    );
  }

  return (
    <button type={type} onClick={onClick} className={merged} disabled={disabled} id={id}>
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </button>
  );
};
