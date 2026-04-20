import React from 'react';
import clsx from 'clsx';

interface DipBadgeProps {
  dip: 'DEEP' | 'MID' | 'LATE' | 'NONE';
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
}

const dipConfig = {
  DEEP: {
    label: 'DEEP DIP',
    color: 'text-green-300',
    bg: 'bg-green-950/60 border-green-500/30',
    dot: 'bg-green-400',
    glow: 'animate-pulse-glow-green',
    description: 'Strong entry signal',
  },
  MID: {
    label: 'MID DIP',
    color: 'text-amber-300',
    bg: 'bg-amber-950/60 border-amber-500/30',
    dot: 'bg-amber-400',
    glow: 'animate-pulse-glow-amber',
    description: 'Moderate opportunity',
  },
  LATE: {
    label: 'LATE DIP',
    color: 'text-red-300',
    bg: 'bg-red-950/60 border-red-500/30',
    dot: 'bg-red-400',
    glow: 'animate-pulse-glow',
    description: 'Extended recovery',
  },
  NONE: {
    label: 'NO DIP',
    color: 'text-gray-400',
    bg: 'bg-gray-800/60 border-gray-600/30',
    dot: 'bg-gray-500',
    glow: '',
    description: 'Neutral trend',
  },
};

const sizeMap = {
  sm: 'px-2.5 py-1 text-xs',
  md: 'px-3.5 py-1.5 text-sm',
  lg: 'px-4 py-2 text-base',
};

export const DipBadge: React.FC<DipBadgeProps> = ({
  dip,
  size = 'md',
  pulse = true,
}) => {
  const config = dipConfig[dip];

  return (
    <div className="inline-flex flex-col items-center gap-1">
      <div
        className={clsx(
          'inline-flex items-center gap-2 font-mono font-bold rounded-full border',
          'transition-all duration-300',
          config.bg,
          config.color,
          sizeMap[size],
          pulse && config.glow,
        )}
      >
        <span className={clsx('w-2 h-2 rounded-full', config.dot, pulse && 'animate-pulse')} />
        {config.label}
      </div>
      <span className="text-[10px] text-gray-500 font-mono">{config.description}</span>
    </div>
  );
};
