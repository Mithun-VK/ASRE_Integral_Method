import React, { useEffect, useState, useRef } from 'react';
import clsx from 'clsx';

interface ScoreBarProps {
  label: string;
  value: number;         // 0-100
  color?: 'red' | 'green' | 'amber' | 'blue';
  suffix?: string;       // e.g., "%" or " pts"
  description?: string;  // extra text below the bar
  delay?: number;        // ms delay before animation starts
  animate?: boolean;     // trigger animation
}

const colorMap = {
  red: {
    bar: 'bg-gradient-to-r from-red-700 via-ark-red to-red-400',
    text: 'text-ark-red',
    glow: 'shadow-ark-red/30',
    bg: 'bg-red-950/40',
  },
  green: {
    bar: 'bg-gradient-to-r from-green-700 via-green-500 to-green-400',
    text: 'text-green-400',
    glow: 'shadow-green-500/30',
    bg: 'bg-green-950/40',
  },
  amber: {
    bar: 'bg-gradient-to-r from-amber-700 via-ark-gold to-amber-400',
    text: 'text-ark-gold',
    glow: 'shadow-ark-gold/30',
    bg: 'bg-amber-950/40',
  },
  blue: {
    bar: 'bg-gradient-to-r from-blue-700 via-blue-500 to-blue-400',
    text: 'text-blue-400',
    glow: 'shadow-blue-500/30',
    bg: 'bg-blue-950/40',
  },
};

function getTier(v: number): string {
  if (v >= 75) return 'A-tier';
  if (v >= 60) return 'B-tier';
  if (v >= 45) return 'C-tier';
  return 'D-tier';
}

export const ScoreBar: React.FC<ScoreBarProps> = ({
  label,
  value,
  color = 'red',
  suffix = '%',
  description,
  delay = 0,
  animate = true,
}) => {
  const [displayValue, setDisplayValue] = useState(0);
  const [barWidth, setBarWidth] = useState(0);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    if (!animate) {
      setDisplayValue(value);
      setBarWidth(value);
      return;
    }

    const timer = setTimeout(() => {
      const duration = 1500; // ms
      const startTime = performance.now();

      const tick = (now: number) => {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);

        setDisplayValue(Math.round(value * eased * 10) / 10);
        setBarWidth(value * eased);

        if (progress < 1) {
          frameRef.current = requestAnimationFrame(tick);
        }
      };

      frameRef.current = requestAnimationFrame(tick);
    }, delay);

    return () => {
      clearTimeout(timer);
      cancelAnimationFrame(frameRef.current);
    };
  }, [value, delay, animate]);

  const colors = colorMap[color];
  const tier = getTier(value);

  return (
    <div className="space-y-1.5 sm:space-y-2">
      {/* Label row — wraps on very small screens */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 flex-wrap flex-1 min-w-0">
          <span className="text-xs sm:text-sm font-semibold text-gray-200 whitespace-nowrap">{label}</span>
          <span className={clsx(
            'text-[10px] font-mono px-1.5 py-0.5 rounded flex-shrink-0',
            'bg-white/5 border border-white/10',
            colors.text
          )}>
            {tier}
          </span>
        </div>
        <span className={clsx('text-base sm:text-lg font-bold font-mono tabular-nums flex-shrink-0', colors.text)}>
          {displayValue.toFixed(1)}{suffix}
        </span>
      </div>

      <div className={clsx('h-2.5 sm:h-3 rounded-full overflow-hidden', colors.bg)}>
        <div
          className={clsx(
            'h-full rounded-full transition-none',
            colors.bar,
            barWidth > 0 && `shadow-lg ${colors.glow}`
          )}
          style={{ width: `${barWidth}%` }}
        />
      </div>

      {description && (
        <p className="text-xs text-gray-500 font-mono">{description}</p>
      )}
    </div>
  );
};
