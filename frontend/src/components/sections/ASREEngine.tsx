import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

const SCORES = [
  {
    label: 'F-Score',
    fullLabel: 'Fundamental',
    measures: 'Revenue quality, debt coverage, earnings consistency',
    example: 'RELIANCE: 74.2% A-tier',
    color: 'from-emerald-600 to-emerald-400',
    textColor: 'text-emerald-400',
    bgColor: 'bg-emerald-950/40',
    value: 74.2,
  },
  {
    label: 'T-Score',
    fullLabel: 'Technical',
    measures: 'Price action, volume profile, RSI divergence',
    example: '545 walk-forward iterations',
    color: 'from-blue-600 to-blue-400',
    textColor: 'text-blue-400',
    bgColor: 'bg-blue-950/40',
    value: 61.8,
  },
  {
    label: 'M-Score',
    fullLabel: 'Momentum',
    measures: 'Trend persistence, sector rotation, dip detection',
    example: 'SUZLON: 48.1 MID dip',
    color: 'from-amber-600 to-amber-400',
    textColor: 'text-amber-400',
    bgColor: 'bg-amber-950/40',
    value: 48.1,
  },
];

export const ASREEngine: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.15 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} id="asre-engine" className="section-padding relative">
      {/* Subtle bg glow */}
      <div className="absolute top-1/2 left-0 w-[500px] h-[500px] bg-ark-red/[0.03] rounded-full blur-[150px] -translate-y-1/2" />

      <div className="section-container relative">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-gold border border-ark-gold/20 rounded-full mb-4 uppercase tracking-wider">
            The Engine
          </span>
          <h2 className="section-title text-white mb-4">
            ASRE <span className="gradient-text-gold">Scoring Engine</span>
          </h2>
          <p className="section-subtitle mx-auto">
            Three orthogonal scores. One composite signal. Every score hash-stamped before market open.
          </p>
        </div>

        {/* Score cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8 mb-12">
          {SCORES.map((score, i) => (
            <div
              key={score.label}
              className={clsx(
                'glass-card-hover p-6 lg:p-8 group',
                'opacity-0',
                visible && 'animate-slide-up',
              )}
              style={{ animationDelay: `${i * 0.15}s`, animationFillMode: 'forwards' }}
            >
              {/* Score header */}
              <div className="flex items-center justify-between mb-5">
                <div>
                  <span className={clsx('text-2xl font-bold font-mono', score.textColor)}>
                    {score.label}
                  </span>
                  <span className="block text-xs text-gray-500 mt-0.5">{score.fullLabel}</span>
                </div>
                <div className={clsx(
                  'text-3xl font-bold font-mono tabular-nums',
                  score.textColor,
                  'opacity-0',
                  visible && 'animate-fade-in'
                )} style={{ animationDelay: `${0.5 + i * 0.2}s`, animationFillMode: 'forwards' }}>
                  {score.value}%
                </div>
              </div>

              {/* Progress bar */}
              <div className={clsx('h-2 rounded-full overflow-hidden mb-5', score.bgColor)}>
                <div
                  className={clsx('h-full rounded-full bg-gradient-to-r transition-all duration-[1500ms] ease-out', score.color)}
                  style={{ width: visible ? `${score.value}%` : '0%', transitionDelay: `${0.3 + i * 0.15}s` }}
                />
              </div>

              {/* Measures */}
              <div className="space-y-3">
                <div>
                  <span className="text-[10px] text-gray-600 uppercase tracking-wider">Measures</span>
                  <p className="text-sm text-gray-300 mt-1">{score.measures}</p>
                </div>
                <div>
                  <span className="text-[10px] text-gray-600 uppercase tracking-wider">Example</span>
                  <p className={clsx('text-sm font-mono mt-1', score.textColor)}>{score.example}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* AUC Badge */}
        <div className={clsx(
          'flex items-center justify-center opacity-0',
          visible && 'animate-fade-in'
        )} style={{ animationDelay: '0.8s', animationFillMode: 'forwards' }}>
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-white/[0.02] border border-white/[0.06]">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-ark-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
              <span className="text-sm font-bold text-ark-gold">AUC 0.60</span>
            </div>
            <span className="text-xs text-gray-500">—</span>
            <span className="text-sm text-gray-400">Honest signal. No overfitting. Walk-forward validated.</span>
          </div>
        </div>
      </div>
    </section>
  );
};
