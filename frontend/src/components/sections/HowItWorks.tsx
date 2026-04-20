import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

const STEPS = [
  {
    step: '01',
    title: 'Enter Ticker',
    description: 'Type any NSE/BSE symbol. Our engine pulls latest price, volume, fundamentals, and sector data in real time.',
    detail: 'NSE/BSE native → 4-second latency',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    step: '02',
    title: 'ASRE Scores',
    description: 'The engine computes F-Score (fundamentals), T-Score (technical signals), and M-Score (momentum/dip context) using 545+ walk-forward validated iterations.',
    detail: 'F/T/M → 545+ iterations each',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    step: '03',
    title: 'Hash & Sign',
    description: 'Every score is SHA-256 hashed with a unique Run ID, timestamped before market open. Immutable, traceable, audit-ready.',
    detail: 'SHA-256 + Run ID → Immutable',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
      </svg>
    ),
  },
  {
    step: '04',
    title: 'PDF Report',
    description: 'A compliance-ready PDF with full F/T/M breakdown, SEBI AI disclosure, hash proof, and client-ready formatting. Download or share.',
    detail: 'SEBI compliant → Client-ready',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
];

export const HowItWorks: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.1 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} id="how-it-works" className="section-padding relative">
      <div className="section-container">
        {/* Section header */}
        <div className="text-center mb-10 sm:mb-16">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-red border border-ark-red/20 rounded-full mb-4 uppercase tracking-wider">
            How It Works
          </span>
          <h2 className="section-title text-white mb-4">
            Ticker to PDF in <span className="gradient-text">4 Seconds</span>
          </h2>
          <p className="section-subtitle mx-auto">
            From NSE/BSE symbol to a hash-signed, SEBI-compliant research report.
          </p>
        </div>

        {/* Steps — on mobile: vertical cards without the timeline line */}
        <div className="relative max-w-3xl mx-auto">
          {/* Vertical connector line — sm+ only */}
          <div className="absolute left-5 top-5 bottom-5 w-px bg-gradient-to-b from-ark-red/40 via-ark-red/20 to-transparent hidden sm:block" />

          <div className="space-y-4 sm:space-y-8">
            {STEPS.map((step, i) => (
              <div
                key={step.step}
                className={clsx(
                  'opacity-0',
                  visible && 'animate-slide-up',
                )}
                style={{ animationDelay: `${i * 0.15}s`, animationFillMode: 'forwards' }}
              >
                {/* Mobile layout: step number pill + card inline */}
                <div className="flex items-start gap-3 sm:gap-6">
                  {/* Step circle — smaller on mobile */}
                  <div className={clsx(
                    'flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12',
                    'rounded-full bg-ark-red/10 border border-ark-red/20',
                    'flex items-center justify-center text-ark-red',
                    'relative z-10 mt-0.5',
                  )}>
                    {step.icon}
                  </div>

                  {/* Content card */}
                  <div className="glass-card p-4 sm:p-6 flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-[10px] font-mono text-ark-red">{step.step}</span>
                      <h3 className="text-base sm:text-lg font-bold text-white">{step.title}</h3>
                    </div>
                    <p className="text-sm text-gray-400 leading-relaxed mb-2">{step.description}</p>
                    <span className="text-[10px] sm:text-xs font-mono text-gray-600">{step.detail}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
