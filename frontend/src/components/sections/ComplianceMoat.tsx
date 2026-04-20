import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

const PILLARS = [
  {
    title: 'Hash Chain',
    description: 'Every ASRE report generates a unique Run ID and SHA-256 hash before market open. This proves pre-investment generation — no retroactive editing possible.',
    detail: 'Run ID proves pre-investment generation',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
    badge: 'SHA-256',
  },
  {
    title: 'SEBI AI Disclosure',
    description: 'Full compliance with SEBI\'s December 2024 AI usage circular. Every PDF includes AI methodology disclosure, data sources, and model limitations — built in, not bolted on.',
    detail: 'Built into every PDF',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    badge: 'Dec 2024',
  },
  {
    title: 'IA/RA Mode',
    description: 'Toggle between Investment Advisor and Research Analyst formatting with a single click. Different SEBI categories require different disclosure language — ASRE handles both automatically.',
    detail: 'Separate formatting per SEBI category',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
    badge: 'IA + RA',
  },
];

export const ComplianceMoat: React.FC = () => {
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
    <section ref={sectionRef} id="compliance" className="section-padding relative">
      <div className="absolute top-1/2 right-0 w-[500px] h-[500px] bg-ark-gold/[0.02] rounded-full blur-[150px] translate-x-1/3 -translate-y-1/2" />

      <div className="section-container relative">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-emerald-400 border border-emerald-500/20 rounded-full mb-4 uppercase tracking-wider">
            Compliance
          </span>
          <h2 className="section-title text-white mb-4">
            Your <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-emerald-300">Compliance Moat</span>
          </h2>
          <p className="section-subtitle mx-auto">
            Three pillars that make ASRE reports audit-proof. Built for SEBI regulations, not around them.
          </p>
        </div>

        {/* Pillars */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
          {PILLARS.map((pillar, i) => (
            <div
              key={pillar.title}
              className={clsx(
                'relative glass-card-hover p-6 lg:p-8 group',
                'opacity-0',
                visible && 'animate-slide-up',
              )}
              style={{ animationDelay: `${i * 0.15}s`, animationFillMode: 'forwards' }}
            >
              {/* Number badge */}
              <div className="absolute top-4 right-4 w-8 h-8 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center text-xs font-mono text-gray-600">
                {i + 1}
              </div>

              {/* Icon */}
              <div className="w-14 h-14 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-5 group-hover:bg-emerald-500/15 transition-colors">
                {pillar.icon}
              </div>

              {/* Content */}
              <h3 className="text-xl font-bold text-white mb-2">{pillar.title}</h3>
              <p className="text-sm text-emerald-400/80 font-mono mb-4">{pillar.detail}</p>
              <p className="text-sm text-gray-400 leading-relaxed">{pillar.description}</p>

              {/* Badge */}
              <div className="mt-5 pt-4 border-t border-white/[0.04]">
                <span className="px-3 py-1 text-[10px] font-mono text-emerald-400/70 border border-emerald-500/20 rounded-full bg-emerald-500/5">
                  {pillar.badge}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
