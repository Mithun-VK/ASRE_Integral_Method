import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { HolographicCard } from '../ui/HolographicCard';

const SPRING = [0.16, 1, 0.3, 1] as const;

const PILLARS = [
  {
    title: 'Hash Chain',
    description: "Every ASRE report generates a unique Run ID and SHA-256 hash before market open. This proves pre-investment generation — no retroactive editing possible.",
    detail: 'Run ID proves pre-investment generation',
    badge: 'SHA-256',
    glowColor: 'rgba(52,211,153,0.15)',
    iconBg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    badgeStyle: 'text-emerald-400/70 border-emerald-500/20 bg-emerald-500/5',
    detailColor: 'text-emerald-400/80',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
  },
  {
    title: 'SEBI AI Disclosure',
    description: "Full compliance with SEBI's December 2024 AI usage circular. Every PDF includes AI methodology disclosure, data sources, and model limitations — built in, not bolted on.",
    detail: 'Built into every PDF',
    badge: 'Dec 2024',
    glowColor: 'rgba(52,211,153,0.18)',
    iconBg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    badgeStyle: 'text-emerald-400/70 border-emerald-500/20 bg-emerald-500/5',
    detailColor: 'text-emerald-400/80',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
  {
    title: 'IA/RA Mode',
    description: "Toggle between Investment Advisor and Research Analyst formatting with a single click. Different SEBI categories require different disclosure language — ASRE handles both automatically.",
    detail: 'Separate formatting per SEBI category',
    badge: 'IA + RA',
    glowColor: 'rgba(52,211,153,0.12)',
    iconBg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    badgeStyle: 'text-emerald-400/70 border-emerald-500/20 bg-emerald-500/5',
    detailColor: 'text-emerald-400/80',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
  },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.14, delayChildren: 0.05 } },
};

const cardVariants = {
  hidden: { opacity: 0, y: 44, clipPath: 'inset(100% 0 0 0)' },
  visible: {
    opacity: 1, y: 0,
    clipPath: 'inset(0% 0 0 0)',
    transition: { duration: 0.65, ease: SPRING },
  },
};

export const ComplianceMoat: React.FC = () => {
  const [headerRef, headerVisible] = useInView<HTMLDivElement>({ threshold: 0.3, once: true });
  const [cardsRef, cardsVisible] = useInView<HTMLDivElement>({ threshold: 0.1, once: true });
  const reduced = useReducedMotion();

  return (
    <section id="compliance" className="section-padding relative overflow-hidden">

      {/* Background glows */}
      <div className="absolute top-1/2 right-0 w-[500px] h-[500px] bg-emerald-500/[0.03] rounded-full blur-[150px] translate-x-1/3 -translate-y-1/2 pointer-events-none" aria-hidden="true" />
      <div className="absolute top-1/4 left-0 w-[300px] h-[300px] bg-ark-gold/[0.02] rounded-full blur-[100px] pointer-events-none" aria-hidden="true" />

      <div className="section-container relative z-10">

        {/* Section header */}
        <motion.div
          ref={headerRef}
          className="text-center mb-16"
          initial={reduced ? false : { opacity: 0, y: 28 }}
          animate={headerVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, ease: SPRING }}
        >
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-emerald-400 border border-emerald-500/20 rounded-full mb-4 uppercase tracking-wider">
            Compliance
          </span>
          <h2 className="section-title text-white mb-4">
            Your{' '}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-emerald-300">
              Compliance Moat
            </span>
          </h2>
          <p className="section-subtitle mx-auto">
            Three pillars that make ASRE reports audit-proof. Built for SEBI regulations, not around them.
          </p>
        </motion.div>

        {/* Pillars */}
        <motion.div
          ref={cardsRef}
          className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8"
          variants={containerVariants}
          initial={reduced ? false : 'hidden'}
          animate={cardsVisible ? 'visible' : 'hidden'}
        >
          {PILLARS.map((pillar, i) => (
            <motion.div key={pillar.title} variants={cardVariants}>
              <HolographicCard
                className="relative h-full glass-card p-6 lg:p-8 group"
                glowColor={pillar.glowColor}
              >
                {/* Number badge */}
                <motion.div
                  className="absolute top-4 right-4 w-8 h-8 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center text-xs font-mono text-gray-600"
                  initial={reduced ? false : { scale: 0, rotate: -180 }}
                  animate={cardsVisible ? { scale: 1, rotate: 0 } : {}}
                  transition={{ duration: 0.5, delay: i * 0.14 + 0.3, ease: SPRING }}
                >
                  {i + 1}
                </motion.div>

                {/* Icon with entrance */}
                <motion.div
                  className={clsx(
                    'w-14 h-14 rounded-xl border flex items-center justify-center mb-5',
                    pillar.iconBg,
                  )}
                  initial={reduced ? false : { scale: 0.5, opacity: 0 }}
                  animate={cardsVisible ? { scale: 1, opacity: 1 } : {}}
                  transition={{ duration: 0.45, delay: i * 0.14 + 0.2, ease: SPRING }}
                  whileHover={reduced ? {} : { scale: 1.1, rotate: 5 }}
                >
                  {pillar.icon}
                </motion.div>

                {/* Content */}
                <h3 className="text-xl font-bold text-white mb-2">{pillar.title}</h3>
                <p className={clsx('text-sm font-mono mb-4', pillar.detailColor)}>{pillar.detail}</p>
                <p className="text-sm text-gray-400 leading-relaxed">{pillar.description}</p>

                {/* Badge */}
                <div className="mt-5 pt-4 border-t border-white/[0.04]">
                  <motion.span
                    className={clsx(
                      'inline-block px-3 py-1 text-[10px] font-mono rounded-full border',
                      pillar.badgeStyle,
                    )}
                    whileHover={reduced ? {} : { scale: 1.05 }}
                    transition={{ duration: 0.15 }}
                  >
                    {pillar.badge}
                  </motion.span>
                </div>
              </HolographicCard>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
