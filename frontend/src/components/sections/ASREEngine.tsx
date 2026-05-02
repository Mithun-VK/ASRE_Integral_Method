import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import clsx from 'clsx';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { AnimatedCounter } from '../ui/AnimatedCounter';
import { HolographicCard } from '../ui/HolographicCard';

const SPRING = [0.16, 1, 0.3, 1] as const;

const SCORES = [
  {
    label: 'F-Score',
    fullLabel: 'Fundamental',
    measures: 'Revenue quality, debt coverage, earnings consistency',
    example: 'RELIANCE: 74.2% A-tier',
    color: 'from-emerald-600 to-emerald-400',
    textColor: 'text-emerald-400',
    barColor: 'bg-gradient-to-r from-emerald-700 via-emerald-500 to-emerald-400',
    bgColor: 'bg-emerald-950/40',
    glowColor: 'rgba(52,211,153,0.15)',
    value: 74.2,
  },
  {
    label: 'T-Score',
    fullLabel: 'Technical',
    measures: 'Price action, volume profile, RSI divergence',
    example: '545 walk-forward iterations',
    color: 'from-blue-600 to-blue-400',
    textColor: 'text-blue-400',
    barColor: 'bg-gradient-to-r from-blue-700 via-blue-500 to-blue-400',
    bgColor: 'bg-blue-950/40',
    glowColor: 'rgba(96,165,250,0.15)',
    value: 61.8,
  },
  {
    label: 'M-Score',
    fullLabel: 'Momentum',
    measures: 'Trend persistence, sector rotation, dip detection',
    example: 'SUZLON: 48.1 MID dip',
    color: 'from-amber-600 to-amber-400',
    textColor: 'text-amber-400',
    barColor: 'bg-gradient-to-r from-amber-700 via-amber-500 to-amber-400',
    bgColor: 'bg-amber-950/40',
    glowColor: 'rgba(251,191,36,0.15)',
    value: 48.1,
  },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.14, delayChildren: 0.05 } },
};

const cardVariants = {
  hidden: { opacity: 0, y: 48, clipPath: 'inset(100% 0 0 0)' },
  visible: {
    opacity: 1, y: 0,
    clipPath: 'inset(0% 0 0 0)',
    transition: { duration: 0.65, ease: SPRING },
  },
};

// Animated progress bar — fill driven by inView
const ScoreProgressBar: React.FC<{
  value: number;
  barColor: string;
  bgColor: string;
  glowColor: string;
  inView: boolean;
  reduced: boolean;
  delay: number;
}> = ({ value, barColor, bgColor, glowColor, inView, reduced, delay }) => (
  <div className={clsx('relative h-2.5 rounded-full overflow-hidden', bgColor)}>
    <motion.div
      className={clsx('absolute inset-y-0 left-0 rounded-full', barColor)}
      initial={{ width: '0%' }}
      animate={{ width: inView ? `${value}%` : '0%' }}
      transition={reduced
        ? { duration: 0 }
        : { duration: 1.4, delay, ease: [0.25, 0.46, 0.45, 0.94] as const }
      }
      style={{
        boxShadow: inView ? `0 0 12px ${glowColor}, 0 0 4px ${glowColor}` : 'none',
      }}
    />
    {/* Shimmer sweep on the bar */}
    {inView && !reduced && (
      <motion.div
        className="absolute inset-y-0 w-16 rounded-full"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
        }}
        initial={{ left: '-10%' }}
        animate={{ left: '110%' }}
        transition={{ duration: 1.0, delay: delay + 0.6, ease: 'easeOut' }}
      />
    )}
  </div>
);

export const ASREEngine: React.FC = () => {
  const [headerRef, headerVisible] = useInView<HTMLDivElement>({ threshold: 0.3, once: true });
  const [cardsRef, cardsVisible] = useInView<HTMLDivElement>({ threshold: 0.1, once: true });
  const [badgeRef, badgeVisible] = useInView<HTMLDivElement>({ threshold: 0.5, once: true });
  const reduced = useReducedMotion();

  // Parallax glow orb
  const orbRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: orbRef, offset: ['start end', 'end start'] });
  const orbY = useTransform(scrollYProgress, [0, 1], ['20%', '-20%']);

  return (
    <section id="asre-engine" className="section-padding relative overflow-hidden">

      {/* Parallax glow orb */}
      <motion.div
        ref={orbRef}
        className="absolute top-1/2 left-0 w-[600px] h-[600px] bg-ark-red/[0.04] rounded-full blur-[150px] -translate-y-1/2 pointer-events-none"
        style={{ y: reduced ? 0 : orbY }}
        aria-hidden="true"
      />
      <div
        className="absolute top-1/3 right-0 w-[400px] h-[400px] bg-ark-gold/[0.02] rounded-full blur-[120px] pointer-events-none"
        aria-hidden="true"
      />

      <div className="section-container relative z-10">

        {/* Section header */}
        <motion.div
          ref={headerRef}
          className="text-center mb-16"
          initial={reduced ? false : { opacity: 0, y: 28 }}
          animate={headerVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, ease: SPRING }}
        >
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-gold border border-ark-gold/20 rounded-full mb-4 uppercase tracking-wider">
            The Engine
          </span>
          <h2 className="section-title text-white mb-4">
            ASRE <span className="gradient-text-gold">Scoring Engine</span>
          </h2>
          <p className="section-subtitle mx-auto">
            Three orthogonal scores. One composite signal. Every score hash-stamped before market open.
          </p>
        </motion.div>

        {/* Score cards */}
        <motion.div
          ref={cardsRef}
          className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8 mb-12"
          variants={containerVariants}
          initial={reduced ? false : 'hidden'}
          animate={cardsVisible ? 'visible' : 'hidden'}
        >
          {SCORES.map((score, i) => (
            <motion.div key={score.label} variants={cardVariants}>
              <HolographicCard
                className="h-full p-6 lg:p-8 glass-card group"
                glowColor={score.glowColor}
              >
                {/* Score header row */}
                <div className="flex items-start justify-between mb-5">
                  <div>
                    <span className={clsx('text-2xl font-bold font-mono', score.textColor)}>
                      {score.label}
                    </span>
                    <span className="block text-xs text-gray-500 mt-0.5">{score.fullLabel}</span>
                  </div>

                  {/* Animated counter */}
                  <div className={clsx('text-3xl font-bold font-mono tabular-nums', score.textColor)}>
                    {cardsVisible ? (
                      <AnimatedCounter
                        target={score.value}
                        decimals={1}
                        suffix="%"
                        duration={1400}
                      />
                    ) : '0.0%'}
                  </div>
                </div>

                {/* Animated progress bar */}
                <div className="mb-5">
                  <ScoreProgressBar
                    value={score.value}
                    barColor={score.barColor}
                    bgColor={score.bgColor}
                    glowColor={score.glowColor}
                    inView={cardsVisible}
                    reduced={reduced}
                    delay={i * 0.15 + 0.3}
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
              </HolographicCard>
            </motion.div>
          ))}
        </motion.div>

        {/* AUC Badge */}
        <motion.div
          ref={badgeRef}
          className="flex items-center justify-center"
          initial={reduced ? false : { opacity: 0, scale: 0.9, y: 20 }}
          animate={badgeVisible ? { opacity: 1, scale: 1, y: 0 } : {}}
          transition={{ duration: 0.55, ease: SPRING }}
        >
          <motion.div
            className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-white/[0.02] border border-white/[0.06]"
            whileHover={reduced ? {} : {
              borderColor: 'rgba(245,158,11,0.25)',
              backgroundColor: 'rgba(245,158,11,0.03)',
              scale: 1.02,
            }}
            transition={{ duration: 0.2 }}
          >
            <motion.div
              className="flex items-center gap-2"
              animate={badgeVisible && !reduced ? { rotate: [0, 8, -4, 0] } : {}}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <svg className="w-5 h-5 text-ark-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
              <span className="text-sm font-bold text-ark-gold gold-pulse rounded-full px-1">AUC 0.60</span>
            </motion.div>
            <span className="text-xs text-gray-600">—</span>
            <span className="text-sm text-gray-400">Honest signal. No overfitting. Walk-forward validated.</span>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};
