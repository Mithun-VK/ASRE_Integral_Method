import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { HolographicCard } from '../ui/HolographicCard';

const SPRING = [0.16, 1, 0.3, 1] as const;

const PROBLEMS = [
  {
    quote: '"I use broker notes"',
    problem: 'No Run ID / Hash. SEBI audits BASIS.',
    description: 'Broker notes lack traceable audit trails. When SEBI requests documentation of your investment rationale, you need timestamped, hash-signed proof — not forwarded emails.',
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    risk: 'High audit risk',
    riskColor: 'text-red-400 border-red-500/20 bg-red-950/40',
    dotColor: 'bg-red-500',
  },
  {
    quote: '"Excel has my model"',
    problem: 'Retroactively editable. No timestamp proof.',
    description: "Spreadsheets can be modified after the fact. There's no immutable record proving your analysis existed before the investment decision was made.",
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
    risk: 'Compliance gap',
    riskColor: 'text-amber-400 border-amber-500/20 bg-amber-950/40',
    dotColor: 'bg-amber-500',
  },
  {
    quote: '"My own analysis"',
    problem: 'Can you document F/T/M scores in 5 min?',
    description: "Manual analysis can't produce structured Fundamental, Technical, and Momentum scores with hash proofs fast enough to scale across your client portfolio.",
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    risk: 'Scalability limit',
    riskColor: 'text-orange-400 border-orange-500/20 bg-orange-950/40',
    dotColor: 'bg-orange-500',
  },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.14, delayChildren: 0.1 } },
};

const cardVariants = {
  hidden: { opacity: 0, y: 40, clipPath: 'inset(100% 0 0 0)' },
  visible: {
    opacity: 1, y: 0,
    clipPath: 'inset(0% 0 0 0)',
    transition: { duration: 0.65, ease: SPRING },
  },
};

const headerVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: SPRING } },
};

export const ProblemSection: React.FC = () => {
  const [ref, visible] = useInView<HTMLElement>({ threshold: 0.12, once: true });
  const reduced = useReducedMotion();

  return (
    <section ref={ref} id="problem" className="section-padding relative overflow-hidden">

      {/* Ambient glow */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-ark-red/[0.03] rounded-full blur-[100px] pointer-events-none"
        aria-hidden="true"
      />

      <div className="section-container relative z-10">

        {/* Section header */}
        <motion.div
          className="text-center mb-16"
          variants={headerVariants}
          initial={reduced ? false : 'hidden'}
          animate={visible ? 'visible' : 'hidden'}
        >
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-red border border-ark-red/20 rounded-full mb-4 uppercase tracking-wider">
            The Problem
          </span>
          <h2 className="section-title text-white mb-4">
            What RIAs Tell Us <span className="gradient-text">Every Week</span>
          </h2>
          <p className="section-subtitle mx-auto">
            Three conversations we hear from every SEBI-registered advisor before they switch to ASRE.
          </p>
        </motion.div>

        {/* Cards */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8"
          variants={containerVariants}
          initial={reduced ? false : 'hidden'}
          animate={visible ? 'visible' : 'hidden'}
        >
          {PROBLEMS.map((item, i) => (
            <motion.div key={i} variants={cardVariants}>
              <HolographicCard className="h-full glass-card p-6 lg:p-8 group">

                {/* Icon */}
                <motion.div
                  className="w-12 h-12 rounded-xl bg-ark-red/10 border border-ark-red/20 flex items-center justify-center text-ark-red mb-5"
                  whileHover={reduced ? {} : { scale: 1.1, rotate: 3 }}
                  transition={{ duration: 0.2 }}
                >
                  {item.icon}
                </motion.div>

                {/* Quote */}
                <p className="text-xl font-bold text-white mb-2">{item.quote}</p>

                {/* Problem highlight */}
                <p className="text-sm font-mono text-ark-red mb-4">{item.problem}</p>

                {/* Description */}
                <p className="text-sm text-gray-400 leading-relaxed mb-5">{item.description}</p>

                {/* Risk tag */}
                <div className={clsx(
                  'inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono uppercase tracking-wider',
                  item.riskColor,
                )}>
                  <span className={clsx('w-1.5 h-1.5 rounded-full animate-pulse', item.dotColor)} />
                  {item.risk}
                </div>

              </HolographicCard>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
