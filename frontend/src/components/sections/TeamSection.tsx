import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';

const SPRING = [0.16, 1, 0.3, 1] as const;

const TEAM = [
  {
    name: 'Sriranjan',
    role: 'CEO & Business Development',
    description: 'Drives partnerships with SEBI-registered RIAs and PMS firms. Responsible for go-to-market strategy and advisor relationships.',
    tags: ['Strategy', 'Partnerships', 'GTM'],
    avatar: 'S',
    color: 'from-ark-red to-red-400',
    shadowColor: 'rgba(220,38,38,0.3)',
    ringColor: 'rgba(220,38,38,0.2)',
  },
  {
    name: 'Mithun',
    role: 'CTO & ASRE Engine',
    description: 'Architect of the ASRE scoring engine. Built the F/T/M framework, walk-forward validation pipeline, and hash-chain infrastructure.',
    tags: ['Engineering', 'ML/Quant', 'Infrastructure'],
    avatar: 'M',
    color: 'from-blue-600 to-blue-400',
    shadowColor: 'rgba(96,165,250,0.3)',
    ringColor: 'rgba(96,165,250,0.2)',
  },
  {
    name: 'Shachin',
    role: 'Research & Compliance',
    description: 'NISM Series XV certified. Ensures all ASRE outputs meet SEBI regulatory requirements. Leads research methodology and compliance frameworks.',
    tags: ['NISM XV', 'Compliance', 'Research'],
    avatar: 'S',
    color: 'from-emerald-600 to-emerald-400',
    shadowColor: 'rgba(52,211,153,0.3)',
    ringColor: 'rgba(52,211,153,0.2)',
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

const tagVariants = {
  hidden: { opacity: 0, scale: 0.7 },
  visible: (i: number) => ({
    opacity: 1, scale: 1,
    transition: { delay: i * 0.07, duration: 0.3, ease: SPRING },
  }),
};

export const TeamSection: React.FC = () => {
  const [headerRef, headerVisible] = useInView<HTMLDivElement>({ threshold: 0.3, once: true });
  const [cardsRef, cardsVisible] = useInView<HTMLDivElement>({ threshold: 0.1, once: true });
  const reduced = useReducedMotion();

  return (
    <section id="team" className="section-padding relative overflow-hidden">

      {/* Background glow */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] bg-ark-red/[0.02] rounded-full blur-[120px] pointer-events-none"
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
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-gray-400 border border-white/10 rounded-full mb-4 uppercase tracking-wider">
            Team
          </span>
          <h2 className="section-title text-white mb-4">
            Built by <span className="gradient-text">Practitioners</span>
          </h2>
          <p className="section-subtitle mx-auto">
            A team that understands both quantitative finance and SEBI compliance — because we've lived it.
          </p>
        </motion.div>

        {/* Team cards */}
        <motion.div
          ref={cardsRef}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8 max-w-4xl mx-auto"
          variants={containerVariants}
          initial={reduced ? false : 'hidden'}
          animate={cardsVisible ? 'visible' : 'hidden'}
        >
          {TEAM.map((member) => (
            <motion.div key={member.name} variants={cardVariants}>
              <motion.div
                className="glass-card p-6 lg:p-8 text-center group h-full flex flex-col"
                whileHover={reduced ? {} : {
                  borderColor: member.ringColor,
                  backgroundColor: 'rgba(255,255,255,0.04)',
                  y: -4,
                }}
                transition={{ duration: 0.25 }}
              >
                {/* Avatar with gradient spin ring on hover */}
                <div className="relative mx-auto mb-5 w-fit">
                  {/* Spin ring — only visible on hover */}
                  {!reduced && (
                    <motion.div
                      className={clsx(
                        'absolute -inset-1 rounded-2xl bg-gradient-to-br opacity-0 group-hover:opacity-100',
                        member.color,
                      )}
                      style={{ filter: 'blur(6px)' }}
                      transition={{ duration: 0.3 }}
                    />
                  )}

                  <motion.div
                    className={clsx(
                      'relative w-16 h-16 rounded-2xl flex items-center justify-center bg-gradient-to-br shadow-lg',
                      member.color,
                    )}
                    whileHover={reduced ? {} : { scale: 1.1, rotate: 3 }}
                    transition={{ duration: 0.3, ease: SPRING }}
                    style={{
                      boxShadow: `0 8px 24px ${member.shadowColor}`,
                    }}
                  >
                    <span className="text-2xl font-bold text-white">{member.avatar}</span>
                  </motion.div>
                </div>

                {/* Info */}
                <h3 className="text-lg font-bold text-white mb-1">{member.name}</h3>
                <p className="text-sm text-gray-400 mb-4">{member.role}</p>
                <p className="text-xs text-gray-500 leading-relaxed mb-5 flex-1">{member.description}</p>

                {/* Tags — staggered pop-in when card visible */}
                <div className="flex items-center justify-center gap-2 flex-wrap">
                  {member.tags.map((tag, j) => (
                    <motion.span
                      key={tag}
                      custom={j}
                      variants={tagVariants}
                      className="px-2.5 py-0.5 text-[10px] font-mono text-gray-500 border border-white/[0.06] rounded-full bg-white/[0.02] hover:text-gray-300 hover:border-white/10 transition-colors duration-200 cursor-default"
                    >
                      {tag}
                    </motion.span>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
