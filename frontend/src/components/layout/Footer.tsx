import React from 'react';
import { motion } from 'framer-motion';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';

const SPRING = [0.16, 1, 0.3, 1] as const;

const FOOTER_LINKS = {
  Product: [
    { label: 'On-Demand Report', href: '#pricing' },
    { label: 'Research Pack', href: '#pricing' },
    { label: 'Backtest Report', href: '#pricing' },
    { label: 'Advisor Dashboard', href: '#pricing' },
  ],
  Company: [
    { label: 'About', href: '#team' },
    { label: 'Compliance', href: '#compliance' },
    { label: 'FAQ', href: '#faq' },
  ],
  Legal: [
    { label: 'Privacy Policy', href: '#' },
    { label: 'Terms of Service', href: '#' },
    { label: 'SEBI Disclosures', href: '#compliance' },
  ],
};

const BADGES = ['SEBI Dec 2024 Compliant', 'NSE/BSE Native', 'Hash Audit Trail'];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08, delayChildren: 0.05 } },
};

const colVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: SPRING } },
};

export const Footer: React.FC = () => {
  const [ref, visible] = useInView<HTMLElement>({ threshold: 0.1, once: true });
  const reduced = useReducedMotion();

  const scrollTo = (href: string) => {
    if (href === '#') return;
    document.querySelector(href)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <footer
      ref={ref}
      className="relative border-t border-white/[0.06] bg-ark-bg-primary overflow-hidden"
    >
      {/* Subtle top glow line */}
      <motion.div
        className="absolute top-0 left-1/2 -translate-x-1/2 h-px w-0 bg-gradient-to-r from-transparent via-ark-red/40 to-transparent"
        animate={visible && !reduced ? { width: '60%' } : {}}
        transition={{ duration: 1.2, delay: 0.2, ease: 'easeOut' }}
        aria-hidden="true"
      />

      <div className="section-container py-12 sm:py-16 relative z-10">
        <motion.div
          className="grid grid-cols-2 md:grid-cols-5 gap-8"
          variants={containerVariants}
          initial={reduced ? false : 'hidden'}
          animate={visible ? 'visible' : 'hidden'}
        >
          {/* Brand column */}
          <motion.div variants={colVariants} className="col-span-2 md:col-span-2">
            <div className="flex items-center gap-2.5 mb-4">
              <motion.div
                className="w-10 h-10 rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-black/30 flex-shrink-0"
                whileHover={reduced ? {} : { scale: 1.08, rotate: 3 }}
                transition={{ duration: 0.25 }}
              >
                <img
                  src="/Logo.jpeg"
                  alt="Ark Angel Logo"
                  className="w-full h-full object-contain bg-ark-bg-secondary"
                  width={40}
                  height={40}
                  loading="lazy"
                />
              </motion.div>
              <div>
                <span className="text-white font-bold text-lg tracking-tight">ARK ANGEL</span>
                <span className="block text-[9px] text-gray-500 uppercase tracking-[0.2em] -mt-0.5">
                  Research Infrastructure
                </span>
              </div>
            </div>

            <p className="text-sm text-gray-500 max-w-sm mb-5 leading-relaxed">
              Quantitative research infrastructure for SEBI-registered Investment Advisors
              and Portfolio Managers. Hash-signed, audit-ready, compliant.
            </p>

            {/* Badges */}
            <div className="flex items-center gap-2 flex-wrap">
              {BADGES.map((badge, i) => (
                <motion.span
                  key={badge}
                  className="px-3 py-1 text-[10px] font-mono text-gray-500 border border-white/[0.06] rounded-full hover:border-white/[0.12] hover:text-gray-400 transition-colors duration-200 cursor-default"
                  initial={reduced ? false : { opacity: 0, scale: 0.85 }}
                  animate={visible ? { opacity: 1, scale: 1 } : {}}
                  transition={{ delay: i * 0.08 + 0.3, duration: 0.35, ease: SPRING }}
                >
                  {badge}
                </motion.span>
              ))}
            </div>
          </motion.div>

          {/* Link columns */}
          {Object.entries(FOOTER_LINKS).map(([title, links]) => (
            <motion.div key={title} variants={colVariants}>
              <h4 className="text-sm font-semibold text-white mb-4">{title}</h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link.label}>
                    <button
                      onClick={() => scrollTo(link.href)}
                      className="text-sm text-gray-500 hover:text-gray-200 transition-colors duration-200 relative group"
                    >
                      {link.label}
                      {/* Underline slide-in on hover */}
                      <span className="absolute -bottom-0.5 left-0 w-0 h-px bg-ark-red group-hover:w-full transition-all duration-300" />
                    </button>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </motion.div>

        {/* Bottom bar */}
        <motion.div
          className="mt-12 pt-8 border-t border-white/[0.04] flex flex-col sm:flex-row items-center justify-between gap-4"
          initial={reduced ? false : { opacity: 0 }}
          animate={visible ? { opacity: 1 } : {}}
          transition={{ delay: 0.6, duration: 0.5 }}
        >
          <p className="text-xs text-gray-600">
            © {new Date().getFullYear()} Ark Angel Technologies. All rights reserved.
          </p>
          <p className="text-[10px] text-gray-700 text-center sm:text-right max-w-md">
            Disclaimer: ASRE scores are quantitative signals, not investment advice.
            Consult your SEBI-registered advisor before making investment decisions.
          </p>
        </motion.div>
      </div>
    </footer>
  );
};
