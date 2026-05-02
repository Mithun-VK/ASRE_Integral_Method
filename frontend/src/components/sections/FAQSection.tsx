import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { BookDemo } from './BookDemo';

const SPRING = [0.16, 1, 0.3, 1] as const;

const FAQS = [
  {
    q: 'What is ASRE?',
    a: 'ASRE (Adaptive Stock Rating Engine) is a quantitative scoring framework that assigns F-Score (Fundamental), T-Score (Technical), and M-Score (Momentum) to NSE/BSE-listed stocks. Each score is walk-forward validated across 545+ iterations and hash-signed for audit trails.',
  },
  {
    q: 'How does the hash chain work?',
    a: 'Every time the ASRE engine scores a stock, it generates a unique Run ID and SHA-256 hash of the entire output — scores, dip context, timestamp, and metadata. This hash is computed before market open, creating an immutable proof that the analysis existed before any investment decision.',
  },
  {
    q: 'Is this SEBI compliant?',
    a: "Yes. ASRE reports are built to comply with SEBI's December 2024 AI usage circular for Investment Advisors and Research Analysts. Every PDF includes mandatory AI methodology disclosure, data source attribution, model limitations, and the IA/RA toggle ensures correct regulatory formatting.",
  },
  {
    q: 'What does "AUC 0.60" mean?',
    a: "AUC (Area Under the Curve) of 0.60 means our model has honest predictive signal — better than random (0.50) but we don't overfit to show inflated numbers. This is walk-forward validated, meaning the model never sees future data during backtesting. We believe in honest metrics over marketing numbers.",
  },
  {
    q: 'Can I use ASRE for client recommendations?',
    a: "ASRE provides quantitative research infrastructure — structured scores with audit trails. It's designed to augment your research process, not replace your judgment. How you use the scores in your client communications is subject to your SEBI registration category (IA/RA/PMS) and compliance framework.",
  },
  {
    q: 'What exchanges and markets do you cover?',
    a: 'ASRE currently covers NSE and BSE-listed equities. We process real-time price, volume, and fundamental data from both exchanges. The engine supports any actively traded symbol on either exchange.',
  },
  {
    q: 'Do you store my client data?',
    a: 'No. ASRE processes ticker symbols and returns scores. We do not store, access, or process any of your client data, portfolio holdings, or personal information. Your client relationship remains entirely yours.',
  },
  {
    q: 'How do I get started?',
    a: "Book a 10-minute demo above. We'll walk through a live scoring session, show you a sample PDF report, and discuss which plan fits your practice. No commitment required.",
  },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.05 } },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20, clipPath: 'inset(100% 0 0 0)' },
  visible: { opacity: 1, y: 0, clipPath: 'inset(0% 0 0 0)', transition: { duration: 0.5, ease: SPRING } },
};

export const FAQSection: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [headerRef, headerVisible] = useInView<HTMLDivElement>({ threshold: 0.3, once: true });
  const [listRef, listVisible] = useInView<HTMLDivElement>({ threshold: 0.05, once: true });
  const [ctaRef, ctaVisible] = useInView<HTMLDivElement>({ threshold: 0.5, once: true });
  const reduced = useReducedMotion();

  const toggle = (i: number) =>
    setOpenIndex(openIndex === i ? null : i);

  return (
    <section id="faq" className="section-padding relative overflow-hidden">

      {/* Ambient glow */}
      <div
        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-ark-red/[0.03] rounded-full blur-[120px] pointer-events-none"
        aria-hidden="true"
      />

      <div className="section-container relative z-10">

        {/* Section header */}
        <motion.div
          ref={headerRef}
          className="text-center mb-10 sm:mb-16"
          initial={reduced ? false : { opacity: 0, y: 24 }}
          animate={headerVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.55, ease: SPRING }}
        >
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-gray-400 border border-white/10 rounded-full mb-4 uppercase tracking-wider">
            FAQ
          </span>
          <h2 className="section-title text-white mb-4">Common Questions</h2>
          <p className="section-subtitle mx-auto">
            Everything SEBI-registered advisors ask before onboarding.
          </p>
        </motion.div>

        {/* FAQ items */}
        <motion.div
          ref={listRef}
          className="max-w-3xl mx-auto space-y-3"
          variants={containerVariants}
          initial={reduced ? false : 'hidden'}
          animate={listVisible ? 'visible' : 'hidden'}
        >
          {FAQS.map((faq, i) => {
            const isOpen = openIndex === i;

            return (
              <motion.div
                key={i}
                variants={itemVariants}
                className="rounded-xl border overflow-hidden"
                animate={{
                  borderColor: isOpen
                    ? 'rgba(220,38,38,0.25)'
                    : 'rgba(255,255,255,0.06)',
                  backgroundColor: isOpen
                    ? 'rgba(255,255,255,0.03)'
                    : 'rgba(255,255,255,0.01)',
                }}
                transition={{ duration: 0.25 }}
              >
                <button
                  onClick={() => toggle(i)}
                  className="w-full flex items-center justify-between p-4 sm:p-5 text-left min-h-[52px] [touch-action:manipulation] group"
                  id={`faq-${i}`}
                  aria-expanded={isOpen}
                >
                  <span className={clsx(
                    'text-sm font-semibold pr-4 transition-colors duration-200',
                    isOpen ? 'text-white' : 'text-gray-300 group-hover:text-white',
                  )}>
                    {faq.q}
                  </span>

                  {/* Animated chevron */}
                  <motion.svg
                    className={clsx(
                      'w-5 h-5 flex-shrink-0',
                      isOpen ? 'text-ark-red' : 'text-gray-600',
                    )}
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ duration: 0.3, ease: SPRING }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </motion.svg>
                </button>

                {/* Accordion body */}
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      key="content"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] as const }}
                      style={{ overflow: 'hidden' }}
                    >
                      <motion.p
                        className="px-5 pb-5 text-sm text-gray-400 leading-relaxed"
                        initial={{ y: -8 }}
                        animate={{ y: 0 }}
                        transition={{ duration: 0.3, ease: SPRING }}
                      >
                        {faq.a}
                      </motion.p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Bottom CTA */}
        <motion.div
          ref={ctaRef}
          className="text-center mt-12"
          initial={reduced ? false : { opacity: 0, y: 20 }}
          animate={ctaVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, ease: SPRING }}
        >
          <p className="text-sm text-gray-500 mb-4">Still have questions?</p>
          <BookDemo
            variant="primary"
            size="lg"
            id="hero-book-demo"
            className="w-full sm:w-auto"
            label="Book 10-Min Demo"
          />
        </motion.div>
      </div>
    </section>
  );
};
