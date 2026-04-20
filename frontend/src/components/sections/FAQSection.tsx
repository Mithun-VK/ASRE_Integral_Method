import React, { useState } from 'react';
import clsx from 'clsx';

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
    a: 'Yes. ASRE reports are built to comply with SEBI\'s December 2024 AI usage circular for Investment Advisors and Research Analysts. Every PDF includes mandatory AI methodology disclosure, data source attribution, model limitations, and the IA/RA toggle ensures correct regulatory formatting.',
  },
  {
    q: 'What does "AUC 0.60" mean?',
    a: 'AUC (Area Under the Curve) of 0.60 means our model has honest predictive signal — better than random (0.50) but we don\'t overfit to show inflated numbers. This is walk-forward validated, meaning the model never sees future data during backtesting. We believe in honest metrics over marketing numbers.',
  },
  {
    q: 'Can I use ASRE for client recommendations?',
    a: 'ASRE provides quantitative research infrastructure — structured scores with audit trails. It\'s designed to augment your research process, not replace your judgment. How you use the scores in your client communications is subject to your SEBI registration category (IA/RA/PMS) and compliance framework.',
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
    a: 'Book a 10-minute demo above. We\'ll walk through a live scoring session, show you a sample PDF report, and discuss which plan fits your practice. No commitment required.',
  },
];

export const FAQSection: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (i: number) => {
    setOpenIndex(openIndex === i ? null : i);
  };

  return (
    <section id="faq" className="section-padding relative">
      <div className="section-container">
        {/* Section header */}
        <div className="text-center mb-10 sm:mb-16">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-gray-400 border border-white/10 rounded-full mb-4 uppercase tracking-wider">
            FAQ
          </span>
          <h2 className="section-title text-white mb-4">
            Common Questions
          </h2>
          <p className="section-subtitle mx-auto">
            Everything SEBI-registered advisors ask before onboarding.
          </p>
        </div>

        {/* FAQ items */}
        <div className="max-w-3xl mx-auto space-y-3">
          {FAQS.map((faq, i) => (
            <div
              key={i}
              className={clsx(
                'rounded-xl border transition-all duration-300',
                openIndex === i
                  ? 'bg-white/[0.03] border-ark-red/20'
                  : 'bg-white/[0.01] border-white/[0.06] hover:border-white/[0.1]',
              )}
            >
              <button
                onClick={() => toggle(i)}
                className="w-full flex items-center justify-between p-4 sm:p-5 text-left min-h-[52px] [touch-action:manipulation]"
                id={`faq-${i}`}
              >
                <span className={clsx(
                  'text-sm font-semibold pr-4 transition-colors',
                  openIndex === i ? 'text-white' : 'text-gray-300'
                )}>
                  {faq.q}
                </span>
                <svg
                  className={clsx(
                    'w-5 h-5 flex-shrink-0 transition-transform duration-300',
                    openIndex === i ? 'text-ark-red rotate-180' : 'text-gray-600'
                  )}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              <div
                className={clsx(
                  'overflow-hidden transition-all duration-300',
                  openIndex === i ? 'max-h-96 pb-5' : 'max-h-0'
                )}
              >
                <p className="px-5 text-sm text-gray-400 leading-relaxed">
                  {faq.a}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-12">
          <p className="text-sm text-gray-500 mb-4">Still have questions?</p>
          <button
            onClick={() => {
              const el = document.querySelector('#demo');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }}
            className="px-6 py-3 text-sm font-semibold bg-ark-red hover:bg-red-700 text-white rounded-xl shadow-lg shadow-ark-red/25 hover:shadow-ark-red/40 transition-all duration-300 hover:-translate-y-0.5"
            id="faq-book-demo"
          >
            Book a 10-Min Demo →
          </button>
        </div>
      </div>
    </section>
  );
};
