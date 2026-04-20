import React from 'react';
import { CTAButton } from '../ui/CTAButton';

const SOCIAL_PROOF = [
  { value: '545+', label: 'Walk-forward iterations/stock' },
  { value: 'NSE/BSE', label: 'Native Coverage' },
  { value: 'SEBI', label: 'Dec 2024 Compliant' },
  { value: 'SHA-256', label: 'Hash Audit Trail' },
];

export const HeroSection: React.FC = () => {
  const scrollToDemo = () => {
    const el = document.querySelector('#demo');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const downloadReport = () => {
    const a = document.createElement('a');
    a.href = '/sample-report.pdf';
    a.download = 'TCS_ASRE_Report_Demo.pdf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <section
      id="hero"
      className="relative min-h-[100svh] flex items-center justify-center pt-16 sm:pt-20 pb-10 sm:pb-12 overflow-hidden hero-mesh"
    >
      {/* Grid pattern overlay */}
      <div className="absolute inset-0 grid-bg opacity-60" />

      {/* Radial gradient accent */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] sm:w-[800px] h-[400px] sm:h-[800px] bg-ark-red/[0.04] rounded-full blur-[80px] sm:blur-[120px]" />

      <div className="relative section-container text-center w-full">
        {/* Top badge */}
        <div className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full bg-white/[0.04] border border-white/[0.08] mb-6 sm:mb-8 animate-fade-in">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse flex-shrink-0" />
          <span className="text-[11px] sm:text-xs text-gray-400 font-medium">Live scoring engine — 4-second latency</span>
        </div>

        {/* Main headline */}
        <h1 className="text-3xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-4 sm:mb-6 animate-slide-up max-w-5xl mx-auto leading-[1.15] sm:leading-[1.1] px-2">
          Quantitative Research{' '}
          <br className="hidden sm:inline" />
          Infrastructure for{' '}
          <span className="gradient-text">SEBI-Registered</span>{' '}
          Advisors
        </h1>

        {/* Sub-headline */}
        <p
          className="text-base sm:text-xl text-gray-400 max-w-xl sm:max-w-2xl mx-auto mb-8 sm:mb-10 animate-slide-up leading-relaxed px-2"
          style={{ animationDelay: '0.15s' }}
        >
          Score NSE/BSE stocks in 4s. Hash-signed, SEBI Dec 2024 AI-compliant, audit-ready.
        </p>

        {/* CTAs — stack on mobile, row on sm+ */}
        <div
          className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 mb-10 sm:mb-16 animate-slide-up px-4 sm:px-0"
          style={{ animationDelay: '0.3s' }}
        >
          <CTAButton
            variant="primary"
            size="lg"
            onClick={scrollToDemo}
            id="hero-book-demo"
            className="w-full sm:w-auto"
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            }
          >
            Book 10-Min Demo
          </CTAButton>

          <CTAButton
            variant="secondary"
            size="lg"
            onClick={downloadReport}
            id="hero-download-report"
            className="w-full sm:w-auto"
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            }
          >
            Download Sample Report
          </CTAButton>
        </div>

        {/* Social proof bar — wraps gracefully on mobile */}
        <div className="animate-fade-in" style={{ animationDelay: '0.5s' }}>
          <div className="grid grid-cols-2 sm:inline-grid sm:grid-cols-4 gap-px bg-white/[0.04] border border-white/[0.06] rounded-2xl overflow-hidden mx-auto max-w-xs sm:max-w-none">
            {SOCIAL_PROOF.map((item, i) => (
              <div
                key={i}
                className="flex flex-col items-center gap-1 px-5 sm:px-8 py-4 bg-ark-bg-primary/80"
              >
                <span className="text-base sm:text-xl font-bold text-white font-mono">{item.value}</span>
                <span className="text-[9px] sm:text-[10px] text-gray-500 text-center">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-24 sm:h-32 bg-gradient-to-t from-ark-bg-primary to-transparent" />
    </section>
  );
};
