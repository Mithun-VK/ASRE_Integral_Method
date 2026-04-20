import React from 'react';

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

export const Footer: React.FC = () => {
  const scrollTo = (href: string) => {
    if (href === '#') return;
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <footer className="relative border-t border-white/[0.06] bg-ark-bg-primary">
      <div className="section-container py-12 sm:py-16">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8 lg:gap-8">
          {/* Brand — full width on mobile, 2 cols on md, 2 of 5 on lg */}
          <div className="col-span-2 md:col-span-2 lg:col-span-2">
            <div className="flex items-center gap-2.5 mb-4">
              <div className="w-10 h-10 rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-black/30 flex-shrink-0">
                <img
                  src="/Logo.jpeg"
                  alt="Ark Angel Logo"
                  className="w-full h-full object-contain bg-ark-bg-secondary"
                />
              </div>
              <div>
                <span className="text-white font-bold text-lg tracking-tight">ARK ANGEL</span>
                <span className="block text-[9px] text-gray-500 uppercase tracking-[0.2em] -mt-0.5">Research Infrastructure</span>
              </div>
            </div>
            <p className="text-sm text-gray-500 max-w-sm mb-5 leading-relaxed">
              Quantitative research infrastructure for SEBI-registered Investment Advisors and Portfolio Managers.
              Hash-signed, audit-ready, compliant.
            </p>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-3 py-1 text-[10px] font-mono text-gray-500 border border-white/[0.06] rounded-full">
                SEBI Dec 2024 Compliant
              </span>
              <span className="px-3 py-1 text-[10px] font-mono text-gray-500 border border-white/[0.06] rounded-full">
                NSE/BSE Native
              </span>
              <span className="px-3 py-1 text-[10px] font-mono text-gray-500 border border-white/[0.06] rounded-full">
                Hash Audit Trail
              </span>
            </div>
          </div>

          {/* Link columns */}
          {Object.entries(FOOTER_LINKS).map(([title, links]) => (
            <div key={title}>
              <h4 className="text-sm font-semibold text-white mb-4">{title}</h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link.label}>
                    <button
                      onClick={() => scrollTo(link.href)}
                      className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
                    >
                      {link.label}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="mt-12 pt-8 border-t border-white/[0.04] flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-gray-600">
            © {new Date().getFullYear()} Ark Angel Technologies. All rights reserved.
          </p>
          <p className="text-[10px] text-gray-700 text-center sm:text-right max-w-md">
            Disclaimer: ASRE scores are quantitative signals, not investment advice. 
            Consult your SEBI-registered advisor before making investment decisions.
          </p>
        </div>
      </div>
    </footer>
  );
};
