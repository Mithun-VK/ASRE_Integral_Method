import React, { useState, useEffect } from 'react';
import clsx from 'clsx';

const NAV_LINKS = [
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'ASRE Engine', href: '#asre-engine' },
  { label: 'Demo', href: '#demo' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'Compliance', href: '#compliance' },
  { label: 'Team', href: '#team' },
];

export const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollTo = (href: string) => {
    setMobileOpen(false);
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <nav
      id="navbar"
      className={clsx(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-500',
        scrolled
          ? 'bg-ark-bg-primary/80 backdrop-blur-xl border-b border-white/[0.06] shadow-lg shadow-black/20'
          : 'bg-transparent'
      )}
    >
      <div className="section-container">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <a href="#" className="flex items-center gap-2.5 group" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <div className="w-10 h-10 rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-black/30 group-hover:shadow-ark-red/20 transition-shadow flex-shrink-0">
              <img
                src="/Logo.jpeg"
                alt="Ark Angel Logo"
                className="w-full h-full object-contain bg-ark-bg-secondary"
              />
            </div>
            <div>
              <span className="text-white font-bold text-lg tracking-tight">ARK ANGEL</span>
              <span className="hidden sm:block text-[9px] text-gray-500 uppercase tracking-[0.2em] -mt-0.5">Research Infrastructure</span>
            </div>
          </a>

          {/* Desktop nav */}
          <div className="hidden lg:flex items-center gap-1">
            {NAV_LINKS.map((link) => (
              <button
                key={link.href}
                onClick={() => scrollTo(link.href)}
                className="px-3 py-2 text-sm text-gray-400 hover:text-white rounded-lg hover:bg-white/[0.04] transition-all duration-200"
              >
                {link.label}
              </button>
            ))}
          </div>

          {/* Desktop CTA */}
          <div className="hidden lg:flex items-center gap-3">
            <a
              href="/sample-report.pdf"
              download="TCS_ASRE_Report.pdf"
              className="px-4 py-2 text-sm text-gray-300 hover:text-white border border-white/10 hover:border-ark-red/30 rounded-xl transition-all duration-300 hover:-translate-y-0.5 flex items-center gap-1.5"
              id="nav-download-report"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Sample Report
            </a>
            <button
              onClick={() => scrollTo('#demo')}
              className="px-5 py-2.5 text-sm font-semibold bg-ark-red hover:bg-red-700 text-white rounded-xl shadow-lg shadow-ark-red/25 hover:shadow-ark-red/40 transition-all duration-300 hover:-translate-y-0.5"
              id="nav-book-demo"
            >
              Book 10-Min Demo
            </button>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="lg:hidden text-gray-400 hover:text-white p-2"
            aria-label="Toggle menu"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {mobileOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile menu */}
        <div
          className={clsx(
            'lg:hidden overflow-hidden transition-all duration-300',
            mobileOpen ? 'max-h-96 pb-6' : 'max-h-0'
          )}
        >
          <div className="space-y-1 pt-2">
            {NAV_LINKS.map((link) => (
              <button
                key={link.href}
                onClick={() => scrollTo(link.href)}
                className="block w-full text-left px-4 py-3 text-sm text-gray-300 hover:text-white hover:bg-white/[0.04] rounded-lg transition-colors"
              >
                {link.label}
              </button>
            ))}
            <div className="pt-4 space-y-2 px-4">
              <a
                href="/sample-report.pdf"
                download="TCS_ASRE_Report.pdf"
                className="block w-full text-center px-4 py-3 text-sm text-gray-300 border border-white/10 rounded-xl hover:border-ark-red/30 transition-colors"
              >
                Download Sample Report (TCS)
              </a>
              <button
                onClick={() => scrollTo('#demo')}
                className="block w-full px-4 py-3 text-sm font-semibold bg-ark-red text-white rounded-xl shadow-lg shadow-ark-red/25"
              >
                Book 10-Min Demo
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};
