import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';
import clsx from 'clsx';
import { PopupModal, useCalendlyEventListener } from 'react-calendly';
import { useReducedMotion } from '../../hooks/useReducedMotion';

// ── Replace with your Calendly link ──────────────────────────────────────────
const CALENDLY_URL = 'https://calendly.com/mithunvk216/30min';

const NAV_LINKS = [
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'ASRE Engine', href: '#asre-engine' },
  { label: 'Demo', href: '#demo' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'Compliance', href: '#compliance' },
];

const SPRING = [0.16, 1, 0.3, 1] as const;
const EXIT = [0.4, 0, 1, 1] as const;

const topBarVariants = { closed: { rotate: 0, y: 0 }, open: { rotate: 45, y: 8 } };
const midBarVariants = { closed: { opacity: 1, scaleX: 1 }, open: { opacity: 0, scaleX: 0 } };
const botBarVariants = { closed: { rotate: 0, y: 0 }, open: { rotate: -45, y: -8 } };

const mobileMenuVariants = {
  hidden: { height: 0, opacity: 0 },
  visible: { height: 'auto' as const, opacity: 1, transition: { duration: 0.35, ease: SPRING } },
  exit: { height: 0, opacity: 0, transition: { duration: 0.25, ease: EXIT } },
};

const menuItemVariants = {
  hidden: { opacity: 0, x: -12 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3, ease: SPRING } },
};

// ── Calendly popup hook — isolated so event listener is always mounted ────────
const useCalendlyModal = () => {
  const [open, setOpen] = useState(false);
  const [booked, setBooked] = useState(false);

  useCalendlyEventListener({
    onEventScheduled: () => {
      setBooked(true);
      setOpen(false);
      // Auto-dismiss toast after 5 s
      setTimeout(() => setBooked(false), 5000);
    },
  });

  return { open, setOpen, booked, setBooked };
};

// ── Demo button — shared between desktop + mobile ─────────────────────────────
interface DemoButtonProps {
  onClick: () => void;
  reduced: boolean;
  className?: string;
  mobile?: boolean;
}

const DemoButton: React.FC<DemoButtonProps> = ({ onClick, reduced, className = '', mobile = false }) => (
  <motion.button
    onClick={onClick}
    id={mobile ? 'mobile-book-demo' : 'nav-book-demo'}
    className={clsx(
      'relative overflow-hidden font-semibold bg-ark-red hover:bg-red-700 text-white rounded-xl',
      'shadow-lg shadow-ark-red/25 transition-colors duration-200',
      'flex items-center justify-center gap-2',
      mobile ? 'w-full px-4 py-3 text-sm' : 'px-5 py-2.5 text-sm',
      className,
    )}
    whileHover={reduced ? {} : {
      y: mobile ? 0 : -2,
      boxShadow: '0 8px 24px rgba(220,38,38,0.4)',
    }}
    whileTap={reduced ? {} : { scale: 0.96 }}
    transition={{ duration: 0.2 }}
    aria-label="Open demo booking calendar"
  >
    {/* Shimmer sweep */}
    {!reduced && (
      <motion.span
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.13) 50%, transparent 70%)',
        }}
        animate={{ x: ['-100%', '200%'] }}
        transition={{ duration: 2.2, repeat: Infinity, repeatDelay: 3.5, ease: 'linear' }}
        aria-hidden="true"
      />
    )}

    {/* Calendar icon */}
    <svg className="w-4 h-4 shrink-0 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>

    <span className="relative z-10">Schedule a Meeting</span>
  </motion.button>
);

// ── Booking confirmed toast ───────────────────────────────────────────────────
const BookedToast: React.FC<{ onDismiss: () => void }> = ({ onDismiss }) => (
  <motion.div
    className={clsx(
      'fixed bottom-6 right-6 z-[9999]',
      'flex items-center gap-3 px-4 py-3 rounded-xl',
      'bg-emerald-950/90 border border-emerald-500/30',
      'backdrop-blur-md shadow-xl',
    )}
    initial={{ opacity: 0, y: 24, scale: 0.95 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    exit={{ opacity: 0, y: 12, scale: 0.95 }}
    transition={{ duration: 0.4, ease: SPRING }}
    role="status"
    aria-live="polite"
  >
    <span className="text-emerald-400 shrink-0">
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    </span>
    <div>
      <p className="text-sm font-semibold text-white">Meeting scheduled!</p>
      <p className="text-xs text-gray-400">Check your inbox for the confirmation.</p>
    </div>
    <button
      className="ml-2 text-gray-500 hover:text-gray-300 transition-colors"
      onClick={onDismiss}
      aria-label="Dismiss notification"
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </motion.div>
);

// ─────────────────────────────────────────────────────────────────────────────
export const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeLink, setActiveLink] = useState('');
  const reduced = useReducedMotion();
  const { scrollY } = useScroll();

  const { open: calendlyOpen, setOpen: setCalendlyOpen, booked, setBooked } =
    useCalendlyModal();

  // Calendly popup needs a real DOM root element
  const [rootEl, setRootEl] = useState<HTMLElement | null>(null);
  useEffect(() => {
    setRootEl(document.getElementById('root') ?? document.body);
  }, []);

  // Scroll detection
  useMotionValueEvent(scrollY, 'change', (latest) => setScrolled(latest > 20));

  // Active section detection
  useEffect(() => {
    const sections = NAV_LINKS.map(l => document.querySelector(l.href));
    const observer = new IntersectionObserver(
      entries => entries.forEach(e => {
        if (e.isIntersecting) setActiveLink(`#${e.target.id}`);
      }),
      { threshold: 0.4 },
    );
    sections.forEach(s => s && observer.observe(s));
    return () => observer.disconnect();
  }, []);

  // Close mobile menu on desktop resize
  useEffect(() => {
    const onResize = () => { if (window.innerWidth >= 1024) setMobileOpen(false); };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const scrollTo = (href: string) => {
    setMobileOpen(false);
    document.querySelector(href)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const openCalendly = () => {
    setMobileOpen(false); // close mobile menu if open
    setCalendlyOpen(true);
  };

  const springT = reduced ? { duration: 0 } : { duration: 0.5, ease: SPRING };

  return (
    <>
      {/* ── Nav ─────────────────────────────────────────────────────────── */}
      <motion.nav
        id="navbar"
        className="fixed top-0 left-0 right-0 z-50"
        animate={scrolled ? 'scrolled' : 'top'}
        variants={{
          top: {
            backgroundColor: 'rgba(15,15,15,0)',
            borderColor: 'rgba(255,255,255,0)',
            boxShadow: '0 0 0 rgba(0,0,0,0)',
          },
          scrolled: {
            backgroundColor: 'rgba(15,15,15,0.85)',
            borderColor: 'rgba(220,38,38,0.15)',
            boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
          },
        }}
        transition={springT}
        style={{
          borderBottomWidth: 1,
          borderBottomStyle: 'solid',
          backdropFilter: scrolled ? 'blur(20px)' : 'blur(0px)',
        }}
      >
        <div className="section-container">
          <div className="flex items-center justify-between h-16 lg:h-20">

            {/* ── Logo ── */}
            <motion.a
              href="#"
              className="flex items-center gap-2.5 group"
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              whileHover={reduced ? {} : { scale: 1.02 }}
              whileTap={reduced ? {} : { scale: 0.98 }}
              transition={{ duration: 0.2 }}
            >
              <div className="w-10 h-10 rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-black/30 group-hover:shadow-ark-red/20 transition-shadow flex-shrink-0">
                <img
                  src="/Logo.jpeg"
                  alt="Ark Angel Logo"
                  width={40} height={40}
                  loading="eager"
                  className="w-full h-full object-contain bg-ark-bg-secondary"
                />
              </div>
              <div>
                <span className="text-white font-bold text-lg tracking-tight">Ark Angel</span>
                <span className="hidden sm:block text-[9px] text-gray-500 uppercase tracking-[0.2em] -mt-0.5">
                  Research Infrastructure
                </span>
              </div>
            </motion.a>

            {/* ── Desktop Nav Links ── */}
            <div className="hidden lg:flex items-center gap-1">
              {NAV_LINKS.map((link) => (
                <motion.button
                  key={link.href}
                  onClick={() => scrollTo(link.href)}
                  className={clsx(
                    'relative px-3 py-2 text-sm rounded-lg transition-colors duration-200',
                    activeLink === link.href
                      ? 'text-white bg-white/[0.06]'
                      : 'text-gray-400 hover:text-white hover:bg-white/[0.04]',
                  )}
                  whileHover={reduced ? {} : { y: -1 }}
                  whileTap={reduced ? {} : { y: 0 }}
                  transition={{ duration: 0.15 }}
                >
                  {link.label}
                  <motion.span
                    className="absolute bottom-0.5 left-3 right-3 h-px bg-ark-red rounded-full"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: activeLink === link.href ? 1 : 0 }}
                    transition={{ duration: 0.3, ease: SPRING }}
                    style={{ transformOrigin: 'left center' }}
                  />
                </motion.button>
              ))}
            </div>

            {/* ── Desktop CTAs ── */}
            <div className="hidden lg:flex items-center gap-3">
              {/* Sample report download */}
              <motion.a
                href="/sample-report.pdf"
                download="TCS_ASRE_Report.pdf"
                id="nav-download-report"
                className="px-4 py-2 text-sm text-gray-300 hover:text-white border border-white/10 hover:border-ark-red/30 rounded-xl transition-colors duration-300 flex items-center gap-1.5"
                whileHover={reduced ? {} : { y: -2 }}
                whileTap={reduced ? {} : { y: 0, scale: 0.97 }}
                transition={{ duration: 0.2 }}
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Sample Report
              </motion.a>

              {/* ✅ Calendly demo button */}
              <DemoButton onClick={openCalendly} reduced={reduced} />
            </div>

            {/* ── Animated Hamburger ── */}
            <motion.button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="lg:hidden text-gray-400 hover:text-white p-2 touch-target flex flex-col justify-center items-center gap-[5px]"
              aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={mobileOpen}
              whileTap={reduced ? {} : { scale: 0.9 }}
            >
              {([topBarVariants, midBarVariants, botBarVariants] as const).map((barV, i) => (
                <motion.span
                  key={i}
                  className="block w-6 h-[2px] bg-current rounded-full origin-center"
                  variants={barV}
                  animate={mobileOpen ? 'open' : 'closed'}
                  transition={reduced ? { duration: 0 } : { duration: 0.3, ease: SPRING }}
                />
              ))}
            </motion.button>
          </div>

          {/* ── Mobile Menu ── */}
          <AnimatePresence>
            {mobileOpen && (
              <motion.div
                className="lg:hidden overflow-hidden"
                variants={mobileMenuVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
              >
                <div className="space-y-1 pt-2 pb-6">
                  {NAV_LINKS.map((link, i) => (
                    <motion.button
                      key={link.href}
                      initial="hidden"
                      animate="visible"
                      variants={menuItemVariants}
                      transition={{ delay: i * 0.04, duration: 0.3, ease: SPRING }}
                      onClick={() => scrollTo(link.href)}
                      className={clsx(
                        'block w-full text-left px-4 py-3 text-sm rounded-lg transition-colors',
                        activeLink === link.href
                          ? 'text-white bg-white/[0.06] border-l-2 border-ark-red pl-3'
                          : 'text-gray-300 hover:text-white hover:bg-white/[0.04]',
                      )}
                    >
                      {link.label}
                    </motion.button>
                  ))}

                  {/* Mobile CTAs */}
                  <motion.div
                    className="pt-4 space-y-2 px-4"
                    initial="hidden"
                    animate="visible"
                    variants={menuItemVariants}
                    transition={{ delay: NAV_LINKS.length * 0.04, duration: 0.3, ease: SPRING }}
                  >
                    <a
                      href="/sample-report.pdf"
                      download="TCS_ASRE_Report.pdf"
                      className="block w-full text-center px-4 py-3 text-sm text-gray-300 border border-white/10 rounded-xl hover:border-ark-red/30 transition-colors"
                    >
                      Download Sample Report (TCS)
                    </a>

                    {/* ✅ Calendly demo button — mobile */}
                    <DemoButton
                      onClick={openCalendly}
                      reduced={reduced}
                      mobile
                    />
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.nav>

      {/* ── Calendly popup modal (portalled to body/root) ─────────────── */}
      {rootEl && (
        <PopupModal
          url={CALENDLY_URL}
          open={calendlyOpen}
          onModalClose={() => setCalendlyOpen(false)}
          rootElement={rootEl}
          pageSettings={{
            backgroundColor: '060608',
            primaryColor: 'DC2626',
            textColor: 'e8e6ff',
            hideLandingPageDetails: false,
            hideEventTypeDetails: false,
          }}
          utm={{
            utmSource: 'ark-angl-website',
            utmMedium: 'navbar-cta',
            utmCampaign: 'book-demo',
          }}
        />
      )}

      {/* ── Booking confirmed toast ────────────────────────────────────── */}
      <AnimatePresence>
        {booked && <BookedToast onDismiss={() => setBooked(false)} />}
      </AnimatePresence>
    </>
  );
};
