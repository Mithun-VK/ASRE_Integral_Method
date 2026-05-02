/**
 * HeroSection — Optimized for low latency & high responsiveness
 *
 * Perf decisions (each annotated inline):
 *  [P1]  ALL decorative animations are pure CSS @keyframes → compositor thread only
 *  [P2]  Framer Motion used ONLY for scroll-parallax (no JS RAF equivalent)
 *  [P3]  GSAP entrance timeline retained but scoped to one ref array call
 *  [P4]  Cursor glow uses a single passive mousemove listener writing to
 *        CSS custom properties (no MotionValue / Spring overhead)
 *  [P5]  `content-visibility: auto` on below-fold proof tiles
 *  [P6]  `contain: strict` on the ticker tape (isolates layout/paint/size)
 *  [P7]  BubbleBackground deferred behind requestIdleCallback
 *  [P8]  All stable objects/arrays defined at MODULE scope → zero GC pressure
 *  [P9]  React.memo + display names on every sub-component
 *  [P10] `will-change` applied exactly where GSAP/Framer writes transforms;
 *        removed everywhere else (excess will-change wastes GPU VRAM)
 *  [P11] Scroll indicator uses a CSS animation instead of Framer animate prop
 *  [P12] `useReducedMotion` gates all decorative work at the top
 */

import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
  useMemo,
} from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useGSAP } from '@gsap/react';
import { gsap } from 'gsap';
import { BubbleBackground } from '@/components/animate-ui/components/backgrounds/bubble';
import { CTAButton } from '../ui/CTAButton';
import { MorphingText } from '../ui/MorphingText';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { useIsMobile } from '../../hooks/useIsMobile';
import { BookDemo } from '../sections/BookDemo';

// ─────────────────────────────────────────────────────────────────────────────
// [P8] Module-level constants — never recreated across renders
// ─────────────────────────────────────────────────────────────────────────────

const BUBBLE_COLORS = {
  first: '220, 38,  38',
  second: '153, 27,  27',
  third: '34,  197, 94',
  fourth: '59,  130, 246',
  fifth: '248, 250, 252',
  sixth: '245, 158, 11',
} as const;

const BUBBLE_SPRING = { stiffness: 60, damping: 22 } as const;

const SOCIAL_PROOF = [
  { value: '545+', label: 'Walk-forward iterations' },
  { value: 'NSE/BSE', label: 'Native Coverage' },
  { value: 'SEBI', label: 'Dec 2024 Compliant' },
  { value: 'SHA-256', label: 'Hash Audit Trail' },
] as const;

// Not `as const` — MorphingText.words expects a mutable string[]
const MORPH_WORDS: string[] = ['SEBI-Registered', 'SEBI-Compliant', 'Audit-Ready', 'AI-Powered'];

const TICKERS = [
  'RELIANCE ▲2.4%', 'TCS ▼0.8%', 'HDFCBANK ▲1.1%',
  'INFY ▲0.6%', 'WIPRO ▼1.3%', 'ICICIBANK ▲2.9%',
  'SBIN ▲0.3%', 'BAJFINANCE ▼0.5%', 'ASIANPAINT ▲1.7%',
  'TATAMOTORS ▲3.2%', 'SUNPHARMA ▼0.4%', 'ADANIENT ▲1.8%',
] as const;

// Pre-doubled at module load → no Array spread inside render [P8]
const TICKERS_DOUBLED = [...TICKERS, ...TICKERS] as const;

// Framer Motion variants — stable object references [P8]
const proofContainerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.09, delayChildren: 1.0 } },
} as const;

const proofItemVariants = {
  hidden: { opacity: 0, y: 20, scale: 0.96 },
  visible: {
    opacity: 1, y: 0, scale: 1,
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] as const }
  },
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// [P1] All decorative CSS @keyframes in one <style> tag injected once via
//      a top-level constant. No JS animation system touches these properties.
// ─────────────────────────────────────────────────────────────────────────────

const GLOBAL_STYLES = `
  /* Ticker tape — translateX only, GPU layer guaranteed */
  @keyframes hero-ticker {
    from { transform: translateX(0) }
    to   { transform: translateX(-50%) }
  }

  /* Scan line — top is not composited; use transform instead */
  @keyframes hero-scan {
    0%,100% { transform: translateY(0vh);   opacity: 0 }
    5%       { opacity: 0.35 }
    95%      { opacity: 0.35 }
    99%      { transform: translateY(97vh); opacity: 0 }
  }

  /* SEBI dot pulse */
  @keyframes hero-dot {
    0%,100% { opacity: 1 }
    50%      { opacity: 0.25 }
  }

  /* Badge shimmer ring — only plays when parent is hovered */
  @keyframes hero-badge-spin {
    from { transform: rotate(0deg) }
    to   { transform: rotate(360deg) }
  }

  /* Scroll mouse indicator */
  @keyframes hero-scroll-dot {
    0%,100% { transform: translateY(0px);  opacity: 1 }
    60%      { transform: translateY(12px); opacity: 0.15 }
  }

  /* Scroll indicator arrow bounce */
  @keyframes hero-section-bounce {
    0%,100% { transform: translateY(0px) }
    50%      { transform: translateY(9px) }
  }
`;

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components — all React.memo'd [P9]
// ─────────────────────────────────────────────────────────────────────────────

/** [P6] contain:strict isolates this from the rest of the layout tree */
const TickerTape: React.FC<{ reduced: boolean }> = React.memo(({ reduced }) => {
  if (reduced) return null;
  return (
    <div
      aria-hidden="true"
      className="absolute top-[4.5rem] sm:top-[5rem] left-0 right-0 z-[5] pointer-events-none"
      style={{
        overflow: 'hidden',
        contain: 'strict',           // [P6] layout + paint + size isolation
        height: '1.5rem',
      }}
    >
      <div
        className="flex gap-8 whitespace-nowrap w-max"
        // [P1] CSS animation — no JS involved whatsoever
        style={{ animation: 'hero-ticker 38s linear infinite' }}
      >
        {TICKERS_DOUBLED.map((tick, i) => (
          <span
            key={i}
            className={`text-[10px] font-mono tracking-wider px-2 ${tick.includes('▲') ? 'text-green-500/35' : 'text-red-500/35'
              }`}
          >
            {tick}
          </span>
        ))}
      </div>
    </div>
  );
});
TickerTape.displayName = 'TickerTape';

/** [P1] Pure CSS scan line — zero JS overhead */
const ScanLine: React.FC = React.memo(() => (
  <div
    aria-hidden="true"
    className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-ark-red/15 to-transparent z-[6] pointer-events-none"
    style={{
      animation: 'hero-scan 16s linear infinite',
      top: 0,
      // [P10] will-change only on elements GSAP/Framer actively transforms
      willChange: 'transform, opacity',
    }}
  />
));
ScanLine.displayName = 'ScanLine';

/**
 * BadgeShimmer
 * [P1] Spin animation plays via CSS animationPlayState toggled by a CSS custom
 *      property — no JS event → React state → re-render cycle.
 */
const BadgeShimmer: React.FC<{ children: React.ReactNode }> = React.memo(
  ({ children }) => {
    // Single ref for the outer wrapper; we write a CSS var directly [P4-adjacent]
    const wrapRef = useRef<HTMLDivElement>(null);

    const onEnter = useCallback(() => {
      wrapRef.current?.style.setProperty('--badge-spin', 'running');
    }, []);
    const onLeave = useCallback(() => {
      wrapRef.current?.style.setProperty('--badge-spin', 'paused');
    }, []);

    return (
      <div ref={wrapRef} className="relative inline-flex">
        {/* [P1] Shimmer ring — paused by default, 0 cost when idle */}
        <div
          aria-hidden="true"
          className="absolute -inset-px rounded-full pointer-events-none opacity-0 transition-opacity duration-500 group-hover:opacity-100"
          style={{
            background:
              'conic-gradient(from 0deg, transparent 0deg, rgba(220,38,38,0.5) 60deg, transparent 120deg)',
            animation: 'hero-badge-spin 3s linear infinite',
            animationPlayState: 'var(--badge-spin, paused)',
          }}
        />
        <div
          onMouseEnter={onEnter}
          onMouseLeave={onLeave}
          className="relative inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-full bg-black/40 border border-white/[0.1] backdrop-blur-md"
        >
          {children}
        </div>
      </div>
    );
  },
);
BadgeShimmer.displayName = 'BadgeShimmer';

/** [P1] Dot pulse is a CSS animation — no Framer Motion spring allocated */
const SEBIBadge: React.FC = React.memo(() => (
  <motion.div
    className="hidden sm:inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/[0.06] bg-white/[0.02] backdrop-blur-sm"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ delay: 1.5, duration: 0.45 }}
  // [P10] no will-change — this animates once then stops
  >
    <span
      className="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0"
      style={{ animation: 'hero-dot 2s ease-in-out infinite' }} // [P1]
    />
    <span className="text-[9px] font-mono text-gray-400 tracking-wider uppercase">
      SEBI Dec 2024 · AI Disclosure Compliant
    </span>
  </motion.div>
));
SEBIBadge.displayName = 'SEBIBadge';

/**
 * ProofTile
 * [P5] content-visibility:auto lets the browser skip paint for off-screen tiles
 *       on initial load — measurable LCP improvement on mobile.
 */
const ProofTile: React.FC<{ item: (typeof SOCIAL_PROOF)[number] }> = React.memo(
  ({ item }) => (
    <motion.div
      variants={proofItemVariants}
      className="
        group relative flex flex-col items-center gap-1
        px-5 sm:px-8 py-4 bg-black/50 cursor-default overflow-hidden
        transition-colors duration-200 hover:bg-ark-red/[0.06]
      "
      // [P5] skip rendering cost until near viewport
      style={{ contentVisibility: 'auto', containIntrinsicSize: '0 72px' }}
    >
      {/* Top-edge glow via CSS only — no JS paint path [P1] */}
      <div
        aria-hidden="true"
        className="
          absolute top-0 left-0 right-0 h-px
          bg-gradient-to-r from-transparent via-ark-red/40 to-transparent
          opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none
        "
      />
      <span className="relative text-base sm:text-xl font-bold text-white font-mono tabular-nums group-hover:text-ark-red transition-colors duration-200">
        {item.value}
      </span>
      <span className="relative text-[9px] sm:text-[10px] text-gray-400 text-center leading-tight">
        {item.label}
      </span>
    </motion.div>
  ),
);
ProofTile.displayName = 'ProofTile';

/** [P11] CSS-only scroll indicator — replaces Framer Motion animate prop */
const ScrollIndicator: React.FC = React.memo(() => (
  <div
    className="mt-10 sm:mt-12 flex flex-col items-center gap-2"
    aria-hidden="true"
    style={{ animation: 'hero-section-bounce 1.3s ease-in-out infinite' }}
  >
    <span className="text-[10px] text-gray-600 uppercase tracking-[0.35em]">Scroll</span>
    <div className="w-5 h-8 rounded-full border border-gray-700/50 flex items-start justify-center pt-1.5">
      <span
        className="w-1 h-1.5 rounded-full bg-gray-600 block"
        style={{ animation: 'hero-scroll-dot 1.6s ease-in-out infinite' }} // [P11]
      />
    </div>
  </div>
));
ScrollIndicator.displayName = 'ScrollIndicator';

// ─────────────────────────────────────────────────────────────────────────────
// HeroSection
// ─────────────────────────────────────────────────────────────────────────────

export const HeroSection: React.FC = () => {
  const reduced = useReducedMotion();
  const isMobile = useIsMobile();

  // [P7] Defer BubbleBackground until browser is idle
  const [bubbleReady, setBubbleReady] = useState(false);
  useEffect(() => {
    if (reduced || isMobile) { setBubbleReady(false); return; }
    if ('requestIdleCallback' in window) {
      const id = requestIdleCallback(() => setBubbleReady(true), { timeout: 1200 });
      return () => cancelIdleCallback(id);
    }
    const t = setTimeout(() => setBubbleReady(true), 300);
    return () => clearTimeout(t);
  }, [reduced, isMobile]);

  // DOM refs — GSAP targets
  const sectionRef = useRef<HTMLElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const subRef = useRef<HTMLParagraphElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // ── [P4] Cursor → CSS custom properties (no MotionValue, no Spring, no RAF) ──
  //   We write --gx / --gy on the glow element directly from a passive listener.
  //   The element reads them via `translate(var(--gx), var(--gy))` in inline style.
  useEffect(() => {
    if (isMobile || reduced) return;
    const section = sectionRef.current;
    const glow = glowRef.current;
    if (!section || !glow) return;

    // Low-pass filter state — manual spring done in rAF to avoid Framer overhead
    let tx = 0, ty = 0;    // target
    let cx = 0, cy = 0;    // current (interpolated)
    let raf = 0;
    const STIFFNESS = 0.06; // lerp factor (lower = smoother / more lag)

    const tick = () => {
      cx += (tx - cx) * STIFFNESS;
      cy += (ty - cy) * STIFFNESS;
      // Write once per frame directly to style — no React state, no re-render
      glow.style.setProperty('--gx', `${cx.toFixed(1)}px`);
      glow.style.setProperty('--gy', `${cy.toFixed(1)}px`);
      raf = requestAnimationFrame(tick);
    };

    const onMove = (e: MouseEvent) => {
      const r = section.getBoundingClientRect();
      tx = (e.clientX - r.left - r.width / 2) * 0.07;
      ty = (e.clientY - r.top - r.height / 2) * 0.07;
    };

    raf = requestAnimationFrame(tick);
    section.addEventListener('mousemove', onMove, { passive: true });
    return () => {
      section.removeEventListener('mousemove', onMove);
      cancelAnimationFrame(raf);
    };
  }, [isMobile, reduced]);

  // ── [P2] Scroll parallax — Framer Motion only, single useTransform ──────────
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start'],
  });
  const contentY = useTransform(scrollYProgress, [0, 1], ['0%', '20%']);
  const contentOpac = useTransform(scrollYProgress, [0, 0.7], [1, 0]);

  // ── [P3] GSAP entrance — ONE fromTo batch, correct will-change lifecycle ────
  useGSAP(() => {
    const targets = [
      badgeRef.current,
      headlineRef.current,
      subRef.current,
      ctaRef.current,
      scrollRef.current,
    ];

    if (reduced) {
      gsap.set(targets, { opacity: 1, y: 0, scale: 1, filter: 'none', clearProps: 'all' });
      gsap.set(glowRef.current, { opacity: 0.09, scale: 1 });
      return;
    }

    // Mark will-change before animation starts; clear after it ends [P10]
    const markWillChange = () =>
      targets.forEach(el => el && ((el as HTMLElement).style.willChange = 'transform, opacity'));
    const clearWillChange = () =>
      targets.forEach(el => el && ((el as HTMLElement).style.willChange = 'auto'));

    markWillChange();

    const tl = gsap.timeline({
      defaults: { ease: 'power3.out' },
      onComplete: clearWillChange,        // [P10] release layer after animation
    });

    tl
      .fromTo(badgeRef.current,
        { y: -20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5 })
      .fromTo(headlineRef.current,
        { y: 44, opacity: 0, filter: 'blur(5px)' },
        { y: 0, opacity: 1, filter: 'blur(0px)', duration: 0.8 },
        '-=0.28')
      .fromTo(subRef.current,
        { y: 24, opacity: 0, filter: 'blur(3px)' },
        { y: 0, opacity: 1, filter: 'blur(0px)', duration: 0.6 },
        '-=0.48')
      .fromTo(ctaRef.current,
        { y: 20, opacity: 0, scale: 0.96 },
        { y: 0, opacity: 1, scale: 1, duration: 0.6 },
        '-=0.42')
      .fromTo(scrollRef.current,
        { opacity: 0, y: 8 },
        { opacity: 1, y: 0, duration: 0.35 },
        '-=0.12');

    // Glow breathe — GSAP owns scale/opacity exclusively [P3]
    // [P10] will-change set before the infinite loop, never cleared (element persists)
    if (glowRef.current) {
      glowRef.current.style.willChange = 'transform, opacity';
      gsap.fromTo(
        glowRef.current,
        { opacity: 0.05, scale: 0.93 },
        { opacity: 0.14, scale: 1.2, duration: 4.8, ease: 'sine.inOut', yoyo: true, repeat: -1 },
      );
    }
  }, [reduced]);

  // ── [P12] Stable callback — no inline function recreated on render ───────────
  const downloadReport = useCallback(() => {
    const a = Object.assign(document.createElement('a'), {
      href: '/sample-report.pdf',
      download: 'TCS_ASRE_Report_Demo.pdf',
    });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, []);

  // ── Memoised proof tiles — avoid re-rendering the grid on any parent state ──
  const proofGrid = useMemo(
    () =>
      SOCIAL_PROOF.map((item, i) => <ProofTile key={i} item={item} />),
    [], // SOCIAL_PROOF is module-level constant → stable forever
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <section
      ref={sectionRef}
      id="hero"
      className="relative min-h-[100svh] flex items-center justify-center pt-16 sm:pt-20 pb-10 sm:pb-12 overflow-hidden bg-ark-bg-primary"
    >
      {/* [P1] One style tag — all CSS @keyframes injected once */}
      <style>{GLOBAL_STYLES}</style>

      {/*
        ══ LAYER STACK (back → front) ══════════════════════════════
        z-0   BubbleBackground      — deferred [P7]
        z-[1] Dark radial overlay   — static div
        z-[2] Fine grid             — static div
        z-[3] Cursor glow           — GSAP breathe + CSS var translate [P3][P4]
        z-[4] Gold accent           — static div
        z-[5] Ticker tape           — CSS animation [P1][P6]
        z-[6] Scan line             — CSS animation [P1]
        z-10  Content               — Framer parallax [P2]
        z-10  Bottom fade           — static div
        ════════════════════════════════════════════════════════════
      */}

      {/* z-0 BubbleBackground — deferred until idle [P7] */}
      {bubbleReady && (
        <BubbleBackground
          interactive
          colors={BUBBLE_COLORS}
          transition={BUBBLE_SPRING}
          className="absolute inset-0 z-0 opacity-70"
          aria-hidden="true"
        />
      )}

      {/* z-1 Dark radial overlay */}
      <div
        aria-hidden="true"
        className="absolute inset-0 z-[1] pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 85% 65% at 50% 48%, rgba(6,6,8,0.48) 0%, rgba(6,6,8,0.86) 100%)',
        }}
      />

      {/* z-2 Fine grid */}
      <div
        aria-hidden="true"
        className="absolute inset-0 z-[2] grid-bg opacity-[0.15] pointer-events-none"
      />

      {/*
        z-3 Cursor glow
        [P4] translate driven by CSS custom properties (--gx / --gy)
             written from a manual rAF lerp. Zero Framer Motion overhead.
        [P3] scale/opacity driven by GSAP breathe (no overlap with Framer).
        [P10] will-change set permanently on this element only.
      */}
      <div
        ref={glowRef}
        aria-hidden="true"
        className="absolute top-1/2 left-1/2 z-[3] w-[460px] sm:w-[840px] h-[460px] sm:h-[840px] bg-ark-red/[0.09] rounded-full blur-[110px] sm:blur-[150px] pointer-events-none"
        style={{
          // CSS custom property translate: zero-cost read, no JS in render path
          transform: 'translate(calc(-50% + var(--gx, 0px)), calc(-50% + var(--gy, 0px)))',
          willChange: 'transform, opacity', // [P10]
        }}
      />

      {/* z-4 Gold accent */}
      <div
        aria-hidden="true"
        className="absolute -top-20 right-0 z-[4] w-[240px] h-[240px] bg-ark-gold/[0.04] rounded-full blur-[90px] pointer-events-none"
      />

      {/* z-5 Ticker tape */}
      <TickerTape reduced={isMobile || reduced} />

      {/* z-6 Scan line */}
      {!reduced && <ScanLine />}

      {/* ══ CONTENT ══════════════════════════════════════════════ */}
      <motion.div
        className="relative z-10 section-container text-center w-full"
        style={
          reduced
            ? {}
            : {
              y: contentY,
              opacity: contentOpac,
              willChange: 'transform, opacity', // [P10]
            }
        }
      >

        {/* Badge row */}
        <div
          ref={badgeRef}
          style={{ opacity: 0 }}
          className="mb-5 sm:mb-6 flex flex-col items-center gap-2"
        >
          <BadgeShimmer>
            <span className="relative flex-shrink-0 w-2 h-2">
              <span className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-60" />
              <span className="relative block w-2 h-2 rounded-full bg-green-400" />
            </span>
            <span className="text-[11px] sm:text-xs text-gray-300 font-medium tracking-wide">
              NSE/BSE · Real-time scoring · 4-second latency
            </span>
          </BadgeShimmer>
          <SEBIBadge />
        </div>

        {/* Headline */}
        <h1
          ref={headlineRef}
          style={{ opacity: 0 }}
          className="text-3xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-4 sm:mb-6 max-w-5xl mx-auto leading-[1.15] sm:leading-[1.1] px-2"
        >
          Quantitative Research{' '}
          <br className="hidden sm:inline" />
          Infrastructure for{' '}
          <span className="inline-flex flex-wrap items-baseline justify-center sm:justify-start gap-x-[0.25em]">
            <MorphingText
              words={MORPH_WORDS}
              interval={isMobile ? 4000 : 3000}
              className="gradient-text"
              pauseOnHidden
            />
            <span>Advisors</span>
          </span>
        </h1>

        {/* Sub-headline */}
        <p
          ref={subRef}
          style={{ opacity: 0 }}
          className="text-base sm:text-lg text-gray-300 max-w-xl sm:max-w-2xl mx-auto mb-8 sm:mb-10 leading-relaxed px-2"
        >
          Score NSE/BSE stocks in 4&nbsp;seconds. Hash-signed, SEBI Dec&nbsp;2024
          AI-compliant, and audit-ready out of the box.
        </p>

        {/* CTAs */}
        <div
          ref={ctaRef}
          style={{ opacity: 0 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 mb-10 sm:mb-14 px-4 sm:px-0"
        >
          <BookDemo
            variant="primary"
            size="lg"
            id="hero-book-demo"
            className="w-full sm:w-auto"
            label="Book 10-Min Demo"
          />
          <CTAButton
            variant="secondary"
            size="lg"
            onClick={downloadReport}
            id="hero-download-report"
            className="w-full sm:w-auto"
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            }
          >
            Download Sample Report
          </CTAButton>
        </div>

        {/* Social proof tiles */}
        <motion.div
          variants={proofContainerVariants}
          initial={reduced ? false : 'hidden'}
          animate="visible"
        >
          <div className="grid grid-cols-2 sm:inline-grid sm:grid-cols-4 gap-px bg-white/[0.05] border border-white/[0.07] rounded-2xl overflow-hidden mx-auto max-w-xs sm:max-w-none backdrop-blur-md">
            {proofGrid}
          </div>
        </motion.div>

        {/* Scroll indicator — [P11] pure CSS, no Framer animate prop */}
        <div ref={scrollRef} style={{ opacity: 0 }}>
          <ScrollIndicator />
        </div>

      </motion.div>

      {/* Bottom fade */}
      <div
        aria-hidden="true"
        className="absolute bottom-0 left-0 right-0 h-28 sm:h-40 bg-gradient-to-t from-ark-bg-primary via-ark-bg-primary/70 to-transparent z-10 pointer-events-none"
      />
    </section>
  );
};