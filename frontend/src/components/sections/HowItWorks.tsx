import React, { useRef, useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import clsx from 'clsx';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';

const SPRING = [0.16, 1, 0.3, 1] as const;

// ─── Step data ────────────────────────────────────────────────────────────────

const STEPS = [
  {
    step: '01',
    title: 'Enter Ticker',
    description: 'Type any NSE/BSE symbol. Our engine pulls latest price, volume, fundamentals, and sector data in real time.',
    detail: 'NSE/BSE native → 4-second latency',
    color: 'from-ark-red to-red-400',
    iconBg: 'bg-ark-red/10 border-ark-red/20 text-ark-red',
    accentRgb: '220,38,38',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    step: '02',
    title: 'ASRE Scores',
    description: 'Computes F-Score (fundamentals), T-Score (technical signals), and M-Score (momentum) using 545+ walk-forward validated iterations.',
    detail: 'F/T/M → 545+ iterations each',
    color: 'from-blue-600 to-blue-400',
    iconBg: 'bg-blue-950/40 border-blue-500/20 text-blue-400',
    accentRgb: '59,130,246',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    step: '03',
    title: 'Hash & Sign',
    description: 'Every score is SHA-256 hashed with a unique Run ID, timestamped before market open. Immutable, traceable, audit-ready.',
    detail: 'SHA-256 + Run ID → Immutable',
    color: 'from-emerald-600 to-emerald-400',
    iconBg: 'bg-emerald-950/40 border-emerald-500/20 text-emerald-400',
    accentRgb: '34,197,94',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
      </svg>
    ),
  },
  {
    step: '04',
    title: 'PDF Report',
    description: 'A compliance-ready PDF with full F/T/M breakdown, SEBI AI disclosure, hash proof, and client-ready formatting.',
    detail: 'SEBI compliant → Client-ready',
    color: 'from-ark-gold to-yellow-400',
    iconBg: 'bg-amber-950/40 border-amber-500/20 text-amber-400',
    accentRgb: '245,158,11',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
];

// ─── Background canvas ────────────────────────────────────────────────────────
//
// Draws three layered systems that reinforce the step narrative:
//
//   A) Data stream particles — thin horizontal lines flying left→right
//      (represents raw market data flowing into the engine)
//
//   B) Animated binary/hex columns — faint falling characters on left edge
//      (represents computation — hash/score processing)
//
//   C) Score convergence arcs — bezier curves converging toward centre
//      (represents F/T/M signals being aggregated)
//
// All systems run in a single rAF loop. Zero DOM reads inside the loop.

const HowItWorksCanvas: React.FC<{ reduced: boolean }> = ({ reduced }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (reduced) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const DPR = Math.min(window.devicePixelRatio, 2);
    let W = 0, H = 0, raf: number, t = 0;

    const resize = () => {
      W = canvas.offsetWidth;
      H = canvas.offsetHeight;
      canvas.width = W * DPR;
      canvas.height = H * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    // ── A) Data stream particles ──────────────────────────────────────────
    // Each particle = a short horizontal dash flying rightward
    interface Particle {
      x: number; y: number;
      speed: number; length: number;
      opacity: number; colorIdx: number;
    }

    const PART_COLORS = [
      [220, 38, 38],  // red
      [59, 130, 246],  // blue
      [34, 197, 94],  // green
      [245, 158, 11],  // amber
      [248, 250, 252],  // white
    ];

    const particles: Particle[] = Array.from({ length: 55 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      speed: 0.4 + Math.random() * 1.2,
      length: 12 + Math.random() * 40,
      opacity: 0.04 + Math.random() * 0.08,
      colorIdx: Math.floor(Math.random() * PART_COLORS.length),
    }));

    const drawParticles = () => {
      particles.forEach(p => {
        p.x += p.speed;
        if (p.x > W + p.length) {
          p.x = -p.length;
          p.y = Math.random() * H;
          p.speed = 0.4 + Math.random() * 1.2;
          p.length = 12 + Math.random() * 40;
        }
        const [r, g, b] = PART_COLORS[p.colorIdx];
        const grad = ctx.createLinearGradient(p.x - p.length, 0, p.x, 0);
        grad.addColorStop(0, `rgba(${r},${g},${b},0)`);
        grad.addColorStop(1, `rgba(${r},${g},${b},${p.opacity})`);
        ctx.beginPath();
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1;
        ctx.moveTo(p.x - p.length, p.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
      });
    };

    // ── B) Computation columns — falling chars ────────────────────────────
    const CHARS = '0123456789ABCDEF∑βαμσ▲▼';
    const COL_SIZE = 18;
    const numCols = Math.ceil(W / COL_SIZE);

    interface Col {
      x: number; y: number;
      speed: number; chars: string[];
      len: number; opacity: number;
    }

    const cols: Col[] = Array.from({ length: numCols }, (_, i) => ({
      x: i * COL_SIZE,
      y: Math.random() * H,
      speed: 0.3 + Math.random() * 0.5,
      chars: Array.from({ length: 8 + Math.floor(Math.random() * 8) }, () =>
        CHARS[Math.floor(Math.random() * CHARS.length)]),
      len: 8 + Math.floor(Math.random() * 8),
      opacity: 0.02 + Math.random() * 0.03,
    }));

    const drawCols = () => {
      ctx.font = '10px "JetBrains Mono", monospace';
      cols.forEach(col => {
        col.y += col.speed;
        if (col.y > H + col.len * COL_SIZE) {
          col.y = -col.len * COL_SIZE;
          col.opacity = 0.02 + Math.random() * 0.03;
        }
        col.chars.forEach((ch, idx) => {
          const cy = col.y + idx * COL_SIZE;
          if (cy < -COL_SIZE || cy > H + COL_SIZE) return;
          // Head char is brighter
          const alpha = idx === col.chars.length - 1
            ? col.opacity * 3.5
            : col.opacity * (1 - idx / col.chars.length);
          ctx.fillStyle = idx === col.chars.length - 1
            ? `rgba(220,38,38,${alpha})`
            : `rgba(248,250,252,${alpha})`;
          ctx.fillText(ch, col.x, cy);
        });
      });
    };

    // ── C) Score convergence arcs ─────────────────────────────────────────
    // Three arcs (F, T, M) originate from left, right, and top edges
    // and converge toward a central "score nexus" point.
    // They breathe — opacity and curvature oscillate with t.

    const drawConvergenceArcs = () => {
      const cx = W * 0.5;   // nexus x
      const cy = H * 0.48;  // nexus y

      const arcs = [
        // F-Score — from top-left
        {
          sx: W * 0.04, sy: H * 0.18,
          cpx: W * 0.2, cpy: H * 0.35,
          col: [220, 38, 38] as [number, number, number],
          label: 'F',
          phase: 0,
        },
        // T-Score — from top-right
        {
          sx: W * 0.96, sy: H * 0.22,
          cpx: W * 0.78, cpy: H * 0.3,
          col: [59, 130, 246] as [number, number, number],
          label: 'T',
          phase: 1.2,
        },
        // M-Score — from bottom-left
        {
          sx: W * 0.08, sy: H * 0.82,
          cpx: W * 0.25, cpy: H * 0.65,
          col: [34, 197, 94] as [number, number, number],
          label: 'M',
          phase: 2.4,
        },
        // β — from bottom-right
        {
          sx: W * 0.94, sy: H * 0.78,
          cpx: W * 0.75, cpy: H * 0.62,
          col: [245, 158, 11] as [number, number, number],
          label: 'β',
          phase: 0.7,
        },
      ];

      arcs.forEach(arc => {
        const pulse = 0.5 + 0.5 * Math.sin(t * 0.9 + arc.phase);
        const alpha = 0.06 + 0.06 * pulse;
        const [r, g, b] = arc.col;

        // Glow pass
        ctx.beginPath();
        ctx.moveTo(arc.sx, arc.sy);
        ctx.quadraticCurveTo(arc.cpx, arc.cpy, cx, cy);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha * 0.4})`;
        ctx.lineWidth = 4;
        ctx.shadowColor = `rgba(${r},${g},${b},0.3)`;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Crisp line pass
        ctx.beginPath();
        ctx.moveTo(arc.sx, arc.sy);
        ctx.quadraticCurveTo(arc.cpx, arc.cpy, cx, cy);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha * 1.8})`;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Animated dot travelling along the arc
        const progress = (t * 0.18 + arc.phase * 0.15) % 1;
        // Quadratic bezier point
        const bx = (1 - progress) * (1 - progress) * arc.sx
          + 2 * (1 - progress) * progress * arc.cpx
          + progress * progress * cx;
        const by = (1 - progress) * (1 - progress) * arc.sy
          + 2 * (1 - progress) * progress * arc.cpy
          + progress * progress * cy;

        ctx.beginPath();
        ctx.arc(bx, by, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${0.55 + 0.3 * pulse})`;
        ctx.fill();

        // Label at origin
        ctx.font = '600 9px "JetBrains Mono", monospace';
        ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + 0.15 * pulse})`;
        ctx.fillText(arc.label, arc.sx - 6, arc.sy - 6);
      });

      // Nexus point — glowing dot where all arcs meet
      const nexusPulse = 0.5 + 0.5 * Math.sin(t * 1.4);
      ctx.beginPath();
      ctx.arc(cx, cy, 3 + nexusPulse * 2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(220,38,38,${0.5 + 0.3 * nexusPulse})`;
      ctx.fill();

      // Nexus outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, 6 + nexusPulse * 8, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(220,38,38,${0.15 * (1 - nexusPulse)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    // ── Main loop ─────────────────────────────────────────────────────────
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      t += 0.006;
      drawCols();
      drawParticles();
      drawConvergenceArcs();
      raf = requestAnimationFrame(draw);
    };

    const onVis = () => {
      if (document.hidden) cancelAnimationFrame(raf);
      else raf = requestAnimationFrame(draw);
    };
    document.addEventListener('visibilitychange', onVis);
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [reduced]);

  if (reduced) return null;

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    />
  );
};

// ─── Step active glow — fires when a step enters view ─────────────────────────
// Each step card emits a matching accent glow behind it when visible

const StepGlow: React.FC<{ accentRgb: string; visible: boolean }> = ({
  accentRgb, visible,
}) => (
  <motion.div
    className="absolute -inset-2 rounded-2xl pointer-events-none"
    style={{
      background: `radial-gradient(ellipse at 20% 50%, rgba(${accentRgb},0.06) 0%, transparent 70%)`,
    }}
    animate={{ opacity: visible ? 1 : 0 }}
    transition={{ duration: 0.6 }}
    aria-hidden="true"
  />
);

// ─── Step card ────────────────────────────────────────────────────────────────

const StepCard: React.FC<{
  step: typeof STEPS[0];
  index: number;
  reduced: boolean;
}> = ({ step, index, reduced }) => {
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.25, once: true });

  return (
    <motion.div
      ref={ref}
      className="relative"
      initial={reduced ? false : { opacity: 0, x: -36 }}
      animate={inView ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.55, delay: index * 0.1, ease: SPRING }}
    >
      {/* Per-step ambient glow */}
      <StepGlow accentRgb={step.accentRgb} visible={inView} />

      <div className="flex items-start gap-3 sm:gap-6 relative">

        {/* Icon circle */}
        <div className="relative flex-shrink-0 mt-0.5 z-10">
          <motion.div
            className={clsx(
              'w-10 h-10 sm:w-12 sm:h-12 rounded-full border flex items-center justify-center',
              step.iconBg,
            )}
            initial={reduced ? false : { scale: 0, rotate: -90 }}
            animate={inView ? { scale: 1, rotate: 0 } : {}}
            transition={{ duration: 0.5, delay: index * 0.1 + 0.15, ease: SPRING }}
          >
            {step.icon}
          </motion.div>

          {/* Entry ping */}
          {inView && !reduced && (
            <motion.div
              className={clsx('absolute inset-0 rounded-full border-2', step.iconBg)}
              initial={{ scale: 1, opacity: 0.6 }}
              animate={{ scale: 2, opacity: 0 }}
              transition={{ duration: 1.0, delay: index * 0.1 + 0.3, ease: 'easeOut' }}
            />
          )}

          {/* Continuous subtle pulse ring on visible steps */}
          {inView && !reduced && (
            <motion.div
              className="absolute inset-0 rounded-full"
              style={{
                border: `1px solid rgba(${step.accentRgb},0.25)`,
              }}
              animate={{ scale: [1, 1.5, 1], opacity: [0.4, 0, 0.4] }}
              transition={{ duration: 3, repeat: Infinity, delay: index * 0.4, ease: 'easeInOut' }}
            />
          )}
        </div>

        {/* Content card */}
        <motion.div
          className="glass-card p-4 sm:p-6 flex-1 min-w-0 group relative overflow-hidden"
          whileHover={reduced ? {} : {
            borderColor: `rgba(${step.accentRgb},0.22)`,
            backgroundColor: 'rgba(255,255,255,0.04)',
          }}
          transition={{ duration: 0.22 }}
        >
          {/* Card inner shimmer on hover */}
          {!reduced && (
            <motion.div
              className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
              style={{
                background: `linear-gradient(105deg,
                  transparent 30%,
                  rgba(${step.accentRgb},0.04) 50%,
                  transparent 70%)`,
              }}
              aria-hidden="true"
            />
          )}

          <div className="flex items-center gap-2 mb-1.5 relative z-10">
            <span className={clsx(
              'text-[10px] font-mono bg-gradient-to-r bg-clip-text text-transparent',
              step.color,
            )}>
              {step.step}
            </span>
            <h3 className="text-base sm:text-lg font-bold text-white">{step.title}</h3>

            {/* Live indicator dot — only on step 01 and 02 (active pipeline steps) */}
            {(index === 0 || index === 1) && inView && !reduced && (
              <motion.span
                className="w-1.5 h-1.5 rounded-full ml-1"
                style={{ backgroundColor: `rgb(${step.accentRgb})` }}
                animate={{ opacity: [1, 0.2, 1], scale: [1, 0.7, 1] }}
                transition={{ duration: 1.8, repeat: Infinity }}
              />
            )}
          </div>

          <p className="text-sm text-gray-400 leading-relaxed mb-2 relative z-10">
            {step.description}
          </p>

          <motion.div
            className="flex items-center gap-2 relative z-10"
            initial={reduced ? false : { opacity: 0 }}
            animate={inView ? { opacity: 1 } : {}}
            transition={{ delay: index * 0.1 + 0.4, duration: 0.4 }}
          >
            {/* Accent bar */}
            <motion.div
              className="h-px flex-1 max-w-[40px]"
              style={{
                transformOrigin: 'left',
                background: `linear-gradient(to right, rgba(${step.accentRgb},0.6), transparent)`,
                height: '1px',
                maxWidth: '40px',
                flex: '1',
              }}
              initial={reduced ? false : { scaleX: 0 }}
              animate={inView ? { scaleX: 1 } : {}}
              transition={{ delay: index * 0.1 + 0.5, duration: 0.5, ease: SPRING }}
            />
            <span className={clsx(
              'text-[10px] sm:text-xs font-mono bg-gradient-to-r bg-clip-text text-transparent',
              step.color,
            )}>
              {step.detail}
            </span>
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  );
};

// ─── Connector line ───────────────────────────────────────────────────────────

const ConnectorLine: React.FC<{ reduced: boolean }> = ({ reduced }) => {
  const lineRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: lineRef,
    offset: ['start 80%', 'end 20%'],
  });
  const scaleY = useTransform(scrollYProgress, [0, 1], [0, 1]);

  return (
    <div
      ref={lineRef}
      className="absolute left-5 sm:left-6 top-5 bottom-5 w-px hidden sm:block overflow-hidden"
      aria-hidden="true"
    >
      {/* Base line */}
      <div className="absolute inset-0 bg-white/[0.04]" />

      {/* Scroll-driven fill */}
      {!reduced && (
        <motion.div
          className="absolute top-0 left-0 right-0 origin-top"
          style={{
            scaleY,
            height: '100%',
            background: 'linear-gradient(to bottom, #DC2626 0%, rgba(220,38,38,0.4) 60%, transparent 100%)',
          }}
        />
      )}

      {/* Travelling dot on the fill line */}
      {!reduced && (
        <motion.div
          className="absolute left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-ark-red"
          style={{ top: useTransform(scrollYProgress, [0, 1], ['0%', '95%']) }}
        />
      )}
    </div>
  );
};

// ─── Section ─────────────────────────────────────────────────────────────────

export const HowItWorks: React.FC = () => {
  const [headerRef, headerVisible] = useInView<HTMLDivElement>({ threshold: 0.3, once: true });
  const reduced = useReducedMotion();

  return (
    <section
      id="how-it-works"
      className="section-padding relative overflow-hidden"
    >

      {/* ── Canvas background ── */}
      <HowItWorksCanvas reduced={reduced} />

      {/* ── Dark base overlay — keeps content readable over canvas ── */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 90% 70% at 50% 50%, rgba(6,6,8,0.55) 0%, rgba(6,6,8,0.88) 100%)',
        }}
        aria-hidden="true"
      />

      {/* ── Ambient static glows ── */}
      {/* Top-left: red (step 01 / F-Score) */}
      <div
        className="absolute top-0 left-0 w-[320px] h-[280px] bg-ark-red/[0.035] rounded-full blur-[90px] pointer-events-none"
        aria-hidden="true"
      />
      {/* Bottom-right: blue (step 02 / T-Score) */}
      <div
        className="absolute bottom-0 right-0 w-[300px] h-[260px] bg-blue-600/[0.03] rounded-full blur-[80px] pointer-events-none"
        aria-hidden="true"
      />
      {/* Centre: emerald (step 03 / Hash) */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] bg-emerald-900/[0.025] rounded-full blur-[100px] pointer-events-none"
        aria-hidden="true"
      />

      {/* ── Content ── */}
      <div className="section-container relative z-10">

        {/* Section header */}
        <motion.div
          ref={headerRef}
          className="text-center mb-10 sm:mb-16"
          initial={reduced ? false : { opacity: 0, y: 28 }}
          animate={headerVisible ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, ease: SPRING }}
        >
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-red border border-ark-red/20 rounded-full mb-4 uppercase tracking-wider">
            How It Works
          </span>
          <h2 className="section-title text-white mb-4">
            Ticker to PDF in{' '}
            <span className="gradient-text">4 Seconds</span>
          </h2>
          <p className="section-subtitle mx-auto">
            From NSE/BSE symbol to a hash-signed, SEBI-compliant research report.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="relative max-w-3xl mx-auto">
          <ConnectorLine reduced={reduced} />

          <div className="space-y-4 sm:space-y-8">
            {STEPS.map((step, i) => (
              <StepCard key={step.step} step={step} index={i} reduced={reduced} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
