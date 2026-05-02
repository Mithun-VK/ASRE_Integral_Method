import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { useASREScore } from '../../hooks/useASREScore';
import { ScoreBar } from '../ui/ScoreBadge';
import { DipBadge } from '../ui/DipBadge';
import { HashProof } from '../ui/HashProof';
import { EmailGateModal } from '../ui/DemoModal';
import { CTAButton } from '../ui/CTAButton';
import { useInView } from '../../hooks/useInView';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { SUPPORTED_TICKERS } from '../../services/api';

const SPRING = [0.16, 1, 0.3, 1] as const;
const SUGGESTED = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'BAJFINANCE'];

function SignalBadge({ signal }: { signal: string }) {
  const color = signal.includes('STRONG BUY')
    ? 'text-green-400 bg-green-950/50 border-green-500/30'
    : signal.includes('BUY')
      ? 'text-emerald-400 bg-emerald-950/50 border-emerald-500/30'
      : signal.includes('SELL')
        ? 'text-red-400 bg-red-950/50 border-red-500/30'
        : 'text-gray-400 bg-gray-800/50 border-gray-600/30';

  return (
    <motion.span
      className={clsx('text-xs font-mono font-bold px-2.5 py-1 rounded-full border', color)}
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.35, ease: SPRING }}
    >
      {signal}
    </motion.span>
  );
}

export const DemoWidget: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [showEmailGate, setShowEmailGate] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [inputFocused, setInputFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const reduced = useReducedMotion();

  const {
    score, loading, error, attemptCount,
    needsEmail, dataSource, fetchScore, resetGate,
  } = useASREScore();

  const [headerRef, headerVisible] = useInView<HTMLDivElement>({ threshold: 0.3, once: true });
  const [cardRef, cardVisible] = useInView<HTMLDivElement>({ threshold: 0.1, once: true });

  const handleScore = () => {
    const cleaned = ticker.trim().toUpperCase().replace('.NS', '').replace('.BO', '');
    if (!cleaned) return;
    if (needsEmail) { setShowEmailGate(true); return; }
    if (!SUPPORTED_TICKERS.includes(cleaned)) {
      setLocalError(`'${cleaned}' is not supported. Try: ${SUGGESTED.join(', ')}.`);
      return;
    }
    setLocalError(null);
    fetchScore(cleaned);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleScore();
  };

  const handleSuggest = (t: string) => {
    setTicker(t);
    setLocalError(null);
    if (needsEmail) { setShowEmailGate(true); return; }
    fetchScore(t);
  };

  const handlePdfDownload = () => {
    if (attemptCount >= 1) { setShowEmailGate(true); return; }
    const a = document.createElement('a');
    a.href = '/sample-report.pdf';
    a.download = 'TCS_ASRE_Report_Demo.pdf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleEmailSuccess = () => {
    resetGate();
    setShowEmailGate(false);
  };

  return (
    <>
      <section id="demo" className="section-padding relative overflow-hidden">

        {/* Ambient glow */}
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-ark-red/[0.04] rounded-full blur-[150px] translate-x-1/2 -translate-y-1/4 pointer-events-none" aria-hidden="true" />
        <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-ark-gold/[0.02] rounded-full blur-[100px] pointer-events-none" aria-hidden="true" />

        <div className="section-container relative z-10">

          {/* Section header */}
          <motion.div
            ref={headerRef}
            className="text-center mb-8 sm:mb-12"
            initial={reduced ? false : { opacity: 0, y: 28 }}
            animate={headerVisible ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, ease: SPRING }}
          >
            <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-red border border-ark-red/20 rounded-full mb-4 uppercase tracking-wider">
              Live Demo
            </span>
            <h2 className="section-title text-white mb-3 sm:mb-4">
              Try the <span className="gradient-text">ASRE Engine</span>
            </h2>
            <p className="section-subtitle mx-auto px-2">
              Enter any NSE blue-chip ticker to get real F/T/M scores from the live ASRE engine.
            </p>
          </motion.div>

          {/* Demo card */}
          <motion.div
            ref={cardRef}
            className="max-w-2xl mx-auto"
            initial={reduced ? false : { opacity: 0, y: 40 }}
            animate={cardVisible ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.65, delay: 0.1, ease: SPRING }}
          >
            <motion.div
              className="glass-card p-4 sm:p-6 lg:p-8"
              animate={cardVisible && !reduced ? {
                boxShadow: [
                  '0 0 0px rgba(220,38,38,0)',
                  '0 0 40px rgba(220,38,38,0.12), 0 0 120px rgba(220,38,38,0.04)',
                  '0 0 30px rgba(220,38,38,0.08), 0 0 80px rgba(220,38,38,0.03)',
                ],
              } : {}}
              transition={{ duration: 1.5, delay: 0.4 }}
            >

              {/* Input row */}
              <div className="flex flex-col sm:flex-row gap-3 mb-4">
                <div className="flex-1 relative">

                  {/* Animated focus ring */}
                  <motion.div
                    className="absolute inset-0 rounded-xl pointer-events-none"
                    animate={{
                      boxShadow: inputFocused
                        ? '0 0 0 2px rgba(220,38,38,0.4), 0 0 20px rgba(220,38,38,0.1)'
                        : '0 0 0 0px rgba(220,38,38,0)',
                    }}
                    transition={{ duration: 0.25 }}
                    aria-hidden="true"
                  />

                  <input
                    ref={inputRef}
                    id="demo-ticker-input"
                    type="text"
                    value={ticker}
                    onChange={(e) => {
                      setTicker(e.target.value.toUpperCase());
                      setLocalError(null);
                    }}
                    onKeyDown={handleKeyDown}
                    onFocus={() => setInputFocused(true)}
                    onBlur={() => setInputFocused(false)}
                    placeholder="NSE ticker: TCS, RELIANCE…"
                    className={clsx(
                      'w-full px-4 py-3.5 sm:px-5 sm:py-4 rounded-xl text-base font-mono',
                      'bg-white/[0.04] border',
                      localError
                        ? 'border-red-500/50'
                        : 'border-white/[0.08] focus:border-ark-red/40',
                      'text-white placeholder:text-gray-600',
                      'focus:outline-none transition-colors duration-200',
                    )}
                  />

                  {/* Clear button */}
                  <AnimatePresence>
                    {ticker && !loading && (
                      <motion.button
                        initial={{ opacity: 0, scale: 0.6 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.6 }}
                        transition={{ duration: 0.15 }}
                        onClick={() => { setTicker(''); setLocalError(null); inputRef.current?.focus(); }}
                        className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center text-gray-600 hover:text-gray-300 transition-colors rounded-lg hover:bg-white/[0.06]"
                        aria-label="Clear ticker"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </motion.button>
                    )}
                  </AnimatePresence>
                </div>

                <CTAButton
                  variant="primary"
                  size="lg"
                  onClick={handleScore}
                  disabled={!ticker.trim()}
                  loading={loading}
                  id="demo-score-btn"
                  className="w-full sm:w-auto"
                >
                  {loading ? 'Scoring…' : 'Score →'}
                </CTAButton>
              </div>

              {/* Suggested tickers */}
              <div className="flex items-center gap-2 flex-wrap mb-3">
                <span className="text-xs text-gray-600 flex-shrink-0">Try:</span>
                {SUGGESTED.map((t, i) => (
                  <motion.button
                    key={t}
                    initial={reduced ? false : { opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.06 + 0.2, duration: 0.3, ease: SPRING }}
                    onClick={() => handleSuggest(t)}
                    className={clsx(
                      'px-3 py-1.5 text-xs font-mono rounded-lg border transition-all duration-200 min-h-[36px]',
                      ticker === t
                        ? 'bg-ark-red/10 border-ark-red/30 text-ark-red'
                        : 'bg-white/[0.02] border-white/[0.06] text-gray-500 hover:text-white hover:border-white/[0.12] active:bg-white/[0.05]'
                    )}
                  >
                    {t}
                  </motion.button>
                ))}
              </div>

              {/* Supported tickers — scrollable */}
              <div className="mb-4">
                <p className="text-[10px] text-gray-600 font-mono mb-1">Supported NSE stocks:</p>
                <div className="flex gap-1.5 overflow-x-auto pb-1 -mx-1 px-1" style={{ scrollbarWidth: 'none' }}>
                  {SUPPORTED_TICKERS.map((t) => (
                    <button
                      key={t}
                      onClick={() => handleSuggest(t)}
                      className={clsx(
                        'flex-shrink-0 px-2 py-0.5 text-[9px] font-mono rounded border transition-colors',
                        ticker === t
                          ? 'border-ark-red/40 text-ark-red bg-ark-red/5'
                          : 'border-white/[0.05] text-gray-600 hover:text-gray-400 hover:border-white/[0.10]'
                      )}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              {/* Rate limit */}
              <AnimatePresence>
                {attemptCount > 0 && (
                  <motion.div
                    className="flex items-center gap-2 mb-4 text-xs text-gray-600"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <span className="font-mono">{Math.max(0, 3 - attemptCount)}/3</span>
                    <span>free lookups remaining</span>
                    {attemptCount >= 2 && (
                      <motion.span
                        className="text-ark-gold ml-auto"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                      >
                        → Enter email for unlimited
                      </motion.span>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Error */}
              <AnimatePresence>
                {(localError || error) && (
                  <motion.div
                    className="mb-4 p-3 rounded-lg bg-red-950/30 border border-red-500/20 text-sm text-red-400 flex items-start gap-2"
                    initial={{ opacity: 0, y: -8, height: 0 }}
                    animate={{ opacity: 1, y: 0, height: 'auto' }}
                    exit={{ opacity: 0, y: -8, height: 0 }}
                    transition={{ duration: 0.25, ease: SPRING }}
                  >
                    <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{localError ?? error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Results */}
              <AnimatePresence mode="wait">
                {score && (
                  <motion.div
                    key={score.run_id}
                    className="space-y-6"
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -16 }}
                    transition={{ duration: 0.45, ease: SPRING }}
                  >
                    {/* Company header */}
                    <div className="flex items-start justify-between gap-3 pb-4 border-b border-white/[0.06] flex-wrap sm:flex-nowrap">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h3 className="text-base sm:text-lg font-bold text-white truncate">
                            {score.company_name}
                          </h3>
                          {dataSource === 'live' ? (
                            <motion.span
                              className="flex items-center gap-1 text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-green-950/60 border border-green-500/30 text-green-400 flex-shrink-0"
                              animate={{ opacity: [1, 0.5, 1] }}
                              transition={{ duration: 2, repeat: Infinity }}
                            >
                              <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                              LIVE
                            </motion.span>
                          ) : (
                            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-amber-950/60 border border-amber-500/30 text-amber-400 flex-shrink-0">
                              DEMO
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs font-mono text-gray-500">{score.ticker}</span>
                          {score.rfinal !== undefined && (
                            <span className="text-xs font-mono text-gray-400">
                              ASRE: <span className="text-white font-bold">{score.rfinal}</span>/100
                            </span>
                          )}
                        </div>

                        {score.signal && (
                          <div className="mt-2">
                            <SignalBadge signal={score.signal} />
                          </div>
                        )}
                      </div>

                      <div className="flex-shrink-0">
                        <DipBadge dip={score.dip_context} size="sm" />
                      </div>
                    </div>

                    {/* Score bars — staggered */}
                    <motion.div
                      className="space-y-5"
                      initial="hidden"
                      animate="visible"
                      variants={{
                        hidden: {},
                        visible: { transition: { staggerChildren: 0.1 } },
                      }}
                    >
                      {[
                        { label: 'F-Score (Fundamental)', value: score.f_score, color: 'green' as const, delay: 0 },
                        { label: 'T-Score (Technical)', value: score.t_score, color: 'blue' as const, delay: 200 },
                        { label: 'M-Score (Momentum)', value: score.m_score, color: 'amber' as const, delay: 400 },
                      ].map((bar) => (
                        <motion.div
                          key={bar.label}
                          variants={{
                            hidden: { opacity: 0, x: -16 },
                            visible: { opacity: 1, x: 0, transition: { duration: 0.4, ease: SPRING } },
                          }}
                        >
                          <ScoreBar
                            label={bar.label}
                            value={bar.value}
                            color={bar.color}
                            delay={bar.delay}
                            animate
                          />
                        </motion.div>
                      ))}
                    </motion.div>

                    {/* Hash proof */}
                    <HashProof runId={score.run_id} hash={score.hash} />

                    {/* Demo disclaimer */}
                    {dataSource === 'mock' && (
                      <motion.div
                        className="flex items-center gap-2 p-3 rounded-lg bg-amber-950/20 border border-amber-500/20 text-[11px] text-amber-400/70"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                      >
                        <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Demo data shown — connect to backend for live scores.
                      </motion.div>
                    )}

                    {/* Action buttons */}
                    <motion.div
                      className="flex flex-col sm:flex-row gap-3 pt-2"
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.45, duration: 0.4, ease: SPRING }}
                    >
                      <CTAButton
                        variant="gold" size="md"
                        onClick={handlePdfDownload}
                        id="demo-download-pdf"
                        className="flex-1"
                        icon={
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        }
                      >
                        Download Full PDF
                      </CTAButton>
                      <CTAButton
                        variant="secondary" size="md"
                        onClick={() => document.querySelector('#pricing')?.scrollIntoView({ behavior: 'smooth' })}
                        id="demo-see-pricing"
                        className="flex-1"
                      >
                        See Pricing →
                      </CTAButton>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </motion.div>
        </div>
      </section>

      <EmailGateModal
        isOpen={showEmailGate}
        onClose={() => setShowEmailGate(false)}
        onSuccess={handleEmailSuccess}
        source="demo_widget"
      />
    </>
  );
};
