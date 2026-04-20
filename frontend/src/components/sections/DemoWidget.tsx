import React, { useState } from 'react';
import clsx from 'clsx';
import { useASREScore } from '../../hooks/useASREScore';
import { ScoreBar } from '../ui/ScoreBadge';
import { DipBadge } from '../ui/DipBadge';
import { HashProof } from '../ui/HashProof';
import { EmailGateModal } from '../ui/DemoModal';
import { CTAButton } from '../ui/CTAButton';
import { SUPPORTED_TICKERS } from '../../services/api';

// Representative tickers across Indian NSE sectors
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
    <span className={clsx('text-xs font-mono font-bold px-2.5 py-1 rounded-full border', color)}>
      {signal}
    </span>
  );
}

export const DemoWidget: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const { score, loading, error, attemptCount, needsEmail, dataSource, fetchScore, resetGate } = useASREScore();
  const [showEmailGate, setShowEmailGate] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleScore = () => {
    const cleaned = ticker.trim().toUpperCase().replace('.NS', '').replace('.BO', '');
    if (!cleaned) return;
    if (needsEmail) { setShowEmailGate(true); return; }

    // Client-side guard: validate against supported NSE list before API call
    if (!SUPPORTED_TICKERS.includes(cleaned)) {
      setLocalError(
        `'${cleaned}' is not in the supported list. Try: ${SUGGESTED.join(', ')} or see the full list below.`
      );
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
    setLocalError(null);  // suggested tickers are always valid
    if (needsEmail) { setShowEmailGate(true); return; }
    fetchScore(t);
  };

  const handlePdfDownload = () => {
    if (attemptCount >= 1) { setShowEmailGate(true); return; }
    // Programmatic download — forces save dialog with correct filename
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
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-ark-red/[0.03] rounded-full blur-[150px] translate-x-1/2 -translate-y-1/4" />

        <div className="section-container relative">
          {/* Section header */}
          <div className="text-center mb-8 sm:mb-12">
            <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-red border border-ark-red/20 rounded-full mb-4 uppercase tracking-wider">
              Live Demo
            </span>
            <h2 className="section-title text-white mb-3 sm:mb-4">
              Try the <span className="gradient-text">ASRE Engine</span>
            </h2>
            <p className="section-subtitle mx-auto px-2">
              Enter any NSE blue-chip ticker to get real F/T/M scores from the live ASRE engine.
            </p>
          </div>

          {/* Demo card */}
          <div className="max-w-2xl mx-auto">
            <div className="glass-card p-4 sm:p-6 lg:p-8 red-glow">
              {/* Input area */}
              <div className="flex flex-col sm:flex-row gap-3 mb-4">
                <div className="flex-1 relative">
                  <input
                    id="demo-ticker-input"
                    type="text"
                    value={ticker}
                    onChange={(e) => {
                    setTicker(e.target.value.toUpperCase());
                    setLocalError(null);  // clear error as user types
                  }}
                    onKeyDown={handleKeyDown}
                    placeholder="NSE ticker: TCS, RELIANCE…"
                    className={clsx(
                      'w-full px-4 py-3.5 sm:px-5 sm:py-4 rounded-xl text-base font-mono',
                      'bg-white/[0.04] border',
                      localError
                        ? 'border-red-500/50 ring-2 ring-red-500/20'
                        : 'border-white/[0.08] focus:border-ark-red/50 focus:ring-2 focus:ring-ark-red/20',
                      'text-white placeholder:text-gray-600',
                      'focus:outline-none',
                      'transition-all duration-300'
                    )}
                  />
                  {ticker && !loading && (
                    <button
                      onClick={() => setTicker('')}
                      className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center text-gray-600 hover:text-gray-400 transition-colors"
                      aria-label="Clear ticker"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>

                <CTAButton
                  variant="primary"
                  size="lg"
                  onClick={handleScore}
                  disabled={loading || !ticker.trim()}
                  id="demo-score-btn"
                  className="w-full sm:w-auto"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Scoring…
                    </span>
                  ) : 'Score →'}
                </CTAButton>
              </div>

              {/* Suggested tickers — touch-friendly min height */}
              <div className="flex items-center gap-2 flex-wrap mb-3">
                <span className="text-xs text-gray-600 flex-shrink-0">Try:</span>
                {SUGGESTED.map((t) => (
                  <button
                    key={t}
                    onClick={() => handleSuggest(t)}
                    className={clsx(
                      'px-3 py-1.5 text-xs font-mono rounded-lg border transition-all duration-200 min-h-[36px]',
                      ticker === t
                        ? 'bg-ark-red/10 border-ark-red/30 text-ark-red'
                        : 'bg-white/[0.02] border-white/[0.06] text-gray-500 hover:text-white hover:border-white/[0.12] active:bg-white/[0.05]'
                    )}
                  >
                    {t}
                  </button>
                ))}
              </div>

              {/* Supported tickers — scrollable on mobile */}
              <div className="mb-4">
                <p className="text-[10px] text-gray-600 font-mono mb-1">Supported NSE stocks:</p>
                <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-hide -mx-1 px-1">
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

              {/* Rate limit notice */}
              {attemptCount > 0 && (
                <div className="flex items-center gap-2 mb-4 text-xs text-gray-600">
                  <span className="font-mono">{Math.max(0, 3 - attemptCount)}/3</span>
                  <span>free lookups remaining</span>
                  {attemptCount >= 2 && (
                    <span className="text-ark-gold ml-auto">→ Enter email for unlimited</span>
                  )}
                </div>
              )}

              {/* Error — client-side validation OR API error */}
              {(localError || error) && (
                <div className="mb-4 p-3 rounded-lg bg-red-950/30 border border-red-500/20 text-sm text-red-400 flex items-start gap-2">
                  <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{localError ?? error}</span>
                </div>
              )}

              {/* Results */}
              {score && (
                <div className="space-y-6 animate-slide-up">
                  {/* Company header */}
                  <div className="flex items-start justify-between gap-3 pb-4 border-b border-white/[0.06] flex-wrap sm:flex-nowrap">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <h3 className="text-base sm:text-lg font-bold text-white truncate">{score.company_name}</h3>
                        {/* LIVE or DEMO badge */}
                        {dataSource === 'live' ? (
                          <span className="flex items-center gap-1 text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-green-950/60 border border-green-500/30 text-green-400 flex-shrink-0">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                            LIVE
                          </span>
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
                    {/* DipBadge — pushed to right, never wraps */}
                    <div className="flex-shrink-0">
                      <DipBadge dip={score.dip_context} size="sm" />
                    </div>
                  </div>

                  {/* Score bars */}
                  <div className="space-y-5">
                    <ScoreBar label="F-Score (Fundamental)" value={score.f_score} color="green" delay={0} />
                    <ScoreBar label="T-Score (Technical)" value={score.t_score} color="blue" delay={200} />
                    <ScoreBar label="M-Score (Momentum)" value={score.m_score} color="amber" delay={400} />
                  </div>

                  {/* Hash proof */}
                  <HashProof runId={score.run_id} hash={score.hash} />

                  {/* Mock disclaimer */}
                  {dataSource === 'mock' && (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-950/20 border border-amber-500/20 text-[11px] text-amber-400/70">
                      <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Demo data shown — connect to backend for live scores.
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex flex-col sm:flex-row gap-3 pt-2">
                    <CTAButton
                      variant="gold"
                      size="md"
                      onClick={handlePdfDownload}
                      id="demo-download-pdf"
                      className="flex-1"
                      icon={
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      }
                    >
                      Download Full PDF
                    </CTAButton>
                    <CTAButton
                      variant="secondary"
                      size="md"
                      onClick={() => document.querySelector('#pricing')?.scrollIntoView({ behavior: 'smooth' })}
                      id="demo-see-pricing"
                      className="flex-1"
                    >
                      See Pricing →
                    </CTAButton>
                  </div>
                </div>
              )}
            </div>
          </div>
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
