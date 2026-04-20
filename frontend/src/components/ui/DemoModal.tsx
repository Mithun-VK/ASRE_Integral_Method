import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import { CTAButton } from './CTAButton';
import { useWaitlist } from '../../hooks/useWaitlist';

interface EmailGateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  source?: string;
}

export const EmailGateModal: React.FC<EmailGateModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  source = 'demo_widget',
}) => {
  const [email, setEmail] = useState('');
  const [sebiReg, setSebiReg] = useState('');
  const { submitting, submitted, error, submit } = useWaitlist();

  // Lock body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    if (isOpen) document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    await submit(email, sebiReg || undefined, source);
    onSuccess();
  };

  return (
    /* Full-screen overlay — strictly inset-0, no padding that could cause overflow */
    <div
      className="fixed inset-0 z-50 modal-overlay"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      {/* Flex column: on mobile fills from bottom, on sm+ centers */}
      <div className="flex flex-col sm:items-center sm:justify-center h-full w-full pointer-events-none">

        {/* Spacer — tappable to close on mobile */}
        <div className="flex-1 sm:hidden pointer-events-auto" onClick={onClose} />

        {/* Modal panel */}
        <div
          className={clsx(
            // Layout — full width, safe horizontal containment
            'relative w-full pointer-events-auto',
            // Desktop: constrained width, centered
            'sm:max-w-md sm:mx-auto',
            // Colours & shape
            'bg-[#1a1a1a] border border-white/[0.10]',
            'rounded-t-3xl sm:rounded-2xl',
            // Spacing — tighter on mobile
            'px-5 pt-4 pb-6 sm:px-8 sm:pt-6 sm:pb-8',
            // Safe area for notched phones (home bar)
            'pb-[calc(1.5rem+env(safe-area-inset-bottom,0px))] sm:pb-8',
            // Shadow & animation
            'shadow-2xl shadow-black/60 animate-scale-in',
            // Prevent inner content overflow
            'overflow-hidden',
          )}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Drag handle pill — mobile only */}
          <div className="w-10 h-1 rounded-full bg-white/[0.12] mx-auto mb-5 sm:hidden" />

          {/* Close button — top right, 44px touch target */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-9 h-9 flex items-center justify-center rounded-xl text-gray-500 hover:text-white hover:bg-white/[0.06] transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {!submitted ? (
            <>
              {/* Header */}
              <div className="mb-5 sm:mb-6 pr-8">
                {/* Icon */}
                <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-xl bg-ark-red/10 border border-ark-red/20 flex items-center justify-center mb-4">
                  <svg className="w-5 h-5 sm:w-6 sm:h-6 text-ark-red" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <h3 className="text-lg sm:text-xl font-bold text-white mb-1.5">
                  Get Your Full ASRE Report
                </h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  Enter your details for the complete PDF report with F/T/M breakdown, dip context, and hash audit trail.
                </p>
              </div>

              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-3 sm:space-y-4">
                {/* Email */}
                <div>
                  <label htmlFor="gate-email" className="block text-xs font-medium text-gray-400 mb-1.5">
                    Work Email <span className="text-ark-red">*</span>
                  </label>
                  <input
                    id="gate-email"
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@firm.com"
                    autoComplete="email"
                    className={clsx(
                      'w-full px-4 py-3 rounded-xl text-sm',
                      'bg-white/[0.04] border border-white/[0.08]',
                      'text-white placeholder:text-gray-600',
                      'focus:outline-none focus:border-ark-red/50 focus:ring-1 focus:ring-ark-red/30',
                      'transition-all duration-200',
                      // Prevent iOS zoom — must be exactly 16px
                      'text-[16px] sm:text-sm',
                    )}
                  />
                </div>

                {/* SEBI Reg */}
                <div>
                  <label htmlFor="gate-sebi" className="block text-xs font-medium text-gray-400 mb-1.5">
                    SEBI Registration # <span className="text-gray-600">(optional)</span>
                  </label>
                  <input
                    id="gate-sebi"
                    type="text"
                    value={sebiReg}
                    onChange={(e) => setSebiReg(e.target.value)}
                    placeholder="INA000XXXXXX"
                    autoComplete="off"
                    className={clsx(
                      'w-full px-4 py-3 rounded-xl text-sm',
                      'bg-white/[0.04] border border-white/[0.08]',
                      'text-white placeholder:text-gray-600',
                      'focus:outline-none focus:border-ark-red/50 focus:ring-1 focus:ring-ark-red/30',
                      'transition-all duration-200',
                      'text-[16px] sm:text-sm',
                    )}
                  />
                </div>

                {/* Inline error */}
                {error && (
                  <p className="text-sm text-red-400 flex items-center gap-1.5">
                    <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {error}
                  </p>
                )}

                {/* Submit */}
                <CTAButton
                  type="submit"
                  variant="primary"
                  size="lg"
                  className="w-full mt-1"
                  disabled={submitting || !email}
                  id="gate-submit"
                >
                  {submitting ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Submitting…
                    </span>
                  ) : 'Get Full Report →'}
                </CTAButton>

                {/* Trust line */}
                <p className="text-[10px] text-gray-600 text-center pt-1">
                  No spam. We only contact SEBI-registered professionals.
                </p>
              </form>
            </>
          ) : (
            /* ── Success State ── */
            <div className="text-center py-4 sm:py-6">
              <div className="w-16 h-16 rounded-full bg-green-500/10 border border-green-500/20 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-white mb-2">You're In!</h3>
              <p className="text-sm text-gray-400 mb-6 leading-relaxed">
                Check your inbox for the full PDF report and a link to book your 10-minute demo.
              </p>
              <CTAButton
                variant="secondary"
                onClick={onClose}
                id="gate-close-success"
                className="w-full sm:w-auto"
              >
                Continue Exploring
              </CTAButton>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
