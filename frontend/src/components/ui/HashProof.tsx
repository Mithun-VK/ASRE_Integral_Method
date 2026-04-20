import React, { useState } from 'react';
import clsx from 'clsx';

interface HashProofProps {
  runId: string;
  hash: string;
  className?: string;
}

export const HashProof: React.FC<HashProofProps> = ({ runId, hash, className }) => {
  const [copied, setCopied] = useState<'runId' | 'hash' | null>(null);

  const copyToClipboard = async (text: string, field: 'runId' | 'hash') => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(field);
      setTimeout(() => setCopied(null), 2000);
    } catch { /* ignore */ }
  };

  // Truncate long hashes for mobile display
  const truncateHash = (h: string, chars = 20) =>
    h.length > chars ? `${h.slice(0, chars)}…` : h;

  return (
    <div className={clsx(
      'rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 sm:p-4',
      'font-mono text-xs',
      className
    )}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <svg className="w-4 h-4 text-ark-gold flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
        <span className="text-ark-gold font-semibold text-xs uppercase tracking-wider">Audit Proof</span>
      </div>

      {/* Fields */}
      <div className="space-y-2.5">
        {/* Run ID */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-start gap-2 min-w-0 flex-1">
            <span className="text-gray-500 flex-shrink-0 w-12 sm:w-16 pt-px">Run ID</span>
            <span className="text-gray-300 break-all text-[10px] sm:text-xs leading-relaxed">{runId}</span>
          </div>
          <button
            onClick={() => copyToClipboard(runId, 'runId')}
            className="flex-shrink-0 text-gray-600 hover:text-gray-400 transition-colors p-1 -m-1 touch-target"
            title="Copy Run ID"
            aria-label="Copy Run ID"
          >
            {copied === 'runId' ? (
              <svg className="w-3 h-3 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            )}
          </button>
        </div>

        {/* Hash */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-start gap-2 min-w-0 flex-1">
            <span className="text-gray-500 flex-shrink-0 w-12 sm:w-16 pt-px">Hash</span>
            <div className="min-w-0 flex-1">
              {/* Mobile: truncated */}
              <span className="text-green-400/80 text-[10px] sm:hidden leading-relaxed block">
                {truncateHash(hash, 24)}
              </span>
              {/* Desktop: full */}
              <span className="text-green-400/80 text-xs hidden sm:block break-all leading-relaxed">
                {hash}
              </span>
            </div>
          </div>
          <button
            onClick={() => copyToClipboard(hash, 'hash')}
            className="flex-shrink-0 text-gray-600 hover:text-gray-400 transition-colors p-1 -m-1 touch-target"
            title="Copy Hash"
            aria-label="Copy Hash"
          >
            {copied === 'hash' ? (
              <svg className="w-3 h-3 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center gap-1.5 text-[10px] text-gray-600">
        <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
        SHA-256 · Generated pre-investment · Immutable
      </div>
    </div>
  );
};
