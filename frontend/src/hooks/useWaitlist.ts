import { useState, useCallback } from 'react';
import { submitWaitlist, type WaitlistPayload, type WaitlistResponse } from '../services/api';

export interface UseWaitlistReturn {
  submitting: boolean;
  submitted: boolean;
  error: string | null;
  submit: (email: string, sebiReg?: string, source?: string) => Promise<void>;
  reset: () => void;
}

export function useWaitlist(): UseWaitlistReturn {
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async (email: string, sebiReg?: string, source: string = 'demo_widget') => {
    setSubmitting(true);
    setError(null);

    const payload: WaitlistPayload = {
      email,
      sebi_reg: sebiReg,
      source,
    };

    try {
      await submitWaitlist(payload);
      setSubmitted(true);
    } catch {
      // If API unavailable, still show success (store locally)
      try {
        const existing = JSON.parse(localStorage.getItem('ark_waitlist') || '[]');
        existing.push({ ...payload, ts: new Date().toISOString() });
        localStorage.setItem('ark_waitlist', JSON.stringify(existing));
      } catch { /* ignore */ }
      setSubmitted(true);
    } finally {
      setSubmitting(false);
    }
  }, []);

  const reset = useCallback(() => {
    setSubmitted(false);
    setError(null);
  }, []);

  return { submitting, submitted, error, submit, reset };
}
