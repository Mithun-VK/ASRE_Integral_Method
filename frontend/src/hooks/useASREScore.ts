import { useState, useCallback, useRef } from 'react';
import { fetchDemoScore, getMockScore, type ASREScoreResponse } from '../services/api';

const RATE_LIMIT = 3;
const STORAGE_KEY = 'ark_demo_count';

function getStoredCount(): number {
  try { return parseInt(localStorage.getItem(STORAGE_KEY) || '0', 10); }
  catch { return 0; }
}

function incrementCount(): number {
  const next = getStoredCount() + 1;
  try { localStorage.setItem(STORAGE_KEY, String(next)); } catch { /* ignore */ }
  return next;
}

export interface UseASREScoreReturn {
  score: ASREScoreResponse | null;
  loading: boolean;
  error: string | null;
  attemptCount: number;
  needsEmail: boolean;
  dataSource: 'live' | 'mock' | null;   // NEW — so UI can show LIVE badge
  fetchScore: (ticker: string) => Promise<void>;
  resetGate: () => void;
}

export function useASREScore(): UseASREScoreReturn {
  const [score, setScore] = useState<ASREScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attemptCount, setAttemptCount] = useState(getStoredCount);
  const [needsEmail, setNeedsEmail] = useState(false);
  const [dataSource, setDataSource] = useState<'live' | 'mock' | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchScore = useCallback(async (ticker: string) => {
    const currentCount = getStoredCount();
    if (currentCount >= RATE_LIMIT) {
      setNeedsEmail(true);
      return;
    }

    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setScore(null);
    setDataSource(null);

    try {
      let data: ASREScoreResponse;
      let source: 'live' | 'mock' = 'live';

      try {
        // ── Try the real backend first ──
        data = await fetchDemoScore(ticker);
        source = 'live';
      } catch (apiErr) {
        // ── Backend unreachable or ticker not supported → fall back to mock ──
        const errMsg = apiErr instanceof Error ? apiErr.message : String(apiErr);
        console.warn(`[ASRE] Backend unavailable (${errMsg}), using mock data`);

        // Simulate realistic latency
        await new Promise(r => setTimeout(r, 600 + Math.random() * 600));
        data = getMockScore(ticker);
        source = 'mock';
      }

      const newCount = incrementCount();
      setAttemptCount(newCount);
      setScore(data);
      setDataSource(source);
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err.message || 'Failed to fetch score');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const resetGate = useCallback(() => {
    setNeedsEmail(false);
    try { localStorage.setItem(STORAGE_KEY, '0'); } catch { /* ignore */ }
    setAttemptCount(0);
  }, []);

  return { score, loading, error, attemptCount, needsEmail, dataSource, fetchScore, resetGate };
}
