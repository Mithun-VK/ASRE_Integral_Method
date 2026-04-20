/* ─── API Service — wired to FastAPI backend (Indian NSE mode) ─── */

const API_BASE = 'http://localhost:8000';

// ─── Our frontend's unified score shape ───────────────────────────────────────
export interface ASREScoreResponse {
  f_score: number;
  t_score: number;
  m_score: number;
  dip_context: 'DEEP' | 'MID' | 'LATE' | 'NONE';
  run_id: string;
  hash: string;
  ticker: string;
  company_name: string;
  signal: string;
  rfinal: number;
  quality_tier?: string | null;
  close_price?: number | null;
  category?: string;
  is_mock: boolean;
}

// ─── Raw shape returned by GET /api/stocks/{ticker} ───────────────────────────
interface BackendStockRating {
  ticker: string;
  rfinal: number;
  rasre: number;
  fscore: number;
  tscore: number;
  mscore: number;
  signal: string;
  category: string;
  dip_quality?: number | null;
  dip_stage?: string | null;
  context?: string;
  timestamp: string;
  peg_ratio?: number | null;
  quality_tier?: string | null;
  close_price?: number | null;
  // ── Authentic audit proof fields ────────────────────────────────────────────
  run_id?: string | null;      // ASRE-{TICKER}-{YYYYMMDD}-{SEQ3}
  score_hash?: string | null;  // SHA-256 of score DataFrame CSV
  ai_explanation?: string | null;
  momentum_trap?: unknown;
}

// ─── Waitlist ─────────────────────────────────────────────────────────────────
export interface WaitlistPayload {
  email: string;
  sebi_reg?: string;
  source: string;
}

export interface WaitlistResponse {
  success: boolean;
  message: string;
}

// ─── Helper: map backend dip_stage → our badge type ──────────────────────────
function mapDipStage(stage: string | null | undefined): ASREScoreResponse['dip_context'] {
  if (!stage) return 'NONE';
  const upper = stage.toUpperCase();
  if (upper === 'DEEP' || upper === 'EARLY' || upper === 'STRUCTURAL BREAK') return 'DEEP';
  if (upper === 'MID') return 'MID';
  if (upper === 'LATE') return 'LATE';
  return 'NONE';
}

// ─── Helper: fallback run_id when backend doesn't return one ─────────────────
function fallbackRunId(ticker: string, timestamp: string): string {
  const dateStr = (timestamp ?? new Date().toISOString()).slice(0, 10).replace(/-/g, '');
  return `ASRE-${ticker.slice(0, 6).toUpperCase()}-${dateStr}-???`;
}

// ─── Helper: format score_hash for display ────────────────────────────────────
function formatHash(hash: string | null | undefined): string {
  if (!hash) return 'pending';
  return hash.startsWith('0x') ? hash : `0x${hash}`;
}

// ─── Indian NSE company names ─────────────────────────────────────────────────
const COMPANY_NAMES: Record<string, string> = {
  // IT
  TCS: 'Tata Consultancy Services Ltd',
  INFY: 'Infosys Ltd',
  WIPRO: 'Wipro Ltd',
  HCLTECH: 'HCL Technologies Ltd',
  // Banking
  HDFCBANK: 'HDFC Bank Ltd',
  ICICIBANK: 'ICICI Bank Ltd',
  SBIN: 'State Bank of India',
  KOTAKBANK: 'Kotak Mahindra Bank Ltd',
  AXISBANK: 'Axis Bank Ltd',
  BAJFINANCE: 'Bajaj Finance Ltd',
  // Energy / Conglomerate
  RELIANCE: 'Reliance Industries Ltd',
  ONGC: 'Oil & Natural Gas Corporation Ltd',
  // FMCG
  HINDUNILVR: 'Hindustan Unilever Ltd',
  ITC: 'ITC Ltd',
  // Auto
  MARUTI: 'Maruti Suzuki India Ltd',
  TATAMOTORS: 'Tata Motors Ltd',
  // Pharma
  SUNPHARMA: 'Sun Pharmaceutical Industries Ltd',
  DRREDDY: 'Dr. Reddy\'s Laboratories Ltd',
};

function getCompanyName(ticker: string): string {
  return COMPANY_NAMES[ticker.toUpperCase()] ?? `${ticker.toUpperCase()} (NSE)`;
}

// ─── Map backend response → unified frontend shape ────────────────────────────
function mapBackendRating(data: BackendStockRating): ASREScoreResponse {
  // Use authentic run_id and score_hash from backend (SHA-256, same as cli.py).
  const run_id = data.run_id ?? fallbackRunId(data.ticker, data.timestamp);
  const hash = formatHash(data.score_hash);

  return {
    f_score: Math.round(data.fscore * 10) / 10,
    t_score: Math.round(data.tscore * 10) / 10,
    m_score: Math.round(data.mscore * 10) / 10,
    dip_context: mapDipStage(data.dip_stage),
    run_id,
    hash,
    ticker: data.ticker,
    company_name: getCompanyName(data.ticker),
    signal: data.signal,
    rfinal: Math.round(data.rfinal * 10) / 10,
    quality_tier: data.quality_tier,
    close_price: data.close_price,
    category: data.category,
    is_mock: false,
  };
}

// =============================================================================
// PUBLIC API FUNCTIONS
// =============================================================================

/**
 * Fetch real ASRE scores from GET /api/stocks/{ticker}
 * Throws on non-2xx so the hook falls back to mock.
 */
export async function fetchDemoScore(ticker: string): Promise<ASREScoreResponse> {
  const cleanTicker = ticker.replace('.NS', '').replace('.BO', '').toUpperCase();

  const res = await fetch(
    `${API_BASE}/api/stocks/${cleanTicker}?include_trap_analysis=false`,
    { method: 'GET', headers: { 'Accept': 'application/json' } }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  const data: BackendStockRating = await res.json();
  return mapBackendRating(data);
}

/** POST /waitlist */
export async function submitWaitlist(payload: WaitlistPayload): Promise<WaitlistResponse> {
  const res = await fetch(`${API_BASE}/waitlist`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Waitlist API error: ${res.status}`);
  return res.json();
}

// =============================================================================
// MOCK DATA — Indian NSE blue-chips (used when backend is unreachable)
// =============================================================================

const MOCK_DATA: Record<string, ASREScoreResponse> = {
  RELIANCE: { f_score: 88.0, t_score: 75.0, m_score: 65.0, dip_context: 'NONE', run_id: 'ASRE-RELIANCE-DEMO-001', hash: '0xdemo_reliance_001', ticker: 'RELIANCE', company_name: 'Reliance Industries Ltd', signal: 'STRONG BUY', rfinal: 84.7, quality_tier: 'A', is_mock: true },
  TCS: { f_score: 82.0, t_score: 52.0, m_score: 48.0, dip_context: 'MID', run_id: 'ASRE-TCS-DEMO-002', hash: '0xdemo_tcs_002', ticker: 'TCS', company_name: 'Tata Consultancy Services Ltd', signal: 'BUY', rfinal: 78.5, quality_tier: 'A', is_mock: true },
  HDFCBANK: { f_score: 85.0, t_score: 68.0, m_score: 62.0, dip_context: 'NONE', run_id: 'ASRE-HDFCBANK-DEMO-003', hash: '0xdemo_hdfcbank_003', ticker: 'HDFCBANK', company_name: 'HDFC Bank Ltd', signal: 'STRONG BUY', rfinal: 81.2, quality_tier: 'A', is_mock: true },
  INFY: { f_score: 76.0, t_score: 45.0, m_score: 55.0, dip_context: 'MID', run_id: 'ASRE-INFY-DEMO-004', hash: '0xdemo_infy_004', ticker: 'INFY', company_name: 'Infosys Ltd', signal: 'BUY', rfinal: 72.3, quality_tier: 'B', is_mock: true },
  ICICIBANK: { f_score: 83.0, t_score: 65.0, m_score: 70.0, dip_context: 'LATE', run_id: 'ASRE-ICICIBANK-DEMO-005', hash: '0xdemo_icicibank_005', ticker: 'ICICIBANK', company_name: 'ICICI Bank Ltd', signal: 'BUY', rfinal: 79.6, quality_tier: 'A', is_mock: true },
  BAJFINANCE: { f_score: 87.0, t_score: 72.0, m_score: 68.0, dip_context: 'NONE', run_id: 'ASRE-BAJFIN-DEMO-006', hash: '0xdemo_bajfin_006', ticker: 'BAJFINANCE', company_name: 'Bajaj Finance Ltd', signal: 'STRONG BUY', rfinal: 83.4, quality_tier: 'A', is_mock: true },
  SBIN: { f_score: 70.0, t_score: 55.0, m_score: 58.0, dip_context: 'MID', run_id: 'ASRE-SBIN-DEMO-007', hash: '0xdemo_sbin_007', ticker: 'SBIN', company_name: 'State Bank of India', signal: 'BUY', rfinal: 65.4, quality_tier: 'B', is_mock: true },
  WIPRO: { f_score: 65.0, t_score: 38.0, m_score: 42.0, dip_context: 'DEEP', run_id: 'ASRE-WIPRO-DEMO-008', hash: '0xdemo_wipro_008', ticker: 'WIPRO', company_name: 'Wipro Ltd', signal: 'HOLD', rfinal: 61.8, quality_tier: 'B', is_mock: true },
  MARUTI: { f_score: 77.0, t_score: 62.0, m_score: 55.0, dip_context: 'NONE', run_id: 'ASRE-MARUTI-DEMO-009', hash: '0xdemo_maruti_009', ticker: 'MARUTI', company_name: 'Maruti Suzuki India Ltd', signal: 'BUY', rfinal: 73.6, quality_tier: 'A', is_mock: true },
  HINDUNILVR: { f_score: 80.0, t_score: 58.0, m_score: 52.0, dip_context: 'LATE', run_id: 'ASRE-HUL-DEMO-010', hash: '0xdemo_hul_010', ticker: 'HINDUNILVR', company_name: 'Hindustan Unilever Ltd', signal: 'BUY', rfinal: 77.3, quality_tier: 'A', is_mock: true },
};

export function getMockScore(ticker: string): ASREScoreResponse {
  const key = ticker.replace('.NS', '').replace('.BO', '').toUpperCase();
  if (MOCK_DATA[key]) return MOCK_DATA[key];

  const f = Math.round((45 + Math.random() * 40) * 10) / 10;
  const t = Math.round((35 + Math.random() * 45) * 10) / 10;
  const m = Math.round((30 + Math.random() * 50) * 10) / 10;
  const dips: ASREScoreResponse['dip_context'][] = ['DEEP', 'MID', 'LATE', 'NONE'];
  const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, '');

  return {
    f_score: f, t_score: t, m_score: m,
    dip_context: dips[Math.floor(Math.random() * dips.length)],
    run_id: `ASRE-${key.slice(0, 6)}-${dateStr}-DEMO`,
    hash: '0xdemo' + Array.from({ length: 36 }, () => Math.floor(Math.random() * 16).toString(16)).join(''),
    ticker: key,
    company_name: getCompanyName(key),
    signal: 'HOLD',
    rfinal: Math.round((f + t + m) / 3 * 10) / 10,
    quality_tier: 'B',
    is_mock: true,
  };
}

// Supported Indian NSE tickers (clean — no .NS suffix)
export const SUPPORTED_TICKERS = [
  'TCS', 'INFY', 'WIPRO', 'HCLTECH',
  'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'BAJFINANCE',
  'RELIANCE', 'ONGC',
  'HINDUNILVR', 'ITC',
  'MARUTI', 'TATAMOTORS',
  'SUNPHARMA', 'DRREDDY',
];
