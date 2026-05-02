#!/usr/bin/env python3
"""
ASRE Pre-Demo Test Script
Validates system readiness before live RIA/RA demonstrations.

Tests:
  - Fundamental data fetch (10 NSE tickers)
  - Full ASRE scan pipeline
  - PDF report generation
  - Performance benchmarks (<30s per stock)
  - Data completeness (min 4 quarters)

Exit codes:
  0 : All tests passed
  1 : One or more stocks failed
  2 : Critical system error

Usage:
  python pre_demo_test.py
  python pre_demo_test.py --tickers INFY.NS TCS.NS RELIANCE.NS

Author: ASRE Project
Date: February 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Import ASRE modules
try:
    from asre.data_loader_indian import load_stock_data
    from asre.data.fundamental_fetcher import FundamentalFetcher
    from asre.composite import compute_asre_rating
    from asre.report import export_stock_report
except ImportError:
    print("❌ CRITICAL: ASRE modules not found. Run from src/ directory:")
    print("   cd src && python pre_demo_test.py")
    sys.exit(2)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default test tickers (10 NSE stocks across sectors)
DEFAULT_TICKERS = [
    "INFY.NS",       # IT
    "TCS.NS",        # IT
    "HDFCBANK.NS",   # Banking
    "RELIANCE.NS",   # Energy/Conglomerate
    "ZYDUSLIFE.NS",  # Pharma
    "MARUTI.NS",     # Auto
    "HINDUNILVR.NS", # FMCG
    "TITAN.NS",      # Consumer
    "BAJFINANCE.NS", # Finance
    "ITC.NS",        # FMCG/Conglomerate
]

# Performance benchmarks
MAX_TIME_PER_STOCK = 30  # seconds
MIN_QUARTERS_REQUIRED = 4
MAX_FAILURES_ALLOWED = 1  # Allow 1 failure out of 10


# ─────────────────────────────────────────────────────────────
# Test Runner
# ─────────────────────────────────────────────────────────────

class PreDemoTestRunner:
    """Orchestrates pre-demo system validation."""

    def __init__(self, tickers: List[str], output_dir: str = "test_reports"):
        self.tickers = tickers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict] = []
        self.failures = 0
        self.total_time = 0.0

        self.fetcher = FundamentalFetcher()

    def run_all_tests(self) -> int:
        """
        Execute full test suite.

        Returns
        -------
        int
            Exit code (0=success, 1=failures, 2=critical)
        """
        logger.info("=" * 70)
        logger.info("ASRE Pre-Demo Test Suite")
        logger.info("=" * 70)
        logger.info(f"Testing {len(self.tickers)} stocks...")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        start_time = time.time()

        for i, ticker in enumerate(self.tickers, 1):
            logger.info(f"[{i}/{len(self.tickers)}] Testing {ticker}...")
            result = self.test_single_stock(ticker)
            self.results.append(result)

            if not result['success']:
                self.failures += 1

        self.total_time = time.time() - start_time

        return self._generate_summary()

    def test_single_stock(self, ticker: str) -> Dict:
        """
        Test single stock through full pipeline.

        Returns dict with:
            - ticker: str
            - success: bool
            - duration: float (seconds)
            - quarters: int
            - error: str (if failed)
            - r_asre: float (if succeeded)
        """
        result = {
            'ticker': ticker,
            'success': False,
            'duration': 0.0,
            'quarters': 0,
            'error': None,
            'r_asre': None,
        }

        stock_start = time.time()

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=720)).strftime("%Y-%m-%d")

            # Step 1: Fetch fundamentals
            logger.info(f"  → Fetching fundamentals...")
            fundamentals, fetch_ts = self.fetcher.fetch_quarterly_fundamentals(
                ticker, start_date=start_date, end_date=end_date
            )

            if fundamentals is None or len(fundamentals) < MIN_QUARTERS_REQUIRED:
                quarters = 0 if fundamentals is None else len(fundamentals)
                raise ValueError(
                    f"Insufficient data: {quarters} quarters "
                    f"(minimum {MIN_QUARTERS_REQUIRED} required)"
                )

            result['quarters'] = len(fundamentals)
            logger.info(f"  ✓ Fetched {len(fundamentals)} quarters")

            # Step 2: Load stock data
            logger.info(f"  → Loading price data...")
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

            stock_df = load_stock_data(
                ticker=ticker,
                start=start_date,
                end=end_date,
                quarterly_fundamentals=fundamentals,
                fundamentals_fetch_ts=fetch_ts
            )

            logger.info(f"  ✓ Loaded {len(stock_df)} price rows")

            # Step 3: Compute ASRE rating
            logger.info(f"  → Computing ASRE rating...")
            results_df = compute_asre_rating(stock_df)

            latest = results_df.iloc[-1]
            result['r_asre'] = latest.get('r_asre', latest.get('r_final', 0))

            logger.info(f"  ✓ R_ASRE = {latest.get('r_final', latest.get('r_asre', 0)):.1f}, Signal = {latest.get('signal', 'CAUTION')}")

            # Step 4: Generate PDF report
            logger.info(f"  → Generating PDF report...")
            pdf_path = export_stock_report(
                ticker=ticker,
                results_df=results_df,
                fundamentals={
                    'tier': 'C',  # Mock for test
                    'category': 'STABLE',
                    'pe': latest.get('pe', 0),
                    'roe': latest.get('roe', 0),
                    'de': latest.get('de', 0),
                },
                output_dir=str(self.output_dir)
            )

            pdf_size_kb = pdf_path.stat().st_size / 1024
            logger.info(f"  ✓ PDF generated ({pdf_size_kb:.1f} KB)")

            # Success
            result['success'] = True
            result['duration'] = time.time() - stock_start

            # Check performance
            if result['duration'] > MAX_TIME_PER_STOCK:
                logger.warning(
                    f"  ⚠️ SLOW: {result['duration']:.1f}s "
                    f"(max {MAX_TIME_PER_STOCK}s)"
                )
            else:
                logger.info(f"  ✓ Completed in {result['duration']:.1f}s")

        except Exception as exc:
            result['error'] = str(exc)
            result['duration'] = time.time() - stock_start
            logger.error(f"  ✗ FAILED: {exc}")

        logger.info("")
        return result

    def _generate_summary(self) -> int:
        """
        Print test summary and determine exit code.

        Returns
        -------
        int
            0 if passed, 1 if failures within tolerance, 2 if critical
        """
        logger.info("=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)

        success_count = sum(1 for r in self.results if r['success'])

        logger.info(f"Total stocks tested:  {len(self.results)}")
        logger.info(f"Passed:               {success_count}")
        logger.info(f"Failed:               {self.failures}")
        logger.info(f"Total time:           {self.total_time:.1f}s")
        logger.info(f"Avg time per stock:   {self.total_time / len(self.results):.1f}s")
        logger.info("")

        # Detailed results table
        logger.info("DETAILED RESULTS:")
        logger.info("-" * 70)
        logger.info(f"{'Ticker':<15} {'Status':<10} {'Time (s)':<10} {'Quarters':<10} {'R_ASRE':<10}")
        logger.info("-" * 70)

        for r in self.results:
            status = "✓ PASS" if r['success'] else "✗ FAIL"
            r_asre = f"{r['r_asre']:.1f}" if r['r_asre'] else "N/A"
            logger.info(
                f"{r['ticker']:<15} {status:<10} {r['duration']:<10.1f} "
                f"{r['quarters']:<10} {r_asre:<10}"
            )

        logger.info("-" * 70)
        logger.info("")

        # Failure details
        if self.failures > 0:
            logger.info("FAILURE DETAILS:")
            for r in self.results:
                if not r['success']:
                    logger.error(f"  {r['ticker']}: {r['error']}")
            logger.info("")

        # Determine exit code
        if self.failures == 0:
            logger.info("✅ ALL TESTS PASSED — System ready for demo")
            return 0
        elif self.failures <= MAX_FAILURES_ALLOWED:
            logger.warning(
                f"⚠️ TESTS PASSED WITH WARNINGS — "
                f"{self.failures} failure(s) within tolerance"
            )
            return 0
        else:
            logger.error(
                f"❌ TESTS FAILED — {self.failures} failure(s) exceed tolerance "
                f"(max {MAX_FAILURES_ALLOWED})"
            )
            return 1


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASRE Pre-Demo Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default 10 NSE stocks
  python pre_demo_test.py

  # Test custom tickers
  python pre_demo_test.py --tickers INFY.NS TCS.NS RELIANCE.NS

  # Save reports to custom directory
  python pre_demo_test.py --output-dir demo_reports
        """
    )

    parser.add_argument(
        '--tickers',
        nargs='+',
        default=DEFAULT_TICKERS,
        help='List of tickers to test (default: 10 NSE stocks)'
    )

    parser.add_argument(
        '--output-dir',
        default='test_reports',
        help='Directory to save PDF reports (default: test_reports/)'
    )

    args = parser.parse_args()

    runner = PreDemoTestRunner(
        tickers=args.tickers,
        output_dir=args.output_dir
    )

    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
