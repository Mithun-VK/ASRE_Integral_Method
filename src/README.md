# ASRE - Institutional-Grade Stock Selection System

## New Structure (v3.0)

### What Changed?
- ? Added point-in-time fundamental data fetching
- ? Implemented dynamic R_ASRE calculation
- ? Eliminated look-ahead bias
- ? Institutional-grade backtesting

### Directory Structure
\\\
src/
+-- asre/
¦   +-- core/           # Core ASRE calculation logic
¦   +-- data/           # Data fetching & point-in-time management
¦   +-- backtest/       # Dynamic backtest engine
¦   +-- strategy/       # Trading strategies
¦   +-- utils/          # Utilities
 config/             # Configuration files
+-- examples/           # Usage examples
 tests/              # Unit tests
 archive/            # Old files (v2.1)
\\\

### Quick Start

1. **Install dependencies:**
   \\\ash
   pip install -r requirements.txt
   \\\

2. **Setup API key:**
   - Get free key from: https://financialmodelingprep.com/developer/docs/
   - Edit \.env\ and add: \FMP_API_KEY=your_key_here\

3. **Run example:**
   \\\ash
   python examples/run_backtest_v3.py
   \\\

### Next Steps

1. **Week 1:** Implement \undamental_fetcher.py\ (use provided code)
2. **Week 2:** Implement \point_in_time.py\ (use provided code)
3. **Week 3:** Implement \engine_v3.py\ (use provided code)
4. **Week 4:** Run backtests and validate results

### Key Differences from v2.1

| Feature | v2.1 (Old) | v3.0 (New) |
|---------|------------|------------|
| R_ASRE | Static (latest data) | Dynamic (quarterly recalculation) |
| Fundamentals | Latest only | Historical quarterly with announcement dates |
| Look-ahead bias |  Present |  Eliminated |
| Backtest validity |  Questionable |  Institutional-grade |

### Documentation

See \IMPLEMENTATION_GUIDE.md\ for detailed implementation steps.

### Support

For questions or issues, refer to the implementation guide.
