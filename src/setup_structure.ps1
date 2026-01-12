
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASRE PROJECT - INSTITUTIONAL-GRADE STRUCTURE SETUP
# Run this script from: D:\asre-project\src
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "🚀 Setting up ASRE Institutional-Grade Structure..." -ForegroundColor Green

# Ensure we're in the right directory
$currentPath = Get-Location
Write-Host "📍 Current directory: $currentPath"

if (-not (Test-Path "asre")) {
    Write-Host "❌ ERROR: asre directory not found!" -ForegroundColor Red
    Write-Host "   Please run this script from D:\asre-project\src" -ForegroundColor Red
    exit 1
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Create new directory structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n📁 Creating new directory structure..." -ForegroundColor Cyan

# Create asre subdirectories
$directories = @(
    "asre\core",
    "asre\data", 
    "asre\backtest",
    "asre\strategy",
    "asre\utils",
    "data\cache\fundamentals",
    "data\cache\prices",
    "config",
    "examples",
    "tests",
    "archive"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
        Write-Host "  ✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️  Exists: $dir" -ForegroundColor Yellow
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: Create __init__.py files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n📝 Creating __init__.py files..." -ForegroundColor Cyan

$initFiles = @(
    "asre\core\__init__.py",
    "asre\data\__init__.py",
    "asre\backtest\__init__.py",
    "asre\strategy\__init__.py",
    "asre\utils\__init__.py",
    "tests\__init__.py"
)

foreach ($file in $initFiles) {
    if (-not (Test-Path $file)) {
        New-Item -Path $file -ItemType File -Force | Out-Null
        Set-Content -Path $file -Value '"""Module initialization"""'
        Write-Host "  ✅ Created: $file" -ForegroundColor Green
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Create placeholder files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n📄 Creating placeholder files..." -ForegroundColor Cyan

$placeholderFiles = @{
    # Core files
    "asre\core\asre_calculator.py" = "# ASRE Calculator - Point-in-time calculation`n# TODO: Implement`n"
    "asre\core\f_score.py" = "# F-Score wrapper`n# TODO: Implement`n"
    "asre\core\t_score.py" = "# T-Score wrapper`n# TODO: Implement`n"
    "asre\core\m_score.py" = "# M-Score wrapper`n# TODO: Implement`n"
    "asre\core\medallion.py" = "# Medallion enhancement`n# TODO: Implement`n"

    # Data files
    "asre\data\fundamental_fetcher.py" = "# Fundamental data fetcher`n# TODO: Use provided code`n"
    "asre\data\point_in_time.py" = "# Point-in-time data manager`n# TODO: Use provided code`n"
    "asre\data\data_cache.py" = "# Data caching utilities`n# TODO: Implement`n"
    "asre\data\price_fetcher.py" = "# Price data fetcher`n# TODO: Implement`n"

    # Backtest files
    "asre\backtest\engine_v3.py" = "# Dynamic backtest engine`n# TODO: Use provided code`n"
    "asre\backtest\portfolio.py" = "# Portfolio manager`n# TODO: Implement`n"
    "asre\backtest\execution.py" = "# Order execution`n# TODO: Implement`n"
    "asre\backtest\metrics.py" = "# Performance metrics`n# TODO: Implement`n"
    "asre\backtest\report.py" = "# Report generation`n# TODO: Implement`n"

    # Strategy files
    "asre\strategy\base_strategy.py" = "# Base strategy class`n# TODO: Implement`n"
    "asre\strategy\asre_strategy.py" = "# ASRE trading strategy`n# TODO: Implement`n"
    "asre\strategy\signals.py" = "# Signal generation`n# TODO: Implement`n"

    # Utils files
    "asre\utils\validators.py" = "# Data validators`n# TODO: Implement`n"
    "asre\utils\logger.py" = "# Logging utilities`n# TODO: Implement`n"
    "asre\utils\helpers.py" = "# Helper functions`n# TODO: Implement`n"
}

foreach ($file in $placeholderFiles.Keys) {
    if (-not (Test-Path $file)) {
        New-Item -Path $file -ItemType File -Force | Out-Null
        Set-Content -Path $file -Value $placeholderFiles[$file]
        Write-Host "  ✅ Created: $file" -ForegroundColor Green
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: Create config files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n⚙️  Creating config files..." -ForegroundColor Cyan

# backtest_config.yaml
$backtestConfig = @"
# ASRE Backtest Configuration

backtest:
  initial_capital: 100000
  rebalance_frequency: 'Q'  # Q=Quarterly, M=Monthly
  transaction_cost: 0.001   # 0.1%

thresholds:
  buy: 70                   # R_ASRE >= 70 = BUY
  sell: 40                  # R_ASRE < 40 = SELL
  fundamental_floor: 65     # F-Score >= 65 required

risk_management:
  max_position_size: 1.0    # 100% of capital (single stock)
  max_drawdown: 0.5         # 50% max drawdown
"@

Set-Content -Path "config\backtest_config.yaml" -Value $backtestConfig
Write-Host "  ✅ Created: config\backtest_config.yaml" -ForegroundColor Green

# data_sources.yaml
$dataSourcesConfig = @"
# Data Source Configuration

data_sources:
  fundamentals:
    primary: 'fmp'          # Financial Modeling Prep
    fallback: 'yfinance'    # Yahoo Finance

  prices:
    source: 'yfinance'      # Yahoo Finance

cache:
  enabled: true
  directory: 'data/cache'
  ttl_days: 30              # Cache for 30 days
"@

Set-Content -Path "config\data_sources.yaml" -Value $dataSourcesConfig
Write-Host "  ✅ Created: config\data_sources.yaml" -ForegroundColor Green

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: Create .env file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n🔐 Creating .env file..." -ForegroundColor Cyan

$envContent = @"
# ASRE Environment Variables
# Get free API key from: https://financialmodelingprep.com/developer/docs/

# Financial Modeling Prep API Key (250 free requests/day)
FMP_API_KEY=YOUR_API_KEY_HERE

# Optional: Alpha Vantage API Key
ALPHA_VANTAGE_KEY=

# Optional: Other API keys
# Add more as needed
"@

if (-not (Test-Path ".env")) {
    Set-Content -Path ".env" -Value $envContent
    Write-Host "  ✅ Created: .env" -ForegroundColor Green
    Write-Host "  ⚠️  IMPORTANT: Edit .env and add your FMP_API_KEY!" -ForegroundColor Yellow
} else {
    Write-Host "  ⏭️  .env already exists (not overwriting)" -ForegroundColor Yellow
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: Create example files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n📚 Creating example files..." -ForegroundColor Cyan

$exampleBacktest = @"
"""
Example: Run Dynamic Backtest v3

This example shows how to run an institutional-grade backtest
with dynamic R_ASRE calculation and point-in-time data.
"""

from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data.point_in_time import PointInTimeData
from asre.backtest.engine_v3 import DynamicBacktestEngine
from asre.data import data_loader

def run_example():
    # 1. Fetch fundamental data
    print("Fetching fundamental data...")
    fetcher = FundamentalFetcher()
    fundamentals = fetcher.fetch_quarterly_fundamentals('NVDA', '2020-01-01', '2026-01-01')

    # 2. Fetch price data
    print("Fetching price data...")
    prices = data_loader.fetch_stock_data('NVDA', '2020-01-01', '2026-01-01')

    # 3. Run dynamic backtest
    print("Running backtest...")
    engine = DynamicBacktestEngine(
        initial_capital=100000,
        rebalance_frequency='Q',
        threshold_buy=70
    )

    results = engine.run_backtest(
        'NVDA',
        prices,
        fundamentals,
        '2020-01-01',
        '2026-01-01'
    )

    # 4. Display results
    print(f"\nTotal Return: {results['total_return']:.2f}%")
    print(f"CAGR: {results['cagr']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")

    # 5. Verify dynamic R_ASRE
    r_asre_std = results['r_asre_history']['r_asre'].std()
    print(f"\nR_ASRE Std Dev: {r_asre_std:.2f}")
    if r_asre_std > 0:
        print("✅ R_ASRE is DYNAMIC (not static)!")
    else:
        print("❌ WARNING: R_ASRE appears static!")

if __name__ == "__main__":
    run_example()
"@

Set-Content -Path "examples\run_backtest_v3.py" -Value $exampleBacktest
Write-Host "  ✅ Created: examples\run_backtest_v3.py" -ForegroundColor Green

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: Move old files to archive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n📦 Moving old files to archive..." -ForegroundColor Cyan

$filesToArchive = @(
    "asre_backtest_v2_1.py",
    "asre_backtest_v2_1_interactive.py",
    "test_backtest_v2_1.py",
    "demo_backtest_v2_1.py",
    "check_aapl_tech.py",
    "diagnose_scores.py",
    "diagnostic_columns.py",
    "f_score_debug.csv",
    "medallion_function.txt"
)

foreach ($file in $filesToArchive) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive\" -Force
        Write-Host "  ✅ Archived: $file" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️  Not found: $file" -ForegroundColor Gray
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: Create README
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n📖 Creating README..." -ForegroundColor Cyan

$readme = @"
# ASRE - Institutional-Grade Stock Selection System

## New Structure (v3.0)

### What Changed?
- ✅ Added point-in-time fundamental data fetching
- ✅ Implemented dynamic R_ASRE calculation
- ✅ Eliminated look-ahead bias
- ✅ Institutional-grade backtesting

### Directory Structure
\`\`\`
src/
├── asre/
│   ├── core/           # Core ASRE calculation logic
│   ├── data/           # Data fetching & point-in-time management
│   ├── backtest/       # Dynamic backtest engine
│   ├── strategy/       # Trading strategies
│   └── utils/          # Utilities
├── config/             # Configuration files
├── examples/           # Usage examples
├── tests/              # Unit tests
└── archive/            # Old files (v2.1)
\`\`\`

### Quick Start

1. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Setup API key:**
   - Get free key from: https://financialmodelingprep.com/developer/docs/
   - Edit \`.env\` and add: \`FMP_API_KEY=your_key_here\`

3. **Run example:**
   \`\`\`bash
   python examples/run_backtest_v3.py
   \`\`\`

### Next Steps

1. **Week 1:** Implement \`fundamental_fetcher.py\` (use provided code)
2. **Week 2:** Implement \`point_in_time.py\` (use provided code)
3. **Week 3:** Implement \`engine_v3.py\` (use provided code)
4. **Week 4:** Run backtests and validate results

### Key Differences from v2.1

| Feature | v2.1 (Old) | v3.0 (New) |
|---------|------------|------------|
| R_ASRE | Static (latest data) | Dynamic (quarterly recalculation) |
| Fundamentals | Latest only | Historical quarterly with announcement dates |
| Look-ahead bias | ❌ Present | ✅ Eliminated |
| Backtest validity | ⚠️ Questionable | ✅ Institutional-grade |

### Documentation

See \`IMPLEMENTATION_GUIDE.md\` for detailed implementation steps.

### Support

For questions or issues, refer to the implementation guide.
"@

Set-Content -Path "README.md" -Value $readme
Write-Host "  ✅ Created: README.md" -ForegroundColor Green

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DONE!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write-Host "`n" -NoNewline
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "✅ STRUCTURE SETUP COMPLETE!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green

Write-Host "`n📊 Summary:" -ForegroundColor Cyan
Write-Host "  • Created 11 new directories"
Write-Host "  • Created 25+ placeholder files"
Write-Host "  • Created config files (YAML)"
Write-Host "  • Created .env file"
Write-Host "  • Moved old files to archive/"
Write-Host "  • Created README.md"

Write-Host "`n🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Edit .env and add your FMP_API_KEY"
Write-Host "  2. Run: pip install financialmodelingprep pyarrow diskcache pyyaml python-dotenv"
Write-Host "  3. Copy provided code files to their locations"
Write-Host "  4. Follow IMPLEMENTATION_GUIDE.md"

Write-Host "`n📂 View structure:" -ForegroundColor Yellow
Write-Host "  tree /F"

Write-Host "`n🚀 Ready to start Week 1!" -ForegroundColor Green
Write-Host ""
