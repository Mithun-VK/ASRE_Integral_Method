# ASRE - Advanced Stock Rating Engine

A quantitative stock rating system combining momentum, technical, and fundamental analysis with statistical calibration and Kalman filtering.

## Installation

\\\ash
pip install -e .
\\\

## Quick Start

\\\python
from asre.data_loader import load_stock_data
from asre.composite import compute_asre_rating

# Load data
df = load_stock_data('AAPL', start_date='2020-01-01')

# Compute ASRE rating
ratings = compute_asre_rating(df)
print(ratings.tail())
\\\

## Project Structure

- \config/\ - Configuration files (YAML)
- \data/\ - Raw, processed, and example data
- \
otebooks/\ - Jupyter notebooks for exploration
- \src/asre/\ - Main package modules
- \	ests/\ - Unit and integration tests

## Modules

- \indicators.py\ - Technical indicators (RSI, EMA, volatility, etc.)
- \momentum.py\ - Momentum score (M-score)
- \	echnical.py\ - Technical score (T-score)
- \undamentals.py\ - Fundamental score (F-score)
- \composite.py\ - Composite rating (R_final, R_ASRE)
- \calibration.py\ - MLE and Kalman filter
- \acktest.py\ - Backtesting framework

## Configuration

Edit \config/asre_params.yaml\ to adjust hyperparameters:

\\\yaml
momentum:
  kappa: 0.03
  beta_m: 0.2
  window_60d: 60

technical:
  gamma: 0.1
  theta: 0.1
  window_200d: 200
  window_20d: 20

fundamentals:
  alpha: 0.02
  window_252d: 252
\\\

## Running Tests

\\\ash
pytest tests/ -v
pytest tests/ --cov=src/asre
\\\

## CLI Commands

\\\ash
# Compute ASRE rating for a stock
python -m asre.cli run-daily --ticker AAPL --date 2024-01-15

# Backtest strategy
python -m asre.cli backtest --start 2020-01-01 --end 2024-12-31 --universe sp500

# Optimize hyperparameters
python -m asre.cli optimize --config config/asre_params.yaml
\\\

## References

- ASRE Formula Documentation: \docs/ASRE_Complete_Formulae.md\
- Concept Guide: \docs/ASRE_Concepts.md\

## License

MIT
