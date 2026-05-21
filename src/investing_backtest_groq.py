"""
Investment-Focused ASRE Backtest - WITH TRADE LOG + AI EXPLAINABILITY
============================================================
✅ ADDED: Detailed trade log showing Buy/Sell dates, prices, and P&L.
✅ NEW: Groq-powered AI explanations for ASRE scores and trade decisions.
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import os
from typing import Dict, List

# Import ASRE modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data_loader import DataLoader
from asre.composite import compute_complete_asre
from asre.backtest import generate_backtest_report

# ✅ NEW: Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq not installed. Run: pip install groq")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ===========================================================================
# ✅ NEW: AI EXPLAINABILITY ENGINE
# ===========================================================================

class ASREExplainer:
    """AI-powered explainability engine using Groq."""
    
    def __init__(self, api_key: str = None):
        """Initialize Groq client."""
        if not GROQ_AVAILABLE:
            self.client = None
            logger.warning("⚠️ AI Explainability disabled (Groq not installed)")
            return
        
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            self.client = None
            logger.warning("⚠️ AI Explainability disabled (No GROQ_API_KEY found)")
        else:
            self.client = Groq(api_key=api_key)
            logger.info("✅ AI Explainability Engine initialized (Groq)")
    
    def explain_asre_score(self, ticker: str, asre_score: float, 
                          components: Dict, price: float, sma: float = None) -> str:
        """Generate natural language explanation of ASRE score."""
        if not self.client:
            return "AI explanations unavailable (Groq not configured)"
        
        # Extract component scores
        momentum = components.get('r_momentum', 'N/A')
        technical = components.get('r_technical', 'N/A')
        fundamental = components.get('r_fundamental', 'N/A')
        
        # Determine quality tier
        if asre_score >= 80:
            tier = "A (Exceptional)"
        elif asre_score >= 70:
            tier = "B (Excellent)"
        elif asre_score >= 60:
            tier = "C (Strong)"
        elif asre_score >= 50:
            tier = "D (Moderate)"
        else:
            tier = "F (Weak)"
        
        # Build prompt
        prompt = f"""You are an expert investment analyst. Explain this ASRE score in simple terms for retail investors.

Stock: {ticker}
Current Price: ${price:.2f}
ASRE Score: {asre_score:.1f}/100 (Tier {tier})

Component Breakdown:
- Momentum Score: {momentum}
- Technical Score: {technical}
- Fundamental Score: {fundamental}

{"Price vs 200-day SMA: " + f"${sma:.2f} ({'Above' if price > sma else 'Below'} SMA)" if sma else ""}

Provide a 3-4 sentence explanation covering:
1. Overall quality assessment
2. What's driving the score (strengths/weaknesses)
3. Investment implication (bullish/bearish/neutral)

Keep it conversational and avoid jargon."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fast, high-quality model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"AI explanation error: {e}")
            return f"AI explanation unavailable (Error: {str(e)})"
    
    def explain_trade_decision(self, action: str, ticker: str, price: float,
                              asre_score: float, reason: str, 
                              position_size: float) -> str:
        """Generate natural language explanation of trade decision."""
        if not self.client:
            return f"{action} executed based on {reason}"
        
        prompt = f"""You are an investment advisor. Explain this portfolio action to a client.

Action: {action}
Stock: {ticker}
Price: ${price:.2f}
Position Size: {position_size*100:.0f}%
ASRE Score: {asre_score:.1f}/100
Trigger: {reason}

Provide a 2-3 sentence explanation that:
1. States what happened and why
2. Explains the risk/reward logic
3. Reassures the investor this is systematic

Be confident but measured. No jargon."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"{action} - {reason}"
    
    def generate_portfolio_summary(self, report: Dict, ticker: str, 
                                   trade_log: List) -> str:
        """Generate executive summary of backtest results."""
        if not self.client:
            return "Portfolio summary unavailable (AI disabled)"
        
        prompt = f"""You are a portfolio manager presenting results to a client.

Stock: {ticker}
Backtest Period: {report['years']:.1f} years

Performance:
- Total Return: {report['total_return']:.1f}%
- CAGR: {report['cagr']:.1f}%
- Sharpe Ratio: {report['sharpe']:.2f}
- Max Drawdown: {report['max_drawdown']:.1f}%

Trading:
- Rebalances: {report['num_trades']}
- Win Rate: {report['win_rate']:.1f}%
- Time in Market: {report['time_in_market']:.1f}%

Write a 4-5 sentence executive summary that:
1. Characterizes overall performance (excellent/good/poor)
2. Highlights the best metric
3. Mentions the biggest risk (drawdown)
4. Concludes with investment suitability

Be honest, professional, and client-focused."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ASRE strategy for {ticker} achieved {report['cagr']:.1f}% CAGR with {report['sharpe']:.2f} Sharpe ratio."

# ===========================================================================
# INVESTMENT SIGNAL ENGINE (UNCHANGED)
# ===========================================================================

class InvestmentSignalEngine:
    def __init__(self,
                 overweight_threshold: float = 75.0,
                 full_threshold: float = 60.0,
                 reduce_threshold: float = 45.0,
                 exit_threshold: float = 30.0,
                 rebalance_period: str = 'W'):
        self.overweight = overweight_threshold
        self.full = full_threshold
        self.reduce = reduce_threshold
        self.exit = exit_threshold
        self.rebalance_period = rebalance_period

    def generate_signals(self, df: pd.DataFrame, rating_col: str = 'r_asre') -> pd.DataFrame:
        """Generate tiered allocation signals (FIXED: merge_asof)."""
        df = df.copy()

        resample_rule = self.rebalance_period.replace('M', 'ME')
        df_resampled = df.resample(resample_rule).last()

        def get_allocation(rating):
            if pd.isna(rating): return 0.0
            elif rating >= self.overweight: return 1.20
            elif rating >= self.full: return 1.00
            elif rating >= self.reduce: return 0.50
            elif rating >= self.exit: return 0.25
            else: return 0.00

        df_resampled['target_allocation'] = df_resampled[rating_col].apply(get_allocation)

        df_resampled = df_resampled.sort_index()
        temp_alloc = df_resampled[['target_allocation']].reset_index()
        temp_alloc.columns = ['rebalance_date', 'target_allocation']

        df_reset = df.reset_index()

        df_merged = pd.merge_asof(
            df_reset,
            temp_alloc,
            left_on='date',
            right_on='rebalance_date',
            direction='backward'
        )

        df_merged.index = df.index
        df['allocation'] = df_merged['target_allocation'].fillna(method='ffill').fillna(0.0)

        df['prev_allocation'] = df['allocation'].shift(1).fillna(0.0)
        df['is_rebalance_date'] = df['allocation'] != df['prev_allocation']

        if df['allocation'].iloc[0] > 0:
            df.iloc[0, df.columns.get_loc('is_rebalance_date')] = True

        return df

# ===========================================================================
# INVESTMENT POSITION MANAGER (UPDATED FOR AI LOGGING)
# ===========================================================================

def compute_investment_returns(
    df: pd.DataFrame,
    allocation_col: str = 'allocation',
    price_col: str = 'close',
    transaction_cost: float = 0.0005,
    stop_loss_pct: float = 0.25,
    trailing_stop_pct: float = 0.15,
    ltcg_tax: float = 0.10,
    stcg_tax: float = 0.15,
    explainer: ASREExplainer = None,  # ✅ NEW
    ticker: str = "STOCK",  # ✅ NEW
) -> pd.DataFrame:

    result_df = df.copy()
    result_df['position'] = 0.0
    result_df['entry_price'] = np.nan
    result_df['entry_date'] = pd.NaT
    result_df['position_high'] = np.nan
    result_df['exit_reason'] = ''
    result_df['trade_pnl'] = 0.0
    result_df['days_held'] = 0
    result_df['tax_applied'] = 0.0

    # Trade Log List
    trade_log = []

    current_position = 0.0
    entry_price = np.nan
    entry_date = None
    position_high = np.nan

    for idx, row in result_df.iterrows():
        target_allocation = row[allocation_col]
        current_price = row[price_col]

        if pd.isna(position_high) and current_position > 0:
            position_high = current_price
        elif current_position > 0:
            position_high = max(position_high, current_price)

        exit_triggered = False
        exit_reason = ''

        if current_position > 0:
            sl_price = entry_price * (1 - stop_loss_pct)
            if current_price <= sl_price:
                exit_triggered = True
                exit_reason = 'Stop_Loss'
            elif not pd.isna(position_high):
                trailing_price = position_high * (1 - trailing_stop_pct)
                if current_price <= trailing_price:
                    exit_triggered = True
                    exit_reason = 'Trailing_Stop'

        if exit_triggered:
            trade_return = (current_price - entry_price) / entry_price
            days_held = (idx - entry_date).days if entry_date else 0
            tax_rate = stcg_tax if days_held < 365 else ltcg_tax
            net_pnl = trade_return * (1 - tax_rate) if trade_return > 0 else trade_return

            result_df.at[idx, 'trade_pnl'] = net_pnl * current_position
            result_df.at[idx, 'exit_reason'] = exit_reason
            result_df.at[idx, 'days_held'] = days_held
            result_df.at[idx, 'tax_applied'] = tax_rate if trade_return > 0 else 0.0

            # ✅ AI EXPLANATION
            asre_score = row.get('r_asre', 0)
            ai_explanation = ""
            if explainer:
                ai_explanation = explainer.explain_trade_decision(
                    "SELL", ticker, current_price, asre_score, 
                    exit_reason, 0.0
                )

            trade_log.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Action': 'SELL (Risk Exit)',
                'Reason': exit_reason,
                'Price': f"{current_price:.2f}",
                'Size': f"{current_position*100:.0f}%",
                'PnL': f"{net_pnl*100:.2f}%",
                'AI_Explanation': ai_explanation  # ✅ NEW
            })

            current_position = 0.0
            entry_price = np.nan
            entry_date = None
            position_high = np.nan

        if row.get('is_rebalance_date', False) or exit_triggered:
            position_change = target_allocation - current_position

            if abs(position_change) > 0.01:
                action = 'BUY' if position_change > 0 else 'SELL'
                reason = 'Rebalance'

                if position_change > 0:
                    if current_position == 0:
                        entry_price = current_price
                        entry_date = idx
                        position_high = current_price

                    # ✅ AI EXPLANATION
                    asre_score = row.get('r_asre', 0)
                    ai_explanation = ""
                    if explainer:
                        ai_explanation = explainer.explain_trade_decision(
                            f"{action}", ticker, current_price, asre_score,
                            'Entry/Increase', target_allocation
                        )

                    trade_log.append({
                        'Date': idx.strftime('%Y-%m-%d'),
                        'Action': f"{action} (+{position_change*100:.0f}%)",
                        'Reason': 'Entry/Increase',
                        'Price': f"{current_price:.2f}",
                        'Size': f"{target_allocation*100:.0f}%",
                        'PnL': '-',
                        'AI_Explanation': ai_explanation  # ✅ NEW
                    })

                    current_position = target_allocation

                elif position_change < 0:
                    if target_allocation == 0:
                        trade_return = (current_price - entry_price) / entry_price if not pd.isna(entry_price) else 0
                        days_held = (idx - entry_date).days if entry_date else 0
                        tax_rate = stcg_tax if days_held < 365 else ltcg_tax
                        net_pnl = trade_return * (1 - tax_rate) if trade_return > 0 else trade_return

                        result_df.at[idx, 'trade_pnl'] = net_pnl * current_position
                        result_df.at[idx, 'exit_reason'] = 'Signal_Exit'
                        result_df.at[idx, 'days_held'] = days_held
                        result_df.at[idx, 'tax_applied'] = tax_rate if trade_return > 0 else 0.0

                        # ✅ AI EXPLANATION
                        asre_score = row.get('r_asre', 0)
                        ai_explanation = ""
                        if explainer:
                            ai_explanation = explainer.explain_trade_decision(
                                "SELL", ticker, current_price, asre_score,
                                'Signal < Exit', 0.0
                            )

                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': 'SELL (Full Exit)',
                            'Reason': 'Signal < Exit',
                            'Price': f"{current_price:.2f}",
                            'Size': '0%',
                            'PnL': f"{net_pnl*100:.2f}%",
                            'AI_Explanation': ai_explanation  # ✅ NEW
                        })

                        entry_price = np.nan
                        entry_date = None
                        position_high = np.nan

                    else:
                        # ✅ AI EXPLANATION
                        asre_score = row.get('r_asre', 0)
                        ai_explanation = ""
                        if explainer:
                            ai_explanation = explainer.explain_trade_decision(
                                "SELL", ticker, current_price, asre_score,
                                'Reduce Exposure', target_allocation
                            )

                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': f"SELL ({position_change*100:.0f}%)",
                            'Reason': 'Reduce Exposure',
                            'Price': f"{current_price:.2f}",
                            'Size': f"{target_allocation*100:.0f}%",
                            'PnL': '-',
                            'AI_Explanation': ai_explanation  # ✅ NEW
                        })

                    current_position = target_allocation

        result_df.at[idx, 'position'] = current_position
        result_df.at[idx, 'entry_price'] = entry_price
        result_df.at[idx, 'entry_date'] = entry_date
        result_df.at[idx, 'position_high'] = position_high

    result_df['price_return'] = result_df[price_col].pct_change()
    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['price_return']
    result_df['position_change'] = result_df['position'].diff().abs()
    result_df['transaction_cost_incurred'] = result_df['position_change'] * transaction_cost
    result_df['net_return'] = result_df['strategy_return'] - result_df['transaction_cost_incurred']
    result_df['cumulative_return'] = (1 + result_df['net_return']).cumprod()

    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax

    return result_df, trade_log

# ===========================================================================
# DATA FETCHING (UNCHANGED)
# ===========================================================================

def fetch_all_data(ticker: str, start_date: str, end_date: str):
    logger.info(f"\n{'='*80}")
    logger.info(f"FETCHING DATA FOR {ticker}")
    logger.info(f"{'='*80}")

    logger.info(f"\n📊 Step 1: Fetching Quarterly Fundamentals...")
    fetcher = FundamentalFetcher()
    try:
        df_fundamentals = fetcher.fetch_quarterly_fundamentals(ticker, start_date, end_date)
        logger.info(f"   ✅ Fetched {len(df_fundamentals)} quarters")
    except Exception as e:
        logger.error(f"   ❌ Error fetching fundamentals: {e}")
        df_fundamentals = None

    logger.info(f"\n📈 Step 2: Loading Price Data & Merging Fundamentals...")
    loader = DataLoader()
    try:
        df = loader.load_stock_data(ticker, start_date, end_date, quarterly_fundamentals=df_fundamentals)
        logger.info(f"   ✅ Loaded {len(df)} days of data")
    except Exception as e:
        logger.error(f"   ❌ Error loading data: {e}")
        raise

    logger.info(f"\n🎯 Step 3: Computing ASRE Ratings...")
    try:
        df_complete = compute_complete_asre(df, medallion=True, return_all_components=True)
        logger.info(f"   ✅ ASRE ratings computed")
        df_complete['date'] = pd.to_datetime(df_complete['date'])
        df_complete = df_complete.set_index('date')
    except Exception as e:
        logger.error(f"   ❌ Error computing ASRE: {e}")
        raise

    return df_complete

# ===========================================================================
# INVESTMENT BACKTEST ENGINE (UPDATED WITH AI)
# ===========================================================================

def run_investment_backtest(
    df: pd.DataFrame,
    rating_col: str = 'r_asre',
    overweight_threshold: float = 75.0,
    full_threshold: float = 60.0,
    reduce_threshold: float = 45.0,
    exit_threshold: float = 30.0,
    rebalance_period: str = 'W',
    explainer: ASREExplainer = None,  # ✅ NEW
    ticker: str = "STOCK",  # ✅ NEW
):
    logger.info(f"\n{'='*80}")
    logger.info(f"INVESTMENT BACKTEST ENGINE {'+ AI EXPLAINABILITY' if explainer and explainer.client else ''}")
    logger.info(f"Rating: {rating_col} | Rebalance: {rebalance_period}")
    logger.info(f"Tiers: Exit<{exit_threshold}, Reduce<{reduce_threshold}, Full<{full_threshold}, OW>={overweight_threshold}")
    logger.info(f"{'='*80}")

    engine = InvestmentSignalEngine(
        overweight_threshold=overweight_threshold,
        full_threshold=full_threshold,
        reduce_threshold=reduce_threshold,
        exit_threshold=exit_threshold,
        rebalance_period=rebalance_period
    )

    df_signals = engine.generate_signals(df, rating_col=rating_col)

    df_results, trade_log = compute_investment_returns(
        df_signals,
        allocation_col='allocation',
        price_col='close',
        transaction_cost=0.0005,
        stop_loss_pct=0.25,
        trailing_stop_pct=0.15,
        ltcg_tax=0.10,
        stcg_tax=0.15,
        explainer=explainer,  # ✅ NEW
        ticker=ticker  # ✅ NEW
    )

    report = generate_investment_report(df_results)
    
    # ✅ NEW: Generate ASRE score explanation for latest date
    latest_idx = df_results.index[-1]
    latest_row = df_results.loc[latest_idx]
    asre_explanation = ""
    if explainer and explainer.client:
        components = {
            'r_momentum': latest_row.get('r_momentum', 'N/A'),
            'r_technical': latest_row.get('r_technical', 'N/A'),
            'r_fundamental': latest_row.get('r_fundamental', 'N/A')
        }
        asre_explanation = explainer.explain_asre_score(
            ticker,
            latest_row.get('r_asre', 0),
            components,
            latest_row['close']
        )
    
    return df_results, report, trade_log, asre_explanation  # ✅ UPDATED

# ===========================================================================
# REPORT GENERATION (UNCHANGED)
# ===========================================================================

def generate_investment_report(df: pd.DataFrame) -> dict:
    total_return = (df['cumulative_return'].iloc[-1] - 1) * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((df['cumulative_return'].iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = df['net_return'].std() * np.sqrt(252) * 100
    sharpe = (cagr / volatility) if volatility > 0 else 0
    max_dd = df['drawdown'].min() * 100
    avg_dd = df[df['drawdown'] < 0]['drawdown'].mean() * 100 if (df['drawdown'] < 0).any() else 0

    trades = df[df['trade_pnl'] != 0]
    num_trades = len(trades)
    wins = trades[trades['trade_pnl'] > 0]
    losses = trades[trades['trade_pnl'] < 0]

    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0
    avg_win = wins['trade_pnl'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = losses['trade_pnl'].mean() * 100 if len(losses) > 0 else 0
    profit_factor = abs(wins['trade_pnl'].sum() / losses['trade_pnl'].sum()) if len(losses) > 0 and losses['trade_pnl'].sum() != 0 else 0
    total_tax = trades['tax_applied'].sum() if 'tax_applied' in trades.columns else 0

    total_position_change = df['position_change'].sum()
    annual_turnover = (total_position_change / years) * 100 if years > 0 else 0
    time_in_market = (df['position'] > 0).sum() / len(df) * 100

    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_tax_rate': total_tax,
        'annual_turnover': annual_turnover,
        'time_in_market': time_in_market,
        'years': years
    }

# ✅ UPDATED: Print report with AI summaries
def print_investment_report(report: dict, trade_log: list, ticker: str, 
                           asre_explanation: str = "", explainer: ASREExplainer = None):
    print(f"\n{'='*80}")
    print(f"INVESTMENT PERFORMANCE REPORT - {ticker}")
    print(f"{'='*80}\n")
    
    # ✅ NEW: AI Executive Summary
    if explainer and explainer.client:
        print(f"🤖 AI EXECUTIVE SUMMARY")
        print(f"{'-'*80}")
        ai_summary = explainer.generate_portfolio_summary(report, ticker, trade_log)
        print(f"{ai_summary}\n")
        print(f"{'='*80}\n")
    
    # ✅ NEW: Current ASRE Analysis
    if asre_explanation:
        print(f"📊 CURRENT ASRE ANALYSIS")
        print(f"{'-'*80}")
        print(f"{asre_explanation}\n")
        print(f"{'='*80}\n")
    
    print(f"📈 Return Metrics")
    print(f"  Total Return:            {report['total_return']:.2f}%")
    print(f"  CAGR:                    {report['cagr']:.2f}%")
    print(f"  Volatility (Annual):     {report['volatility']:.2f}%")
    print(f"  Sharpe Ratio:            {report['sharpe']:.3f}")
    print(f"\n📉 Risk Metrics")
    print(f"  Max Drawdown:            {report['max_drawdown']:.2f}%")
    print(f"  Avg Drawdown:            {report['avg_drawdown']:.2f}%")
    print(f"\n💼 Investment Behavior")
    print(f"  Time in Market:          {report['time_in_market']:.1f}%")
    print(f"  Annual Turnover:         {report['annual_turnover']:.1f}%")
    print(f"  Number of Rebalances:    {report['num_trades']}")
    print(f"\n🎯 Trade Statistics")
    print(f"  Win Rate:                {report['win_rate']:.2f}%")
    print(f"  Avg Win:                 {report['avg_win']:.2f}%")
    print(f"  Avg Loss:                {report['avg_loss']:.2f}%")
    print(f"  Profit Factor:           {report['profit_factor']:.2f}")
    print(f"\n💰 Tax Impact")
    print(f"  Total Tax Drag:          {report['total_tax_rate']:.2f}%")

    print(f"\n{'='*80}")
    print(f"📝 DETAILED TRADE LOG ({len(trade_log)} Events)")
    print(f"{'='*80}")
    
    # Print first 10 trades with AI explanations
    display_limit = min(10, len(trade_log))
    for i, t in enumerate(trade_log[:display_limit]):
        print(f"\n[{i+1}] {t['Date']} - {t['Action']}")
        print(f"    Price: {t['Price']} | Size: {t['Size']} | PnL: {t['PnL']}")
        if t.get('AI_Explanation'):
            print(f"    💬 AI: {t['AI_Explanation']}")
    
    if len(trade_log) > display_limit:
        print(f"\n... ({len(trade_log) - display_limit} more trades in CSV)")
    
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Investment-Focused ASRE Backtest with AI Explainability")
    parser.add_argument("ticker", help="Stock ticker (e.g., AAPL, META)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--rating", default='r_asre', choices=['r_asre', 'r_final'],
                       help="Rating column (default: r_asre)")
    parser.add_argument("--rebalance", default='W', choices=['W', 'M', 'Q'],
                       help="Rebalancing frequency (default: W)")
    parser.add_argument("--overweight", type=float, default=75.0, help="Overweight threshold")
    parser.add_argument("--full", type=float, default=60.0, help="Full position threshold")
    parser.add_argument("--reduce", type=float, default=45.0, help="Reduce position threshold")
    parser.add_argument("--exit", type=float, default=30.0, help="Exit threshold")
    parser.add_argument("--groq-api-key", help="Groq API key (or set GROQ_API_KEY env var)")  # ✅ NEW
    parser.add_argument("--no-ai", action="store_true", help="Disable AI explanations")  # ✅ NEW

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"🏦 INVESTMENT BACKTEST - {args.ticker}")
    print(f"{'='*80}")

    # ✅ NEW: Initialize AI Explainer
    explainer = None
    if not args.no_ai:
        explainer = ASREExplainer(api_key=args.groq_api_key)

    try:
        df = fetch_all_data(args.ticker, args.start_date, args.end_date)
        df_results, report, trade_log, asre_explanation = run_investment_backtest(
            df,
            rating_col=args.rating,
            overweight_threshold=args.overweight,
            full_threshold=args.full,
            reduce_threshold=args.reduce,
            exit_threshold=args.exit,
            rebalance_period=args.rebalance,
            explainer=explainer,  # ✅ NEW
            ticker=args.ticker  # ✅ NEW
        )
        print_investment_report(report, trade_log, args.ticker, asre_explanation, explainer)
        
        # Save results
        output_file = f"investment_backtest_{args.ticker}_{args.rating}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(output_file)
        
        # ✅ NEW: Save trade log with AI explanations
        trade_log_file = f"trade_log_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(trade_log).to_csv(trade_log_file, index=False)
        
        print(f"\n✅ Results saved to {output_file}")
        print(f"✅ Trade log saved to {trade_log_file}")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
