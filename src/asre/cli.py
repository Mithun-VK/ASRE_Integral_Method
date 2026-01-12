"""
ASRE Command-Line Interface (ENHANCED v2.1 - DIP QUALITY INTEGRATED)

✅ NEW: Dip Quality Score integration in Market Context
✅ NEW: Shows both R_Final and R_ASRE (Medallion) scores
✅ NEW: Dual-score comparison tables
✅ NEW: Intelligent buy-the-dip detection with quality assessment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from time import sleep

import pandas as pd
import numpy as np

# Optional rich library for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich import box
    from rich.tree import Tree
    from rich.align import Align
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("⚠️  Warning: 'rich' library not installed. Install with: pip install rich")
    print("   Using fallback plain text output.\n")

from .config import (
    MomentumConfig,
    TechnicalConfig,
    FundamentalsConfig,
    CompositeConfig,
    BacktestConfig,
)
from .data_loader import load_stock_data
from .composite import compute_complete_asre, validate_asre_rating
from .backtest import Backtester
from .optimization import ASREOptimizer, DEFAULT_PARAM_SPACES


# ---------------------------------------------------------------------------
# Constants & Theme
# ---------------------------------------------------------------------------

THEME_COLORS = {
    'primary': 'cyan',
    'secondary': 'blue',
    'success': 'green',
    'warning': 'yellow',
    'danger': 'red',
    'info': 'magenta',
    'muted': 'bright_black',
}

SIGNAL_EMOJI = {
    'strong_buy': '🚀',
    'buy': '📈',
    'hold': '⚖️',
    'caution': '⚠️',
    'sell': '📉',
    'strong_sell': '🔻',
}

logger = logging.getLogger(__name__)

if HAS_RICH:
    console = Console()
else:
    console = None


# ---------------------------------------------------------------------------
# Setup & Utilities
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO

    if HAS_RICH:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[RichHandler(
                rich_tracebacks=True,
                show_time=False,
                show_path=False,
            )]
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )


def print_header(title: str, subtitle: Optional[str] = None):
    """Print elegant header with branding."""
    if HAS_RICH and console:
        header_text = f"[bold cyan]ASRE[/bold cyan] [dim]|[/dim] {title}"

        if subtitle:
            content = f"{header_text}\n[dim]{subtitle}[/dim]"
        else:
            content = header_text

        console.print(Panel(
            Align.center(content),
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        ))
        console.print()
    else:
        print(f"\n{'='*70}")
        print(f"ASRE | {title}")
        if subtitle:
            print(subtitle)
        print(f"{'='*70}\n")


def print_success(message: str):
    """Print success message."""
    if HAS_RICH and console:
        console.print(f"✓ [green]{message}[/green]")
    else:
        print(f"✓ {message}")


def print_error(message: str):
    """Print error message."""
    if HAS_RICH and console:
        console.print(f"✗ [red]{message}[/red]")
    else:
        print(f"✗ {message}")


def print_info(message: str):
    """Print info message."""
    if HAS_RICH and console:
        console.print(f"ℹ [cyan]{message}[/cyan]")
    else:
        print(f"ℹ {message}")


def safe_date_format(date_val: Any, default: str = "N/A") -> str:
    """Safely format date value to string."""
    if pd.isna(date_val):
        return default

    if isinstance(date_val, (pd.Timestamp, datetime)):
        return date_val.strftime('%Y-%m-%d')
    elif isinstance(date_val, str):
        try:
            return pd.to_datetime(date_val).strftime('%Y-%m-%d')
        except:
            return date_val
    else:
        return str(date_val)


def get_signal_interpretation(rating: float) -> Dict[str, str]:
    """Get signal interpretation from rating."""
    if rating >= 75:
        return {
            'signal': 'STRONG BUY',
            'emoji': SIGNAL_EMOJI['strong_buy'],
            'color': 'bold green',
            'interpretation': 'Outstanding outlook across all factors. Strong conviction buy.',
        }
    elif rating >= 60:
        return {
            'signal': 'BUY',
            'emoji': SIGNAL_EMOJI['buy'],
            'color': 'green',
            'interpretation': 'Positive momentum and fundamentals. Good entry opportunity.',
        }
    elif rating >= 45:
        return {
            'signal': 'HOLD',
            'emoji': SIGNAL_EMOJI['hold'],
            'color': 'yellow',
            'interpretation': 'Neutral outlook. Suitable for existing positions.',
        }
    elif rating >= 35:
        return {
            'signal': 'CAUTION',
            'emoji': SIGNAL_EMOJI['caution'],
            'color': 'orange1',
            'interpretation': 'Weakening signals. Monitor closely for deterioration.',
        }
    elif rating >= 25:
        return {
            'signal': 'SELL',
            'emoji': SIGNAL_EMOJI['sell'],
            'color': 'red',
            'interpretation': 'Negative outlook. Consider reducing exposure.',
        }
    else:
        return {
            'signal': 'STRONG SELL',
            'emoji': SIGNAL_EMOJI['strong_sell'],
            'color': 'bold red',
            'interpretation': 'Severe weakness across factors. Exit recommended.',
        }


# ✅ UPDATED: Detect buy-the-dip scenarios (FALLBACK ONLY)
def detect_scenario(f_score: float, t_score: float, m_score: float) -> Dict[str, Any]:
    """
    Detect special trading scenarios (FALLBACK when Dip Quality Score not available).

    NOTE: This is now a fallback. The primary source is 'market_context' from
    compute_complete_asre() which uses Dip Quality Score.
    """
    variance = np.std([f_score, t_score, m_score])

    is_buy_dip = (f_score >= 70) and (t_score <= 20)
    is_pump_risk = (f_score <= 50) and (t_score >= 80)
    is_balanced = variance < 25

    if is_buy_dip:
        return {
            'scenario': 'BUY THE DIP',
            'emoji': '🎯',
            'color': 'bold green',
            'description': f'Strong fundamentals ({f_score:.0f}%) at oversold levels ({t_score:.0f}%)'
        }
    elif is_pump_risk:
        return {
            'scenario': 'PUMP RISK',
            'emoji': '⚠️',
            'color': 'bold red',
            'description': f'Weak fundamentals ({f_score:.0f}%) overbought ({t_score:.0f}%)'
        }
    elif is_balanced:
        return {
            'scenario': 'BALANCED',
            'emoji': '⚖️',
            'color': 'cyan',
            'description': 'All components aligned'
        }
    else:
        return {
            'scenario': 'DIVERGENT',
            'emoji': '📊',
            'color': 'yellow',
            'description': f'High component variance (σ={variance:.1f})'
        }


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from JSON file."""
    if config_path is None:
        return {}

    config_file = Path(config_path)

    if not config_file.exists():
        print_error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        print_success(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        sys.exit(1)


def create_configs_from_dict(config_dict: Dict) -> Dict:
    """Create config objects from dictionary."""
    configs = {}

    if 'momentum' in config_dict:
        configs['momentum'] = MomentumConfig(**config_dict['momentum'])

    if 'technical' in config_dict:
        configs['technical'] = TechnicalConfig(**config_dict['technical'])

    if 'fundamentals' in config_dict:
        configs['fundamentals'] = FundamentalsConfig(**config_dict['fundamentals'])

    if 'composite' in config_dict:
        configs['composite'] = CompositeConfig(**config_dict['composite'])

    if 'backtest' in config_dict:
        configs['backtest'] = BacktestConfig(**config_dict['backtest'])

    return configs


# ---------------------------------------------------------------------------
# Command: compute (✅ UPDATED TO SHOW BOTH SCORES)
# ---------------------------------------------------------------------------

def command_compute(args):
    """Compute ASRE rating for multiple stocks with dual-score display."""
    setup_logging(args.verbose)

    print_header("Multi-Stock Analysis", f"Processing {len(args.tickers)} tickers")

    config_dict = load_config(args.config)
    configs = create_configs_from_dict(config_dict)

    if args.date:
        end_date = pd.to_datetime(args.date)
        start_date = end_date - timedelta(days=365)
    else:
        start_date = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=365)
        end_date = pd.to_datetime(args.end) if args.end else datetime.now()

    all_results = []

    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing stocks...", total=len(args.tickers))

            for ticker in args.tickers:
                progress.update(task, description=f"[cyan]Processing {ticker.upper()}...")

                try:
                    df = load_stock_data(
                        ticker,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                    )

                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])

                    df = compute_complete_asre(
                        df,
                        config=configs.get('composite'),
                        fundamentals_config=configs.get('fundamentals'),
                        technical_config=configs.get('technical'),
                        momentum_config=configs.get('momentum'),
                        medallion=True,
                        return_all_components=True,
                    )

                    if 'date' in df.columns:
                        df = df.set_index('date')

                    latest = df.iloc[-1]
                    all_results.append({
                        'ticker': ticker.upper(),
                        'df': df,
                        'latest': latest
                    })

                except Exception as e:
                    print_error(f"Failed {ticker}: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

                progress.advance(task)
    else:
        for ticker in args.tickers:
            print(f"Processing {ticker.upper()}...")
            try:
                df = load_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                df = compute_complete_asre(df, config=configs.get('composite'), medallion=True)

                if 'date' in df.columns:
                    df = df.set_index('date')

                all_results.append({
                    'ticker': ticker.upper(),
                    'df': df,
                    'latest': df.iloc[-1]
                })
            except Exception as e:
                print_error(f"Failed {ticker}: {e}")

    print_success(f"Completed {len(all_results)}/{len(args.tickers)} stocks!")

    # Display results for each stock
    for result in all_results:
        ticker = result['ticker']
        df_display = result['df'].tail(args.last) if not args.date else result['df'].tail(1)
        display_ratings_dual_score(df_display, ticker, args.output_format)
        print()

    # Export if requested
    if args.output:
        combined_df = pd.concat([result['df'].assign(ticker=result['ticker']) for result in all_results])
        export_results(combined_df.reset_index(), args.output, args.output_format)
        print_success(f"Exported {len(all_results)} stocks to {args.output}")


# ✅ ENHANCED: Display with Dip Quality Score integration
def display_ratings_dual_score(df: pd.DataFrame, ticker: str, format_type: str = 'table'):
    """Display ASRE ratings showing BOTH R_Final and R_ASRE with Dip Quality Score."""

    if format_type == 'json':
        output = df[['f_score', 't_score', 'm_score', 'r_final', 'r_asre']].to_json(
            orient='records',
            date_format='iso',
            indent=2
        )
        print(output)
        return

    if format_type == 'csv':
        print(df[['f_score', 't_score', 'm_score', 'r_final', 'r_asre']].to_csv())
        return

    latest = df.iloc[-1]
    latest_r_final = latest.get('r_final', 0)
    latest_r_asre = latest.get('r_asre', 0)
    latest_f = latest.get('f_score', 0)
    latest_t = latest.get('t_score', 0)
    latest_m = latest.get('m_score', 0)

    # Get signals for both scores
    signal_final = get_signal_interpretation(latest_r_final)
    signal_asre = get_signal_interpretation(latest_r_asre)

    if HAS_RICH and console:
        # ✅ MAIN TIMELINE TABLE WITH BOTH SCORES
        table = Table(
            title=f"📊 ASRE Ratings Timeline - {ticker}",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )

        table.add_column("Date", style="cyan", no_wrap=True, justify="center")
        table.add_column("F %", style="green", justify="right", width=6)
        table.add_column("T %", style="blue", justify="right", width=6)
        table.add_column("M %", style="magenta", justify="right", width=6)
        table.add_column("R_Final", style="bold yellow", justify="right", width=8)
        table.add_column("R_ASRE", style="bold cyan", justify="right", width=8)
        table.add_column("Signal", justify="center", width=14)

        for idx, row in df.tail(10).iterrows():
            date_str = safe_date_format(idx)

            row_r_final = row.get('r_final', 0)
            row_r_asre = row.get('r_asre', 0)
            row_signal = get_signal_interpretation(row_r_asre)

            # Determine color based on ASRE score
            if row_r_asre >= 70:
                asre_color = "bold green"
            elif row_r_asre >= 50:
                asre_color = "yellow"
            else:
                asre_color = "bold red"

            table.add_row(
                date_str,
                f"[green]{row['f_score']:.0f}%[/green]",
                f"[blue]{row['t_score']:.0f}%[/blue]",
                f"[magenta]{row['m_score']:.0f}%[/magenta]",
                f"[yellow]{row_r_final:.1f}[/yellow]",
                f"[{asre_color}]{row_r_asre:.1f}[/{asre_color}]",
                f"{row_signal['emoji']} [{row_signal['color']}]{row_signal['signal']}[/{row_signal['color']}]",
            )

        console.print()
        console.print(table)
        console.print()

        # ✅ SCORE CARDS
        score_panels = [
            Panel(
                Align.center(f"[bold green]{latest_f:.0f}%[/bold green]\n[dim]({latest_f:.1f}/100.0)[/dim]\nFundamentals"),
                border_style="green",
                box=box.ROUNDED,
            ),
            Panel(
                Align.center(f"[bold blue]{latest_t:.0f}%[/bold blue]\n[dim]({latest_t:.1f}/100)[/dim]\nTechnical"),
                border_style="blue",
                box=box.ROUNDED,
            ),
            Panel(
                Align.center(f"[bold magenta]{latest_m:.0f}%[/bold magenta]\n[dim]({latest_m:.1f}/100)[/dim]\nMomentum"),
                border_style="magenta",
                box=box.ROUNDED,
            ),
        ]

        console.print(Columns(score_panels, equal=True, expand=True))
        console.print()

        # ✅ DUAL-SCORE PANEL
        latest_date = safe_date_format(latest.name)
        dual_score_panel = Panel(
            Align.center(
                f"[bold yellow]R_Final:[/bold yellow] {latest_r_final:.1f}/100\n"
                f"[dim]Composite weighted score[/dim]\n\n"
                f"[{signal_asre['color']}]R_ASRE:[/{signal_asre['color']}] {latest_r_asre:.1f}/100\n"
                f"[dim]Risk-parity Medallion score[/dim]\n\n"
                f"[{signal_asre['color']}]{signal_asre['emoji']} {signal_asre['signal']}[/{signal_asre['color']}]\n\n"
                f"[dim]{signal_asre['interpretation']}[/dim]"
            ),
            title=f"[bold cyan]{ticker}[/bold cyan] Assessment",
            subtitle=f"[dim]{latest_date}[/dim]",
            border_style=signal_asre['color'].replace("bold ", ""),
            box=box.DOUBLE,
            padding=(1, 2),
        )
        console.print(dual_score_panel)
        console.print()

        # ✅ ENHANCED MARKET CONTEXT WITH DIP QUALITY SCORE
        if 'market_context' in latest and pd.notna(latest['market_context']):
            market_context = str(latest['market_context'])

            # Color coding based on dip quality
            if 'HIGH QUALITY' in market_context or latest.get('dip_dip_quality_score', 0) >= 80:
                context_color = "bold green"
            elif 'LATE-STAGE' in market_context or 'RECOVERY' in market_context or latest.get('dip_dip_quality_score', 0) <= 30:
                context_color = "bold red"
            elif 'GOOD' in market_context or latest.get('dip_dip_quality_score', 0) >= 60:
                context_color = "bold yellow"
            elif 'PUMP RISK' in market_context:
                context_color = "bold red"
            else:
                context_color = "cyan"

            market_context_display = market_context
        else:
            # Fallback to old scenario detection
            scenario = detect_scenario(latest_f, latest_t, latest_m)
            market_context_display = f"{scenario['emoji']} [{scenario['color']}]{scenario['scenario']}[/{scenario['color']}]\n{scenario['description']}"
            context_color = scenario['color'].replace("bold ", "")

        scenario_panel = Panel(
            market_context_display,
            title="📌 Market Context",
            border_style=context_color,
            box=box.ROUNDED,
        )
        console.print(scenario_panel)
        console.print()

        # ✅ DIP QUALITY DETAILED METRICS (if available)
        if 'dip_dip_quality_score' in latest and pd.notna(latest['dip_dip_quality_score']):
            dip_quality = latest['dip_dip_quality_score']
            dip_stage = latest.get('dip_dip_stage', 'N/A')
            entry_timing = latest.get('dip_entry_timing_score', 0)
            expected_upside = latest.get('dip_expected_upside', 0)
            risk_reward = latest.get('dip_risk_reward_ratio', 0)
            confidence = latest.get('dip_confidence', 0)

            # Color code dip quality
            if dip_quality >= 80:
                dip_color = "bold green"
            elif dip_quality >= 60:
                dip_color = "bold yellow"
            elif dip_quality >= 40:
                dip_color = "yellow"
            else:
                dip_color = "bold red"

            dip_metrics_panel = Panel(
                f"[{dip_color}]Overall Score:[/{dip_color}] [{dip_color}]{dip_quality:.0f}/100[/{dip_color}]\n"
                f"Stage: {dip_stage} | Entry Timing: {entry_timing:.0f}/100\n"
                f"Expected Upside: {expected_upside:.1f}% | R/R: {risk_reward:.2f}\n"
                f"Confidence: {confidence:.0f}%",
                title="🎯 Dip Quality Metrics",
                border_style=dip_color.replace("bold ", ""),
                box=box.ROUNDED,
            )
            console.print(dip_metrics_panel)
            console.print()

        # Confidence interval
        if 'confidence_lower' in latest and 'confidence_upper' in latest:
            ci_lower = latest.get('confidence_lower', latest_r_asre)
            ci_upper = latest.get('confidence_upper', latest_r_asre)
            ci_width = ci_upper - ci_lower
            confidence_pct = max(0, 100 - (ci_width / max(latest_r_asre, 1) * 100))

            confidence_panel = Panel(
                f"Range: [{ci_lower:.1f}, {ci_upper:.1f}]\n"
                f"Confidence: {confidence_pct:.0f}%",
                title="📈 Prediction Confidence",
                border_style="cyan",
                box=box.ROUNDED,
            )
            console.print(confidence_panel)
            console.print()

    else:
        # Plain text fallback
        print(f"\n{'='*80}")
        print(f"ASRE Ratings - {ticker}")
        print(f"{'='*80}\n")

        display_df = df[['f_score', 't_score', 'm_score', 'r_final', 'r_asre']].tail()
        print(display_df.round(1).to_string())
        print(f"\n{'='*80}\n")

        print(f"R_Final:  {latest_r_final:.1f}/100")
        print(f"R_ASRE:   {latest_r_asre:.1f}/100 - {signal_asre['signal']}")
        print(f"F: {latest_f:.0f}% | T: {latest_t:.0f}% | M: {latest_m:.0f}%")

        # Show market context or fallback scenario
        if 'market_context' in latest and pd.notna(latest['market_context']):
            print(f"\nMarket Context:\n{latest['market_context']}")
        else:
            scenario = detect_scenario(latest_f, latest_t, latest_m)
            print(f"{scenario['scenario']}: {scenario['description']}")

        print(f"\n{signal_asre['interpretation']}\n")


# ---------------------------------------------------------------------------
# Command: backtest
# ---------------------------------------------------------------------------

def command_backtest(args):
    """Run strategy backtest."""
    setup_logging(args.verbose)

    print_header(
        "Strategy Backtest",
        f"Testing ASRE strategy on {args.ticker.upper()}"
    )

    config_dict = load_config(args.config)
    configs = create_configs_from_dict(config_dict)

    start_date = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=365*2)
    end_date = pd.to_datetime(args.end) if args.end else datetime.now()

    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading historical data...", total=100)

            try:
                df = load_stock_data(
                    args.ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                )
                progress.update(task, completed=30)

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                progress.update(task, description="[cyan]Computing ASRE ratings...")
                df = compute_complete_asre(
                    df,
                    config=configs.get('composite'),
                    fundamentals_config=configs.get('fundamentals'),
                    technical_config=configs.get('technical'),
                    momentum_config=configs.get('momentum'),
                    medallion=True
                )
                progress.update(task, completed=70)

                progress.update(task, description="[cyan]Running backtest simulation...")
                bt_config = configs.get('backtest', BacktestConfig(
                    threshold_long=args.threshold_long,
                    threshold_short=args.threshold_short,
                ))

                bt = Backtester(df, rating_col='r_asre', config=bt_config)
                bt.run(
                    signal_type=args.signal_type,
                    transaction_cost=args.transaction_cost,
                    slippage=args.slippage,
                    max_position=args.max_position,
                )
                progress.update(task, completed=100)

            except Exception as e:
                print_error(f"Backtest failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
    else:
        print("Loading data...")
        df = load_stock_data(args.ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        print("Computing ratings...")
        df = compute_complete_asre(df, medallion=True)
        print("Running backtest...")
        bt = Backtester(df, rating_col='r_asre')
        bt.run(signal_type=args.signal_type)

    print_success("Backtest complete!")

    report = bt.get_report()
    display_backtest_elegant(report, args.ticker.upper())

    if args.output:
        export_backtest(bt.results_df, report, args.output, args.output_format)
        print_success(f"Exported to {args.output}")


def display_backtest_elegant(report: Dict, ticker: str):
    """Display backtest results with elegant formatting."""

    if HAS_RICH and console:
        table = Table(
            title="📈 Performance Metrics",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("", style="dim")

        table.add_row("Total Return", f"{report.get('total_return', 0):.2%}", "📊")
        table.add_row("CAGR", f"[bold]{report.get('cagr', 0):.2%}[/bold]", "📈")
        table.add_row("Volatility", f"{report.get('volatility', 0):.2%}", "📉")
        table.add_row("", "", "")

        sharpe = report.get('sharpe_ratio', 0)
        sharpe_color = "green" if sharpe >= 1.0 else "yellow"
        table.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe:.3f}[/{sharpe_color}]", "⚡")
        table.add_row("Sortino Ratio", f"{report.get('sortino_ratio', 0):.3f}", "✨")
        table.add_row("Calmar Ratio", f"{report.get('calmar_ratio', 0):.3f}", "🎯")
        table.add_row("", "", "")

        table.add_row("Max Drawdown", f"[red]{report.get('max_drawdown', 0):.2%}[/red]", "📉")
        table.add_row("VaR (95%)", f"{report.get('var_95', 0):.2%}", "⚠️")
        table.add_row("", "", "")

        table.add_row("Win Rate", f"{report.get('win_rate', 0):.2%}", "🎲")
        table.add_row("Profit Factor", f"{report.get('profit_factor', 0):.3f}", "💰")
        table.add_row("Trades", f"{report.get('num_trades', 0):.0f}", "🔄")

        console.print()
        console.print(table)
        console.print()

        if sharpe >= 2.0:
            rating = "EXCELLENT 🌟"
            rating_color = "bold green"
            interpretation = "Outstanding risk-adjusted returns. Institutional-grade performance."
        elif sharpe >= 1.0:
            rating = "GOOD ✅"
            rating_color = "green"
            interpretation = "Solid risk-adjusted returns. Competitive performance."
        elif sharpe >= 0.5:
            rating = "FAIR ⚖️"
            rating_color = "yellow"
            interpretation = "Moderate risk-adjusted returns. Room for improvement."
        else:
            rating = "POOR ❌"
            rating_color = "red"
            interpretation = "Weak risk-adjusted returns. Strategy needs refinement."

        assessment = Panel(
            Align.center(
                f"[{rating_color}]{rating}[/{rating_color}]\n\n"
                f"[dim]{interpretation}[/dim]\n\n"
                f"Sharpe: [{rating_color}]{sharpe:.3f}[/{rating_color}] | "
                f"Max DD: [red]{report.get('max_drawdown', 0):.1%}[/red] | "
                f"Win Rate: {report.get('win_rate', 0):.0%}"
            ),
            title=f"[bold cyan]{ticker}[/bold cyan] Strategy Assessment",
            border_style=rating_color.replace("bold ", ""),
            box=box.DOUBLE,
            padding=(1, 2),
        )

        console.print(assessment)
        console.print()

    else:
        print(f"\n{'='*80}")
        print(f"Backtest Results - {ticker}")
        print(f"{'='*80}\n")
        print(f"Total Return:  {report.get('total_return', 0):.2%}")
        print(f"CAGR:          {report.get('cagr', 0):.2%}")
        print(f"Sharpe Ratio:  {report.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown:  {report.get('max_drawdown', 0):.2%}")
        print(f"Win Rate:      {report.get('win_rate', 0):.2%}")
        print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# Command: compare (✅ UPDATED WITH DIP QUALITY INDICATORS)
# ---------------------------------------------------------------------------

def command_compare(args):
    """Compare multiple stocks with dual-score display."""
    setup_logging(args.verbose)

    print_header(
        "Stock Comparison",
        f"Comparing {len(args.tickers)} stocks"
    )

    config_dict = load_config(args.config)
    configs = create_configs_from_dict(config_dict)

    end_date = pd.to_datetime(args.date) if args.date else datetime.now()
    start_date = end_date - timedelta(days=365)

    results = []

    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing stocks...",
                total=len(args.tickers)
            )

            for ticker in args.tickers:
                progress.update(task, description=f"[cyan]Processing {ticker.upper()}...")

                try:
                    df = load_stock_data(
                        ticker,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                    )

                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')

                    df = compute_complete_asre(
                        df,
                        config=configs.get('composite'),
                        fundamentals_config=configs.get('fundamentals'),
                        technical_config=configs.get('technical'),
                        momentum_config=configs.get('momentum'),
                        medallion=True,
                        return_all_components=True
                    )

                    latest = df.iloc[-1]

                    results.append({
                        'ticker': ticker.upper(),
                        'date': latest.name,
                        'f_score': latest.get('f_score', 0),
                        't_score': latest.get('t_score', 0),
                        'm_score': latest.get('m_score', 0),
                        'r_final': latest.get('r_final', 0),
                        'r_asre': latest.get('r_asre', 0),
                        'dip_quality': latest.get('dip_dip_quality_score', None),
                        'dip_stage': latest.get('dip_dip_stage', None),
                        'is_buy_dip': latest.get('is_buy_dip', False),
                    })

                except Exception as e:
                    logger.error(f"Failed to process {ticker}: {e}")

                progress.advance(task)
    else:
        for ticker in args.tickers:
            print(f"Processing {ticker}...")
            try:
                df = load_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                df = compute_complete_asre(df, medallion=True, return_all_components=True)
                latest = df.iloc[-1]
                results.append({
                    'ticker': ticker.upper(),
                    'r_final': latest.get('r_final', 0),
                    'r_asre': latest.get('r_asre', 0),
                    'dip_quality': latest.get('dip_dip_quality_score', None),
                    'dip_stage': latest.get('dip_dip_stage', None),
                })
            except Exception as e:
                logger.error(f"Failed: {e}")

    if len(results) == 0:
        print_error("No results to display")
        return

    print_success(f"Analyzed {len(results)} stocks")

    # Sort by R_ASRE (Medallion)
    results_df = pd.DataFrame(results).sort_values('r_asre', ascending=False)

    display_comparison_dual_score(results_df)

    if args.output:
        results_df.to_csv(args.output, index=False)
        print_success(f"Exported to {args.output}")


# ✅ ENHANCED: Comparison table with dip quality indicators
def display_comparison_dual_score(results_df: pd.DataFrame):
    """Display comparison with both R_Final and R_ASRE scores + Dip Quality."""

    if HAS_RICH and console:
        table = Table(
            title="🏆 Stock Rankings",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Rank", justify="center", style="bold")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("F", style="green", justify="right")
        table.add_column("T", style="blue", justify="right")
        table.add_column("M", style="magenta", justify="right")
        table.add_column("R_Final", style="yellow", justify="right")
        table.add_column("R_ASRE", style="bold cyan", justify="right")
        table.add_column("Signal", justify="center")
        table.add_column("Context", style="dim")

        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            r_final = row.get('r_final', 0)
            r_asre = row.get('r_asre', 0)
            f_score = row.get('f_score', 0)
            t_score = row.get('t_score', 0)
            m_score = row.get('m_score', 0)
            dip_quality = row.get('dip_quality', None)
            dip_stage = row.get('dip_stage', None)

            signal_info = get_signal_interpretation(r_asre)

            # Medal for top 3
            medal = ""
            rank_style = "white"
            if rank == 1:
                medal = "🥇"
                rank_style = "bold yellow"
            elif rank == 2:
                medal = "🥈"
                rank_style = "bold white"
            elif rank == 3:
                medal = "🥉"
                rank_style = "bold orange1"

            # Dip quality indicator
            context_indicator = ""
            if pd.notna(dip_quality) and dip_quality is not None:
                if dip_stage in ["EARLY", "MID"] and dip_quality >= 60:
                    context_indicator = "🎯 DIP"
                elif dip_stage == "LATE":
                    context_indicator = "⚠️ LATE"
                elif dip_stage == "RECOVERY":
                    context_indicator = "❌ RECOV"
            elif row.get('is_buy_dip', False):
                context_indicator = "🎯 DIP?"

            table.add_row(
                f"[{rank_style}]{rank}[/{rank_style}]",
                row['ticker'],
                f"{f_score:.0f}",
                f"{t_score:.0f}",
                f"{m_score:.0f}",
                f"[yellow]{r_final:.1f}[/yellow]",
                f"[{signal_info['color']}]{r_asre:.1f}[/{signal_info['color']}]",
                f"{signal_info['emoji']} [{signal_info['color']}]{signal_info['signal']}[/{signal_info['color']}]",
                f"{medal} {context_indicator}",
            )

        console.print()
        console.print(table)
        console.print()

    else:
        print("\n" + results_df.to_string(index=False) + "\n")


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def export_results(df: pd.DataFrame, output_path: str, format_type: str = 'csv'):
    """Export results to file."""
    output_file = Path(output_path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)

    if format_type == 'csv':
        df.to_csv(output_file, index=False)
    elif format_type == 'json':
        df.to_json(output_file, orient='records', date_format='iso', indent=2)
    else:
        df.to_csv(output_file, index=False)


def export_backtest(results_df: pd.DataFrame, report: Dict, output_path: str, format_type: str = 'csv'):
    """Export backtest results."""
    output_file = Path(output_path)

    if format_type == 'json':
        output = {
            'report': report,
            'equity_curve': results_df['cumulative_return'].to_dict(),
            'returns': results_df['net_return'].to_dict(),
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
    else:
        results_df.to_csv(output_file)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def create_parser():
    """Create argument parser with elegant help."""
    parser = argparse.ArgumentParser(
        prog='asre',
        description='🚀 ASRE - Advanced Stock Rating Engine (v2.1 - Dip Quality Integrated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick rating check
  asre compute AAPL

  # Historical analysis
  asre compute AAPL --start 2024-01-01 --end 2024-12-31

  # Backtest strategy
  asre backtest AAPL --start 2020-01-01

  # Compare stocks
  asre compare AAPL MSFT GOOGL TSLA

For detailed help on any command:
  asre <command> --help
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config', type=str, help='Configuration file (JSON)')
    parser.add_argument('--version', action='version', version='ASRE 2.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Compute command
    compute_parser = subparsers.add_parser('compute', help='Compute ASRE rating')
    compute_parser.add_argument('tickers', nargs='+', type=str, help='Stock tickers (e.g., AAPL MSFT SPY)')
    compute_parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD)')
    compute_parser.add_argument('--start', type=str, help='Start date')
    compute_parser.add_argument('--end', type=str, help='End date')
    compute_parser.add_argument('--last', type=int, default=10, help='Show last N days (default: 10)')
    compute_parser.add_argument('-o', '--output', type=str, help='Export to file')
    compute_parser.add_argument('--output-format', choices=['table', 'csv', 'json'], default='table')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtest')
    backtest_parser.add_argument('ticker', type=str, help='Stock ticker')
    backtest_parser.add_argument('--start', type=str, help='Start date')
    backtest_parser.add_argument('--end', type=str, help='End date')
    backtest_parser.add_argument('--signal-type', choices=['threshold', 'quantile', 'regime'], default='threshold')
    backtest_parser.add_argument('--threshold-long', type=float, default=70.0, help='Long threshold (default: 70)')
    backtest_parser.add_argument('--threshold-short', type=float, default=30.0, help='Short threshold (default: 30)')
    backtest_parser.add_argument('--transaction-cost', type=float, default=0.001)
    backtest_parser.add_argument('--slippage', type=float, default=0.0005)
    backtest_parser.add_argument('--max-position', type=float, default=1.0)
    backtest_parser.add_argument('-o', '--output', type=str, help='Export to file')
    backtest_parser.add_argument('--output-format', choices=['csv', 'json'], default='csv')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple stocks')
    compare_parser.add_argument('tickers', nargs='+', help='Stock tickers (space-separated)')
    compare_parser.add_argument('--date', type=str, help='Comparison date')
    compare_parser.add_argument('-o', '--output', type=str, help='Export to file')

    return parser


def main():
    """Main entry point with elegant error handling."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        if HAS_RICH and console:
            print_header("Advanced Stock Rating Engine", "v2.1.0 - Dip Quality Integrated")
            console.print("[dim]Use --help to see available commands[/dim]\n")
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'compute':
            command_compute(args)
        elif args.command == 'backtest':
            command_backtest(args)
        elif args.command == 'compare':
            command_compare(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print_info("\nOperation cancelled by user")
        sys.exit(0)

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


__all__ = ['main', 'create_parser']
