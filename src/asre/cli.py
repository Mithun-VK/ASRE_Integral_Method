"""
ASRE Command-Line Interface

Production-grade CLI for ASRE rating system with elegant UX.

Design Principles:
- Progressive disclosure: Show essential info first, details on demand
- Clear visual hierarchy with colors and spacing
- Consistent interaction patterns across commands
- Helpful defaults and smart prompts
- Real-time progress feedback
- Beautiful, scannable output

Commands:
1. compute: Compute ASRE rating for a stock
2. backtest: Run strategy backtest
3. optimize: Optimize parameters
4. compare: Compare multiple stocks
5. export: Export ratings to CSV/JSON
6. watch: Real-time monitoring (interactive)

Features:
- Rich formatted output with tables and colors
- Progress tracking for long operations
- Configuration file support
- Interactive mode for exploratory analysis
- Keyboard shortcuts for power users
- Smart error handling with recovery suggestions
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
    elif rating >= 65:
        return {
            'signal': 'BUY',
            'emoji': SIGNAL_EMOJI['buy'],
            'color': 'green',
            'interpretation': 'Positive momentum and fundamentals. Good entry opportunity.',
        }
    elif rating >= 55:
        return {
            'signal': 'HOLD',
            'emoji': SIGNAL_EMOJI['hold'],
            'color': 'yellow',
            'interpretation': 'Neutral outlook. Suitable for existing positions.',
        }
    elif rating >= 45:
        return {
            'signal': 'CAUTION',
            'emoji': SIGNAL_EMOJI['caution'],
            'color': 'orange1',
            'interpretation': 'Weakening signals. Monitor closely for deterioration.',
        }
    elif rating >= 35:
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
# Command: compute
# ---------------------------------------------------------------------------

def command_compute(args):
    """Compute ASRE rating for multiple stocks with elegant UX."""
    setup_logging(args.verbose)
    
    print_header("Multi-Stock Analysis", f"Processing {len(args.tickers)} tickers")
    
    # Load configuration ONCE (optimization)
    config_dict = load_config(args.config)
    configs = create_configs_from_dict(config_dict)
    
    # Determine date range ONCE
    if args.date:
        end_date = pd.to_datetime(args.date)
        start_date = end_date - timedelta(days=365)
    else:
        start_date = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=365)
        end_date = pd.to_datetime(args.end) if args.end else datetime.now()
    
    all_results = []
    
    # 🟢 PROCESS EACH TICKER
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
                    # Load data
                    df = load_stock_data(
                        ticker,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                    )
                    
                    # Ensure date column is datetime
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # Compute ASRE
                    df = compute_complete_asre(
                        df,
                        config=configs.get('composite'),
                        fundamentals_config=configs.get('fundamentals'),
                        technical_config=configs.get('technical'),
                        momentum_config=configs.get('momentum'),
                        medallion=True,
                        return_all_components=True,
                    )
                    
                    # Store results
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
        # Plain text fallback
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
    
    # 🟢 DISPLAY RESULTS FOR EACH STOCK
    for result in all_results:
        ticker = result['ticker']
        df_display = result['df'].tail(args.last) if not args.date else result['df'].tail(1)
        display_ratings_elegant(df_display, ticker, args.output_format)
        print()  # Spacer
    
    # 🟢 SUMMARY TABLE
    if len(all_results) > 1 and HAS_RICH and console:
        summary_table = Table(title="📊 Portfolio Summary", box=box.ROUNDED)
        summary_table.add_column("Ticker", style="cyan")
        summary_table.add_column("ASRE", style="bold yellow", justify="right")
        summary_table.add_column("F-Score", style="green", justify="right")
        summary_table.add_column("Signal", justify="center")
        
        for result in all_results:
            r_asre = result['latest'].get('r_asre', 0)
            signal_info = get_signal_interpretation(r_asre)
            summary_table.add_row(
                result['ticker'],
                f"{r_asre:.1f}",
                f"{result['latest'].get('f_score', 0):.1f}",
                f"{signal_info['emoji']} {signal_info['signal']}"
            )
        
        console.print(summary_table)
        console.print()
    
    # 🟢 EXPORT ALL RESULTS
    if args.output:
        combined_df = pd.concat([result['df'].assign(ticker=result['ticker']) for result in all_results])
        export_results(combined_df.reset_index(), args.output, args.output_format)
        print_success(f"Exported {len(all_results)} stocks to {args.output}")

def display_ratings_elegant(df: pd.DataFrame, ticker: str, format_type: str = 'table'):
    """Display ASRE ratings with elegant formatting + PROPER SCALING."""
    
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
    
    # 🟢 SCALE SCORES FOR DISPLAY (0-100)
    df_display = df.copy()
    # Fundamentals: still normalize (raw max = 14)
    df_display['f_score_scaled'] = np.clip((df['f_score'] / 14.0) * 100, 0, 100)

    # Technical: ALREADY normalized in engine → DO NOT rescale again
    df_display['t_score_scaled'] = np.clip(df['t_score'], 0, 100)

    # Momentum: already 0–100
    df_display['m_score_scaled'] = np.clip(df['m_score'], 0, 100)
    
    latest = df_display.iloc[-1]
    r_asre = latest.get('r_asre', 0)
    signal_info = get_signal_interpretation(r_asre)
    
    # Color-coded rating
    if r_asre >= 70:
        rating_style = "bold green"
    elif r_asre >= 50:
        rating_style = "yellow"
    else:
        rating_style = "bold red"
    
    if HAS_RICH and console:
        # 🟢 MAIN TIMELINE TABLE (showing scaled scores)
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
        table.add_column("ASRE", style="bold yellow", justify="right", width=8)
        table.add_column("Signal", justify="center", width=12)
        
        for idx, row in df_display.tail(10).iterrows():  # Last 10 days
            date_str = safe_date_format(idx)
            table.add_row(
                date_str,
                f"[green]{row['f_score_scaled']:.0f}%[/green]",
                f"[blue]{row['t_score_scaled']:.0f}%[/blue]",
                f"[magenta]{row['m_score_scaled']:.0f}%[/magenta]",
                f"[{rating_style}]{r_asre:.1f}[/{rating_style}]",
                f"{signal_info['emoji']} [{signal_info['color']}]{signal_info['signal']}[/{signal_info['color']}]",
            )
        
        console.print()
        console.print(table)
        console.print()
        
        # 🟢 SCORE CARDS (SCALED!)
        score_panels = [
            Panel(
                Align.center(f"[bold green]{latest['f_score_scaled']:.0f}%[/bold green]\n[dim]({latest['f_score']:.1f}/14.0)[/dim]\nFundamentals"),
                border_style="green",
                box=box.ROUNDED,
            ),
            Panel(
                Align.center(f"[bold blue]{latest['t_score_scaled']:.0f}%[/bold blue]\n[dim]({latest['t_score']:.1f}/100)[/dim]\nTechnical"),
                border_style="blue",
                box=box.ROUNDED,
            ),
            Panel(
                Align.center(f"[bold magenta]{latest['m_score_scaled']:.0f}%[/bold magenta]\n[dim]({latest['m_score']:.1f}/100)[/dim]\nMomentum"),
                border_style="magenta",
                box=box.ROUNDED,
            ),
        ]
        
        console.print(Columns(score_panels, equal=True, expand=True))
        console.print()
        
        # 🟢 MAIN SIGNAL PANEL
        latest_date = safe_date_format(latest.name)
        main_panel = Panel(
            Align.center(
                f"{signal_info['emoji']} [bold]{r_asre:.1f}/100[/bold]\n\n"
                f"[{signal_info['color']}]{signal_info['signal']}[/{signal_info['color']}]\n\n"
                f"[dim]{signal_info['interpretation']}[/dim]"
            ),
            title=f"[bold cyan]{ticker}[/bold cyan] Assessment",
            subtitle=f"[dim]{latest_date}[/dim]",
            border_style=signal_info['color'],
            box=box.DOUBLE,
            padding=(1, 2),
        )
        console.print(main_panel)
        console.print()
        
        # 🟢 CONFIDENCE (if available)
        if 'confidence_lower' in latest and 'confidence_upper' in latest:
            ci_lower = latest.get('confidence_lower', r_asre)
            ci_upper = latest.get('confidence_upper', r_asre)
            ci_width = ci_upper - ci_lower
            confidence_pct = max(0, 100 - (ci_width / max(r_asre, 1) * 100))
            
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
        # Plain text fallback (SCALED!)
        print(f"\n{'='*80}")
        print(f"ASRE Ratings - {ticker} (Scaled 0-100%)")
        print(f"{'='*80}\n")
        
        display_df = df_display[['f_score_scaled', 't_score_scaled', 'm_score_scaled', 'r_asre']].tail()
        display_df.columns = ['F %', 'T %', 'M %', 'ASRE']
        print(display_df.round(1).to_string())
        print(f"\n{'='*80}\n")
        
        print(f"Latest: {r_asre:.1f}/100 - {signal_info['signal']}")
        print(f"F: {latest['f_score_scaled']:.0f}% | T: {latest['t_score_scaled']:.0f}% | M: {latest['m_score_scaled']:.0f}%")
        print(f"{signal_info['interpretation']}\n")

# ---------------------------------------------------------------------------
# Command: backtest
# ---------------------------------------------------------------------------

def command_backtest(args):
    """Run strategy backtest with elegant progress and output."""
    setup_logging(args.verbose)
    
    print_header(
        "Strategy Backtest",
        f"Testing ASRE strategy on {args.ticker.upper()}"
    )
    
    # Load configuration
    config_dict = load_config(args.config)
    configs = create_configs_from_dict(config_dict)
    
    # Date range
    start_date = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=365*2)
    end_date = pd.to_datetime(args.end) if args.end else datetime.now()
    
    # Load and process with progress
    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Load data
            task = progress.add_task("[cyan]Loading historical data...", total=100)
            
            try:
                df = load_stock_data(
                    args.ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                )
                progress.update(task, completed=30)
                
                # FIX: Ensure date is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Compute ASRE
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
                
                # Run backtest
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
    
    # Generate report
    report = bt.get_report()
    
    # Display results
    display_backtest_elegant(report, args.ticker.upper())
    
    # Export if requested
    if args.output:
        export_backtest(bt.results_df, report, args.output, args.output_format)
        print_success(f"Exported to {args.output}")

def display_backtest_elegant(report: Dict, ticker: str):
    """Display backtest results with elegant formatting."""
    
    if HAS_RICH and console:
        # Performance metrics table
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
        
        # Returns
        table.add_row("Total Return", f"{report.get('total_return', 0):.2%}", "📊")
        table.add_row("CAGR", f"[bold]{report.get('cagr', 0):.2%}[/bold]", "📈")
        table.add_row("Volatility", f"{report.get('volatility', 0):.2%}", "📉")
        table.add_row("", "", "")
        
        # Risk-adjusted
        sharpe = report.get('sharpe_ratio', 0)
        sharpe_color = "green" if sharpe >= 1.0 else "yellow"
        table.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe:.3f}[/{sharpe_color}]", "⚡")
        table.add_row("Sortino Ratio", f"{report.get('sortino_ratio', 0):.3f}", "✨")
        table.add_row("Calmar Ratio", f"{report.get('calmar_ratio', 0):.3f}", "🎯")
        table.add_row("", "", "")
        
        # Risk
        table.add_row("Max Drawdown", f"[red]{report.get('max_drawdown', 0):.2%}[/red]", "📉")
        table.add_row("VaR (95%)", f"{report.get('var_95', 0):.2%}", "⚠️")
        table.add_row("", "", "")
        
        # Win/Loss
        table.add_row("Win Rate", f"{report.get('win_rate', 0):.2%}", "🎲")
        table.add_row("Profit Factor", f"{report.get('profit_factor', 0):.3f}", "💰")
        table.add_row("Trades", f"{report.get('num_trades', 0):.0f}", "🔄")
        
        console.print()
        console.print(table)
        console.print()
        
        # Strategy rating
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
        # Plain text
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
# Command: compare
# ---------------------------------------------------------------------------

def command_compare(args):
    """Compare multiple stocks with elegant visualization."""
    setup_logging(args.verbose)
    
    print_header(
        "Stock Comparison",
        f"Comparing {len(args.tickers)} stocks"
    )
    
    # Load config
    config_dict = load_config(args.config)
    configs = create_configs_from_dict(config_dict)
    
    # Date
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
                        medallion=True
                    )
                    
                    latest = df.iloc[-1]
                    
                    results.append({
                        'ticker': ticker.upper(),
                        'date': latest.name,
                        'f_score': latest.get('f_score', 0),
                        't_score': latest.get('t_score', 0),
                        'm_score': latest.get('m_score', 0),
                        'r_asre': latest.get('r_asre', 0),
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
                df = compute_complete_asre(df, medallion=True)
                latest = df.iloc[-1]
                results.append({
                    'ticker': ticker.upper(),
                    'r_asre': latest.get('r_asre', 0),
                })
            except Exception as e:
                logger.error(f"Failed: {e}")
    
    if len(results) == 0:
        print_error("No results to display")
        return
    
    print_success(f"Analyzed {len(results)} stocks")
    
    results_df = pd.DataFrame(results).sort_values('r_asre', ascending=False)
    
    display_comparison_elegant(results_df)
    
    if args.output:
        results_df.to_csv(args.output, index=False)
        print_success(f"Exported to {args.output}")

def display_comparison_elegant(results_df: pd.DataFrame):
    """Display comparison with elegant ranking."""
    
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
        table.add_column("ASRE", style="bold yellow", justify="right")
        table.add_column("Signal", justify="center")
        table.add_column("", style="dim")
        
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            r_asre = row.get('r_asre', 0)
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
            
            table.add_row(
                f"[{rank_style}]{rank}[/{rank_style}]",
                row['ticker'],
                f"{row.get('f_score', 0):.0f}",
                f"{row.get('t_score', 0):.0f}",
                f"{row.get('m_score', 0):.0f}",
                f"[{signal_info['color']}]{r_asre:.1f}[/{signal_info['color']}]",
                f"{signal_info['emoji']} [{signal_info['color']}]{signal_info['signal']}[/{signal_info['color']}]",
                medal,
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
    
    # Reset index if date is index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # Convert date to string for export
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
        description='🚀 ASRE - Advanced Stock Rating Engine',
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
    parser.add_argument('--version', action='version', version='ASRE 1.0.0')
    
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
            print_header("Advanced Stock Rating Engine", "v1.0.0")
            console.print("[dim]Use --help to see available commands[/dim]\n")
        parser.print_help()
        sys.exit(0)
    
    try:
        # Route to command handler
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

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = ['main', 'create_parser']
