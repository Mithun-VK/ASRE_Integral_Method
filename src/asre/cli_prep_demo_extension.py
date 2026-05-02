"""
CLI extension: prep-demo command
Add this code to your existing cli.py file.

Location: Merge into asre/cli.py after the existing 'compute' command.
"""

import shutil
from pathlib import Path

import click
from rich import _console

from asre import cli


@cli.command(name="prep-demo")
@click.option("--prospect", required=True, help="Prospect name (e.g., 'Rajesh Kumar')")
@click.option("--tickers", required=True, multiple=True, help="Tickers to pre-scan (e.g., INFY.NS TCS.NS)")
@click.option("--output-dir", default="demos", help="Base directory for demo folders")
def prep_demo(prospect: str, tickers: tuple, output_dir: str):
    """
    Pre-load stock data and generate reports for prospect demos.

    Creates a folder structure:
        demos/
            Rajesh_Kumar_20260226/
                cache/          (pre-fetched fundamentals)
                reports/        (PDF reports)
                prescan_output.txt
                metadata.json

    Example:
        asre prep-demo --prospect "Rajesh Kumar" --tickers INFY.NS TCS.NS HDFCBANK.NS
    """
    from datetime import datetime
    import json
    from asre.data.fundamental_fetcher import FundamentalFetcher
    from asre.data_loader_indian import load_stock_data
    from asre.composite import compute_asre_rating
    from asre.report_generator import export_stock_report

    _console.print()
    console.print(panel.fit(
        f"[bold cyan]ASRE Demo Preparation[/bold cyan]\n"
        f"Prospect: {prospect}\n"
        f"Tickers: {', '.join(tickers)}",
        border_style="cyan"
    ))
    console.print()

    # Create folder structure
    safe_name = prospect.replace(" ", "_").replace("/", "_")
    date_str = datetime.now().strftime("%Y%m%d")
    demo_dir = Path(output_dir) / f"{safe_name}_{date_str}"

    cache_dir = demo_dir / "cache"
    reports_dir = demo_dir / "reports"

    demo_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    logger.info(f"Created demo directory: {demo_dir}")

    # Initialize components
    fetcher = FundamentalFetcher(cache_dir=str(cache_dir))

    results = []
    scan_output = []

    scan_output.append("=" * 70)
    scan_output.append(f"ASRE Pre-Demo Scan — {prospect}")
    scan_output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    scan_output.append("=" * 70)
    scan_output.append("")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            f"Pre-scanning {len(tickers)} stocks...",
            total=len(tickers)
        )

        for ticker in tickers:
            progress.update(task, description=f"Processing {ticker}...")

            try:
                # Fetch fundamentals
                fundamentals, fetch_ts = fetcher.fetch_quarterly_fundamentals(ticker)

                if fundamentals is None or len(fundamentals) < 4:
                    raise ValueError(f"Insufficient data ({len(fundamentals) if fundamentals else 0} quarters)")

                # Load data
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

                stock_df = load_stock_data(
                    ticker=ticker,
                    start=start_date,
                    end=end_date,
                    quarterly_fundamentals=fundamentals,
                    fundamentals_fetch_ts=fetch_ts
                )

                # Compute rating
                rating_df = compute_asre_rating(stock_df)
                latest = rating_df.iloc[-1]

                # Generate PDF
                pdf_path = export_stock_report(
                    ticker=ticker,
                    results_df=rating_df,
                    fundamentals={
                        'tier': 'C',
                        'category': 'STABLE',
                        'pe': latest.get('pe', 0),
                        'roe': latest.get('roe', 0),
                        'de': latest.get('de', 0),
                    },
                    output_dir=str(reports_dir)
                )

                # Store result
                result = {
                    'ticker': ticker,
                    'success': True,
                    'r_asre': float(latest['r_asre']),
                    'signal': latest['signal'],
                    'quarters': len(fundamentals),
                    'pdf': str(pdf_path.name),
                }
                results.append(result)

                # Add to scan output
                scan_output.append(f"✓ {ticker}")
                scan_output.append(f"  R_ASRE: {latest['r_asre']:.1f}/100")
                scan_output.append(f"  Signal: {latest['signal']}")
                scan_output.append(f"  Quarters: {len(fundamentals)}")
                scan_output.append(f"  PDF: {pdf_path.name}")
                scan_output.append("")

            except Exception as exc:
                logger.error(f"Failed to process {ticker}: {exc}")
                results.append({
                    'ticker': ticker,
                    'success': False,
                    'error': str(exc),
                })
                scan_output.append(f"✗ {ticker}")
                scan_output.append(f"  Error: {exc}")
                scan_output.append("")

            progress.advance(task)

    # Save prescan output
    prescan_path = demo_dir / "prescan_output.txt"
    with open(prescan_path, "w") as f:
        f.write("\n".join(scan_output))

    # Save metadata
    metadata = {
        'prospect': prospect,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        'tickers': list(tickers),
        'results': results,
        'asre_version': '2.0.1',
    }

    metadata_path = demo_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Demo preparation complete![/bold green]\n\n"
        f"Location: {demo_dir}\n"
        f"Success: {sum(r['success'] for r in results)}/{len(results)} stocks\n"
        f"Reports: {len(list(reports_dir.glob('*.pdf')))} PDFs generated\n"
        f"Cache: {len(list(cache_dir.glob('*.json')))} fundamentals cached\n\n"
        f"[cyan]Ready for prospect demo — share {demo_dir} folder[/cyan]",
        border_style="green"
    ))
    console.print()


# Add this after your existing 'export' command (if any)
@cli.command(name="test-demo")
@click.option("--output-dir", default="test_reports", help="Output directory for test reports")
def test_demo(output_dir: str):
    """
    Run pre-demo test suite to validate system readiness.

    Tests 10 NSE stocks and generates performance report.

    Example:
        asre test-demo
        asre test-demo --output-dir demo_test_output
    """
    import subprocess
    import sys

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Running ASRE Pre-Demo Test Suite[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Run pre_demo_test.py script
    script_path = Path(__file__).parent.parent / "pre_demo_test.py"

    if not script_path.exists():
        console.print("[bold red]Error: pre_demo_test.py not found[/bold red]")
        console.print(f"Expected location: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path), "--output-dir", output_dir]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)