import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys
from scripts.utility.diagnose_cursor_mcp import diagnose_cursor_mcp
from scripts.utility.swap_mdc import swap_mdc
from scripts.utility.get_sector_industry_list import get_sector_industry_summary
from scripts.utility.simulate_market import run_etf_vs_momentum_simulation

console = Console()

def print_rich_help():
    """Print a modern, styled help message using rich."""
    console.rule("[bold blue]Fundamentals CLI")
    usage = """
    [bold]Usage:[/bold]  python fund_cli.py [cyan]<subcommand>[/cyan] [yellow][options][/yellow]
    """
    console.print(Panel(usage.strip(), title="[bold magenta]How to Use", border_style="blue"))

    table = Table(title="Available Subcommands", box=None, header_style="bold green", show_lines=False)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_row("diagnose-cursor-mcp", "Diagnose Cursor MCP logs for errors and successes. Optionally specify a range of logs to analyze.")
    table.add_row("swap-mdc", "Swap .mdc and .md file extensions in .cursor/rules/")
    table.add_row("spread-sim", "Run ETF vs. momentum basket spread simulation and print results.")
    table.add_row("sector-industry-list", "Build a Sector-Industry summary from Yahoo Finance classifications. Outputs CSV files with summary and full company info.")
    console.print(table)

    notes = """
    [bold]Options:[/bold]
      [cyan]-h[/cyan], [cyan]--help[/cyan]    Show this help message and exit

    [bold]For subcommand-specific options, use:[/bold]
      python fund_cli.py [cyan]<subcommand>[/cyan] [yellow]--help[/yellow]
    """
    console.print(Panel(notes.strip(), title="[bold magenta]Notes", border_style="blue"))

def run_diagnose_cursor_mcp(args):
    """Run diagnostics on Cursor MCP logs."""
    diagnose_cursor_mcp(range_arg=args.range)

def run_swap_mdc(args):
    """Swap .mdc and .md file extensions in .cursor/rules/."""
    swap_mdc()

def run_spread_sim(args):
    """Run ETF vs. momentum basket spread simulation and print results."""
    run_etf_vs_momentum_simulation()

def run_sector_industry_list(args):
    """Build a Sector-Industry summary from Yahoo Finance classifications."""
    get_sector_industry_summary(
        market_category=args.market_category,
        exclude_etf=args.exclude_etf,
        min_round_lot=args.min_round_lot,
        max_tickers=args.max_tickers
    )

def main():
    parser = argparse.ArgumentParser(
        description=argparse.SUPPRESS,
        add_help=False
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)

    # diagnose-cursor-mcp
    parser_dcm = subparsers.add_parser(
        "diagnose-cursor-mcp",
        help=argparse.SUPPRESS
    )
    parser_dcm.add_argument(
        "range", nargs="?", default=None, help="Range of reverse-time-sorted logs to analyze: N, N:M, N:, :M (optional)"
    )
    parser_dcm.set_defaults(func=run_diagnose_cursor_mcp)

    # swap-mdc
    parser_swap = subparsers.add_parser(
        "swap-mdc",
        help=argparse.SUPPRESS
    )
    parser_swap.set_defaults(func=run_swap_mdc)

    # spread-sim
    parser_spread = subparsers.add_parser(
        "spread-sim",
        help=argparse.SUPPRESS
    )
    parser_spread.set_defaults(func=run_spread_sim)

    # sector-industry-list
    parser_sector = subparsers.add_parser(
        "sector-industry-list",
        help=argparse.SUPPRESS
    )
    parser_sector.add_argument('--market-category', nargs='+', default=['Q', 'G'], choices=['Q', 'G', 'S'],
                        help='Market categories to include (default: Q G)')
    parser_sector.add_argument('--exclude-etf', action='store_true', default=True,
                        help='Exclude ETFs (default: True)')
    parser_sector.add_argument('--min-round-lot', type=int, default=100,
                        help='Minimum round lot size (default: 100)')
    parser_sector.add_argument('--max-tickers', type=int, default=None,
                        help='Maximum number of tickers to process (default: all)')
    parser_sector.set_defaults(func=run_sector_industry_list)

    if len(sys.argv) == 1 or any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        print_rich_help()
        sys.exit(0)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Execution Failed[/bold red]"))
        sys.exit(1)

if __name__ == "__main__":
    main()
