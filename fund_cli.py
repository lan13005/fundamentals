import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fundamentals.utility.diagnose_cursor_mcp import diagnose_cursor_mcp
from fundamentals.utility.get_sector_industry_list import get_sector_industry_summary
from fundamentals.utility.macrotrends_scraper import run_macrotrends_scraper
from fundamentals.utility.simulate_market import run_etf_vs_momentum_simulation
from fundamentals.utility.swap_mdc import swap_mdc

console = Console()


def print_rich_help(parser=None, subcommand=None):
    """Print a modern, styled help message using rich.

    Args:
        parser: The ArgumentParser instance to get help from
        subcommand: The subcommand name if showing subcommand help
    """
    console.rule("[bold blue]Fundamentals CLI")

    if subcommand:
        # Show subcommand-specific help
        subparser = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        cmd_parser = subparser.choices[subcommand]

        usage = f"""
        [bold]Usage:[/bold]  fund {subcommand} [yellow][options][/yellow]
        """
        console.print(Panel(usage.strip(), title="[bold magenta]How to Use", border_style="blue"))

        # Description
        if cmd_parser.description:
            console.print(Panel(cmd_parser.description, title="[bold magenta]Description", border_style="blue"))

        # Arguments
        if cmd_parser._actions:
            table = Table(title="Arguments", box=None, header_style="bold green", show_lines=False)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            table.add_column("Default", style="yellow")

            for action in cmd_parser._actions:
                if action.help != argparse.SUPPRESS:
                    # Format option names
                    if action.option_strings:
                        option = ", ".join(action.option_strings)
                    else:
                        option = action.dest

                    # Get default value if any
                    default = ""
                    if action.default is not None and action.default != argparse.SUPPRESS:
                        if isinstance(action.default, list):
                            default = f"[{', '.join(map(str, action.default))}]"
                        else:
                            default = str(action.default)

                    # Add choices if any
                    help_text = action.help or ""
                    if action.choices:
                        help_text += f" (choices: {', '.join(map(str, action.choices))})"

                    table.add_row(option, help_text, default)

            console.print(table)
    else:
        # Show main help
        usage = """
        [bold]Usage:[/bold]  fund [cyan]<subcommand>[/cyan] [yellow][options][/yellow]
        """
        console.print(Panel(usage.strip(), title="[bold magenta]How to Use", border_style="blue"))

        table = Table(
            title="Available Subcommands",
            box=None,
            header_style="bold green",
            show_lines=False,
        )
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        # Get subcommands from parser
        subparser = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        for cmd, cmd_parser in subparser.choices.items():
            table.add_row(cmd, cmd_parser.description or cmd_parser.help or "")

        console.print(table)

        notes = """
        [bold]Options:[/bold]
          [cyan]-h[/cyan], [cyan]--help[/cyan]    Show this help message and exit

        [bold]For subcommand-specific options, use:[/bold]
          fund [cyan]<subcommand>[/cyan] [yellow]--help[/yellow]
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
        max_tickers=args.max_tickers,
    )


def run_macrotrends(args):
    """Fetch financial data from Macrotrends for specified tickers."""
    slug_map_dict = dict(pair.split(":", 1) for pair in args.slug_map) if args.slug_map else None
    run_macrotrends_scraper(
        symbols=args.symbols,
        slug_map=slug_map_dict,
        freq=args.freq,
        force=args.force,
        date=args.date,
    )


def main():
    parser = argparse.ArgumentParser(description="Fundamentals CLI - Tools for financial data analysis", add_help=False)
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)

    # diagnose-cursor-mcp
    parser_dcm = subparsers.add_parser(
        "diagnose-cursor-mcp",
        help="Diagnose Cursor MCP logs for errors and successes",
        description="Analyze Cursor MCP logs for errors and successes. Optionally specify a range of logs to analyze.",
        add_help=False,
    )
    parser_dcm.add_argument(
        "range",
        nargs="?",
        default=None,
        help="Range of reverse-time-sorted logs to analyze: N, N:M, N:, :M (optional)",
    )
    parser_dcm.set_defaults(func=run_diagnose_cursor_mcp)

    # swap-mdc
    parser_swap = subparsers.add_parser(
        "swap-mdc",
        help="Swap .mdc and .md file extensions in .cursor/rules/",
        description="Swap .mdc and .md file extensions in the .cursor/rules/ directory.",
        add_help=False,
    )
    parser_swap.set_defaults(func=run_swap_mdc)

    # spread-sim
    parser_spread = subparsers.add_parser(
        "spread-sim",
        help="Run ETF vs. momentum basket spread simulation",
        description="Run ETF vs. momentum basket spread simulation and print results.",
        add_help=False,
    )
    parser_spread.set_defaults(func=run_spread_sim)

    # sector-industry-list
    parser_sector = subparsers.add_parser(
        "sector-industry-list",
        help="Build a Sector-Industry summary from Yahoo Finance",
        description="Build a Sector-Industry summary from Yahoo Finance classifications. Outputs CSV files with summary and full company info.",
        add_help=False,
    )
    parser_sector.add_argument(
        "--market-category",
        nargs="+",
        default=["Q", "G"],
        choices=["Q", "G", "S"],
        help="Market categories to include (default: Q G)",
    )
    parser_sector.add_argument(
        "--exclude-etf",
        action="store_true",
        default=True,
        help="Exclude ETFs (default: True)",
    )
    parser_sector.add_argument(
        "--min-round-lot",
        type=int,
        default=100,
        help="Minimum round lot size (default: 100)",
    )
    parser_sector.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Maximum number of tickers to process (default: all)",
    )
    parser_sector.set_defaults(func=run_sector_industry_list)

    # macrotrends
    parser_macrotrends = subparsers.add_parser(
        "macrotrends",
        help="Fetch financial data from Macrotrends",
        description="Fetch financial data from Macrotrends for specified tickers and store in parquet/duckdb format.",
        add_help=False,
    )
    parser_macrotrends.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        metavar="TICKER",
        help="List of ticker symbols to process",
    )
    parser_macrotrends.add_argument(
        "--slug-map",
        nargs="+",
        default=[],
        metavar="TICKER:slug",
        help="Optional ticker:slug mappings to override defaults",
    )
    parser_macrotrends.add_argument(
        "--freq",
        choices=["Q", "A"],
        default="Q",
        help="Frequency of data - Q for quarterly or A for annual (default: Q)",
    )
    parser_macrotrends.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and force fetch from web",
    )
    parser_macrotrends.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Snapshot date (default: today)",
    )
    parser_macrotrends.set_defaults(func=run_macrotrends)

    # Handle help flags
    if len(sys.argv) == 1:
        print_rich_help(parser)
        sys.exit(0)

    # Check for help on main command or subcommand
    if "-h" in sys.argv or "--help" in sys.argv:
        if len(sys.argv) > 2 and sys.argv[1] in subparsers.choices:
            print_rich_help(parser, sys.argv[1])
        else:
            print_rich_help(parser)
        sys.exit(0)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Execution Failed[/bold red]"))
        sys.exit(1)


if __name__ == "__main__":
    main()
