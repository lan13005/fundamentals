import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def run_print_mcp_logs(args):
    """Run diagnostics on Cursor MCP logs."""
    from fundamentals.utility.print_mcp_logs import print_mcp_logs

    print_mcp_logs(range_arg=args.range)


def run_swap_mdc(args):
    """Swap .mdc and .md file extensions in .cursor/rules/."""
    from fundamentals.utility.swap_mdc import swap_mdc

    swap_mdc()


def run_macrotrends(args):
    """Fetch financial data from Macrotrends for specified tickers."""
    from fundamentals.utility.macrotrends_scraper import run_macrotrends_scraper

    slug_map_dict = dict(pair.split(":", 1) for pair in args.slug_map) if args.slug_map else None
    run_macrotrends_scraper(
        symbols=args.symbols,
        slug_map=slug_map_dict,
        freq=args.freq,
        force=args.force,
        date=args.date,
    )


def run_housing(args):
    """Run Zillow housing data analysis for specified city and state."""
    from fundamentals.utility.zillow_housing import run_zillow_housing_analysis
    
    city = args.city.lower()
    city = city[0].upper() + city[1:]
    state = args.state.upper()
    
    run_zillow_housing_analysis(
        city=city,
        state=state,
        ignore_cache=args.ignore_cache,
    )


def run_company_info(args):
    """Fetch company information using yfinance for S&P 500 or custom tickers."""
    from fundamentals.utility.company_info import get_company_info

    get_company_info(
        tickers=args.tickers,
        file_name=args.file_name,
        use_sp500=args.use_sp500,
        output_dir=args.output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Fundamentals CLI - Tools for financial data analysis", add_help=False)
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)

    # company-info
    parser_company = subparsers.add_parser(
        "company-info",
        help="Fetch company information using yfinance",
        description="Fetch company information using yfinance for S&P 500 tickers or custom ticker list. Saves data to parquet format.",
        add_help=False,
    )
    parser_company.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="List of ticker symbols (if not provided, uses S&P 500)",
    )
    parser_company.add_argument(
        "--file-name",
        default=None,
        help="Output file name (required when using custom tickers, defaults to 'SP500' for S&P 500)",
    )
    parser_company.add_argument(
        "--use-sp500",
        action="store_true",
        default=True,
        help="Use S&P 500 tickers from Wikipedia (default: True)",
    )
    parser_company.add_argument(
        "--output-dir",
        default="macro_data",
        help="Directory to save parquet file (default: macro_data)",
    )
    parser_company.set_defaults(func=run_company_info)

    # diagnose-cursor-mcp
    parser_dcm = subparsers.add_parser(
        "print-mcp-logs",
        help="Print Cursor MCP logs for errors and successes",
        description="Print Cursor MCP logs for errors and successes. Optionally specify a range of logs to analyze.",
        add_help=False,
    )
    parser_dcm.add_argument(
        "range",
        nargs="?",
        default=None,
        help="Range of reverse-time-sorted logs to analyze: N, N:M, N:, :M (optional)",
    )
    parser_dcm.set_defaults(func=run_print_mcp_logs)

    # swap-mdc
    parser_swap = subparsers.add_parser(
        "swap-mdc",
        help="Swap .mdc and .md file extensions in .cursor/rules/",
        description="Swap .mdc and .md file extensions in the .cursor/rules/ directory.",
        add_help=False,
    )
    parser_swap.set_defaults(func=run_swap_mdc)

    # housing
    parser_housing = subparsers.add_parser(
        "housing",
        help="Analyze Zillow housing data for a specific city and state",
        description="Download and analyze Zillow Home Value Index (ZHVI) data for a specified city and state. Creates visualizations and summary statistics for different housing tiers.",
        add_help=False,
    )
    parser_housing.add_argument(
        "--city",
        default="Denver",
        help="City name to analyze (default: Denver)",
    )
    parser_housing.add_argument(
        "--state",
        default="CO", 
        help="State abbreviation (default: CO)",
    )
    parser_housing.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore existing cache and re-download data from Zillow",
    )
    parser_housing.set_defaults(func=run_housing)

    # macrotrends
    parser_macrotrends = subparsers.add_parser(
        "macrotrends",
        help="Fetch financial data from Macrotrends",
        description="Fetch financial data from Macrotrends for specified tickers and store in parquet format.",
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
