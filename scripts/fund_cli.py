import argparse
from rich.console import Console
from rich.panel import Panel
import sys
from scripts.utility import diagnose_cursor_mcp, swap_mdc
from scripts.utility import simulate_market

console = Console()

def run_diagnose_cursor_mcp(args):
    # Pass the range argument as a string or None
    diagnose_cursor_mcp.main(range_arg=args.range)

def run_swap_mdc(args):
    swap_mdc.main()

def run_spread_sim(args):
    simulate_market.main()

def main():
    parser = argparse.ArgumentParser(
        description="Fundamentals CLI: Dispatch utility for all scripts in the scripts/ folder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)

    # diagnose-cursor-mcp
    parser_dcm = subparsers.add_parser(
        "diagnose-cursor-mcp",
        help="Diagnose Cursor MCP logs for errors and successes."
    )
    parser_dcm.add_argument(
        "range", nargs="?", default=None, help="Range of reverse-time-sorted logs to analyze: N, N:M, N:, :M (optional)"
    )
    parser_dcm.set_defaults(func=run_diagnose_cursor_mcp)

    # swap-mdc
    parser_swap = subparsers.add_parser(
        "swap-mdc",
        help="Swap .mdc and .md file extensions in .cursor/rules/."
    )
    parser_swap.set_defaults(func=run_swap_mdc)

    # spread-sim
    parser_spread = subparsers.add_parser(
        "spread-sim",
        help="Run ETF vs. momentum basket spread simulation and print results."
    )
    parser_spread.set_defaults(func=run_spread_sim)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Execution Failed[/bold red]"))
        sys.exit(1)

if __name__ == "__main__":
    main()
