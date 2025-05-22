import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Parameters (hyperparameters)
weekly_contrib = 100  # Weekly contribution in dollars
years = 20            # Investment duration in years
weeks = years * 52    # Total number of weeks

# Growth rates
index_annual = 0.07       # Annualized return for index ETF
momentum_annual = 0.09    # Annualized return for momentum basket
index_weekly = (1 + index_annual)**(1/52) - 1
momentum_weekly = (1 + momentum_annual)**(1/52) - 1

# ETF cost parameters
etf_half_spread = 0.00015     # Half-spread per trade for ETF
etf_expense_annual = 0.0003   # Annual expense ratio for ETF
etf_expense_weekly = etf_expense_annual / 52

# Stock basket cost parameters
stock_half_spread = 0.0005    # Half-spread per trade for stock basket
stock_roundtrip = 2 * stock_half_spread
purge_interval = 26           # Weeks between portfolio purges
purge_fraction = 0.10         # Fraction of portfolio purged each time

def simulate(strategy: str, weekly_growth: float) -> tuple[float, float, float]:
    """
    Simulate the growth of a portfolio using either an ETF or a momentum stock basket.

    Args:
        strategy (str): Either "ETF" or "Stock".
        weekly_growth (float): Weekly growth rate for the strategy.

    Returns:
        tuple[float, float, float]: Final value, total spread cost, total fee cost.

    The simulation models weekly contributions, trading costs, and (for stocks) periodic portfolio purges.
    """
    value, spread, fee = 0.0, 0.0, 0.0
    for week in range(1, weeks + 1):
        value += weekly_contrib
        if strategy == "ETF":
            s = weekly_contrib * etf_half_spread
        else:
            s = weekly_contrib * stock_half_spread
        value -= s
        spread += s
        value *= (1 + weekly_growth)
        if strategy == "ETF":
            f = value * etf_expense_weekly
            value -= f
            fee += f
        if strategy == "Stock" and week % purge_interval == 0:
            turnover = value * purge_fraction
            cost = turnover * stock_roundtrip
            value -= cost
            spread += cost
    return value, spread, fee

def main() -> None:
    """
    Run the ETF vs. momentum basket spread simulation and print results using rich.

    Emphasizes hyperparameters and explains the simulation logic in the output.
    """
    console = Console()
    console.rule("[bold blue]ETF vs. Momentum Basket Spread Simulation[/bold blue]")

    # Emphasize hyperparameters
    hyper_table = Table(title="Simulation Hyperparameters", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    hyper_table.add_column("Parameter", style="cyan", no_wrap=True)
    hyper_table.add_column("Value", style="yellow")
    hyper_table.add_row("Weekly Contribution", f"${weekly_contrib}")
    hyper_table.add_row("Years", str(years))
    hyper_table.add_row("Index Annual Return", f"{index_annual*100:.2f}%")
    hyper_table.add_row("Momentum Annual Return", f"{momentum_annual*100:.2f}%")
    hyper_table.add_row("ETF Half-Spread", f"{etf_half_spread*100:.3f}%")
    hyper_table.add_row("ETF Expense Ratio (Annual)", f"{etf_expense_annual*100:.3f}%")
    hyper_table.add_row("Stock Half-Spread", f"{stock_half_spread*100:.3f}%")
    hyper_table.add_row("Purge Interval (weeks)", str(purge_interval))
    hyper_table.add_row("Purge Fraction", f"{purge_fraction*100:.1f}%")
    console.print(hyper_table)

    # Run simulations
    etf_val, etf_spread, etf_fee = simulate("ETF", index_weekly)
    mom_val, mom_spread, mom_fee = simulate("Stock", momentum_weekly)

    # Prepare results table
    results_table = Table(title="Simulation Results", box=box.ROUNDED, show_header=True, header_style="bold green")
    results_table.add_column("Strategy", style="bold")
    results_table.add_column("Ending Value", justify="right")
    results_table.add_column("Total Spread ($)", justify="right")
    results_table.add_column("Expense Ratio ($)", justify="right")
    results_table.add_column("Net vs. ETF ($)", justify="right")
    results_table.add_row(
        "VTI (Index 7%)",
        f"${etf_val:,.2f}",
        f"${etf_spread:,.2f}",
        f"${etf_fee:,.2f}",
        f"$0.00"
    )
    results_table.add_row(
        "Momentum basket (9%)",
        f"${mom_val:,.2f}",
        f"${mom_spread:,.2f}",
        f"${mom_fee:,.2f}",
        f"${mom_val - etf_val:,.2f}"
    )
    console.print(results_table)

    # Explanation panel
    explanation = (
        "[bold]Simulation Explanation:[/bold]\n"
        "- Each week, a fixed contribution is made and trading costs are deducted.\n"
        "- ETF strategy incurs a small spread and annual expense ratio.\n"
        "- Momentum basket incurs higher spread and, every {purge_interval} weeks, a fraction of the portfolio is purged (sold and repurchased) to simulate turnover.\n"
        "- Results show the impact of costs and turnover on long-term returns."
    )
    console.print(Panel(explanation, title="Explanation", border_style="blue"))

if __name__ == "__main__":
    main()