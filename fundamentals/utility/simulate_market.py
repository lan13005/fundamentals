###########################################################
# Simulate ETF vs. Momentum Basket Spread using Monte Carlo
# simulation.
# Outputs:
#     - Prints simulation results and saves histogram to
#       sim_market_histogram.png
###########################################################

import os

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Parameters (hyperparameters)
weekly_contrib = 100  # Weekly contribution in dollars (Constant)
years = 2  # Investment duration in years (Constant)
weeks = years * 52  # Total number of weeks (Constant)

# Growth rates (lognormal, parameters are for the underlying normal log-returns)
index_annual_mean = 0.07  # Mean annualized return for index ETF (Lognormal, mean=7%)
index_annual_sigma = 0.15  # Annualized volatility for index ETF (Lognormal, sigma=15%)
momentum_annual_mean = 0.09  # Mean annualized return for momentum basket (Lognormal, mean=9%)
momentum_annual_sigma = 0.20  # Annualized volatility for momentum basket (Lognormal, sigma=20%)

# ETF cost parameters (you can free the std also but why)
etf_half_spread_mean = 0.00010  # ETF half-spread per trade (Constant, 1bp)
etf_half_spread_sigma = 0.0  # ETF half-spread stddev (Constant)
etf_expense_annual_mean = 0.0003  # ETF annual expense ratio (Constant, 0.03%)
etf_expense_annual_sigma = 0.0  # ETF expense ratio stddev (Constant)

# Stock basket cost parameters
stock_half_spread_mean = 0.0004  # Stock basket half-spread per trade (Normal, mean=4bp)
stock_half_spread_sigma = 0.0001  # Stock basket half-spread stddev (Normal, sigma=1bp)
purge_fraction_mean = 0.50  # Fraction of portfolio purged each time (Normal, mean=50%)
purge_fraction_sigma = 0.10  # Purge fraction stddev (Normal, sigma=10%)
purge_interval = 26  # Weeks between portfolio purges (Constant, 6 months)

n_sim = 5000  # Number of Monte Carlo runs (Constant)

rng = np.random.default_rng()


def simulate(
    strategy: str,
    weekly_logret_mean: float,
    weekly_logret_sigma: float,
    etf_half_spread: float,
    etf_expense_weekly: float,
    stock_half_spread: float,
    stock_roundtrip: float,
    purge_fraction: float,
) -> tuple[float, float, float]:
    """
    Simulate the growth of a portfolio using either an ETF or a momentum stock basket.
    Args:
        strategy (str): Either "ETF" or "Stock".
        weekly_logret_mean (float): Mean of weekly log-return for the strategy.
        weekly_logret_sigma (float): Stddev of weekly log-return for the strategy.
        etf_half_spread (float): ETF half-spread per trade.
        etf_expense_weekly (float): ETF weekly expense ratio.
        stock_half_spread (float): Stock basket half-spread per trade.
        stock_roundtrip (float): Stock basket roundtrip cost.
        purge_fraction (float): Fraction of portfolio purged at each interval.
    Returns:
        tuple[float, float, float]: Final value, total spread cost, total fee cost.
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
        # Sample a new weekly log-return
        logret = rng.normal(weekly_logret_mean, weekly_logret_sigma)
        weekly_growth = np.expm1(logret)
        value *= 1 + weekly_growth
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


def run_monte_carlo():
    etf_vals, etf_spreads, etf_fees = [], [], []
    mom_vals, mom_spreads, mom_fees = [], [], []
    # Calculate weekly log-return mean and sigma for both strategies
    index_weekly_logret_mean = (np.log1p(index_annual_mean) - 0.5 * index_annual_sigma**2) / 52
    index_weekly_logret_sigma = index_annual_sigma / np.sqrt(52)
    momentum_weekly_logret_mean = (np.log1p(momentum_annual_mean) - 0.5 * momentum_annual_sigma**2) / 52
    momentum_weekly_logret_sigma = momentum_annual_sigma / np.sqrt(52)
    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Monte Carlo simulations...", total=n_sim)
        for _ in range(n_sim):
            # ETF costs: constant or normal as documented
            etf_half_spread = max(0, rng.normal(etf_half_spread_mean, etf_half_spread_sigma))
            etf_expense_annual = max(0, rng.normal(etf_expense_annual_mean, etf_expense_annual_sigma))
            etf_expense_weekly = etf_expense_annual / 52
            # Stock basket costs: normal as documented
            stock_half_spread = max(0, rng.normal(stock_half_spread_mean, stock_half_spread_sigma))
            stock_roundtrip = 2 * stock_half_spread
            purge_fraction = max(0, rng.normal(purge_fraction_mean, purge_fraction_sigma))
            # Simulate ETF
            etf_val, etf_spread, etf_fee = simulate(
                "ETF",
                index_weekly_logret_mean,
                index_weekly_logret_sigma,
                etf_half_spread,
                etf_expense_weekly,
                stock_half_spread,
                stock_roundtrip,
                purge_fraction,
            )
            etf_vals.append(etf_val)
            etf_spreads.append(etf_spread)
            etf_fees.append(etf_fee)
            # Simulate Momentum
            mom_val, mom_spread, mom_fee = simulate(
                "Stock",
                momentum_weekly_logret_mean,
                momentum_weekly_logret_sigma,
                etf_half_spread,
                etf_expense_weekly,
                stock_half_spread,
                stock_roundtrip,
                purge_fraction,
            )
            mom_vals.append(mom_val)
            mom_spreads.append(mom_spread)
            mom_fees.append(mom_fee)
            progress.update(task, advance=1)
    return (
        np.array(etf_vals),
        np.array(etf_spreads),
        np.array(etf_fees),
        np.array(mom_vals),
        np.array(mom_spreads),
        np.array(mom_fees),
    )


def median_and_sigma_pm(arr):
    med = np.median(arr)
    p16 = np.percentile(arr, 16)
    p84 = np.percentile(arr, 84)
    return p16, med, p84


def fmt_pm(p16, med, p84, prec=2):
    return f"${p16:,.{prec}f} | ${med:,.{prec}f} | ${p84:,.{prec}f}"


def modern_hist(ax, data, bins, label, color):
    counts, edges, patches = ax.hist(data, bins=bins, histtype="step", linewidth=2, color=color, label=label)
    ax.stairs(counts, edges, color=color, linewidth=2)
    ax.set_xlabel("Ending Value ($)")
    ax.set_ylabel("Simulations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    return counts, edges


def run_etf_vs_momentum_simulation():
    """
    Run the ETF vs. momentum basket spread simulation and print results using rich.
    Emphasizes hyperparameters and explains the simulation logic in the output.
    """
    console = Console()
    console.rule("[bold blue]ETF vs. Momentum Basket Spread Simulation (Monte Carlo)")

    # Emphasize hyperparameters
    hyper_table = Table(
        title="Simulation Hyperparameters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    hyper_table.add_column("Parameter", style="cyan", no_wrap=True)
    hyper_table.add_column("Value", style="yellow")
    hyper_table.add_row("Weekly Contribution", f"${weekly_contrib}")
    hyper_table.add_row("Years", str(years))
    hyper_table.add_row(
        "Index Annual Return (mean ± σ)",
        f"{index_annual_mean*100:.2f}% ± {index_annual_sigma*100:.2f}%",
    )
    hyper_table.add_row(
        "Momentum Annual Return (mean ± σ)",
        f"{momentum_annual_mean*100:.2f}% ± {momentum_annual_sigma*100:.2f}%",
    )
    hyper_table.add_row(
        "ETF Half-Spread (mean ± σ)",
        f"{etf_half_spread_mean*100:.3f}% ± {etf_half_spread_sigma*100:.3f}%",
    )
    hyper_table.add_row(
        "ETF Expense Ratio (Annual, mean ± σ)",
        f"{etf_expense_annual_mean*100:.3f}% ± {etf_expense_annual_sigma*100:.3f}%",
    )
    hyper_table.add_row(
        "Stock Half-Spread (mean ± σ)",
        f"{stock_half_spread_mean*100:.3f}% ± {stock_half_spread_sigma*100:.3f}%",
    )
    hyper_table.add_row("Purge Interval (weeks)", str(purge_interval))
    hyper_table.add_row(
        "Purge Fraction (mean ± σ)",
        f"{purge_fraction_mean*100:.1f}% ± {purge_fraction_sigma*100:.1f}%",
    )
    console.print(hyper_table)

    # Run Monte Carlo simulations
    etf_vals, etf_spreads, etf_fees, mom_vals, mom_spreads, mom_fees = run_monte_carlo()

    etf_val_minus, etf_val_med, etf_val_plus = median_and_sigma_pm(etf_vals)
    etf_spread_minus, etf_spread_med, etf_spread_plus = median_and_sigma_pm(etf_spreads)
    etf_fee_minus, etf_fee_med, etf_fee_plus = median_and_sigma_pm(etf_fees)
    mom_val_minus, mom_val_med, mom_val_plus = median_and_sigma_pm(mom_vals)
    mom_spread_minus, mom_spread_med, mom_spread_plus = median_and_sigma_pm(mom_spreads)
    mom_fee_minus, mom_fee_med, mom_fee_plus = median_and_sigma_pm(mom_fees)

    net_vs_etf = mom_vals - etf_vals
    net_vs_etf_minus, net_vs_etf_med, net_vs_etf_plus = median_and_sigma_pm(net_vs_etf)

    # Prepare results table
    results_table = Table(
        title="Simulation Results (-σ | Median | +σ)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
    )
    results_table.add_column("Strategy", style="bold", min_width=22)
    results_table.add_column("Ending Value", justify="right", min_width=28)
    results_table.add_column("Total Spread ($)", justify="right", min_width=28)
    results_table.add_column("Expense Ratio ($)", justify="right", min_width=28)
    results_table.add_column("Net vs. ETF ($)", justify="right", min_width=28)
    results_table.add_row(
        "VTI (Index 7%)",
        fmt_pm(etf_val_minus, etf_val_med, etf_val_plus),
        fmt_pm(etf_spread_minus, etf_spread_med, etf_spread_plus),
        fmt_pm(etf_fee_minus, etf_fee_med, etf_fee_plus),
        fmt_pm(0, 0, 0),
    )
    results_table.add_row(
        "Momentum basket (9%)",
        fmt_pm(mom_val_minus, mom_val_med, mom_val_plus),
        fmt_pm(mom_spread_minus, mom_spread_med, mom_spread_plus),
        fmt_pm(mom_fee_minus, mom_fee_med, mom_fee_plus),
        fmt_pm(net_vs_etf_minus, net_vs_etf_med, net_vs_etf_plus),
    )
    console.print(results_table)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    all_vals = np.concatenate([etf_vals, mom_vals])
    xlow = np.percentile(all_vals, 0)
    xhigh = np.percentile(all_vals, 100)
    bins = np.linspace(xlow, xhigh, 100)
    modern_hist(ax, etf_vals, bins, label="ETF (VTI)", color="#1f77b4")
    modern_hist(ax, mom_vals, bins, label="Momentum Basket", color="#d62728")
    ax.set_xlim(xlow, xhigh)
    ax.set_title("Distribution of Ending Portfolio Values (Monte Carlo)", fontsize=14)
    plt.tight_layout()
    out_path = os.path.abspath("sim_market_histogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]Histogram saved to:[/green] [bold]{out_path}[/bold]")

    # Explanation panel
    explanation = f"""[bold]Simulation Explanation:[/bold]
- Each week, a fixed contribution is made and trading costs are deducted.\n
- ETF and momentum returns, spreads, and costs are drawn from approximate but realistic distributions each run.\n
- ETF strategy incurs a small spread and annual expense ratio.\n
- Momentum basket incurs higher spread and, every {purge_interval} weeks, a fraction of the portfolio is purged (sold and repurchased) to simulate turnover.\n
- Results show the impact of costs, turnover, and market randomness on long-term returns.\n
- Table shows median ± 1σ (68%) range from Monte Carlo ensemble.\n
- Histogram shows the full distribution of outcomes."""
    console.print(Panel(explanation, title="Explanation", border_style="blue"))
