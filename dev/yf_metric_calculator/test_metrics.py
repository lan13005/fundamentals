from compute_metrics import compute_metrics_from_quarters
import yfinance as yf
from rich.console import Console
from rich.table import Table

# NOTE
# finance metrics are very difficult to reproduce and are very noisy
# it is quite difficult to recreate yf metrics from the quarterly / yearly SEC reports


def test_compare_metrics(ticker: str):
    """
    Compare computed metrics to yfinance's and print results.
    """
    console = Console()
    computed = compute_metrics_from_quarters(ticker)
    yf_metrics = yf.Ticker(ticker).info

    table = Table(title=f"Metric Comparison for {ticker}")
    table.add_column("Metric")
    table.add_column("Computed")
    table.add_column("YFinance")
    table.add_column("Relative Difference")
    table.add_column("Match")

    for key, val in computed.items():
        yf_val = yf_metrics.get(key, None)
        if val is None or yf_val is None:
            match = "[red]❌[/red]"
        else:
            try:
                # Use relative difference, allow 2% tolerance
                rel_diff = abs(val - yf_val) / (abs(yf_val) + 1e-8)
                match = "[green]✅[/green]" if rel_diff <= 0.10 else "[red]❌[/red]"
            except Exception:
                match = "[red]❌[/red]"
        table.add_row(key, str(val), str(yf_val), f"{rel_diff:.2f}", match)

    console.print(table)


if __name__ == "__main__":
    test_compare_metrics("MSFT")
