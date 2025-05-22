from compute_metrics import compute_metrics_from_quarters
import yfinance as yf
from rich.console import Console
from rich.table import Table

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
    table.add_column("Match")

    for key, val in computed.items():
        yf_val = yf_metrics.get(key, None)
        if val is None or yf_val is None:
            match = "[red]❌[/red]"
        else:
            try:
                match = "[green]✅[/green]" if abs(val - yf_val) < 1e-2 else "[red]❌[/red]"
            except Exception:
                match = "[red]❌[/red]"
        table.add_row(key, str(val), str(yf_val), match)

    console.print(table)

if __name__ == "__main__":
    test_compare_metrics("AAPL")
