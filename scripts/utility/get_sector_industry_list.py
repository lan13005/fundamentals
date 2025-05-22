#!/usr/bin/env python3
# %matplotlib inline

# + [markdown]
"""
Build a Sector-Industry summary from Yahoo Finance classifications.

Outputs:
    1. yahoo_sector_industry_summary.csv  (three-column summary requested)
    2. yahoo_company_info.csv  (full line-by-line dump)
"""
# -

import io
import requests
import pandas as pd
from pathlib import Path
import yfinance as yf
import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

console = Console()

# 1. Get the master ticker lists (Nasdaq Trader FTP – no login required)
# ##############################################################################

LISTING_URLS = {
    "nasdaqlisted": (
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    ),
    "otherlisted": (
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    ),
}

# Dictionary of important keys used for value investing
# NOTE: Leave at top level for user understanding
key_importance = {
    # --- Governance & Stewardship (quality gate) ----------------------------
    "overallRisk":               "ISS QualityScore decile rank (1 best, 10 worst); quick proxy for governance quality and scandal risk",
    "auditRisk":                 "Audit-committee and accounting oversight score; high values can foreshadow restatements or weak controls",
    "boardRisk":                 "Board independence / diversity score; strong boards improve capital-allocation discipline",
    "compensationRisk":          "Pay-for-performance alignment; mis-aligned incentives erode long-term value",
    "shareHolderRightsRisk":     "Minority-rights protection (one-share-one-vote, no poison pill); low risk limits dilution events",
    # --- Profitability & Economic Moat --------------------------------------
    "returnOnEquity":            "Core measure of capital efficiency; >15 % across cycles signals durable competitive advantage",
    "returnOnAssets":            "Balance-sheet–agnostic profitability; useful cross-sector check on ROE",
    "operatingMargins":          "Captures operating-level pricing power before unusuals; stability is key",
    "ebitdaMargins":             "Cash-flow proxy margin; less accounting noise than net margin",
    "grossMargins":              "Up-stream pricing power and cost moat; early warning of eroding advantage",
    # --- Growth Sustainability ----------------------------------------------
    "revenueGrowth":             "Top-line CAGR driver; must be ≥ nominal GDP for real expansion",
    "earningsGrowth":            "Bottom-line compounding rate; confirms operating leverage",
    "earningsQuarterlyGrowth":   "Near-term momentum signal; persistent acceleration is a tail-wind",
    "freeCashflow":              "Fuel for reinvestment, dividends, and buy-backs; positive FCF validates accrual earnings",
    # --- Balance-Sheet Resilience -------------------------------------------
    "totalCash":                 "Liquidity buffer in absolute terms",
    "totalDebt":                 "Absolute leverage gauge; interpret with debt-to-equity",
    "debtToEquity":              "Leverage ratio; <1 preferred for sleep-at-night safety",
    "currentRatio":              "Short-term solvency (liabilites/assets); <1 can stress working capital during downturns",
    # --- Capital Allocation Track Record ------------------------------------
    "dividendYield":             "Shareholder pay-out today; combine with payoutRatio for sustainability",
    "payoutRatio":               "Earnings share returned to owners; <60 % gives reinvestment headroom",
    "fiveYearAvgDividendYield":  "History of income return; smooths one-off spikes/cuts",
    # --- Valuation vs. Quality (entry discipline) ---------------------------
    "forwardPE":                 "Price vs. next-year EPS forecast; embeds market expectations (forward-looking)",
    "trailingPE":                "Price vs. TTM EPS; backward-looking baseline",
    "priceToBook":               "Asset-based value gauge; useful for financials and cyclical firms",
    "enterpriseToEbitda":        "Capital-structure-neutral multiple; handy for cross-sector comps",
    "pegRatio":                  "PE adjusted for projected EPS growth; screens growth-at-reasonable-price",
    # --- Ownership & Liquidity ----------------------------------------------
    "heldPercentInsiders":       "Management skin-in-the-game; higher = better alignment",
    "heldPercentInstitutions":   "Institutional sponsorship; low values may imply under-researched 'value' pockets",
    "marketCap":                 "Size proxy; informs liquidity and index inclusion",
    "beta":                      "Historical volatility vs. market; helps size positions in a momentum overlay"
}

def load_tickers(market_categories=None, exclude_etf=True, min_round_lot=100, max_tickers=None):
    dfs = []
    for name, url in LISTING_URLS.items():
        txt = requests.get(url, timeout=30).text
        txt_clean = "\n".join(
            line for line in txt.splitlines() if not line.startswith("File Creation Time")
        )
        df = pd.read_csv(io.StringIO(txt_clean), sep="|")
        symcol = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        # Apply filters
        if 'Market Category' in df.columns:
            if market_categories:
                df = df[df['Market Category'].isin(market_categories)]
        if 'ETF' in df.columns and exclude_etf:
            df = df[df['ETF'] != 'Y']
        if 'Round Lot Size' in df.columns:
            df = df[df['Round Lot Size'] >= min_round_lot]
        if 'Test Issue' in df.columns:
            df = df[df['Test Issue'] != 'Y']
        if 'NextShares' in df.columns:
            df = df[df['NextShares'] != 'Y']
        dfs.append(df[[symcol]].rename(columns={symcol: "ticker"}))
    master = pd.concat(dfs, ignore_index=True).drop_duplicates()
    master["ticker"] = master["ticker"].str.upper()
    tickers = master["ticker"].tolist()
    if max_tickers:
        tickers = tickers[:max_tickers]
    return tickers

# 2. Pull Yahoo Finance metadata in batches (to respect rate limits)
# ##############################################################################

def fetch_yahoo_metadata(tickers, key_importance):
    """
    Fetch Yahoo Finance metadata for a list of tickers using rich progress bar.
    """
    BATCH = 50
    PAUSE = 1.0
    records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Downloading Yahoo metadata...", total=len(tickers))
        for i in range(0, len(tickers), BATCH):
            batch = tickers[i : i + BATCH]
            data = yf.Tickers(" ".join(batch))
            for tk, obj in data.tickers.items():
                try:
                    info = obj.info
                except Exception:
                    continue
                sector = info.get("sector")
                industry = info.get("industry")
                if not sector or not industry:
                    continue
                record = {
                    "ticker": tk,
                    "company": info.get("shortName", ""),
                    "sector": sector,
                    "industry": industry,
                }
                for k in key_importance:
                    record[k] = info.get(k, None)
                records.append(record)
            progress.update(task, advance=len(batch))
            time.sleep(PAUSE)
    return records

def format_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame so that ints are written as ints and floats as 4 decimals.
    """
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_float_dtype(df_out[col]):
            # If all values are integer-like, cast to int
            if (df_out[col].dropna() % 1 == 0).all():
                df_out[col] = df_out[col].dropna().astype('Int64')
            else:
                df_out[col] = df_out[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
    return df_out

def get_top_companies_by_marketcap(df: pd.DataFrame, n=3) -> pd.Series:
    """
    For each sector-industry, return the top n tickers by market cap as a comma-separated string.
    """
    def top_n(group):
        sorted_group = group.sort_values("marketCap", ascending=False)
        return ", ".join(sorted_group["ticker"].head(n).astype(str))
    return top_n

def build_sector_industry_summary(df_full: pd.DataFrame, n_examples=3) -> pd.DataFrame:
    """
    Build a summary DataFrame with sector-industry, example companies (by market cap), and count.
    """
    df_full = df_full.copy()
    df_full["sector_industry"] = df_full["sector"] + " - " + df_full["industry"]
    summary = (
        df_full.groupby("sector_industry")
        .apply(lambda g: pd.Series({
            "Example Companies": ", ".join(g.sort_values("marketCap", ascending=False)["ticker"].head(n_examples).astype(str)),
            "Number of Companies": g["ticker"].count()
        }), include_groups=False)
        .reset_index()
        .rename(columns={"sector_industry": "Sector - Industry"})
        .sort_values("Number of Companies", ascending=False)
    )
    return summary

def print_sector_industry_table(summary: pd.DataFrame):
    table = Table(title="Sector-Industry Summary", show_lines=True)
    table.add_column("Sector - Industry", style="cyan", no_wrap=True)
    table.add_column("Example Companies", style="magenta")
    table.add_column("Number of Companies", style="green")
    for _, row in summary.iterrows():
        table.add_row(str(row["Sector - Industry"]), str(row["Example Companies"]), str(row["Number of Companies"]))
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Build a Sector-Industry summary from Yahoo Finance classifications.")
    parser.add_argument('--market-category', nargs='+', default=['Q', 'G'], choices=['Q', 'G', 'S'],
                        help='Market categories to include (default: Q G)')
    parser.add_argument('--exclude-etf', action='store_true', default=True,
                        help='Exclude ETFs (default: True)')
    parser.add_argument('--min-round-lot', type=int, default=100,
                        help='Minimum round lot size (default: 100)')
    parser.add_argument('--max-tickers', type=int, default=None,
                        help='Maximum number of tickers to process (default: all)')
    args = parser.parse_args()

    console.print(Panel("[bold cyan]Fetching tickers with filters:[/bold cyan]\n"
                            f"Market Category: {args.market_category}\n"
                            f"Exclude ETF: {args.exclude_etf}\n"
                            f"Min Round Lot: {args.min_round_lot}", title="[bold green]Ticker Loader Filters"))
    tickers = load_tickers(
        market_categories=args.market_category,
        exclude_etf=args.exclude_etf,
        min_round_lot=args.min_round_lot,
        max_tickers=args.max_tickers
    )
    console.print(f"[bold yellow]Fetched {len(tickers):,} U.S.-listed tickers after filtering[/bold yellow]")
    records = fetch_yahoo_metadata(tickers, key_importance)
    df_full = pd.DataFrame.from_records(records)
    df_full_formatted = format_dataframe_for_csv(df_full)
    df_full_formatted.to_csv("yahoo_company_info.csv", index=False)
    summary = build_sector_industry_summary(df_full, n_examples=3)
    summary.to_csv("yahoo_sector_industry_summary.csv", index=False)
    print_sector_industry_table(summary)
    console.print("\n✔ Done!  Files written:")
    for f in Path(".").glob("*sector_industry*.csv"):
        console.print(f"  • [bold green]{f.resolve()}[/bold green]")

if __name__ == "__main__":
    main()
