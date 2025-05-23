#!/usr/bin/env python3
# %matplotlib inline

#########################################################
# Build a Sector-Industry summary from Yahoo Finance
# classifications.
# Outputs:
#     1. yahoo_sector_industry_summary.csv  (three-column
#        summary requested)
#     2. yahoo_company_info.csv  (full line-by-line dump)
#########################################################

import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
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

console = Console()

# 1. Get the master ticker lists (Nasdaq Trader FTP - no login required)
# ##############################################################################

LISTING_URLS = {
    "nasdaqlisted": ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
    "otherlisted": ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"),
}

# ---------------------------------------------------------------------------
# Dictionary of important keys used for value investing
# NOTE: Leave at top level for user understanding
# ---------------------------------------------------------------------------
key_importance = {
    # --- Governance & Stewardship (quality gate) ----------------------------
    "overallRisk": "ISS QualityScore decile rank (1 best, 10 worst); quick proxy for governance quality and scandal risk",
    "auditRisk": "Audit-committee & accounting oversight score; high values can foreshadow restatements or weak controls",
    "boardRisk": "Board independence / diversity score; strong boards improve capital-allocation discipline",
    "compensationRisk": "Pay-for-performance alignment; mis-aligned incentives erode long-term value",
    "shareHolderRightsRisk": "Minority-rights protection (one-share-one-vote, no poison pill); low risk limits dilution events",
    # --- Profitability & Economic Moat --------------------------------------
    "returnOnEquity": "Core measure of capital efficiency; >15 % across cycles signals durable competitive advantage",
    "returnOnAssets": "Balance-sheet-agnostic profitability; useful cross-sector check on ROE",
    "operatingMargins": "Captures operating-level pricing power before unusuals; stability is key",
    "ebitdaMargins": "Cash-flow proxy margin; less accounting noise than net margin",
    "grossMargins": "Up-stream pricing power and cost moat; early warning of eroding advantage",
    "profitMargins": "Bottom-line (net) profitability; confirms that revenue converts to cash",
    # --- Growth Sustainability ----------------------------------------------
    "revenueGrowth": "Top-line CAGR driver; must be ≥ nominal GDP for real expansion",
    "earningsGrowth": "Bottom-line compounding rate; confirms operating leverage",
    "earningsQuarterlyGrowth": "Near-term momentum signal; persistent acceleration is a tail-wind",
    "freeCashflow": "Fuel for reinvestment, dividends, and buy-backs; positive FCF validates accrual earnings",
    # --- Balance-Sheet Resilience -------------------------------------------
    "totalCash": "Liquidity buffer in absolute terms",
    "totalDebt": "Absolute leverage gauge; interpret with debt-to-equity",
    "debtToEquity": "Leverage ratio; <1 preferred for sleep-at-night safety",
    "currentRatio": "Short-term solvency; <1 can stress working capital during downturns",
    # --- Capital-Allocation Track Record ------------------------------------
    "dividendYield": "Shareholder pay-out today; combine with payoutRatio for sustainability",
    "payoutRatio": "Earnings share returned to owners; <60 % gives reinvestment headroom",
    "fiveYearAvgDividendYield": "History of income return; smooths one-off spikes/cuts",
    # --- Valuation vs. Quality (entry discipline) ---------------------------
    "forwardPE": "Price vs. next-year EPS forecast; embeds market expectations (forward-looking)",
    "trailingPE": "Price vs. TTM EPS; backward-looking baseline",
    "priceToBook": "Asset-based value gauge; useful for financials & cyclicals",
    "priceToSalesTrailing12Months": "Sales multiple; helpful when earnings are depressed",
    "enterpriseToEbitda": "Capital-structure-neutral multiple; handy for cross-sector comps",
    "trailingPegRatio": "PEG based on trailing EPS and forecast growth; screens growth-at-reasonable-price",
    # --- Ownership & Liquidity ----------------------------------------------
    "heldPercentInsiders": "Management skin-in-the-game; higher = better alignment",
    "heldPercentInstitutions": "Institutional sponsorship; low values may imply under-researched 'value' pockets",
    "marketCap": "Size proxy; informs liquidity and index inclusion",
    "beta": "Historical volatility vs. market; helps size positions within a momentum overlay",
    # --- Holistic Valuation Context -----------------------------------------
    "enterpriseValue": "Total operating valuation incl. debt & cash; cross-capital-structure metric",
}


def load_tickers(market_categories=None, exclude_etf=True, min_round_lot=100, max_tickers=None):
    dfs = []
    for _name, url in LISTING_URLS.items():
        txt = requests.get(url, timeout=30).text
        txt_clean = "\n".join(line for line in txt.splitlines() if not line.startswith("File Creation Time"))
        df = pd.read_csv(io.StringIO(txt_clean), sep="|")
        symcol = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        # Apply filters
        if "Market Category" in df.columns:
            if market_categories:
                df = df[df["Market Category"].isin(market_categories)]
        if "ETF" in df.columns and exclude_etf:
            df = df[df["ETF"] != "Y"]
        if "Round Lot Size" in df.columns:
            df = df[df["Round Lot Size"] >= min_round_lot]
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]
        if "NextShares" in df.columns:
            df = df[df["NextShares"] != "Y"]
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
    Returns:
        records: List[dict] of successfully loaded company info
        failed_tickers: List[str] of tickers that failed to load or lacked sector/industry
        sector_industry_counts: Dict[(sector, industry), {'requested': int, 'loaded': int, 'failed': int}]
    """
    BATCH = 50
    PAUSE = 1.0
    records = []
    failed_tickers = []
    sector_industry_counts = {}
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
            for tk in batch:
                obj = data.tickers.get(tk)
                if obj is None:
                    failed_tickers.append(tk)
                    continue
                try:
                    info = obj.info
                except Exception:
                    failed_tickers.append(tk)
                    continue
                sector = info.get("sector")
                industry = info.get("industry")
                if not sector or not industry:
                    failed_tickers.append(tk)
                    continue
                key = (sector, industry)
                if key not in sector_industry_counts:
                    sector_industry_counts[key] = {
                        "requested": 0,
                        "loaded": 0,
                        "failed": 0,
                    }
                sector_industry_counts[key]["loaded"] += 1
                record = {
                    "ticker": tk,
                    "company": info.get("shortName", ""),
                    "sector": sector,
                    "industry": industry,
                }
                for k in key_importance:
                    record[k] = info.get(k, None)
                records.append(record)
            for tk in batch:
                # Count all requested per sector/industry (even if failed)
                obj = data.tickers.get(tk)
                sector = None
                industry = None
                if obj is not None:
                    try:
                        info = obj.info
                        sector = info.get("sector")
                        industry = info.get("industry")
                    except Exception:
                        pass
                key = (sector, industry) if sector and industry else ("Unknown", "Unknown")
                if key not in sector_industry_counts:
                    sector_industry_counts[key] = {
                        "requested": 0,
                        "loaded": 0,
                        "failed": 0,
                    }
                sector_industry_counts[key]["requested"] += 1
                if tk in failed_tickers:
                    sector_industry_counts[key]["failed"] += 1
            progress.update(task, advance=len(batch))
            time.sleep(PAUSE)
    return records, failed_tickers, sector_industry_counts


def format_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame so that ints are written as ints and floats as 4 decimals.
    """
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_float_dtype(df_out[col]):
            # If all values are integer-like, cast to int
            if (df_out[col].dropna() % 1 == 0).all():
                df_out[col] = df_out[col].dropna().astype("Int64")
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


def human_format(num):
    """
    Convert a number to a human-readable string with K, M, B, or T suffix.
    Args:
        num (float or int): The number to format.
    Returns:
        str: Human-readable string.
    """
    if num is None or pd.isna(num):
        return "N/A"
    num = float(num)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            if unit == "":
                return f"{num:.0f}"
            return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"  # For numbers >= 1 quadrillion


def build_sector_industry_summary(df_full: pd.DataFrame, n_examples=3) -> pd.DataFrame:
    """
    Build a summary DataFrame with sector-industry, example companies (by market cap), count, and total market cap.
    """
    df_full = df_full.copy()
    df_full["sector_industry"] = df_full["sector"] + " - " + df_full["industry"]
    summary = (
        df_full.groupby("sector_industry")
        .apply(
            lambda g: pd.Series(
                {
                    "Example Companies": ", ".join(
                        g.sort_values("marketCap", ascending=False)["ticker"].head(n_examples).astype(str)
                    ),
                    "Number of Companies": g["ticker"].count(),
                    "Total Market Cap": g["marketCap"].sum() if "marketCap" in g else 0,
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"sector_industry": "Sector - Industry"})
        .sort_values("Number of Companies", ascending=False)
    )
    # Format Total Market Cap as human readable for display and CSV
    summary["Total Market Cap"] = summary["Total Market Cap"].apply(human_format)
    return summary


def print_sector_industry_table(summary: pd.DataFrame):
    table = Table(title="Sector-Industry Summary", show_lines=True)
    table.add_column("Sector - Industry", style="cyan", no_wrap=True)
    table.add_column("Example Companies", style="magenta")
    table.add_column("Number of Companies", style="green")
    table.add_column("Total Market Cap", style="yellow")
    for _, row in summary.iterrows():
        table.add_row(
            str(row["Sector - Industry"]),
            str(row["Example Companies"]),
            str(row["Number of Companies"]),
            str(row["Total Market Cap"]) if pd.notnull(row["Total Market Cap"]) else "N/A",
        )
    console.print(table)


def get_sector_industry_summary(market_category=["Q", "G"], exclude_etf=True, min_round_lot=100, max_tickers=None):
    """
    Build a Sector-Industry summary from Yahoo Finance classifications and write CSV outputs.
    Args:
        market_category (list[str]): Market categories to include (default: ['Q', 'G'])
        exclude_etf (bool): Exclude ETFs (default: True)
        min_round_lot (int): Minimum round lot size (default: 100)
        max_tickers (int or None): Maximum number of tickers to process (default: all)
    Returns:
        pd.DataFrame: The summary DataFrame
    """
    console.print(
        Panel(
            "[bold cyan]Fetching tickers with filters:[/bold cyan]\n"
            f"Market Category: {market_category}\n"
            f"Exclude ETF: {exclude_etf}\n"
            f"Min Round Lot: {min_round_lot}",
            title="[bold green]Ticker Loader Filters",
        )
    )
    tickers = load_tickers(
        market_categories=market_category,
        exclude_etf=exclude_etf,
        min_round_lot=min_round_lot,
        max_tickers=max_tickers,
    )
    console.print(f"[bold yellow]Fetched {len(tickers):,} U.S.-listed tickers after filtering[/bold yellow]")
    records, failed_tickers, sector_industry_counts = fetch_yahoo_metadata(tickers, key_importance)
    df_full = pd.DataFrame.from_records(records)
    df_full_formatted = format_dataframe_for_csv(df_full)
    df_full_formatted.to_csv("yahoo_company_info.csv", index=False)
    summary = build_sector_industry_summary(df_full, n_examples=3)
    summary.to_csv("yahoo_sector_industry_summary.csv", index=False)
    # Print stats
    total_loaded = len(records)
    total_requested = len(tickers)
    percent_loaded = 100.0 * total_loaded / total_requested if total_requested else 0
    console.print(
        Panel(
            f"[bold green]Loaded {total_loaded:,} / {total_requested:,} tickers ({percent_loaded:.2f}%) successfully.[/bold green]",
            title="[bold cyan]Overall Load Success",
        )
    )
    # Per sector/industry
    table = Table(title="Per Sector/Industry Load Stats", show_lines=True)
    table.add_column("Sector")
    table.add_column("Industry")
    table.add_column("Requested", justify="right")
    table.add_column("Loaded", justify="right")
    table.add_column("Failed", justify="right")
    for (sector, industry), stats in sector_industry_counts.items():
        table.add_row(
            str(sector),
            str(industry),
            str(stats["requested"]),
            str(stats["loaded"]),
            str(stats["failed"]),
        )
    console.print(table)
    print_sector_industry_table(summary)
    console.print("\n✔ Done!  Files written:")
    for f in Path(".").glob("*sector_industry*.csv"):
        console.print(f"  • [bold green]{f.resolve()}[/bold green]")
    return summary
