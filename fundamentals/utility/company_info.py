import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from fundamentals.utility.general import get_latest_quarter_end

console = Console()

def get_company_info(
    tickers: list[str] = None,
    file_name: str = None,
    use_sp500: bool = True,
    output_dir: str = "macro_data"
) -> pd.DataFrame:
    """
    Get company information using yfinance for specified tickers or S&P 500.

    Args:
        tickers: List of ticker symbols. If None and use_sp500=True, fetches S&P 500 tickers.
        file_name: Name for the output file (without extension). Required if tickers is provided.
        use_sp500: Whether to use S&P 500 tickers from Wikipedia (default: True).
        output_dir: Directory to save the parquet file (default: "macro_data").

    Returns:
        pd.DataFrame: DataFrame containing company information.

    Raises:
        ValueError: If tickers are provided but file_name is not specified.
    """
    # Validate inputs
    if tickers is not None and file_name is None:
        raise ValueError("file_name must be provided when using custom tickers")

    # Get tickers
    if tickers is not None:
        # Custom tickers provided
        use_sp500 = False
        console.print(f"[blue]Using provided tickers: {tickers}[/blue]")
    elif use_sp500:
        console.print("[blue]Fetching S&P 500 tickers from Wikipedia...[/blue]")
        # Grab the current S&P 500 tickers from Wikipedia
        wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
        table = BeautifulSoup(wiki, "lxml").find("table", {"id":"constituents"})
        tickers = [row.find_all("td")[0].text.strip() for row in table.find_all("tr")[1:]]
        tickers = [t.replace('.', '-') for t in tickers]
        file_name = "SP500"
        console.print(f"[green]Found {len(tickers)} S&P 500 tickers[/green]")
    else:
        raise ValueError("Either use_sp500=True or provide custom tickers")

    # Define fields to extract
    fields = [
        # Corporate identifiers & profile
        "shortName",
        "city",
        "state",
        "country",
        "website",
        "sectorKey",
        "industryKey",
        "longBusinessSummary",

        # Governance & risk metrics (all on a 1-10 scale; higher means greater risk)
        # Interestingly, if you look at the following risk metrics for S&P 500 companies
        #   you will notice that there are ~50 companies in each bin. This suggests that
        #   the score system could be using percentile rankings.
        "auditRisk",               # Audit-committee & accounting oversight score; high values can foreshadow restatements or weak controls
        "boardRisk",               # Board independence / diversity score; strong boards improve capital-allocation discipline
        "compensationRisk",        # Pay-for-performance alignment; mis-aligned incentives erode long-term value
        "shareHolderRightsRisk",   # Minority-rights protection (one-share-one-vote, no poison pill); low risk limits dilution events
        "overallRisk",             # ISS QualityScore decile rank (1 best, 10 worst); quick proxy for governance quality and scandal risk

        # Timestamps for governance data (seconds since UNIX epoch)
        "governanceEpochDate",         # when governance metrics were last updated
        "compensationAsOfEpochDate",   # as-of date for compensation data

        # Leadership & IR
        "companyOfficers", # officer's maxAge can be wrong
        "irWebsite",       # investor relations website

        # Ownership & analyst consensus
        "heldPercentInstitutions",     # Percent of shares held by institutions; interestingly this value can be greater than 1
        "heldPercentInsiders",         # Percent of shares held by insiders
        "recommendationKey",           # e.g. 'buy', 'hold', 'sell'
        "recommendationMean",          # average analyst recommendation (numeric, 1=Strong Buy…5=Strong Sell)
        "averageAnalystRating",        # string like '2.1 - Buy'; numeric part on 1-5 scale
        "numberOfAnalystOpinions",     # Number of analysts covering the stock
        "targetLowPrice",              # Lowest analyst target price
        "targetMeanPrice",             # Average analyst target price
        "targetHighPrice",             # Highest analyst target price
        "targetMedianPrice",           # Median analyst target price

        # Capital structure & normalized profitability
        "debtToEquity",     # Leverage ratio; <1 preferred for sleep-at-night safety
        "totalDebt",        # Absolute leverage gauge; interpret with debt-to-equity
        "totalCash",        # Liquidity buffer in absolute terms
        "enterpriseValue",  # Total operating valuation incl. debt & cash; cross-capital-structure metric
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "profitMargins",    # Bottom-line (net) profitability; confirms that revenue converts to cash
        "grossMargins",     # Up-stream pricing power and cost moat; early warning of eroding advantage
        "ebitdaMargins",    # Cash-flow proxy margin; less accounting noise than net margin
        "operatingMargins", # Captures operating-level pricing power before unusuals; stability is key
        "returnOnAssets",   # Balance-sheet-agnostic profitability; useful cross-sector check on ROE
        "returnOnEquity",   # Core measure of capital efficiency; >15 % across cycles signals durable competitive advantage
        "dividendYield",    # Shareholder pay-out today; combine with payoutRatio for sustainability
        "dividendRate",     # Dividend yield; annualized payout as a percentage of price
        "payoutRatio",      # Earnings share returned to owners; <60 % gives reinvestment headroom
        "fiveYearAvgDividendYield", # History of income return; smooths one-off spikes/cuts
        "beta",             # Historical volatility vs. market; helps size positions within a momentum overlay
    ]

    console.print(f"[blue]Fetching company info from yfinance for {len(tickers)} tickers...[/blue]")

    # Fetch data using yfinance
    bundle = yf.Tickers(" ".join(tickers))

    tickers_info = defaultdict(list)

    # Process tickers with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing tickers...", total=len(tickers))

        for sym in tickers:
            if sym not in tickers_info:
                tickers_info['ticker'].append(sym)
            for field in fields:
                value = np.nan
                if field in bundle.tickers[sym].info:
                    value = bundle.tickers[sym].info[field]
                tickers_info[field].append(value)

            progress.advance(task)

    # Create DataFrame
    df = pd.DataFrame(tickers_info)

    # Save to parquet file
    os.makedirs(output_dir, exist_ok=True)
    date_str = get_latest_quarter_end().strftime("%Y-%m-%d")
    output_file = os.path.join(output_dir, f"{file_name}-{date_str}.parquet")

    df.to_parquet(output_file, index=False)
    console.print(f"[green]Company info saved to: {output_file}[/green]")
    console.print(f"[cyan]DataFrame shape: {df.shape}[/cyan]")

    return df
