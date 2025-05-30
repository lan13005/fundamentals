from __future__ import annotations

import asyncio
import datetime as dt
import json
import pathlib
import re
import warnings
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import yfinance as yf
from aiohttp import ClientTimeout
from rich.console import Console
from rich.table import Table

from fundamentals.utility.general import get_latest_quarter_end

warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape")
console = Console()

# ───────────── configuration ───────────── #
DATA_DIR = pathlib.Path("macro_data/parquet")
UTIL_DIR = pathlib.Path("fundamentals/utility")
OVERRIDE_FILE = UTIL_DIR / "slug_overrides.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UTIL_DIR.mkdir(parents=True, exist_ok=True)

URL_TMPL = "https://www.macrotrends.net/stocks/charts/{sym}/{slug}/{page}?freq={freq}"

_RE_ORIG = re.compile(r"originalData\s*=\s*(\[\{.*?\}\]);", re.S)
_RE_CLEAN = re.compile(r"<.*?>")
_RE_NON = re.compile(r"^[^a-zA-Z]*|[^a-zA-Z]*$")

DEFAULT_PAGES = ["income-statement", "balance-sheet", "cash-flow-statement", "financial-ratios"]

# ───────────── overrides I/O ───────────── #
def load_file_overrides() -> dict[str, str]:
    if OVERRIDE_FILE.exists():
        try:
            with OVERRIDE_FILE.open() as f:
                return {k.upper(): v for k, v in json.load(f).items()}
        except Exception as e:
            console.print(f"[red]Warning:[/] could not read {OVERRIDE_FILE}: {e}")
    return {}


def save_file_overrides(mapping: dict[str, str]) -> None:
    try:
        with OVERRIDE_FILE.open("w") as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        console.print(f"[red]Warning:[/] could not write {OVERRIDE_FILE}: {e}")


FILE_OVERRIDES = load_file_overrides()


# ───────────── helper funcs ───────────── #
def strip_non_letters(s: str) -> str:
    return _RE_NON.sub("", s)


def derive_slug(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        base = info.get("displayName") or info.get("shortName") or ticker
    except Exception:
        base = ticker
    return strip_non_letters(base.lower().replace(" ", "-"))


def parse_original_data(html: str) -> pd.DataFrame:
    m = _RE_ORIG.search(html)
    if not m:
        raise ValueError("originalData not found")
    df = pd.DataFrame(eval(m.group(1)))
    if "field_name" in df.columns:
        df["field_name"] = (
            df["field_name"].str.replace(_RE_CLEAN, "", regex=True).str.replace(r"\\/", "/", regex=True).str.strip()
        )
    return df.drop(columns=["popup_icon"], errors="ignore")


def clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns are safe for Parquet output by converting mixed or problematic types.
    Then transform the dataframe structure with proper datatypes:
    - Transpose with field_name as columns and dates as index
    - Convert numeric columns to float with NaN filled as 0
    - Convert date column to datetime
    """
    df_clean = df.copy()
    # Initial cleaning for mixed/problematic types
    for col in df_clean.columns:
        col_data = df_clean[col]
        # Convert bytes columns to string
        if col_data.dtype == object and any(isinstance(x, bytes) for x in col_data.dropna()):
            df_clean[col] = col_data.apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        # Convert mixed types to string
        elif col_data.dtype == object and not all(isinstance(x, str) or pd.isna(x) for x in col_data.dropna()):
            df_clean[col] = col_data.astype(str)
        # Convert float columns to nullable float
        elif pd.api.types.is_float_dtype(col_data):
            df_clean[col] = pd.to_numeric(col_data, errors="coerce")
        # Convert columns with all NaN to string
        elif col_data.isnull().all():
            df_clean[col] = col_data.astype(str)
    return df_clean

def finalize_merged_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Finalize the merged dataframe by:
    - Transposing the dataframe to have features as columns rather than datetimes
    - Converting all non-date columns to numeric
    - Converting the date column to datetime
    - Fetching raw price & dividends, resampling to quarter-end
    - Aligning all datasets
    - Calculating additional annualized features
        - EPS (+derived features) uses 3yr rolling average
    """
    # Finally, transform the dataframe to have features as columns rather than datetimes
    if 'field_name' in df.columns:
        df = df.set_index('field_name').T
        df.index.name = "date"
        df = df.reset_index()

        # Convert all non-date columns to numeric
        # Filling NaNs with 0s makes sense for this type of data
        for col in df.columns:
            if col == 'date':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

    # 1. Set up yfinance and date range
    yt     = yf.Ticker(ticker)
    start  = df['date'].min()
    end    = df['date'].max() + pd.Timedelta(days=1) # offset to include the last day

    # 2. Fetch raw price & dividends
    prices = yt.history(start=start, end=end)[['Close']].rename(columns={'Close':'Price'})
    divs   = yt.dividends.rename("Dividends Per Share")

    # 3. Resample to quarter-end
    price_q = prices['Price'].resample('QE').last()
    price_q.index = price_q.index.tz_localize(None)           # drop tz-info
    price_q = price_q.loc[start:end]

    div_q = divs.resample('QE').sum()
    if div_q.index.tz is not None:
        div_q.index = div_q.index.tz_localize(None)
    div_q = div_q.loc[start:end].fillna(0)                    # fill missing quarters with 0

    # Align all datasets, print a remark on any alignment issues like data chopped off
    new_start = max(start, div_q.index.min(), price_q.index.min())
    new_end   = min(end, div_q.index.max(), price_q.index.max()) + pd.Timedelta(days=1)
    price_q = price_q.loc[new_start:new_end]
    div_q   = div_q.loc[new_start:new_end]
    df = df[(df['date'] > new_start) * (df['date'] < new_end)]

    # # 4. Align back to your df dates
    df = df.set_index('date').sort_index()
    df['Price']               = price_q.reindex(df.index, method='ffill')
    df['Dividends Per Share'] = div_q.reindex(df.index).fillna(0)       # already 0 where missing

    df['NWC']                  = df['Total Current Assets'] - df['Total Current Liabilities']
    df['Net Fixed Assets']     = df['Property, Plant, And Equipment']
    df['Capital Employed']     = df['NWC'] + df['Net Fixed Assets']
    df['ROCE']                 = df['Operating Income'] / df['Capital Employed']

    # Statutory tax rates changed in 2017 from 35% to 21%
    #   This rate does not include state taxes, credits, international income, etc.
    stat_rates = pd.Series(
        df.index.year.map(lambda y: 0.35 if y < 2018 else 0.21),
        index=df.index
    )

    # Growth metrics
    df['Revenue YoY'] = df['Revenue'].pct_change(periods=4)

    # NOTE: our data is sorted with latest date first. Need to reverse, then roll, then reverse again
    df['Taxes LTM']   = df['Income Taxes'].rolling(window=4).sum()
    df['PreTax LTM']  = df['Pre-Tax Income'].rolling(window=4).sum()
    df['ETR LTM']     = (df['Taxes LTM'] / df['PreTax LTM']).clip(0,1).ffill().fillna(stat_rates)
    df['NOPAT LTM']   = df['Operating Income'].rolling(window=4).sum() * (1 - df['ETR LTM'])
    df['Capital Employed Avg'] = df['Capital Employed'].rolling(window=4).mean() # average of 4 quarter-ends
    df['ROIC LTM']   = df['NOPAT LTM'] / df['Capital Employed Avg'] # annualized ROIC

    df['Total Debt']           = df['Long Term Debt'] + df['Net Current Debt']
    df['Debt to Equity']       = df['Total Debt'] / df['Share Holder Equity']
    df['Equity to Assets']     = df['Share Holder Equity'] / df['Total Assets']
    df['Earnings LTM']         = df['Net Income'].rolling(window=4).sum()
    df['Avg Earnings 3y']      = df['Earnings LTM'].rolling(window=3).mean() # backward rolling window
    df['FCF']                  = (
        df['Cash Flow From Operating Activities']
    + df['Net Change In Property, Plant, And Equipment']
    + df['Net Change In Intangible Assets']
    )
    df['FCF Margin']     = df['FCF'] / df['Revenue']
    df['FCF LTM']        = df['FCF'].rolling(window=4).sum()
    df['FCF Margin LTM'] = df['FCF'].rolling(window=4).sum() / df['Revenue'].rolling(window=4).sum()

    df['Market Cap']           = df['Price'] * df['Shares Outstanding']
    df['EPS 3y']               = df['Avg Earnings 3y'] / df['Shares Outstanding']
    df['PE Ratio']             = df['Price'] / df['EPS 3y']
    df['BV per share']         = df['Share Holder Equity'] / df['Shares Outstanding']
    df['PB Ratio']             = df['Price'] / df['BV per share']
    df['BV to Tangible Assets']= (
        (df['Share Holder Equity'] - df['Goodwill And Intangible Assets'])
        / df['Total Assets']
    )
    df['Enterprise Value']     = df['Market Cap'] + df['Total Debt'] - df['Cash On Hand']
    df['EV to EBITDA']         = df['Enterprise Value'] / df['EBITDA']
    df['Dividend Yield LTM']   = df['Dividends Per Share'].rolling(window=4).sum()
    df['Dividend Yield']       = df['Dividend Yield LTM'] / df['Price']
    df['FCF LTM']              = df['FCF'].rolling(window=4).sum()
    df['FCF Yield LTM']        = df['FCF LTM'] / df['Enterprise Value']

    # Total return over 5 years using Compound Annual Growth Rate
    df['TR Factor 5y'] = (df['Price'] + df['Dividends Per Share'].rolling(window=4*5).sum()) / df['Price'].shift(4*5)
    df['TR CAGR 5y']  = df['TR Factor 5y'] ** (1/5) - 1
    
    # Everything at this point should be space-separated
    #   Switch to kebab-case for column names which is easier to reference
    df.columns = df.columns.str.replace(",", " ")
    df.columns = df.columns.str.replace("/", " ")
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    df.columns = df.columns.str.replace(".", "-")
    df.columns = df.columns.str.replace(" ", "-")
    df.columns = df.columns.str.replace("-{2,}", "-", regex=True)

    return df


def merged_parquet_name(sym: str, snap_date: dt.date) -> pathlib.Path:
    """Return the merged parquet file name for a symbol and quarter."""
    return DATA_DIR / f"{sym.upper()}_{snap_date.isoformat()}.parquet"


# ───────────── async scraping ───────────── #
async def fetch_table(
    session: aiohttp.ClientSession,
    sym: str,
    slug: str,
    page: str,
    snap_date: dt.date,
    freq: str,
    force: bool,
    mismatches: list[Tuple[str, str]],
    overrides: dict[str, str],
) -> pd.DataFrame:
    """Fetch a single page and return its cleaned dataframe (no file writing)."""
    url = URL_TMPL.format(sym=sym, slug=slug, page=page, freq=freq)
    console.print(f"[yellow]WEB [/yellow]{url}")
    async with session.get(url, allow_redirects=True) as r:
        html = await r.text()
        final = str(r.url)

    # detect redirect / extract correct slug
    if f"freq={freq}" not in final:
        correct_slug = final.split("/")[6]  # …/TICKER/<slug>/<page>?freq=A
        mismatches.append((sym, slug, correct_slug))
        overrides[sym.upper()] = correct_slug
        console.print(
            f"[red]Redirected[/red] → new slug '[bold]{correct_slug}[/bold]'."
            " Annual data was returned; rerun to fetch quarterly."
        )

    try:
        df = parse_original_data(html)
        df_clean = clean_dataframe_for_parquet(df)
        return df_clean
    except Exception as e:
        console.print(f"[red]Error fetching data for {sym} {page}: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return pd.DataFrame()


async def scrape_many(
    symbols: List[Tuple[str, str]],
    pages: Iterable[str],
    snap_date: dt.date,
    freq: str,
    force: bool,
    mismatches: list[Tuple[str, str]],
    overrides: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Fetch all pages for all symbols and return a dict of symbol: merged dataframe."""
    results = {}
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=20)) as sess:
        for s, slug in symbols:
            dfs = []
            for p in pages:
                df = await fetch_table(sess, s, slug, p, snap_date, freq, force, mismatches, overrides)
                if not df.empty:
                    dfs.append(df)
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                merged = finalize_merged_dataframe(merged, s)
                results[s] = merged
    return results


def build_symbol_list(symbols: List[str], slug_map: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """Build a list of (symbol, slug) tuples using override mappings.

    Args:
        symbols: List of ticker symbols to process
        slug_map: Optional dictionary of ticker:slug mappings to override defaults

    Returns:
        List of (symbol, slug) tuples for processing
    """
    cli_overrides = {k.upper(): v for k, v in (slug_map or {}).items()}
    merged = {**FILE_OVERRIDES, **cli_overrides}
    return [(sym, merged.get(sym.upper(), derive_slug(sym))) for sym in symbols]


def run_macrotrends_scraper(
    symbols: List[str],
    slug_map: Optional[Dict[str, str]] = None,
    freq: str = "Q",
    force: bool = False,
    date: Optional[str] = None,
) -> pd.DataFrame:
    """Run the Macrotrends scraper to fetch financial data for all DEFAULT_PAGES and return a single merged DataFrame per symbol. Writes only one merged parquet file per symbol per quarter."""
    pages = DEFAULT_PAGES
    if date:
        snap_date = dt.date.fromisoformat(date)
    else:
        snap_date = get_latest_quarter_end()
    symbols_list = build_symbol_list(symbols, slug_map)

    mismatches: list[Tuple[str, str]] = []
    overrides_updated = FILE_OVERRIDES.copy()

    # Fetch all dataframes for all symbols
    results = asyncio.run(scrape_many(symbols_list, pages, snap_date, freq, force, mismatches, overrides_updated))

    # persist any new overrides discovered
    if overrides_updated != FILE_OVERRIDES:
        save_file_overrides(overrides_updated)
        console.print("[bold magenta]↺ slug_overrides.json updated.[/bold magenta]")

    # pretty-print mismatches
    if mismatches:
        t = Table(title="Redirects (annual data fetched)")
        t.add_column("Ticker", style="red")
        t.add_column("Old slug")
        t.add_column("New slug (saved)")
        for sym, old, new in mismatches:
            t.add_row(sym, old, new)
        console.print(t)
        console.print(
            "\n[yellow]⚠ Because the site redirected, annual data were downloaded.[/yellow]\n"
            "   [yellow]Rerun the program to obtain quarterly data with the corrected slug.[/yellow]\n"
        )

    # Write merged parquet files and return merged dataframe
    all_dfs = []
    for sym, df in results.items():
        if not df.empty:
            fn = merged_parquet_name(sym, snap_date)
            clean_dataframe_for_parquet(df).to_parquet(fn, compression="snappy")
            console.print(f"[cyan]SAVE[/cyan] {fn.name}")
            all_dfs.append(df)
    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        console.print(f"[bold green]✓ Loaded {len(result)} rows for {len(symbols)} symbol(s) for {snap_date}.[/bold green]")
        return result
    else:
        console.print("[red]No dataframes loaded.[/red]")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True, metavar="TICKER")
    parser.add_argument("--slug-map", nargs="+", default=[], metavar="TICKER:slug")
    parser.add_argument("--freq", choices=["Q", "A"], default="Q")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD")

    args = parser.parse_args()
    slug_map_dict = dict(pair.split(":", 1) for pair in args.slug_map) if args.slug_map else None

    df = run_macrotrends_scraper(
        symbols=args.symbols, slug_map=slug_map_dict, freq=args.freq, force=args.force, date=args.date
    )
    if not df.empty:
        console.print(df.head())

        console.print(df.head())
