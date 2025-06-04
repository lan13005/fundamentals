from __future__ import annotations

import asyncio
import datetime as dt
import json
import pathlib
import random
import re
import time
import warnings
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf
from aiohttp import ClientTimeout
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from fundamentals.utility.general import get_latest_quarter_end, get_nasdaq_tickers, get_sp500_tickers
from fundamentals.utility.wacc import compute_rolling_beta, get_historical_erp, get_risk_free_rate

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

# ───────────── safety configuration ───────────── #
MAX_CONCURRENT_REQUESTS = 3  # Conservative limit for macrotrends
REQUEST_DELAY = 0.5  # 500ms between requests (will be replaced by token bucket)
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential backoff multiplier
JITTER_PERCENT = 0.1  # ±10% random jitter for retries
PAGE_DELAY = 0.2  # Base delay between pages for same symbol
PAGE_JITTER_PERCENT = 0.1  # ±10% jitter for page delays

# Token bucket configuration
TOKEN_BUCKET_CAPACITY = 5  # Burst capacity
TOKEN_REFILL_RATE = 2  # Tokens per second
INITIAL_SEMAPHORE_SIZE = 3  # Starting semaphore size (reduced from 10)
MIN_SEMAPHORE_SIZE = 1  # Minimum semaphore size (reduced from 2)
MAX_SEMAPHORE_SIZE = 8  # Maximum semaphore size (reduced from 20)

# Headers to be respectful
HEADERS = {
    "User-Agent": "fundamentals-scraper/1.0 (Financial Research Tool)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ───────────── Token Bucket Rate Limiter ───────────── #
class TokenBucket:
    """Enhanced token bucket rate limiter with adaptive behavior."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = min(capacity // 2, 2)  # Start with fewer tokens to be more conservative
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
        self._rate_limit_backoff = 1.0  # Exponential backoff multiplier for rate limits
        self._last_rate_limit = 0

    async def consume(self, tokens: int = 1) -> None:
        """Consume tokens from the bucket, waiting if necessary."""
        async with self._lock:
            now = time.time()

            # Apply exponential backoff if we've been rate limited recently
            if now - self._last_rate_limit < 60:  # Within last minute
                backoff_delay = self._rate_limit_backoff * 0.5  # Extra delay
                await asyncio.sleep(backoff_delay)
                console.print(f"[cyan]Token bucket backoff: {backoff_delay:.1f}s[/cyan]")

            # Add tokens based on time elapsed
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now

            # Wait if we don't have enough tokens
            while self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
                now = time.time()
                time_passed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
                self.last_refill = now

            self.tokens -= tokens

    async def record_rate_limit(self):
        """Record that we hit a rate limit and increase backoff."""
        async with self._lock:
            self._last_rate_limit = time.time()
            self._rate_limit_backoff = min(
                8.0, self._rate_limit_backoff * 1.5
            )  # Exponential backoff, max 8x (increased from 4x)
            # Reduce tokens on rate limit to be more conservative
            self.tokens = max(0, self.tokens - 2)
            console.print(f"[yellow]Token bucket: increased backoff to {self._rate_limit_backoff:.1f}x[/yellow]")

    async def record_success(self):
        """Record successful requests to gradually reduce backoff."""
        async with self._lock:
            if time.time() - self._last_rate_limit > 30:  # No rate limits for 30s
                self._rate_limit_backoff = max(1.0, self._rate_limit_backoff * 0.9)  # Gradually reduce


# ───────────── Adaptive Semaphore ───────────── #
class AdaptiveSemaphore:
    """Adaptive semaphore that aggressively adjusts based on 429 response rates."""

    def __init__(self, initial_size: int, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
        self._semaphore = asyncio.Semaphore(initial_size)
        self._current_size = initial_size
        self._requests_made = 0
        self._rate_limited_responses = 0
        self._consecutive_rate_limits = 0
        self._last_rate_limit_time = 0
        self._circuit_breaker_until = 0
        self._lock = asyncio.Lock()
        self._adaptation_threshold = 5  # Check every 5 requests instead of 20

    async def acquire(self):
        """Acquire the semaphore, with circuit breaker check."""
        # Circuit breaker: if we've had too many rate limits recently, force a delay
        now = time.time()
        if now < self._circuit_breaker_until:
            wait_time = self._circuit_breaker_until - now
            console.print(f"[red]Circuit breaker active: waiting {wait_time:.1f}s[/red]")
            await asyncio.sleep(wait_time)

        await self._semaphore.acquire()

    def release(self):
        """Release the semaphore."""
        self._semaphore.release()

    async def record_request(self, was_rate_limited: bool = False):
        """Record a request and aggressively adjust semaphore size."""
        async with self._lock:
            self._requests_made += 1
            now = time.time()

            if was_rate_limited:
                self._rate_limited_responses += 1
                self._consecutive_rate_limits += 1
                self._last_rate_limit_time = now

                # Immediate aggressive reduction on rate limit
                if self._consecutive_rate_limits == 1:
                    # First rate limit: reduce by 40%
                    new_size = max(self.min_size, int(self._current_size * 0.6))
                    await self._resize_semaphore(new_size)
                    console.print(f"[yellow]Rate limit detected! Reduced semaphore to {new_size} (-40%)[/yellow]")

                elif self._consecutive_rate_limits >= 2:
                    # Multiple consecutive rate limits: activate circuit breaker
                    new_size = self.min_size
                    await self._resize_semaphore(new_size)
                    # Circuit breaker: force a pause based on consecutive rate limits
                    breaker_time = min(30, 5 * self._consecutive_rate_limits)  # 5s per consecutive, max 30s
                    self._circuit_breaker_until = now + breaker_time
                    console.print(
                        f"[red]Multiple rate limits! Circuit breaker: {breaker_time}s pause, semaphore: {new_size}[/red]"
                    )

            else:
                # Reset consecutive counter on successful request
                self._consecutive_rate_limits = 0

            # Adaptive adjustment every few requests
            if self._requests_made % self._adaptation_threshold == 0:
                rate_limited_fraction = self._rate_limited_responses / self._requests_made

                if rate_limited_fraction > 0.05:  # More than 5% rate limited (more aggressive)
                    new_size = max(self.min_size, int(self._current_size * 0.7))  # Reduce by 30%
                    if new_size != self._current_size:
                        await self._resize_semaphore(new_size)
                        console.print(
                            f"[yellow]High rate limit rate ({rate_limited_fraction:.1%}): reduced semaphore to {new_size}[/yellow]"
                        )

                elif rate_limited_fraction < 0.01 and (now - self._last_rate_limit_time) > 60:
                    # Only increase if no rate limits for 60+ seconds and very low rate
                    new_size = min(self.max_size, int(self._current_size * 1.2))  # Increase by 20%
                    if new_size != self._current_size:
                        await self._resize_semaphore(new_size)
                        console.print(
                            f"[green]Low rate limit rate ({rate_limited_fraction:.1%}): increased semaphore to {new_size}[/green]"
                        )

                # Reset counters but keep recent history
                if self._requests_made >= 50:  # Reset every 50 requests to keep adaptation responsive
                    self._requests_made = self._adaptation_threshold
                    self._rate_limited_responses = max(
                        0, int(self._rate_limited_responses * 0.3)
                    )  # Keep 30% of history

    async def _resize_semaphore(self, new_size: int):
        """Resize the semaphore by creating a new one."""
        self._semaphore = asyncio.Semaphore(new_size)
        self._current_size = new_size

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


def add_jitter(delay: float, jitter_percent: float = JITTER_PERCENT) -> float:
    """Add random jitter to a delay value."""
    jitter = delay * jitter_percent * (2 * random.random() - 1)  # ±jitter_percent
    return max(0, delay + jitter)


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


# ───────────── WACC calculation functions ───────────── #


def calculate_wacc_components(df: pd.DataFrame, ticker: str, verbose: bool = False) -> pd.DataFrame:
    """
    Calculate WACC components and add them to the existing dataframe.

    Args:
        df (pd.DataFrame): Dataframe with financial data (with datetime index)
        ticker (str): Stock ticker symbol
        verbose (bool): Whether to print debug information

    Returns:
        pd.DataFrame: Dataframe with WACC components added ['Rf', 'beta', 'ERP', 'R-e', 'WACC']
    """
    if df.empty:
        console.print(f"[yellow]Warning: Empty dataframe for WACC calculation for {ticker}[/yellow]")
        return df

    try:
        # Get date range from dataframe
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")

        if verbose:
            console.print(f"[cyan]Calculating WACC components for {ticker} from {start_date} to {end_date}[/cyan]")

        # Get risk-free rate
        rf_quarterly = get_risk_free_rate(start_date, end_date, verbose)

        # Compute rolling beta
        beta_quarterly = compute_rolling_beta(ticker, rf_quarterly, start_date, end_date, verbose=verbose)

        # Get historical ERP
        erp_quarterly = get_historical_erp(start_date, end_date, verbose)

        # Add WACC components to dataframe by aligning with existing index
        df_wacc = df.copy()

        # Align data with existing dataframe index
        df_wacc["Risk-Free-Rate"] = rf_quarterly.reindex(df_wacc.index, method="ffill")
        df_wacc["beta"] = beta_quarterly.reindex(df_wacc.index, method="ffill")
        df_wacc["Equity-Risk-Premium"] = erp_quarterly.reindex(df_wacc.index, method="ffill")

        # Fill any remaining NaN values with reasonable defaults
        df_wacc["Risk-Free-Rate"] = df_wacc["Risk-Free-Rate"].fillna(0.025)  # 2.5% default risk-free rate
        df_wacc["beta"] = df_wacc["beta"].fillna(1.0)  # Default beta of 1.0
        df_wacc["Equity-Risk-Premium"] = df_wacc["Equity-Risk-Premium"].fillna(0.055)  # 5.5% default ERP

        # Calculate cost of equity
        df_wacc["Cost-of-Equity"] = df_wacc["Risk-Free-Rate"] + df_wacc["beta"] * df_wacc["Equity-Risk-Premium"]

        # Map column names from kebab-case to the expected format for WACC calculations
        # Need to handle both Total-Debt components and Interest-Expense
        total_debt = df_wacc.get("Long-Term-Debt", 0) + df_wacc.get("Net-Current-Debt", 0)
        market_cap = df_wacc.get("Market-Cap", 0)

        # Calculate interest expense from available columns
        # Look for Interest Expense or similar columns in original names before kebab conversion
        interest_expense = 0
        for col in df_wacc.columns:
            if "interest" in col.lower() and "expense" in col.lower():
                interest_expense = df_wacc[col]
                break

        # If no interest expense found, use 0 (companies with no debt)
        if isinstance(interest_expense, int | float) and interest_expense == 0:
            interest_expense = pd.Series(0, index=df_wacc.index)

        # Calculate cost of debt components
        debt_avg = (total_debt.shift(1) + total_debt) / 2

        # Calculate R_d_pre_tax, but fill with 0 if no debt or interest expense
        r_d_pre_tax = np.where(
            (debt_avg > 0) & (interest_expense.notna()) & (interest_expense > 0),
            interest_expense / debt_avg,
            0.0,  # Zero cost if no debt or no interest expense
        )

        # Calculate tax rate from Pre-Tax-Income and Income-Taxes
        pre_tax_income = df_wacc.get("Pre-Tax-Income", 0)
        income_taxes = df_wacc.get("Income-Taxes", 0)

        tax_rate = np.where(
            (pre_tax_income != 0) & (pre_tax_income.notna()) & (income_taxes.notna()),
            income_taxes / pre_tax_income,
            0.21,  # Use standard corporate tax rate as fallback
        )

        # Ensure tax rate is between 0 and 1
        tax_rate = np.clip(tax_rate, 0, 1)

        # After-tax cost of debt (will be 0 if r_d_pre_tax is 0)
        r_d = r_d_pre_tax * (1 - tax_rate)

        # Add cost of debt to dataframe
        df_wacc["Cost-of-Debt"] = r_d

        # Capital structure weights
        market_value_equity = market_cap.fillna(0.0)
        market_value_debt = total_debt.fillna(0.0)
        total_capital = market_value_equity + market_value_debt

        # Calculate weights with safe division
        total_capital_safe = total_capital.replace(0, np.nan)

        weight_equity = np.where(
            total_capital > 0, market_value_equity / total_capital_safe, 1.0  # 100% equity if no total capital data
        )

        weight_debt = np.where(
            total_capital > 0, market_value_debt / total_capital_safe, 0.0  # 0% debt if no total capital data
        )

        # Calculate WACC
        wacc = np.where(
            df_wacc["Cost-of-Equity"].notna() & (total_capital > 0),
            weight_equity * df_wacc["Cost-of-Equity"] + weight_debt * df_wacc["Cost-of-Debt"],
            np.nan,
        )

        df_wacc["WACC"] = wacc

        if verbose:
            non_null_wacc = pd.Series(wacc).dropna()
            if len(non_null_wacc) > 0:
                console.print(f"[green]✓ Calculated WACC for {len(non_null_wacc)} quarters for {ticker}[/green]")
                console.print(
                    f"[cyan]Average WACC: {non_null_wacc.mean():.4f} ({non_null_wacc.mean()*100:.2f}%)[/cyan]"
                )
            else:
                console.print(f"[yellow]Warning: No valid WACC calculations for {ticker}[/yellow]")

        return df_wacc

    except Exception as e:
        console.print(f"[red]Error calculating WACC components for {ticker}: {e}[/red]")
        import traceback

        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return df


def finalize_merged_dataframe(df: pd.DataFrame, ticker: str, include_wacc: bool = True) -> pd.DataFrame:
    """
    Finalize the merged dataframe by:
    - Transposing the dataframe to have features as columns rather than datetimes
    - Converting all non-date columns to numeric
    - Converting the date column to datetime
    - Fetching raw price & dividends, resampling to quarter-end
    - Aligning all datasets
    - Calculating additional annualized features
        - EPS (+derived features) uses 3yr rolling average
    - Optionally calculating WACC components (Rf, beta, ERP, R_e, WACC)

    Args:
        df (pd.DataFrame): Input dataframe with financial data
        ticker (str): Stock ticker symbol
        include_wacc (bool): Whether to calculate and include WACC components

    Returns:
        pd.DataFrame: Finalized dataframe with all calculated metrics
    """
    try:
        # Check if input dataframe is empty
        if df.empty:
            console.print(f"[red]Warning: Empty input dataframe for {ticker}[/red]")
            return pd.DataFrame()

        # Finally, transform the dataframe to have features as columns rather than datetimes
        if "field_name" in df.columns:
            df = df.set_index("field_name").T
            df.index.name = "date"
            df = df.reset_index()

            # Convert all non-date columns to numeric
            # Filling NaNs with 0s makes sense for this type of data
            for col in df.columns:
                if col == "date":
                    continue
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Convert date column to datetime
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Check if we have valid dates
            if df["date"].isna().all():
                console.print(f"[red]Warning: No valid dates found for {ticker}[/red]")
                return pd.DataFrame()
        else:
            console.print(f"[red]Warning: No 'field_name' column found for {ticker}[/red]")
            return pd.DataFrame()

        # Check if we have any data after transformation
        if df.empty or len(df) == 0:
            console.print(f"[red]Warning: Empty dataframe after transformation for {ticker}[/red]")
            return pd.DataFrame()

        # Try to fetch yfinance data with error handling
        try:
            # 1. Set up yfinance and date range
            yt = yf.Ticker(ticker)
            start = df["date"].min()
            end = df["date"].max() + pd.Timedelta(days=1)  # offset to include the last day

            # 2. Fetch raw price & dividends with timeout
            try:
                prices = yt.history(start=start, end=end)[["Close"]].rename(columns={"Close": "Price"})
                divs = yt.dividends.rename("Dividends Per Share")
            except Exception as yf_error:
                console.print(f"[yellow]Warning: Failed to fetch yfinance data for {ticker}: {yf_error}[/yellow]")
                console.print("[yellow]Proceeding without price/dividend data[/yellow]")
                # Continue without yfinance data
                prices = pd.DataFrame()
                divs = pd.Series(dtype="float64", name="Dividends Per Share")

            # Only proceed with price/dividend calculations if we have the data
            if not prices.empty and len(divs) > 0:
                # 3. Resample to quarter-end
                price_q = prices["Price"].resample("QE").last()
                price_q.index = price_q.index.tz_localize(None)  # drop tz-info
                price_q = price_q.loc[start:end]

                div_q = divs.resample("QE").sum()
                if div_q.index.tz is not None:
                    div_q.index = div_q.index.tz_localize(None)
                div_q = div_q.loc[start:end].fillna(0)  # fill missing quarters with 0

                # Align all datasets, print a remark on any alignment issues like data chopped off
                new_start = max(start, div_q.index.min(), price_q.index.min())
                new_end = min(end, div_q.index.max(), price_q.index.max()) + pd.Timedelta(days=1)
                console.print(f"[cyan]Debug: Calculated alignment range: {new_start} to {new_end}[/cyan]")
                console.print(
                    f"[cyan]Debug: Financial data date range: {df['date'].min()} to {df['date'].max()}[/cyan]"
                )
                console.print(f"[cyan]Debug: Financial data before filtering: {len(df)} rows[/cyan]")

                price_q = price_q.loc[new_start:new_end]
                div_q = div_q.loc[new_start:new_end]

                # FIXED: Use flexible approach instead of restrictive filtering
                # Keep ALL financial data and add price/dividend data where available
                console.print("[yellow]Using flexible approach: keeping all financial data[/yellow]")
                df = df.set_index("date").sort_index()

                # Add price and dividend data by reindexing (fills NaN where no price data available)
                df["Price"] = price_q.reindex(df.index, method="ffill")
                df["Dividends Per Share"] = div_q.reindex(df.index).fillna(0)

                # Fill NaN prices with 0 for periods without price data
                df["Price"] = df["Price"].fillna(0)
                console.print(f"[cyan]Debug: Final shape after flexible alignment: {df.shape}[/cyan]")
            else:
                # Set df index to date and add placeholder columns
                console.print("[yellow]No price/dividend data available, using placeholders[/yellow]")
                df = df.set_index("date").sort_index()
                df["Price"] = 0.0
                df["Dividends Per Share"] = 0.0

        except Exception as yf_error:
            console.print(f"[yellow]Warning: Error processing yfinance data for {ticker}: {yf_error}[/yellow]")
            # Continue without yfinance data
            df = df.set_index("date").sort_index()
            df["Price"] = 0.0
            df["Dividends Per Share"] = 0.0

        # Check if we have required columns before proceeding with calculations
        required_cols = ["Total Current Assets", "Total Current Liabilities", "Property, Plant, And Equipment"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            console.print(f"[yellow]Warning: Missing columns for {ticker}: {missing_cols}[/yellow]")
            console.print(f"[yellow]Available columns: {list(df.columns)[:10]}... (showing first 10)[/yellow]")
            # Add missing columns with zeros to allow calculations to proceed
            for col in missing_cols:
                df[col] = 0.0

        # Safely calculate derived columns with error handling
        try:
            df["NWC"] = df.get("Total Current Assets", 0) - df.get("Total Current Liabilities", 0)
            df["Net Fixed Assets"] = df.get("Property, Plant, And Equipment", 0)
            df["Capital Employed"] = df["NWC"] + df["Net Fixed Assets"]
            df["ROCE"] = df.get("Operating Income", 0) / df["Capital Employed"].replace(0, 1)  # Avoid division by zero

            # Statutory tax rates changed in 2017 from 35% to 21%
            #   This rate does not include state taxes, credits, international income, etc.
            stat_rates = pd.Series(df.index.year.map(lambda y: 0.35 if y < 2018 else 0.21), index=df.index)

            # Growth metrics
            df["Revenue YoY"] = df.get("Revenue", pd.Series(0, index=df.index)).pct_change(periods=4)

            # NOTE: our data is sorted with latest date first. Need to reverse, then roll, then reverse again
            df["Taxes LTM"] = df.get("Income Taxes", 0).rolling(window=4).sum()
            df["PreTax LTM"] = df.get("Pre-Tax Income", 0).rolling(window=4).sum()
            df["ETR LTM"] = (df["Taxes LTM"] / df["PreTax LTM"].replace(0, 1)).clip(0, 1).ffill().fillna(stat_rates)
            df["NOPAT LTM"] = df.get("Operating Income", 0).rolling(window=4).sum() * (1 - df["ETR LTM"])
            df["Capital Employed Avg"] = df["Capital Employed"].rolling(window=4).mean()  # average of 4 quarter-ends
            df["ROIC LTM"] = df["NOPAT LTM"] / df["Capital Employed Avg"].replace(0, 1)  # annualized ROIC

            df["Total Debt"] = df.get("Long Term Debt", 0) + df.get("Net Current Debt", 0)
            df["Debt to Equity"] = df["Total Debt"] / df.get("Share Holder Equity", 1).replace(0, 1)
            df["Equity to Assets"] = df.get("Share Holder Equity", 0) / df.get("Total Assets", 1).replace(0, 1)
            df["Earnings LTM"] = df.get("Net Income", 0).rolling(window=4).sum()
            df["Avg Earnings 3y"] = df["Earnings LTM"].rolling(window=3).mean()  # backward rolling window

            cash_flow_ops = df.get("Cash Flow From Operating Activities", 0)
            ppe_change = df.get("Net Change In Property, Plant, And Equipment", 0)
            intangible_change = df.get("Net Change In Intangible Assets", 0)
            df["FCF"] = cash_flow_ops + ppe_change + intangible_change

            df["FCF Margin"] = df["FCF"] / df.get("Revenue", 1).replace(0, 1)
            df["FCF LTM"] = df["FCF"].rolling(window=4).sum()
            df["FCF Margin LTM"] = df["FCF"].rolling(window=4).sum() / df.get("Revenue", 1).rolling(
                window=4
            ).sum().replace(0, 1)

            df["Market Cap"] = df["Price"] * df.get("Shares Outstanding", 0)
            df["EPS 3y"] = df["Avg Earnings 3y"] / df.get("Shares Outstanding", 1).replace(0, 1)
            df["PE Ratio"] = df["Price"] / df["EPS 3y"].replace(0, 1)
            df["BV per share"] = df.get("Share Holder Equity", 0) / df.get("Shares Outstanding", 1).replace(0, 1)
            df["PB Ratio"] = df["Price"] / df["BV per share"].replace(0, 1)
            df["BV to Tangible Assets"] = (
                df.get("Share Holder Equity", 0) - df.get("Goodwill And Intangible Assets", 0)
            ) / df.get("Total Assets", 1).replace(0, 1)
            df["Enterprise Value"] = df["Market Cap"] + df["Total Debt"] - df.get("Cash On Hand", 0)
            df["EV to EBITDA"] = df["Enterprise Value"] / df.get("EBITDA", 1).replace(0, 1)
            df["Dividend Yield LTM"] = df["Dividends Per Share"].rolling(window=4).sum()
            df["Dividend Yield"] = df["Dividend Yield LTM"] / df["Price"].replace(0, 1)
            df["FCF LTM"] = df["FCF"].rolling(window=4).sum()
            df["FCF Yield LTM"] = df["FCF LTM"] / df["Enterprise Value"].replace(0, 1)

            # Total return over 5 years using Compound Annual Growth Rate
            df["TR Factor 5y"] = (df["Price"] + df["Dividends Per Share"].rolling(window=4 * 5).sum()) / df[
                "Price"
            ].shift(4 * 5).replace(0, 1)
            df["TR CAGR 5y"] = df["TR Factor 5y"] ** (1 / 5) - 1

        except Exception as calc_error:
            console.print(f"[yellow]Warning: Error calculating derived metrics for {ticker}: {calc_error}[/yellow]")
            # Continue with basic data if calculations fail

        # Everything at this point should be space-separated
        #   Switch to kebab-case for column names which is easier to reference
        df.columns = df.columns.str.replace(",", " ")
        df.columns = df.columns.str.replace("/", " ")
        df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
        df.columns = df.columns.str.replace(".", "-")
        df.columns = df.columns.str.replace(" ", "-")
        df.columns = df.columns.str.replace("-{2,}", "-", regex=True)

        # Calculate WACC components and add to dataframe
        if include_wacc:
            try:
                console.print(f"[cyan]Adding WACC components to {ticker}...[/cyan]")
                df = calculate_wacc_components(df, ticker, verbose=False)
                console.print(f"[green]✓ Successfully added WACC components to {ticker}[/green]")
            except Exception as wacc_error:
                console.print(f"[yellow]Warning: Error calculating WACC components for {ticker}: {wacc_error}[/yellow]")
                # Continue without WACC data - add placeholder columns with NaN
                for wacc_col in [
                    "Risk-Free-Rate",
                    "beta",
                    "Equity-Risk-Premium",
                    "Cost-of-Equity",
                    "Cost-of-Debt",
                    "WACC",
                ]:
                    df[wacc_col] = np.nan
        else:
            console.print(f"[cyan]Skipping WACC calculation for {ticker}[/cyan]")

        # Final check
        if df.empty:
            console.print(f"[red]Warning: Final dataframe is empty for {ticker}[/red]")
            return pd.DataFrame()

        console.print(f"[green]✓ Successfully processed {len(df)} rows for {ticker}[/green]")
        return df

    except Exception as e:
        console.print(f"[red]Error in finalize_merged_dataframe for {ticker}: {e}[/red]")
        import traceback

        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return pd.DataFrame()


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
    mismatches: list[Tuple[str, str, str]],  # Updated to include old slug
    overrides: dict[str, str],
    semaphore: AdaptiveSemaphore,
    token_bucket: TokenBucket,
    etag_cache: dict[str, dict],
) -> pd.DataFrame:
    """
    Fetch a single page and return its cleaned dataframe.

    If a redirect is detected (wrong slug), automatically retries with the corrected slug
    to fetch the requested frequency data instead of defaulting to annual data.

    Updates the shared overrides dictionary so subsequent calls can use the correct slug
    without redirecting.

    Includes aggressive rate limiting, exponential backoff, and circuit breaker for respectful scraping.
    """
    # Check if we have an updated slug from a previous redirect in this session
    current_slug = overrides.get(sym.upper(), slug)

    # More persistent retry logic - track 429s separately
    consecutive_429s = 0
    max_429_retries = 5  # Allow more 429-specific retries

    for attempt in range(MAX_RETRIES):
        try:
            # Consume token before making request
            await token_bucket.consume()

            # Add jittered delay between requests (except first request)
            if attempt > 0:
                base_delay = REQUEST_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                delay = add_jitter(base_delay)
                console.print(f"[yellow]Retry {attempt}/{MAX_RETRIES} for {sym}-{page} after {delay:.1f}s[/yellow]")
                await asyncio.sleep(delay)

            url = URL_TMPL.format(sym=sym.replace("-", "."), slug=current_slug, page=page, freq=freq)
            console.print(f"[yellow]WEB [/yellow]{url}")

            async with semaphore:
                async with session.get(url, headers=HEADERS, allow_redirects=True) as r:
                    # Record the request for adaptive systems
                    was_rate_limited = r.status == 429
                    await semaphore.record_request(was_rate_limited)

                    if was_rate_limited:
                        await token_bucket.record_rate_limit()
                        consecutive_429s += 1
                    else:
                        await token_bucket.record_success()
                        consecutive_429s = 0  # Reset on success

                    if r.status == 429:  # Rate limited
                        # Don't give up on 429s - be more persistent
                        if consecutive_429s <= max_429_retries:
                            # More aggressive retry_after handling with exponential backoff
                            retry_after = int(r.headers.get("Retry-After", 20))  # Increased default
                            retry_after = min(retry_after, 45)  # Increased cap

                            # Add exponential backoff based on consecutive 429s
                            backoff_multiplier = 1.8**consecutive_429s  # Increased from 1.5
                            retry_after = min(retry_after * backoff_multiplier, 120)  # Max 2 minutes

                            jittered_delay = add_jitter(retry_after, 0.2)  # 20% jitter
                            console.print(
                                f"[red]Rate limited! Waiting {jittered_delay:.1f}s (429 #{consecutive_429s}, backoff: {backoff_multiplier:.1f}x)[/red]"
                            )
                            await asyncio.sleep(jittered_delay)
                            continue
                        else:
                            console.print(
                                f"[red]Too many consecutive 429s ({consecutive_429s}) for {sym}-{page}, giving up[/red]"
                            )
                            return pd.DataFrame()

                    elif r.status >= 500:  # Server error
                        console.print(f"[red]Server error {r.status} for {sym}-{page}[/red]")
                        continue

                    elif r.status != 200:
                        console.print(f"[red]HTTP {r.status} for {sym}-{page}[/red]")
                        return pd.DataFrame()

                    html = await r.text()
                    final = str(r.url)

            # detect redirect / extract correct slug and update shared overrides
            if f"freq={freq}" not in final:
                correct_slug = final.split("/")[6]  # …/TICKER/<slug>/<page>?freq=A
                old_slug = current_slug
                mismatches.append((sym, old_slug, correct_slug))  # Track old and new slug

                # Update shared overrides immediately so other calls can use it
                overrides[sym.upper()] = correct_slug
                current_slug = correct_slug  # Update for potential retry

                console.print(
                    f"[red]Redirected[/red] → new slug '[bold]{correct_slug}[/bold]'. "
                    f"[yellow]Retrying with correct slug to fetch {freq} data...[/yellow]"
                )

                # Automatically retry with the correct slug and original frequency
                jittered_delay = add_jitter(REQUEST_DELAY * 0.5, 0.1)  # Shorter delay for redirects
                await asyncio.sleep(jittered_delay)

                # Consume another token for the retry
                await token_bucket.consume()

                retry_url = URL_TMPL.format(sym=sym.replace("-", "."), slug=correct_slug, page=page, freq=freq)
                console.print(f"[yellow]RETRY[/yellow] {retry_url}")

                # Retry with same persistence for redirects
                for redirect_attempt in range(3):  # Allow 3 attempts for redirects
                    async with semaphore:
                        async with session.get(retry_url, headers=HEADERS, allow_redirects=True) as retry_r:
                            await semaphore.record_request(retry_r.status == 429)

                            if retry_r.status == 429:
                                await token_bucket.record_rate_limit()
                                if redirect_attempt < 2:  # Don't give up immediately on 429
                                    retry_wait = add_jitter(30 * (redirect_attempt + 1), 0.2)
                                    console.print(
                                        f"[yellow]Redirect retry {redirect_attempt + 1} hit 429, waiting {retry_wait:.1f}s[/yellow]"
                                    )
                                    await asyncio.sleep(retry_wait)
                                    continue
                                else:
                                    console.print(
                                        f"[red]Redirect retry failed with HTTP 429 after {redirect_attempt + 1} attempts[/red]"
                                    )
                                    return pd.DataFrame()
                            else:
                                await token_bucket.record_success()

                            if retry_r.status != 200:
                                console.print(f"[red]Redirect retry failed with HTTP {retry_r.status}[/red]")
                                if redirect_attempt == 2:  # Last attempt
                                    return pd.DataFrame()
                                continue

                            html = await retry_r.text()
                            console.print(f"[green]✓ Successfully fetched {freq} data with corrected slug[/green]")
                            break

            # Parse the data
            df = parse_original_data(html)
            df_clean = clean_dataframe_for_parquet(df)
            return df_clean

        except aiohttp.ClientError as e:
            console.print(f"[red]Network error for {sym}-{page} (attempt {attempt + 1}): {e}[/red]")
            if attempt == MAX_RETRIES - 1:
                return pd.DataFrame()
        except Exception as e:
            console.print(f"[red]Error fetching data for {sym} {page}: {e}[/red]")
            import traceback

            console.print(traceback.format_exc())
            return pd.DataFrame()

    return pd.DataFrame()


async def scrape_many(
    symbols: List[Tuple[str, str]],
    pages: Iterable[str],
    snap_date: dt.date,
    freq: str,
    force: bool,
    mismatches: list[Tuple[str, str, str]],  # Updated type annotation
    overrides: dict[str, str],
    include_wacc: bool = True,  # New parameter for WACC calculation
) -> dict[str, pd.DataFrame]:
    """
    Fetch all pages for all symbols and return a dict of symbol: merged dataframe.

    Automatically handles redirects by retrying with corrected slugs to ensure
    the requested frequency data is fetched.

    Uses controlled concurrency, rate limiting, and sequential page fetching for respectful scraping.
    The overrides dictionary is shared across all fetch calls, so when one call discovers
    a new slug via redirect, subsequent calls for the same symbol use the correct slug.

    Saves dataframes immediately after each symbol completes to prevent data loss.
    """
    results = {}
    semaphore = AdaptiveSemaphore(INITIAL_SEMAPHORE_SIZE, MIN_SEMAPHORE_SIZE, MAX_SEMAPHORE_SIZE)
    token_bucket = TokenBucket(TOKEN_BUCKET_CAPACITY, TOKEN_REFILL_RATE)
    # Note: etag_cache kept for compatibility but not actively used anymore
    etag_cache = {}

    # Track 429 rate to adjust behavior dynamically
    recent_429s = 0
    total_requests = 0

    # Time tracking for estimated completion
    start_time = time.time()
    total_symbols = len(symbols)

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS * 2,  # Total connection pool size
        limit_per_host=MAX_CONCURRENT_REQUESTS,  # Per-host limit
        ttl_dns_cache=300,  # Cache DNS for 5 minutes
        use_dns_cache=True,
    )

    async with aiohttp.ClientSession(
        timeout=ClientTimeout(total=30, connect=10), connector=connector, headers=HEADERS
    ) as sess:
        for symbol_index, (s, slug) in enumerate(symbols, 1):
            # Calculate time estimates
            elapsed_time = time.time() - start_time
            if symbol_index > 1:  # Only calculate after first symbol
                avg_time_per_symbol = elapsed_time / (symbol_index - 1)
                remaining_symbols = total_symbols - symbol_index + 1
                estimated_remaining_time = avg_time_per_symbol * remaining_symbols

                # Format time displays
                elapsed_str = f"{elapsed_time/60:.1f}m" if elapsed_time > 60 else f"{elapsed_time:.0f}s"
                remaining_str = (
                    f"{estimated_remaining_time/60:.1f}m"
                    if estimated_remaining_time > 60
                    else f"{estimated_remaining_time:.0f}s"
                )

                console.print(
                    f"[bold blue]Processing symbol: {s}[/bold blue] "
                    f"[dim]({symbol_index}/{total_symbols}) "
                    f"Elapsed: {elapsed_str}, ETA: {remaining_str}[/dim]"
                )
            else:
                console.print(
                    f"[bold blue]Processing symbol: {s}[/bold blue] " f"[dim]({symbol_index}/{total_symbols})[/dim]"
                )

            # Dynamic rate limit adjustment based on recent 429s
            if total_requests > 10 and recent_429s / total_requests > 0.15:  # More than 15% 429s
                extra_delay = 2.0 + (recent_429s / total_requests) * 5  # 2-6s extra delay
                console.print(
                    f"[red]High 429 rate ({recent_429s}/{total_requests}): adding {extra_delay:.1f}s delay[/red]"
                )
                await asyncio.sleep(extra_delay)

            # Fetch pages sequentially for each symbol with jittered delays
            valid_dfs = []
            pages_list = list(pages)

            for i, p in enumerate(pages_list):
                # Add jittered delay between pages (except first page)
                if i > 0:
                    base_delay = PAGE_DELAY
                    # Increase delay if we've been hitting 429s
                    if recent_429s > 0:
                        base_delay *= 1 + recent_429s * 0.5  # Increase by 50% per recent 429
                    page_delay = add_jitter(base_delay, PAGE_JITTER_PERCENT)
                    await asyncio.sleep(page_delay)

                try:
                    # Use the shared overrides dict so slug updates from previous pages are used
                    df = await fetch_table(
                        sess,
                        s,
                        slug,
                        p,
                        snap_date,
                        freq,
                        force,
                        mismatches,
                        overrides,
                        semaphore,
                        token_bucket,
                        etag_cache,
                    )

                    total_requests += 1

                    if isinstance(df, Exception):
                        console.print(f"[red]Task failed for {s}-{p}: {df}[/red]")
                    elif not df.empty:
                        valid_dfs.append(df)
                        console.print(f"[green]✓ Fetched {s}-{p}[/green]")
                    else:
                        console.print(f"[yellow]Empty result for {s}-{p}[/yellow]")

                except Exception as e:
                    console.print(f"[red]Exception for {s}-{p}: {e}[/red]")
                    # Check if it was a 429-related exception
                    if "429" in str(e) or "rate" in str(e).lower():
                        recent_429s += 1

            # Process and save dataframe immediately after symbol completion
            if valid_dfs:
                console.print(f"[cyan]Debug: Found {len(valid_dfs)} valid dataframes for {s}[/cyan]")
                for i, df in enumerate(valid_dfs):
                    console.print(
                        f"[cyan]Debug: DataFrame {i} shape: {df.shape}, columns: {list(df.columns)[:5]}...[/cyan]"
                    )

                merged = pd.concat(valid_dfs, ignore_index=True)
                console.print(f"[cyan]Debug: Merged dataframe shape: {merged.shape}[/cyan]")
                console.print(f"[cyan]Debug: Merged columns: {list(merged.columns)[:10]}...[/cyan]")

                merged = finalize_merged_dataframe(merged, s, include_wacc=include_wacc)
                console.print(f"[cyan]Debug: Final dataframe shape after finalization: {merged.shape}[/cyan]")

                # Save immediately to prevent data loss
                fn = merged_parquet_name(s, snap_date)
                clean_dataframe_for_parquet(merged).to_parquet(fn, compression="snappy")
                console.print(f"[cyan]SAVE[/cyan] {fn.name}")

                results[s] = merged
                console.print(f"[green]✓ Completed {s} ({len(valid_dfs)}/{len(pages_list)} pages)[/green]")
            else:
                console.print(f"[red]✗ No valid data for {s}[/red]")

            # Reset recent 429 counter periodically
            if total_requests % 20 == 0:
                recent_429s = max(0, recent_429s - 1)  # Gradually forget old 429s

    return results


def build_symbol_list(symbols: List[str], slug_map: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """Build a list of (symbol, slug) tuples using override mappings.

    Args:
        symbols: List of ticker symbols to process
        slug_map: Optional dictionary of ticker:slug mappings to override defaults

    Returns:
        List of (symbol, slug) tuples for processing
    """
    global FILE_OVERRIDES  # Declare global at the top

    cli_overrides = {k.upper(): v for k, v in (slug_map or {}).items()}
    merged = {**FILE_OVERRIDES, **cli_overrides}

    result = []
    new_slugs = {}  # Track newly derived slugs to save
    symbols_needing_derivation = []

    # First pass: identify symbols that need slug derivation
    for sym in symbols:
        if sym.upper() in merged:
            result.append((sym, merged[sym.upper()]))
        else:
            symbols_needing_derivation.append(sym)

    # Second pass: derive slugs with progress bar if needed
    if symbols_needing_derivation:
        console.print(
            f"[cyan]Deriving slugs for {len(symbols_needing_derivation)} symbols  (patience, loading from yfinance)...[/cyan]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Constructing symbol list", total=len(symbols_needing_derivation))

            for sym in symbols_needing_derivation:
                slug = derive_slug(sym)
                result.append((sym, slug))
                new_slugs[sym.upper()] = slug
                progress.advance(task)

        # Save newly derived slugs to the overrides file
        if new_slugs:
            updated_overrides = {**FILE_OVERRIDES, **new_slugs}
            save_file_overrides(updated_overrides)
            # Update the global FILE_OVERRIDES
            FILE_OVERRIDES = updated_overrides
            console.print(f"[green]✓ Saved {len(new_slugs)} new slugs to overrides file[/green]")

    return result


def load_cached_data(symbols: List[str], snap_date: dt.date) -> dict[str, pd.DataFrame]:
    """
    Load existing cached parquet files for symbols.

    Args:
        symbols: List of ticker symbols to check for cached data
        snap_date: The snapshot date to check for

    Returns:
        Dictionary of symbol: dataframe for cached data found
    """
    cached_results = {}

    for sym in symbols:
        parquet_file = merged_parquet_name(sym, snap_date)
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                cached_results[sym] = df
                console.print(f"[green]✓ Found cached data for {sym}[/green] → {parquet_file.name}")
            except Exception as e:
                console.print(f"[red]Warning: Could not load cached data for {sym}: {e}[/red]")

    return cached_results


def expand_special_symbols(symbols: List[str]) -> List[str]:
    """
    Expand special symbols like 'sp500' and 'nasdaq' to their full ticker lists.

    Args:
        symbols: List of symbols that may contain special symbols

    Returns:
        Expanded list with special symbols replaced by actual tickers
    """
    expanded = []

    for symbol in symbols:
        symbol_lower = symbol.lower()

        if symbol_lower == "sp500":
            console.print("[bold blue]Expanding 'sp500' to S&P 500 tickers...[/bold blue]")
            try:
                sp500_tickers = get_sp500_tickers()
                expanded.extend(sp500_tickers)
                console.print(f"[green]✓ Added {len(sp500_tickers)} S&P 500 tickers[/green]")
            except Exception as e:
                console.print(f"[red]Error fetching S&P 500 tickers: {e}[/red]")

        elif symbol_lower == "nasdaq":
            console.print("[bold blue]Expanding 'nasdaq' to NASDAQ tickers...[/bold blue]")
            try:
                nasdaq_tickers = get_nasdaq_tickers()
                expanded.extend(nasdaq_tickers)
                console.print(f"[green]✓ Added {len(nasdaq_tickers)} NASDAQ tickers[/green]")
            except Exception as e:
                console.print(f"[red]Error fetching NASDAQ tickers: {e}[/red]")

        else:
            expanded.append(symbol)

    return expanded


def run_macrotrends_scraper(
    symbols: List[str],
    slug_map: Optional[Dict[str, str]] = None,
    freq: str = "Q",
    force: bool = False,
    date: Optional[str] = None,
    safety_preset: Optional[str] = "conservative",  # Default to conservative
    include_wacc: bool = True,  # New parameter to enable/disable WACC calculations
    **safety_kwargs,
) -> pd.DataFrame:
    """
    Run the Macrotrends scraper to fetch financial data for all DEFAULT_PAGES and return a single merged DataFrame per symbol.

    Automatically handles site redirections by retrying with corrected slugs to ensure
    the requested frequency data is obtained in a single run. Writes only one merged
    parquet file per symbol per quarter.

    Args:
        symbols: List of ticker symbols to scrape (supports 'sp500' and 'nasdaq' special symbols)
        slug_map: Optional mapping of ticker to slug overrides
        freq: Frequency - "Q" for quarterly, "A" for annual
        force: Force re-download even if data exists
        date: Specific date in YYYY-MM-DD format, defaults to latest quarter
        safety_preset: One of "conservative" (default), "balanced", "aggressive", "maximum"
        include_wacc: Whether to calculate and include WACC components (Rf, beta, ERP, R_e, WACC)
        **safety_kwargs: Direct safety parameters (max_concurrent, request_delay, etc.)

    Returns:
        Merged DataFrame with all scraped data including WACC components if enabled
    """
    # Initialize tracking statistics
    failed_symbols = []
    empty_symbols = []
    successful_symbols = []
    cached_symbols = []

    # Configure safety settings with conservative default
    if safety_preset or safety_kwargs:
        if safety_preset:
            presets = get_safety_presets()
            if safety_preset not in presets:
                raise ValueError(f"Invalid safety preset. Choose from: {list(presets.keys())}")
            configure_scraping_safety(**presets[safety_preset])
            console.print(f"[green]Using safety preset: {safety_preset}[/green]")
        if safety_kwargs:
            configure_scraping_safety(**safety_kwargs)
    else:
        # Apply conservative default if nothing specified
        presets = get_safety_presets()
        configure_scraping_safety(**presets["conservative"])
        console.print("[green]Using default safety preset: conservative[/green]")

    # Expand special symbols (sp500, nasdaq) to full ticker lists
    expanded_symbols = expand_special_symbols(symbols)

    pages = DEFAULT_PAGES
    if date:
        snap_date = dt.date.fromisoformat(date)
    else:
        snap_date = get_latest_quarter_end()

    # Check for cached data if not forcing re-download
    cached_data = {}
    symbols_to_scrape = expanded_symbols

    if not force:
        cached_data = load_cached_data(expanded_symbols, snap_date)
        symbols_to_scrape = [sym for sym in expanded_symbols if sym not in cached_data]

        # Track cached symbols
        cached_symbols = list(cached_data.keys())

        if cached_data:
            console.print(f"[cyan]Found cached data for {len(cached_data)} symbol(s)[/cyan]")

            if not symbols_to_scrape:
                console.print("[green]All requested data is already cached. Use --force to re-download.[/green]")
                if cached_data:
                    # Check which cached symbols actually have data
                    valid_cached = {}
                    for sym, df in cached_data.items():
                        if df.empty:
                            empty_symbols.append(sym)
                        else:
                            valid_cached[sym] = df
                            successful_symbols.append(sym)

                    if valid_cached:
                        result = pd.concat(list(valid_cached.values()), ignore_index=False, sort=True)
                        console.print(
                            f"[bold green]✓ Loaded {len(result)} rows from cache for {len(valid_cached)} symbol(s).[/bold green]"
                        )

                        # Print summary statistics
                        _print_scraping_statistics(successful_symbols, empty_symbols, failed_symbols, cached_symbols)
                        return result
                    else:
                        console.print("[red]All cached data is empty.[/red]")
                        _print_scraping_statistics(successful_symbols, empty_symbols, failed_symbols, cached_symbols)
                        return pd.DataFrame()
                else:
                    _print_scraping_statistics(successful_symbols, empty_symbols, failed_symbols, cached_symbols)
                    return pd.DataFrame()

    # Only scrape symbols that don't have cached data (or if force=True)
    if symbols_to_scrape:
        symbols_list = build_symbol_list(symbols_to_scrape, slug_map)
        console.print(f"[blue]Scraping {len(symbols_to_scrape)} symbol(s) from web...[/blue]")

        mismatches: list[Tuple[str, str, str]] = []
        overrides_updated = FILE_OVERRIDES.copy()

        # Fetch all dataframes for symbols that need scraping
        scraped_results = asyncio.run(
            scrape_many(
                symbols_list, pages, snap_date, freq, force, mismatches, overrides_updated, include_wacc=include_wacc
            )
        )

        # Track which symbols succeeded, failed, or returned empty data
        for sym in symbols_to_scrape:
            if sym in scraped_results:
                if scraped_results[sym].empty:
                    empty_symbols.append(sym)
                else:
                    successful_symbols.append(sym)
            else:
                failed_symbols.append(sym)

        # persist any new overrides discovered during redirect handling
        if overrides_updated != FILE_OVERRIDES:
            save_file_overrides(overrides_updated)
            console.print("[bold magenta]↺ slug_overrides.json updated with corrected slugs.[/bold magenta]")

        # pretty-print any redirects that occurred (but were automatically handled)
        if mismatches:
            t = Table(title="Redirects Handled (correct data fetched automatically)")
            t.add_column("Ticker", style="green")
            t.add_column("Old slug")
            t.add_column("New slug (saved)")
            for sym, old, new in mismatches:
                t.add_row(sym, old, new)
            console.print(t)
            console.print(
                f"\n[green]✓ Redirects were automatically handled and {freq} data was fetched.[/green]\n"
                "   [cyan]Updated slugs have been saved for future runs.[/cyan]\n"
            )
    else:
        scraped_results = {}

    # Combine cached and scraped results
    all_results = {**cached_data, **scraped_results}

    # Track cached symbols that had data
    for sym in cached_symbols:
        if sym in all_results and not all_results[sym].empty:
            if sym not in successful_symbols:  # Don't double-count
                successful_symbols.append(sym)
        elif sym in cached_data and cached_data[sym].empty:
            if sym not in empty_symbols:  # Don't double-count
                empty_symbols.append(sym)

    # Write merged parquet files for newly scraped data and return merged dataframe
    all_dfs = []
    for sym, df in all_results.items():
        if not df.empty:
            # Only write to file if this was newly scraped (not from cache)
            if sym in scraped_results:
                fn = merged_parquet_name(sym, snap_date)
                clean_dataframe_for_parquet(df).to_parquet(fn, compression="snappy")
                console.print(f"[cyan]SAVE[/cyan] {fn.name}")
            all_dfs.append(df)

    # Print comprehensive statistics
    _print_scraping_statistics(successful_symbols, empty_symbols, failed_symbols, cached_symbols)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=False, sort=True)
        total_cached = len([s for s in cached_symbols if s in successful_symbols])
        total_scraped = len([s for s in successful_symbols if s not in cached_symbols])
        console.print(
            f"[bold green]✓ Loaded {len(result)} rows for {len(all_results)} symbol(s) "
            f"({total_cached} cached, {total_scraped} scraped) for {snap_date}.[/bold green]"
        )
        return result
    else:
        console.print("[red]No dataframes loaded.[/red]")
        return pd.DataFrame()


def _print_scraping_statistics(
    successful_symbols: List[str], empty_symbols: List[str], failed_symbols: List[str], cached_symbols: List[str]
) -> None:
    """Print comprehensive scraping statistics to the user."""
    console.print("\n" + "=" * 60)
    console.print("[bold]📊 SCRAPING STATISTICS SUMMARY[/bold]")
    console.print("=" * 60)

    total_symbols = len(successful_symbols) + len(empty_symbols) + len(failed_symbols)

    if successful_symbols:
        console.print(f"[green]✅ Successfully loaded data:[/green] {len(successful_symbols)} symbols")
        if len(successful_symbols) <= 10:
            console.print(f"   {', '.join(successful_symbols)}")
        else:
            console.print(f"   {', '.join(successful_symbols[:10])}... (showing first 10)")

    if cached_symbols:
        cached_successful = [s for s in cached_symbols if s in successful_symbols]
        console.print(f"[cyan]💾 Used cached data:[/cyan] {len(cached_successful)} symbols")
        if len(cached_successful) <= 10:
            console.print(f"   {', '.join(cached_successful)}")
        else:
            console.print(f"   {', '.join(cached_successful[:10])}... (showing first 10)")

    if empty_symbols:
        console.print(f"[yellow]⚠️  Empty dataframes (no financial data found):[/yellow] {len(empty_symbols)} symbols")
        console.print(f"   {', '.join(empty_symbols)}")
        console.print("   [dim]These symbols were fetched but contained no usable financial data[/dim]")

    if failed_symbols:
        console.print(f"[red]❌ Failed to scrape:[/red] {len(failed_symbols)} symbols")
        console.print(f"   {', '.join(failed_symbols)}")
        console.print("   [dim]These symbols could not be fetched due to network/server errors[/dim]")

    # Success rate calculation
    if total_symbols > 0:
        success_rate = (len(successful_symbols) / total_symbols) * 100
        console.print(f"\n[bold]Success Rate:[/bold] {success_rate:.1f}% ({len(successful_symbols)}/{total_symbols})")

        if empty_symbols or failed_symbols:
            console.print("\n[yellow]💡 TROUBLESHOOTING TIPS:[/yellow]")
            if empty_symbols:
                console.print(
                    "• Empty dataframes: Check if the ticker symbols are correct and have available financial data"
                )
                console.print("• Some symbols may be new listings without sufficient historical data")
            if failed_symbols:
                console.print("• Failed symbols: Try running with --force flag or check network connectivity")
                console.print("• Consider using a more conservative safety preset if encountering rate limits")

    console.print("=" * 60 + "\n")


def configure_scraping_safety(
    max_concurrent: int = 3,
    request_delay: float = 0.5,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    token_capacity: int = 5,
    token_refill_rate: float = 2.0,
    page_delay: float = 0.2,
    jitter_percent: float = 0.1,
) -> None:
    """
    Configure scraping safety parameters.

    Args:
        max_concurrent: Maximum concurrent requests (1-10 recommended)
        request_delay: Base delay between requests in seconds (0.1-2.0 recommended)
        max_retries: Maximum retry attempts (1-5 recommended)
        backoff_factor: Exponential backoff multiplier (1.5-3.0 recommended)
        token_capacity: Token bucket burst capacity (3-10 recommended)
        token_refill_rate: Token bucket refill rate per second (1-5 recommended)
        page_delay: Base delay between pages for same symbol (0.1-1.0 recommended)
        jitter_percent: Random jitter percentage for delays (0.05-0.2 recommended)

    Presets:
        - Conservative: configure_scraping_safety(max_concurrent=2, request_delay=1.0, token_capacity=3, token_refill_rate=1.0, page_delay=0.5)
        - Balanced: configure_scraping_safety(max_concurrent=3, request_delay=0.5, token_capacity=5, token_refill_rate=2.0, page_delay=0.2)  # Default
        - Aggressive: configure_scraping_safety(max_concurrent=5, request_delay=0.2, token_capacity=8, token_refill_rate=3.0, page_delay=0.1)
        - Maximum: configure_scraping_safety(max_concurrent=8, request_delay=0.1, token_capacity=10, token_refill_rate=5.0, page_delay=0.05)
    """
    global MAX_CONCURRENT_REQUESTS, REQUEST_DELAY, MAX_RETRIES, BACKOFF_FACTOR
    global TOKEN_BUCKET_CAPACITY, TOKEN_REFILL_RATE, PAGE_DELAY, JITTER_PERCENT

    # Validate parameters
    if not 1 <= max_concurrent <= 10:
        raise ValueError("max_concurrent must be between 1 and 10")
    if not 0.05 <= request_delay <= 5.0:
        raise ValueError("request_delay must be between 0.05 and 5.0 seconds")
    if not 1 <= max_retries <= 10:
        raise ValueError("max_retries must be between 1 and 10")
    if not 1.0 <= backoff_factor <= 5.0:
        raise ValueError("backoff_factor must be between 1.0 and 5.0")
    if not 1 <= token_capacity <= 20:
        raise ValueError("token_capacity must be between 1 and 20")
    if not 0.5 <= token_refill_rate <= 10.0:
        raise ValueError("token_refill_rate must be between 0.5 and 10.0")
    if not 0.05 <= page_delay <= 2.0:
        raise ValueError("page_delay must be between 0.05 and 2.0 seconds")
    if not 0.01 <= jitter_percent <= 0.5:
        raise ValueError("jitter_percent must be between 0.01 and 0.5")

    MAX_CONCURRENT_REQUESTS = max_concurrent
    REQUEST_DELAY = request_delay
    MAX_RETRIES = max_retries
    BACKOFF_FACTOR = backoff_factor
    TOKEN_BUCKET_CAPACITY = token_capacity
    TOKEN_REFILL_RATE = token_refill_rate
    PAGE_DELAY = page_delay
    JITTER_PERCENT = jitter_percent

    console.print("[cyan]Scraping safety configured:[/cyan]")
    console.print(f"  Max concurrent: {max_concurrent}")
    console.print(f"  Request delay: {request_delay}s")
    console.print(f"  Max retries: {max_retries}")
    console.print(f"  Backoff factor: {backoff_factor}")
    console.print(f"  Token capacity: {token_capacity}")
    console.print(f"  Token refill rate: {token_refill_rate}/s")
    console.print(f"  Page delay: {page_delay}s")
    console.print(f"  Jitter: ±{jitter_percent*100:.0f}%")


def get_safety_presets() -> dict:
    """Return available safety presets for easy configuration, optimized to minimize rate limits."""
    return {
        "conservative": {
            "max_concurrent": 1,
            "request_delay": 2.0,
            "max_retries": 3,
            "backoff_factor": 2.0,
            "token_capacity": 2,
            "token_refill_rate": 0.5,
            "page_delay": 1.0,
            "jitter_percent": 0.15,
        },
        "balanced": {
            "max_concurrent": 2,
            "request_delay": 1.0,
            "max_retries": 3,
            "backoff_factor": 2.0,
            "token_capacity": 3,
            "token_refill_rate": 1.0,
            "page_delay": 0.5,
            "jitter_percent": 0.1,
        },
        "aggressive": {
            "max_concurrent": 3,
            "request_delay": 0.5,
            "max_retries": 3,
            "backoff_factor": 1.5,
            "token_capacity": 5,
            "token_refill_rate": 2.0,
            "page_delay": 0.2,
            "jitter_percent": 0.1,
        },
        "maximum": {
            "max_concurrent": 4,
            "request_delay": 0.3,
            "max_retries": 2,
            "backoff_factor": 1.5,
            "token_capacity": 6,
            "token_refill_rate": 3.0,
            "page_delay": 0.1,
            "jitter_percent": 0.1,
        },
    }
