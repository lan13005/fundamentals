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
import duckdb
import pandas as pd
import yfinance as yf
from aiohttp import ClientTimeout
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape")
console = Console()

# ───────────── configuration ───────────── #
DATA_DIR = pathlib.Path("macro_data/parquet")
DB_PATH = pathlib.Path("macro_data/macrotrends.duckdb")
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


def parquet_name(sym: str, page: str, snap_date: dt.date) -> pathlib.Path:
    return DATA_DIR / f"{sym.upper()}_{page}_{snap_date.isoformat()}.parquet"


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
) -> None:
    fn = parquet_name(sym, page, snap_date)
    if not force and fn.exists():
        console.print(f"[green]CACHE[/] {fn.name}")
        return

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

    parse_original_data(html).to_parquet(fn, compression="snappy")
    console.print(f"[cyan]SAVE[/cyan] {fn.name}")


async def scrape_many(
    symbols: List[Tuple[str, str]],
    pages: Iterable[str],
    snap_date: dt.date,
    freq: str,
    force: bool,
    mismatches: list[Tuple[str, str]],
    overrides: dict[str, str],
) -> None:
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=20)) as sess:
        await asyncio.gather(
            *[
                fetch_table(sess, s, slug, p, snap_date, freq, force, mismatches, overrides)
                for s, slug in symbols
                for p in pages
            ]
        )


# ───────────── DuckDB refresh ───────────── #
def materialise_duckdb() -> None:
    duckdb.execute(
        f"""
        CREATE OR REPLACE TABLE macrotrends AS
        SELECT *,
               regexp_extract(filename, '^([A-Z\\.]+)_', 1)           AS ticker,
               regexp_extract(filename, '^[A-Z\\.]+_([a-z\\-]+)_', 1) AS page,
               regexp_extract(filename, '_(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet$', 1)
                                                                     AS snapshot_date
        FROM read_parquet('{DATA_DIR}/*.parquet', union_by_name=TRUE)
        """
    )
    console.print("[bold green]✓ macrotrends.duckdb refreshed[/bold green]")


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
    pages: Optional[List[str]] = None,
    slug_map: Optional[Dict[str, str]] = None,
    freq: str = "Q",
    force: bool = False,
    date: Optional[str] = None,
) -> None:
    """Run the Macrotrends scraper to fetch financial data.

    Args:
        symbols: List of ticker symbols to process
        pages: List of pages to scrape (defaults to income statement, balance sheet, etc.)
        slug_map: Optional dictionary of ticker:slug mappings to override defaults
        freq: Frequency of data to fetch - 'Q' for quarterly or 'A' for annual (default: 'Q')
        force: Whether to ignore cache and force fetch from web (default: False)
        date: Snapshot date in YYYY-MM-DD format (default: today)
    """
    pages = pages or DEFAULT_PAGES
    snap_date = dt.date.fromisoformat(date) if date else dt.date.today()
    symbols_list = build_symbol_list(symbols, slug_map)

    mismatches: list[Tuple[str, str]] = []
    overrides_updated = FILE_OVERRIDES.copy()

    asyncio.run(scrape_many(symbols_list, pages, snap_date, freq, force, mismatches, overrides_updated))
    materialise_duckdb()

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True, metavar="TICKER")
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--slug-map", nargs="+", default=[], metavar="TICKER:slug")
    parser.add_argument("--freq", choices=["Q", "A"], default="Q")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--date", default=dt.date.today().isoformat(), metavar="YYYY-MM-DD")

    args = parser.parse_args()
    slug_map_dict = dict(pair.split(":", 1) for pair in args.slug_map) if args.slug_map else None

    run_macrotrends_scraper(
        symbols=args.symbols,
        pages=args.pages,
        slug_map=slug_map_dict,
        freq=args.freq,
        force=args.force,
        date=args.date
    )
