# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from rich.console import Console

company_info = pd.read_csv("yahoo_company_info_orig.csv")
company_info = company_info.sort_values("marketCap", ascending=False).reset_index(drop=True)

# This should be removed once we recreate our initial csv
# yfinance growth metrics kinda suck?
remove_keys = [
    "pegRatio",
    "overallRisk",
    "auditRisk",
    "boardRisk",
    "compensationRisk",
    "shareHolderRightsRisk",
    "revenueGrowth",
    "earningsGrowth",
    "earningsQuarterlyGrowth",
]
company_info = company_info.drop(remove_keys, axis=1)

# +
company = yf.Ticker("AAPL")

# company_5quarter_metrics = pd.concat([company.quarterly_income_stmt, company.quarterly_balance_sheet, company.quarterly_cashflow], join='inner')
# company_5quarter_metrics.loc[['Capital Expenditure Reported', 'Operating Cash Flow']]
# for v in company_5quarter_metrics.index:
#     print(v)
# -

# company.quarterly_income_stmt.columns

# provides Ex-Dividend Dates and are not aligned with quarterly schedule
# Would need Annualized Dividend Per Share and divide by stock price
# company.dividends

# +
# Initial Filters
# 1. company: company nane can be NaN, these companies are insignificant
# 2. returnOnEquity: NaN means its either a newer/delisted company
# 3. returnOnAssets: NaN means its either a newer/delisted company
# 4. heldPercentInsiders: Appear to be smaller companies or very new
# There are lots of companies with
# - trailingPE attempts to answer "How many times are investors willing to pay for each dollar of the company's earnings?"
#   Therefore a negative EPS is meaningless for this question.
#   We should probably calculate this
# - forwardPE can also be NaN if expected future EPS if <= 0.
drop_na = [
    "company",
    "returnOnEquity",
    "returnOnAssets",
    "marketCap",
    "heldPercentInsiders",
    "heldPercentInstitutions",
    "trailingPE",
]
company_info = company_info.dropna(subset=drop_na)

# yfinance sets dividendYield to NaN even though 0 makes perfect sense
company_info["dividendYield"] = company_info["dividendYield"].fillna(0)
company_info["fiveYearAvgDividendYield"] = company_info["fiveYearAvgDividendYield"].fillna(0)

# - freeCashflow can have a lot of NaNs ()
# - enterpriseToEbitda also has a lot of NaNs, Remove metric
drop_for_now = [
    "freeCashflow",
    "enterpriseToEbitda",
    "debtToEquity",  # important to calculate but lots missing
    "beta",  # somewhat important but ARM is somehow missing
    "forwardPE",
    "payoutRatio",
]
company_info = company_info.dropna(subset=drop_for_now)
# -

# company_info.columns

for col in company_info.columns:
    percent_na = company_info[col].isna().sum() / len(company_info)
    print(f"{col} has {percent_na:0.5f} NaN rows")

company_info[company_info["beta"].isna()]

company_info.query("marketCap >= 1_000_000_000")


def compute_cagr(end_value: float, start_value: float, periods: float) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).

    Args:
        end_value (float): Final value
        start_value (float): Initial value
        periods (float): Number of periods (years)

    Returns:
        float: CAGR as a decimal (e.g., 0.15 for 15% growth)
    """
    if start_value == 0 or end_value == 0:
        return 0
    try:
        return (abs(end_value) / abs(start_value)) ** (1 / periods) - 1
    except Exception:
        return 0


def compute_metrics_from_quarters(ticker: str) -> Dict[str, Any]:
    """
    Compute all important_keys metrics from company_5quarter_metrics and yfinance price data.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        Dict[str, Any]: Dictionary of computed metrics.
    """
    console = Console()
    company = yf.Ticker(ticker)
    try:
        quarterly_income = company.quarterly_income_stmt
        quarterly_balance = company.quarterly_balance_sheet
        quarterly_cashflow = company.quarterly_cashflow
        dividends = company.dividends

        # Get annual statements for growth calculations
        annual_income = company.income_stmt

        console.log("[blue]Available metrics in quarterly quarterly_income statement:[/blue]")
        console.log(quarterly_income.index.tolist())
        console.log("\n[blue]Available metrics in annual quarterly_income statement:[/blue]")
        console.log(annual_income.index.tolist())

        if "Basic EPS" in annual_income.index:
            eps_vals = annual_income.loc["Basic EPS"].values
            console.log(f"\n[blue]Annual Basic EPS values: {eps_vals}[/blue]")
    except Exception as e:
        console.log(f"[red]Failed to load statements: {e}[/red]")
        return {"error": f"Failed to load statements: {e}"}

    def safe_ttm(df, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                vals = df.loc[key].values[:4]
                return vals.sum()
            except Exception:
                continue
        return 0

    def safe_get(df, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                return df.loc[key].iloc[0]
            except Exception:
                continue
        return 0

    def safe_diff(df, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                vals = df.loc[key].values[:2]
                if len(vals) == 2 and vals[1] != 0:
                    return (vals[0] - vals[1]) / vals[1]
            except Exception:
                continue
        return 0

    # Get latest close price and shares outstanding
    hist = company.history(period="5d")
    price = float(hist["Close"].iloc[-1])
    shares_out = safe_get(quarterly_balance, ["Ordinary Shares Number", "Share Issued"])
    shares_out = float(shares_out)
    market_cap = price * shares_out

    # TTM values (use all plausible keys in order of preference)
    ttm_net_income = safe_ttm(
        quarterly_income, ["Net Income Common Stockholders", "Diluted NI Availto Com Stockholders", "Net Income"]
    )
    ttm_equity = safe_get(
        quarterly_balance, ["Common Stock Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"]
    )
    ttm_total_assets = safe_get(quarterly_balance, ["Total Assets"])
    ttm_op_income = safe_ttm(quarterly_income, ["Operating Income"])
    ttm_revenue = safe_ttm(quarterly_income, ["Total Revenue", "Operating Revenue"])
    ttm_ebitda = safe_ttm(quarterly_income, ["EBITDA"])
    ttm_gross_profit = safe_ttm(quarterly_income, ["Gross Profit"])
    ttm_op_cf = safe_ttm(quarterly_cashflow, ["Operating Cash Flow"])
    ttm_capex = safe_ttm(quarterly_cashflow, ["Capital Expenditure"])
    ttm_dividends_paid = safe_ttm(quarterly_cashflow, ["Cash Dividends Paid", "Common Stock Dividend Paid"])
    total_cash = safe_get(
        quarterly_balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]
    )
    total_debt = safe_get(quarterly_balance, ["Total Debt"])
    current_assets = safe_get(quarterly_balance, ["Current Assets"])
    current_liabilities = safe_get(quarterly_balance, ["Current Liabilities"])

    metrics = {}
    # --- Profitability & Economic Moat ---
    metrics["returnOnEquity"] = ttm_net_income / ttm_equity
    try:
        assets_vals = quarterly_balance.loc["Total Assets"].values[:5]
        avg_assets = sum(assets_vals) / len(assets_vals)
    except Exception:
        avg_assets = ttm_total_assets
    metrics["returnOnAssets"] = ttm_net_income / avg_assets
    metrics["operatingMargins"] = ttm_op_income / ttm_revenue
    metrics["ebitdaMargins"] = ttm_ebitda / ttm_revenue
    metrics["grossMargins"] = ttm_gross_profit / ttm_revenue
    metrics["profitMargins"] = ttm_net_income / ttm_revenue

    # Growth metrics using CAGR
    # For quarterly metrics, we use 1 year (4 quarters) as the period
    # Revenue growth - using quarterly data over 1 year
    _tr = annual_income.loc["Total Revenue"]
    metrics["revenueGrowth"] = compute_cagr(_tr[0], _tr[3], 3)  # 3 year CAGR

    # Earnings growth - using Basic EPS from annual data

    # DO NOT DELETE THIS BLOCK
    # Option 1: Using quarterly data like so matches yfinance
    #           Choose to go with CAGR with longest period possible
    #           which is 4 years
    # _beps = quarterly_income.loc['Basic EPS']
    # metrics["earningsGrowth"] = (_beps[0] - _beps[4]) / _beps[4]
    _beps = annual_income.loc["Basic EPS"]
    metrics["earningsGrowth"] = compute_cagr(_beps[0], _beps[3], 3)

    _net_income = annual_income.loc["Net Income"]
    metrics["earningsQuarterlyGrowth"] = compute_cagr(_net_income[0], _net_income[3], 3)

    metrics["freeCashflow"] = ttm_op_cf + ttm_capex

    # --- Balance-Sheet Resilience ---
    metrics["totalCash"] = float(total_cash)
    metrics["totalDebt"] = float(total_debt)
    metrics["debtToEquity"] = (float(total_debt) / ttm_equity) * 100
    metrics["currentRatio"] = float(current_assets) / float(current_liabilities)

    # --- Capital-Allocation Track Record ---
    if not dividends.empty and price != 0:
        annual_div = dividends[-4:].sum()  # last 4 quarters
        metrics["dividendYield"] = (annual_div / price) * 100
        # 5yr avg dividend yield: use last 20 dividends (approx 5 years)
        fiveyr_div = dividends[-20:].sum()
        metrics["fiveYearAvgDividendYield"] = (fiveyr_div / price) * 100 / 5
    else:
        metrics["dividendYield"] = 0
        metrics["fiveYearAvgDividendYield"] = 0
    metrics["payoutRatio"] = abs(ttm_dividends_paid) / ttm_net_income

    # --- Valuation vs. Quality ---
    metrics["marketCap"] = market_cap
    eps_ttm = ttm_net_income / shares_out
    metrics["trailingPE"] = price / eps_ttm
    # Forward PE: use forwardEps from company.info if available
    forward_eps = company.info.get("forwardEps", 0)
    if forward_eps:
        metrics["forwardPE"] = price / forward_eps
    else:
        metrics["forwardPE"] = 0
    book_value_per_share = ttm_equity / shares_out
    metrics["priceToBook"] = price / book_value_per_share
    metrics["priceToSalesTrailing12Months"] = market_cap / ttm_revenue
    enterprise_value = market_cap + float(total_debt) - float(total_cash)
    metrics["enterpriseValue"] = enterprise_value
    metrics["enterpriseToEbitda"] = enterprise_value / ttm_ebitda
    # PEG ratio: trailingPE / (100 * earningsGrowth) (earningsGrowth as YoY fraction)
    metrics["trailingPegRatio"] = metrics["trailingPE"] / (100 * metrics["earningsGrowth"])

    # --- Ownership & Liquidity ---
    # IMPORTANT: DO NOT MODIFY BELOW THIS LINE
    metrics["heldPercentInsiders"] = company.info.get("heldPercentInsiders", 0)
    metrics["heldPercentInstitutions"] = company.info.get("heldPercentInstitutions", 0)
    metrics["beta"] = company.info.get("beta", 0)

    return metrics
