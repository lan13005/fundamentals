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
import pandas as pd
import yfinance as yf
from typing import Dict, Any
from rich.console import Console

company_info = pd.read_csv("yahoo_company_info_orig.csv")
company_info = company_info.sort_values('marketCap', ascending=False).reset_index(drop=True)

# This should be removed once we recreate our initial csv
# yfinance growth metrics kinda suck?
remove_keys = ['pegRatio', 'overallRisk', 'auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk', 
                'revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth']
company_info = company_info.drop(remove_keys, axis=1)

# +
company = yf.Ticker("AAPL")

# company_5quarter_metrics = pd.concat([company.quarterly_income_stmt, company.quarterly_balance_sheet, company.quarterly_cashflow], join='inner')
# company_5quarter_metrics.loc[['Capital Expenditure Reported', 'Operating Cash Flow']]
# for v in company_5quarter_metrics.index:
#     print(v)
# -

company.quarterly_income_stmt.columns

# provides Ex-Dividend Dates and are not aligned with quarterly schedule
# Would need Annualized Dividend Per Share and divide by stock price
company.dividends 

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
drop_na = ['company', 'returnOnEquity', 'returnOnAssets', 'marketCap',
           'heldPercentInsiders', 'heldPercentInstitutions', 'trailingPE']
company_info = company_info.dropna(subset=drop_na)

# yfinance sets dividendYield to NaN even though 0 makes perfect sense
company_info['dividendYield'] = company_info['dividendYield'].fillna(0)
company_info['fiveYearAvgDividendYield'] = company_info['fiveYearAvgDividendYield'].fillna(0)

# - freeCashflow can have a lot of NaNs ()
# - enterpriseToEbitda also has a lot of NaNs, Remove metric
drop_for_now = ['freeCashflow', 'enterpriseToEbitda', 'debtToEquity', # important to calculate but lots missing
                'beta', # somewhat important but ARM is somehow missing
                'forwardPE', 'payoutRatio']
company_info = company_info.dropna(subset=drop_for_now)
# -

company_info.columns

for col in company_info.columns:
    percent_na = company_info[col].isna().sum() / len(company_info)
    print(f"{col} has {percent_na:0.5f} NaN rows")

company_info[company_info['beta'].isna()]

company_info.query('marketCap >= 1_000_000_000')

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
        income = company.quarterly_income_stmt
        balance = company.quarterly_balance_sheet
        cashflow = company.quarterly_cashflow
        dividends = company.dividends
        
        print("available indices:")
        print(f"income: {income.index}")
        print(f"balance: {balance.index}")
        print(f"cashflow: {cashflow.index}")
        print(f"dividends: {dividends.index}")
        
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
        return None

    def safe_get(df, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                return df.loc[key].iloc[0]
            except Exception:
                continue
        return None

    def safe_diff(df, keys):
        # For growth metrics: (latest - prev) / prev
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                vals = df.loc[key].values[:2]
                if len(vals) == 2 and vals[1] != 0:
                    return (vals[0] - vals[1]) / vals[1]
            except Exception:
                continue
        return None

    # Get latest close price and shares outstanding
    try:
        hist = company.history(period="5d")
        price = float(hist["Close"].iloc[-1])
    except Exception:
        price = None
    shares_out = safe_get(balance, ["Ordinary Shares Number", "Common Stock Shares Outstanding", "Share Issued"])
    if shares_out is not None:
        try:
            shares_out = float(shares_out)
        except Exception:
            shares_out = None
    # Market Cap
    market_cap = price * shares_out if price is not None and shares_out is not None else None

    # TTM values
    ttm_net_income = safe_ttm(income, ["Net Income", "NetIncome"])
    ttm_equity = safe_get(balance, ["Total Stockholder Equity", "Total Equity"])
    ttm_total_assets = safe_get(balance, ["Total Assets"])
    ttm_op_income = safe_ttm(income, ["Operating Income"])
    ttm_revenue = safe_ttm(income, ["Total Revenue", "Revenue"])
    ttm_ebitda = safe_ttm(income, ["EBITDA"])
    ttm_gross_profit = safe_ttm(income, ["Gross Profit"])
    ttm_op_cf = safe_ttm(cashflow, ["Operating Cash Flow"])
    ttm_capex = safe_ttm(cashflow, ["Capital Expenditures"])
    ttm_dividends_paid = safe_ttm(cashflow, ["Dividends Paid"])
    total_cash = safe_get(balance, ["Cash", "Cash And Cash Equivalents"])
    total_debt = safe_get(balance, ["Total Debt"])
    current_assets = safe_get(balance, ["Total Current Assets"])
    current_liabilities = safe_get(balance, ["Total Current Liabilities"])

    metrics = {}
    # --- Profitability & Economic Moat ---
    metrics["returnOnEquity"] = (ttm_net_income / ttm_equity) if ttm_net_income is not None and ttm_equity not in (None, 0) else None
    metrics["returnOnAssets"] = (ttm_net_income / ttm_total_assets) if ttm_net_income is not None and ttm_total_assets not in (None, 0) else None
    metrics["operatingMargins"] = (ttm_op_income / ttm_revenue) if ttm_op_income is not None and ttm_revenue not in (None, 0) else None
    metrics["ebitdaMargins"] = (ttm_ebitda / ttm_revenue) if ttm_ebitda is not None and ttm_revenue not in (None, 0) else None
    metrics["grossMargins"] = (ttm_gross_profit / ttm_revenue) if ttm_gross_profit is not None and ttm_revenue not in (None, 0) else None
    metrics["profitMargins"] = (ttm_net_income / ttm_revenue) if ttm_net_income is not None and ttm_revenue not in (None, 0) else None

    # --- Growth Sustainability ---
    metrics["revenueGrowth"] = safe_diff(income, ["Total Revenue", "Revenue"])
    metrics["earningsGrowth"] = safe_diff(income, ["Net Income", "NetIncome"])
    metrics["earningsQuarterlyGrowth"] = safe_diff(income, ["Net Income", "NetIncome"])
    metrics["freeCashflow"] = (ttm_op_cf + ttm_capex) if ttm_op_cf is not None and ttm_capex is not None else None

    # --- Balance-Sheet Resilience ---
    metrics["totalCash"] = float(total_cash) if total_cash is not None else None
    metrics["totalDebt"] = float(total_debt) if total_debt is not None else None
    metrics["debtToEquity"] = (float(total_debt) / ttm_equity) if total_debt is not None and ttm_equity not in (None, 0) else None
    metrics["currentRatio"] = (float(current_assets) / float(current_liabilities)) if current_assets not in (None, 0) and current_liabilities not in (None, 0) else None

    # --- Capital-Allocation Track Record ---
    try:
        if not dividends.empty and price is not None:
            annual_div = dividends[-4:].sum()
            metrics["dividendYield"] = (annual_div / price)
        else:
            metrics["dividendYield"] = 0
    except Exception:
        metrics["dividendYield"] = None
    metrics["payoutRatio"] = (abs(ttm_dividends_paid) / ttm_net_income) if ttm_dividends_paid is not None and ttm_net_income not in (None, 0) else None
    # 5yr avg dividend yield: fallback to None (not in statements)
    metrics["fiveYearAvgDividendYield"] = None

    # --- Valuation vs. Quality ---
    metrics["marketCap"] = market_cap
    # EPS (TTM): Net income / shares_out
    eps_ttm = (ttm_net_income / shares_out) if ttm_net_income is not None and shares_out not in (None, 0) else None
    # Trailing PE: price / EPS (TTM)
    metrics["trailingPE"] = (price / eps_ttm) if price is not None and eps_ttm not in (None, 0) else None
    # Forward PE: not available from statements, fallback to None
    metrics["forwardPE"] = None
    # Book value per share: equity / shares_out
    book_value_per_share = (ttm_equity / shares_out) if ttm_equity is not None and shares_out not in (None, 0) else None
    metrics["priceToBook"] = (price / book_value_per_share) if price is not None and book_value_per_share not in (None, 0) else None
    metrics["priceToSalesTrailing12Months"] = (market_cap / ttm_revenue) if market_cap is not None and ttm_revenue not in (None, 0) else None
    # Enterprise Value = market cap + total debt - cash
    enterprise_value = (market_cap + float(total_debt) - float(total_cash)) if market_cap is not None and total_debt is not None and total_cash is not None else None
    metrics["enterpriseValue"] = enterprise_value
    metrics["enterpriseToEbitda"] = (enterprise_value / ttm_ebitda) if enterprise_value is not None and ttm_ebitda not in (None, 0) else None
    # PEG ratio: trailingPE / earnings growth
    metrics["trailingPegRatio"] = (metrics["trailingPE"] / metrics["earningsQuarterlyGrowth"]) if metrics["trailingPE"] not in (None, 0) and metrics["earningsQuarterlyGrowth"] not in (None, 0) else None

    # --- Ownership & Liquidity ---
    # These are not in statements, so fallback to None
    metrics["heldPercentInsiders"] = None
    metrics["heldPercentInstitutions"] = None
    metrics["beta"] = None

    return metrics


