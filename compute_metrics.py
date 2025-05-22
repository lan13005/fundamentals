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
    company = yf.Ticker(ticker)
    try:
        company_5quarter_metrics = pd.concat([
            company.quarterly_income_stmt,
            company.quarterly_balance_sheet,
            company.quarterly_cashflow
        ], join='inner')
    except Exception as e:
        return {"error": f"Failed to load statements: {e}"}

    def safe_get(df, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                return df.loc[key].iloc[0]
            except Exception:
                continue
        return None

    info = company.info
    metrics = {}

    # --- Profitability & Economic Moat ---
    net_income = safe_get(company.quarterly_income_stmt, ["Net Income", "NetIncome"])
    equity = safe_get(company.quarterly_balance_sheet, ["Total Stockholder Equity", "Total Equity"])
    metrics["returnOnEquity"] = (net_income / equity) if net_income is not None and equity else info.get("returnOnEquity")

    total_assets = safe_get(company.quarterly_balance_sheet, ["Total Assets"])
    metrics["returnOnAssets"] = (net_income / total_assets) if net_income is not None and total_assets else info.get("returnOnAssets")

    op_income = safe_get(company.quarterly_income_stmt, ["Operating Income"])
    revenue = safe_get(company.quarterly_income_stmt, ["Total Revenue", "Revenue"])
    metrics["operatingMargins"] = (op_income / revenue) if op_income is not None and revenue else info.get("operatingMargins")

    ebitda = safe_get(company.quarterly_income_stmt, ["EBITDA"])
    metrics["ebitdaMargins"] = (ebitda / revenue) if ebitda is not None and revenue else info.get("ebitdaMargins")

    gross_profit = safe_get(company.quarterly_income_stmt, ["Gross Profit"])
    metrics["grossMargins"] = (gross_profit / revenue) if gross_profit is not None and revenue else info.get("grossMargins")

    metrics["profitMargins"] = (net_income / revenue) if net_income is not None and revenue else info.get("profitMargins")

    # --- Growth Sustainability ---
    try:
        revs = company.quarterly_income_stmt.loc[["Total Revenue", "Revenue"]].values[0]
        metrics["revenueGrowth"] = ((revs[0] - revs[1]) / revs[1]) if len(revs) > 1 and revs[1] != 0 else info.get("revenueGrowth")
    except Exception:
        metrics["revenueGrowth"] = info.get("revenueGrowth")

    try:
        earnings = company.quarterly_income_stmt.loc[["Net Income", "NetIncome"]].values[0]
        metrics["earningsGrowth"] = ((earnings[0] - earnings[1]) / earnings[1]) if len(earnings) > 1 and earnings[1] != 0 else info.get("earningsGrowth")
        metrics["earningsQuarterlyGrowth"] = ((earnings[0] - earnings[1]) / earnings[1]) if len(earnings) > 1 and earnings[1] != 0 else info.get("earningsQuarterlyGrowth")
    except Exception:
        metrics["earningsGrowth"] = info.get("earningsGrowth")
        metrics["earningsQuarterlyGrowth"] = info.get("earningsQuarterlyGrowth")

    op_cf = safe_get(company.quarterly_cashflow, ["Operating Cash Flow"])
    capex = safe_get(company.quarterly_cashflow, ["Capital Expenditures"])
    if op_cf is not None and capex is not None:
        metrics["freeCashflow"] = op_cf + capex
    else:
        metrics["freeCashflow"] = info.get("freeCashflow")

    # --- Balance-Sheet Resilience ---
    metrics["totalCash"] = safe_get(company.quarterly_balance_sheet, ["Cash", "Cash And Cash Equivalents"]) or info.get("totalCash")
    metrics["totalDebt"] = safe_get(company.quarterly_balance_sheet, ["Total Debt"]) or info.get("totalDebt")
    total_debt = metrics["totalDebt"]
    metrics["debtToEquity"] = (total_debt / equity) if total_debt is not None and equity else info.get("debtToEquity")
    current_assets = safe_get(company.quarterly_balance_sheet, ["Total Current Assets"])
    current_liabilities = safe_get(company.quarterly_balance_sheet, ["Total Current Liabilities"])
    metrics["currentRatio"] = (current_assets / current_liabilities) if current_assets is not None and current_liabilities else info.get("currentRatio")

    # --- Capital-Allocation Track Record ---
    try:
        dividends = company.dividends
        if not dividends.empty:
            annual_div = dividends[-4:].sum()  # last 4 quarters
            price = info.get("regularMarketPrice", None)
            metrics["dividendYield"] = (annual_div / price) if price else info.get("dividendYield")
        else:
            metrics["dividendYield"] = 0
    except Exception:
        metrics["dividendYield"] = info.get("dividendYield")

    try:
        dividends_paid = safe_get(company.quarterly_cashflow, ["Dividends Paid"])
        metrics["payoutRatio"] = (abs(dividends_paid) / net_income) if dividends_paid is not None and net_income else info.get("payoutRatio")
    except Exception:
        metrics["payoutRatio"] = info.get("payoutRatio")

    metrics["fiveYearAvgDividendYield"] = info.get("fiveYearAvgDividendYield", None)

    # --- Valuation vs. Quality ---
    market_cap = info.get("marketCap", None)
    metrics["forwardPE"] = info.get("forwardPE", None)
    metrics["trailingPE"] = info.get("trailingPE", None)
    metrics["priceToBook"] = (market_cap / equity) if market_cap and equity else info.get("priceToBook")
    metrics["priceToSalesTrailing12Months"] = (market_cap / (revenue * 4)) if market_cap and revenue else info.get("priceToSalesTrailing12Months")
    try:
        enterprise_value = market_cap + total_debt - metrics["totalCash"] if market_cap and total_debt and metrics["totalCash"] else info.get("enterpriseValue")
        metrics["enterpriseValue"] = enterprise_value
        metrics["enterpriseToEbitda"] = (enterprise_value / (ebitda * 4)) if enterprise_value and ebitda else info.get("enterpriseToEbitda")
    except Exception:
        metrics["enterpriseValue"] = info.get("enterpriseValue")
        metrics["enterpriseToEbitda"] = info.get("enterpriseToEbitda")
    metrics["trailingPegRatio"] = info.get("trailingPegRatio", None)

    # --- Ownership & Liquidity ---
    metrics["heldPercentInsiders"] = info.get("heldPercentInsiders", None)
    metrics["heldPercentInstitutions"] = info.get("heldPercentInstitutions", None)
    metrics["marketCap"] = market_cap
    metrics["beta"] = info.get("beta", None)

    # --- Holistic Valuation Context ---
    metrics["enterpriseValue"] = enterprise_value if 'enterprise_value' in locals() else info.get("enterpriseValue")

    return metrics


