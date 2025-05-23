# +
from datetime import datetime

import pandas as pd
from stockdex import Ticker

# -

ticker = Ticker(ticker="AAPL")


# Function to safely convert and clean numeric string data
def safe_numeric_conversion(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("M", "e6")
        .str.replace("B", "e9")
        .str.replace("T", "e12")
        .str.replace("$", "")
        .str.replace(",", "")
        .str.replace("%", "")
        .str.replace("--", "nan")
        .str.strip(),
        errors="coerce",
    )


# # Get dataframes

# +
## Yahoo API + Web scraped
_yahoo_api_price_data = ticker.yahoo_api_price(range="1y", dataGranularity="1d")
_yahoo_web_summary = ticker.yahoo_web_summary
_yahoo_api_income_statement_quarterly = ticker.yahoo_api_income_statement(frequency="quarterly")
_yahoo_api_cash_flow = ticker.yahoo_api_cash_flow(format="raw")
_yahoo_api_balance_sheet = ticker.yahoo_api_balance_sheet(period1=datetime(2019, 1, 1))
_yahoo_api_financials = ticker.yahoo_api_financials(period1=datetime(2022, 1, 1), period2=datetime.today())
_yahoo_web_income_statement = ticker.yahoo_web_income_stmt
_yahoo_web_balance_sheet = ticker.yahoo_web_balance_sheet
_yahoo_web_cashflow = ticker.yahoo_web_cashflow
_yahoo_web_valuation_measures = ticker.yahoo_web_valuation_measures
_yahoo_web_major_holders = ticker.yahoo_web_major_holders
_yahoo_web_trading_information = ticker.yahoo_web_trading_information
_digrin_payout_ratio = ticker.digrin_payout_ratio

## Dirgin dataframes
_digrin_dividend_data = ticker.digrin_dividend
_digrin_price = ticker.digrin_price
_digrin_assets_vs_liabilities = ticker.digrin_assets_vs_liabilities
_digrin_free_cash_flow = ticker.digrin_free_cash_flow
_digrin_net_income = ticker.digrin_net_income
_digrin_cash_and_debt = ticker.digrin_cash_and_debt
_digrin_shares_outstanding = ticker.digrin_shares_outstanding
_digrin_expenses = ticker.digrin_expenses
_digrin_cost_of_revenue = ticker.digrin_cost_of_revenue
_digrin_upcoming_estimated_earnings = ticker.digrin_upcoming_estimated_earnings
_digrin_dgr3 = ticker.digrin_dgr3
_digrin_dgr5 = ticker.digrin_dgr5
_digrin_dgr10 = ticker.digrin_dgr10

## Macrotrends (these are scraped so requests can be slow)
_macrotrends_income_statement = ticker.macrotrends_income_statement
_macrotrends_balance_sheet = ticker.macrotrends_balance_sheet
_macrotrends_cash_flow = ticker.macrotrends_cash_flow
_macrotrends_gross_margin = ticker.macrotrends_gross_margin
_macrotrends_operating_margin = ticker.macrotrends_operating_margin
_macrotrends_ebitda_margin = ticker.macrotrends_ebitda_margin
_macrotrends_net_margin = ticker.macrotrends_net_margin

# -

# # Apply safe conversion to all relevant dataframes
#

# Macrotrends Income Statement
macrotrends_income_statement = _macrotrends_income_statement.apply(safe_numeric_conversion)
# Macrotrends Balance Sheet
macrotrends_balance_sheet = _macrotrends_balance_sheet.apply(safe_numeric_conversion)
# Macrotrends Cash Flow
macrotrends_cash_flow = _macrotrends_cash_flow.apply(safe_numeric_conversion)
# Yahoo API Price Data
yahoo_api_price_data = _yahoo_api_price_data.copy()
# Yahoo Web Major Holders
yahoo_web_major_holders = _yahoo_web_major_holders.copy()
# Yahoo API Income Statement (Quarterly) - columns already numeric, need to strip the index strings if converting them
yahoo_api_income_statement_quarterly = _yahoo_api_income_statement_quarterly.copy()
yahoo_api_income_statement_quarterly.columns = [
    col.replace("quarterly", "") for col in yahoo_api_income_statement_quarterly.columns
]
# Yahoo API Cash Flow
yahoo_api_cash_flow = _yahoo_api_cash_flow.copy()
yahoo_api_cash_flow.columns = [col.replace("annual", "") for col in yahoo_api_cash_flow.columns]
yahoo_api_cash_flow = yahoo_api_cash_flow.apply(safe_numeric_conversion)
# Yahoo API Balance Sheet
yahoo_api_balance_sheet = _yahoo_api_balance_sheet.copy()
yahoo_api_balance_sheet.columns = [col.replace("annual", "") for col in yahoo_api_balance_sheet.columns]
yahoo_api_balance_sheet = yahoo_api_balance_sheet.apply(safe_numeric_conversion)
# Yahoo API Financials
yahoo_api_financials = _yahoo_api_financials.copy()
yahoo_api_financials.columns = [col.replace("annual", "") for col in yahoo_api_financials.columns]
yahoo_api_financials = yahoo_api_financials.apply(safe_numeric_conversion)
# Yahoo Web Summary (specific columns)
yahoo_web_summary = _yahoo_web_summary.copy()
for col in [0]:
    if col in yahoo_web_summary.columns:
        yahoo_web_summary.loc[
            ["marketCap", "trailingPE", "targetMeanPrice", "regularMarketPrice", "regularMarketChange"], col
        ] = safe_numeric_conversion(
            yahoo_web_summary.loc[
                ["marketCap", "trailingPE", "targetMeanPrice", "regularMarketPrice", "regularMarketChange"], col
            ]
        )
        yahoo_web_summary.loc[["regularMarketVolume", "averageVolume"], col] = safe_numeric_conversion(
            yahoo_web_summary.loc[["regularMarketVolume", "averageVolume"], col]
        )
        # yahoo_web_summary.loc["regularMarketChangePercent", col] = (
        #     safe_numeric_conversion(yahoo_web_summary.loc["regularMarketChangePercent", col]) / 100
        # )
# Yahoo Web Income Statement
yahoo_web_income_statement = _yahoo_web_income_statement.apply(safe_numeric_conversion)
# Yahoo Web Balance Sheet
yahoo_web_balance_sheet = _yahoo_web_balance_sheet.apply(safe_numeric_conversion)
# Yahoo Web Cashflow
yahoo_web_cashflow = _yahoo_web_cashflow.apply(safe_numeric_conversion)
# Yahoo Web Valuation Measures
yahoo_web_valuation_measures = _yahoo_web_valuation_measures.apply(safe_numeric_conversion)
# Yahoo Web Trading Information
yahoo_web_trading_information = _yahoo_web_trading_information.apply(safe_numeric_conversion)
# Digrin Dividend Data
digrin_dividend_data = _digrin_dividend_data.copy()
for col in ["Dividend amount (change)", "Adjusted Price", "Close Price"]:
    digrin_dividend_data[col] = safe_numeric_conversion(digrin_dividend_data[col].astype(str).str.split(" ").str[0])
# Digrin Payout Ratio
digrin_payout_ratio = _digrin_payout_ratio.copy()
digrin_payout_ratio["Payout ratio"] = safe_numeric_conversion(digrin_payout_ratio["Payout ratio"]) / 100
digrin_payout_ratio["PE ratio"] = safe_numeric_conversion(digrin_payout_ratio["PE ratio"])
# Digrin DGR dataframes
digrin_dgr3 = _digrin_dgr3.copy()
digrin_dgr5 = _digrin_dgr5.copy()
digrin_dgr10 = _digrin_dgr10.copy()
for df in [digrin_dgr3, digrin_dgr5, digrin_dgr10]:
    df["Dividend"] = safe_numeric_conversion(df["Dividend"])
    df["Estimated Yield on Cost"] = safe_numeric_conversion(df["Estimated Yield on Cost"]) / 100
# Macrotrends Margin dataframes
macrotrends_gross_margin = _macrotrends_gross_margin.copy()
macrotrends_operating_margin = _macrotrends_operating_margin.copy()
macrotrends_ebitda_margin = _macrotrends_ebitda_margin.copy()
macrotrends_net_margin = _macrotrends_net_margin.copy()
for df in [macrotrends_gross_margin, macrotrends_operating_margin, macrotrends_ebitda_margin, macrotrends_net_margin]:
    df["TTM Revenue"] = safe_numeric_conversion(df["TTM Revenue"])
    df[df.columns[2]] = safe_numeric_conversion(
        df[df.columns[2]]
    )  # TTM Gross Profit/Operating Income/EBITDA/Net Income
    df[df.columns[3]] = safe_numeric_conversion(df[df.columns[3]]) / 100  # Margin
# Digrin Price
digrin_price = _digrin_price.copy()
for col in ["Adjusted price", "Real price"]:
    digrin_price[col] = safe_numeric_conversion(digrin_price[col])
# Digrin Assets vs Liabilities
digrin_assets_vs_liabilities = _digrin_assets_vs_liabilities.apply(safe_numeric_conversion)
# Digrin Free Cash Flow
digrin_free_cash_flow = _digrin_free_cash_flow.apply(safe_numeric_conversion)
# Digrin Net Income
digrin_net_income = _digrin_net_income.apply(safe_numeric_conversion)
# Digrin Cash and Debt
digrin_cash_and_debt = _digrin_cash_and_debt.apply(safe_numeric_conversion)
# Digrin Shares Outstanding
digrin_shares_outstanding = _digrin_shares_outstanding.apply(safe_numeric_conversion)
# Digrin Expenses
digrin_expenses = _digrin_expenses.apply(safe_numeric_conversion)
# Digrin Cost of Revenue
digrin_cost_of_revenue = _digrin_cost_of_revenue.apply(safe_numeric_conversion)
# Digrin Upcoming Estimated Earnings
digrin_upcoming_estimated_earnings = _digrin_upcoming_estimated_earnings.copy()
digrin_upcoming_estimated_earnings["Low Revenue"] = safe_numeric_conversion(
    digrin_upcoming_estimated_earnings["Low Revenue"]
)
digrin_upcoming_estimated_earnings["Actual / Estimated Revenue"] = safe_numeric_conversion(
    digrin_upcoming_estimated_earnings["Actual / Estimated Revenue"]
)
digrin_upcoming_estimated_earnings["High Revenue"] = safe_numeric_conversion(
    digrin_upcoming_estimated_earnings["High Revenue"]
)

# +
print("--- Profitability & Economic Moat ---")

print("\nreturnOnEquity:")
# Calculation using Macrotrends Income Statement and Balance Sheet
# Latest available year for Macrotrends Income Statement is '2024-09-30'
# Latest available year for Macrotrends Balance Sheet is '2024-09-30'
net_income_macro_2024 = macrotrends_income_statement.loc["Net Income", "2024-09-30"]
share_holder_equity_macro_2024 = macrotrends_balance_sheet.loc["Share Holder Equity", "2024-09-30"]
if (
    not pd.isna(net_income_macro_2024)
    and not pd.isna(share_holder_equity_macro_2024)
    and share_holder_equity_macro_2024 != 0
):
    print(f"Macrotrends (2024-09-30): {net_income_macro_2024 / share_holder_equity_macro_2024:.4f}")
else:
    print("Macrotrends (2024-09-30): Data not available for calculation.")

# +

# Calculation using Yahoo Web Income Statement and Yahoo Web Balance Sheet
net_income_web_2024 = yahoo_web_income_statement.loc["Net Income Common Stockholders", "9/30/2024"]
total_equity_web_2024 = yahoo_web_balance_sheet.loc["Total Equity Gross Minority Interest", "9/30/2024"]
if not pd.isna(net_income_web_2024) and not pd.isna(total_equity_web_2024) and total_equity_web_2024 != 0:
    print(f"Yahoo Web (9/30/2024): {net_income_web_2024 / total_equity_web_2024:.4f}")
else:
    print("Yahoo Web (9/30/2024): Data not available for calculation.")

# +

# Calculation using Yahoo API Financials and Yahoo API Balance Sheet
net_income_api_2023 = yahoo_api_financials.loc["2023-09-30", "NetIncomeCommonStockholders"]
stockholders_equity_api_2023 = yahoo_api_balance_sheet.loc["2023-09-30", "StockholdersEquity"]
if not pd.isna(net_income_api_2023) and not pd.isna(stockholders_equity_api_2023) and stockholders_equity_api_2023 != 0:
    print(f"Yahoo API (2023-09-30): {net_income_api_2023 / stockholders_equity_api_2023:.4f}")
else:
    print("Yahoo API (2023-09-30): Data not available for calculation.")

# +

print("\nreturnOnAssets:")

# Calculation using Macrotrends Income Statement and Balance Sheet
net_income_macro_2024 = macrotrends_income_statement.loc["Net Income", "2024-09-30"]
total_assets_macro_2024 = macrotrends_balance_sheet.loc["Total Assets", "2024-09-30"]
if not pd.isna(net_income_macro_2024) and not pd.isna(total_assets_macro_2024) and total_assets_macro_2024 != 0:
    print(f"Macrotrends (2024-09-30): {net_income_macro_2024 / total_assets_macro_2024:.4f}")
else:
    print("Macrotrends (2024-09-30): Data not available for calculation.")

# +

# Calculation using Yahoo Web Income Statement and Yahoo Web Balance Sheet
net_income_web_2024 = yahoo_web_income_statement.loc["Net Income Common Stockholders", "9/30/2024"]
total_assets_web_2024 = yahoo_web_balance_sheet.loc["Total Assets", "9/30/2024"]
if not pd.isna(net_income_web_2024) and not pd.isna(total_assets_web_2024) and total_assets_web_2024 != 0:
    print(f"Yahoo Web (9/30/2024): {net_income_web_2024 / total_assets_web_2024:.4f}")
else:
    print("Yahoo Web (9/30/2024): Data not available for calculation.")

# +

# Calculation using Yahoo API Financials and Yahoo API Balance Sheet
net_income_api_2023 = yahoo_api_financials.loc["2023-09-30", "NetIncomeCommonStockholders"]
total_assets_api_2023 = yahoo_api_balance_sheet.loc["2023-09-30", "TotalAssets"]
if not pd.isna(net_income_api_2023) and not pd.isna(total_assets_api_2023) and total_assets_api_2023 != 0:
    print(f"Yahoo API (2023-09-30): {net_income_api_2023 / total_assets_api_2023:.4f}")
else:
    print("Yahoo API (2023-09-30): Data not available for calculation.")

# +

print("\noperatingMargins:")
# From Macrotrends Operating Margin
if not macrotrends_operating_margin.empty:
    print(
        f"Macrotrends Operating Margin (latest): {macrotrends_operating_margin.loc[macrotrends_operating_margin.index[-1], 'Operating Margin']:.4f}"
    )
else:
    print("Macrotrends Operating Margin: Data not available.")

# +

# Calculation using Macrotrends Income Statement
operating_income_macro_2024 = macrotrends_income_statement.loc["Operating Income", "2024-09-30"]
revenue_macro_2024 = macrotrends_income_statement.loc["Revenue", "2024-09-30"]
if not pd.isna(operating_income_macro_2024) and not pd.isna(revenue_macro_2024) and revenue_macro_2024 != 0:
    print(f"Macrotrends (2024-09-30) calculated: {operating_income_macro_2024 / revenue_macro_2024:.4f}")
else:
    print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo Web Income Statement
operating_income_web_2024 = yahoo_web_income_statement.loc["Operating Income", "9/30/2024"]
total_revenue_web_2024 = yahoo_web_income_statement.loc["Total Revenue", "9/30/2024"]
if not pd.isna(operating_income_web_2024) and not pd.isna(total_revenue_web_2024) and total_revenue_web_2024 != 0:
    print(f"Yahoo Web (9/30/2024) calculated: {operating_income_web_2024 / total_revenue_web_2024:.4f}")
else:
    print("Yahoo Web (9/30/2024) calculated: Data not available for calculation.")

# +

# Using Yahoo API Income Statement (Quarterly)
operating_income_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "OperatingIncome"]
total_revenue_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "TotalRevenue"]
if (
    not pd.isna(operating_income_api_q3_2024)
    and not pd.isna(total_revenue_api_q3_2024)
    and total_revenue_api_q3_2024 != 0
):
    print(
        f"Yahoo API Quarterly (2024-09-30) calculated: {operating_income_api_q3_2024 / total_revenue_api_q3_2024:.4f}"
    )
else:
    print("Yahoo API Quarterly (2024-09-30) calculated: Data not available for calculation.")

# +

# Using Yahoo API Financials
operating_income_api_2023 = yahoo_api_financials.loc["2023-09-30", "OperatingIncome"]
total_revenue_api_2023 = yahoo_api_financials.loc["2023-09-30", "TotalRevenue"]
if not pd.isna(operating_income_api_2023) and not pd.isna(total_revenue_api_2023) and total_revenue_api_2023 != 0:
    print(f"Yahoo API Financials (2023-09-30) calculated: {operating_income_api_2023 / total_revenue_api_2023:.4f}")
else:
    print("Yahoo API Financials (2023-09-30) calculated: Data not available for calculation.")

# +

print("\nebitdaMargins:")
# From Macrotrends EBITDA Margin
if not macrotrends_ebitda_margin.empty:
    print(
        f"Macrotrends EBITDA Margin (latest): {macrotrends_ebitda_margin.loc[macrotrends_ebitda_margin.index[-1], 'EBITDA Margin']:.4f}"
    )
else:
    print("Macrotrends EBITDA Margin: Data not available.")

# +

# Calculation using Macrotrends Income Statement
ebitda_macro_2024 = macrotrends_income_statement.loc["EBITDA", "2024-09-30"]
revenue_macro_2024 = macrotrends_income_statement.loc["Revenue", "2024-09-30"]
if not pd.isna(ebitda_macro_2024) and not pd.isna(revenue_macro_2024) and revenue_macro_2024 != 0:
    print(f"Macrotrends (2024-09-30) calculated: {ebitda_macro_2024 / revenue_macro_2024:.4f}")
else:
    print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo Web Income Statement
ebitda_web_2024 = yahoo_web_income_statement.loc["EBITDA", "9/30/2024"]
total_revenue_web_2024 = yahoo_web_income_statement.loc["Total Revenue", "9/30/2024"]
if not pd.isna(ebitda_web_2024) and not pd.isna(total_revenue_web_2024) and total_revenue_web_2024 != 0:
    print(f"Yahoo Web (9/30/2024) calculated: {ebitda_web_2024 / total_revenue_web_2024:.4f}")
else:
    print("Yahoo Web (9/30/2024) calculated: Data not available for calculation.")

# +

# Using Yahoo API Income Statement (Quarterly)
ebitda_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "EBITDA"]
total_revenue_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "TotalRevenue"]
if not pd.isna(ebitda_api_q3_2024) and not pd.isna(total_revenue_api_q3_2024) and total_revenue_api_q3_2024 != 0:
    print(f"Yahoo API Quarterly (2024-09-30) calculated: {ebitda_api_q3_2024 / total_revenue_api_q3_2024:.4f}")
else:
    print("Yahoo API Quarterly (2024-09-30) calculated: Data not available for calculation.")

# +

# Using Yahoo API Financials
ebitda_api_2023 = yahoo_api_financials.loc["2023-09-30", "EBITDA"]
total_revenue_api_2023 = yahoo_api_financials.loc["2023-09-30", "TotalRevenue"]
if not pd.isna(ebitda_api_2023) and not pd.isna(total_revenue_api_2023) and total_revenue_api_2023 != 0:
    print(f"Yahoo API Financials (2023-09-30) calculated: {ebitda_api_2023 / total_revenue_api_2023:.4f}")
else:
    print("Yahoo API Financials (2023-09-30) calculated: Data not available for calculation.")

# +

print("\ngrossMargins:")
# From Macrotrends Gross Margin
if not macrotrends_gross_margin.empty:
    print(
        f"Macrotrends Gross Margin (latest): {macrotrends_gross_margin.loc[macrotrends_gross_margin.index[-1], 'Gross Margin']:.4f}"
    )
else:
    print("Macrotrends Gross Margin: Data not available.")

# +

# Calculation using Macrotrends Income Statement
gross_profit_macro_2024 = macrotrends_income_statement.loc["Gross Profit", "2024-09-30"]
revenue_macro_2024 = macrotrends_income_statement.loc["Revenue", "2024-09-30"]
if not pd.isna(gross_profit_macro_2024) and not pd.isna(revenue_macro_2024) and revenue_macro_2024 != 0:
    print(f"Macrotrends (2024-09-30) calculated: {gross_profit_macro_2024 / revenue_macro_2024:.4f}")
else:
    print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo Web Income Statement
gross_profit_web_2024 = yahoo_web_income_statement.loc["Gross Profit", "9/30/2024"]
total_revenue_web_2024 = yahoo_web_income_statement.loc["Total Revenue", "9/30/2024"]
if not pd.isna(gross_profit_web_2024) and not pd.isna(total_revenue_web_2024) and total_revenue_web_2024 != 0:
    print(f"Yahoo Web (9/30/2024) calculated: {gross_profit_web_2024 / total_revenue_web_2024:.4f}")
else:
    print("Yahoo Web (9/30/2024) calculated: Data not available for calculation.")

# +

# Using Yahoo API Income Statement (Quarterly)
gross_profit_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "GrossProfit"]
total_revenue_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "TotalRevenue"]
if not pd.isna(gross_profit_api_q3_2024) and not pd.isna(total_revenue_api_q3_2024) and total_revenue_api_q3_2024 != 0:
    print(f"Yahoo API Quarterly (2024-09-30) calculated: {gross_profit_api_q3_2024 / total_revenue_api_q3_2024:.4f}")
else:
    print("Yahoo API Quarterly (2024-09-30) calculated: Data not available for calculation.")

# +

# Using Yahoo API Financials
gross_profit_api_2023 = yahoo_api_financials.loc["2023-09-30", "GrossProfit"]
total_revenue_api_2023 = yahoo_api_financials.loc["2023-09-30", "TotalRevenue"]
if not pd.isna(gross_profit_api_2023) and not pd.isna(total_revenue_api_2023) and total_revenue_api_2023 != 0:
    print(f"Yahoo API Financials (2023-09-30) calculated: {gross_profit_api_2023 / total_revenue_api_2023:.4f}")
else:
    print("Yahoo API Financials (2023-09-30) calculated: Data not available for calculation.")

# +

print("\nprofitMargins:")
# From Macrotrends Net Margin
if not macrotrends_net_margin.empty:
    print(
        f"Macrotrends Net Margin (latest): {macrotrends_net_margin.loc[macrotrends_net_margin.index[-1], 'Net Margin']:.4f}"
    )
else:
    print("Macrotrends Net Margin: Data not available.")

# +

# Calculation using Macrotrends Income Statement
net_income_macro_2024 = macrotrends_income_statement.loc["Net Income", "2024-09-30"]
revenue_macro_2024 = macrotrends_income_statement.loc["Revenue", "2024-09-30"]
if not pd.isna(net_income_macro_2024) and not pd.isna(revenue_macro_2024) and revenue_macro_2024 != 0:
    print(f"Macrotrends (2024-09-30) calculated: {net_income_macro_2024 / revenue_macro_2024:.4f}")
else:
    print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo Web Income Statement
net_income_web_2024 = yahoo_web_income_statement.loc["Net Income Common Stockholders", "9/30/2024"]
total_revenue_web_2024 = yahoo_web_income_statement.loc["Total Revenue", "9/30/2024"]
if not pd.isna(net_income_web_2024) and not pd.isna(total_revenue_web_2024) and total_revenue_web_2024 != 0:
    print(f"Yahoo Web (9/30/2024) calculated: {net_income_web_2024 / total_revenue_web_2024:.4f}")
else:
    print("Yahoo Web (9/30/2024) calculated: Data not available for calculation.")

# +

# Using Yahoo API Income Statement (Quarterly)
net_income_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "NetIncome"]
total_revenue_api_q3_2024 = yahoo_api_income_statement_quarterly.loc["2024-09-30", "TotalRevenue"]
if not pd.isna(net_income_api_q3_2024) and not pd.isna(total_revenue_api_q3_2024) and total_revenue_api_q3_2024 != 0:
    print(f"Yahoo API Quarterly (2024-09-30) calculated: {net_income_api_q3_2024 / total_revenue_api_q3_2024:.4f}")
else:
    print("Yahoo API Quarterly (2024-09-30) calculated: Data not available for calculation.")

# +

# Using Yahoo API Financials
net_income_api_2023 = yahoo_api_financials.loc["2023-09-30", "NetIncome"]
total_revenue_api_2023 = yahoo_api_financials.loc["2023-09-30", "TotalRevenue"]
if not pd.isna(net_income_api_2023) and not pd.isna(total_revenue_api_2023) and total_revenue_api_2023 != 0:
    print(f"Yahoo API Financials (2023-09-30) calculated: {net_income_api_2023 / total_revenue_api_2023:.4f}")
else:
    print("Yahoo API Financials (2023-09-30) calculated: Data not available for calculation.")

print("\n--- Growth Sustainability ---")

# +

print("\nrevenueGrowth:")
# Macrotrends Income Statement
if len(macrotrends_income_statement.columns) >= 2:
    revenue_current = macrotrends_income_statement.loc["Revenue", "2024-09-30"]
    revenue_previous = macrotrends_income_statement.loc["Revenue", "2023-09-30"]
    if not pd.isna(revenue_current) and not pd.isna(revenue_previous) and revenue_previous != 0:
        print(f"Macrotrends (2024-09-30 vs 2023-09-30): {(revenue_current / revenue_previous - 1):.4f}")
    else:
        print("Macrotrends (2024-09-30 vs 2023-09-30): Data not available for calculation.")

# +

# Yahoo Web Income Statement
if len(yahoo_web_income_statement.columns) >= 2:
    revenue_web_current = yahoo_web_income_statement.loc["Total Revenue", "9/30/2024"]
    revenue_web_previous = yahoo_web_income_statement.loc["Total Revenue", "9/30/2023"]
    if not pd.isna(revenue_web_current) and not pd.isna(revenue_web_previous) and revenue_web_previous != 0:
        print(f"Yahoo Web (9/30/2024 vs 9/30/2023): {(revenue_web_current / revenue_web_previous - 1):.4f}")
    else:
        print("Yahoo Web (9/30/2024 vs 9/30/2023): Data not available for calculation.")

# +

# Yahoo API Financials
if len(yahoo_api_financials.index) >= 2:
    revenue_api_current = yahoo_api_financials.loc["2023-09-30", "TotalRevenue"]
    revenue_api_previous = yahoo_api_financials.loc["2022-09-30", "TotalRevenue"]
    if not pd.isna(revenue_api_current) and not pd.isna(revenue_api_previous) and revenue_api_previous != 0:
        print(
            f"Yahoo API Financials (2023-09-30 vs 2022-09-30): {(revenue_api_current / revenue_api_previous - 1):.4f}"
        )
    else:
        print("Yahoo API Financials (2023-09-30 vs 2022-09-30): Data not available for calculation.")

# +

print("\nearningsGrowth:")
# Macrotrends Income Statement
if len(macrotrends_income_statement.columns) >= 2:
    net_income_current = macrotrends_income_statement.loc["Net Income", "2024-09-30"]
    net_income_previous = macrotrends_income_statement.loc["Net Income", "2023-09-30"]
    if not pd.isna(net_income_current) and not pd.isna(net_income_previous) and net_income_previous != 0:
        print(f"Macrotrends (2024-09-30 vs 2023-09-30): {(net_income_current / net_income_previous - 1):.4f}")
    else:
        print("Macrotrends (2024-09-30 vs 2023-09-30): Data not available for calculation.")

# +

# Yahoo Web Income Statement
if len(yahoo_web_income_statement.columns) >= 2:
    net_income_web_current = yahoo_web_income_statement.loc["Net Income Common Stockholders", "9/30/2024"]
    net_income_web_previous = yahoo_web_income_statement.loc["Net Income Common Stockholders", "9/30/2023"]
    if not pd.isna(net_income_web_current) and not pd.isna(net_income_web_previous) and net_income_web_previous != 0:
        print(f"Yahoo Web (9/30/2024 vs 9/30/2023): {(net_income_web_current / net_income_web_previous - 1):.4f}")
    else:
        print("Yahoo Web (9/30/2024 vs 9/30/2023): Data not available for calculation.")

# +

# Yahoo API Financials
if len(yahoo_api_financials.index) >= 2:
    net_income_api_current = yahoo_api_financials.loc["2023-09-30", "NetIncomeCommonStockholders"]
    net_income_api_previous = yahoo_api_financials.loc["2022-09-30", "NetIncomeCommonStockholders"]
    if not pd.isna(net_income_api_current) and not pd.isna(net_income_api_previous) and net_income_api_previous != 0:
        print(
            f"Yahoo API Financials (2023-09-30 vs 2022-09-30): {(net_income_api_current / net_income_api_previous - 1):.4f}"
        )
    else:
        print("Yahoo API Financials (2023-09-30 vs 2022-09-30): Data not available for calculation.")

# +

print("\nearningsQuarterlyGrowth:")
# Yahoo API Income Statement (Quarterly)
if len(yahoo_api_income_statement_quarterly.index) >= 2:
    net_income_q_current = yahoo_api_income_statement_quarterly.loc["2025-03-31", "NetIncome"]
    net_income_q_previous = yahoo_api_income_statement_quarterly.loc["2024-12-31", "NetIncome"]
    if not pd.isna(net_income_q_current) and not pd.isna(net_income_q_previous) and net_income_q_previous != 0:
        print(
            f"Yahoo API Quarterly (2025-03-31 vs 2024-12-31): {(net_income_q_current / net_income_q_previous - 1):.4f}"
        )
    else:
        print("Yahoo API Quarterly (2025-03-31 vs 2024-12-31): Data not available for calculation.")

# +

print("\nfreeCashflow:")
# Yahoo Web Cashflow (TTM)
if "Free Cash Flow" in yahoo_web_cashflow.index:
    print(f"Yahoo Web Cashflow (TTM): {yahoo_web_cashflow.loc['Free Cash Flow', 'TTM']:.2f}")
else:
    print("Yahoo Web Cashflow (TTM): Data not available.")

# +

# Digrin Free Cash Flow (latest)
if not digrin_free_cash_flow.empty:
    print(
        f"Digrin Free Cash Flow (latest): {digrin_free_cash_flow.loc[digrin_free_cash_flow.index[0], 'Free Cash Flow']:.2f}"
    )
else:
    print("Digrin Free Cash Flow: Data not available.")

# +

# Calculation from Yahoo Web Cashflow (Operating Cash Flow - Capital Expenditure) for a specific year
if "Operating Cash Flow" in yahoo_web_cashflow.index and "Capital Expenditure" in yahoo_web_cashflow.index:
    op_cash_flow_web_2024 = yahoo_web_cashflow.loc["Operating Cash Flow", "9/30/2024"]
    capex_web_2024 = yahoo_web_cashflow.loc["Capital Expenditure", "9/30/2024"]
    if not pd.isna(op_cash_flow_web_2024) and not pd.isna(capex_web_2024):
        print(f"Yahoo Web (9/30/2024) calculated: {op_cash_flow_web_2024 - capex_web_2024:.2f}")
    else:
        print("Yahoo Web (9/30/2024) calculated: Data not available for calculation.")

# +

# Calculation from Macrotrends Cash Flow (Cash Flow From Operating Activities - Net Change In Property, Plant, And Equipment) for a specific year
if (
    "Cash Flow From Operating Activities" in macrotrends_cash_flow.index
    and "Net Change In Property, Plant, And Equipment" in macrotrends_cash_flow.index
):
    op_cash_flow_macro_2024 = macrotrends_cash_flow.loc["Cash Flow From Operating Activities", "2024-09-30"]
    ppe_change_macro_2024 = macrotrends_cash_flow.loc["Net Change In Property, Plant, And Equipment", "2024-09-30"]
    if not pd.isna(op_cash_flow_macro_2024) and not pd.isna(ppe_change_macro_2024):
        print(
            f"Macrotrends (2024-09-30) calculated: {op_cash_flow_macro_2024 + ppe_change_macro_2024:.2f}"
        )  # Note: PPE change is usually negative for capex
    else:
        print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation from Yahoo API Cash Flow (OperatingCashFlow - CapitalExpenditure) for a specific year
if "OperatingCashFlow" in yahoo_api_cash_flow.columns and "CapitalExpenditure" in yahoo_api_cash_flow.columns:
    op_cash_flow_api_2024 = yahoo_api_cash_flow.loc["2024-09-30", "OperatingCashFlow"]
    capex_api_2024 = yahoo_api_cash_flow.loc["2024-09-30", "CapitalExpenditure"]
    if not pd.isna(op_cash_flow_api_2024) and not pd.isna(capex_api_2024):
        print(
            f"Yahoo API (2024-09-30) calculated: {op_cash_flow_api_2024 + capex_api_2024:.2f}"
        )  # Note: CapitalExpenditure is typically negative
    else:
        print("Yahoo API (2024-09-30) calculated: Data not available for calculation.")

print("\n--- Balance-Sheet Resilience ---")

# +

print("\ntotalCash:")
# Macrotrends Balance Sheet
if "Cash On Hand" in macrotrends_balance_sheet.index:
    print(f"Macrotrends Balance Sheet (2024-09-30): {macrotrends_balance_sheet.loc['Cash On Hand', '2024-09-30']:.2f}")
else:
    print("Macrotrends Balance Sheet: Cash On Hand data not available.")

# +

# Digrin Cash and Debt
if not digrin_cash_and_debt.empty:
    print(f"Digrin Cash and Debt (latest): {digrin_cash_and_debt.loc[0, 'Cash']:.2f}")
else:
    print("Digrin Cash and Debt: Data not available.")

# +

# Yahoo API Balance Sheet
if "CashAndCashEquivalents" in yahoo_api_balance_sheet.columns:
    print(
        f"Yahoo API Balance Sheet (2024-09-30): {yahoo_api_balance_sheet.loc['2024-09-30', 'CashAndCashEquivalents']:.2f}"
    )
else:
    print("Yahoo API Balance Sheet: CashAndCashEquivalents data not available.")

# +

print("\ntotalDebt:")
# Yahoo Web Balance Sheet
if "Total Debt" in yahoo_web_balance_sheet.index:
    print(f"Yahoo Web Balance Sheet (9/30/2024): {yahoo_web_balance_sheet.loc['Total Debt', '9/30/2024']:.2f}")
else:
    print("Yahoo Web Balance Sheet: Total Debt data not available.")

# +

# Digrin Cash and Debt
if not digrin_cash_and_debt.empty:
    print(f"Digrin Cash and Debt (latest): {digrin_cash_and_debt.loc[0, 'Debt']:.2f}")
else:
    print("Digrin Cash and Debt: Data not available.")

# +

# Yahoo API Balance Sheet
if "TotalDebt" in yahoo_api_balance_sheet.columns:
    print(f"Yahoo API Balance Sheet (2024-09-30): {yahoo_api_balance_sheet.loc['2024-09-30', 'TotalDebt']:.2f}")
else:
    print("Yahoo API Balance Sheet: TotalDebt data not available.")

# +

print("\ndebtToEquity:")
# Calculation using Macrotrends Balance Sheet
if "Total Liabilities" in macrotrends_balance_sheet.index and "Share Holder Equity" in macrotrends_balance_sheet.index:
    total_liabilities_macro_2024 = macrotrends_balance_sheet.loc["Total Liabilities", "2024-09-30"]
    share_holder_equity_macro_2024 = macrotrends_balance_sheet.loc[
        "Share Holder Equity", "2024-09-30"
    ]  # Corrected typo here
    if (
        not pd.isna(total_liabilities_macro_2024)
        and not pd.isna(share_holder_equity_macro_2024)
        and share_holder_equity_macro_2024 != 0
    ):
        print(
            f"Macrotrends (2024-09-30) calculated: {total_liabilities_macro_2024 / share_holder_equity_macro_2024:.4f}"
        )
    else:
        print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo Web Balance Sheet
if (
    "Total Liabilities Net Minority Interest" in yahoo_web_balance_sheet.index
    and "Total Equity Gross Minority Interest" in yahoo_web_balance_sheet.index
):
    total_liabilities_web_2024 = yahoo_web_balance_sheet.loc["Total Liabilities Net Minority Interest", "9/30/2024"]
    total_equity_web_2024 = yahoo_web_balance_sheet.loc["Total Equity Gross Minority Interest", "9/30/2024"]
    if not pd.isna(total_liabilities_web_2024) and not pd.isna(total_equity_web_2024) and total_equity_web_2024 != 0:
        print(f"Yahoo Web (9/30/2024) calculated: {total_liabilities_web_2024 / total_equity_web_2024:.4f}")
    else:
        print("Yahoo Web (9/30/2024) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo API Balance Sheet
if (
    "TotalLiabilitiesNetMinorityInterest" in yahoo_api_balance_sheet.columns
    and "StockholdersEquity" in yahoo_api_balance_sheet.columns
):
    total_liabilities_api_2024 = yahoo_api_balance_sheet.loc["2024-09-30", "TotalLiabilitiesNetMinorityInterest"]
    stockholders_equity_api_2024 = yahoo_api_balance_sheet.loc["2024-09-30", "StockholdersEquity"]
    if (
        not pd.isna(total_liabilities_api_2024)
        and not pd.isna(stockholders_equity_api_2024)
        and stockholders_equity_api_2024 != 0
    ):
        print(f"Yahoo API (2024-09-30) calculated: {total_liabilities_api_2024 / stockholders_equity_api_2024:.4f}")
    else:
        print("Yahoo API (2024-09-30) calculated: Data not available for calculation.")

# +

print("\ncurrentRatio:")
# Calculation using Macrotrends Balance Sheet
if (
    "Total Current Assets" in macrotrends_balance_sheet.index
    and "Total Current Liabilities" in macrotrends_balance_sheet.index
):
    current_assets_macro_2024 = macrotrends_balance_sheet.loc["Total Current Assets", "2024-09-30"]
    current_liabilities_macro_2024 = macrotrends_balance_sheet.loc["Total Current Liabilities", "2024-09-30"]
    if (
        not pd.isna(current_assets_macro_2024)
        and not pd.isna(current_liabilities_macro_2024)
        and current_liabilities_macro_2024 != 0
    ):
        print(f"Macrotrends (2024-09-30) calculated: {current_assets_macro_2024 / current_liabilities_macro_2024:.4f}")
    else:
        print("Macrotrends (2024-09-30) calculated: Data not available for calculation.")

# +

# Calculation using Yahoo API Balance Sheet
if "CurrentAssets" in yahoo_api_balance_sheet.columns and "CurrentLiabilities" in yahoo_api_balance_sheet.columns:
    current_assets_api_2024 = yahoo_api_balance_sheet.loc["2024-09-30", "CurrentAssets"]
    current_liabilities_api_2024 = yahoo_api_balance_sheet.loc["2024-09-30", "CurrentLiabilities"]
    if (
        not pd.isna(current_assets_api_2024)
        and not pd.isna(current_liabilities_api_2024)
        and current_liabilities_api_2024 != 0
    ):
        print(f"Yahoo API (2024-09-30) calculated: {current_assets_api_2024 / current_liabilities_api_2024:.4f}")
    else:
        print("Yahoo API (2024-09-30) calculated: Data not available for calculation.")

# +

print("\n--- Capital-Allocation Track Record ---")

print("\ndividendYield:")
# Yahoo Web Trading Information
if "Forward Annual Dividend Yield 4" in yahoo_web_trading_information.index:
    print(
        f"Yahoo Web Trading Information (Forward Annual Dividend Yield): {yahoo_web_trading_information.loc['Forward Annual Dividend Yield 4', 'Value']:.4f}"
    )
else:
    print("Yahoo Web Trading Information: Forward Annual Dividend Yield data not available.")
if "Trailing Annual Dividend Yield 3" in yahoo_web_trading_information.index:
    print(
        f"Yahoo Web Trading Information (Trailing Annual Dividend Yield): {yahoo_web_trading_information.loc['Trailing Annual Dividend Yield 3', 'Value']:.4f}"
    )
else:
    print("Yahoo Web Trading Information: Trailing Annual Dividend Yield data not available.")

# +

# Digrin DGR3
if not digrin_dgr3.empty:
    print(
        f"Digrin DGR3 (2025 Estimated Yield on Cost): {digrin_dgr3.loc[digrin_dgr3.index[digrin_dgr3['Year'] == 2025].tolist()[0], 'Estimated Yield on Cost']:.4f}"
    )
else:
    print("Digrin DGR3: Data not available.")

# +

# Calculation (Latest Dividend / Latest Close Price)
if not digrin_dividend_data.empty and not yahoo_api_price_data.empty:
    latest_dividend = digrin_dividend_data.loc[0, "Dividend amount (change)"]
    latest_close_price = yahoo_api_price_data.loc[yahoo_api_price_data.index[-1], "close"]
    if not pd.isna(latest_dividend) and not pd.isna(latest_close_price) and latest_close_price != 0:
        print(f"Calculated (Latest Dividend / Latest Close Price): {latest_dividend / latest_close_price:.4f}")
    else:
        print("Calculated (Latest Dividend / Latest Close Price): Data not available for calculation.")

# +

print("\npayoutRatio:")
# Yahoo Web Trading Information
if "Payout Ratio 4" in yahoo_web_trading_information.index:
    print(f"Yahoo Web Trading Information: {yahoo_web_trading_information.loc['Payout Ratio 4', 'Value']:.4f}")
else:
    print("Yahoo Web Trading Information: Payout Ratio data not available.")

# +

# Digrin Payout Ratio
if not digrin_payout_ratio.empty:
    print(f"Digrin Payout Ratio (latest): {digrin_payout_ratio.loc[digrin_payout_ratio.index[0], 'Payout ratio']:.4f}")
else:
    print("Digrin Payout Ratio: Data not available.")

# +

print("\nfiveYearAvgDividendYield:")
# Yahoo Web Trading Information
if "5 Year Average Dividend Yield 4" in yahoo_web_trading_information.index:
    print(
        f"Yahoo Web Trading Information: {yahoo_web_trading_information.loc['5 Year Average Dividend Yield 4', 'Value']:.4f}"
    )
else:
    print("Yahoo Web Trading Information: 5 Year Average Dividend Yield data not available.")

print("\n--- Valuation vs. Quality (entry discipline) ---")

# +

print("\nforwardPE:")
# Yahoo Web Valuation Measures
if "Forward P/E" in yahoo_web_valuation_measures.index:
    print(f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Forward P/E', 'Current']:.2f}")
else:
    print("Yahoo Web Valuation Measures: Forward P/E data not available.")

# +

print("\ntrailingPE:")
# Yahoo Web Summary
if "trailingPE" in yahoo_web_summary.index:
    print(f"Yahoo Web Summary: {yahoo_web_summary.loc['trailingPE', 0]:.2f}")
else:
    print("Yahoo Web Summary: trailingPE data not available.")

# +

# Yahoo Web Valuation Measures
if "Trailing P/E" in yahoo_web_valuation_measures.index:
    print(f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Trailing P/E', 'Current']:.2f}")
else:
    print("Yahoo Web Valuation Measures: Trailing P/E data not available.")

# +

print("\npriceToBook:")
# Yahoo Web Valuation Measures
if "Price/Book" in yahoo_web_valuation_measures.index:
    print(f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Price/Book', 'Current']:.2f}")
else:
    print("Yahoo Web Valuation Measures: Price/Book data not available.")

# +

print("\npriceToSalesTrailing12Months:")
# Yahoo Web Valuation Measures
if "Price/Sales" in yahoo_web_valuation_measures.index:
    print(f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Price/Sales', 'Current']:.2f}")
else:
    print("Yahoo Web Valuation Measures: Price/Sales data not available.")

# +

print("\nenterpriseToEbitda:")
# Yahoo Web Valuation Measures
if "Enterprise Value/EBITDA" in yahoo_web_valuation_measures.index:
    print(
        f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Enterprise Value/EBITDA', 'Current']:.2f}"
    )
else:
    print("Yahoo Web Valuation Measures: Enterprise Value/EBITDA data not available.")

# +

print("\ntrailingPegRatio:")
# Yahoo Web Valuation Measures (PEG Ratio is the closest, based on 5yr expected growth)
if "PEG Ratio (5yr expected)" in yahoo_web_valuation_measures.index:
    print(
        f"Yahoo Web Valuation Measures (PEG Ratio (5yr expected) for Current): {yahoo_web_valuation_measures.loc['PEG Ratio (5yr expected)', 'Current']:.2f}"
    )
else:
    print("Yahoo Web Valuation Measures: PEG Ratio data not available.")

# +

print("\n--- Ownership & Liquidity ---")

print("\nheldPercentInsiders:")
# Yahoo Web Trading Information
if "% Held by Insiders 1" in yahoo_web_trading_information.index:
    print(
        f"Yahoo Web Trading Information: {yahoo_web_trading_information.loc['% Held by Insiders 1', 'Value'] / 100:.4f}"
    )
else:
    print("Yahoo Web Trading Information: % Held by Insiders data not available.")
# Yahoo Web Major Holders
if not yahoo_web_major_holders.empty and 0 in yahoo_web_major_holders.index and 0 in yahoo_web_major_holders.columns:
    print(f"Yahoo Web Major Holders: {yahoo_web_major_holders.loc[0, 0] / 100:.4f}")
else:
    print("Yahoo Web Major Holders: % of Shares Held by All Insider data not available.")

# +

print("\nheldPercentInstitutions:")
# Yahoo Web Trading Information
if "% Held by Institutions 1" in yahoo_web_trading_information.index:
    print(
        f"Yahoo Web Trading Information: {yahoo_web_trading_information.loc['% Held by Institutions 1', 'Value'] / 100:.4f}"
    )
else:
    print("Yahoo Web Trading Information: % Held by Institutions data not available.")
# Yahoo Web Major Holders
if not yahoo_web_major_holders.empty and 1 in yahoo_web_major_holders.index and 0 in yahoo_web_major_holders.columns:
    print(f"Yahoo Web Major Holders: {yahoo_web_major_holders.loc[1, 0] / 100:.4f}")
else:
    print("Yahoo Web Major Holders: % of Shares Held by Institutions data not available.")

# +

print("\nmarketCap:")
# Yahoo Web Summary
if "marketCap" in yahoo_web_summary.index:
    print(f"Yahoo Web Summary: {yahoo_web_summary.loc['marketCap', 0]:.2f}")
else:
    print("Yahoo Web Summary: marketCap data not available.")
# Yahoo Web Valuation Measures
if "Market Cap" in yahoo_web_valuation_measures.index:
    print(f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Market Cap', 'Current']:.2f}")
else:
    print("Yahoo Web Valuation Measures: Market Cap data not available.")

# +

print("\nbeta:")
# Yahoo Web Trading Information
if "Beta (5Y Monthly)" in yahoo_web_trading_information.index:
    print(f"Yahoo Web Trading Information: {yahoo_web_trading_information.loc['Beta (5Y Monthly)', 'Value']:.2f}")
else:
    print("Yahoo Web Trading Information: Beta data not available.")

# +

print("\n--- Holistic Valuation Context ---")

print("\nenterpriseValue:")
# Yahoo Web Valuation Measures
if "Enterprise Value" in yahoo_web_valuation_measures.index:
    print(
        f"Yahoo Web Valuation Measures (Current): {yahoo_web_valuation_measures.loc['Enterprise Value', 'Current']:.2f}"
    )
else:
    print("Yahoo Web Valuation Measures: Enterprise Value data not available.")

# Yahoo Web Summary (specific columns)
yahoo_web_summary = _yahoo_web_summary.copy()
