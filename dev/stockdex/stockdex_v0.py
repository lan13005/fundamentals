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
from datetime import datetime

import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from stockdex import Ticker

ticker = Ticker(ticker="AAPL")

console = Console()


def print_dataframe_with_metadata(df: pd.DataFrame, title: str):
    """Print a dataframe with rich formatting and metadata."""
    # Print metadata
    metadata = {"shape": df.shape, "columns": list(df.columns), "index": list(df.index), "dtypes": df.dtypes.to_dict()}

    console.print(f"\n[bold blue]{title} Metadata:[/bold blue]")
    rprint(metadata)

    # Create rich table
    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    for column in df.columns:
        table.add_column(str(column))

    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)


# +
# Price data (use range and dataGranularity to make range and granularity more specific) from Yahoo Finance API
print_dataframe_with_metadata(ticker.yahoo_api_price(range="1y", dataGranularity="1d"), "Yahoo API Price Data")

# # plot financial data using Plotly
# ticker = Ticker(ticker="MSFT")
# ticker.plot_yahoo_api_financials(group_by="field")

# +

# Complete historical data of the stock in certain categories from digrin website
print_dataframe_with_metadata(ticker.digrin_dividend, "Digrin Dividend Data")

# +

# Financial data from macrotrends website
print_dataframe_with_metadata(ticker.macrotrends_income_statement, "Macrotrends Income Statement")

# +

# Summary including general financial information from Yahoo Finance website
print_dataframe_with_metadata(ticker.yahoo_web_summary, "Yahoo Web Summary")

# # Yahoo Finance Data

# +

# Current trading period of the stock (pre-market, regular, post-market trading periods)
print_dataframe_with_metadata(ticker.yahoo_api_current_trading_period, "Current Trading Period")

# +

# Fundamental data (use frequency, format, period1 and period2 to fine-tune the returned data)
print_dataframe_with_metadata(
    ticker.yahoo_api_income_statement(frequency="quarterly"), "Yahoo API Income Statement (Quarterly)"
)

# +

print_dataframe_with_metadata(ticker.yahoo_api_cash_flow(format="raw"), "Yahoo API Cash Flow")

# +

print_dataframe_with_metadata(ticker.yahoo_api_balance_sheet(period1=datetime(2019, 1, 1)), "Yahoo API Balance Sheet")

# +

print_dataframe_with_metadata(
    ticker.yahoo_api_financials(period1=datetime(2022, 1, 1), period2=datetime.today()), "Yahoo API Financials"
)

# +

# ticker.plot_yahoo_api_financials(group_by="field")

# +

# ticker.plot_yahoo_api_income_statement(group_by="timeframe")

# +

# ticker.plot_yahoo_api_cash_flow(frequency="quarterly")

# +
# ticker.plot_yahoo_api_balance_sheet(frequency="quarterly")
# -

# # Sankey Chart

# +
# ticker.plot_sankey_chart()
# -

# # Summary

# +

# Summary including general financial information
print_dataframe_with_metadata(ticker.yahoo_web_summary, "Yahoo Web Summary")

# +

# Financial data as it is seen in the yahoo finance website
print_dataframe_with_metadata(ticker.yahoo_web_income_stmt, "Yahoo Web Income Statement")

# +

print_dataframe_with_metadata(ticker.yahoo_web_balance_sheet, "Yahoo Web Balance Sheet")

# +

print_dataframe_with_metadata(ticker.yahoo_web_cashflow, "Yahoo Web Cashflow")

# +

# Analysts and estimates
print_dataframe_with_metadata(ticker.yahoo_web_valuation_measures, "Yahoo Web Valuation Measures")

# +

# Data about options
print_dataframe_with_metadata(ticker.yahoo_web_calls, "Yahoo Web Calls")

# +

print_dataframe_with_metadata(ticker.yahoo_web_puts, "Yahoo Web Puts")

# +

console.print("\n[bold blue]Yahoo Web Description:[/bold blue]")
console.print(ticker.yahoo_web_description)
# -

console.print("\n[bold blue]Yahoo Web Corporate Governance:[/bold blue]")
console.print(ticker.yahoo_web_corporate_governance)

# +

print_dataframe_with_metadata(ticker.yahoo_web_major_holders, "Yahoo Web Major Holders")

# +

print_dataframe_with_metadata(ticker.yahoo_web_top_institutional_holders, "Yahoo Web Top Institutional Holders")

# +

print_dataframe_with_metadata(ticker.yahoo_web_top_mutual_fund_holders, "Yahoo Web Top Mutual Fund Holders")

# +

print_dataframe_with_metadata(ticker.yahoo_web_valuation_measures, "Yahoo Web Valuation Measures")

# +

print_dataframe_with_metadata(ticker.yahoo_web_trading_information, "Yahoo Web Trading Information")
# -

# # Digrin Data

# Complete historical data of the stock in certain categories
print_dataframe_with_metadata(ticker.digrin_dividend, "Digrin Dividend Data")

# +

print_dataframe_with_metadata(ticker.digrin_payout_ratio, "Digrin Payout Ratio")

# +

print_dataframe_with_metadata(ticker.digrin_stock_splits, "Digrin Stock Splits")

# +

print_dataframe_with_metadata(ticker.digrin_price, "Digrin Price")
# -

print_dataframe_with_metadata(ticker.digrin_assets_vs_liabilities, "Digrin Assets vs Liabilities")

print_dataframe_with_metadata(ticker.digrin_free_cash_flow, "Digrin Free Cash Flow")

# +

print_dataframe_with_metadata(ticker.digrin_net_income, "Digrin Net Income")

# +

print_dataframe_with_metadata(ticker.digrin_cash_and_debt, "Digrin Cash and Debt")

# +

print_dataframe_with_metadata(ticker.digrin_shares_outstanding, "Digrin Shares Outstanding")

# +

print_dataframe_with_metadata(ticker.digrin_expenses, "Digrin Expenses")

# +

print_dataframe_with_metadata(ticker.digrin_cost_of_revenue, "Digrin Cost of Revenue")

# +

print_dataframe_with_metadata(ticker.digrin_upcoming_estimated_earnings, "Digrin Upcoming Estimated Earnings")

# +

# Dividend data
print_dataframe_with_metadata(ticker.digrin_dividend, "Digrin Dividend Data")

# +

print_dataframe_with_metadata(ticker.digrin_dgr3, "Digrin DGR3")

# +

print_dataframe_with_metadata(ticker.digrin_dgr5, "Digrin DGR5")

# +

print_dataframe_with_metadata(ticker.digrin_dgr10, "Digrin DGR10")
# -

# # Plotting

# +
# ticker.plot_digrin_shares_outstanding()
# ticker.plot_digrin_price()
# ticker.plot_digrin_dividend()
# ticker.plot_digrin_assets_vs_liabilities()
# ticker.plot_digrin_free_cash_flow()
# ticker.plot_digrin_cash_and_debt()
# ticker.plot_digrin_net_income()
# ticker.plot_digrin_expenses()
# ticker.plot_digrin_cost_of_revenue()


# ticker.plot_digrin_price()
# -

# # Macrotrends Data

# +

# Financial data
print_dataframe_with_metadata(ticker.macrotrends_income_statement, "Macrotrends Income Statement")
# -

print_dataframe_with_metadata(ticker.macrotrends_balance_sheet, "Macrotrends Balance Sheet")

print_dataframe_with_metadata(ticker.macrotrends_cash_flow, "Macrotrends Cash Flow")

# +

# Margins
print_dataframe_with_metadata(ticker.macrotrends_gross_margin, "Macrotrends Gross Margin")
# -

print_dataframe_with_metadata(ticker.macrotrends_operating_margin, "Macrotrends Operating Margin")

print_dataframe_with_metadata(ticker.macrotrends_ebitda_margin, "Macrotrends EBITDA Margin")

print_dataframe_with_metadata(ticker.macrotrends_pre_tax_margin, "Macrotrends Pre-Tax Margin")

print_dataframe_with_metadata(ticker.macrotrends_net_margin, "Macrotrends Net Margin")

# +

# ticker.plot_macrotrends_income_statement()
# ticker.plot_macrotrends_balance_sheet()
# ticker.plot_macrotrends_cash_flow()


# # Dashboards

# from stockdex.ticker import Ticker
# from stockdex.lib import plot_multiple_categories

# # choose the stock
# ticker = Ticker(ticker="MSFT")

# # Here you will choose arbitrary figures to plot. In this example we will plot data extracted from digrin website
# # IMPORTANT: make sure to set show_plot=False in each function to return the plotly figure object instead of showing the plot. Not setting this parameter will show the plots in separate tabs.
# figures = [
#     ticker.plot_digrin_shares_outstanding(show_plot=False),
#     ticker.plot_digrin_price(show_plot=False),
#     ticker.plot_digrin_dividend(show_plot=False),
#     ticker.plot_digrin_assets_vs_liabilities(show_plot=False),
#     ticker.plot_digrin_free_cash_flow(show_plot=False),
#     ticker.plot_digrin_net_income(show_plot=False),
#     ticker.plot_digrin_cash_and_debt(show_plot=False),
#     ticker.plot_digrin_expenses(show_plot=False),
#     ticker.plot_digrin_cost_of_revenue(show_plot=False),
# ]

# # main functions that will create the dash app
# plot_multiple_categories(ticker=ticker.ticker, figures=figures)
