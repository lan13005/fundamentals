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

import edgar as et
import datetime
from rich.console import Console
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
console = Console()

et.set_identity("lng1492@gmail.com")

# # Getting filings

# +
ticker = "RKLB"
company = et.Company(ticker)
filings = company.get_filings()
filing_df = filings.to_pandas()

summary = {
    "ticker": company.tickers,
    "name": company.name,
    "industry": company.industry,
    "mailing_address": str(company.mailing_address()),
    "cik": company.cik,
    "sic": company.sic,
    "exchanges": company.get_exchanges(),
    "latest_filing_date": filing_df.filing_date.max().strftime("%Y-%m-%d"),
    "earliest_filing_date": filing_df.filing_date.min().strftime("%Y-%m-%d"),
    "form_counts": filing_df.form.value_counts().to_dict() # Dict[str, int]
}

# edgartools/edgar/ownership/ownershipforms.py
#  Form3,4,5 where Ownership class can be converted to Form3,4,5 class using .obj()
#  From here I think we can access the information between date ranges also by filtering and looping over Form filings
#       converting using .obj() then getting activities, holdings, etc.

# -

filtered_filings = filings.filter(form="4", date="2024-01-01:")
merged_df = [filing.obj().to_dataframe() for filing in filtered_filings]
merged_df = pd.concat(merged_df)

# Convert Date column to datetime if it's not already
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Get stock price data
stock_data = yf.download(ticker, start=merged_df.Date.min(), end=merged_df.Date.max())

# Create interactive plot
fig = make_subplots(rows=1, cols=1)

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Stock Price'
    ),
    row=1, col=1
)

# Add insider trading indicators
for _, row in merged_df.iterrows():
    # Find the nearest trading day
    trade_date = pd.to_datetime(row['Date'])
    nearest_date = stock_data.index[stock_data.index.get_indexer([trade_date], method='nearest')[0]]
    
    # Determine if it's a purchase or sale
    is_purchase = row['Shares'] > 0
    
    # Create arrow annotation
    fig.add_annotation(
        x=nearest_date,
        y=stock_data.loc[nearest_date, 'High'] if is_purchase else stock_data.loc[nearest_date, 'Low'],
        text='↑' if is_purchase else '↓',
        showarrow=False,
        font=dict(size=20, color='green' if is_purchase else 'red'),
        row=1, col=1
    )

# Update layout
fig.update_layout(
    title=f'{ticker} Stock Price with Insider Trading Indicators',
    yaxis_title='Price',
    xaxis_title='Date',
    template='plotly_white'
)

# Show the plot
fig.show()

######################################################################
# NOTE: Do not touch any of the code/comments below!
# You can use them as references to understand the code better though
######################################################################

# filings = company.get_filings()
# # filings.get_filings(2024, [3, 4])

# company.get_filings().filter(date="2025-03-01:")

# import datetime
# company_insider_filings = company.get_filings(form=[3,4,5])
# company_insider_filings.filter

# # +
# company_insider_filings = company.get_filings(form=[3,4,5])
# company_insider_filings[0].obj() # prints additional information (basically renders the form)
# # These two produce the same output
# print(company_insider_filings[0].obj())
# # console.print(rklb_insider_filings[0].obj().get_ownership_summary())

# print(f"rklb_insider_filings[0] type: {type(company_insider_filings[0])}")
# print(f"rklb_insider_filings[0].obj() type: {type(company_insider_filings[0].obj())}")

# # +
# rklb_filings = rklb.get_filings() # get all filings
# # rklb_filings[0].open() # opens browser to the filing
# rklb_10q = rklb.get_filings(form='10-Q') 

# # rklb_filings.data # pyarrow table of filings
# # -

# rklb_10q[0].accession_number

# rklb.get_filings(form='3')[3].view()

# # # Facts Metadata

# # +
# rklb_facts = rklb.get_facts()
# rklb_facts_df = rklb_facts.to_pandas() # converts pyarrow table to pandas df

# ## Gets metadata about the facts including fact description
# unique_desc = rklb_facts.fact_meta['description'].value_counts()
# # for desc in unique_desc[:3]:
# #     print(desc)
# # -

# unique_desc

# # # Attachments in a filing

# # +
# # Attachments
# ## Get the attachments for the first 10q filing
# rklb_10q[0].attachments

# ## Prints the actual text of the attachment 10q
# # text = rklb_10q[0].attachments[1].text()

# ## Atleast for the 10q attachments, I cant find anything useful
# # rklb_10q[0].attachments[84].download('.') # download the attachment
# # -

# # # Financials
# #

# # Some functionality is deprecated, should use XBRLs instead of i.e. MultiFinancials(filings)

# rklb_financials = rklb.get_financials()

# # +
# # Income Statement
# rklb_balance_sheet = rklb_financials.balance_sheet()
# rklb_balance_sheet = rklb_balance_sheet.to_dataframe()

# rklb_cash_flow = rklb_financials.cashflow_statement()
# rklb_cash_flow = rklb_cash_flow.to_dataframe()

# rklb_income_statement = rklb_financials.income_statement()
# rklb_income_statement = rklb_income_statement.to_dataframe()

# # +
# from edgar.xbrl.xbrl import XBRL
# from edgar.xbrl import XBRLS

# filing = rklb.latest("10-K")
# xbrl = XBRL.from_filing(filing)
# rklb_income_statement = xbrl.statements.income_statement()

# # +
# filings = rklb.latest("10-K", 5)
# xbrls = XBRLS.from_filings(filings)
# stitched_statements = xbrls.statements

# stitched_statements

# # +
# balance_sheet = stitched_statements.balance_sheet()
# income_statement = stitched_statements.income_statement()
# cash_flow = stitched_statements.cashflow_statement()
# statement_of_equity = stitched_statements.statement_of_equity()

# # You can also access by type
# comprehensive_income = stitched_statements["ComprehensiveIncome"]

# # income_statement.to_dataframe() # converts to pandas df
# # -

# income_trend = stitched_statements.income_statement(max_periods=3)

# from rich.console import Console
# console = Console()
# console.print(stitched_statements.balance_sheet())

# # # Insider Trading

# # - Form 3: Filed by insiders to report their initial ownership of company stock - typically filed when an insider joins a company or becomes an officer or director.
# # - Form 4: Filed to report any changes in ownership of company stock - typically filed when an insider buys or sells company stock.
# # - Form 5: Includes any transactions that were not reported on Form 4 - typically filed at the end of the fiscal year.
# #
# # NOTE: A "filing" is returned as an `edgar.entity.filings.EntityFiling` object. We have to use `.obj()` to convert it get something useful like form4 `edgar.ownership.ownershipforms.Form4`

# rklb_insider_filings = rklb.get_filings(form=[3,4,5])

# rklb_insider_filings[0].obj() # prints additional information (basically renders the form)

# # +
# # These two produce the same output
# console.print(rklb_insider_filings[0].obj())
# # console.print(rklb_insider_filings[0].obj().get_ownership_summary())

# console.print(f"rklb_insider_filings[0] type: {type(rklb_insider_filings[0])}")
# console.print(f"rklb_insider_filings[0].obj() type: {type(rklb_insider_filings[0].obj())}")
# # -

