import edgar
from fastmcp import Context
from typing import Annotated, Dict, Any
from pydantic import Field, BaseModel
from edgar import Company
from dotenv import load_dotenv
from rich.console import Console

console = Console()

load_dotenv()
class CompanyFilingInfo(BaseModel):
    """Model for company filing information."""
    ticker: str
    form: str
    filing_text: str
    filing_date: str
    accession_number: str

async def print_company_info_impl(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    filing_index: Annotated[int, Field(description="Index of the filing to retrieve (0 for most recent)")],
) -> Dict[str, Any]:
    try:
        # Get company information
        company = Company(ticker)
        
        # Get latest filing and return text
        filings = company.get_filings(form=form)
        if not filings:
            return {
                "status": 1,
                "message": f"No filings found for {ticker}",
                "data": None
            }
            
        if filing_index >= len(filings):
            return {
                "status": 1,
                "message": f"Filing index {filing_index} out of range (0-{len(filings)-1})",
                "data": None
            }
            
        filing = filings[filing_index]
        
        # Handle filing_date being either a string or a datetime object
        filing_date = filing.filing_date
        if hasattr(filing_date, 'strftime'):
            filing_date = filing_date.strftime("%Y-%m-%d")
        filing_info = CompanyFilingInfo(
            ticker=ticker,
            form=form,
            filing_text=filing.text(),
            filing_date=filing_date,
            accession_number=filing.accession_number
        )
        
        return {
            "status": 0,
            "message": "Success",
            "data": filing_info.dict()
        }
        
    except Exception as e:
        print(f"Error getting company info: {str(e)}")
        return {
            "status": 1,
            "message": f"Error getting company info: {str(e)}",
            "data": None
        } 




async def get_statement_impl(ticker: str, form: str, date: str, statement: str, ctx: Context):

    available_forms = set([
        "10-Q",
        "10-K",
        "8-K"
    ])

    if form not in available_forms:
        await ctx.error(f"Form {form} is not available: choose from {available_forms}")
        return {}
    
    available_statements = set([
        "AccountingPolicies",       
        "BalanceSheet",             
        "BalanceSheetParenthetical",
        "CashFlowStatement",        
        "ComprehensiveIncome",
        "CoverPage",                
        "Disclosures",              
        "IncomeStatement",          
        "SegmentDisclosure",        
        "StatementOfEquity"
    ])

    if statement not in available_statements:
        await ctx.error(f"Statement {statement} is not available: choose from {available_statements}")
        return {}

    cache_key = f"{ticker}_{form}_{date}_{statement}"

    if cache_key in ctx.session:
        return ctx.session[cache_key]

    company = edgar.Company(ticker)
    filings = company.get_filings()
    xbrl = edgar.XBRLS.from_filings(filings.filter(form=form, date=date))
    statements = xbrl.statements
    statement = statements[statement]
    
    ctx.session[cache_key] = statement
    return statement

async def summarize_financial_report_impl(ticker: str, form: str, date: str, statement: str, ctx: Context):
    """
    Generate a financial reports summary using the output of get_statement_impl and an LLM prompt.
    """
    # Get the statement data
    statement_data = await get_statement_impl(ticker, form, date, statement, ctx)
    if not statement_data:
        await ctx.error("No statement data available to summarize.")
        return {"status": 1, "message": "No statement data available to summarize.", "summary": None}

    # Prepare the prompt for the LLM
    prompt = (
        f"You are a financial analyst. Given the following {statement} data from a {form} filing for {ticker} (date: {date}), "
        "write a concise, clear financial report summary suitable for an investor. "
        "Highlight key figures, trends, and any notable changes.\n\n"
        f"Statement Data:\n{statement_data}\n\nSummary:"
    )

    # Use the LLM to generate the summary
    summary = await ctx.prompt(prompt)
    return {"status": 0, "message": "Success", "summary": summary}
