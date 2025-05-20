import edgar
from fastmcp import Context
from typing import Annotated, Dict, Any
from pydantic import Field, BaseModel
from edgar import Company
from dotenv import load_dotenv
from rich.console import Console
from edgar.xbrl import XBRLS

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
    console.log(f"[bold blue]Entering print_company_info_impl[/bold blue] with ticker={ticker}, form={form}, filing_index={filing_index}")
    try:
        # Get company information
        company = Company(ticker)
        console.log(f"[green]Fetched company object for {ticker}")
        
        # Get latest filing and return text
        filings = company.get_filings(form=form)
        console.log(f"[green]Fetched {len(filings)} filings for {ticker} and form {form}")
        if not filings:
            console.log(f"[yellow]No filings found for {ticker}")
            return {
                "status": 1,
                "message": f"No filings found for {ticker}",
                "data": None
            }
            
        if filing_index >= len(filings):
            console.log(f"[yellow]Filing index {filing_index} out of range for {ticker} (max {len(filings)-1})")
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
        console.log(f"[green]Returning filing info for {ticker} {form} index {filing_index}")
        return {
            "status": 0,
            "message": "Success",
            "data": filing_info.model_dump()
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        console.print(f"[red]Error getting company info: {str(e)}\nTraceback:\n{tb}[/red]")
        return {
            "status": 1,
            "message": f"Error getting company info: {str(e)}",
            "data": None
        } 

async def get_statement_impl(ticker: str, form: str, date: str, statement_type: str, ctx: Context):
    console.log(f"[bold blue]Entering get_statement_impl[/bold blue] with ticker={ticker}, form={form}, date={date}, statement_type={statement_type}")
    available_forms = set([
        "10-Q",
        "10-K",
        "8-K"
    ])

    if form not in available_forms:
        console.log(f"[yellow]Form {form} is not available")
        await ctx.error(f"Form {form} is not available: choose from {available_forms}")
        return {"error": f"Form {form} is not available: choose from {available_forms}"}
    
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

    if statement_type not in available_statements:
        console.log(f"[yellow]Statement {statement_type} is not available")
        await ctx.error(f"Statement {statement_type} is not available: choose from {available_statements}")
        return {"error": f"Statement {statement_type} is not available: choose from {available_statements}"}

    try:
        company = edgar.Company(ticker)
        console.log(f"[green]Fetched company object for {ticker}")
        filings = company.get_filings()
        console.log(f"[green]Fetched filings for {ticker}")
        filtered_filings = filings.filter(form=form, date=date)
        console.log(f"[green]Filtered filings for form={form}, date={date}")
        xbrls = XBRLS.from_filings(filtered_filings)
        statements = xbrls.statements
        stitched_statement = statements[statement_type]
        
        # Check that we actually loaded some statements across periods
        found_stmt_types = set()
        found_periods = xbrls.get_periods()
        for xbrl in stitched_statement.xbrls.xbrl_list:
            statement = xbrl.get_all_statements()
            for stmt in statement:
                if stmt['type']:
                    found_stmt_types.add(stmt['type'])
        period_count = len(found_periods)
        if period_count == 0 or len(found_stmt_types) == 0:
            console.log(f"[yellow]No statements found for {statement_type}")
            await ctx.error(f"No statements found for {statement_type}")
            return {"error": f"No statements found for {statement_type}"}
        
        console.log(f"[green]Returning statement for {statement_type}")
        return {"stitched_statement": stitched_statement}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        console.print(f"[red]Error in get_statement_impl: {str(e)}\nTraceback:\n{tb}[/red]")
        await ctx.error(f"Error in get_statement_impl: {str(e)}")
        return {"error": f"Error in get_statement_impl: {str(e)}"}

async def summarize_financial_report_impl(ticker: str, form: str, date: str, statement_type: str, ctx: Context):
    console.log(f"[bold blue]Entering summarize_financial_report_impl[/bold blue] with ticker={ticker}, form={form}, date={date}, statement_type={statement_type}")
    """
    Generate a financial reports summary using the output of get_statement_impl and an LLM prompt.
    """
    # Get the statement data
    statement_result = await get_statement_impl(ticker, form, date, statement_type, ctx)
    if not statement_result or "error" in statement_result:
        error_msg = statement_result.get("error", "No statement data available to summarize.")
        console.log(f"[yellow]{error_msg}")
        await ctx.error(error_msg)
        return {"error": error_msg}

    stitched_statement = statement_result["stitched_statement"]

    # Prepare the prompt for the LLM
    prompt = (
        f"You are a financial analyst. Given the following {statement_type} data from a {form} filing for {ticker} (date: {date}), "
        "write a concise, clear financial report summary suitable for an investor. "
        "Highlight key figures, trends, and any notable changes.\n\n"
        f"Statement Data:\n{stitched_statement}\n\nSummary:"
    )

    # Use the LLM to generate the summary
    console.log("[green]Prompt prepared for LLM. Sending to ctx.prompt...")
    summary = await ctx.prompt(prompt)
    console.log("[green]LLM summary received.")
    return {"summary": summary}
