from typing import Annotated, Dict, Any
from fastmcp import FastMCP, Context
from pydantic import Field
from fundamentals.tools import get_statement_impl, summarize_financial_report_impl, print_company_info_impl

fun_mcp = FastMCP()

@fun_mcp.tool()
async def get_statement(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    date: Annotated[str, Field(description="Date to retrieve filings for")],
    statement: Annotated[str, Field(description="Statement to retrieve")],
    ctx: Annotated[Context, Field(description="Context object")]
) -> Dict[str, Any]:
    """Get a financial statement from SEC EDGAR filings.
    
    Args:
        ticker: Company stock ticker symbol (e.g., 'AAPL' for Apple)
        form: SEC filing form type, one of:
            - "10-Q" - Quarterly report
            - "10-K" - Annual report
            - "8-K" - Current report
        date: Filing date range in one of these formats:
            - "2024-01-01:" denotes all filings from 2024-01-01 to present
            - ":2024-01-01" denotes all filings up to and including 2024-01-01
            - "2024-01-01:2024-01-02" denotes all filings between 2024-01-01 and 2024-01-02
        statement: Type of financial statement to retrieve, one of:
            - "AccountingPolicies"
            - "BalanceSheet"
            - "BalanceSheetParenthetical"
            - "CashFlowStatement"
            - "ComprehensiveIncome" 
            - "CoverPage"
            - "Disclosures"
            - "IncomeStatement"
            - "SegmentDisclosure"
            - "StatementOfEquity"
        ctx: Context object for error reporting and progress updates
            
    Returns:
        Dict[str, Any]: Dictionary containing either:
            - On success: {"stitched_statement": StitchedStatement} where StitchedStatement is an object of type edgar.xbrl.statements.StitchedStatement
            - On error: {"error": str} with error message
            
    Raises:
        ValueError: If return value is not in the expected format
    """
    result = await get_statement_impl(ticker, form, date, statement, ctx)
    if not (isinstance(result, dict) and ("stitched_statement" in result or "error" in result)):
        raise ValueError("get_statement must return a dict with a 'stitched_statement' or 'error' key.")
    return result


@fun_mcp.tool()
async def print_company_info(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    filing_index: Annotated[int, Field(description="Index of the filing to retrieve (0 for most recent)")],
) -> Dict[str, Any]:
    """Get company filing text from SEC EDGAR database.
    
    Args:
        ticker: Company stock ticker symbol (e.g., 'AAPL' for Apple)
        form: SEC filing form type (e.g., '10-K', '10-Q'), see [secforms.md](mdc:docs/edgartools/secforms.md)
        filing_index: Index of the filing to retrieve (0 for most recent)
        
    Returns:
        Dict[str, Any]: Dictionary containing filing information including:
            - status: int (0 for success, non-zero for error)
            - message: str (error message if status is non-zero)
            - data: CompanyFilingInfo (filing information if status is 0)
        
    Raises:
        ValueError: If no filings are found or if there's an error getting company info
    """
    return await print_company_info_impl(ticker, form, filing_index)

@fun_mcp.tool()
async def summarize_financial_report(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    date: Annotated[str, Field(description="Date to retrieve filings for")],
    statement: Annotated[str, Field(description="Statement to retrieve")],
    ctx: Annotated[Context, Field(description="Context object")]
) -> Dict[str, Any]:
    """
    Generate a financial report summary using an LLM prompt based on the output of get_statement.

    Returns:
        Dict[str, Any]:
            On success: {"summary": ...}
            On error: {"error": ...}
    
    Downstream agents MUST use the 'summary' key for the result, or check for 'error'.
    """
    result = await summarize_financial_report_impl(ticker, form, date, statement, ctx)
    if not (isinstance(result, dict) and ("summary" in result or "error" in result)):
        raise ValueError("summarize_financial_report must return a dict with a 'summary' or 'error' key.")
    return result

def main():
    """Run the fun_mcp server using fastfun_mcp."""
    fun_mcp.run()

if __name__ == "__main__":
    main()