from typing import Annotated, Dict, Any
from fastmcp import FastMCP, Context
from pydantic import Field
from server_utils.tools import get_statement_impl, summarize_financial_report_impl, print_company_info_impl

# All function definitions should be imported. Keep this file clean
# by only making calls to imported functions for use in the MCP server
# as a tool.
# Example tool creation using FastMCP:
# @mcp.tool()
# def example():
#     """
#     This is an example tool. 
#     """
#     pass

mcp = FastMCP(title="notegen MCP Server")

@mcp.tool()
async def get_statement(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    date: Annotated[str, Field(description="Date to retrieve filings for")],
    statement: Annotated[str, Field(description="Statement to retrieve")],
    ctx: Annotated[Context, Field(description="Context object")]
) -> Dict[str, Any]:
    """
    date: 
        - "2024-01-01:" denotes all filings from 2024-01-01 to present
        - ":2024-01-01" denotes all filings up to and including 2024-01-01
        - "2024-01-01:2024-01-02" denotes all filings between 2024-01-01 and 2024-01-02
    form: 
        - "10-Q", 
        - "10-K", 
        - "8-K"
    statement: 
        - "AccountingPolicies", 
        - "BalanceSheet", 
        - "BalanceSheetParenthetical", 
        - "CashFlowStatement", 
        - "ComprehensiveIncome", 
        - "CoverPage", 
        - "Disclosures", 
        - "IncomeStatement", 
        - "SegmentDisclosure", 
        - "StatementOfEquity"
    """
    return await get_statement_impl(ticker, form, date, statement, ctx)


@mcp.tool()
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

@mcp.tool()
async def summarize_financial_report(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    date: Annotated[str, Field(description="Date to retrieve filings for")],
    statement: Annotated[str, Field(description="Statement to retrieve")],
    ctx: Annotated[Context, Field(description="Context object")]
) -> Dict[str, Any]:
    """
    Generate a financial report summary using an LLM prompt based on the output of get_statement.
    """
    return await summarize_financial_report_impl(ticker, form, date, statement, ctx)

def main():
    """Run the MCP server using fastmcp."""
    mcp.run()

if __name__ == "__main__":
    main()