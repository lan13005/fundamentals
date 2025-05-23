from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field
from rich.console import Console

from fundamentals.tools import get_statements_impl

console = Console()
fun_mcp = FastMCP()


@fun_mcp.tool()
async def get_statements(
    ticker: Annotated[str, Field(description="Company stock ticker symbol")],
    form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
    date: Annotated[str, Field(description="Date to retrieve filings for")],
    statement: Annotated[str, Field(description="Statement to retrieve")],
    ctx: Annotated[Context, Field(description="Context object")],
) -> dict[str, Any]:
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
            - "latest" uses the most recent ~400 days of filings (current date minus 400 days to present)
        statement: Type of financial statement to retrieve, one of:
            - "BalanceSheet"
            - "CashFlowStatement"
            - "ComprehensiveIncome"
            - "IncomeStatement"
            - "StatementOfEquity"
        ctx: Context object for error reporting and progress updates

    Note:
        Any date range starting before 2009-01-01 will be automatically adjusted to start at 2009-01-01.
        The special value 'latest' is supported for the date argument.
        This adjustment is logged for debugging.

    Returns:
        Dict[str, Any]: Dictionary containing either:
            - On success: {"stitched_statement": StitchedStatement} where StitchedStatement is an object of type
            edgar.xbrl.statements.StitchedStatement
            - On error: {"error": str} with error message

    Raises:
        ValueError: If return value is not in the expected format
    """
    result = await get_statements_impl(ticker, form, date, statement, ctx)
    if date != result.get("date", date):
        console.log(
            f"[server.py] Date range adjusted from {date} to {result.get('date', date)} to avoid pre-2009 data "
            "when XBRLS became mandated"
        )
    if not (isinstance(result, dict) and ("stitched_statement" in result or "error" in result)):
        raise ValueError("get_statements must return a dict with a 'stitched_statement' or 'error' key.")
    return result


# DO NOT DELETE THIS COMMENTED OUT TOOL
#    mcp/server/session.py makes an MCP request with method="sampling/createMessage" which Cursor does not recognize
# @fun_mcp.tool()
# async def summarize_financial_report(
#     ticker: Annotated[str, Field(description="Company stock ticker symbol")],
#     form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
#     date: Annotated[str, Field(description="Date to retrieve filings for")],
#     statement: Annotated[str, Field(description="Statement to retrieve")],
#     ctx: Annotated[Context, Field(description="Context object")]
# ) -> Dict[str, Any]:
#     """
#     Generate a financial report summary using an LLM prompt based on the output of get_statements.

#     Note:
#         Any date range starting before 2009-01-01 will be automatically adjusted to start at 2009-01-01.
#         The special value 'latest' is supported for the date argument, which uses the most recent ~400 days of filings
#         This adjustment is logged for debugging.

#     Returns:
#         Dict[str, Any]:
#             On success: {"summary": ...}
#             On error: {"error": ...}

#     Downstream agents MUST use the 'summary' key for the result, or check for 'error'.
#     """
#     result = await summarize_financial_report_impl(ticker, form, date, statement, ctx)
#     if date != result.get('date', date):
#         console.log(f"[server.py] Date range adjusted from {date} to {result.get('date', date)} to avoid pre-2009 data
#             "when XBRLS became mandated"
#         )
#     if not (isinstance(result, dict) and ("summary" in result or "error" in result)):
#         raise ValueError("summarize_financial_report must return a dict with a 'summary' or 'error' key.")
#     return result


def main():
    """Run the fun_mcp server using fastmcp."""
    fun_mcp.run()


if __name__ == "__main__":
    main()
