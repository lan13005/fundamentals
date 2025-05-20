import pytest
from fundamentals.tools import print_company_info_impl, get_statement_impl, summarize_financial_report_impl
from fastmcp import Context, FastMCP, Client
from rich.console import Console
from typing import Annotated, Dict, Any
from pydantic import Field
import asyncio
from unittest.mock import patch, AsyncMock
from edgar.xbrl.statements import StitchedStatement

console = Console()

############ REAL DATA ############

# @pytest.mark.asyncio
# async def test_print_company_info_real_data():
#     console.log("[integration] test_print_company_info_real_data entry")
#     result = await print_company_info_impl(
#         ticker="AAPL",
#         form="10-K",
#         filing_index=0
#     )
#     console.log(f"[integration] Result: {result}")
#     
#     assert result["status"] == 0
#     assert result["message"] == "Success"
#     assert isinstance(result["data"], dict)
#     assert result["data"]["ticker"] == "AAPL"
#     assert result["data"]["form"] == "10-K"
#     assert isinstance(result["data"]["filing_text"], str)
#     assert len(result["data"]["filing_text"]) > 1000  # Should be a long text
#     assert isinstance(result["data"]["filing_date"], str)
#     assert isinstance(result["data"]["accession_number"], str)
# console.log("[integration] test_print_company_info_real_data exit")

@patch('fastmcp.Context')
@pytest.mark.asyncio
async def test_get_statement_impl_real_data(MockContext):
    console.log("[integration] test_get_statement_impl_real_data entry")
    ctx = MockContext()
    ctx.debug = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.prompt = AsyncMock(return_value={})
    result = await get_statement_impl(
        ticker="AAPL",
        form="10-K",
        date="2023-10-27:",
        statement_type="BalanceSheet",
        ctx=ctx
    )
    console.log(f"[integration] Result: {result}")
    assert isinstance(result, dict)
    assert ("stitched_statement" in result or "error" in result)
    if "stitched_statement" in result:
        stitched_statement = result["stitched_statement"]
        assert isinstance(stitched_statement, StitchedStatement)
    else:
        assert isinstance(result["error"], str)
    console.log("[integration] test_get_statement_impl_real_data exit")

@patch('fastmcp.Context')
@pytest.mark.asyncio
async def test_summarize_financial_report_impl_real_data(MockContext):
    console.log("[integration] test_summarize_financial_report_impl_real_data entry")
    ctx = MockContext()
    ctx.debug = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.prompt = AsyncMock(return_value="This is a mock summary.")
    result = await summarize_financial_report_impl(
        ticker="AAPL",
        form="10-K",
        date="2023-10-27:",
        statement_type="BalanceSheet",
        ctx=ctx
    )
    console.log(f"[integration] Result: {result}")
    assert isinstance(result, dict)
    assert ("summary" in result or "error" in result)
    if "summary" in result:
        summary = result["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 10
    else:
        assert isinstance(result["error"], str)
    console.log("[integration] test_summarize_financial_report_impl_real_data exit")


##### DO NOT MODIFY BELOW THIS LINE, FOR REFERENCE ONLY #####

# @pytest.fixture
# def mcp_server():
    
#     fun_mcp = FastMCP()
    
#     @fun_mcp.tool()
#     async def print_company_info(
#         ticker: Annotated[str, Field(description="Company stock ticker symbol")],
#         form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
#         filing_index: Annotated[int, Field(description="Index of the filing to retrieve (0 for most recent)")],
#     ) -> Dict[str, Any]:
#         """Get company filing text from SEC EDGAR database.
        
#         Args:
#             ticker: Company stock ticker symbol (e.g., 'AAPL' for Apple)
#             form: SEC filing form type (e.g., '10-K', '10-Q'), see [secforms.md](mdc:docs/edgartools/secforms.md)
#             filing_index: Index of the filing to retrieve (0 for most recent)
            
#         Returns:
#             Dict[str, Any]: Dictionary containing filing information including:
#                 - status: int (0 for success, non-zero for error)
#                 - message: str (error message if status is non-zero)
#                 - data: CompanyFilingInfo (filing information if status is 0)
            
#         Raises:
#             ValueError: If no filings are found or if there's an error getting company info
#         """
#         return await print_company_info_impl(ticker, form, filing_index)
    
#     @fun_mcp.tool()
#     async def summarize_financial_report(
#         ticker: Annotated[str, Field(description="Company stock ticker symbol")],
#         form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
#         date: Annotated[str, Field(description="Date to retrieve filings for")],
#         statement: Annotated[str, Field(description="Statement to retrieve")],
#         ctx: Annotated[Context, Field(description="Context object")]
#     ) -> Dict[str, Any]:
#         """
#         Generate a financial report summary using an LLM prompt based on the output of get_statement.
#         """
#         return await summarize_financial_report_impl(ticker, form, date, statement, ctx)
        
#     @fun_mcp.tool()
#     async def get_statement(
#         ticker: Annotated[str, Field(description="Company stock ticker symbol")],
#         form: Annotated[str, Field(description="SEC filing form type (e.g., '10-K', '10-Q')")],
#         date: Annotated[str, Field(description="Date to retrieve filings for")],
#         statement: Annotated[str, Field(description="Statement to retrieve")],
#         ctx: Annotated[Context, Field(description="Context object")]
#     ) -> Dict[str, Any]:
#         """
#         date: 
#             - "2024-01-01:" denotes all filings from 2024-01-01 to present
#             - ":2024-01-01" denotes all filings up to and including 2024-01-01
#             - "2024-01-01:2024-01-02" denotes all filings between 2024-01-01 and 2024-01-02
#         form: 
#             - "10-Q", 
#             - "10-K", 
#             - "8-K"
#         statement: 
#             - "AccountingPolicies", 
#             - "BalanceSheet", 
#             - "BalanceSheetParenthetical", 
#             - "CashFlowStatement", 
#             - "ComprehensiveIncome", 
#             - "CoverPage", 
#             - "Disclosures", 
#             - "IncomeStatement", 
#             - "SegmentDisclosure", 
#             - "StatementOfEquity"
#         """
#         return await get_statement_impl(ticker, form, date, statement, ctx)
        
#     return fun_mcp

# async def test_tool_functionality(mcp_server):
#     # Pass the server directly to the Client constructor
#     async with Client(mcp_server) as client:
        
#         print("test_tool_functionality entry")
        
#         tools = await client.list_tools()
#         print(f"Tools: {tools}")
        
#         # result = await client.call_tool("print_company_info", {"ticker": "AAPL", "form": "10-K", "filing_index": 0})
#         # assert result[0].text != ""
        
#         result = await client.call_tool("get_statement", {"ticker": "AAPL", "form": "10-K", "date": "2023-10-27:", "statement": "BalanceSheet"})
#         print(result)