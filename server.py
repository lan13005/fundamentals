from typing import Annotated, Dict, Any
from fastmcp import FastMCP
from pydantic import Field
from server_utils.company_info import print_company_info_impl

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

def main():
    """Run the MCP server using fastmcp."""
    mcp.run()

if __name__ == "__main__":
    main()