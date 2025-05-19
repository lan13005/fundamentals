from typing import Annotated, Dict, Any
from pydantic import Field, BaseModel
from edgar import Company
from dotenv import load_dotenv
from datetime import datetime

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
        
        filing_info = CompanyFilingInfo(
            ticker=ticker,
            form=form,
            filing_text=filing.text(),
            filing_date="", # filing.filing_date.strftime("%Y-%m-%d"),
            accession_number="" #filing.accession_number
        )
        
        return {
            "status": 0,
            "message": "Success",
            "data": filing_info.dict()
        }
        
    except Exception as e:
        return {
            "status": 1,
            "message": f"Error getting company info: {str(e)}",
            "data": None
        } 