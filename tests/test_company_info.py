import pytest
from unittest.mock import Mock, patch
from server_utils.tools import print_company_info_impl

@pytest.mark.asyncio
async def test_print_company_info_success():
    # Mock the Company and Filing objects
    mock_filing = Mock()
    mock_filing.text.return_value = "Test filing text"
    mock_filing.filing_date = "2024-03-15"
    mock_filing.accession_number = "0001193125-24-000001"
    
    mock_company = Mock()
    mock_company.get_filings.return_value = [mock_filing]
    
    with patch('server_utils.tools.Company', return_value=mock_company):
        result = await print_company_info_impl(
            ticker="AAPL",
            form="10-K",
            filing_index=0
        )
        
        assert result["status"] == 0
        assert result["message"] == "Success"
        assert isinstance(result["data"], dict)
        assert result["data"]["ticker"] == "AAPL"
        assert result["data"]["form"] == "10-K"
        assert result["data"]["filing_text"] == "Test filing text"
        assert result["data"]["filing_date"] == "2024-03-15"
        assert result["data"]["accession_number"] == "0001193125-24-000001"
        
        # Verify the mock was called correctly
        mock_company.get_filings.assert_called_once_with(form="10-K")
        mock_filing.text.assert_called_once()

@pytest.mark.asyncio
async def test_print_company_info_no_filings():
    mock_company = Mock()
    mock_company.get_filings.return_value = []
    
    with patch('server_utils.tools.Company', return_value=mock_company):
        result = await print_company_info_impl(
            ticker="AAPL",
            form="10-K",
            filing_index=0
        )
        
        assert result["status"] == 1
        assert "No filings found" in result["message"]
        assert result["data"] is None

@pytest.mark.asyncio
async def test_print_company_info_invalid_index():
    mock_filing = Mock()
    mock_company = Mock()
    mock_company.get_filings.return_value = [mock_filing]
    
    with patch('server_utils.tools.Company', return_value=mock_company):
        result = await print_company_info_impl(
            ticker="AAPL",
            form="10-K",
            filing_index=1  # Index out of range
        )
        
        assert result["status"] == 1
        assert "out of range" in result["message"]
        assert result["data"] is None