import pytest
from unittest.mock import Mock, patch, AsyncMock
from server_utils.tools import print_company_info_impl, get_statement_impl, summarize_financial_report_impl
from fastmcp import Context
from rich.console import Console

console = Console()


############ PRINT COMPANY INFO ############

@pytest.mark.asyncio    
async def test_print_company_info_success():
    console.log("[test] test_print_company_info_success entry")
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
        console.log(f"[test] Result: {result}")
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
    console.log("[test] test_print_company_info_success exit")

@pytest.mark.asyncio
async def test_print_company_info_no_filings():
    console.log("[test] test_print_company_info_no_filings entry")
    mock_company = Mock()
    mock_company.get_filings.return_value = []
    
    with patch('server_utils.tools.Company', return_value=mock_company):
        result = await print_company_info_impl(
            ticker="AAPL",
            form="10-K",
            filing_index=0
        )
        console.log(f"[test] Result: {result}")
        assert result["status"] == 1
        assert "No filings found" in result["message"]
        assert result["data"] is None
    console.log("[test] test_print_company_info_no_filings exit")

@pytest.mark.asyncio
async def test_print_company_info_invalid_index():
    console.log("[test] test_print_company_info_invalid_index entry")
    mock_filing = Mock()
    mock_company = Mock()
    mock_company.get_filings.return_value = [mock_filing]
    
    with patch('server_utils.tools.Company', return_value=mock_company):
        result = await print_company_info_impl(
            ticker="AAPL",
            form="10-K",
            filing_index=1  # Index out of range
        )
        console.log(f"[test] Result: {result}")
        assert result["status"] == 1
        assert "out of range" in result["message"]
        assert result["data"] is None
    console.log("[test] test_print_company_info_invalid_index exit")


############ GET STATEMENT ############

@pytest.mark.asyncio
async def test_get_statement_impl_success():
    console.log("[test] test_get_statement_impl_success entry")
    mock_ctx = Mock(spec=Context)
    mock_ctx.session = {}
    
    mock_filings = Mock()
    mock_filings.filter.return_value = 'filtered_filings'
    mock_company = Mock()
    mock_company.get_filings.return_value = mock_filings
    
    mock_xbrl = Mock()
    mock_xbrl.statements = {'BalanceSheet': {'assets': 1000}}
    
    with patch('server_utils.tools.edgar.Company', return_value=mock_company), \
         patch('server_utils.tools.edgar.xbrl.XBRLS.from_filings', return_value=mock_xbrl):
        result = await get_statement_impl(
            ticker="AAPL",
            form="10-K",
            date="2024-03-15",
            statement="BalanceSheet",
            ctx=mock_ctx
        )
        console.log(f"[test] Result: {result}")
        assert result == {'assets': 1000}
        assert mock_ctx.session['AAPL_10-K_2024-03-15_BalanceSheet'] == {'assets': 1000}
    console.log("[test] test_get_statement_impl_success exit")

@pytest.mark.asyncio
async def test_get_statement_impl_invalid_form():
    console.log("[test] test_get_statement_impl_invalid_form entry")
    mock_ctx = AsyncMock(spec=Context)
    mock_ctx.session = {}
    result = await get_statement_impl(
        ticker="AAPL",
        form="20-F",
        date="2024-03-15",
        statement="BalanceSheet",
        ctx=mock_ctx
    )
    console.log(f"[test] Result: {result}")
    assert result == {}
    mock_ctx.error.assert_awaited_once()
    assert "not available" in mock_ctx.error.call_args[0][0]
    console.log("[test] test_get_statement_impl_invalid_form exit")

@pytest.mark.asyncio
async def test_get_statement_impl_invalid_statement():
    console.log("[test] test_get_statement_impl_invalid_statement entry")
    mock_ctx = AsyncMock(spec=Context)
    mock_ctx.session = {}
    result = await get_statement_impl(
        ticker="AAPL",
        form="10-K",
        date="2024-03-15",
        statement="NonExistentStatement",
        ctx=mock_ctx
    )
    console.log(f"[test] Result: {result}")
    assert result == {}
    mock_ctx.error.assert_awaited_once()
    assert "not available" in mock_ctx.error.call_args[0][0]
    console.log("[test] test_get_statement_impl_invalid_statement exit")

@pytest.mark.asyncio
async def test_get_statement_impl_cache():
    console.log("[test] test_get_statement_impl_cache entry")
    mock_ctx = Mock(spec=Context)
    mock_ctx.session = {"AAPL_10-K_2024-03-15_BalanceSheet": {"cached": True}}
    result = await get_statement_impl(
        ticker="AAPL",
        form="10-K",
        date="2024-03-15",
        statement="BalanceSheet",
        ctx=mock_ctx
    )
    console.log(f"[test] Result: {result}")
    assert result == {"cached": True}
    console.log("[test] test_get_statement_impl_cache exit")

############ SUMMARIZE FINANCIAL REPORT ############

@pytest.mark.asyncio
async def test_summarize_financial_report_impl_success():
    console.log("[test] test_summarize_financial_report_impl_success entry")
    mock_ctx = AsyncMock(spec=Context)
    mock_ctx.session = {}
    # Patch get_statement_impl to return statement data
    with patch('server_utils.tools.get_statement_impl', new=AsyncMock(return_value={'assets': 1000})):
        mock_ctx.prompt = AsyncMock(return_value="Summary text.")
        result = await summarize_financial_report_impl(
            ticker="AAPL",
            form="10-K",
            date="2024-03-15",
            statement="BalanceSheet",
            ctx=mock_ctx
        )
        console.log(f"[test] Result: {result}")
        assert result["status"] == 0
        assert result["message"] == "Success"
        assert result["summary"] == "Summary text."
        mock_ctx.prompt.assert_awaited_once()
    console.log("[test] test_summarize_financial_report_impl_success exit")

@pytest.mark.asyncio
async def test_summarize_financial_report_impl_no_statement():
    console.log("[test] test_summarize_financial_report_impl_no_statement entry")
    mock_ctx = AsyncMock(spec=Context)
    mock_ctx.session = {}
    with patch('server_utils.tools.get_statement_impl', new=AsyncMock(return_value={})), \
         patch.object(mock_ctx, 'error', new=AsyncMock()) as mock_error:
        result = await summarize_financial_report_impl(
            ticker="AAPL",
            form="10-K",
            date="2024-03-15",
            statement="BalanceSheet",
            ctx=mock_ctx
        )
        console.log(f"[test] Result: {result}")
        assert result["status"] == 1
        assert "No statement data" in result["message"]
        assert result["summary"] is None
        mock_error.assert_awaited_once()
    console.log("[test] test_summarize_financial_report_impl_no_statement exit")

@pytest.mark.asyncio
async def test_print_company_info_real_data():
    console.log("[integration] test_print_company_info_real_data entry")
    result = await print_company_info_impl(
        ticker="AAPL",
        form="10-K",
        filing_index=0
    )
    console.log(f"[integration] Result: {result}")
    assert result["status"] == 0
    assert result["message"] == "Success"
    assert isinstance(result["data"], dict)
    assert result["data"]["ticker"] == "AAPL"
    assert result["data"]["form"] == "10-K"
    assert isinstance(result["data"]["filing_text"], str)
    assert len(result["data"]["filing_text"]) > 1000  # Should be a long text
    assert isinstance(result["data"]["filing_date"], str)
    assert isinstance(result["data"]["accession_number"], str)
    console.log("[integration] test_print_company_info_real_data exit")

@pytest.mark.asyncio
async def test_get_statement_impl_real_data():
    console.log("[integration] test_get_statement_impl_real_data entry")
    class DummyCtx:
        def __init__(self):
            self.session = {}
        async def error(self, msg):
            console.log(f"[integration] DummyCtx error: {msg}")
    ctx = DummyCtx()
    # Use a known recent 10-K date for Apple (update if needed)
    result = await get_statement_impl(
        ticker="AAPL",
        form="10-K",
        date="2023-10-27",
        statement="BalanceSheet",
        ctx=ctx
    )
    console.log(f"[integration] Result: {result}")
    assert isinstance(result, dict)
    assert len(result) > 0
    assert any(key in result for key in ["Assets", "assets", "Liabilities", "liabilities"])
    console.log("[integration] test_get_statement_impl_real_data exit")

@pytest.mark.asyncio
async def test_summarize_financial_report_impl_real_data():
    console.log("[integration] test_summarize_financial_report_impl_real_data entry")
    class DummyCtx:
        def __init__(self):
            self.session = {}
        async def error(self, msg):
            console.log(f"[integration] DummyCtx error: {msg}")
        async def prompt(self, prompt):
            # Simulate LLM summary
            return "This is a summary of the Balance Sheet."
    ctx = DummyCtx()
    result = await summarize_financial_report_impl(
        ticker="AAPL",
        form="10-K",
        date="2023-10-27",
        statement="BalanceSheet",
        ctx=ctx
    )
    console.log(f"[integration] Result: {result}")
    assert result["status"] == 0
    assert result["message"] == "Success"
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10
    console.log("[integration] test_summarize_financial_report_impl_real_data exit")