import edgar
from fastmcp import Context
from dotenv import load_dotenv
from rich.console import Console
from edgar.xbrl import XBRLS
from fundamentals.utility.parsing import reformat_markdown_financial_table

console = Console()

load_dotenv()

async def get_statements_impl(ticker: str, form: str, date: str, statement_type: str, ctx: Context):
    """
    Retrieve a financial statement from SEC EDGAR filings.
    Args:
        ticker (str): Company stock ticker symbol.
        form (str): SEC filing form type.
        date (str): Date to retrieve filings for.
        statement_type (str): Statement to retrieve.
        ctx (Context): Context object for error reporting and progress updates.
    Returns:
        Dict[str, Any]: Dictionary containing either an instance of StitchedStatement or an error message.
    """
    console.log(f"[bold blue]Entering get_statements_impl[/bold blue] with ticker={ticker}, form={form}, date={date}, statement_type={statement_type}")
    available_forms = set([
        "10-Q",
        "10-K",
        "8-K"
    ])

    if form not in available_forms:
        console.log(f"[yellow]Form {form} is not available[/yellow]")
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

    # Disallow certain statements as per requirements
    disallowed_statements = set([
        "Disclosures", # Limited useful information
        "CoverPage", # Limited useful information
        "BalanceSheetParenthetical", # Limited useful information
        "AccountingPolicies", # Limited useful information
        "SegmentDisclosure"  # This statement is difficult to parse
    ])
    if statement_type in disallowed_statements:
        console.log(f"[yellow]Statement {statement_type} is not allowed[/yellow]")
        await ctx.error(f"Statement {statement_type} is not allowed.")
        return {"error": f"Statement {statement_type} is not allowed."}

    if statement_type not in available_statements:
        console.log(f"[yellow]Statement {statement_type} is not available[/yellow]")
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
            msg = f"No statements found for {statement_type} (form={form}, ticker={ticker}, date={date})"
            console.log(f"[yellow]{msg}[/yellow]")
            await ctx.error(msg)
            return {"error": msg}
        
        console.log(f"[green]Returning statement for {statement_type}")
        return {"stitched_statement": stitched_statement}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_msg = f"Error in get_statements_impl for ticker={ticker}, form={form}, date={date}, statement={statement_type}: {str(e)}"
        console.log(f"[red]{error_msg}\nTraceback:\n{tb}[/red]")
        await ctx.error(error_msg)
        return {"error": error_msg}

async def summarize_financial_report_impl(ticker: str, form: str, date: str, statement_type: str, ctx: Context):
    """
    Generate a financial reports summary using the output of get_statements_impl and an LLM prompt.
    """
    import traceback  # Ensure traceback is available for all exception blocks
    console.log(f"[bold blue]Entering summarize_financial_report_impl[/bold blue] with ticker={ticker}, form={form}, date={date}, statement_type={statement_type}")
    try:
        # Get the statement data
        statement_result = await get_statements_impl(ticker, form, date, statement_type, ctx)
        if not statement_result or "error" in statement_result:
            error_msg = statement_result.get("error", "No statement data available to summarize.")
            console.log(f"[yellow]Error in get_statements_impl: {error_msg}[/yellow]")
            await ctx.error(error_msg)
            return {"error": error_msg}
        
        # Convert the stitched statement a simpler markdown format
        try:
            stitched_statement = statement_result["stitched_statement"]
            markdown_text = stitched_statement.render().to_markdown()
            markdown_text = reformat_markdown_financial_table(markdown_text)
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Failed to render or reformat statement for LLM prompt: {str(e)}"
            console.log(f"[red]{error_msg}\nTraceback:\n{tb}[/red]")
            await ctx.error(error_msg)
            return {"error": error_msg}

        # Prepare the prompt for the LLM
        prompt = (
            f"You are a financial analyst. Given the following {statement_type} data from a {form} filing for {ticker} (date: {date}), "
            "write a concise, clear financial report summary suitable for an investor. "
            "Highlight key figures, trends, and any notable changes.\n\n"
            f"Statement Data in Markdown Format:\n{markdown_text}\n\nSummary:"
        )

        # Use the LLM to generate the summary
        console.log("[green]Prompt prepared for LLM. Sending to ctx.prompt...")
        try:
            summary = await ctx.prompt(prompt)
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"LLM prompt failed for ticker={ticker}, form={form}, date={date}, statement={statement_type}: {str(e)}"
            console.log(f"[red]{error_msg}\nTraceback:\n{tb}[/red]")
            await ctx.error(error_msg)
            return {"error": error_msg}
        console.log("[green]LLM summary received.")
        return {"summary": summary}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_msg = f"Unexpected error in summarize_financial_report_impl for ticker={ticker}, form={form}, date={date}, statement={statement_type}: {str(e)}"
        console.log(f"[red]{error_msg}\nTraceback:\n{tb}[/red]")
        await ctx.error(error_msg)
        return {"error": error_msg}
