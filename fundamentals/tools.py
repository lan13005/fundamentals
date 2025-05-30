import datetime
import os
import traceback  # Import traceback at the top for use in all exception blocks

import edgar
from dotenv import load_dotenv
from edgar.xbrl import XBRLS
from fastmcp import Context
from rich.console import Console

from fundamentals.utility.general import reformat_markdown_financial_table

console = Console(file=open(os.devnull, "w"))

load_dotenv()


def format_error_msg(context: str, e: Exception) -> str:
    """
    Formats an error message with traceback for consistent error reporting.

    Args:
        context (str): Contextual information about where the error occurred.
        e (Exception): The exception instance.

    Returns:
        str: Formatted error message with traceback.
    """
    tb = traceback.format_exc()
    return f"{context}: {e!s}\nTraceback:\n{tb}"


def adjust_date_range(date_str):
    """
    Adjusts the date range for SEC EDGAR queries.

    Supports:
        - Formats like '2024-01-01:', ':2024-12-31', '2008-01-01:2024-12-31'
        - 'latest' as a special value, using the current date minus 400 days as min_date
        - Random strings or invalid formats are treated as unspecified, defaulting to '2009-01-01:'

    If start == end, one day is added to the end date to ensure a valid range.

    Args:
        date_str (str): Date string or 'latest'.

    Returns:
        str: Adjusted date string for EDGAR queries.
    """
    DEFAULT_MIN_DATE = datetime.date(2009, 1, 1)

    def parse_date_safe(date_str):
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None

    if date_str == "latest":
        min_date = datetime.date.today() - datetime.timedelta(days=400)
        return f"{min_date.isoformat()}:"

    if ":" in date_str:
        start_str, end_str = ([*date_str.split(":"), ""])[:2]
        start = parse_date_safe(start_str.strip()) if start_str.strip() else None
        end = parse_date_safe(end_str.strip()) if end_str.strip() else None

        if not start or start < DEFAULT_MIN_DATE:
            start = DEFAULT_MIN_DATE

        if end:
            if end == start:
                end += datetime.timedelta(days=1)
            return f"{start.isoformat()}:{end.isoformat()}"
        else:
            return f"{start.isoformat()}:"
    else:
        return f"{DEFAULT_MIN_DATE.isoformat()}:"


async def get_statements_impl(ticker: str, form: str, date: str, statement_type: str, ctx: Context):
    """
    Retrieve a financial statement from SEC EDGAR filings.
    Args:
        ticker (str): Company stock ticker symbol.
        form (str): SEC filing form type.
        date (str): Date to retrieve filings for. Note: Any date range starting before 2009-01-01 will be automatically adjusted to start at 2009-01-01.
        statement_type (str): Statement to retrieve.
        ctx (Context): Context object for error reporting and progress updates.
    Returns:
        Dict[str, Any]: Dictionary containing either an instance of StitchedStatement or an error message.
    """
    # Guard against dates before 2009-01-01
    original_date = date
    date = adjust_date_range(date)

    if date != original_date:
        console.log(f"Date range adjusted from {original_date} to {date} to avoid pre-2009 data")
        await ctx.warning(f"Date range adjusted from {original_date} to {date} to avoid pre-2009 data")

    console.log(
        f"Entering get_statements_impl with ticker={ticker}, form={form}, date={date}, statement_type={statement_type}"
    )
    available_forms = set(["10-Q", "10-K", "8-K"])

    if form not in available_forms:
        console.log(f"Form {form} is not available")
        await ctx.error(f"Form {form} is not available: choose from {available_forms}")
        return {"error": f"Form {form} is not available: choose from {available_forms}"}

    available_statements = set(
        [
            "AccountingPolicies",
            "BalanceSheet",
            "BalanceSheetParenthetical",
            "CashFlowStatement",
            "ComprehensiveIncome",
            "CoverPage",
            "Disclosures",
            "IncomeStatement",
            "SegmentDisclosure",
            "StatementOfEquity",
        ]
    )

    # Disallow certain statements as per requirements
    disallowed_statements = set(
        [
            "Disclosures",  # Limited useful information
            "CoverPage",  # Limited useful information
            "BalanceSheetParenthetical",  # Limited useful information
            "AccountingPolicies",  # Limited useful information
            "SegmentDisclosure",  # This statement is difficult to parse
        ]
    )
    if statement_type in disallowed_statements:
        console.log(f"Statement {statement_type} is not allowed")
        await ctx.error(f"Statement {statement_type} is not allowed.")
        return {"error": f"Statement {statement_type} is not allowed."}

    if statement_type not in available_statements:
        console.log(f"Statement {statement_type} is not available")
        await ctx.error(f"Statement {statement_type} is not available: choose from {available_statements}")
        return {"error": f"Statement {statement_type} is not available: choose from {available_statements}"}

    try:
        company = edgar.Company(ticker)
        console.log(f"Fetched company object for {ticker}")
        filings = company.get_filings()
        console.log(f"Fetched filings for {ticker}")
        filtered_filings = filings.filter(form=form, date=date)
        console.log(f"Filtered filings for form={form}, date={date}")
        xbrls = XBRLS.from_filings(filtered_filings)
        statements = xbrls.statements
        stitched_statement = statements[statement_type]  # StitchedStatement object

        # Check that we actually loaded some statements across periods
        found_stmt_types = set()
        found_periods = xbrls.get_periods()
        for xbrl in stitched_statement.xbrls.xbrl_list:
            statement = xbrl.get_all_statements()
            for stmt in statement:
                if stmt["type"]:
                    found_stmt_types.add(stmt["type"])
        period_count = len(found_periods)
        if period_count == 0 or len(found_stmt_types) == 0:
            msg = f"No statements found for {statement_type} (form={form}, ticker={ticker}, date={date})"
            console.log(f"{msg}")
            await ctx.error(msg)
            return {"error": msg}

        # Convert to markdown for JSON output, StitchedStatement is not serializable
        stitched_statement_md = stitched_statement.render().to_markdown()
        stitched_statement_md = reformat_markdown_financial_table(stitched_statement_md)

        console.log(f"Returning statement for {statement_type}")
        return {"stitched_statement": stitched_statement_md}
    except Exception as e:
        error_msg = format_error_msg(
            f"Error in get_statements_impl for ticker={ticker}, form={form}, date={date}, statement={statement_type}",
            e,
        )
        console.log(error_msg)
        await ctx.error(error_msg)
        return {"error": error_msg}


# DO NOT DELETE THIS COMMENTED OUT TOOL
#    mcp/server/session.py makes an MCP request with method="sampling/createMessage" which Cursor does not recognize
async def summarize_financial_report_impl(ticker: str, form: str, date: str, statement_type: str, ctx: Context):
    """
    Generate a financial reports summary using the output of get_statements_impl and an LLM prompt.
    Note: Any date range starting before 2009-01-01 will be automatically adjusted to start at 2009-01-01.
    The 'latest' value is supported for date, which uses the most recent ~400 days of filings.
    """
    # Guard against dates before 2009-01-01 and support 'latest'
    original_date = date
    date = adjust_date_range(date)
    if date != original_date:
        console.log(
            f"Date range adjusted from {original_date} to {date} to avoid pre-2009 data or to use 'latest' option"
        )
        await ctx.error(
            f"Date range adjusted from {original_date} to {date} to avoid pre-2009 data or to use 'latest' option"
        )

    console.log(
        f"Entering summarize_financial_report_impl with ticker={ticker}, form={form}, date={date}, statement_type={statement_type}"
    )
    try:
        # Get the statement data
        statement_result = await get_statements_impl(ticker, form, date, statement_type, ctx)
        if not statement_result or "error" in statement_result:
            error_msg = statement_result.get("error", "No statement data available to summarize.")
            console.log(f"Error in get_statements_impl: {error_msg}")
            await ctx.error(error_msg)
            return {"error": error_msg}

        # Convert the stitched statement a simpler markdown format
        markdown_text = statement_result["stitched_statement"]

        # Prepare the prompt for the LLM
        prompt = (
            f"You are a financial analyst. Given the following {statement_type} data from a {form} filing for {ticker} (date: {date}), "
            "write a concise, clear financial report summary suitable for an investor. "
            "Highlight key figures, trends, and any notable changes.\n\n"
            f"Statement Data in Markdown Format:\n{markdown_text}\n\nSummary:"
        )

        # Use the LLM to generate the summary
        console.log("Prompt prepared for LLM. Sending to ctx.sample...")
        if not hasattr(ctx, "sample"):
            error_msg = "Context object missing required 'sample' method"
            console.log(f"{error_msg}")
            await ctx.error(error_msg)
            return {"error": error_msg}
        try:
            response = await ctx.sample(prompt)
            if not hasattr(response, "text"):
                console.log(f"Unexpected response type: {type(response)}")
                raise Exception(f"Unexpected response type: {type(response)}")
            summary = response.text
        except AttributeError as e:
            error_msg = format_error_msg(
                f"LLM response missing required attributes in summarize_financial_report_impl for ticker={ticker}, form={form}, date={date}, statement={statement_type}",
                e,
            )
            console.log(error_msg)
            await ctx.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = format_error_msg(
                f"LLM prompt failed for ticker={ticker}, form={form}, date={date}, statement={statement_type}",
                e,
            )
            console.log(error_msg)
            await ctx.error(error_msg)
            return {"error": error_msg}
        console.log("LLM summary received.")

        return {"summary": summary}

    except Exception as e:
        error_msg = format_error_msg(
            f"Unexpected error in summarize_financial_report_impl for ticker={ticker}, form={form}, date={date}, statement={statement_type}",
            e,
        )
        console.log(error_msg)
        await ctx.error(error_msg)
        return {"error": error_msg}
