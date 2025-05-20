import edgar
from fastmcp import Context

async def get_statement_impl(ticker: str, form: str, date: str, statement: str, ctx: Context):

    available_forms = set([
        "10-Q",
        "10-K",
        "8-K"
    ])

    if form not in available_forms:
        await ctx.error(f"Form {form} is not available: choose from {available_forms}")
        return {}
    
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

    if statement not in available_statements:
        await ctx.error(f"Statement {statement} is not available: choose from {available_statements}")
        return {}

    cache_key = f"{ticker}_{form}_{date}_{statement}"

    if cache_key in ctx.session:
        return ctx.session[cache_key]

    company = edgar.Company(ticker)
    filings = company.get_filings()
    xbrl = edgar.XBRLS.from_filings(filings.filter(form=form, date=date))
    statements = xbrl.statements
    statement = statements[statement]
    
    ctx.session[cache_key] = statement
    return statement
