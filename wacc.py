import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


def get_financial_data(ticker_symbol, start_date="2010-01-01", end_date="2025-06-01", verbose=False):
    """
    Extract quarterly financial data from yfinance.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: Quarterly financial data
    """
    if verbose:
        print(f"Fetching financial data for {ticker_symbol}...")
    
    ticker = yf.Ticker(ticker_symbol)
    income_q = ticker.quarterly_financials
    balance_q = ticker.quarterly_balance_sheet
    
    if verbose:
        print("Available income statement fields:")
        print(income_q.index.tolist()[:10])  # Show first 10 fields
        print("\nAvailable balance sheet fields:")
        print(balance_q.index.tolist()[:10])  # Show first 10 fields
    
    # Field mapping for different yfinance naming conventions
    field_mappings = {
        'Interest-Expense': ['Interest Expense', 'Interest Expense Non Operating', 'Net Interest Income'],
        'Total-Debt': ['Total Debt', 'Long Term Debt', 'Current Debt', 'Short Long Term Debt'],
        'Income-Taxes': ['Tax Provision', 'Income Tax Expense', 'Provision For Income Taxes', 'Tax Effect Of Unusual Items'],
        'Pre-Tax-Income': ['Pretax Income', 'Income Before Tax', 'Earnings Before Tax', 'Operating Income']
    }
    
    financial_data = {}
    
    # Extract Interest Expense from income statement
    for field_name, possible_names in field_mappings.items():
        found_data = None
        source_statement = income_q if field_name in ['Interest-Expense', 'Income-Taxes', 'Pre-Tax-Income'] else balance_q
        
        for name in possible_names:
            if name in source_statement.index:
                found_data = source_statement.loc[name]
                if verbose:
                    print(f"Found {field_name} as '{name}'")
                break
        
        if found_data is not None:
            financial_data[field_name] = found_data
        else:
            if verbose:
                print(f"Warning: {field_name} not found, using zeros")
            financial_data[field_name] = pd.Series(index=source_statement.columns, data=0, dtype=float)
    
    # Get market cap data
    info = ticker.info
    current_shares = info.get('sharesOutstanding', 0) if info else 0
    current_market_cap = info.get('marketCap', 0) if info else 0
    
    # Get historical price data for market cap calculation
    try:
        price_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        prices_quarterly = price_data.resample('QE').last()
        
        if current_shares > 0:
            # Calculate historical market cap
            market_cap_series = prices_quarterly * current_shares
            # Align with financial data dates using more robust matching
            market_cap_aligned = pd.Series(index=income_q.columns, dtype=float)
            
            for date in income_q.columns:
                # Convert to pandas datetime if needed
                target_date = pd.to_datetime(date)
                
                # Find the closest price date that's on or before the target date
                valid_prices = market_cap_series[market_cap_series.index <= target_date]
                if len(valid_prices) > 0:
                    closest_date = valid_prices.index[-1]  # Most recent date <= target
                    market_cap_aligned[date] = market_cap_series[closest_date]
                else:
                    # If no price data before this date, use the first available price
                    if len(market_cap_series) > 0:
                        market_cap_aligned[date] = market_cap_series.iloc[0]
                    
            financial_data['Market-Cap'] = market_cap_aligned
        else:
            if verbose:
                print("Warning: Could not get shares outstanding")
            financial_data['Market-Cap'] = pd.Series(index=income_q.columns, data=current_market_cap, dtype=float)
            
    except Exception as e:
        if verbose:
            print(f"Error getting price data: {e}")
        # Use current market cap as fallback for all periods
        financial_data['Market-Cap'] = pd.Series(index=income_q.columns, data=current_market_cap, dtype=float)
    
    return pd.DataFrame(financial_data)


def get_risk_free_rate(start_date="2010-01-01", end_date="2025-06-01", verbose=False):
    """
    Fetch 10-year Treasury yield from FRED as risk-free rate.
    
    Args:
        start_date (str): Start date
        end_date (str): End date
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.Series: Quarterly risk-free rates
    """
    if verbose:
        print("Fetching risk-free rate data from FRED...")
    
    fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GS10&scale=left&cosd={start_date}&coed={end_date}&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin"
    
    try:
        response = requests.get(fred_url)
        response.raise_for_status()
        rf_data = pd.read_csv(StringIO(response.text))
        rf_data['observation_date'] = pd.to_datetime(rf_data['observation_date'])
        rf_data = rf_data.set_index('observation_date')
        rf_daily = rf_data['GS10'] / 100  # Convert percentage to decimal
        rf_quarterly = rf_daily.resample('QE').last()
        if verbose:
            print(f"Successfully fetched {len(rf_daily)} risk-free rate observations")
        return rf_quarterly
    except Exception as e:
        if verbose:
            print(f"Error fetching FRED data: {e}")
            print("Using approximation of 2.5% risk-free rate")
        quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="QE")
        return pd.Series(index=quarterly_dates, data=0.025, name='GS10')


def compute_rolling_beta(ticker_symbol, risk_free_quarterly, start_date="2010-01-01", end_date="2025-06-01", window_months=24, verbose=False):
    """
    Compute rolling beta using 24-month rolling window.
    
    Args:
        ticker_symbol (str): Stock ticker
        risk_free_quarterly (pd.Series): Quarterly risk-free rates
        start_date (str): Start date
        end_date (str): End date
        window_months (int): Rolling window in months
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.Series: Quarterly beta values
    """
    if verbose:
        print("Computing rolling beta...")
    
    # Download price data
    stock_price = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    market_price = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # Handle potential multi-level columns from yfinance
    if hasattr(stock_price, 'columns'):
        stock_price = stock_price.iloc[:, 0] if len(stock_price.columns) > 0 else stock_price
    if hasattr(market_price, 'columns'):
        market_price = market_price.iloc[:, 0] if len(market_price.columns) > 0 else market_price
    
    # Compute monthly returns
    stock_ret = stock_price.resample('ME').last().pct_change().dropna()
    market_ret = market_price.resample('ME').last().pct_change().dropna()
    
    # Convert quarterly risk-free to monthly (approximate)
    rf_monthly = risk_free_quarterly.resample('ME').ffill() / 12
    
    # Align indexes and compute excess returns
    common_idx = stock_ret.index.intersection(market_ret.index).intersection(rf_monthly.index)
    
    if verbose:
        print(f"Computing beta for {len(common_idx)} months, window={window_months}")
    
    if len(common_idx) == 0:
        print("ERROR: No common dates found between stock, market, and risk-free data!")
        return pd.Series(dtype=float, name='beta')
    
    # Check for NaN values in each component
    stock_ret_aligned = stock_ret.loc[common_idx]
    market_ret_aligned = market_ret.loc[common_idx]
    rf_monthly_aligned = rf_monthly.loc[common_idx]
    
    # Compute excess returns with better handling
    stock_ex = stock_ret_aligned - rf_monthly_aligned
    market_ex = market_ret_aligned - rf_monthly_aligned
    
    # Compute rolling beta
    betas = []
    beta_dates = []
    
    valid_beta_count = 0
    
    for i in range(window_months, len(common_idx)):
        sub_idx = common_idx[i-window_months:i]
        y_sub = stock_ex.loc[sub_idx].values
        x_sub = market_ex.loc[sub_idx].values
        
        # Ensure we have scalar values, not arrays
        if len(y_sub.shape) > 1:
            y_sub = y_sub.flatten()
        if len(x_sub.shape) > 1:
            x_sub = x_sub.flatten()
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(y_sub) | np.isnan(x_sub))
        valid_count = valid_mask.sum()
        
        if valid_count < 12:  # Need at least 12 months of data
            betas.append(np.nan)
            beta_dates.append(common_idx[i])
            continue
            
        y_clean = y_sub[valid_mask]
        x_clean = x_sub[valid_mask]
        
        # Linear regression using numpy
        X_sub = np.column_stack([np.ones(len(x_clean)), x_clean])
        
        try:
            coeffs = np.linalg.solve(X_sub.T @ X_sub, X_sub.T @ y_clean)
            beta = float(coeffs[1])  # Ensure scalar value
            
            if not np.isnan(beta) and np.isfinite(beta):
                valid_beta_count += 1
            else:
                beta = np.nan
                
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(X_sub, y_clean, rcond=None)[0]
            beta = float(coeffs[1])  # Ensure scalar value
            if not np.isnan(beta) and np.isfinite(beta):
                valid_beta_count += 1
            else:
                beta = np.nan
        except Exception:
            beta = np.nan
        
        betas.append(beta)
        beta_dates.append(common_idx[i])
    
    if verbose:
        print(f"Calculated {valid_beta_count} valid betas out of {len(betas)} attempts")
    
    # Convert to quarterly - ensure we get scalar values
    beta_series = pd.Series(betas, index=beta_dates, name='beta')
    beta_quarterly = beta_series.resample('QE').last()
    
    return beta_quarterly


def scrape_damodaran_erp(verbose=False):
    """
    Scrape historical equity risk premium data from Damodaran's website.
    
    Args:
        verbose (bool): Whether to print debug information
        
    Returns:
        dict: Dictionary mapping year to ERP value
    """
    if verbose:
        print("Scraping historical ERP data from Damodaran's website...")
    
    try:
        url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        erp_data = {}
        
        for table in tables:
            rows = table.find_all('tr')
            header_found = False
            erp_column_idx = None
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) > 4:
                    cell_texts = [cell.get_text().strip() for cell in cells]
                    
                    # Look for header row
                    if any('Year' in text for text in cell_texts) and any('ERP' in text or 'Premium' in text for text in cell_texts):
                        header_found = True
                        for j, text in enumerate(cell_texts):
                            if 'Implied ERP' in text or ('ERP' in text and 'FCFE' in text):
                                erp_column_idx = j
                                break
                        continue
                    
                    # Parse data rows
                    if header_found and erp_column_idx is not None and len(cell_texts) > erp_column_idx:
                        year_match = re.search(r'(\d{4})', cell_texts[0])
                        erp_match = re.search(r'([\d.]+)%', cell_texts[erp_column_idx])
                        
                        if year_match and erp_match:
                            year = int(year_match.group(1))
                            erp_value = float(erp_match.group(1)) / 100
                            erp_data[year] = erp_value
            
            if erp_data:
                break
        
        # Fallback parsing if table parsing fails
        if not erp_data:
            page_text = soup.get_text()
            for line in page_text.split('\n'):
                if re.search(r'\d{4}.*\d+\.\d+%', line):
                    year_match = re.search(r'(\d{4})', line)
                    erp_matches = re.findall(r'(\d+\.\d+)%', line)
                    
                    if year_match and erp_matches:
                        year = int(year_match.group(1))
                        erp_value = float(erp_matches[-1]) / 100
                        erp_data[year] = erp_value
        
        return erp_data
        
    except Exception as e:
        if verbose:
            print(f"Error scraping ERP data: {e}")
        return {}


def get_historical_erp(start_date="2010-01-01", end_date="2025-06-01", verbose=False):
    """
    Get historical ERP data and convert to quarterly time series.
    
    Args:
        start_date (str): Start date
        end_date (str): End date
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.Series: Quarterly ERP values
    """
    historical_erp = scrape_damodaran_erp(verbose)
    
    if historical_erp:
        if verbose:
            print(f"Successfully scraped ERP data for {len(historical_erp)} years")
            print(f"Year range: {min(historical_erp.keys())} - {max(historical_erp.keys())}")
        
        # Create quarterly ERP series
        quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="QE")
        erp_quarterly = pd.Series(index=quarterly_dates, name='ERP', dtype=float)
        
        for quarter_date in quarterly_dates:
            year = quarter_date.year
            # Find most recent ERP value
            erp_value = None
            for search_year in range(year, year - 10, -1):
                if search_year in historical_erp:
                    erp_value = historical_erp[search_year]
                    break
            
            if erp_value is not None:
                erp_quarterly.loc[quarter_date] = erp_value
        
        # Fill any remaining missing values
        avg_erp = erp_quarterly.mean()
        if pd.isna(avg_erp):
            avg_erp = 0.055  # fallback
        erp_quarterly = erp_quarterly.fillna(avg_erp)
        
        return erp_quarterly
    else:
        if verbose:
            print("Failed to scrape ERP data, using fallback value of 5.5%")
        quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="QE")
        return pd.Series(index=quarterly_dates, data=0.055, name='ERP')


def wacc(ticker_symbol, start_date="2010-01-01", end_date="2025-06-01", verbose=False):
    """
    Calculate Weighted Average Cost of Capital (WACC) for a given ticker.
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: Quarterly WACC analysis with all components
    """
    print(f"\n{'='*60}")
    print(f"WACC Analysis for {ticker_symbol}")
    print(f"{'='*60}")
    
    # Ensure we don't calculate beyond current date for safety
    current_date = pd.Timestamp.now()
    max_end_date = pd.Timestamp(end_date)
    if max_end_date > current_date:
        end_date = current_date.strftime('%Y-%m-%d')
        print(f"Limiting analysis to {end_date} to avoid unreliable future data")
    
    # 1. Get financial data
    financial_data = get_financial_data(ticker_symbol, start_date, end_date, verbose)
    
    # 2. Get risk-free rate
    rf_quarterly = get_risk_free_rate(start_date, end_date, verbose)
    
    # 3. Compute rolling beta
    beta_quarterly = compute_rolling_beta(ticker_symbol, rf_quarterly, start_date, end_date, verbose=verbose)
    
    # 4. Get historical ERP
    erp_quarterly = get_historical_erp(start_date, end_date, verbose)
    
    # 5. Create main quarterly DataFrame - limit to current date
    quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="QE")
    # Filter out future quarters beyond current date - use most recent completed quarter
    # Since we're in June 2025, Q1 2025 (March 31) should be available
    current_quarter = ((current_date.month - 1) // 3 + 1)
    current_year = current_date.year
    
    # Be conservative - only include quarters that are definitely completed
    if current_date.month <= 3:  # Q1
        safe_quarter_end = pd.Timestamp(current_year - 1, 12, 31)  # Previous year Q4
    elif current_date.month <= 6:  # Q2  
        safe_quarter_end = pd.Timestamp(current_year, 3, 31)  # Current year Q1
    elif current_date.month <= 9:  # Q3
        safe_quarter_end = pd.Timestamp(current_year, 6, 30)  # Current year Q2  
    else:  # Q4
        safe_quarter_end = pd.Timestamp(current_year, 9, 30)  # Current year Q3
    
    quarterly_dates = quarterly_dates[quarterly_dates <= safe_quarter_end]
    df_wacc = pd.DataFrame(index=quarterly_dates)
    
    # 6. Merge all data
    df_wacc = df_wacc.merge(financial_data, left_index=True, right_index=True, how='left')
    df_wacc = df_wacc.merge(rf_quarterly.to_frame('Rf'), left_index=True, right_index=True, how='left')
    df_wacc = df_wacc.merge(beta_quarterly.to_frame('beta'), left_index=True, right_index=True, how='left')
    df_wacc = df_wacc.merge(erp_quarterly.to_frame('ERP'), left_index=True, right_index=True, how='left')
    
    # 7. Calculate WACC components
    if verbose:
        print("Computing WACC components...")
        print(f"Available quarters with financial data: {df_wacc['Total-Debt'].count()}")
        print(f"Available quarters with market cap: {df_wacc['Market-Cap'].count()}")
        print(f"Available quarters with interest expense: {df_wacc['Interest-Expense'].count()}")
    
    # Cost of debt - fill with zeros for companies with no debt
    df_wacc['Debt_Avg'] = (df_wacc['Total-Debt'].shift(1) + df_wacc['Total-Debt']) / 2
    
    # Calculate R_d_pre_tax, but fill with 0 if no debt or interest expense
    df_wacc['R_d_pre_tax'] = np.where(
        (df_wacc['Debt_Avg'] > 0) & (df_wacc['Interest-Expense'].notna()) & (df_wacc['Interest-Expense'] > 0),
        df_wacc['Interest-Expense'] / df_wacc['Debt_Avg'],
        0.0  # Zero cost if no debt or no interest expense
    )
    
    # Calculate tax rate, use corporate rate as fallback
    df_wacc['Tax_Rate'] = np.where(
        (df_wacc['Pre-Tax-Income'] != 0) & (df_wacc['Pre-Tax-Income'].notna()) & (df_wacc['Income-Taxes'].notna()),
        df_wacc['Income-Taxes'] / df_wacc['Pre-Tax-Income'],
        0.21  # Use standard corporate tax rate as fallback
    )
    
    # Ensure tax rate is between 0 and 1
    df_wacc['Tax_Rate'] = df_wacc['Tax_Rate'].clip(0, 1)
    
    # After-tax cost of debt (will be 0 if R_d_pre_tax is 0)
    df_wacc['R_d'] = df_wacc['R_d_pre_tax'] * (1 - df_wacc['Tax_Rate'])
    
    # Cost of equity
    df_wacc['R_e'] = df_wacc['Rf'] + df_wacc['beta'] * df_wacc['ERP']
    
    # Capital structure weights - fix FutureWarning with proper handling
    df_wacc['Market_Value_Equity'] = df_wacc['Market-Cap']
    
    # Suppress FutureWarning by setting option temporarily
    with pd.option_context('future.no_silent_downcasting', True):
        df_wacc['Market_Value_Debt'] = df_wacc['Total-Debt'].fillna(0.0)
    
    # Calculate total capital - use equity value if debt is missing
    df_wacc['Total_Capital'] = df_wacc['Market_Value_Equity'].fillna(0.0) + df_wacc['Market_Value_Debt']
    
    # Calculate weights - handle cases where we have equity but no debt - use safe division
    with pd.option_context('future.no_silent_downcasting', True):
        total_capital_safe = df_wacc['Total_Capital'].replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    
    df_wacc['Weight_Equity'] = np.where(
        df_wacc['Total_Capital'] > 0,
        df_wacc['Market_Value_Equity'].fillna(0.0) / total_capital_safe,
        1.0  # 100% equity if no total capital data
    )
    
    df_wacc['Weight_Debt'] = np.where(
        df_wacc['Total_Capital'] > 0,
        df_wacc['Market_Value_Debt'] / total_capital_safe,
        0.0  # 0% debt if no total capital data
    )
    
    # WACC calculation - calculate wherever we have cost of equity
    # If R_d is 0 (no debt), WACC = Weight_Equity * R_e + Weight_Debt * 0 = Weight_Equity * R_e
    df_wacc['WACC'] = np.where(
        df_wacc['R_e'].notna() & (df_wacc['Total_Capital'] > 0),
        df_wacc['Weight_Equity'] * df_wacc['R_e'] + df_wacc['Weight_Debt'] * df_wacc['R_d'],
        np.nan
    )
    
    # Forward fill missing market cap for better coverage
    df_wacc['Market_Value_Equity'] = df_wacc['Market_Value_Equity'].ffill()
    
    # Fix FutureWarning for debt fillna
    with pd.option_context('future.no_silent_downcasting', True):
        df_wacc['Market_Value_Debt'] = df_wacc['Market_Value_Debt'].fillna(0.0)
    
    # Recalculate weights with forward-filled data
    df_wacc['Total_Capital'] = df_wacc['Market_Value_Equity'].fillna(0.0) + df_wacc['Market_Value_Debt']
    
    # Use safe division again
    with pd.option_context('future.no_silent_downcasting', True):
        total_capital_safe = df_wacc['Total_Capital'].replace(0, np.nan)
    
    df_wacc['Weight_Equity'] = np.where(
        df_wacc['Total_Capital'] > 0,
        df_wacc['Market_Value_Equity'].fillna(0.0) / total_capital_safe,
        1.0
    )
    
    df_wacc['Weight_Debt'] = np.where(
        df_wacc['Total_Capital'] > 0,
        df_wacc['Market_Value_Debt'] / total_capital_safe,
        0.0
    )
    
    # Recalculate WACC with better data coverage
    df_wacc['WACC'] = np.where(
        df_wacc['R_e'].notna() & (df_wacc['Total_Capital'] > 0),
        df_wacc['Weight_Equity'] * df_wacc['R_e'] + df_wacc['Weight_Debt'] * df_wacc['R_d'],
        np.nan
    )
    
    # Clean up columns for final output
    output_columns = [
        'Rf', 'beta', 'ERP', 'R_e', 'R_d_pre_tax', 'Tax_Rate', 'R_d',
        'Market_Value_Equity', 'Market_Value_Debt', 'Weight_Equity', 'Weight_Debt', 'WACC'
    ]
    
    df_final = df_wacc[output_columns].copy()
    
    # Final filter: Remove any future dates that might have slipped through
    df_final = df_final[df_final.index <= safe_quarter_end]
    
    # Display results
    print(f"\n{'WACC Analysis Results':<30}")
    print("="*50) 
    
    # Show recent data
    recent_data = df_final.dropna(subset=['WACC']).tail(8)
    if not recent_data.empty:
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        current_wacc = recent_data['WACC'].iloc[-1]
        avg_wacc = recent_data['WACC'].mean()
        current_beta = recent_data['beta'].iloc[-1] if 'beta' in recent_data.columns else np.nan
        current_debt_weight = recent_data['Weight_Debt'].iloc[-1]
        current_equity_weight = recent_data['Weight_Equity'].iloc[-1]
        
        # Handle array values by extracting scalar
        if hasattr(current_wacc, '__len__') and len(current_wacc) > 1:
            current_wacc = current_wacc[0] if len(current_wacc) > 0 else np.nan
        if hasattr(avg_wacc, '__len__') and len(avg_wacc) > 1:
            avg_wacc = avg_wacc[0] if len(avg_wacc) > 0 else np.nan
        if hasattr(current_beta, '__len__') and len(current_beta) > 1:
            current_beta = current_beta[0] if len(current_beta) > 0 else np.nan
        if hasattr(current_debt_weight, '__len__') and len(current_debt_weight) > 1:
            current_debt_weight = current_debt_weight[0] if len(current_debt_weight) > 0 else np.nan
        if hasattr(current_equity_weight, '__len__') and len(current_equity_weight) > 1:
            current_equity_weight = current_equity_weight[0] if len(current_equity_weight) > 0 else np.nan
        
        try:
            print(f"Current WACC: {float(current_wacc):.4f} ({float(current_wacc)*100:.2f}%)")
            print(f"Average WACC: {float(avg_wacc):.4f} ({float(avg_wacc)*100:.2f}%)")
            if not np.isnan(current_beta):
                print(f"Current Beta: {float(current_beta):.4f}")
            print(f"Current Debt Weight: {float(current_debt_weight):.4f} ({float(current_debt_weight)*100:.1f}%)")
            print(f"Current Equity Weight: {float(current_equity_weight):.4f} ({float(current_equity_weight)*100:.1f}%)")
        except (ValueError, TypeError) as e:
            print(f"Note: Some values could not be formatted: {e}")
            
    else:
        print("No valid WACC data computed")
        if verbose:
            print("\nDebugging information:")
            print(f"Financial data shape: {financial_data.shape}")
            print(f"Financial data columns: {financial_data.columns.tolist()}")
            print(f"RF quarterly shape: {rf_quarterly.shape}")
            print(f"Beta quarterly shape: {beta_quarterly.shape}")
            print(f"ERP quarterly shape: {erp_quarterly.shape}")
            print(f"Final dataframe shape: {df_final.shape}")
            print(f"Non-null counts in final dataframe:")
            print(df_final.count())
            
            # Show sample of data for debugging
            print(f"\nSample data (last 5 rows):")
            sample_cols = ['Rf', 'beta', 'ERP', 'R_e', 'WACC']
            print(df_final[sample_cols].tail().round(4))
    
    return df_final


# Example usage
if __name__ == "__main__":
    # Calculate WACC for ENPH - if it fails due to network, try AAPL
    try:
        print("Attempting ENPH analysis...")
        apple_wacc = wacc("ENPH", verbose=False)  # Set to True for debug output
        ticker_used = "ENPH"
    except Exception as e:
        print(f"ENPH failed ({e}), trying AAPL...")
        apple_wacc = wacc("AAPL", verbose=False)
        ticker_used = "AAPL"
    
    # Display the last few quarters
    print(f"\nFinal WACC DataFrame for {ticker_used} (last 10 quarters):")
    result = apple_wacc.dropna().tail(n=10)
    if not result.empty:
        print(result.round(4))
    else:
        print("No valid WACC data available")