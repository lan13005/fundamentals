"""
Helper module for WACC (Weighted Average Cost of Capital) calculations.

This module provides core WACC calculation functions that are used by 
macrotrends_scraper.py to add cost of capital metrics to financial data.
"""

import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


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
