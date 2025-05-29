#!/usr/bin/env python3
"""
Test script for get_company_info function
"""

from fundamentals.utility.company_info import get_company_info

# Test with just a few custom tickers
print("Testing with custom tickers...")
df = get_company_info(tickers=['AAPL', 'GOOGL', 'TSLA'], file_name='test_3stocks')
print(f'Successfully processed {len(df)} companies')
print(f'Columns: {list(df.columns)[:10]}...')  # Show first 10 columns
print()

# Test with S&P 500 default (this would take longer in real usage)
print("Testing S&P 500 functionality by checking if it would work...")
# We'll just test the logic without actually running it on all 500 companies
try:
    # This will fetch the S&P 500 list but we'll limit it to first 2 for demo
    import requests
    from bs4 import BeautifulSoup
    
    wiki = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text
    table = BeautifulSoup(wiki, 'lxml').find('table', {'id':'constituents'})
    sp500_tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:3]]  # Just first 2
    sp500_tickers = [t.replace('.', '-') for t in sp500_tickers]
    
    # Test the S&P 500 logic with limited tickers
    df_sp500 = get_company_info(tickers=sp500_tickers, file_name='sp500_sample')
    print(f'S&P 500 sample test successful! Processed {len(df_sp500)} companies')
    
except Exception as e:
    print(f'Error in S&P 500 test: {e}') 