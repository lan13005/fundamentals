#!/usr/bin/env python3
"""
Demonstration of WACC Integration in Macrotrends Scraper

This script demonstrates how to use the integrated WACC calculation
functionality in the macrotrends scraper.
"""

import sys
import pandas as pd
from pathlib import Path

# Add the fundamentals package to path
sys.path.insert(0, str(Path(__file__).parent))

from fundamentals.utility.macrotrends_scraper import run_macrotrends_scraper


def demo_wacc_integration():
    """Demonstrate WACC integration with real examples."""
    print("=" * 70)
    print("WACC Integration Demo - Macrotrends Scraper")
    print("=" * 70)
    
    # Demo symbols
    demo_symbols = ["AAPL"]  # Use well-known symbols
    
    print(f"\nFetching financial data with WACC calculations for: {', '.join(demo_symbols)}")
    print("This may take a few moments as it fetches:")
    print("  - Historical financial data from Macrotrends")
    print("  - Risk-free rates from FRED")
    print("  - Beta calculations from stock/market returns")
    print("  - Equity Risk Premium from Damodaran's data")
    print("-" * 70)
    
    try:
        # Fetch data with WACC calculations enabled
        df = run_macrotrends_scraper(
            symbols=demo_symbols,
            force=False,  # Use cached data if available for faster demo
            include_wacc=True,  # Enable WACC calculations
            safety_preset="conservative"  # Use conservative scraping settings
        )
        
        if df.empty:
            print("❌ No data retrieved. Please check the symbols or try again.")
            return
        
        print(f"✅ Successfully retrieved data!")
        print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show which columns are available
        wacc_columns = ['Rf', 'beta', 'ERP', 'R-e', 'WACC']
        financial_columns = ['Market-Cap', 'Long-Term-Debt', 'Income-Taxes', 'Pre-Tax-Income']
        
        available_wacc = [col for col in wacc_columns if col in df.columns]
        available_financial = [col for col in financial_columns if col in df.columns]
        
        print(f"\n📈 WACC Components Available: {available_wacc}")
        print(f"💰 Key Financial Data Available: {available_financial}")
        
        # Show summary statistics for WACC components
        if available_wacc:
            print(f"\n📊 WACC Summary Statistics:")
            wacc_summary = df[available_wacc].describe()
            print(wacc_summary.round(4))
        
        # Show recent data for each symbol
        print(f"\n📋 Recent Data by Symbol:")
        print("-" * 50)
        
        # Group by symbol if available
        if 'symbol' in df.columns or len(demo_symbols) > 1:
            for symbol in demo_symbols:
                # Filter data for this symbol (you may need to adjust this based on how data is structured)
                symbol_data = df.head(10)  # For demo, show first 10 rows
                
                print(f"\n🏢 {symbol}:")
                if available_wacc:
                    recent_wacc = symbol_data[available_wacc].dropna()
                    if not recent_wacc.empty:
                        latest_row = recent_wacc.iloc[-1]
                        print(f"   📊 Latest WACC Components:")
                        for col in available_wacc:
                            value = latest_row[col]
                            if pd.notna(value):
                                if col == 'WACC':
                                    print(f"      {col}: {value:.4f} ({value*100:.2f}%)")
                                elif col in ['Rf', 'ERP', 'R-e']:
                                    print(f"      {col}: {value:.4f} ({value*100:.2f}%)")
                                else:  # beta
                                    print(f"      {col}: {value:.4f}")
                    else:
                        print(f"   ⚠️  No valid WACC data available")
                else:
                    print(f"   ❌ WACC columns not found")
        
        # Show comparison of cost components
        if 'R-e' in df.columns and 'Rf' in df.columns:
            print(f"\n📈 Cost of Equity vs Risk-Free Rate Analysis:")
            valid_data = df[['Rf', 'R-e']].dropna()
            if not valid_data.empty:
                rf_avg = valid_data['Rf'].mean()
                re_avg = valid_data['R-e'].mean()
                equity_premium = re_avg - rf_avg
                print(f"   📊 Average Risk-Free Rate: {rf_avg:.4f} ({rf_avg*100:.2f}%)")
                print(f"   📊 Average Cost of Equity: {re_avg:.4f} ({re_avg*100:.2f}%)")
                print(f"   📊 Equity Risk Premium: {equity_premium:.4f} ({equity_premium*100:.2f}%)")
        
        # Usage tips
        print(f"\n💡 Usage Tips:")
        print(f"   • The dataframe includes historical quarterly data with datetime index")
        print(f"   • WACC columns: {wacc_columns}")
        print(f"   • Financial data from Macrotrends with kebab-case column names")
        print(f"   • Set include_wacc=False to skip WACC calculations for faster processing")
        print(f"   • Use force=True to refresh cached data")
        
        print(f"\n📝 Example: Calculate debt-to-equity ratios")
        if 'Long-Term-Debt' in df.columns and 'Market-Cap' in df.columns:
            sample_data = df[['Long-Term-Debt', 'Market-Cap']].dropna().tail(5)
            if not sample_data.empty:
                sample_data['Debt-to-Market-Cap'] = sample_data['Long-Term-Debt'] / sample_data['Market-Cap']
                print(sample_data.round(4))
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting Tips:")
        print(f"   • Check internet connection for data fetching")
        print(f"   • Verify symbols are valid ticker symbols")
        print(f"   • Try with a single symbol first")
        print(f"   • Check if cached data exists in macro_data/parquet/")
    
    print(f"\n" + "=" * 70)
    print("Demo Complete! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    demo_wacc_integration() 