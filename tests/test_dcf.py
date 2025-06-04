#!/usr/bin/env python3
"""
Test script for DCF model integration with WACC data from macrotrends scraper.

This script demonstrates:
1. Loading financial data with WACC calculations
2. Using DCF model with WACC-based discount rates
3. Testing different discount rate scaling factors
4. Monte Carlo simulation with WACC data
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, uniform

# Add the fundamentals package to path
sys.path.insert(0, str(Path(__file__).parent))

from fundamentals.utility.dcf import StandardDCFModel
from fundamentals.utility.macrotrends_scraper import run_macrotrends_scraper


def test_dcf_wacc_integration():
    """Test DCF model integration with WACC data."""
    print("=" * 70)
    print("DCF-WACC Integration Test")
    print("=" * 70)
    
    # 1. Load financial data with WACC
    print("\n1. Loading financial data with WACC calculations...")
    df = run_macrotrends_scraper(['AAPL'], include_wacc=True, safety_preset='conservative')
    
    # Verify WACC data
    wacc_val = df['WACC'].iloc[-1]
    print(f"✓ Successfully loaded WACC data")
    print(f"   Current WACC: {wacc_val:.4f} ({wacc_val*100:.2f}%)")
    
    # 2. Test basic DCF calculation
    print("\n2. Testing basic DCF calculation with WACC...")
    model = StandardDCFModel()
    
    pv_fcf_fraction, dcf_val, stock_price = model.calculate_dcf(
        df, 
        discount_rate_scale=1.0,
        terminal_growth=0.025,
        time_horizon=10
    )
    
    print(f"✓ DCF calculation successful:")
    print(f"   FCF PV Fraction: {pv_fcf_fraction:.3f}")
    print(f"   DCF Value: ${dcf_val:,.0f}M")
    print(f"   Implied Stock Price: ${stock_price:.2f}")
    
    # 3. Test discount rate scaling
    print("\n3. Testing discount rate scaling effects...")
    print("   Scale Factor | Discount Rate | Stock Price")
    print("   -------------|---------------|------------")
    
    for scale in [0.8, 1.0, 1.2, 1.5]:
        _, _, price = model.calculate_dcf(df, discount_rate_scale=scale)
        effective_rate = wacc_val * scale
        print(f"   {scale:>8.1f}x | {effective_rate:>10.2%} | ${price:>9.2f}")
    
    # 4. Test Monte Carlo simulation
    print("\n4. Testing Monte Carlo simulation with WACC...")
    
    # Configure prior distributions
    distributions = {
        'discount_rate_scale': uniform(loc=0.8, scale=0.4),  # 0.8x to 1.2x WACC
        'growth_rate': norm(loc=0.05, scale=0.02),           # 5% ± 2% growth
        'terminal_growth': uniform(loc=0.02, scale=0.02),    # 2% to 4% terminal
        'time_horizon': uniform(loc=8, scale=4)              # 8-12 years lookback
    }
    
    # Simple correlation matrix
    correlation = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.1],
        [0.1, 0.2, 0.1, 1.0]
    ])
    
    model.configure_priors(distributions, correlation)
    model.simulate(df, n_samples=1000, random_state=42)
    
    # Summary statistics
    print(f"\n✓ Monte Carlo simulation complete:")
    print(f"   Stock Price: ${model.stock_prices.mean():.2f} ± ${model.stock_prices.std():.2f}")
    print(f"   Range: ${model.stock_prices.min():.2f} - ${model.stock_prices.max():.2f}")
    print(f"   Median: ${np.median(model.stock_prices):.2f}")
    
    # Show parameter samples
    print(f"\n   Parameter Samples Summary:")
    for param_name, samples in model.parameter_samples.items():
        if param_name == 'discount_rate_scale':
            print(f"   {param_name}: {samples.mean():.2f}x ± {samples.std():.2f}")
        elif 'rate' in param_name or 'growth' in param_name:
            print(f"   {param_name}: {samples.mean():.2%} ± {samples.std():.2%}")
        else:
            print(f"   {param_name}: {samples.mean():.2f} ± {samples.std():.2f}")
    
    # 5. Compare with current market price
    print("\n5. Market comparison...")
    current_price = df['Price'].iloc[-1]
    implied_price = model.stock_prices.mean()
    
    print(f"   Current Market Price: ${current_price:.2f}")
    print(f"   DCF Implied Price: ${implied_price:.2f}")
    print(f"   Difference: {((implied_price - current_price) / current_price) * 100:+.1f}%")
    
    if implied_price > current_price:
        print(f"   → DCF suggests stock may be UNDERVALUED")
    else:
        print(f"   → DCF suggests stock may be OVERVALUED")
    
    print("\n" + "=" * 70)
    print("✅ DCF-WACC Integration Test Complete!")
    print("=" * 70)
    
    return model, df


if __name__ == "__main__":
    test_dcf_wacc_integration()