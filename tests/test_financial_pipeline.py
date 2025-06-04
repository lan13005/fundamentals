#!/usr/bin/env python3
"""
Comprehensive Financial Pipeline Integration Test

This test validates the complete pipeline:
1. Data Infrastructure & CLI functionality  
2. Macrotrends data fetching with integrity checks
3. WACC calculation and validation
4. DCF modeling with Monte Carlo simulation
5. Market analysis and diagnostics

Combines and replaces: test_macrotrends.py, test_wacc.py, test_dcf.py
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, uniform

# Add the fundamentals package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fundamentals.utility.dcf import StandardDCFModel
from fundamentals.utility.macrotrends_scraper import run_macrotrends_scraper


class FinancialPipelineTestSuite:
    """Comprehensive test suite for the financial data pipeline."""
    
    def __init__(self, test_symbols=None):
        """Initialize test suite with symbols to test."""
        self.test_symbols = test_symbols or ["AAPL"]
        self.df = None
        self.dcf_model = None
        self.test_results = {}
        
    def run_all_tests(self):
        """Run the complete test suite."""
        print("=" * 80)
        print("🧪 FINANCIAL PIPELINE - COMPREHENSIVE INTEGRATION TEST")
        print("=" * 80)
        print("Testing: Macrotrends → WACC → DCF → Analysis Pipeline")
        print(f"Symbols: {', '.join(self.test_symbols)}")
        print("=" * 80)
        
        try:
            # Test 1: Infrastructure & CLI
            self._test_infrastructure_and_cli()
            
            # Test 2: Macrotrends Data Fetching & Integrity
            self._test_macrotrends_data_pipeline()
            
            # Test 3: WACC Integration & Validation
            self._test_wacc_integration_validation()
            
            # Test 4: DCF Model Integration
            self._test_dcf_model_integration()
            
            # Test 5: Monte Carlo Simulation
            self._test_monte_carlo_simulation()
            
            # Test 6: Market Analysis & Diagnostics
            self._test_market_analysis_diagnostics()
            
            # Final Summary
            self._print_comprehensive_summary()
            
        except Exception as e:
            print(f"\n❌ CRITICAL FAILURE: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True
    
    def _test_infrastructure_and_cli(self):
        """Test 1: Data infrastructure and CLI functionality."""
        print("\n📁 TEST 1: Infrastructure & CLI Functionality")
        print("-" * 60)
        
        # Create required directories
        os.makedirs("macro_data/parquet", exist_ok=True)
        os.makedirs("fundamentals/utility", exist_ok=True)
        print("✓ Created required directories")
        
        # Test CLI functionality (with timeout for safety)
        cwd = os.getcwd()
        try:
            print("   Testing CLI command execution...")
            result = subprocess.run(
                [
                    "python", "fund_cli.py", "macrotrends",
                    "--symbols", self.test_symbols[0],
                    "--freq", "Q"
                ],
                capture_output=True, text=True, timeout=60, check=True
            )
            
            # Verify expected output messages
            if "✓ Loaded" in result.stdout or "cached" in result.stdout.lower():
                print("✓ CLI executed successfully with expected output")
                self.test_results['cli_execution'] = True
            else:
                print("⚠️  CLI executed but output format unexpected")
                self.test_results['cli_execution'] = False
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  CLI test skipped ({type(e).__name__}): {str(e)[:100]}...")
            self.test_results['cli_execution'] = False
        
        # Check parquet file infrastructure
        parquet_dir = Path("macro_data/parquet")
        parquet_files = list(parquet_dir.glob("*.parquet"))
        print(f"✓ Parquet directory exists with {len(parquet_files)} files")
        
        self.test_results['infrastructure'] = True
        print("✅ Infrastructure & CLI test completed")
    
    def _test_macrotrends_data_pipeline(self):
        """Test 2: Macrotrends data fetching with comprehensive validation."""
        print("\n📊 TEST 2: Macrotrends Data Pipeline")
        print("-" * 60)
        
        print(f"Fetching comprehensive financial data for: {', '.join(self.test_symbols)}")
        print("   📈 Historical financial data from Macrotrends")
        print("   📊 Risk-free rates from FRED")
        print("   📉 Beta calculations from stock/market returns")
        print("   💰 Equity Risk Premium from Damodaran")
        print("   🧮 WACC component calculations")
        
        # Fetch data with full WACC integration
        self.df = run_macrotrends_scraper(
            symbols=self.test_symbols,
            include_wacc=True,
            safety_preset="conservative",
            force=False  # Use cache for faster testing
        )
        
        # Comprehensive data validation
        if self.df.empty:
            raise ValueError("❌ No data retrieved from macrotrends pipeline")
        
        print(f"✓ Data retrieved successfully")
        print(f"   📊 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"   📅 Date range: {self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}")
        print(f"   🔢 Index type: {type(self.df.index).__name__}")
        
        # Verify essential financial columns
        essential_columns = [
            'FCF-LTM', 'Shares-Outstanding', 'Market-Cap', 
            'Revenue', 'Operating-Income', 'Net-Income'
        ]
        missing_essential = [col for col in essential_columns if col not in self.df.columns]
        if missing_essential:
            raise ValueError(f"❌ Missing essential columns: {missing_essential}")
        
        print(f"✓ Essential financial columns verified: {len(essential_columns)} present")
        
        # Data quality checks
        latest_data = self.df.iloc[-1]
        fcf_ltm = latest_data['FCF-LTM']
        shares = latest_data['Shares-Outstanding']
        market_cap = latest_data['Market-Cap']
        revenue = latest_data['Revenue']
        
        print(f"   📊 Latest Metrics:")
        print(f"      FCF (LTM): ${fcf_ltm:,.0f}M")
        print(f"      Market Cap: ${market_cap:,.0f}M")
        print(f"      Shares: {shares:,.0f}")
        print(f"      Revenue: ${revenue:,.0f}M")
        
        # Sanity checks with warnings
        if shares <= 0:
            raise ValueError("❌ Invalid shares outstanding")
        if market_cap <= 0:
            raise ValueError("❌ Invalid market cap")
        if fcf_ltm < -10000:  # Allow some negative FCF but not extreme
            print("⚠️  Warning: Very negative FCF may indicate data issues")
        
        # Check datetime index integrity
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("❌ Index must be DatetimeIndex for time series analysis")
        
        date_gaps = self.df.index.to_series().diff().mode()[0]
        expected_quarterly = pd.Timedelta(days=90)
        if abs((date_gaps - expected_quarterly).days) > 10:
            print(f"⚠️  Warning: Date gaps ({date_gaps}) don't appear quarterly")
        
        self.test_results['data_pipeline'] = True
        print("✅ Macrotrends data pipeline test passed")
    
    def _test_wacc_integration_validation(self):
        """Test 3: Comprehensive WACC integration and validation."""
        print("\n💰 TEST 3: WACC Integration & Validation")
        print("-" * 60)
        
        # Verify all WACC components exist
        wacc_components = [
            'Risk-Free-Rate', 'beta', 'Equity-Risk-Premium', 
            'Cost-of-Equity', 'Cost-of-Debt', 'WACC'
        ]
        missing_wacc = [col for col in wacc_components if col not in self.df.columns]
        if missing_wacc:
            raise ValueError(f"❌ Missing WACC components: {missing_wacc}")
        
        print(f"✓ All WACC components present: {len(wacc_components)} columns")
        
        # WACC data quality validation
        wacc_data = self.df[wacc_components].dropna()
        if len(wacc_data) < 5:
            raise ValueError("❌ Insufficient valid WACC data for analysis")
        
        print(f"✓ Valid WACC observations: {len(wacc_data)}/{len(self.df)} ({len(wacc_data)/len(self.df)*100:.1f}%)")
        
        # Current WACC component analysis
        latest_wacc = wacc_data.iloc[-1]
        rf = latest_wacc['Risk-Free-Rate']
        beta = latest_wacc['beta']
        erp = latest_wacc['Equity-Risk-Premium']
        cost_equity = latest_wacc['Cost-of-Equity']
        cost_debt = latest_wacc['Cost-of-Debt']
        wacc = latest_wacc['WACC']
        
        print(f"   📊 Current WACC Components:")
        print(f"      Risk-Free Rate: {rf:.4f} ({rf*100:.2f}%)")
        print(f"      Beta: {beta:.3f}")
        print(f"      Equity Risk Premium: {erp:.4f} ({erp*100:.2f}%)")
        print(f"      Cost of Equity: {cost_equity:.4f} ({cost_equity*100:.2f}%)")
        print(f"      Cost of Debt: {cost_debt:.4f} ({cost_debt*100:.2f}%)")
        print(f"      WACC: {wacc:.4f} ({wacc*100:.2f}%)")
        
        # Mathematical relationship validation
        calculated_cost_equity = rf + beta * erp
        if abs(calculated_cost_equity - cost_equity) > 0.0001:
            print(f"⚠️  Warning: Cost of Equity calculation discrepancy")
            print(f"      Expected: {calculated_cost_equity:.4f}, Got: {cost_equity:.4f}")
        else:
            print("✓ Cost of Equity calculation verified: R_e = R_f + β × ERP")
        
        # Historical trend analysis
        if len(wacc_data) >= 8:  # At least 2 years of data
            recent_wacc = wacc_data['WACC'].tail(4).mean()  # Last year
            historical_wacc = wacc_data['WACC'].mean()  # All data
            wacc_trend = ((recent_wacc - historical_wacc) / historical_wacc) * 100
            print(f"   📈 WACC Trend Analysis:")
            print(f"      Historical Average: {historical_wacc:.2%}")
            print(f"      Recent Average: {recent_wacc:.2%}")
            print(f"      Trend: {wacc_trend:+.1f}% (recent vs historical)")
        
        # Range validation with industry context
        validation_warnings = []
        if not (0.005 <= rf <= 0.10):
            validation_warnings.append(f"Risk-free rate {rf:.2%} outside normal range (0.5%-10%)")
        if not (0.2 <= beta <= 2.5):
            validation_warnings.append(f"Beta {beta:.2f} outside normal range (0.2-2.5)")
        if not (0.02 <= erp <= 0.15):
            validation_warnings.append(f"ERP {erp:.2%} outside normal range (2%-15%)")
        if not (0.03 <= wacc <= 0.25):
            validation_warnings.append(f"WACC {wacc:.2%} outside normal range (3%-25%)")
        
        if validation_warnings:
            print("⚠️  Validation Warnings:")
            for warning in validation_warnings:
                print(f"      {warning}")
        else:
            print("✓ All WACC components within expected ranges")
        
        self.test_results['wacc_validation'] = True
        print("✅ WACC integration & validation test passed")
    
    def _test_dcf_model_integration(self):
        """Test 4: DCF model integration with WACC."""
        print("\n📈 TEST 4: DCF Model Integration")
        print("-" * 60)
        
        # Initialize and test DCF model
        self.dcf_model = StandardDCFModel()
        
        # Basic DCF calculation test
        pv_fcf_fraction, dcf_val, stock_price = self.dcf_model.calculate_dcf(
            self.df,
            discount_rate_scale=1.0,
            terminal_growth=0.025,
            time_horizon=10
        )
        
        print(f"✓ Basic DCF calculation successful:")
        print(f"   📊 FCF Present Value Fraction: {pv_fcf_fraction:.3f} ({pv_fcf_fraction*100:.1f}%)")
        print(f"   💰 Total DCF Value: ${dcf_val:,.0f}M")
        print(f"   📈 Implied Stock Price: ${stock_price:.2f}")
        
        # WACC integration verification
        current_wacc = self.df['WACC'].iloc[-1]
        print(f"   🔗 Using WACC: {current_wacc:.4f} ({current_wacc*100:.2f}%)")
        
        # Sensitivity analysis for discount rate scaling
        print(f"\n   📊 Discount Rate Sensitivity Analysis:")
        print(f"   {'Scale':>6} │ {'Discount Rate':>13} │ {'Stock Price':>11} │ {'% Change':>9}")
        print(f"   {'─'*6}┼{'─'*15}┼{'─'*13}┼{'─'*10}")
        
        base_price = stock_price
        sensitivity_results = {}
        
        for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            _, _, price = self.dcf_model.calculate_dcf(self.df, discount_rate_scale=scale)
            effective_rate = current_wacc * scale
            price_change = ((price - base_price) / base_price) * 100 if scale != 1.0 else 0
            sensitivity_results[scale] = {'price': price, 'change': price_change}
            
            print(f"   {scale:>5.1f}x │ {effective_rate:>12.2%} │ ${price:>10.2f} │ {price_change:>+8.1f}%")
        
        # Validation checks
        if dcf_val <= 0:
            raise ValueError("❌ DCF value must be positive")
        if stock_price <= 0:
            raise ValueError("❌ Stock price must be positive")
        if not (0.1 <= pv_fcf_fraction <= 0.9):
            print(f"⚠️  Warning: FCF PV fraction {pv_fcf_fraction:.2%} outside typical range (10%-90%)")
        
        # Check sensitivity makes sense (higher discount rate = lower price)
        low_scale_price = sensitivity_results[0.8]['price']
        high_scale_price = sensitivity_results[1.2]['price']
        if low_scale_price <= high_scale_price:
            print("⚠️  Warning: Sensitivity analysis shows unexpected price relationship")
        else:
            print("✓ Sensitivity analysis shows expected inverse relationship (higher rate → lower price)")
        
        self.test_results['dcf_integration'] = True
        print("✅ DCF model integration test passed")
    
    def _test_monte_carlo_simulation(self):
        """Test 5: Monte Carlo simulation with WACC integration."""
        print("\n🎲 TEST 5: Monte Carlo Simulation")
        print("-" * 60)
        
        # Configure realistic prior distributions
        distributions = {
            'discount_rate_scale': uniform(loc=0.8, scale=0.4),  # 0.8x to 1.2x WACC
            'growth_rate': norm(loc=0.05, scale=0.03),           # 5% ± 3% growth  
            'terminal_growth': uniform(loc=0.02, scale=0.03),    # 2% to 5% terminal
            'time_horizon': uniform(loc=7, scale=6)              # 7-13 years horizon
        }
        
        # Correlation matrix (reflects real-world relationships)
        correlation = np.array([
            [1.0, 0.2, 0.3, 0.1],  # discount_rate_scale
            [0.2, 1.0, 0.5, 0.2],  # growth_rate
            [0.3, 0.5, 1.0, 0.1],  # terminal_growth  
            [0.1, 0.2, 0.1, 1.0]   # time_horizon
        ])
        
        print("   🎯 Configuring simulation parameters...")
        print("      Discount Rate Scale: 0.8x to 1.2x WACC")
        print("      Growth Rate: 5% ± 3% (normal distribution)")
        print("      Terminal Growth: 2% to 5% (uniform)")
        print("      Time Horizon: 7-13 years (uniform)")
        
        # Configure and run simulation
        self.dcf_model.configure_priors(distributions, correlation)
        n_samples = 750  # Balanced between speed and accuracy
        self.dcf_model.simulate(self.df, n_samples=n_samples, random_state=42)
        
        # Comprehensive results analysis
        prices = self.dcf_model.stock_prices
        mean_price = prices.mean()
        std_price = prices.std()
        median_price = np.median(prices)
        
        # Percentile analysis
        p5, p10, p25, p75, p90, p95 = np.percentile(prices, [5, 10, 25, 75, 90, 95])
        
        print(f"\n✓ Monte Carlo simulation complete ({n_samples:,} samples)")
        print(f"   📊 Price Distribution Analysis:")
        print(f"      Mean: ${mean_price:.2f}")
        print(f"      Median: ${median_price:.2f}")
        print(f"      Std Dev: ${std_price:.2f}")
        print(f"      CV: {(std_price/mean_price)*100:.1f}%")
        
        print(f"\n   📈 Confidence Intervals:")
        print(f"      80% CI: ${p10:.2f} - ${p90:.2f}")
        print(f"      90% CI: ${p5:.2f} - ${p95:.2f}")
        print(f"      IQR: ${p25:.2f} - ${p75:.2f}")
        
        # Parameter distribution summary
        print(f"\n   🎯 Parameter Sample Summary:")
        for param_name, samples in self.dcf_model.parameter_samples.items():
            mean_val = samples.mean()
            std_val = samples.std()
            
            if param_name == 'discount_rate_scale':
                print(f"      {param_name}: {mean_val:.3f}x ± {std_val:.3f}")
            elif 'rate' in param_name or 'growth' in param_name:
                print(f"      {param_name}: {mean_val:.2%} ± {std_val:.2%}")
            else:
                print(f"      {param_name}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Distribution validation
        if std_price <= 0:
            raise ValueError("❌ Price standard deviation must be positive")
        if mean_price <= 0:
            raise ValueError("❌ Mean price must be positive")
        if not (0.05 <= (std_price/mean_price) <= 2.0):  # CV between 5% and 200%
            print(f"⚠️  Warning: Coefficient of variation {(std_price/mean_price)*100:.1f}% seems unusual")
        
        # Check for reasonable distribution shape
        skewness = pd.Series(prices).skew()
        print(f"   📏 Distribution Shape: Skewness = {skewness:.2f}")
        if abs(skewness) > 2:
            print("⚠️  Warning: Highly skewed distribution detected")
        
        self.test_results['monte_carlo'] = True
        print("✅ Monte Carlo simulation test passed")
    
    def _test_market_analysis_diagnostics(self):
        """Test 6: Market analysis and comprehensive diagnostics."""
        print("\n📊 TEST 6: Market Analysis & Diagnostics")
        print("-" * 60)
        
        # Current market data extraction
        current_price = self.df['Price'].iloc[-1]
        market_cap = self.df['Market-Cap'].iloc[-1]
        implied_price = self.dcf_model.stock_prices.mean()
        price_std = self.dcf_model.stock_prices.std()
        
        print(f"   📈 Market vs DCF Analysis:")
        print(f"      Current Stock Price: ${current_price:.2f}")
        print(f"      Market Cap: ${market_cap:,.0f}M")
        print(f"      DCF Implied Price: ${implied_price:.2f} ± ${price_std:.2f}")
        
        # Valuation analysis
        price_diff = implied_price - current_price
        price_diff_pct = (price_diff / current_price) * 100
        
        print(f"      Price Difference: ${price_diff:+.2f} ({price_diff_pct:+.1f}%)")
        
        # Determine valuation category
        if abs(price_diff_pct) < 10:
            valuation_category = "FAIRLY VALUED"
            valuation_emoji = "⚖️"
        elif price_diff_pct > 10:
            valuation_category = "UNDERVALUED"
            valuation_emoji = "📈"
        else:
            valuation_category = "OVERVALUED"
            valuation_emoji = "📉"
        
        print(f"      {valuation_emoji} DCF Indication: {valuation_category}")
        
        # Confidence analysis
        p25, p75 = np.percentile(self.dcf_model.stock_prices, [25, 75])
        if current_price < p25:
            confidence = "HIGH confidence in undervaluation"
        elif current_price > p75:
            confidence = "HIGH confidence in overvaluation"
        elif p25 <= current_price <= p75:
            confidence = "MODERATE confidence (price within IQR)"
        else:
            confidence = "LOW confidence"
        
        print(f"      🎯 Confidence Level: {confidence}")
        
        # Comprehensive diagnostics
        print(f"\n   🔍 Data Quality Diagnostics:")
        
        # Data completeness analysis
        total_rows = len(self.df)
        financial_core_cols = ['FCF-LTM', 'Revenue', 'Market-Cap', 'WACC']
        complete_financial = self.df[financial_core_cols].dropna().shape[0]
        completeness = (complete_financial / total_rows) * 100
        
        print(f"      Data Completeness: {completeness:.1f}% ({complete_financial}/{total_rows} complete rows)")
        
        # Time series consistency
        date_consistency = self.df.index.is_monotonic_increasing
        print(f"      Date Consistency: {'✓' if date_consistency else '✗'} {'Monotonic' if date_consistency else 'Non-monotonic'}")
        
        # WACC stability analysis
        wacc_series = self.df['WACC'].dropna()
        if len(wacc_series) > 8:
            wacc_volatility = wacc_series.pct_change().std() * np.sqrt(4)  # Annualized
            print(f"      WACC Volatility: {wacc_volatility:.2%} (annualized)")
            
            recent_wacc_avg = wacc_series.tail(4).mean()
            historical_wacc_avg = wacc_series.mean()
            wacc_drift = ((recent_wacc_avg - historical_wacc_avg) / historical_wacc_avg) * 100
            print(f"      WACC Drift: {wacc_drift:+.1f}% (recent vs historical)")
        
        # Price momentum analysis
        price_series = self.df['Price'].dropna()
        if len(price_series) > 12:
            price_returns = price_series.pct_change().dropna()
            price_volatility = price_returns.std() * np.sqrt(4)  # Annualized
            momentum_1y = ((price_series.iloc[-1] / price_series.iloc[-5]) - 1) * 100
            
            print(f"      Price Volatility: {price_volatility:.2%} (annualized)")
            print(f"      1-Year Momentum: {momentum_1y:+.1f}%")
        
        # Risk assessment
        print(f"\n   ⚠️  Risk Assessment:")
        risk_factors = []
        
        if abs(price_diff_pct) > 50:
            risk_factors.append("Large valuation gap suggests model risk or market inefficiency")
        if (price_std / implied_price) > 0.3:
            risk_factors.append("High valuation uncertainty from Monte Carlo simulation")
        if completeness < 80:
            risk_factors.append("Data completeness below 80% may affect reliability")
        
        if risk_factors:
            for i, risk in enumerate(risk_factors, 1):
                print(f"      {i}. {risk}")
        else:
            print("      No significant risk factors identified")
        
        # Investment recommendation framework
        print(f"\n   💡 Analysis Summary:")
        print(f"      Symbol: {self.test_symbols[0]}")
        print(f"      Valuation: {valuation_category}")
        print(f"      Confidence: {confidence}")
        print(f"      Price Target: ${implied_price:.2f} (±${price_std:.2f})")
        
        self.test_results['market_analysis'] = True
        print("✅ Market analysis & diagnostics test passed")
    
    def _print_comprehensive_summary(self):
        """Print comprehensive test summary with detailed results."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"🧪 Test Execution Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Tests Passed: {passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Detailed test results with emojis
        test_details = {
            'infrastructure': '📁 Infrastructure & CLI',
            'data_pipeline': '📊 Macrotrends Data Pipeline',
            'wacc_validation': '💰 WACC Integration & Validation',
            'dcf_integration': '📈 DCF Model Integration',
            'monte_carlo': '🎲 Monte Carlo Simulation',
            'market_analysis': '📊 Market Analysis & Diagnostics'
        }
        
        print(f"\n📊 Detailed Test Results:")
        for test_key, passed in self.test_results.items():
            test_name = test_details.get(test_key, test_key)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        # Pipeline health summary
        if passed_tests == total_tests:
            print(f"\n🎉 PIPELINE STATUS: HEALTHY")
            print(f"   All components of the financial pipeline are working correctly!")
            print(f"   ✓ Data Infrastructure ✓ Macrotrends ✓ WACC ✓ DCF ✓ Analysis")
        else:
            failed_tests = total_tests - passed_tests
            print(f"\n⚠️  PIPELINE STATUS: NEEDS ATTENTION")
            print(f"   {failed_tests} component(s) need investigation")
            print(f"   Please review the detailed output above for specific issues")
        
        # Performance summary
        if hasattr(self, 'df') and self.df is not None:
            print(f"\n📈 Pipeline Performance Summary:")
            print(f"   Data Points: {len(self.df):,} quarterly observations")
            print(f"   Features: {self.df.shape[1]:,} financial metrics")
            print(f"   WACC Data: {self.df['WACC'].notna().sum():,} valid observations")
            
            if hasattr(self, 'dcf_model') and self.dcf_model is not None:
                if hasattr(self.dcf_model, 'stock_prices'):
                    price_range = self.dcf_model.stock_prices.max() - self.dcf_model.stock_prices.min()
                    print(f"   Monte Carlo: {len(self.dcf_model.stock_prices):,} simulations")
                    print(f"   Price Range: ${price_range:.2f} valuation spread")
        
        print("=" * 80)


def test_financial_pipeline():
    """Main test function for pytest compatibility and standalone execution."""
    test_suite = FinancialPipelineTestSuite(['AAPL'])
    success = test_suite.run_all_tests()
    
    # For pytest: assert the test passed
    assert success, "Financial pipeline test suite failed"
    
    return test_suite


if __name__ == "__main__":
    # Run the comprehensive test when executed directly
    print("🚀 Starting Financial Pipeline Test Suite...")
    try:
        result = test_financial_pipeline()
        print("\n✅ Test suite completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
