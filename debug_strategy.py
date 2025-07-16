#!/usr/bin/env python3
"""
Debug script to understand Sharpe ratio calculations and Bollinger Bands strategy issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test Sharpe ratio calculation
def debug_sharpe_ratio():
    print("=== SHARPE RATIO CALCULATION DEBUG ===")
    
    # Example returns from the console log
    returns = [0.44, -0.10, -0.09, 0.04, 0.54, -0.15]  # 6 trades
    
    print(f"Trade returns: {returns}")
    print(f"Number of trades: {len(returns)}")
    
    # Standard Sharpe calculation
    risk_free_rate = 0.02
    excess_returns = np.array(returns) - risk_free_rate/252  # Daily risk-free rate
    
    print(f"Risk-free rate (annual): {risk_free_rate}")
    print(f"Risk-free rate (daily): {risk_free_rate/252:.6f}")
    print(f"Excess returns: {excess_returns}")
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    print(f"Mean excess return: {mean_excess:.6f}")
    print(f"Std of excess returns: {std_excess:.6f}")
    
    # Annual Sharpe ratio
    sharpe_ratio = mean_excess / std_excess * np.sqrt(252) if std_excess > 0 else 0
    print(f"Sharpe ratio (annualized): {sharpe_ratio:.2f}")
    
    # Problem: We're using trade returns, not daily returns!
    print("\n=== ISSUE IDENTIFIED ===")
    print("The current calculation uses individual trade returns instead of daily portfolio returns.")
    print("This inflates the Sharpe ratio because:")
    print("1. Trade returns are larger than daily returns")
    print("2. We're not accounting for holding periods")
    print("3. The annualization factor (√252) assumes daily returns")
    
    # Correct approach would be to calculate daily portfolio returns
    print("\n=== CORRECT APPROACH ===")
    print("Should calculate daily portfolio returns considering:")
    print("- Position sizes")
    print("- Holding periods")
    print("- Cash positions between trades")

def debug_bollinger_bands():
    print("\n=== BOLLINGER BANDS STRATEGY DEBUG ===")
    
    # Typical Bollinger Band conditions
    print("Common Bollinger Band strategy:")
    print("BUY: price < lower_band (oversold)")
    print("SELL: price > upper_band (overbought)")
    print()
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    df = pd.DataFrame({
        'close': prices,
        'date': dates
    })
    
    # Calculate Bollinger Bands
    window = 20
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_middle'] = rolling_mean
    
    # Check how often price touches bands
    buy_signals = df['close'] < df['bb_lower']
    sell_signals = df['close'] > df['bb_upper']
    
    print(f"Sample data points: {len(df)}")
    print(f"Buy signals (price < lower): {buy_signals.sum()}")
    print(f"Sell signals (price > upper): {sell_signals.sum()}")
    print(f"Buy signal frequency: {buy_signals.sum()/len(df)*100:.1f}%")
    print(f"Sell signal frequency: {sell_signals.sum()/len(df)*100:.1f}%")
    
    print("\n=== POTENTIAL ISSUES ===")
    print("1. Bollinger Band touches are rare (typically 5% of time)")
    print("2. Strategy might be too restrictive")
    print("3. Need sufficient data for accurate band calculation")
    print("4. Market conditions affect band width")
    
    # Show last few rows with bands
    print("\n=== SAMPLE DATA (last 10 rows) ===")
    print(df[['close', 'bb_lower', 'bb_middle', 'bb_upper']].tail(10).round(2))

if __name__ == "__main__":
    debug_sharpe_ratio()
    debug_bollinger_bands()