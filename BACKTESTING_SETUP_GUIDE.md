# Backtesting Setup Guide - Phase 28

## Overview

This guide shows how to set up and run backtests to validate your improvements before live trading.

## What is Backtesting?

Backtesting runs your trading system on historical data to see:
- How many trades it would have generated
- What the win rate would have been
- Total profit/loss
- Maximum drawdown
- Sharpe ratio (risk-adjusted returns)

## Step 1: Get Historical Data

### Option A: Download from Yahoo Finance (Easiest)

**In Python (local or Colab):**

```python
import yfinance as yf
import pandas as pd

# Download 6 months of XAU/USD data
print("📥 Downloading XAU/USD historical data...")
data = yf.download("GC=F", start="2023-10-01", end="2024-04-14", interval="5m")

# Save to CSV
data.to_csv("xau_usd_historical_data.csv")
print(f"✅ Downloaded {len(data)} candles")
print(f"   Columns: {data.columns.tolist()}")
print(f"   Date range: {data.index[0]} to {data.index[-1]}")
```

**Output file:** `xau_usd_historical_data.csv`

### Option B: Use Your Broker's Data

If you have data from Exness or another broker:
1. Export to CSV format
2. Ensure columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
3. Save as `xau_usd_historical_data.csv`

### Option C: Use Free Data API

```python
import requests
import pandas as pd

# Alternative: Use free data source
url = "https://api.example.com/historical-data?symbol=XAUUSD&period=5m"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df.to_csv("xau_usd_historical_data.csv", index=False)
```

## Step 2: Prepare Data for Backtesting

**Create a Python script:** `prepare_backtest_data.py`

```python
import pandas as pd
import numpy as np

# Load historical data
df = pd.read_csv("xau_usd_historical_data.csv")

# Ensure correct column names
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp
df = df.sort_values('timestamp')

# Remove duplicates
df = df.drop_duplicates(subset=['timestamp'])

# Fill missing values
df = df.fillna(method='ffill')

# Remove rows with NaN
df = df.dropna()

print(f"✅ Data prepared:")
print(f"   Total candles: {len(df)}")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Columns: {df.columns.tolist()}")

# Save cleaned data
df.to_csv("xau_usd_historical_data_clean.csv", index=False)
print(f"✅ Saved: xau_usd_historical_data_clean.csv")
```

**Run it:**
```bash
python prepare_backtest_data.py
```

## Step 3: Create a Simple Signal Generator

**Create:** `simple_signal_generator.py`

```python
import pandas as pd
import numpy as np

def simple_signal_generator(df):
    """
    Simple signal generator for backtesting.
    This is a placeholder - replace with your actual signal logic.
    """
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # Calculate EMA
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    
    # Simple signal logic
    if len(df) < 30:
        return "WAIT"
    
    rsi_val = rsi.iloc[-1]
    ema_fast_val = ema_fast.iloc[-1]
    ema_slow_val = ema_slow.iloc[-1]
    
    # BUY: RSI > 50 AND EMA fast > EMA slow
    if rsi_val > 50 and ema_fast_val > ema_slow_val:
        return "BUY"
    
    # SELL: RSI < 50 AND EMA fast < EMA slow
    elif rsi_val < 50 and ema_fast_val < ema_slow_val:
        return "SELL"
    
    else:
        return "WAIT"
```

## Step 4: Run Backtest

**Create:** `run_backtest.py`

```python
import pandas as pd
from backtest import Backtester
from simple_signal_generator import simple_signal_generator

# Load historical data
print("📥 Loading historical data...")
df = pd.read_csv("xau_usd_historical_data_clean.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create backtester
print("🔄 Running backtest...")
backtester = Backtester("xau_usd_historical_data_clean.csv")

# Run backtest
results = backtester.run_backtest(
    start_date="2024-01-01",
    end_date="2024-04-14",
    signal_generator_func=simple_signal_generator
)

# Print results
print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)
print(f"Total Trades:        {results['total_trades']}")
print(f"Wins:                {results['wins']}")
print(f"Losses:              {results['losses']}")
print(f"Win Rate:            {results['win_rate']:.1f}%")
print(f"Total Profit:        ${results['total_profit']:.2f}")
print(f"Average Profit:      ${results['avg_profit_per_trade']:.2f}")
print(f"Final Balance:       ${results['final_balance']:.2f}")
print(f"ROI:                 {results['roi']:.1f}%")
print(f"Max Drawdown:        ${results['max_drawdown']:.2f}")
print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
print(f"Best Trade:          ${results['best_trade']:.2f}")
print(f"Worst Trade:         ${results['worst_trade']:.2f}")
print(f"Avg Win:             ${results['avg_win']:.2f}")
print(f"Avg Loss:            ${results['avg_loss']:.2f}")
print("="*50)

# Export results
backtester.export_results("backtest_results.json")
print("\n✅ Results saved to: backtest_results.json")
```

**Run it:**
```bash
python run_backtest.py
```

## Step 5: Compare Old vs New System

**Create:** `compare_systems.py`

```python
import pandas as pd
from backtest import Backtester, BacktestComparison
from simple_signal_generator import simple_signal_generator

# Load data
df = pd.read_csv("xau_usd_historical_data_clean.csv")

# Test OLD system
print("🔄 Testing OLD system (fixed thresholds)...")
backtester_old = Backtester("xau_usd_historical_data_clean.csv")
results_old = backtester_old.run_backtest(
    start_date="2024-01-01",
    end_date="2024-04-14",
    signal_generator_func=simple_signal_generator
)

# Test NEW system (with improvements)
print("🔄 Testing NEW system (adaptive thresholds)...")
backtester_new = Backtester("xau_usd_historical_data_clean.csv")
results_new = backtester_new.run_backtest(
    start_date="2024-01-01",
    end_date="2024-04-14",
    signal_generator_func=simple_signal_generator  # Replace with improved version
)

# Compare
comparison = BacktestComparison("Old System", "New System")
comparison.set_results(1, results_old)
comparison.set_results(2, results_new)

comp_results = comparison.get_comparison()

print("\n" + "="*60)
print("SYSTEM COMPARISON")
print("="*60)

for metric, data in comp_results["metrics"].items():
    print(f"\n{metric.upper()}:")
    print(f"  Old System: {data['system_1']}")
    print(f"  New System: {data['system_2']}")
    print(f"  Winner: {data['winner']}")

print("\n" + "="*60)
```

**Run it:**
```bash
python compare_systems.py
```

## Step 6: Analyze Results

### Key Metrics to Watch

| Metric | Good Value | Excellent Value |
|--------|-----------|-----------------|
| Win Rate | > 50% | > 60% |
| ROI | > 10% | > 30% |
| Sharpe Ratio | > 1.0 | > 2.0 |
| Max Drawdown | < 20% | < 10% |
| Profit Factor | > 1.5 | > 2.0 |

### Example Results

```
OLD SYSTEM (Before Improvements):
  Win Rate: 45.2%
  Total Profit: $125.50
  ROI: 12.5%
  Sharpe Ratio: 0.8

NEW SYSTEM (After Improvements):
  Win Rate: 58.7%
  Total Profit: $287.25
  ROI: 28.7%
  Sharpe Ratio: 1.4

IMPROVEMENT: +13.5% win rate, +$161.75 profit, +16.2% ROI
```

## Step 7: Optimize Based on Results

If results are not good, try:

1. **Adjust thresholds**
   - Increase B1 threshold from 0.600 to 0.650
   - Increase B2 threshold from 5.0 to 5.5
   - Increase B3 threshold from 6.5 to 7.0

2. **Add filters**
   - Skip trades during low-liquidity hours
   - Skip trades during high volatility
   - Skip trades near economic events

3. **Improve indicators**
   - Add more technical indicators
   - Use multi-timeframe analysis
   - Add volume confirmation

4. **Retrain ML model**
   - Add more features
   - Use more training data
   - Adjust hyperparameters

## Step 8: Validate on Demo Account

Once backtest looks good:

1. **Connect to Exness demo account**
2. **Run live signals for 1 week**
3. **Compare demo results with backtest**
4. **If similar, deploy to live trading**

## Complete Backtesting Workflow

```
1. Get Historical Data
   ↓
2. Prepare Data
   ↓
3. Create Signal Generator
   ↓
4. Run Backtest (Old System)
   ↓
5. Run Backtest (New System)
   ↓
6. Compare Results
   ↓
7. Optimize if Needed
   ↓
8. Test on Demo Account
   ↓
9. Deploy to Live Trading
```

## Files You'll Need

```
xau_usd_historical_data.csv          # Raw historical data
xau_usd_historical_data_clean.csv    # Cleaned data
prepare_backtest_data.py             # Data preparation script
simple_signal_generator.py           # Signal generation logic
run_backtest.py                      # Backtesting script
compare_systems.py                   # Comparison script
backtest_results.json                # Results output
```

## Quick Start (Copy-Paste)

```bash
# 1. Download data
python -c "import yfinance as yf; yf.download('GC=F', start='2023-10-01', end='2024-04-14', interval='5m').to_csv('xau_usd_historical_data.csv')"

# 2. Prepare data
python prepare_backtest_data.py

# 3. Run backtest
python run_backtest.py

# 4. Compare systems
python compare_systems.py
```

## Troubleshooting

### Error: "No module named 'yfinance'"
```bash
pip install yfinance
```

### Error: "CSV file not found"
- Make sure you're in the correct directory
- Check file name spelling

### Backtest shows 0 trades
- Signal generator is too strict
- Adjust thresholds in `simple_signal_generator.py`
- Add debug prints to see what signals are generated

### Results look too good to be true
- Check for look-ahead bias (using future data)
- Verify signal generation logic
- Test on different time periods

## Next Steps

1. **Download historical data**
2. **Prepare the data**
3. **Create signal generator**
4. **Run backtest**
5. **Compare old vs new system**
6. **Optimize if needed**
7. **Test on demo account**
8. **Deploy to live trading**

---

**Phase 28 Backtesting Setup** ✅
