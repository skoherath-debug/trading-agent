"""
Backtesting Framework - Phase 28
Test trading system on historical data before live deployment
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json


class Backtester:
    """Backtest trading system on historical data"""
    
    def __init__(self, historical_data_csv):
        """
        Initialize backtester with historical data.
        
        Args:
            historical_data_csv: Path to CSV with columns:
                                timestamp, open, high, low, close, volume
        """
        self.df = pd.read_csv(historical_data_csv)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        self.trades = []
        self.balance = 1000  # Starting balance
        self.initial_balance = 1000
    
    def run_backtest(self, start_date, end_date, signal_generator_func):
        """
        Run backtest on historical data.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            signal_generator_func: Function that generates signals from price data
        
        Returns:
            Backtest results dictionary
        """
        df_period = self.df[(self.df['timestamp'] >= start_date) & 
                            (self.df['timestamp'] <= end_date)].copy()
        
        open_trade = None
        
        for idx, row in df_period.iterrows():
            # Get signal from historical data up to this point
            df_up_to_now = self.df[self.df['timestamp'] <= row['timestamp']]
            
            try:
                signal = signal_generator_func(df_up_to_now)
            except:
                signal = "WAIT"
            
            if signal == "BUY" and open_trade is None:
                # Open buy trade
                open_trade = {
                    "entry_time": row['timestamp'],
                    "entry_price": row['close'],
                    "signal": "BUY",
                    "entry_index": idx
                }
            elif signal == "SELL" and open_trade is not None:
                # Close trade
                profit = row['close'] - open_trade['entry_price']
                self.trades.append({
                    **open_trade,
                    "exit_time": row['timestamp'],
                    "exit_price": row['close'],
                    "profit": round(profit, 2),
                    "win": profit > 0,
                    "pips": round(profit / 0.01, 0)  # Approximate pips
                })
                self.balance += profit
                open_trade = None
        
        # Close any remaining open trade
        if open_trade is not None:
            last_price = df_period['close'].iloc[-1]
            profit = last_price - open_trade['entry_price']
            self.trades.append({
                **open_trade,
                "exit_time": df_period['timestamp'].iloc[-1],
                "exit_price": last_price,
                "profit": round(profit, 2),
                "win": profit > 0,
                "pips": round(profit / 0.01, 0)
            })
            self.balance += profit
        
        return self.get_results()
    
    def get_results(self):
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                "error": "No trades generated",
                "total_trades": 0,
                "win_rate": 0,
                "total_profit": 0
            }
        
        wins = sum(1 for t in self.trades if t['win'])
        total = len(self.trades)
        total_profit = sum(t['profit'] for t in self.trades)
        avg_profit = total_profit / total if total > 0 else 0
        
        # Calculate max drawdown
        cumulative_profit = 0
        peak_balance = self.initial_balance
        max_drawdown = 0
        
        for trade in self.trades:
            cumulative_profit += trade['profit']
            current_balance = self.initial_balance + cumulative_profit
            
            if current_balance > peak_balance:
                peak_balance = current_balance
            
            drawdown = peak_balance - current_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        if len(self.trades) > 1:
            profits = [t['profit'] for t in self.trades]
            std_dev = np.std(profits) if np.std(profits) > 0 else 1
            sharpe = (avg_profit / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        else:
            sharpe = 0
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round((wins / total * 100), 1) if total > 0 else 0,
            "total_profit": round(total_profit, 2),
            "avg_profit_per_trade": round(avg_profit, 2),
            "final_balance": round(self.balance, 2),
            "roi": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "best_trade": round(max([t['profit'] for t in self.trades]), 2) if self.trades else 0,
            "worst_trade": round(min([t['profit'] for t in self.trades]), 2) if self.trades else 0,
            "avg_win": round(np.mean([t['profit'] for t in self.trades if t['win']]), 2) if wins > 0 else 0,
            "avg_loss": round(np.mean([t['profit'] for t in self.trades if not t['win']]), 2) if (total - wins) > 0 else 0,
        }
    
    def get_recent_trades(self, limit=10):
        """Get last N trades"""
        return self.trades[-limit:]
    
    def export_results(self, filename="backtest_results.json"):
        """Export results to JSON file"""
        results = {
            "summary": self.get_results(),
            "trades": self.trades
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        return filename


class BacktestComparison:
    """Compare two trading systems"""
    
    def __init__(self, system_name_1, system_name_2):
        self.system_1 = {"name": system_name_1, "results": None}
        self.system_2 = {"name": system_name_2, "results": None}
    
    def set_results(self, system_num, results):
        """Set results for a system"""
        if system_num == 1:
            self.system_1["results"] = results
        else:
            self.system_2["results"] = results
    
    def get_comparison(self):
        """Compare the two systems"""
        if not self.system_1["results"] or not self.system_2["results"]:
            return {"error": "Both systems must have results"}
        
        r1 = self.system_1["results"]
        r2 = self.system_2["results"]
        
        comparison = {
            "system_1": self.system_1["name"],
            "system_2": self.system_2["name"],
            "metrics": {
                "win_rate": {
                    "system_1": r1.get("win_rate", 0),
                    "system_2": r2.get("win_rate", 0),
                    "winner": self.system_1["name"] if r1.get("win_rate", 0) > r2.get("win_rate", 0) else self.system_2["name"]
                },
                "total_profit": {
                    "system_1": r1.get("total_profit", 0),
                    "system_2": r2.get("total_profit", 0),
                    "winner": self.system_1["name"] if r1.get("total_profit", 0) > r2.get("total_profit", 0) else self.system_2["name"]
                },
                "roi": {
                    "system_1": r1.get("roi", 0),
                    "system_2": r2.get("roi", 0),
                    "winner": self.system_1["name"] if r1.get("roi", 0) > r2.get("roi", 0) else self.system_2["name"]
                },
                "sharpe_ratio": {
                    "system_1": r1.get("sharpe_ratio", 0),
                    "system_2": r2.get("sharpe_ratio", 0),
                    "winner": self.system_1["name"] if r1.get("sharpe_ratio", 0) > r2.get("sharpe_ratio", 0) else self.system_2["name"]
                },
                "max_drawdown": {
                    "system_1": r1.get("max_drawdown", 0),
                    "system_2": r2.get("max_drawdown", 0),
                    "winner": self.system_1["name"] if r1.get("max_drawdown", 0) < r2.get("max_drawdown", 0) else self.system_2["name"]
                }
            }
        }
        
        return comparison
