"""
Performance Tracking Module - Phase 28
Tracks all trades and calculates performance metrics
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class PerformanceTracker:
    """Track trading performance, win rates, and brain accuracy"""
    
    def __init__(self, filename="performance_log.json"):
        self.filename = filename
        self.trades = self._load_trades()
    
    def _load_trades(self):
        """Load existing trades from file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def log_trade(self, entry_time, entry_price, exit_time, exit_price, signal, 
                  b1_score, b2_score, b3_score, session="UNKNOWN"):
        """Record a completed trade"""
        profit_loss = (exit_price - entry_price) if signal == "BUY" else (entry_price - exit_price)
        
        trade = {
            "entry_time": entry_time,
            "entry_price": round(entry_price, 2),
            "exit_time": exit_time,
            "exit_price": round(exit_price, 2),
            "signal": signal,
            "b1_score": round(b1_score, 3),
            "b2_score": round(b2_score, 1),
            "b3_score": round(b3_score, 1),
            "profit_loss": round(profit_loss, 2),
            "win": profit_loss > 0,
            "session": session,
            "hour": datetime.fromisoformat(entry_time).hour if isinstance(entry_time, str) else entry_time.hour,
            "logged_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.trades.append(trade)
        self._save()
        return trade
    
    def get_win_rate(self):
        """Calculate overall win rate"""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["win"])
        return round((wins / len(self.trades)) * 100, 1)
    
    def get_win_rate_by_session(self):
        """Win rate for each trading session"""
        sessions = {}
        for trade in self.trades:
            session = trade.get("session", "UNKNOWN")
            if session not in sessions:
                sessions[session] = {"wins": 0, "total": 0}
            sessions[session]["total"] += 1
            if trade["win"]:
                sessions[session]["wins"] += 1
        
        return {s: round((v["wins"]/v["total"]*100), 1) if v["total"] > 0 else 0 
                for s, v in sessions.items()}
    
    def get_win_rate_by_hour(self):
        """Win rate by hour of day"""
        hours = {}
        for trade in self.trades:
            hour = trade.get("hour", 0)
            if hour not in hours:
                hours[hour] = {"wins": 0, "total": 0}
            hours[hour]["total"] += 1
            if trade["win"]:
                hours[hour]["wins"] += 1
        
        return {h: round((v["wins"]/v["total"]*100), 1) if v["total"] > 0 else 0 
                for h, v in sorted(hours.items())}
    
    def get_brain_accuracy(self):
        """Which brain is most accurate?"""
        if not self.trades:
            return {"B1": 0, "B2": 0, "B3": 0}
        
        b1_correct = sum(1 for t in self.trades if (t["b1_score"] >= 0.6) == t["win"])
        b2_correct = sum(1 for t in self.trades if (t["b2_score"] >= 5.0) == t["win"])
        b3_correct = sum(1 for t in self.trades if (t["b3_score"] >= 6.5) == t["win"])
        
        total = len(self.trades)
        return {
            "B1": round((b1_correct / total * 100), 1) if total > 0 else 0,
            "B2": round((b2_correct / total * 100), 1) if total > 0 else 0,
            "B3": round((b3_correct / total * 100), 1) if total > 0 else 0,
        }
    
    def get_average_profit_per_trade(self):
        """Average profit/loss per trade"""
        if not self.trades:
            return 0.0
        total_pnl = sum(t["profit_loss"] for t in self.trades)
        return round(total_pnl / len(self.trades), 2)
    
    def get_total_profit(self):
        """Total profit/loss"""
        if not self.trades:
            return 0.0
        return round(sum(t["profit_loss"] for t in self.trades), 2)
    
    def get_best_session(self):
        """Session with highest win rate"""
        by_session = self.get_win_rate_by_session()
        if not by_session:
            return "N/A"
        return max(by_session, key=by_session.get)
    
    def get_recent_trades(self, limit=10):
        """Get last N trades"""
        return self.trades[-limit:]
    
    def get_summary(self):
        """Get complete performance summary"""
        return {
            "total_trades": len(self.trades),
            "win_rate": self.get_win_rate(),
            "total_profit": self.get_total_profit(),
            "avg_profit_per_trade": self.get_average_profit_per_trade(),
            "win_rate_by_session": self.get_win_rate_by_session(),
            "win_rate_by_hour": self.get_win_rate_by_hour(),
            "brain_accuracy": self.get_brain_accuracy(),
            "best_session": self.get_best_session(),
            "recent_trades": self.get_recent_trades(5)
        }
    
    def _save(self):
        """Save trades to file"""
        with open(self.filename, 'w') as f:
            json.dump(self.trades, f, indent=2)


# Global tracker instance
tracker = PerformanceTracker()
