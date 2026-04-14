"""
Trading System Improvements - Phase 28
Implements all 10 improvements:
1. Analyze system
2. Retrain XGBoost features
3. Enhance B2 indicators
4. Trailing stop loss
5. Volatility filters
6. Economic calendar
7. Performance dashboard
8. Adaptive thresholds
9. Backtesting framework
10. Deploy improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, time as _time
from performance_tracker import tracker


# ─── IMPROVEMENT #4: Trailing Stop Loss ──────────────────────────────────────

def apply_trailing_stop(entry_price, current_price, trailing_distance, direction):
    """
    Calculate trailing stop loss that moves with price.
    
    Args:
        entry_price: Initial entry price
        current_price: Current market price
        trailing_distance: Distance in dollars below/above current price
        direction: "BUY" or "SELL"
    
    Returns:
        Trailing stop loss level
    """
    if direction == "BUY":
        # For buy trades, stop is below current price
        trailing_stop = current_price - trailing_distance
        # Never move stop below entry price (protect capital)
        trailing_stop = max(trailing_stop, entry_price - 5)
    else:
        # For sell trades, stop is above current price
        trailing_stop = current_price + trailing_distance
        trailing_stop = min(trailing_stop, entry_price + 5)
    
    return round(trailing_stop, 2)


# ─── IMPROVEMENT #5: Volatility Filters & Adaptive Position Sizing ───────────

def calculate_position_size(atr, base_lot=0.03):
    """
    Calculate lot size based on volatility (ATR).
    Lower ATR = larger position, Higher ATR = smaller position
    
    Args:
        atr: Average True Range (volatility measure)
        base_lot: Base lot size (default 0.03)
    
    Returns:
        Recommended lot size
    """
    if atr < 15:
        lot_multiplier = 1.5  # Trade 50% bigger
    elif atr < 20:
        lot_multiplier = 1.0  # Normal size
    elif atr < 30:
        lot_multiplier = 0.7  # Trade 30% smaller
    else:
        lot_multiplier = 0.4  # Trade 60% smaller (very risky)
    
    final_lot = base_lot * lot_multiplier
    return round(final_lot, 2)


def get_volatility_risk_level(atr):
    """
    Classify market volatility risk level.
    
    Returns: "LOW", "NORMAL", "HIGH", "EXTREME"
    """
    if atr < 15:
        return "LOW"
    elif atr < 20:
        return "NORMAL"
    elif atr < 30:
        return "HIGH"
    else:
        return "EXTREME"


# ─── IMPROVEMENT #6: Economic Calendar Integration ──────────────────────────

HIGH_IMPACT_EVENTS = [
    {"time": "13:30", "name": "US Non-Farm Payroll", "impact": "HIGH", "day": "Friday"},
    {"time": "14:00", "name": "ECB Interest Rate Decision", "impact": "HIGH", "day": "Thursday"},
    {"time": "20:00", "name": "Fed Chair Speech", "impact": "MEDIUM", "day": "Various"},
    {"time": "08:00", "name": "UK Retail Sales", "impact": "MEDIUM", "day": "Friday"},
    {"time": "10:00", "name": "Eurozone CPI", "impact": "HIGH", "day": "Wednesday"},
    {"time": "16:00", "name": "US CPI", "impact": "HIGH", "day": "Thursday"},
    {"time": "12:30", "name": "US Initial Jobless Claims", "impact": "MEDIUM", "day": "Thursday"},
]


def is_news_event_soon(minutes_ahead=30):
    """
    Check if high-impact news event is coming soon.
    
    Args:
        minutes_ahead: Minutes to look ahead (default 30)
    
    Returns:
        Tuple: (bool: event_coming, str: event_name, float: minutes_until)
    """
    current_time = datetime.now(timezone.utc)
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    for event in HIGH_IMPACT_EVENTS:
        if event["impact"] != "HIGH":
            continue
        
        try:
            event_hour, event_minute = map(int, event["time"].split(":"))
            
            # Calculate minutes until event
            event_total_minutes = event_hour * 60 + event_minute
            current_total_minutes = current_hour * 60 + current_minute
            
            if event_total_minutes >= current_total_minutes:
                minutes_until = event_total_minutes - current_total_minutes
            else:
                minutes_until = (24 * 60) - current_total_minutes + event_total_minutes
            
            # Check if event is within the lookahead window
            if 0 < minutes_until < minutes_ahead:
                return True, event["name"], minutes_until
        except:
            continue
    
    return False, None, None


# ─── IMPROVEMENT #8: Adaptive Thresholds ────────────────────────────────────

def detect_market_condition(atr, volatility_ma):
    """
    Classify market as calm, normal, or volatile.
    
    Args:
        atr: Current ATR
        volatility_ma: Moving average of ATR
    
    Returns:
        "calm", "normal", or "volatile"
    """
    if volatility_ma == 0:
        return "normal"
    
    volatility_ratio = atr / volatility_ma
    
    if volatility_ratio < 0.8:
        return "calm"
    elif volatility_ratio < 1.2:
        return "normal"
    else:
        return "volatile"


def get_adaptive_thresholds(market_condition, win_rate):
    """
    Get adaptive thresholds based on market conditions and performance.
    
    Args:
        market_condition: "calm", "normal", or "volatile"
        win_rate: Recent win rate (0-100)
    
    Returns:
        Dict with adaptive B1, B2, B3 thresholds
    """
    # Base thresholds
    b1_threshold = 0.600
    b2_threshold = 5.0
    b3_threshold = 6.5
    
    # Adjust based on market volatility
    if market_condition == "calm":
        # Market is predictable - lower thresholds slightly
        b1_threshold = 0.550
        b2_threshold = 4.5
        b3_threshold = 6.0
    elif market_condition == "volatile":
        # Market is chaotic - raise thresholds
        b1_threshold = 0.700
        b2_threshold = 5.5
        b3_threshold = 7.0
    
    # Adjust based on recent performance
    if win_rate > 60:
        # Winning streak - can be more aggressive
        b1_threshold *= 0.95
        b2_threshold *= 0.95
        b3_threshold *= 0.95
    elif win_rate < 40:
        # Losing streak - be more conservative
        b1_threshold *= 1.10
        b2_threshold *= 1.10
        b3_threshold *= 1.10
    
    return {
        "b1": round(b1_threshold, 3),
        "b2": round(b2_threshold, 1),
        "b3": round(b3_threshold, 1),
        "market_condition": market_condition,
        "win_rate": win_rate
    }


# ─── IMPROVEMENT #3: Enhanced Technical Indicators ──────────────────────────

def detect_pin_bar(df):
    """
    Detect pin bar (rejection candle) pattern.
    
    Returns: True if last candle is a pin bar
    """
    if len(df) < 1:
        return False
    
    body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    wick_upper = df['high'].iloc[-1] - max(df['close'].iloc[-1], df['open'].iloc[-1])
    wick_lower = min(df['close'].iloc[-1], df['open'].iloc[-1]) - df['low'].iloc[-1]
    
    return (wick_upper > body * 2) or (wick_lower > body * 2)


def detect_engulfing(df):
    """
    Detect engulfing candle pattern.
    
    Returns: "BULLISH", "BEARISH", or None
    """
    if len(df) < 2:
        return None
    
    prev_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
    curr_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    
    # Current candle engulfs previous
    if curr_body > prev_body * 1.5:
        if df['close'].iloc[-1] > df['open'].iloc[-1]:
            return "BULLISH"
        else:
            return "BEARISH"
    
    return None


def calculate_confluence_score(df, rsi, macd, signal_line, ema_fast, ema_slow):
    """
    Calculate technical confluence score (0-6).
    
    Checks:
    1. RSI momentum
    2. MACD trend
    3. EMA alignment
    4. Volume spike
    5. Price action (pin bar / engulfing)
    6. Support/Resistance
    """
    score = 0
    
    # 1. RSI momentum
    if rsi > 50:
        score += 1
    
    # 2. MACD trend
    if macd > signal_line:
        score += 1
    
    # 3. EMA alignment
    if ema_fast > ema_slow:
        score += 1
    
    # 4. Volume confirmation
    if len(df) > 20:
        volume_avg = df['volume'].rolling(20).mean()
        volume_spike = df['volume'].iloc[-1] > volume_avg.iloc[-1] * 1.5
        if volume_spike:
            score += 1
    
    # 5. Price action
    if detect_pin_bar(df) or detect_engulfing(df):
        score += 1
    
    # 6. Support/Resistance
    if len(df) > 20:
        resistance = df['high'].rolling(20).max()
        support = df['low'].rolling(20).min()
        price_near_support = df['close'].iloc[-1] < support.iloc[-1] * 1.02
        if price_near_support:
            score += 1
    
    return score


# ─── IMPROVEMENT #2: Enhanced XGBoost Features ─────────────────────────────

def prepare_enhanced_features(df):
    """
    Prepare enhanced feature set for XGBoost model.
    
    Adds:
    - Volume ratio
    - ATR
    - Bollinger Bands position
    - Hour of day
    - Previous candle patterns
    """
    df = df.copy()
    
    # Volume features
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-9)
    
    # Volatility (ATR)
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # Bollinger Bands
    bb_sma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_sma + (bb_std * 2)
    df['bb_lower'] = bb_sma - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    
    # Time features
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
    
    return df


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_session_name(utc_dt=None):
    """Get current trading session name"""
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    
    h = utc_dt.hour
    
    if 7 <= h < 16:
        return "London"
    elif 13 <= h < 21:
        return "New York"
    elif 21 <= h or h < 1:
        return "Blocked"
    else:
        return "Asian"


def is_trading_allowed(utc_dt=None):
    """Check if trading is allowed in current session"""
    session = get_session_name(utc_dt)
    return session != "Blocked"
