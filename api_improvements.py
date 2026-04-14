"""
API Endpoints for Improvements - Phase 28
New endpoints for performance tracking, adaptive thresholds, and metrics
"""

from fastapi import APIRouter
from performance_tracker import tracker
from improvements import (
    get_adaptive_thresholds,
    detect_market_condition,
    is_news_event_soon,
    calculate_position_size,
    get_volatility_risk_level,
    get_session_name,
    is_trading_allowed
)

router = APIRouter(prefix="/api", tags=["improvements"])


@router.get("/performance")
async def get_performance_metrics():
    """Get complete performance summary"""
    return tracker.get_summary()


@router.get("/performance/win-rate")
async def get_win_rate():
    """Get overall win rate"""
    return {
        "win_rate": tracker.get_win_rate(),
        "total_trades": len(tracker.trades)
    }


@router.get("/performance/by-session")
async def get_performance_by_session():
    """Get win rate by trading session"""
    return {
        "by_session": tracker.get_win_rate_by_session(),
        "best_session": tracker.get_best_session()
    }


@router.get("/performance/by-hour")
async def get_performance_by_hour():
    """Get win rate by hour of day"""
    return {
        "by_hour": tracker.get_win_rate_by_hour()
    }


@router.get("/performance/brain-accuracy")
async def get_brain_accuracy():
    """Get accuracy of each brain"""
    return {
        "brain_accuracy": tracker.get_brain_accuracy()
    }


@router.get("/performance/recent-trades")
async def get_recent_trades(limit: int = 10):
    """Get last N trades"""
    return {
        "recent_trades": tracker.get_recent_trades(limit)
    }


@router.get("/adaptive-thresholds")
async def get_adaptive_thresholds_endpoint(atr: float = 18.5, volatility_ma: float = 15.0):
    """
    Get adaptive thresholds based on market conditions.
    
    Args:
        atr: Current ATR (volatility)
        volatility_ma: Moving average of ATR
    
    Returns:
        Adaptive B1, B2, B3 thresholds
    """
    market_condition = detect_market_condition(atr, volatility_ma)
    win_rate = tracker.get_win_rate()
    
    thresholds = get_adaptive_thresholds(market_condition, win_rate)
    
    return {
        "thresholds": thresholds,
        "market_condition": market_condition,
        "current_win_rate": win_rate,
        "atr": atr,
        "volatility_ma": volatility_ma
    }


@router.get("/position-size")
async def get_position_size_endpoint(atr: float = 18.5):
    """
    Get recommended position size based on volatility.
    
    Args:
        atr: Current ATR
    
    Returns:
        Recommended lot size and risk level
    """
    lot_size = calculate_position_size(atr)
    risk_level = get_volatility_risk_level(atr)
    
    return {
        "atr": atr,
        "recommended_lot_size": lot_size,
        "risk_level": risk_level,
        "base_lot": 0.03
    }


@router.get("/economic-calendar")
async def get_economic_calendar():
    """Get upcoming economic events"""
    news_coming, event_name, minutes_until = is_news_event_soon()
    
    return {
        "high_impact_event_soon": news_coming,
        "event_name": event_name,
        "minutes_until": minutes_until,
        "trading_allowed": not news_coming,
        "recommendation": "SKIP TRADE" if news_coming else "PROCEED"
    }


@router.get("/session-status")
async def get_session_status():
    """Get current trading session status"""
    session = get_session_name()
    allowed = is_trading_allowed()
    
    return {
        "current_session": session,
        "trading_allowed": allowed,
        "session_quality": "HIGH" if session == "London/NY overlap" else "MEDIUM" if session in ["London", "New York"] else "LOW"
    }


@router.post("/log-trade")
async def log_trade(
    entry_time: str,
    entry_price: float,
    exit_time: str,
    exit_price: float,
    signal: str,
    b1_score: float,
    b2_score: float,
    b3_score: float,
    session: str = "UNKNOWN"
):
    """
    Log a completed trade.
    
    Args:
        entry_time: Entry time (ISO format)
        entry_price: Entry price
        exit_time: Exit time (ISO format)
        exit_price: Exit price
        signal: Trade signal (BUY/SELL)
        b1_score: B1 score
        b2_score: B2 score
        b3_score: B3 score
        session: Trading session
    
    Returns:
        Logged trade record
    """
    trade = tracker.log_trade(
        entry_time=entry_time,
        entry_price=entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        signal=signal,
        b1_score=b1_score,
        b2_score=b2_score,
        b3_score=b3_score,
        session=session
    )
    
    return {
        "status": "logged",
        "trade": trade,
        "current_win_rate": tracker.get_win_rate(),
        "total_trades": len(tracker.trades)
    }


@router.get("/health/improvements")
async def health_check_improvements():
    """Health check for improvements module"""
    return {
        "status": "ok",
        "improvements": [
            "performance_tracking",
            "adaptive_thresholds",
            "position_sizing",
            "economic_calendar",
            "session_management",
            "brain_accuracy_tracking"
        ],
        "performance_data_available": len(tracker.trades) > 0,
        "total_trades_logged": len(tracker.trades)
    }
