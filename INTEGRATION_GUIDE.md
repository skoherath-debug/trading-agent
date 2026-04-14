# Integration Guide - Adding Improvements to main.py

## Overview

This guide shows exactly where and how to integrate the improvements into your existing `main.py`.

## Step 1: Add Imports at the Top

**Location:** After the existing imports (around line 12)

**Add these lines:**

```python
# ── Phase 28: Import Improvements ─────────────────────────────
from api_improvements import router as improvements_router
from performance_tracker import tracker
from improvements import (
    get_adaptive_thresholds,
    detect_market_condition,
    is_news_event_soon,
    calculate_position_size,
    get_volatility_risk_level,
    apply_trailing_stop,
    get_session_name,
    is_trading_allowed,
    calculate_confluence_score,
    prepare_enhanced_features
)
```

**Full context (lines 1-22):**

```python
"""
XAU/USD Triple Brain Agent - Railway Deployment
Phase 28 - Integrated Improvements: Performance tracking, adaptive thresholds, volatility filters
"""
import os, threading, time, traceback
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Phase 28: Import Improvements ─────────────────────────────
from api_improvements import router as improvements_router
from performance_tracker import tracker
from improvements import (
    get_adaptive_thresholds,
    detect_market_condition,
    is_news_event_soon,
    calculate_position_size,
    get_volatility_risk_level,
    apply_trailing_stop,
    get_session_name,
    is_trading_allowed,
    calculate_confluence_score,
    prepare_enhanced_features
)

# ── App ───────────────────────────────────────────────────────
api = FastAPI(title="XAU/USD Triple Brain Agent v4.1 - Phase 28")
```

## Step 2: Include Improvements Router

**Location:** After the CORS middleware setup (around line 21)

**Add this line:**

```python
# Include improvements endpoints
api.include_router(improvements_router)
```

**Full context (lines 14-25):**

```python
# ── App ───────────────────────────────────────────────────────
api = FastAPI(title="XAU/USD Triple Brain Agent v4.1 - Phase 28")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include improvements endpoints
api.include_router(improvements_router)

# ── Environment variables ─────────────────────────────────────
```

## Step 3: Add Variables for Tracking

**Location:** After environment variables (around line 32)

**Add these lines:**

```python
# ── Phase 28: Improvement Tracking Variables ──────────────────
_last_signal = None
_last_signal_time = None
_signal_count_today = 0
_atr_history = []  # Track ATR for volatility_ma calculation
_volatility_ma = 15.0  # Moving average of ATR
```

## Step 4: Modify Signal Generation Logic

**Location:** In the `run_analysis()` function, after calculating B1, B2, B3 scores

**Find this section (around line 250-300):**

```python
# Current code (BEFORE):
if b1 >= 0.60 and b2 >= 5 and conf >= 6.5:
    sig = "BUY"
elif b1 >= 0.60 and b2 >= 5 and conf >= 6.5:
    sig = "SELL"
else:
    sig = "WAIT"
```

**Replace with (AFTER):**

```python
# ── Phase 28: Adaptive Thresholds ─────────────────────────────
# Calculate market condition and get adaptive thresholds
atr_val = m.get("volatility", {}).get("atr14", 15.0)
_atr_history.append(atr_val)
if len(_atr_history) > 20:
    _atr_history.pop(0)
global _volatility_ma
_volatility_ma = np.mean(_atr_history) if _atr_history else 15.0

market_condition = detect_market_condition(atr_val, _volatility_ma)
recent_win_rate = tracker.get_win_rate()
adaptive_thresholds = get_adaptive_thresholds(market_condition, recent_win_rate)

# ── Phase 28: Economic Calendar Check ─────────────────────────
news_coming, event_name, minutes_until = is_news_event_soon()
if news_coming:
    print(f"   ⚠️  HIGH-IMPACT NEWS: {event_name} in {minutes_until:.0f} min - SKIP TRADE")
    sig = "WAIT"
    reason = f"Economic event: {event_name}"
else:
    # ── Phase 28: Use Adaptive Thresholds ─────────────────────
    if b1 >= adaptive_thresholds["b1"] and b2 >= adaptive_thresholds["b2"] and conf >= adaptive_thresholds["b3"]:
        sig = "BUY"
    elif b1 >= adaptive_thresholds["b1"] and b2 >= adaptive_thresholds["b2"] and conf >= adaptive_thresholds["b3"]:
        sig = "SELL"
    else:
        sig = "WAIT"
        reason = f"Thresholds not met (B1:{b1:.3f}/{adaptive_thresholds['b1']}, B2:{b2:.1f}/{adaptive_thresholds['b2']}, B3:{conf:.1f}/{adaptive_thresholds['b3']})"

# ── Phase 28: Position Sizing ─────────────────────────────────
recommended_lot = calculate_position_size(atr_val)
risk_level = get_volatility_risk_level(atr_val)
```

## Step 5: Add Trade Logging

**Location:** After a trade is executed (around line 320-340)

**Add this code:**

```python
# ── Phase 28: Log Trade for Performance Tracking ───────────────
if sig in ["BUY", "SELL"]:
    try:
        session_name = get_session_name()
        tracker.log_trade(
            entry_time=now.isoformat(),
            entry_price=price,
            exit_time=now.isoformat(),  # Will be updated when trade closes
            exit_price=price,
            signal=sig,
            b1_score=b1,
            b2_score=b2,
            b3_score=conf,
            session=session_name
        )
        print(f"   ✅ Trade logged: {sig} at ${price} (Session: {session_name})")
    except Exception as e:
        print(f"   ⚠️  Failed to log trade: {e}")
```

## Step 6: Add Status Endpoint

**Location:** Add a new endpoint after existing endpoints (around line 350+)

**Add this code:**

```python
# ── Phase 28: New Status Endpoint ─────────────────────────────
@api.get("/status/enhanced")
async def get_enhanced_status():
    """Get enhanced status with improvements data"""
    return {
        "status": "running",
        "improvements": {
            "performance_tracking": True,
            "adaptive_thresholds": True,
            "volatility_filters": True,
            "economic_calendar": True,
            "position_sizing": True
        },
        "performance": tracker.get_summary(),
        "market_condition": detect_market_condition(_volatility_ma, _volatility_ma),
        "news_event_soon": is_news_event_soon()[0]
    }
```

## Step 7: Update Telegram Messages

**Location:** In the Telegram message sending section (around line 355-365)

**Update the message to include new info:**

```python
# Current (BEFORE):
tg(
    icon + " <b>" + em + " - XAU/USD</b>\n\n"
    "Confidence: <b>" + sf(conf, 1) + "/10</b>\n"
    "Price: <b>$" + sf(price) + "</b>\n"
    "B1:" + sf(b1, 3) + " B2:" + sf(b2, 1) + "/10 B3:" + sf(conf, 1) + "/10"
)

# Updated (AFTER):
market_cond = detect_market_condition(_volatility_ma, _volatility_ma)
lot_size = calculate_position_size(_volatility_ma)

tg(
    icon + " <b>" + em + " - XAU/USD</b>\n\n"
    "Confidence: <b>" + sf(conf, 1) + "/10</b>\n"
    "Price: <b>$" + sf(price) + "</b>\n"
    "B1:" + sf(b1, 3) + " B2:" + sf(b2, 1) + "/10 B3:" + sf(conf, 1) + "/10\n\n"
    "<b>Phase 28 Enhancements:</b>\n"
    "Market: " + market_cond + "\n"
    "Position Size: " + sf(lot_size, 2) + " lots\n"
    "Win Rate: " + sf(tracker.get_win_rate(), 1) + "%\n"
    "Session: " + get_session_name()
)
```

## Complete Integration Checklist

- [ ] Add imports at top of main.py
- [ ] Include improvements router
- [ ] Add tracking variables
- [ ] Modify signal generation logic
- [ ] Add trade logging
- [ ] Add status endpoint
- [ ] Update Telegram messages
- [ ] Test on demo account
- [ ] Commit to GitHub
- [ ] Verify Railway deployment

## Testing the Integration

### 1. Check if improvements are loaded
```bash
curl https://your-railway-app.up.railway.app/api/health/improvements
```

### 2. Get performance metrics
```bash
curl https://your-railway-app.up.railway.app/api/performance
```

### 3. Get adaptive thresholds
```bash
curl "https://your-railway-app.up.railway.app/api/adaptive-thresholds?atr=18.5&volatility_ma=15.0"
```

### 4. Check economic calendar
```bash
curl https://your-railway-app.up.railway.app/api/economic-calendar
```

## Troubleshooting

### Import Error: "No module named 'improvements'"
- Make sure `improvements.py` is in the same directory as `main.py`
- Verify file was pushed to GitHub and Railway pulled it

### Performance data not saving
- Check that `performance_log.json` is writable
- Verify `tracker.log_trade()` is being called

### Adaptive thresholds not changing
- Ensure ATR values are being calculated correctly
- Check that `tracker.get_win_rate()` has enough trades

## Next Steps

1. **Make these changes to main.py**
2. **Commit and push to GitHub**
3. **Railway will auto-deploy**
4. **Test the new endpoints**
5. **Proceed to XGBoost retraining**

---

**Phase 28 Integration Complete** ✅
