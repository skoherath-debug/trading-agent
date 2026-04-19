"""
bootstrap_history.py — Phase 31 Week 1
Loads historical candles from Twelve Data (3-key rotation) on startup.

Symbol: XAU/USD
Loads:  300 M5, 200 M15, 100 H1, 50 H4
Used by main_phase31.py to seed the Finnhub feed's history buffers
before the WebSocket has built up enough closed candles.
"""

import os
import logging
import time
import requests

log = logging.getLogger(__name__)

TD_KEYS = [
    os.getenv("TD_KEY",  ""),
    os.getenv("TD_KEY2", ""),
    os.getenv("TD_KEY3", ""),
]
TD_KEYS = [k for k in TD_KEYS if k]

SYMBOL = "XAU/USD"

TF_CONFIG = {
    "m5":  {"interval": "5min",  "outputsize": 300},
    "m15": {"interval": "15min", "outputsize": 200},
    "h1":  {"interval": "1h",    "outputsize": 100},
    "h4":  {"interval": "4h",    "outputsize": 50},
}


def _fetch_td(interval: str, outputsize: int):
    """Try each TD key until one works."""
    last_err = None
    for i, key in enumerate(TD_KEYS):
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol":     SYMBOL,
                "interval":   interval,
                "outputsize": outputsize,
                "apikey":     key,
                "format":     "JSON",
            }
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                msg = data.get("message", "unknown")
                log.warning(f"[BOOTSTRAP] TD key #{i+1} error: {msg}")
                last_err = msg
                continue
            values = data.get("values", [])
            if not values:
                last_err = "empty values"
                continue
            return values
        except Exception as e:
            last_err = str(e)
            log.warning(f"[BOOTSTRAP] TD key #{i+1} exception: {e}")
            time.sleep(1)
    log.error(f"[BOOTSTRAP] All TD keys failed: {last_err}")
    return []


def _to_candles(values):
    """Convert TD response to [{time, open, high, low, close, volume}]."""
    candles = []
    # TD returns newest first — we want oldest first
    for v in reversed(values):
        try:
            dt = v["datetime"]
            # TD gives "YYYY-MM-DD HH:MM:SS" (UTC)
            ts = int(time.mktime(time.strptime(dt, "%Y-%m-%d %H:%M:%S")))
            candles.append({
                "time":   ts,
                "open":   float(v["open"]),
                "high":   float(v["high"]),
                "low":    float(v["low"]),
                "close":  float(v["close"]),
                "volume": float(v.get("volume", 0) or 0),
            })
        except Exception as e:
            log.warning(f"[BOOTSTRAP] skip row: {e}")
    return candles


def bootstrap_all(feed):
    """Load all TFs into the feed. Called once on startup."""
    if not TD_KEYS:
        log.error("[BOOTSTRAP] No TD_KEY env vars set — cannot seed history")
        return False

    log.info("[BOOTSTRAP] Loading historical candles from Twelve Data...")
    ok = True
    for tf, cfg in TF_CONFIG.items():
        log.info(f"[BOOTSTRAP] Fetching {cfg['outputsize']} {tf.upper()} candles ({cfg['interval']})...")
        values = _fetch_td(cfg["interval"], cfg["outputsize"])
        if not values:
            log.error(f"[BOOTSTRAP] Failed to load {tf.upper()}")
            ok = False
            continue
        candles = _to_candles(values)
        feed.seed_history(tf, candles)
        # Be polite to the free tier
        time.sleep(1)

    log.info(f"[BOOTSTRAP] Complete — status={'OK' if ok else 'PARTIAL'}")
    return ok
