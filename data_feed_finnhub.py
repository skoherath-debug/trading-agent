"""
data_feed_finnhub.py — Phase 31 Week 1
Real-time XAU/USD price feed via Finnhub WebSocket.

- Symbol: OANDA:XAU_USD
- Builds live M5, M15, H1, H4 candles from tick stream
- Thread-safe (WebSocket runs in background thread, asyncio main reads)
- Auto-reconnects with exponential backoff
- Tracks data age for health monitoring
"""

import os
import json
import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone

import websocket  # pip install websocket-client

log = logging.getLogger(__name__)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
SYMBOL      = "OANDA:XAU_USD"
WS_URL      = f"wss://ws.finnhub.io?token={FINNHUB_KEY}"


class FinnhubRealTimeFeed:
    def __init__(self):
        # Latest price
        self.latest_mid   = None
        self.last_tick_ts = 0.0      # seconds epoch

        # Live building candle per timeframe
        self.live_m5  = None
        self.live_m15 = None
        self.live_h1  = None
        self.live_h4  = None

        # Closed candle history (ring buffers)
        self.m5_history  = deque(maxlen=500)
        self.m15_history = deque(maxlen=300)
        self.h1_history  = deque(maxlen=200)
        self.h4_history  = deque(maxlen=100)

        # Thread safety
        self._lock = threading.Lock()

        # Connection state
        self.ws               = None
        self.stream_connected = False
        self._retries         = 0
        self._stop            = False

        # Callbacks (fired on candle close)
        self.on_m5_close  = []
        self.on_m15_close = []
        self.on_h1_close  = []
        self.on_h4_close  = []

    # ────────────────────────────────────────────────────────────
    # WebSocket callbacks
    # ────────────────────────────────────────────────────────────
    def _on_open(self, ws):
        self.stream_connected = True
        self._retries = 0
        log.info(f"[FINNHUB] Connected, subscribing to {SYMBOL}")
        ws.send(json.dumps({"type": "subscribe", "symbol": SYMBOL}))

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
        except Exception:
            return

        mtype = msg.get("type")
        if mtype == "ping":
            return  # keep-alive from server
        if mtype != "trade":
            return

        for t in msg.get("data", []):
            price = float(t.get("p", 0))
            ts_ms = float(t.get("t", 0))
            if price <= 0 or ts_ms <= 0:
                continue
            ts_sec = ts_ms / 1000.0
            self._process_tick(price, ts_sec)

    def _on_error(self, ws, error):
        log.warning(f"[FINNHUB] WS error: {error}")

    def _on_close(self, ws, code, msg):
        self.stream_connected = False
        log.warning(f"[FINNHUB] WS closed (code={code}, msg={msg})")

    # ────────────────────────────────────────────────────────────
    # Tick + candle builder
    # ────────────────────────────────────────────────────────────
    def _process_tick(self, price: float, ts_sec: float):
        with self._lock:
            self.latest_mid   = price
            self.last_tick_ts = ts_sec

            self._update_candle("m5",  300,  ts_sec, price)
            self._update_candle("m15", 900,  ts_sec, price)
            self._update_candle("h1",  3600, ts_sec, price)
            self._update_candle("h4",  14400, ts_sec, price)

    def _update_candle(self, tf: str, seconds: int, ts_sec: float, price: float):
        bucket_start = int(ts_sec // seconds) * seconds
        live_attr    = f"live_{tf}"
        hist_attr    = f"{tf}_history"
        cb_attr      = f"on_{tf}_close"

        live = getattr(self, live_attr)

        if live is None:
            setattr(self, live_attr, {
                "time":  bucket_start,
                "open":  price,
                "high":  price,
                "low":   price,
                "close": price,
                "ticks": 1,
            })
            return

        if bucket_start > live["time"]:
            # Close the prior candle
            getattr(self, hist_attr).append(dict(live))
            closed = dict(live)
            # Fire callbacks outside lock (do it after we re-open)
            setattr(self, live_attr, {
                "time":  bucket_start,
                "open":  price,
                "high":  price,
                "low":   price,
                "close": price,
                "ticks": 1,
            })
            # Fire callbacks after state updated
            for cb in getattr(self, cb_attr):
                try:
                    cb(closed)
                except Exception as e:
                    log.error(f"[FINNHUB] Callback error on {tf} close: {e}")
        else:
            # Update live candle
            live["high"]  = max(live["high"], price)
            live["low"]   = min(live["low"],  price)
            live["close"] = price
            live["ticks"] = live.get("ticks", 0) + 1

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────
    def seed_history(self, tf: str, candles: list):
        """Load historical closed candles (from bootstrap_history.py)."""
        attr = f"{tf}_history"
        if not hasattr(self, attr):
            log.error(f"[FINNHUB] Unknown tf for seed: {tf}")
            return
        buf = getattr(self, attr)
        with self._lock:
            buf.clear()
            for c in candles:
                buf.append(c)
        log.info(f"[FINNHUB] Seeded {len(candles)} {tf.upper()} candles")

    def get_data_age_ms(self):
        if self.last_tick_ts == 0:
            return None
        return int((time.time() - self.last_tick_ts) * 1000)

    def is_healthy(self, max_age_ms: int = 5000) -> bool:
        if not self.stream_connected:
            return False
        age = self.get_data_age_ms()
        if age is None:
            return False  # never received tick
        return age < max_age_ms

    def snapshot(self):
        with self._lock:
            return {
                "price":        self.latest_mid,
                "data_age_ms":  self.get_data_age_ms(),
                "healthy":      self.is_healthy(),
                "connected":    self.stream_connected,
                "live_m5":      dict(self.live_m5)  if self.live_m5  else None,
                "live_m15":     dict(self.live_m15) if self.live_m15 else None,
                "live_h1":      dict(self.live_h1)  if self.live_h1  else None,
                "live_h4":      dict(self.live_h4)  if self.live_h4  else None,
                "m5_closed":    list(self.m5_history),
                "m15_closed":   list(self.m15_history),
                "h1_closed":    list(self.h1_history),
                "h4_closed":    list(self.h4_history),
            }

    # ────────────────────────────────────────────────────────────
    # Run loop (blocking — start on a thread)
    # ────────────────────────────────────────────────────────────
    def run_forever(self):
        if not FINNHUB_KEY:
            log.error("[FINNHUB] FINNHUB_API_KEY env var not set — feed disabled")
            return

        while not self._stop:
            try:
                log.info(f"[FINNHUB] Connecting (retry #{self._retries})...")
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                log.error(f"[FINNHUB] run_forever exception: {e}")

            self.stream_connected = False
            self._retries += 1
            delay = min(60, 2 ** min(self._retries, 6))
            log.info(f"[FINNHUB] Reconnecting in {delay}s...")
            time.sleep(delay)

    def start(self):
        """Start WebSocket in background daemon thread."""
        t = threading.Thread(target=self.run_forever, daemon=True, name="finnhub-ws")
        t.start()
        return t

    def stop(self):
        self._stop = True
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
