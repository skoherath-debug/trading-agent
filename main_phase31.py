"""
main_phase31.py — Phase 31 Week 1+
Real-time XAU/USD feed via Finnhub WebSocket.

Endpoints (all JSON, CORS enabled):
  /              — basic health
  /health        — detailed health (feed + mode + age)
  /snapshot      — full state dump (price, candles, history)
  /live-price    — dashboard-compatible live price
  /status        — dashboard-compatible status (for "Connecting..." badge)
  /chart-data    — M5 OHLC candles for the dashboard chart

SAFETY: MODE env var must be PAPER to start.
"""

import os
import sys
import time
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from data_feed_finnhub import FinnhubRealTimeFeed
from bootstrap_history  import bootstrap_all

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    stream = sys.stdout,
)
log = logging.getLogger("phase31")

MODE = os.getenv("MODE", "").upper()
if MODE != "PAPER":
    log.error(f"[SAFETY] MODE must be PAPER — got '{MODE}'. Refusing to start.")
    sys.exit(1)

log.info("=" * 60)
log.info("  PHASE 31 — REAL-TIME XAU/USD FEED")
log.info(f"  MODE: {MODE} (no live trading)")
log.info("=" * 60)

feed = FinnhubRealTimeFeed()


def on_m5_close(candle):
    log.info(
        f"[CANDLE M5] t={candle['time']} "
        f"O={candle['open']:.2f} H={candle['high']:.2f} "
        f"L={candle['low']:.2f} C={candle['close']:.2f} "
        f"ticks={candle.get('ticks', 0)}"
    )


def on_h1_close(candle):
    log.info(
        f"[CANDLE H1] t={candle['time']} "
        f"O={candle['open']:.2f} H={candle['high']:.2f} "
        f"L={candle['low']:.2f} C={candle['close']:.2f}"
    )


feed.on_m5_close.append(on_m5_close)
feed.on_h1_close.append(on_h1_close)


def health_monitor():
    while True:
        try:
            connected = feed.stream_connected
            age       = feed.get_data_age_ms()
            healthy   = feed.is_healthy()
            price     = feed.latest_mid
            m5_n      = len(feed.m5_history)

            age_str = f"{age}ms" if age is not None else "N/A"
            price_str = f"${price:.2f}" if price else "None"

            log.info(
                f"[HEALTH] connected={connected} healthy={healthy} "
                f"age={age_str} price={price_str} m5_history={m5_n}"
            )
        except Exception as e:
            log.error(f"[HEALTH] monitor error: {e}")
        time.sleep(30)


class APIHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path in ("/", "/health"):
            self._send(200, {
                "status":    "ok",
                "service":   "phase31",
                "mode":      MODE,
                "connected": feed.stream_connected,
                "healthy":   feed.is_healthy(),
                "age_ms":    feed.get_data_age_ms(),
                "price":     feed.latest_mid,
                "m5_count":  len(feed.m5_history),
                "m15_count": len(feed.m15_history),
                "h1_count":  len(feed.h1_history),
                "h4_count":  len(feed.h4_history),
                "timestamp": int(time.time()),
            })
            return

        if path == "/snapshot":
            self._send(200, feed.snapshot())
            return

        if path == "/live-price":
            price = feed.latest_mid
            if price is None:
                self._send(200, {
                    "price":   None,
                    "close":   None,
                    "symbol":  "XAU/USD",
                    "ts":      int(time.time()),
                    "healthy": False,
                    "error":   "no ticks yet",
                })
                return
            self._send(200, {
                "price":     round(price, 2),
                "close":     round(price, 2),
                "symbol":    "XAU/USD",
                "ts":        int(time.time()),
                "age_ms":    feed.get_data_age_ms(),
                "healthy":   feed.is_healthy(),
                "source":    "finnhub_ws",
            })
            return

        if path == "/status":
            price = feed.latest_mid
            price_rounded = round(price, 2) if price else None
            data_ok = feed.is_healthy()
            data_msg = (
                f"Finnhub WS · age {feed.get_data_age_ms()}ms · "
                f"{len(feed.m5_history)} M5 bars"
                if price
                else "Waiting for Finnhub ticks"
            )

            self._send(200, {
                "signal":     "WAIT",
                "confidence": 0,
                "price":      price_rounded,
                "health": {
                    "data_feed":   {"ok": data_ok, "msg": data_msg},
                    "brain1":      {"ok": False,  "msg": "not wired yet"},
                    "brain2":      {"ok": False,  "msg": "not wired yet"},
                    "brain3":      {"ok": False,  "msg": "not wired yet"},
                    "h4":          {"ok": False,  "msg": "not wired yet"},
                    "telegram":    {"ok": True,   "msg": "standby"},
                    "daily_limit": {"ok": True,   "msg": "OK — paper mode"},
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                           time.gmtime()),
                "session":         "LIVE (paper)",
                "session_quality": "LIVE" if data_ok else "N/A",
                "session_trade_ok": False,
                "session_icon":    "🟢" if data_ok else "🟡",
                "last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                          time.gmtime()),
                "improvements": "PHASE_31",
                "mode": MODE,
            })
            return

        if path == "/chart-data":
            qs = parse_qs(urlparse(self.path).query)
            tf = qs.get("tf", ["m5"])[0].lower()

            tf_map = {
                "m5":  feed.m5_history,
                "m15": feed.m15_history,
                "h1":  feed.h1_history,
                "h4":  feed.h4_history,
            }
            if tf not in tf_map:
                self._send(400, {"error": f"unknown tf: {tf}"})
                return

            bars = [
                {
                    "time":   c["time"],
                    "open":   round(c["open"],  2),
                    "high":   round(c["high"],  2),
                    "low":    round(c["low"],   2),
                    "close":  round(c["close"], 2),
                    "volume": c.get("ticks", 0),
                }
                for c in list(tf_map[tf])
            ]

            live_attr = getattr(feed, f"live_{tf}", None)
            if live_attr:
                bars.append({
                    "time":   live_attr["time"],
                    "open":   round(live_attr["open"],  2),
                    "high":   round(live_attr["high"],  2),
                    "low":    round(live_attr["low"],   2),
                    "close":  round(live_attr["close"], 2),
                    "volume": live_attr.get("ticks", 0),
                    "live":   True,
                })

            self._send(200, {
                "symbol": "XAU/USD",
                "tf":     tf,
                "bars":   bars,
                "count":  len(bars),
                "source": "finnhub_ws_live",
            })
            return

        self._send(404, {"error": "not found", "path": path})


def run_http():
    port = int(os.getenv("PORT", "8080"))
    srv = HTTPServer(("0.0.0.0", port), APIHandler)
    log.info(f"[HTTP] Listening on 0.0.0.0:{port}")
    log.info("[HTTP] Endpoints: / /health /snapshot /live-price /status /chart-data?tf=m5")
    srv.serve_forever()


def main():
    bootstrap_all(feed)
    feed.start()
    log.info("[MAIN] Finnhub feed started")
    threading.Thread(target=health_monitor, daemon=True, name="health").start()
    log.info("[MAIN] Health monitor started")
    run_http()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("[MAIN] Shutting down...")
        feed.stop()
