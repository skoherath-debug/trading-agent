"""
main_phase31.py — Phase 31 Week 1 Entry Point

What it does:
  1. Starts Finnhub WebSocket feed (background thread)
  2. Bootstraps historical candles from Twelve Data
  3. Logs candle closes (Brains wiring happens Week 2)
  4. Health monitor logs every 30s
  5. Exposes a tiny HTTP /health + /snapshot endpoint on $PORT
     (so Railway knows the service is alive)

SAFETY: MODE env var must be PAPER. Will refuse to start otherwise.
"""

import os
import sys
import time
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from data_feed_finnhub import FinnhubRealTimeFeed
from bootstrap_history  import bootstrap_all

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    stream = sys.stdout,
)
log = logging.getLogger("phase31")

# ──────────────────────────────────────────────────────────────
# Safety check
# ──────────────────────────────────────────────────────────────
MODE = os.getenv("MODE", "").upper()
if MODE != "PAPER":
    log.error(f"[SAFETY] MODE must be PAPER — got '{MODE}'. Refusing to start.")
    sys.exit(1)

log.info("=" * 60)
log.info("  PHASE 31 WEEK 1 — REAL-TIME XAU/USD FEED")
log.info("  MODE: PAPER (no live trading)")
log.info("=" * 60)

# ──────────────────────────────────────────────────────────────
# Feed + callbacks
# ──────────────────────────────────────────────────────────────
feed = FinnhubRealTimeFeed()


def on_m5_close(candle):
    log.info(
        f"[CANDLE M5] t={candle['time']} "
        f"O={candle['open']:.2f} H={candle['high']:.2f} "
        f"L={candle['low']:.2f} C={candle['close']:.2f} "
        f"ticks={candle.get('ticks', 0)}"
    )
    # TODO Week 2: run Brain 1/2/3 + AI Judge here


def on_h1_close(candle):
    log.info(
        f"[CANDLE H1] t={candle['time']} "
        f"O={candle['open']:.2f} H={candle['high']:.2f} "
        f"L={candle['low']:.2f} C={candle['close']:.2f}"
    )


feed.on_m5_close.append(on_m5_close)
feed.on_h1_close.append(on_h1_close)


# ──────────────────────────────────────────────────────────────
# Health monitor (runs every 30s on its own thread)
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# Tiny HTTP server for Railway healthcheck + remote inspection
# ──────────────────────────────────────────────────────────────
class StatusHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self._send(200, {
                "status":    "ok",
                "service":   "phase31-week1",
                "mode":      MODE,
                "connected": feed.stream_connected,
                "healthy":   feed.is_healthy(),
                "age_ms":    feed.get_data_age_ms(),
                "price":     feed.latest_mid,
            })
        elif self.path == "/snapshot":
            self._send(200, feed.snapshot())
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, format, *args):
        return  # silence default access logs


def run_http():
    port = int(os.getenv("PORT", "8080"))
    srv = HTTPServer(("0.0.0.0", port), StatusHandler)
    log.info(f"[HTTP] Listening on 0.0.0.0:{port}")
    srv.serve_forever()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # 1. Seed history from Twelve Data
    bootstrap_all(feed)

    # 2. Start Finnhub WebSocket (background thread)
    feed.start()
    log.info("[MAIN] Finnhub feed started")

    # 3. Start health monitor (background thread)
    threading.Thread(target=health_monitor, daemon=True, name="health").start()
    log.info("[MAIN] Health monitor started")

    # 4. Start HTTP server (blocks main thread)
    run_http()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("[MAIN] Shutting down...")
        feed.stop()
