"""
XAU/USD Triple Brain Agent — Railway Deployment
Phase 27 — Fixed: CORS, last_result, telegram, syntax errors
"""
import os, json, threading, time
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── FastAPI init ──────────────────────────────────────────────
api = FastAPI(title="XAU/USD Triple Brain Agent v4")

# ── CORS — allow dashboard at ceylonpropertylink.com ─────────
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config from Railway env vars ──────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "")
RUN_INTERVAL   = int(os.environ.get("RUN_INTERVAL", "3600"))
PKL_ID         = os.environ.get("MODEL_PKL_ID",  "")
JSON_ID        = os.environ.get("MODEL_JSON_ID", "")
TPSL_ID        = os.environ.get("TPSL_JSON_ID",  "")
TD_KEY         = "2c3dff7091284f92b2361649006448a8"

MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Download models from Google Drive ────────────────────────
def download_file(file_id, dest):
    if os.path.exists(dest):
        print(f"✅ Exists: {os.path.basename(dest)}")
        return True
    if not file_id or "PASTE" in file_id:
        print(f"⚠️  No file ID for {os.path.basename(dest)}")
        return False
    try:
        print(f"⬇️  Downloading {os.path.basename(dest)}...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = req.Session()
        r = session.get(url, stream=True, timeout=60)
        for key, value in r.cookies.items():
            if key.startswith("download_warning"):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                r = session.get(url, stream=True, timeout=60)
                break
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        print(f"✅ Downloaded: {os.path.basename(dest)} ({os.path.getsize(dest):,} bytes)")
        return True
    except Exception as e:
        print(f"❌ Download failed {os.path.basename(dest)}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False

download_file(PKL_ID,  f"{MODELS_DIR}/brain1_xgboost_v4.pkl")
download_file(JSON_ID, f"{MODELS_DIR}/brain1_features_v4.json")
download_file(TPSL_ID, f"{MODELS_DIR}/dynamic_tpsl_config.json")

# ── Load trading_agent.py ─────────────────────────────────────
_agent_loaded = False
try:
    code = open("/app/trading_agent.py").read()
    code = code.replace(
        '"/content/drive/MyDrive/trading_agent/',
        f'"{MODELS_DIR}/'
    )
    exec(code, globals())
    _agent_loaded = True
    print("✅ trading_agent.py loaded")
except Exception as e:
    print(f"❌ Failed to load trading_agent.py: {e}")

# ── numpy/pandas type cleaner for JSON ───────────────────────
def clean(obj):
    if isinstance(obj, dict):          return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):          return [clean(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, float) and (obj != obj):  return None   # NaN → null
    return obj

# ── Telegram sender ───────────────────────────────────────────
def tg(msg: str):
    """Send Telegram message — synchronous, fire and forget."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("⚠️  Telegram not configured")
        return False
    try:
        r = req.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        print(f"⚠️  Telegram error: {e}")
        return False

# ── Shared state — updated by scheduler AND /run-analysis ────
last_result: dict = {}
last_run_time: str = ""
_prev_health: dict = {}

def _save_result(result: dict):
    """Store result globally so /status can serve it instantly."""
    global last_result, last_run_time
    last_result   = result
    last_run_time = datetime.now(timezone.utc).isoformat()

def _check_health(result: dict):
    """Alert Telegram when a health component changes state."""
    global _prev_health
    health = result.get("health", {})
    if not health:
        return
    for key, status in health.items():
        was_ok = _prev_health.get(key, {}).get("ok", True)
        is_ok  = status.get("ok", True)
        msg    = status.get("msg", "")
        if was_ok and not is_ok:
            tg(f"⚠️ FAULT DETECTED: {key}\n{msg}\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        elif not was_ok and is_ok:
            tg(f"✅ RECOVERED: {key}\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    _prev_health = {k: dict(v) for k, v in health.items()}

# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@api.get("/")
def root():
    return {
        "agent":    "XAU/USD Triple Brain v4",
        "status":   "running",
        "time":     datetime.now(timezone.utc).isoformat(),
        "endpoints": ["/status", "/run-analysis", "/chart-data",
                      "/telegram/send", "/files", "/health"]
    }

@api.get("/health")
def health_check():
    return {
        "status":        "ok",
        "agent_loaded":  _agent_loaded,
        "last_run":      last_run_time or "never",
        "last_signal":   last_result.get("signal", "none"),
        "time":          datetime.now(timezone.utc).isoformat()
    }

@api.get("/status")
def get_status():
    """
    Returns the LAST cached analysis result instantly (no API calls).
    Dashboard polls this every 60s — much faster than /run-analysis.
    """
    if not last_result:
        return {
            "signal":     "WAIT",
            "confidence": 0,
            "price":      0,
            "msg":        "No analysis run yet — scheduler fires every "+str(RUN_INTERVAL//60)+" min",
            "last_run":   last_run_time or "never",
            "health": {
                "data_feed":   {"ok": _agent_loaded, "msg": "agent loaded" if _agent_loaded else "agent failed to load"},
                "brain1":      {"ok": False, "msg": "waiting for first run"},
                "brain2":      {"ok": False, "msg": "waiting for first run"},
                "brain3":      {"ok": False, "msg": "waiting for first run"},
                "h4":          {"ok": False, "msg": "waiting"},
                "telegram":    {"ok": bool(TELEGRAM_TOKEN), "msg": "configured" if TELEGRAM_TOKEN else "no token"},
                "daily_limit": {"ok": True,  "msg": "ok"}
            }
        }
    result = dict(last_result)
    result["last_run"] = last_run_time
    return JSONResponse(content=clean(result))

@api.post("/run-analysis")
def run_analysis_endpoint():
    """
    Triggers a fresh analysis. Takes 20-35s.
    Also saves result to last_result so /status serves it next time.
    """
    if not _agent_loaded:
        return JSONResponse(content={
            "signal": "ERROR",
            "error":  "trading_agent.py not loaded — check Railway logs",
            "health": {"data_feed": {"ok": False, "msg": "agent not loaded"}}
        })
    try:
        result = run_analysis()   # from trading_agent.py
        _save_result(result)
        _check_health(result)
        return JSONResponse(content=clean(result))
    except Exception as e:
        err = {
            "signal": "ERROR",
            "error":  str(e),
            "health": {"data_feed": {"ok": False, "msg": str(e)[:200]}}
        }
        return JSONResponse(content=err, status_code=500)

@api.get("/chart-data")
def chart_data():
    """Returns 200 x 15min candles from Twelve Data for the dashboard chart."""
    try:
        r = req.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol":     "XAU/USD",
                "interval":   "15min",
                "outputsize": 200,
                "apikey":     TD_KEY,
                "timezone":   "UTC",
                "format":     "JSON"
            },
            timeout=30
        )
        data = r.json()
        if "values" not in data:
            raise ValueError(data.get("message", "No data from Twelve Data"))
        bars = []
        for v in reversed(data["values"]):
            try:
                bars.append({
                    "time":  int(pd.Timestamp(v["datetime"]).timestamp()),
                    "open":  round(float(v["open"]),  2),
                    "high":  round(float(v["high"]),  2),
                    "low":   round(float(v["low"]),   2),
                    "close": round(float(v["close"]), 2),
                })
            except Exception:
                pass
        return JSONResponse(content={"bars": bars, "symbol": "XAU/USD", "tf": "15m"})
    except Exception as e:
        return JSONResponse(content={"error": str(e), "bars": []}, status_code=500)

@api.post("/telegram/send")
def telegram_send(body: dict):
    """
    Send a Telegram message via the bot.
    Called by dashboard SEND TEST button.
    """
    msg = body.get("message", "Test from dashboard")
    ok  = tg(msg)
    if ok:
        return {"ok": True, "msg": "Sent"}
    else:
        return JSONResponse(
            content={"ok": False, "msg": "Failed — check TELEGRAM_TOKEN and TELEGRAM_CHAT env vars on Railway"},
            status_code=500
        )

@api.get("/files")
def check_files():
    """Debug — check if model files are present."""
    files = {}
    for fname in ["brain1_xgboost_v4.pkl", "brain1_features_v4.json", "dynamic_tpsl_config.json"]:
        path = f"{MODELS_DIR}/{fname}"
        files[fname] = {
            "exists": os.path.exists(path),
            "size_kb": round(os.path.getsize(path) / 1024, 1) if os.path.exists(path) else 0
        }
    return {
        "files": files,
        "env": {
            "MODEL_PKL_ID":  (PKL_ID[:8]  + "...") if PKL_ID  else "NOT SET",
            "MODEL_JSON_ID": (JSON_ID[:8] + "...") if JSON_ID else "NOT SET",
            "TPSL_JSON_ID":  (TPSL_ID[:8] + "...") if TPSL_ID else "NOT SET",
            "TELEGRAM":      "SET" if TELEGRAM_TOKEN else "NOT SET",
        }
    }

@api.get("/debug-data")
def debug_data():
    """Debug — test Twelve Data API connectivity."""
    result = {}
    try:
        r = req.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": "XAU/USD", "interval": "1h",
                "outputsize": 5, "apikey": TD_KEY,
                "timezone": "UTC", "format": "JSON"
            },
            timeout=30
        )
        data = r.json()
        result["twelvedata"] = {
            "ok":     "values" in data,
            "rows":   len(data.get("values", [])),
            "msg":    data.get("message", "OK") if "values" not in data else "OK",
            "sample": data.get("values", [{}])[0] if "values" in data else {}
        }
    except Exception as e:
        result["twelvedata"] = {"ok": False, "msg": str(e)}
    return result

# ─────────────────────────────────────────────────────────────
# SCHEDULER — runs run_analysis() every RUN_INTERVAL seconds
# ─────────────────────────────────────────────────────────────
def scheduler():
    time.sleep(15)   # wait for server to fully start
    tg(
        f"🚀 <b>AGENT STARTED</b> (Railway 24/7)\n"
        f"Schedule: every {RUN_INTERVAL//60} min\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
        f"Dashboard: ceylonpropertylink.com/agent/dashboard.html"
    )
    while True:
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        print(f"[SCHEDULER] Running analysis at {now}")
        try:
            result = run_analysis()           # from trading_agent.py
            _save_result(result)              # ← saves to last_result for /status
            _check_health(result)

            sig   = result.get("signal",     "?")
            conf  = result.get("confidence",  0)
            price = result.get("price",       0)
            b1    = result.get("brain1", {}).get("probability", 0)
            b2    = result.get("brain2", {}).get("score", 0)
            print(f"[SCHEDULER] {sig} | conf={conf}/10 | B1={b1:.3f} | B2={b2:.1f} | ${price:.2f}")

            # Send Telegram signal alert
            if sig in ("BUY", "SELL"):
                tp = result.get("tp", 0)
                sl = result.get("sl", 0)
                tg(
                    f"{'📈' if sig=='BUY' else '📉'} <b>{sig} — XAU/USD</b>\n\n"
                    f"🎯 Confidence: <b>{conf}/10</b>\n"
                    f"💰 Price: <b>${price:.2f}</b>\n"
                    f"✅ TP: ${tp:.2f}\n"
                    f"❌ SL: ${sl:.2f}\n\n"
                    f"B1: {b1:.3f} | B2: {b2:.1f}/10\n"
                    f"🕐 {now}\n\n"
                    f"🤖 XAU/USD Triple Brain v4"
                )

        except Exception as e:
            print(f"[SCHEDULER] ERROR: {e}")
            tg(f"❌ <b>run_analysis failed</b>\n{str(e)[:300]}\n{now}")

        time.sleep(RUN_INTERVAL)

# Start scheduler in background thread
threading.Thread(target=scheduler, daemon=True).start()
print("✅ Scheduler started — interval:", RUN_INTERVAL, "sec")
