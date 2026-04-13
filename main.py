"""
XAU/USD Triple Brain Agent — Railway Deployment
"""
import os, sys, json, pickle, threading, time, traceback
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Download models on startup ────────────────────────────────────────────────
MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_file(file_id, dest):
    """Download from Google Drive using direct download URL."""
    if os.path.exists(dest):
        print(f"✅ Already exists: {dest}")
        return True
    if not file_id or file_id == "PASTE_PKL_ID_HERE" or file_id == "PASTE_JSON_ID_HERE" or file_id == "PASTE_TPSL_ID_HERE":
        print(f"⚠️  No file ID set for {dest}")
        return False
    try:
        print(f"Downloading {os.path.basename(dest)}...")
        # Try direct download first
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = req.Session()
        r = session.get(url, stream=True, timeout=60)
        # Handle Google's virus scan warning for large files
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
        size = os.path.getsize(dest)
        print(f"✅ Downloaded: {os.path.basename(dest)} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(dest)}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False

# Download all model files
PKL_ID  = os.environ.get("MODEL_PKL_ID",  "")
JSON_ID = os.environ.get("MODEL_JSON_ID", "")
TPSL_ID = os.environ.get("TPSL_JSON_ID",  "")

download_file(PKL_ID,  f"{MODELS_DIR}/brain1_xgboost_v4.pkl")
download_file(JSON_ID, f"{MODELS_DIR}/brain1_features_v4.json")
download_file(TPSL_ID, f"{MODELS_DIR}/dynamic_tpsl_config.json")

# ── Load trading agent ────────────────────────────────────────────────────────
try:
    AGENT_CODE = open("/app/trading_agent.py").read()
    AGENT_CODE = AGENT_CODE.replace(
        '"/content/drive/MyDrive/trading_agent/',
        f'"{MODELS_DIR}/'
    )
    exec(AGENT_CODE, globals())
    print("✅ Trading agent loaded")
except Exception as e:
    print(f"❌ Failed to load trading agent: {e}")

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "")
RUN_INTERVAL   = int(os.environ.get("RUN_INTERVAL", "3600"))

# ── numpy cleaner ─────────────────────────────────────────────────────────────
def clean(obj):
    if isinstance(obj, dict):            return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):            return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    return obj

# ── FastAPI ───────────────────────────────────────────────────────────────────
api = FastAPI(title="XAU/USD Triple Brain Agent")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/")
def root():
    return {"agent": "XAU/USD Triple Brain v4", "status": "running",
            "time": datetime.now(timezone.utc).isoformat()}

@api.get("/health")
def health():
    return {"status": "ok", "agent": "XAU/USD Triple Brain v4",
            "time": datetime.now(timezone.utc).isoformat()}

@api.get("/files")
def check_files():
    """Debug endpoint — check if model files are downloaded."""
    files = {}
    for fname in ["brain1_xgboost_v4.pkl", "brain1_features_v4.json", "dynamic_tpsl_config.json"]:
        path = f"{MODELS_DIR}/{fname}"
        files[fname] = {
            "exists": os.path.exists(path),
            "size": os.path.getsize(path) if os.path.exists(path) else 0
        }
    env = {
        "MODEL_PKL_ID":  PKL_ID[:8]+"..." if PKL_ID else "NOT SET",
        "MODEL_JSON_ID": JSON_ID[:8]+"..." if JSON_ID else "NOT SET",
        "TPSL_JSON_ID":  TPSL_ID[:8]+"..." if TPSL_ID else "NOT SET",
    }
    return {"files": files, "env_ids": env}

@api.post("/run-analysis")
def run_analysis_endpoint():
    try:
        return JSONResponse(content=clean(run_analysis()))
    except Exception as e:
        return JSONResponse(content={"signal": "ERROR", "error": str(e),
                "health": {"data_feed": {"ok": False, "msg": str(e)}}})

@api.get("/chart-data")
def chart_data():
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", interval="15m",
                         auto_adjust=False, progress=False)
        df.columns = [col[0].lower() if isinstance(col,tuple) else col.lower()
                      for col in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        df = df[df.index.dayofweek < 5].tail(200)
        bars = [{"time": int(pd.Timestamp(ts).timestamp()),
                 "open":  round(float(r["open"]),  2),
                 "high":  round(float(r["high"]),  2),
                 "low":   round(float(r["low"]),   2),
                 "close": round(float(r["close"]), 2)}
                for ts, r in df.iterrows()]
        return JSONResponse(content={"bars": bars, "symbol": "XAU/USD", "tf": "15m"})
    except Exception as e:
        return JSONResponse(content={"error": str(e), "bars": []})

@api.get("/daily-summary")
def daily_summary():
    try:
        return JSONResponse(content=clean(get_daily_summary()))
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@api.get("/session-status")
def session_status():
    q, allowed = _get_session_quality()
    return {"quality": q, "trade_allowed": allowed}

# ── Telegram ──────────────────────────────────────────────────────────────────
def tg(msg):
    try:
        req.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                 json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=10)
    except: pass

_prev_health = {}
def check_health(result):
    global _prev_health
    health = result.get("health", {})
    if not health: return
    for key, status in health.items():
        was_ok = _prev_health.get(key, {}).get("ok", True)
        is_ok  = status.get("ok", True)
        msg    = status.get("msg", "")
        if was_ok and not is_ok:
            tg(f"⚠️ FAULT: {key}\n{msg}\n{datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        elif not was_ok and is_ok:
            tg(f"✅ RECOVERED: {key}")
    _prev_health = {k: dict(v) for k, v in health.items()}

# ── Scheduler ─────────────────────────────────────────────────────────────────
def scheduler():
    time.sleep(15)
    tg(f"🚀 AGENT STARTED (Railway 24/7)\nScheduler: every {RUN_INTERVAL//60} min\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}\nDashboard: ceylonpropertylink.com/agent/dashboard.html")
    while True:
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        print(f"[SCHEDULER] {now}")
        try:
            result = run_analysis()
            sig   = result.get("signal","?")
            conf  = result.get("confidence", 0)
            price = result.get("price", 0)
            print(f"[SCHEDULER] {sig} | {conf}/10 | ${price}")
            check_health(result)
        except Exception as e:
            print(f"[SCHEDULER] ERROR: {e}")
            tg(f"❌ run_analysis failed\n{str(e)[:200]}")
        time.sleep(RUN_INTERVAL)

threading.Thread(target=scheduler, daemon=True).start()
print("✅ Scheduler started")
