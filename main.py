"""
XAU/USD Triple Brain Agent - Railway Deployment
Phase 27 - Fixed: NoneType crash, safe format helpers, 5min interval
"""
import os, threading, time, traceback
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App ───────────────────────────────────────────────────────
api = FastAPI(title="XAU/USD Triple Brain Agent v4")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment variables ─────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT", "")
RUN_INTERVAL   = int(os.environ.get("RUN_INTERVAL", "300"))
PKL_ID         = os.environ.get("MODEL_PKL_ID", "")
JSON_ID        = os.environ.get("MODEL_JSON_ID", "")
TPSL_ID        = os.environ.get("TPSL_JSON_ID", "")
TD_KEY         = "2c3dff7091284f92b2361649006448a8"
MODELS_DIR     = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── CRITICAL FIX: Safe number helpers ────────────────────────
# These prevent the "NoneType.__format__" crash that was happening
# every hour because run_analysis() returns None for some values.

def sf(v, d=2):
    """Safe float format - NEVER crashes on None/NaN."""
    try:
        if v is None:
            return "0.00"
        f = float(v)
        if f != f:  # NaN check
            return "0.00"
        return "{:.{}f}".format(f, d)
    except Exception:
        return "0.00"

def sn(v, fallback=0):
    """Safe number - returns numeric value or fallback."""
    try:
        if v is None:
            return fallback
        f = float(v)
        return fallback if f != f else f
    except Exception:
        return fallback

# ── Clean numpy/pandas types for JSON ────────────────────────
def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return None  # NaN to null
    return obj

# ── Download models from Google Drive ────────────────────────
def download_file(file_id, dest):
    if os.path.exists(dest):
        print("EXISTS: " + os.path.basename(dest))
        return True
    if not file_id or "PASTE" in file_id:
        print("NO ID: " + os.path.basename(dest))
        return False
    try:
        print("DOWNLOADING: " + os.path.basename(dest))
        url = "https://drive.google.com/uc?export=download&id=" + file_id
        session = req.Session()
        r = session.get(url, stream=True, timeout=60)
        for k, v in r.cookies.items():
            if k.startswith("download_warning"):
                r = session.get(url + "&confirm=" + v, stream=True, timeout=60)
                break
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print("OK: " + os.path.basename(dest) + " (" + str(os.path.getsize(dest)) + " bytes)")
        return True
    except Exception as e:
        print("FAILED: " + str(e))
        if os.path.exists(dest):
            os.remove(dest)
        return False

download_file(PKL_ID,  MODELS_DIR + "/brain1_xgboost_v4.pkl")
download_file(JSON_ID, MODELS_DIR + "/brain1_features_v4.json")
download_file(TPSL_ID, MODELS_DIR + "/dynamic_tpsl_config.json")

# ── Load trading_agent.py ─────────────────────────────────────
_agent_loaded = False
try:
    code = open("/app/trading_agent.py").read()
    code = code.replace(
        '"/content/drive/MyDrive/trading_agent/',
        '"' + MODELS_DIR + '/'
    )
    exec(code, globals())
    _agent_loaded = True
    print("trading_agent.py loaded OK")
except Exception as e:
    print("trading_agent.py FAILED: " + str(e))

# ── Telegram ──────────────────────────────────────────────────
def tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return False
    try:
        r = req.post(
            "https://api.telegram.org/bot" + TELEGRAM_TOKEN + "/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        print("Telegram error: " + str(e))
        return False

# ── Session helper ────────────────────────────────────────────
def get_session():
    now = datetime.now(timezone.utc)
    t = now.hour * 60 + now.minute
    # London: 08:00-17:00 UTC | NY: 13:00-22:00 | Overlap: 13:00-17:00
    if 780 <= t < 1020:
        return "LONDON/NY OVERLAP", "HIGH", True, "🔥"
    elif 480 <= t < 780:
        return "LONDON SESSION", "MEDIUM", True, "🟢"
    elif 1020 <= t < 1320:
        return "NEW YORK SESSION", "MEDIUM", True, "🟡"
    else:
        return "ASIAN SESSION", "LOW", False, "🔵"

# ── Shared state ──────────────────────────────────────────────
last_result   = {}
last_run_time = ""
_prev_health  = {}
_run_count    = 0

def _save_result(result):
    global last_result, last_run_time
    last_result   = result
    last_run_time = datetime.now(timezone.utc).isoformat()

def _check_health(result):
    global _prev_health
    health = result.get("health", {})
    for key, status in health.items():
        was_ok = _prev_health.get(key, {}).get("ok", True)
        is_ok  = status.get("ok", True)
        if was_ok and not is_ok:
            tg("FAULT: " + key + " - " + status.get("msg", ""))
        elif not was_ok and is_ok:
            tg("RECOVERED: " + key)
    _prev_health = {k: dict(v) for k, v in health.items()}

# ── Endpoints ─────────────────────────────────────────────────
@api.get("/")
def root():
    sname, sq, _, si = get_session()
    return {
        "agent":   "XAU/USD Triple Brain v4",
        "status":  "running",
        "session": sname,
        "session_quality": sq,
        "session_icon":    si,
        "time":    datetime.now(timezone.utc).isoformat(),
        "last_run": last_run_time or "never",
        "run_count": _run_count
    }

@api.get("/health")
def health_ep():
    return {
        "status":        "ok",
        "agent_loaded":  _agent_loaded,
        "last_run":      last_run_time or "never",
        "last_signal":   last_result.get("signal", "none"),
        "run_count":     _run_count,
        "time":          datetime.now(timezone.utc).isoformat()
    }

@api.get("/status")
def status_ep():
    sname, sq, stok, si = get_session()
    base = {
        "session":           sname,
        "session_quality":   sq,
        "session_trade_ok":  stok,
        "session_icon":      si,
        "last_run":          last_run_time or "never",
    }
    if not last_result:
        base.update({
            "signal":     "WAIT",
            "confidence": 0,
            "price":      0,
            "msg": "No analysis yet - runs every " + str(RUN_INTERVAL // 60) + " min",
            "health": {
                "data_feed":   {"ok": _agent_loaded, "msg": "agent " + ("ok" if _agent_loaded else "FAILED")},
                "brain1":      {"ok": False, "msg": "pending first run"},
                "brain2":      {"ok": False, "msg": "pending first run"},
                "brain3":      {"ok": False, "msg": "pending first run"},
                "h4":          {"ok": False, "msg": "pending"},
                "telegram":    {"ok": bool(TELEGRAM_TOKEN), "msg": "ok" if TELEGRAM_TOKEN else "no token"},
                "daily_limit": {"ok": True, "msg": "ok"}
            }
        })
        return JSONResponse(content=base)
    result = dict(last_result)
    result.update(base)
    return JSONResponse(content=clean(result))

@api.post("/run-analysis")
def run_ep():
    if not _agent_loaded:
        return JSONResponse(content={
            "signal": "ERROR",
            "error":  "trading_agent.py not loaded - check Railway logs"
        }, status_code=500)
    try:
        result = run_analysis()
        _save_result(result)
        _check_health(result)
        return JSONResponse(content=clean(result))
    except Exception as e:
        tb = traceback.format_exc()
        print("[run-analysis] " + str(e) + "\n" + tb[:500])
        return JSONResponse(content={
            "signal": "ERROR",
            "error":  str(e)
        }, status_code=500)

@api.get("/chart-data")
def chart_ep():
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
def tg_ep(body: dict):
    ok = tg(body.get("message", "Test"))
    if ok:
        return {"ok": True}
    return JSONResponse(content={"ok": False, "msg": "Failed - check TELEGRAM_TOKEN env var"}, status_code=500)

@api.get("/files")
def files_ep():
    files = {}
    for name in ["brain1_xgboost_v4.pkl", "brain1_features_v4.json", "dynamic_tpsl_config.json"]:
        path = MODELS_DIR + "/" + name
        files[name] = {
            "exists":  os.path.exists(path),
            "size_kb": round(os.path.getsize(path) / 1024, 1) if os.path.exists(path) else 0
        }
    return {
        "files":        files,
        "RUN_INTERVAL": RUN_INTERVAL,
        "telegram":     "SET" if TELEGRAM_TOKEN else "NOT SET",
        "agent_loaded": _agent_loaded
    }

# ── Scheduler ─────────────────────────────────────────────────
def scheduler():
    global _run_count
    time.sleep(20)
    sname, _, _, _ = get_session()
    tg(
        "AGENT STARTED - Railway 24/7\n"
        "Interval: every " + str(RUN_INTERVAL // 60) + " min\n"
        "Session: " + sname + "\n"
        "Time: " + datetime.now(timezone.utc).strftime("%H:%M UTC") + "\n"
        "Dashboard: ceylonpropertylink.com/agent/dashboard.html"
    )

    while True:
        now_str  = datetime.now(timezone.utc).strftime("%H:%M UTC")
        sname2, sq2, stok2, _ = get_session()
        print("[" + now_str + "] Running | " + sq2 + " | trade_ok=" + str(stok2))

        try:
            result = run_analysis()
            _run_count += 1
            _save_result(result)
            _check_health(result)

            # ── SAFE extraction - never crash on None ─────────
            sig   = str(result.get("signal") or "WAIT").upper()
            conf  = sn(result.get("confidence"), 0)
            price = sn(result.get("price"),       0)
            b1    = sn((result.get("brain1") or {}).get("probability"), 0)
            b2    = sn((result.get("brain2") or {}).get("score"),       0)
            tp    = sn(result.get("tp"),  0)
            sl    = sn(result.get("sl"),  0)
            h4v   = bool(result.get("h4_veto",        False))
            sb    = bool(result.get("session_blocked", False))

            print("  " + sig + " | B1=" + sf(b1,3) + " | B2=" + sf(b2,1) + " | conf=" + sf(conf,1) + " | $" + sf(price))

            if sig in ("BUY", "SELL"):
                em   = "BUY" if sig == "BUY" else "SELL"
                icon = "📈" if sig == "BUY" else "📉"
                rr   = 0
                if tp and sl and price and abs(sl - price) > 0:
                    rr = abs(tp - price) / abs(sl - price)
                tg(
                    icon + " <b>" + em + " - XAU/USD</b>\n\n"
                    "Confidence: <b>" + sf(conf, 1) + "/10</b>\n"
                    "Price: <b>$" + sf(price) + "</b>\n"
                    "Take Profit: $" + sf(tp) + "\n"
                    "Stop Loss:   $" + sf(sl) + "\n"
                    "RR: 1:" + sf(rr, 1) + "\n\n"
                    "B1:" + sf(b1, 3) + " B2:" + sf(b2, 1) + "/10 B3:" + sf(conf, 1) + "/10\n"
                    + sname2 + " | " + now_str + "\n\n"
                    "Open trade on Exness NOW!\n"
                    "XAU/USD Triple Brain v4"
                )

            elif _run_count % 6 == 0:
                # Status update every 30 min (6 x 5min cycles)
                why = "setup not ready"
                if sb:
                    why = "session blocked (" + sq2 + ")"
                elif h4v:
                    why = "H4 veto active"
                elif b1 < 0.60:
                    why = "B1=" + sf(b1, 3) + " (need 0.600)"
                elif b2 < 5:
                    why = "B2=" + sf(b2, 1) + "/10 (need 5.0)"
                elif conf < 6.5:
                    why = "B3=" + sf(conf, 1) + "/10 (need 6.5)"
                tg(
                    "Status: WAIT - " + why + "\n"
                    "B1:" + sf(b1, 3) + " B2:" + sf(b2, 1) + "/10 B3:" + sf(conf, 1) + "/10\n"
                    "Price: $" + sf(price) + "\n"
                    + sname2 + " | " + now_str
                )

        except Exception as e:
            tb = traceback.format_exc()
            short = str(e)[:250]
            print("[SCHEDULER ERROR] " + short)
            print(tb[:400])
            tg("Analysis error: " + short + "\n" + now_str + "\nCheck Railway logs.")

        time.sleep(RUN_INTERVAL)


threading.Thread(target=scheduler, daemon=True).start()
print("Scheduler started - every " + str(RUN_INTERVAL) + "s (" + str(RUN_INTERVAL // 60) + " min)")
