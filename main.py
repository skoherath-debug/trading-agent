"""
XAU/USD Triple Brain Agent - Railway Deployment
Phase 29 - Smart scheduler, key rotation, live price, canvas chart
"""
import os, threading, time, traceback
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Phase 28/29: Import Improvements ──────────────────────────
try:
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
    )
    IMPROVEMENTS_LOADED = True
    print("✅ Phase 28/29 improvements loaded")
except Exception as e:
    IMPROVEMENTS_LOADED = False
    print(f"⚠️  Improvements not available: {e}")

# ── App ───────────────────────────────────────────────────────
api = FastAPI(title="XAU/USD Triple Brain Agent · Phase 29")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if IMPROVEMENTS_LOADED:
    api.include_router(improvements_router)

# ── Environment variables ─────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "")
RUN_INTERVAL   = int(os.environ.get("RUN_INTERVAL", "300"))
PKL_ID         = os.environ.get("MODEL_PKL_ID",  "")
JSON_ID        = os.environ.get("MODEL_JSON_ID", "")
TPSL_ID        = os.environ.get("TPSL_JSON_ID",  "")
TD_KEY         = os.environ.get("TD_KEY",         "")
MODELS_DIR     = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Phase 29: Tracking ────────────────────────────────────────
_atr_history  = []
_volatility_ma = 15.0

# ── Safe helpers ──────────────────────────────────────────────
def sf(v, d=2):
    try:
        if v is None: return "0.00"
        f = float(v)
        if f != f: return "0.00"
        return "{:.{}f}".format(f, d)
    except Exception: return "0.00"

def sn(v, fallback=0):
    try:
        if v is None: return fallback
        f = float(v)
        return fallback if f != f else f
    except Exception: return fallback

def clean(obj):
    if isinstance(obj, dict):   return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [clean(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, float) and obj != obj: return None
    return obj

# ── Download models ───────────────────────────────────────────
def download_file(file_id, dest):
    if os.path.exists(dest):
        fsize = os.path.getsize(dest)
        if dest.endswith(".pkl") and fsize < 5000:
            print(f"CORRUPT: {os.path.basename(dest)} only {fsize} bytes — re-downloading")
            os.remove(dest)
        else:
            print("EXISTS: " + os.path.basename(dest) + " (" + str(fsize) + " bytes)")
            return True
    if not file_id or "PASTE" in file_id:
        print("NO ID: " + os.path.basename(dest))
        return False
    try:
        print("DOWNLOADING: " + os.path.basename(dest))
        try:
            import gdown
            gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=False)
            if os.path.exists(dest) and os.path.getsize(dest) > 5000:
                print("OK (gdown): " + os.path.basename(dest))
                return True
            elif os.path.exists(dest):
                os.remove(dest)
        except Exception as ge:
            print(f"gdown failed: {ge}")
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
                if chunk: f.write(chunk)
        fsize = os.path.getsize(dest)
        if fsize < 5000 and dest.endswith(".pkl"):
            with open(dest, "rb") as f:
                header = f.read(20)
            if b"<!DOCTYPE" in header or b"<html" in header:
                print("ERROR: Got HTML instead of model")
                os.remove(dest)
                return False
        print("OK: " + os.path.basename(dest) + " (" + str(fsize) + " bytes)")
        return True
    except Exception as e:
        print("FAILED: " + str(e))
        if os.path.exists(dest): os.remove(dest)
        return False

download_file(PKL_ID,  MODELS_DIR + "/brain1_xgboost_v4.pkl")
download_file(JSON_ID, MODELS_DIR + "/brain1_features_v4.json")
download_file(TPSL_ID, MODELS_DIR + "/dynamic_tpsl_config.json")

# ── Load trading_agent.py ─────────────────────────────────────
_agent_loaded = False
try:
    code = open("/app/trading_agent.py").read()
    code = code.replace('"/content/drive/MyDrive/trading_agent/', '"' + MODELS_DIR + '/')
    exec(code, globals())
    _agent_loaded = True
    print("trading_agent.py loaded OK")
except Exception as e:
    print("trading_agent.py FAILED: " + str(e))

# ── Telegram ──────────────────────────────────────────────────
def tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT: return False
    try:
        r = req.post(
            "https://api.telegram.org/bot" + TELEGRAM_TOKEN + "/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        print("Telegram error: " + str(e))
        return False

# ── Session helper ────────────────────────────────────────────
def get_session():
    t = datetime.now(timezone.utc)
    m = t.hour * 60 + t.minute
    if 780 <= m < 1020: return "LONDON/NY OVERLAP", "HIGH",   True,  "🔥"
    elif 480 <= m < 780: return "LONDON SESSION",   "MEDIUM", True,  "🟢"
    elif 1020 <= m < 1320: return "NEW YORK SESSION","MEDIUM", True,  "🟡"
    else:                 return "ASIAN/OFF-HOURS",  "LOW",    False, "🔵"

# ── Smart interval ────────────────────────────────────────────
def get_run_interval():
    m = datetime.now(timezone.utc).hour * 60 + datetime.now(timezone.utc).minute
    return 120 if 420 <= m < 1080 else 1800

# ── Shared state ──────────────────────────────────────────────
last_result   = {}
last_run_time = ""
_prev_health  = {}
_run_count    = 0
_price_cache  = {"price": None, "ts": 0.0}
_chart_cache  = {"bars": [], "ts": 0.0}
CHART_CACHE_TTL = 60

def _save_result(result):
    global last_result, last_run_time, _price_cache
    last_result   = result
    last_run_time = datetime.now(timezone.utc).isoformat()
    if result.get("price"):
        _price_cache = {"price": result["price"], "ts": time.time()}

def _check_health(result):
    global _prev_health
    for key, status in result.get("health", {}).items():
        was_ok = _prev_health.get(key, {}).get("ok", True)
        is_ok  = status.get("ok", True)
        if was_ok and not is_ok:
            tg("⚠️ FAULT: " + key + "\n" + status.get("msg", ""))
        elif not was_ok and is_ok:
            tg("✅ RECOVERED: " + key)
    _prev_health = {k: dict(v) for k, v in result.get("health", {}).items()}

# ── Endpoints ─────────────────────────────────────────────────
@api.get("/")
def root():
    sname, sq, _, si = get_session()
    return {"agent":"XAU/USD Triple Brain · Phase 29","status":"running",
            "session":sname,"session_quality":sq,"session_icon":si,
            "time":datetime.now(timezone.utc).isoformat(),"last_run":last_run_time or "never",
            "run_count":_run_count}

@api.get("/health")
def health_ep():
    return {"status":"ok","agent_loaded":_agent_loaded,"improvements":IMPROVEMENTS_LOADED,
            "last_run":last_run_time or "never","last_signal":last_result.get("signal","none"),
            "run_count":_run_count,"time":datetime.now(timezone.utc).isoformat()}

@api.get("/status")
def status_ep():
    sname, sq, stok, si = get_session()
    base = {"session":sname,"session_quality":sq,"session_trade_ok":stok,"session_icon":si,
            "last_run":last_run_time or "never","improvements":"ENABLED" if IMPROVEMENTS_LOADED else "DISABLED"}
    if not last_result:
        base.update({"signal":"WAIT","confidence":0,"price":0,
            "msg":"No analysis yet","health":{
                "data_feed":   {"ok":_agent_loaded,"msg":"agent "+("ok" if _agent_loaded else "FAILED")},
                "brain1":      {"ok":False,"msg":"pending first run"},
                "brain2":      {"ok":False,"msg":"pending first run"},
                "brain3":      {"ok":False,"msg":"pending first run"},
                "h4":          {"ok":False,"msg":"pending"},
                "telegram":    {"ok":bool(TELEGRAM_TOKEN),"msg":"ok" if TELEGRAM_TOKEN else "no token"},
                "daily_limit": {"ok":True,"msg":"ok"}}})
        return JSONResponse(content=base)
    result = dict(last_result)
    result.update(base)
    return JSONResponse(content=clean(result))

@api.post("/run-analysis")
def run_ep():
    if not _agent_loaded:
        return JSONResponse(content={"signal":"ERROR","error":"trading_agent.py not loaded"},status_code=500)
    try:
        result = run_analysis()
        _save_result(result)
        _check_health(result)
        return JSONResponse(content=clean(result))
    except Exception as e:
        return JSONResponse(content={"signal":"ERROR","error":str(e)},status_code=500)

@api.get("/live-price")
def live_price_ep():
    global _price_cache
    now = time.time()
    if _price_cache["price"] and (now - _price_cache["ts"]) < 30:
        return JSONResponse(content={"price":_price_cache["price"],"cached":True})
    # Try all 3 TD keys
    td_keys = [k for k in [
        os.environ.get("TD_KEY","f3883b7831a540cda02cfafcfe77e082"),
        os.environ.get("TD_KEY2","41c8cfdf490b4bf4a0d388e716a32453"),
        os.environ.get("TD_KEY3","f58b0a482f1443e78fb23cf8975b44d9"),
    ] if k]
    for key in td_keys:
        try:
            r = req.get("https://api.twelvedata.com/price",
                        params={"symbol":"XAU/USD","apikey":key},timeout=8)
            data = r.json()
            if "price" in data:
                price = round(float(data["price"]),2)
                _price_cache = {"price":price,"ts":now}
                if last_result: last_result["price"] = price
                return JSONResponse(content={"price":price,"cached":False,
                                             "time":datetime.now(timezone.utc).isoformat()})
        except Exception: continue
    return JSONResponse(content={"price":last_result.get("price"),"cached":True})

@api.get("/chart-data")
def chart_ep():
    global _chart_cache
    if _chart_cache["bars"] and (time.time()-_chart_cache["ts"]) < CHART_CACHE_TTL:
        return JSONResponse(content={"bars":_chart_cache["bars"],"symbol":"XAU/USD","tf":"15m","cached":True})
    td_keys = [k for k in [
        os.environ.get("TD_KEY","f3883b7831a540cda02cfafcfe77e082"),
        os.environ.get("TD_KEY2","41c8cfdf490b4bf4a0d388e716a32453"),
        os.environ.get("TD_KEY3","f58b0a482f1443e78fb23cf8975b44d9"),
    ] if k]
    for key in td_keys:
        try:
            r = req.get("https://api.twelvedata.com/time_series",
                params={"symbol":"XAU/USD","interval":"15min","outputsize":200,
                        "apikey":key,"timezone":"UTC","format":"JSON"},timeout=30)
            data = r.json()
            if "values" not in data: continue
            bars = []
            for v in reversed(data["values"]):
                try:
                    bars.append({"time":int(pd.Timestamp(v["datetime"]).timestamp()),
                                 "open":round(float(v["open"]),2),"high":round(float(v["high"]),2),
                                 "low":round(float(v["low"]),2),"close":round(float(v["close"]),2)})
                except Exception: pass
            _chart_cache["bars"] = bars
            _chart_cache["ts"]   = time.time()
            return JSONResponse(content={"bars":bars,"symbol":"XAU/USD","tf":"15m"})
        except Exception: continue
    if _chart_cache["bars"]:
        return JSONResponse(content={"bars":_chart_cache["bars"],"symbol":"XAU/USD","tf":"15m","cached":True,"stale":True})
    return JSONResponse(content={"error":"All TD keys failed","bars":[]},status_code=500)

@api.get("/test-telegram")
def test_tg_ep():
    ok = tg("✅ Telegram test\nagent.ceylonpropertylink.com\n" + datetime.now(timezone.utc).strftime("%H:%M UTC"))
    return {"ok":ok}

@api.post("/telegram/send")
def tg_ep(body: dict):
    ok = tg(body.get("message","Test"))
    return {"ok":ok} if ok else JSONResponse(content={"ok":False,"msg":"Failed"},status_code=500)

@api.get("/files")
def files_ep():
    files = {}
    for name in ["brain1_xgboost_v4.pkl","brain1_features_v4.json","dynamic_tpsl_config.json"]:
        path = MODELS_DIR + "/" + name
        files[name] = {"exists":os.path.exists(path),
                       "size_kb":round(os.path.getsize(path)/1024,1) if os.path.exists(path) else 0}
    return {"files":files,"RUN_INTERVAL":RUN_INTERVAL,"telegram":"SET" if TELEGRAM_TOKEN else "NOT SET",
            "agent_loaded":_agent_loaded,"improvements":"ENABLED" if IMPROVEMENTS_LOADED else "DISABLED"}

# ── Scheduler ─────────────────────────────────────────────────
def scheduler():
    global _run_count, _volatility_ma
    time.sleep(20)
    sname, _, _, _ = get_session()
    tg(
        "🚀 AGENT STARTED · Phase 29\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Schedule: 2min (07-18 UTC) / 30min off-hours\n"
        "Session: " + sname + "\n"
        "Time: " + datetime.now(timezone.utc).strftime("%H:%M UTC") + "\n"
        "Improvements: " + ("✅ ENABLED" if IMPROVEMENTS_LOADED else "⚠️ DISABLED") + "\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "agent.ceylonpropertylink.com"
    )

    while True:
        now_str          = datetime.now(timezone.utc).strftime("%H:%M UTC")
        sname2, sq2, stok2, _ = get_session()
        print("[" + now_str + "] Running | " + sq2)

        try:
            result = run_analysis()
            _run_count += 1
            _save_result(result)
            _check_health(result)

            sig   = str(result.get("signal")  or "WAIT").upper()
            conf  = sn(result.get("confidence"), 0)
            price = sn(result.get("price"),       0)
            b1    = sn((result.get("brain1") or {}).get("probability"), 0)
            b2    = sn((result.get("brain2") or {}).get("score"),       0)
            b3v   = str((result.get("brain3") or {}).get("verdict","—"))
            tp    = sn(result.get("tp"),   0)
            tp2   = sn(result.get("tp2"),  0)
            sl    = sn(result.get("sl"),   0)
            h4d   = str((result.get("h4") or {}).get("direction","—"))
            h4v   = bool(result.get("h4_veto",        False))
            sb    = bool(result.get("session_blocked", False))

            print("  " + sig + " | B1=" + sf(b1,3) + " | B2=" + sf(b2,1) + " | conf=" + sf(conf,1) + " | $" + sf(price))

            # Phase 28/29 improvements
            if IMPROVEMENTS_LOADED:
                try:
                    atr_val = sn(result.get("atr", 15.0), 15.0)
                    _atr_history.append(atr_val)
                    if len(_atr_history) > 20: _atr_history.pop(0)
                    _volatility_ma = float(np.mean(_atr_history)) if _atr_history else 15.0
                    news_coming, event_name, minutes_until = is_news_event_soon()
                    if news_coming:
                        print("   NEWS: " + event_name + " in " + str(round(minutes_until)) + " min")
                        sig = "WAIT"
                    market_condition    = detect_market_condition(atr_val, _volatility_ma)
                    recommended_lot     = calculate_position_size(atr_val)
                    risk_level          = get_volatility_risk_level(atr_val)
                    print("   Market: " + market_condition + " | Lot: " + sf(recommended_lot) + " | Risk: " + risk_level)
                except Exception as e:
                    print("   Improvements error: " + str(e))

            # ── Send BUY/SELL alert ──────────────────────────────
            if sig in ("BUY", "SELL"):
                icon = "📈" if sig == "BUY" else "📉"
                rr   = round(abs(tp-price)/abs(sl-price),1) if abs(sl-price) > 0 else 0

                if IMPROVEMENTS_LOADED:
                    try:
                        tracker.log_trade(
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            entry_price=price, exit_time=datetime.now(timezone.utc).isoformat(),
                            exit_price=price,  signal=sig, b1_score=b1, b2_score=b2,
                            b3_score=conf,     session=sname2
                        )
                    except Exception: pass

                tg(
                    icon + " " + sig + " XAU/USD\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "Entry : $" + sf(price) + "\n"
                    "TP1   : $" + sf(tp)  + "  (+" + sf(abs(tp -price)) + ")\n"
                    "TP2   : $" + sf(tp2) + "  (+" + sf(abs(tp2-price)) + ")\n"
                    "SL    : $" + sf(sl)  + "  (-" + sf(abs(sl -price)) + ")\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "Score : " + sf(conf,2) + "/10   RR: 1:" + sf(rr,1) + "\n"
                    "B1: " + sf(b1,3) + "  B2: " + sf(b2,1) + "/10  B3: " + b3v + "\n"
                    "H4: " + h4d + "  WR: 80.3%  Lot: 0.03\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    + sname2 + " | " + now_str + "\n"
                    "agent.ceylonpropertylink.com"
                )

            # ── Periodic WAIT status (every 30 min) ──────────────
            elif _run_count % 15 == 0:
                why = ("session blocked" if sb else
                       "H4 veto" if h4v else
                       "B1=" + sf(b1,3) + " needs 0.600" if b1 < 0.60 else
                       "B2=" + sf(b2,1) + " needs 5.0"   if b2 < 5.0  else
                       "conf=" + sf(conf,1) + " needs 6.5")
                tg(
                    "⏳ WAIT | $" + sf(price) + "\n"
                    "Reason: " + why + "\n"
                    "B1: " + sf(b1,3) + "  B2: " + sf(b2,1) + "/10\n"
                    + sname2 + " | " + now_str
                )

        except Exception as e:
            short = str(e)[:200]
            print("[SCHEDULER ERROR] " + short)
            tg("❌ Analysis error:\n" + short + "\n" + now_str)

        time.sleep(get_run_interval())


threading.Thread(target=scheduler, daemon=True).start()
print("Scheduler started - 2min trading / 30min off-hours · Phase 29")
print("Improvements: " + ("ENABLED ✅" if IMPROVEMENTS_LOADED else "DISABLED ⚠️"))
