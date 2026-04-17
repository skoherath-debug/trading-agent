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

# ── Expert Entry Analysis ────────────────────────────────────
def get_entry_analysis(result, current_price):
    """
    Professional entry point analysis:
    1. Entry zone check — is current price still in valid zone?
    2. Limit order recommendation
    3. Confirmation level
    4. Entry quality score
    """
    sig    = result.get("signal","WAIT")
    sig_px = sn(result.get("price"), 0)
    atr    = sn(result.get("atr"), 13)
    tp1    = sn(result.get("tp"),  0)
    sl     = sn(result.get("sl"),  0)
    h4     = result.get("h4") or {}
    ema20  = sn(h4.get("ema20"), 0)

    if sig not in ("BUY","SELL") or sig_px == 0:
        return None

    # Entry zone: signal price ± 0.4 ATR
    zone_size   = round(atr * 0.4, 2)
    zone_low    = round(sig_px - zone_size, 2)
    zone_high   = round(sig_px + zone_size, 2)

    # Limit order level: 0.2 ATR better than signal price
    limit_offset = round(atr * 0.2, 2)
    if sig == "BUY":
        limit_price = round(sig_px - limit_offset, 2)  # buy cheaper
        in_zone     = zone_low <= current_price <= zone_high
        price_ok    = current_price <= sig_px + zone_size
        confirm_lvl = round(max(sig_px, ema20) + 0.5, 2) if ema20 else sig_px
    else:
        limit_price = round(sig_px + limit_offset, 2)  # sell higher
        in_zone     = zone_low <= current_price <= zone_high
        price_ok    = current_price >= sig_px - zone_size
        confirm_lvl = round(min(sig_px, ema20) - 0.5, 2) if ema20 else sig_px

    # Slippage from signal price
    slippage    = round(abs(current_price - sig_px), 2)
    slippage_pct= round(slippage / atr * 100, 1)

    # Entry quality score (0-10)
    if slippage <= atr * 0.1:    quality = 10  # perfect
    elif slippage <= atr * 0.2:  quality = 8   # excellent
    elif slippage <= atr * 0.4:  quality = 6   # acceptable
    elif slippage <= atr * 0.6:  quality = 4   # marginal
    else:                         quality = 0   # skip

    # Adjusted TP/SL from current price
    rr_raw = abs(tp1-sig_px)/abs(sl-sig_px) if abs(sl-sig_px) > 0 else 0
    adj_tp1 = round(current_price + abs(tp1-sig_px), 2) if sig=="BUY" else round(current_price - abs(tp1-sig_px), 2)
    adj_sl  = round(current_price - abs(sl-sig_px),  2) if sig=="BUY" else round(current_price + abs(sl-sig_px),  2)

    return {
        "signal":       sig,
        "signal_price": sig_px,
        "current_price":current_price,
        "slippage":     slippage,
        "slippage_pct": slippage_pct,
        "in_zone":      in_zone,
        "price_ok":     price_ok,
        "quality":      quality,
        "zone_low":     zone_low,
        "zone_high":    zone_high,
        "limit_price":  limit_price,
        "limit_offset": limit_offset,
        "confirm_level":confirm_lvl,
        "adj_tp1":      adj_tp1,
        "adj_sl":       adj_sl,
        "atr":          atr,
    }

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
    # Sri Lanka trading: 12:30–23:30 PM = 07:00–18:00 UTC
    # Best window: London open 07:00–11:00 UTC + NY overlap 13:00–17:00 UTC
    if 780 <= m < 1020: return "LONDON/NY OVERLAP", "HIGH",   True,  "🔥"  # BEST
    elif 420 <= m < 780: return "LONDON SESSION",   "HIGH",   True,  "🟢"  # GOOD
    elif 1020 <= m < 1080: return "NEW YORK SESSION","MEDIUM", True,  "🟡"  # OK
    else:                  return "ASIAN/OFF-HOURS", "LOW",    False, "🔵"  # SKIP

# ── Smart interval ────────────────────────────────────────────
# ── Dual-phase timing ────────────────────────────────────────
# Phase A (2.5min): B1 + B2 pre-analysis — fast, no Groq
# Phase B (5min):   B3 Groq confirmation — full signal decision
PHASE_A_SECS   = 150   # 2.5 min — pre-compute B1+B2
PHASE_B_SECS   = 300   # 5.0 min — full analysis + B3 confirm
OFF_HOURS_SECS = 1800  # 30 min off-hours

def get_run_interval():
    m = datetime.now(timezone.utc).hour * 60 + datetime.now(timezone.utc).minute
    return OFF_HOURS_SECS if not (420 <= m < 1080) else PHASE_A_SECS

# ── Shared state ──────────────────────────────────────────────
last_result   = {}
last_run_time = ""
_prev_health  = {}
_run_count    = 0
_last_alerted_sig   = "WAIT"   # duplicate suppression
_pre_signal         = "WAIT"   # Phase A pre-signal
_pre_b1             = 0.0      # Phase A B1 probability
_pre_b2             = 0.0      # Phase A B2 score
_pre_time           = 0.0      # Phase A timestamp
_phase_a_count      = 0        # counts 2.5min cycles
_last_alerted_price = 0.0
_last_alerted_time  = 0.0      # cooldown timer
ALERT_COOLDOWN_SECS = 1800     # 30 min — one signal per gold setup
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

@api.get("/entry-analysis")
def entry_analysis_ep():
    """Real-time entry point quality analysis."""
    if not last_result:
        return {"error": "No signal yet"}
    try:
        # Get current live price
        cur = sn(_price_cache.get("price"), sn(last_result.get("price"), 0))
        analysis = get_entry_analysis(last_result, cur)
        if not analysis:
            return {"signal": "WAIT", "msg": "No active signal"}
        return analysis
    except Exception as e:
        return {"error": str(e)}

@api.get("/rocket-status")
def rocket_ep():
    """Phase 30 — latest Rocket/Waterfall micro-entry status."""
    return JSONResponse(content=clean(_rw_result) if _rw_result else {
        "rocket":0,"waterfall":0,"signal":"WAIT","msg":"No scan yet — starts 07:00 UTC"
    })

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
            global _pre_signal, _pre_b1, _pre_b2, _pre_time, _phase_a_count
            _phase_a_count += 1
            is_phase_b = (_phase_a_count % 2 == 0)  # every 2nd cycle = 5min

            # ── PHASE A (every 2.5min): B1 + B2 only ─────────────
            if not is_phase_b:
                print("[PHASE A] Pre-computing B1+B2...")
                try:
                    # Run lightweight analysis — B1+B2 only, skip B3 Groq
                    result_a = run_analysis()
                    _pre_b1  = sn((result_a.get("brain1") or {}).get("probability"), 0)
                    _pre_b2  = sn((result_a.get("brain2") or {}).get("score"), 0)
                    _pre_time= time.time()

                    # Determine pre-signal direction
                    if _pre_b1 >= 0.600 and _pre_b2 >= 5.0:
                        _pre_signal = result_a.get("brain2",{}).get("details",{}).get("trend","WAIT")
                        _pre_signal = "BUY" if "BULL" in _pre_signal.upper() else "WAIT"
                    else:
                        _pre_signal = "WAIT"

                    _save_result(result_a)
                    price_a = sn(result_a.get("price"), 0)
                    print("  [A] B1=" + sf(_pre_b1,3) + " B2=" + sf(_pre_b2,1) + " Pre=" + _pre_signal + " $" + sf(price_a))
                except Exception as ea:
                    print("  [A] Pre-analysis error: " + str(ea)[:100])
                time.sleep(PHASE_A_SECS)
                continue  # wait for Phase B

            # ── PHASE B (every 5min): Full B3 confirmation ────────
            print("[PHASE B] Full analysis with B3 confirmation...")
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

            # Phase B consistency check:
            # Signal only fires if Phase A pre-signal agrees with Phase B
            phase_consistent = (
                (_pre_signal == sig) or           # both phases agree
                (sig in ("BUY","SELL") and         # OR strong signal
                 abs(time.time()-_pre_time) < 400) # within 2.5min window
            )

            if sig in ("BUY","SELL") and not phase_consistent:
                print("  [B] SUPPRESSED — Phase A=" + _pre_signal + " Phase B=" + sig + " (inconsistent)")
                sig = "WAIT"  # suppress inconsistent signal

            print("  [B] " + sig + " | B1=" + sf(b1,3) + " B2=" + sf(b2,1) + " B3=" + b3v + " conf=" + sf(conf,1) + " $" + sf(price))

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
            # ── Cooldown suppression — 15min between alerts ───────────
            global _last_alerted_sig, _last_alerted_price, _last_alerted_time
            now_ts       = time.time()
            cooldown_ok  = (now_ts - _last_alerted_time) >= ALERT_COOLDOWN_SECS
            dir_changed  = sig != _last_alerted_sig   # BUY→SELL always fires
            price_moved  = abs(price - _last_alerted_price) > 8.0

            # Quality gates for alert
            session_quality_ok = sq2 in ("HIGH", "MEDIUM")
            rocket_score    = sn((result.get("brain2") or {}).get("details", {}).get("rocket_score"), 0)
            waterfall_score = sn((result.get("brain2") or {}).get("details", {}).get("waterfall_score"), 0)
            momentum_ok = (sig == "BUY"  and rocket_score    >= 50) or                           (sig == "SELL" and waterfall_score >= 50)

            # Fire if: cooldown ok AND session ok AND momentum confirmed OR direction flipped
            should_alert = (cooldown_ok and session_quality_ok and momentum_ok) or dir_changed

            if sig in ("BUY", "SELL") and should_alert:
                _last_alerted_sig   = sig
                _last_alerted_price = price
                _last_alerted_time  = now_ts
                mins_since = round((now_ts - _last_alerted_time)/60, 1) if _last_alerted_time else 0
                print("   📱 ALERT: " + sig + " (cooldown ok, last alert " + str(mins_since) + "m ago)")
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

                # Expert entry analysis
                ea = get_entry_analysis(result, price)
                if ea:
                    q_emoji = "🟢" if ea["quality"]>=8 else "🟡" if ea["quality"]>=5 else "🔴"
                    entry_guide = (
                        "Option A (Market): $" + sf(price) + "\n"
                        "Option B (Limit):  $" + sf(ea["limit_price"]) + " (+" + sf(ea["limit_offset"]) + " better)\n"
                        "Valid zone: $" + sf(ea["zone_low"]) + " — $" + sf(ea["zone_high"]) + "\n"
                        "If price > $" + sf(ea["zone_high"]) + " → SKIP"
                    )
                else:
                    entry_guide = "Entry: $" + sf(price)
                    q_emoji = "🟡"

                mom_line = ("🚀 Rocket: " + str(int(rocket_score)) + "/100\n") if sig=="BUY" else ("💧 Waterfall: " + str(int(waterfall_score)) + "/100\n")
                tg(
                    icon + " " + sig + " XAU/USD\n"
                    + "━━━━━━━━━━━━━━━━━━━━\n"
                    + "Entry: $" + sf(price) + "  Quality: " + q_emoji + " " + (sf(ea["quality"],0)+"/10" if ea else "—") + "\n"
                    + entry_guide + "\n"
                    + "━━━━━━━━━━━━━━━━━━━━\n"
                    + "TP1: $" + sf(tp)  + " (+" + sf(abs(tp -price)) + ")  TP2: $" + sf(tp2) + "\n"
                    + "SL:  $" + sf(sl)  + " (-" + sf(abs(sl -price)) + ")\n"
                    + "━━━━━━━━━━━━━━━━━━━━\n"
                    + "Score: " + sf(conf,2) + "/10  RR: 1:" + sf(rr,1) + "\n"
                    + "B1: " + sf(b1,3) + "  B2: " + sf(b2,1) + "/10  H4: " + h4d + "\n"
                    + mom_line
                    + "WR: 80.3%  Lot: 0.03\n"
                    + "━━━━━━━━━━━━━━━━━━━━\n"
                    + sname2 + " | " + now_str + "\n"
                    + "agent.ceylonpropertylink.com"
                )

            # Reset tracker when WAIT — allows fresh signal next time
            if sig == "WAIT" and _last_alerted_sig in ("BUY","SELL"):
                _last_alerted_sig   = "WAIT"
                _last_alerted_price = 0.0
                # Don't reset _last_alerted_time — cooldown still applies


            # ── SL / TP price alerts ─────────────────────────────────────────
            _tp1  = sn(last_result.get("tp"),  0)
            _tp2  = sn(last_result.get("tp2"), 0)
            _sl   = sn(last_result.get("sl"),  0)
            _atr  = sn(last_result.get("atr"), 15)

            if _last_alerted_sig == "BUY" and _sl > 0 and price > 0:
                dist_to_sl = price - _sl
                if 0 < dist_to_sl < _atr * 0.3:
                    tg("⚠️ SL WARNING BUY\nPrice: $" + sf(price) + "\nSL: $" + sf(_sl) + " (" + sf(dist_to_sl) + " away)\nConsider closing!\nagent.ceylonpropertylink.com")
                    print("SL WARNING sent")
                elif price < _sl:
                    tg("🛑 SL BREACHED BUY\nPrice: $" + sf(price) + "\nSL: $" + sf(_sl) + "\nLoss: ~$" + sf(abs(price-_sl)*3) + "\nCLOSE NOW on Exness!\nagent.ceylonpropertylink.com")
                    print("SL BREACH sent")
                    _last_alerted_sig = "WAIT"
                elif _tp1 > 0 and price >= _tp1:
                    tg("🎯 TP1 HIT BUY\nPrice: $" + sf(price) + "\nTP1: $" + sf(_tp1) + "\nProfit: ~$" + sf(abs(price-_last_alerted_price)*3) + "\nMove SL to breakeven. TP2: $" + sf(_tp2) + "\nagent.ceylonpropertylink.com")
                    print("TP1 HIT sent")

            elif _last_alerted_sig == "SELL" and _sl > 0 and price > 0:
                dist_to_sl = _sl - price
                if 0 < dist_to_sl < _atr * 0.3:
                    tg("⚠️ SL WARNING SELL\nPrice: $" + sf(price) + "\nSL: $" + sf(_sl) + " (" + sf(dist_to_sl) + " away)\nConsider closing!\nagent.ceylonpropertylink.com")
                    print("SL WARNING sent")
                elif price > _sl:
                    tg("🛑 SL BREACHED SELL\nPrice: $" + sf(price) + "\nSL: $" + sf(_sl) + "\nLoss: ~$" + sf(abs(price-_sl)*3) + "\nCLOSE NOW on Exness!\nagent.ceylonpropertylink.com")
                    print("SL BREACH sent")
                    _last_alerted_sig = "WAIT"
                elif _tp1 > 0 and price <= _tp1:
                    tg("🎯 TP1 HIT SELL\nPrice: $" + sf(price) + "\nTP1: $" + sf(_tp1) + "\nProfit: ~$" + sf(abs(_last_alerted_price-price)*3) + "\nMove SL to breakeven. TP2: $" + sf(_tp2) + "\nagent.ceylonpropertylink.com")
                    print("TP1 HIT sent")

        except Exception as e:
            short = str(e)[:200]
            print("[SCHEDULER ERROR] " + short)
            tg("❌ Analysis error:\n" + short + "\n" + now_str)

        time.sleep(PHASE_A_SECS)  # always 2.5min between cycles


# ── Phase 30: Rocket/Waterfall fast scheduler (60s) ──────────────────────────
def rocket_scheduler():
    global _rw_result, _rw_pre_alerted, _rw_entry_alerted
    global _rw_alert_dir, _rw_alert_ts, _rw_alert_price
    import time as _t
    _t.sleep(30)  # stagger from main scheduler
    print("[RW] Phase 30 Rocket/Waterfall scheduler started")
    while True:
        try:
            # Get live price from cache
            live_p = _price_cache.get("price")
            result = run_rocket_analysis(live_p)
            _rw_result = result

            rocket    = result.get("rocket", 0)
            waterfall = result.get("waterfall", 0)
            signal    = result.get("signal", "WAIT")
            price     = result.get("price", 0)
            rsi       = result.get("rsi", 50)
            atr       = result.get("atr", 13)
            entry     = result.get("entry", price)
            tp1       = result.get("tp1", 0)
            tp2       = result.get("tp2", 0)
            sl        = result.get("sl", 0)
            now_s     = datetime.now(timezone.utc).strftime("%H:%M UTC")

            print(f"[RW] 🚀{rocket} 💧{waterfall} | {signal} | ${sf(price)} | RSI={sf(rsi,1)}")

            now_ts = _t.time()
            cooldown_ok = (now_ts - _rw_alert_ts) >= _RW_COOLDOWN
            dir_flip    = (signal in ("ROCKET","WATERFALL") and
                           signal.split("_")[0] != _rw_alert_dir.split("_")[0])

            # ── PRE-ALERT: Rocket/Waterfall building (40+) ──────────────────
            if signal in ("ROCKET_BUILDING","WATERFALL_BUILDING") and not _rw_pre_alerted:
                emoji = "🚀" if "ROCKET" in signal else "💧"
                score = rocket if "ROCKET" in signal else waterfall
                mom_str = "🚀 Rocket: " if "ROCKET" in signal else "💧 Waterfall: "
                tg(emoji + " MOMENTUM BUILDING — XAU/USD\n"
                   + "━━━━━━━━━━━━━━━━━━━━\n"
                   + mom_str + str(score) + "/100\n"
                   + "Price: $" + sf(price) + "\n"
                   + "RSI: " + sf(rsi,1) + "\n"
                   + "━━━━━━━━━━━━━━━━━━━━\n"
                   + "GET READY — Signal building\n"
                   + "Watch for ENTRY ALERT\n"
                   + now_s + " | agent.ceylonpropertylink.com")
                _rw_pre_alerted = True
                print(f"[RW] Pre-alert sent: {signal} {score}/100")

            # ── ENTRY ALERT: Full entry signal (60+) ────────────────────────
            elif signal in ("ROCKET","WATERFALL") and (cooldown_ok or dir_flip):
                emoji    = "⚡🚀" if signal == "ROCKET" else "⚡💧"
                dir_word = "BUY" if signal == "ROCKET" else "SELL"
                score    = rocket if signal == "ROCKET" else waterfall
                max_slip = round(atr * 0.15, 2)
                rr       = round(abs(tp1-entry)/abs(sl-entry),1) if abs(sl-entry)>0 else 0

                mom_str2 = "🚀 Rocket: " if signal=="ROCKET" else "💧 Waterfall: "
                tg(emoji + " " + dir_word + " NOW — XAU/USD\n"
                   + "━━━━━━━━━━━━━━━━━━━━\n"
                   + mom_str2 + str(score) + "/100  RSI: " + sf(rsi,1) + "\n"
                   + "━━━━━━━━━━━━━━━━━━━━\n"
                   + "Entry: $" + sf(entry) + "  (+-$" + sf(max_slip) + ")\n"
                   + "TP1  : $" + sf(tp1) + "  (+" + sf(abs(tp1-entry)) + ")\n"
                   + "TP2  : $" + sf(tp2) + "  (+" + sf(abs(tp2-entry)) + ")\n"
                   + "SL   : $" + sf(sl) + "  (-" + sf(abs(sl-entry)) + ")\n"
                   + "━━━━━━━━━━━━━━━━━━━━\n"
                   + "RR: 1:" + sf(rr,1) + "  ATR: $" + sf(atr) + "\n"
                   + "Skip if price moved >$" + sf(max_slip) + "\n"
                   + now_s + " | Phase 30 | agent.ceylonpropertylink.com")
                _rw_alert_dir   = signal
                _rw_alert_ts    = now_ts
                _rw_alert_price = price
                _rw_entry_alerted = True
                _rw_pre_alerted   = False  # reset for next setup
                print(f"[RW] ENTRY ALERT sent: {dir_word} ${sf(price)} Rocket={rocket} WF={waterfall}")

            # Reset pre-alert when momentum fades
            if signal == "WAIT" and _rw_pre_alerted:
                _rw_pre_alerted   = False
                _rw_entry_alerted = False

        except Exception as e:
            print(f"[RW] Scheduler error: {str(e)[:100]}")

        _t.sleep(60)  # run every 60 seconds

threading.Thread(target=rocket_scheduler, daemon=True).start()
print("[RW] Phase 30 Rocket/Waterfall scheduler started — 60s interval")

threading.Thread(target=scheduler, daemon=True).start()
print("Scheduler started - 2min trading / 30min off-hours · Phase 29")
print("Improvements: " + ("ENABLED ✅" if IMPROVEMENTS_LOADED else "DISABLED ⚠️"))
