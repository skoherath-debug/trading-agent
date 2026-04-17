"""
main.py — Phase 30 (PATCHED 2026-04-17)

Fixes in this build:
  [FIX-A] Added defensive module-level _rw_result = None init (was NameError
          on /rocket-status endpoint before first scheduler tick)
  [FIX-B] _pre_signal now correctly produces BUY / SELL / WAIT (was BUY/WAIT only)
  [FIX-C] rocket_scheduler now uses canonical names from trading_agent.py:
          _rw_alert_time (not _rw_alert_ts), _RW_ENTRY_COOLDOWN (not _RW_COOLDOWN),
          _rw_alerted (not _rw_entry_alerted). Prevents NameError on first iteration.
  [FIX-D] mins_since debug output now computed BEFORE timestamp reset (was always 0)
  [ADD-A] Added simple heartbeat endpoint & improved /status fallback messaging
"""

import os, threading, time, traceback
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Phase 28/29: Improvements ─────────────────────────────────
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
api = FastAPI(title="XAU/USD Triple Brain Agent · Phase 30")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if IMPROVEMENTS_LOADED:
    api.include_router(improvements_router)

# ── Env ───────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "")
RUN_INTERVAL   = int(os.environ.get("RUN_INTERVAL", "300"))
PKL_ID         = os.environ.get("MODEL_PKL_ID",  "")
JSON_ID        = os.environ.get("MODEL_JSON_ID", "")
TPSL_ID        = os.environ.get("TPSL_JSON_ID",  "")
TD_KEY         = os.environ.get("TD_KEY",         "")
MODELS_DIR     = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# [FIX-A] DEFENSIVE INIT — will be overridden when trading_agent.py exec's,
# but prevents NameError if /rocket-status is called before first scheduler tick.
_rw_result        = None
_rw_pre_alerted   = False
_rw_alerted       = False
_rw_alert_dir     = "WAIT"
_rw_alert_time    = 0.0
_rw_alert_price   = 0.0
_RW_ENTRY_COOLDOWN= 600

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
    sig    = result.get("signal","WAIT")
    sig_px = sn(result.get("price"), 0)
    atr    = sn(result.get("atr"), 13)
    tp1    = sn(result.get("tp"),  0)
    sl     = sn(result.get("sl"),  0)
    h4     = result.get("h4") or {}
    ema20  = sn(h4.get("ema20"), 0)

    if sig not in ("BUY","SELL") or sig_px == 0:
        return None

    zone_size   = round(atr * 0.4, 2)
    zone_low    = round(sig_px - zone_size, 2)
    zone_high   = round(sig_px + zone_size, 2)
    limit_offset = round(atr * 0.2, 2)
    if sig == "BUY":
        limit_price = round(sig_px - limit_offset, 2)
        in_zone     = zone_low <= current_price <= zone_high
        price_ok    = current_price <= sig_px + zone_size
        confirm_lvl = round(max(sig_px, ema20) + 0.5, 2) if ema20 else sig_px
    else:
        limit_price = round(sig_px + limit_offset, 2)
        in_zone     = zone_low <= current_price <= zone_high
        price_ok    = current_price >= sig_px - zone_size
        confirm_lvl = round(min(sig_px, ema20) - 0.5, 2) if ema20 else sig_px

    slippage    = round(abs(current_price - sig_px), 2)
    slippage_pct= round(slippage / atr * 100, 1)

    if slippage <= atr * 0.1:    quality = 10
    elif slippage <= atr * 0.2:  quality = 8
    elif slippage <= atr * 0.4:  quality = 6
    elif slippage <= atr * 0.6:  quality = 4
    else:                         quality = 0

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
    traceback.print_exc()

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
    elif 420 <= m < 780: return "LONDON SESSION",   "HIGH",   True,  "🟢"
    elif 1020 <= m < 1080: return "NEW YORK SESSION","MEDIUM", True,  "🟡"
    else:                  return "ASIAN/OFF-HOURS", "LOW",    False, "🔵"

PHASE_A_SECS   = 150
PHASE_B_SECS   = 300
OFF_HOURS_SECS = 1800

def get_run_interval():
    m = datetime.now(timezone.utc).hour * 60 + datetime.now(timezone.utc).minute
    return OFF_HOURS_SECS if not (420 <= m < 1080) else PHASE_A_SECS

# ── Shared state ──────────────────────────────────────────────
last_result   = {}
last_run_time = ""
_prev_health  = {}
_run_count    = 0
_last_alerted_sig   = "WAIT"
_pre_signal         = "WAIT"
_pre_b1             = 0.0
_pre_b2             = 0.0
_pre_time           = 0.0
_phase_a_count      = 0
_last_alerted_price = 0.0
_last_alerted_time  = 0.0
ALERT_COOLDOWN_SECS = 1800
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
    return {"agent":"XAU/USD Triple Brain · Phase 30","status":"running",
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
            "msg":"Agent warming up — first run in ~2.5min","health":{
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
    if not last_result:
        return {"error": "No signal yet"}
    try:
        cur = sn(_price_cache.get("price"), sn(last_result.get("price"), 0))
        analysis = get_entry_analysis(last_result, cur)
        if not analysis:
            return {"signal": "WAIT", "msg": "No active signal"}
        return analysis
    except Exception as e:
        return {"error": str(e)}

@api.get("/rocket-status")
def rocket_ep():
    """Phase 30 — latest Rocket/Waterfall micro-entry status. [FIX-A] _rw_result pre-initialized."""
    try:
        res = _rw_result  # now safe — pre-initialized at module level
        if res:
            return JSONResponse(content=clean(res))
    except NameError:
        pass
    return JSONResponse(content={
        "rocket":0, "waterfall":0, "signal":"WAIT",
        "msg":"Warming up — first 1m scan in <60s",
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        "🚀 AGENT STARTED · Phase 30\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Schedule: 2.5min/5min dual-phase\n"
        "Rocket scanner: 60s (1m bars)\n"
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
            is_phase_b = (_phase_a_count % 2 == 0)

            # ── PHASE A ───────────────────────────────────────────
            if not is_phase_b:
                print("[PHASE A] Pre-computing B1+B2...")
                try:
                    result_a = run_analysis()
                    _pre_b1  = sn((result_a.get("brain1") or {}).get("probability"), 0)
                    _pre_b2  = sn((result_a.get("brain2") or {}).get("score"), 0)
                    _pre_time= time.time()

                    # [FIX-B] Properly detect BUY / SELL / WAIT direction
                    if _pre_b1 >= 0.600 and _pre_b2 >= 5.0:
                        _trend_raw = (result_a.get("brain2",{}).get("details",{}) or {}).get("trend","")
                        _trend_up  = str(_trend_raw).upper()
                        if "BULL" in _trend_up:
                            _pre_signal = "BUY"
                        elif "BEAR" in _trend_up:
                            _pre_signal = "SELL"
                        else:
                            _pre_signal = "WAIT"
                    else:
                        _pre_signal = "WAIT"

                    _save_result(result_a)
                    price_a = sn(result_a.get("price"), 0)
                    print("  [A] B1=" + sf(_pre_b1,3) + " B2=" + sf(_pre_b2,1) + " Pre=" + _pre_signal + " $" + sf(price_a))
                except Exception as ea:
                    print("  [A] Pre-analysis error: " + str(ea)[:100])
                time.sleep(PHASE_A_SECS)
                continue

            # ── PHASE B ───────────────────────────────────────────
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

            phase_consistent = (
                (_pre_signal == sig) or
                (sig in ("BUY","SELL") and abs(time.time()-_pre_time) < 400)
            )

            if sig in ("BUY","SELL") and not phase_consistent:
                print("  [B] SUPPRESSED — Phase A=" + _pre_signal + " Phase B=" + sig + " (inconsistent)")
                sig = "WAIT"

            print("  [B] " + sig + " | B1=" + sf(b1,3) + " B2=" + sf(b2,1) + " B3=" + b3v + " conf=" + sf(conf,1) + " $" + sf(price))

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

            # ── Alert gating ──────────────────────────────────────
            global _last_alerted_sig, _last_alerted_price, _last_alerted_time
            now_ts       = time.time()
            cooldown_ok  = (now_ts - _last_alerted_time) >= ALERT_COOLDOWN_SECS
            dir_changed  = sig != _last_alerted_sig
            price_moved  = abs(price - _last_alerted_price) > 8.0

            session_quality_ok = sq2 in ("HIGH", "MEDIUM")
            rocket_score    = sn((result.get("brain2") or {}).get("details", {}).get("rocket_score"), 0)
            waterfall_score = sn((result.get("brain2") or {}).get("details", {}).get("waterfall_score"), 0)
            momentum_ok = (
                (sig == "BUY"  and rocket_score    >= 50) or
                (sig == "SELL" and waterfall_score >= 50)
            )

            should_alert = (cooldown_ok and session_quality_ok and momentum_ok) or dir_changed

            if sig in ("BUY", "SELL") and should_alert:
                # [FIX-D] Compute mins_since BEFORE resetting timestamp
                mins_since = round((now_ts - _last_alerted_time)/60, 1) if _last_alerted_time else 0
                _last_alerted_sig   = sig
                _last_alerted_price = price
                _last_alerted_time  = now_ts
                print("   📱 ALERT: " + sig + " (last alert " + str(mins_since) + "m ago)")
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

            if sig == "WAIT" and _last_alerted_sig in ("BUY","SELL"):
                _last_alerted_sig   = "WAIT"
                _last_alerted_price = 0.0

            # ── SL / TP price alerts ──────────────────────────────
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
            traceback.print_exc()
            tg("❌ Analysis error:\n" + short + "\n" + now_str)

        time.sleep(PHASE_A_SECS)


# ═══════════════════════════════════════════════════════════════
# [FIX-C] Rocket/Waterfall scheduler — now uses CANONICAL names
# from trading_agent.py, no more _rw_alert_ts / _RW_COOLDOWN / _rw_entry_alerted
# ═══════════════════════════════════════════════════════════════
def rocket_scheduler():
    global _rw_result, _rw_pre_alerted, _rw_alerted
    global _rw_alert_dir, _rw_alert_time, _rw_alert_price
    import time as _t
    _t.sleep(30)
    print("[RW] Phase 30 Rocket/Waterfall scheduler started (1m bars, 60s tick)")
    while True:
        try:
            # Get fresh price — if cache is older than 15s, fetch a new one directly
            live_p = _price_cache.get("price")
            cache_age = _t.time() - _price_cache.get("ts", 0)
            if not live_p or cache_age > 15:
                try:
                    _k = get_td_key()
                    _r = req.get("https://api.twelvedata.com/price",
                                 params={"symbol":"XAU/USD","apikey":_k}, timeout=5)
                    _p = float(_r.json().get("price", 0))
                    if _p > 100:
                        live_p = round(_p, 2)
                        _price_cache["price"] = live_p
                        _price_cache["ts"]    = _t.time()
                except Exception:
                    pass

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
            cooldown_ok = (now_ts - _rw_alert_time) >= _RW_ENTRY_COOLDOWN
            dir_flip    = (signal in ("ROCKET","WATERFALL") and
                           signal.split("_")[0] != _rw_alert_dir.split("_")[0])

            # PRE-ALERT (building)
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

            # ENTRY ALERT
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
                   + "Entry: $" + sf(entry) + "  (±$" + sf(max_slip) + ")\n"
                   + "TP1  : $" + sf(tp1) + "  (+" + sf(abs(tp1-entry)) + ")\n"
                   + "TP2  : $" + sf(tp2) + "  (+" + sf(abs(tp2-entry)) + ")\n"
                   + "SL   : $" + sf(sl) + "  (-" + sf(abs(sl-entry)) + ")\n"
                   + "━━━━━━━━━━━━━━━━━━━━\n"
                   + "RR: 1:" + sf(rr,1) + "  ATR: $" + sf(atr) + "\n"
                   + "Skip if price moved >$" + sf(max_slip) + "\n"
                   + now_s + " | Phase 30 | agent.ceylonpropertylink.com")
                _rw_alert_dir   = signal
                _rw_alert_time  = now_ts
                _rw_alert_price = price
                _rw_alerted     = True
                _rw_pre_alerted = False
                print(f"[RW] ENTRY ALERT sent: {dir_word} ${sf(price)} R={rocket} W={waterfall}")

            if signal == "WAIT" and _rw_pre_alerted:
                _rw_pre_alerted = False
                _rw_alerted     = False

        except Exception as e:
            print(f"[RW] Scheduler error: {str(e)[:150]}")
            traceback.print_exc()

        _t.sleep(60)

threading.Thread(target=rocket_scheduler, daemon=True).start()
print("[RW] Phase 30 Rocket/Waterfall scheduler thread launched — 60s interval")

threading.Thread(target=scheduler, daemon=True).start()
print("Scheduler started — 2.5/5min dual-phase · Phase 30")
print("Improvements: " + ("ENABLED ✅" if IMPROVEMENTS_LOADED else "DISABLED ⚠️"))
