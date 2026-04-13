"""
XAU/USD Triple Brain Agent — Railway Deployment
Phase 27 — Fixed: NoneType format crash, safe helpers, session-aware Telegram
"""
import os, json, threading, time, traceback
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

api = FastAPI(title="XAU/USD Triple Brain Agent v4")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "")
RUN_INTERVAL   = int(os.environ.get("RUN_INTERVAL", "300"))
PKL_ID         = os.environ.get("MODEL_PKL_ID",  "")
JSON_ID        = os.environ.get("MODEL_JSON_ID", "")
TPSL_ID        = os.environ.get("TPSL_JSON_ID",  "")
TD_KEY         = "2c3dff7091284f92b2361649006448a8"
MODELS_DIR     = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── SAFE NUMBER HELPERS — fixes NoneType.__format__ crash ────
def sf(v, d=2):
    try:
        if v is None: return "—"
        f = float(v)
        return "—" if f != f else f"{f:.{d}f}"
    except: return "—"

def sn(v, fb=0):
    try:
        if v is None: return fb
        f = float(v)
        return fb if f != f else f
    except: return fb

def clean(obj):
    if isinstance(obj, dict):        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [clean(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, float) and obj != obj: return None
    return obj

def download_file(file_id, dest):
    if os.path.exists(dest): print(f"OK: {os.path.basename(dest)}"); return True
    if not file_id or "PASTE" in file_id: print(f"No ID: {os.path.basename(dest)}"); return False
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        s = req.Session(); r = s.get(url, stream=True, timeout=60)
        for k,v in r.cookies.items():
            if k.startswith("download_warning"):
                r = s.get(f"{url}&confirm={v}", stream=True, timeout=60); break
        r.raise_for_status()
        with open(dest,"wb") as f:
            for chunk in r.iter_content(32768):
                if chunk: f.write(chunk)
        print(f"Downloaded: {os.path.basename(dest)} ({os.path.getsize(dest):,}b)")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(dest): os.remove(dest)
        return False

download_file(PKL_ID,  f"{MODELS_DIR}/brain1_xgboost_v4.pkl")
download_file(JSON_ID, f"{MODELS_DIR}/brain1_features_v4.json")
download_file(TPSL_ID, f"{MODELS_DIR}/dynamic_tpsl_config.json")

_agent_loaded = False
try:
    code = open("/app/trading_agent.py").read()
    code = code.replace('"/content/drive/MyDrive/trading_agent/', f'"{MODELS_DIR}/')
    exec(code, globals())
    _agent_loaded = True
    print("trading_agent.py loaded OK")
except Exception as e:
    print(f"trading_agent.py FAILED: {e}")

def tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT: return False
    try:
        r = req.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id":TELEGRAM_CHAT,"text":msg,"parse_mode":"HTML"}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}"); return False

def get_session():
    t = datetime.now(timezone.utc).hour*60 + datetime.now(timezone.utc).minute
    if 480<=t<780:   return "LONDON SESSION",        "MEDIUM", True,  "🟢"
    if 780<=t<1020:  return "LONDON/NY OVERLAP",     "HIGH",   True,  "🔥"
    if 1020<=t<1320: return "NEW YORK SESSION",      "MEDIUM", True,  "🟡"
    return "ASIAN SESSION", "LOW", False, "🔵"

last_result = {}; last_run_time = ""; _prev_health = {}; _run_count = 0

def _save(r):
    global last_result, last_run_time
    last_result = r; last_run_time = datetime.now(timezone.utc).isoformat()

def _health_check(result):
    global _prev_health
    for k, s in result.get("health",{}).items():
        was = _prev_health.get(k,{}).get("ok",True)
        now_ok = s.get("ok",True)
        if was and not now_ok: tg(f"FAULT: {k} — {s.get('msg','')}")
        elif not was and now_ok: tg(f"RECOVERED: {k}")
    _prev_health = {k:dict(v) for k,v in result.get("health",{}).items()}

@api.get("/")
def root():
    sn2,sq,_,si = get_session()
    return {"agent":"XAU/USD Triple Brain v4","status":"running",
            "session":sn2,"session_quality":sq,"session_icon":si,
            "time":datetime.now(timezone.utc).isoformat(),
            "last_run":last_run_time or "never","run_count":_run_count}

@api.get("/health")
def health():
    return {"status":"ok","agent_loaded":_agent_loaded,"last_run":last_run_time or "never",
            "last_signal":last_result.get("signal","none"),"run_count":_run_count}

@api.get("/status")
def status():
    sname,sq,stok,si = get_session()
    if not last_result:
        return JSONResponse(content={"signal":"WAIT","confidence":0,"price":0,
            "session":sname,"session_quality":sq,"session_trade_ok":stok,"session_icon":si,
            "msg":f"No analysis yet — runs every {RUN_INTERVAL//60}min","last_run":last_run_time or "never",
            "health":{"data_feed":{"ok":_agent_loaded,"msg":"agent "+("ok" if _agent_loaded else "FAILED")},
                "brain1":{"ok":False,"msg":"pending"},"brain2":{"ok":False,"msg":"pending"},
                "brain3":{"ok":False,"msg":"pending"},"h4":{"ok":False,"msg":"pending"},
                "telegram":{"ok":bool(TELEGRAM_TOKEN),"msg":"ok" if TELEGRAM_TOKEN else "no token"},
                "daily_limit":{"ok":True,"msg":"ok"}}})
    r = dict(last_result)
    r["last_run"] = last_run_time
    r.setdefault("session", sname); r["session_quality"] = sq
    r["session_trade_ok"] = stok;   r["session_icon"]    = si
    return JSONResponse(content=clean(r))

@api.post("/run-analysis")
def run_endpoint():
    if not _agent_loaded:
        return JSONResponse(content={"signal":"ERROR","error":"agent not loaded"},status_code=500)
    try:
        r = run_analysis(); _save(r); _health_check(r)
        return JSONResponse(content=clean(r))
    except Exception as e:
        tb2 = traceback.format_exc()
        print(f"run-analysis error: {e}\n{tb2}")
        return JSONResponse(content={"signal":"ERROR","error":str(e)},status_code=500)

@api.get("/chart-data")
def chart():
    try:
        r = req.get("https://api.twelvedata.com/time_series",
            params={"symbol":"XAU/USD","interval":"15min","outputsize":200,
                    "apikey":TD_KEY,"timezone":"UTC","format":"JSON"},timeout=30)
        data = r.json()
        if "values" not in data: raise ValueError(data.get("message","no data"))
        bars = []
        for v in reversed(data["values"]):
            try: bars.append({"time":int(pd.Timestamp(v["datetime"]).timestamp()),
                "open":round(float(v["open"]),2),"high":round(float(v["high"]),2),
                "low":round(float(v["low"]),2),"close":round(float(v["close"]),2)})
            except: pass
        return JSONResponse(content={"bars":bars,"symbol":"XAU/USD","tf":"15m"})
    except Exception as e:
        return JSONResponse(content={"error":str(e),"bars":[]},status_code=500)

@api.post("/telegram/send")
def tg_send(body: dict):
    ok = tg(body.get("message","Test"))
    return {"ok":ok} if ok else JSONResponse(content={"ok":False,"msg":"send failed"},status_code=500)

@api.get("/files")
def files():
    fs = {}
    for n in ["brain1_xgboost_v4.pkl","brain1_features_v4.json","dynamic_tpsl_config.json"]:
        p = f"{MODELS_DIR}/{n}"
        fs[n] = {"exists":os.path.exists(p),"kb":round(os.path.getsize(p)/1024,1) if os.path.exists(p) else 0}
    return {"files":fs,"RUN_INTERVAL":RUN_INTERVAL,"telegram":"SET" if TELEGRAM_TOKEN else "NOT SET"}

# ── SCHEDULER ─────────────────────────────────────────────────
def scheduler():
    global _run_count
    time.sleep(20)
    sname,_,_,_ = get_session()
    tg(f"AGENT STARTED\nInterval: {RUN_INTERVAL//60}min\nSession: {sname}\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}\nDashboard: ceylonpropertylink.com/agent/dashboard.html")

    while True:
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        sname2, sq2, stok2, _ = get_session()
        print(f"[{now}] Running analysis | {sq2} session | trade_ok={stok2}")
        try:
            result = run_analysis()
            _run_count += 1
            _save(result)
            _health_check(result)

            sig   = str(result.get("signal") or "WAIT").upper()
            conf  = sn(result.get("confidence"), 0)
            price = sn(result.get("price"),       0)
            b1    = sn((result.get("brain1") or {}).get("probability"), 0)
            b2    = sn((result.get("brain2") or {}).get("score"),       0)
            tp    = sn(result.get("tp"), 0)
            sl    = sn(result.get("sl"), 0)
            h4v   = bool(result.get("h4_veto", False))
            sb    = bool(result.get("session_blocked", False))

            print(f"  {sig} | conf={sf(conf,1)} | B1={sf(b1,3)} | B2={sf(b2,1)} | ${sf(price)}")

            if sig in ("BUY","SELL"):
                rr = abs(tp-price)/abs(sl-price) if sl and price and abs(sl-price)>0 else 0
                em = "📈" if sig=="BUY" else "📉"
                tg(f"{em} <b>{sig} — XAU/USD</b>\n\n"
                   f"Confidence: <b>{sf(conf,1)}/10</b>\n"
                   f"Price: <b>${sf(price)}</b>\n"
                   f"Take Profit: ${sf(tp)}\n"
                   f"Stop Loss: ${sf(sl)}\n"
                   f"RR: 1:{sf(rr,1)}\n\n"
                   f"B1:{sf(b1,3)} B2:{sf(b2,1)}/10 B3:{sf(conf,1)}/10\n"
                   f"{sname2} | {now}\n\n"
                   f"Open trade on Exness NOW!\n"
                   f"XAU/USD Triple Brain v4")
            elif _run_count % 6 == 0:
                why = "setup not ready"
                if sb:        why = f"session blocked ({sq2})"
                elif h4v:     why = "H4 veto active"
                elif b1<0.60: why = f"B1={sf(b1,3)} (need 0.600)"
                elif b2<5:    why = f"B2={sf(b2,1)}/10 (need 5.0)"
                elif conf<6.5:why = f"B3={sf(conf,1)}/10 (need 6.5)"
                tg(f"Status: WAIT — {why}\nB1:{sf(b1,3)} B2:{sf(b2,1)}/10 B3:{sf(conf,1)}/10\n${sf(price)} | {sname2} | {now}")

        except Exception as e:
            tb2 = traceback.format_exc()
            short = str(e)[:250]
            print(f"[SCHEDULER ERROR] {short}\n{tb2[:300]}")
            tg(f"Analysis error: {short}\n{now}\nCheck Railway logs for details.")
        time.sleep(RUN_INTERVAL)

threading.Thread(target=scheduler, daemon=True).start()
print(f"Scheduler started — every {RUN_INTERVAL}s ({RUN_INTERVAL//60}min)")
