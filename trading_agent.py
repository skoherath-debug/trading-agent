"""
trading_agent.py — Phase 30 (PATCHED 2026-04-17)

Fixes in this build:
  [FIX-1] Removed broken Rocket/Waterfall gate that referenced `sig` before definition
          — this was a NameError bomb every time fused >= 6.3
  [FIX-2] Consolidated three duplicate gate blocks into ONE (after sig is set)
  [FIX-3] Added `global _td_key_index` so the key rotation actually persists
  [FIX-4] b2_dir now handles BUY / SELL / WAIT cleanly (was BUY/WAIT only)
  [ADD-1] Switched Phase 30 to 1-minute bars (was 5min) for faster rocket detection
  [ADD-2] Exposed _RW_COOLDOWN / _rw_alert_ts / _rw_entry_alerted / _rw_result aliases
          so main.py's name mismatches don't crash the rocket scheduler
  [ADD-3] rocket-status returns signal_time so frontend can show signal age
"""

import os, pickle, json, requests, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timezone, time as _time

# ── Env vars ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8172828888:AAFWCvtCl1F-Kj5yOv_EFEB9vxL-ir-dD9I")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "7132630179")

_GROQ_KEYS = [k for k in [
    os.environ.get("GROQ_API_KEY",  "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"),
    os.environ.get("GROQ_API_KEY2", "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"),
    os.environ.get("GROQ_API_KEY3", "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"),
] if k.strip()]
_groq_key_index = 0
GROQ_API_KEY = _GROQ_KEYS[0]

_TD_KEYS = [k for k in [
    os.environ.get("TD_KEY",  "f3883b7831a540cda02cfafcfe77e082"),
    os.environ.get("TD_KEY2", "41c8cfdf490b4bf4a0d388e716a32453"),
    os.environ.get("TD_KEY3", "f58b0a482f1443e78fb23cf8975b44d9"),
] if k.strip()]
_td_key_index = 0

def get_td_key():
    global _td_key_index
    if not _TD_KEYS: return ""
    return _TD_KEYS[_td_key_index % len(_TD_KEYS)]

def rotate_td_key(reason=""):
    global _td_key_index
    if len(_TD_KEYS) <= 1: return
    _td_key_index += 1
    print(f"[TD KEY] Rotated to key {(_td_key_index % len(_TD_KEYS)) + 1}/{len(_TD_KEYS)} — {reason}")

TD_KEY = get_td_key()

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]
_groq_model_index = 0
GROQ_MODEL = GROQ_MODELS[0]
SYMBOL = "GC=F"

def _to_python(obj):
    if isinstance(obj, dict):          return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):          return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, np.bool_):      return bool(obj)
    return obj

def calculate_atr_dynamic(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_dynamic_tpsl(df, direction, entry_price):
    try:
        with open("dynamic_tpsl_config.json") as f:
            cfg = json.load(f)
    except Exception:
        return 15.0, 10.0, entry_price + 5, entry_price - 3.33
    atr_val = float(calculate_atr_dynamic(df, cfg["atr_period"]).iloc[-1])
    if pd.isna(atr_val): atr_val = 16.78
    tp_pts = atr_val * cfg["tp_atr_mult"]
    sl_pts = atr_val * cfg["sl_atr_mult"]
    px     = cfg["px_to_usd"]
    tp_usd = round(min(max(tp_pts * px, cfg["tp_clamp"][0]), cfg["tp_clamp"][1]), 2)
    sl_usd = round(min(max(sl_pts * px, cfg["sl_clamp"][0]), cfg["sl_clamp"][1]), 2)
    if direction == "BUY":
        tp_level = round(entry_price + tp_pts, 2)
        sl_level = round(entry_price - sl_pts, 2)
    else:
        tp_level = round(entry_price - tp_pts, 2)
        sl_level = round(entry_price + sl_pts, 2)
    return tp_usd, sl_usd, tp_level, sl_level

_BLOCK_START = _time(21, 0)
_BLOCK_END   = _time(1,  0)

def _is_session_blocked(utc_dt=None):
    if utc_dt is None: utc_dt = datetime.now(timezone.utc)
    t = utc_dt.time().replace(second=0, microsecond=0)
    return t >= _BLOCK_START or t < _BLOCK_END

def _get_session_quality(utc_dt=None):
    if utc_dt is None: utc_dt = datetime.now(timezone.utc)
    if _is_session_blocked(utc_dt): return "BLOCKED", False
    h = utc_dt.hour
    london   = 7  <= h < 16
    new_york = 13 <= h < 21
    if london and new_york: return "HIGH (London/NY overlap)", True
    elif london or new_york: return "MEDIUM", True
    else: return "LOW (no major session)", False

def _get_h4_trend(df_h1):
    try:
        df = df_h1.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        h4 = df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        if len(h4) < 55:
            return {"direction":"NEUTRAL","reason":"insufficient H4 data","score":5,
                    "ema20":None,"ema50":None,"rsi":None,"atr":None}
        c = h4["close"]
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1]
        tr    = pd.concat([h4["high"]-h4["low"],(h4["high"]-h4["close"].shift()).abs(),(h4["low"]-h4["close"].shift()).abs()],axis=1).max(axis=1)
        atr   = tr.rolling(14).mean().iloc[-1]
        price = c.iloc[-1]; e20 = ema20.iloc[-1]; e50 = ema50.iloc[-1]; e20_p = ema20.iloc[-2]
        bull_pts = sum([price > e20, e20 > e50, e20 > e20_p, rsi > 50])
        if bull_pts >= 3:   direction, score, reason = "BUY",  8 if bull_pts==4 else 7, f"H4 bullish ({bull_pts}/4 conditions)"
        elif bull_pts <= 1: direction, score, reason = "SELL", 2 if bull_pts==0 else 3, f"H4 bearish ({bull_pts}/4 conditions)"
        else:               direction, score, reason = "NEUTRAL", 5, f"H4 neutral ({bull_pts}/4 conditions)"
        return {"direction":direction,"score":score,"reason":reason,
                "ema20":round(e20,2),"ema50":round(e50,2),"rsi":round(rsi,2),
                "atr":round(atr,2),"bull_pts":bull_pts,"candles":len(h4)}
    except Exception as e:
        return {"direction":"NEUTRAL","reason":f"H4 error: {e}","score":5,
                "ema20":None,"ema50":None,"rsi":None,"atr":None}

_DAILY_PNL_PATH    = "daily_pnl.json"
_DAILY_LOSS_LIMIT  = 30.0
_MAX_TRADES_PER_DAY = 6

def _load_daily_pnl():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with open(_DAILY_PNL_PATH) as f: data = json.load(f)
        if data.get("date") != today: raise ValueError("stale")
        return data
    except Exception:
        return {"date": today, "trades": [], "total_pnl": 0.0, "trade_count": 0}

def _save_daily_pnl(data):
    try:
        with open(_DAILY_PNL_PATH, "w") as f: json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[DAILY PNL] Save error: {e}")

def _check_daily_limit():
    pnl        = _load_daily_pnl()
    total_loss = abs(min(pnl["total_pnl"], 0))
    count      = pnl["trade_count"]
    if total_loss >= _DAILY_LOSS_LIMIT:
        return True, f"Daily loss limit hit (${total_loss:.2f}/${_DAILY_LOSS_LIMIT})", pnl
    if count >= _MAX_TRADES_PER_DAY:
        return True, f"Max trades reached ({count}/{_MAX_TRADES_PER_DAY})", pnl
    return False, f"OK — loss=${total_loss:.2f}/${_DAILY_LOSS_LIMIT} | trades={count}/{_MAX_TRADES_PER_DAY}", pnl

def record_trade_result(signal, entry, exit_price, lot=0.03, px_to_usd=3.0):
    pts     = (exit_price - entry) if signal == "BUY" else (entry - exit_price)
    pnl_usd = round(pts * px_to_usd, 2)
    pnl_data = _load_daily_pnl()
    pnl_data["trades"].append({"time":datetime.now(timezone.utc).isoformat(),
                                "signal":signal,"entry":entry,"exit":exit_price,"pnl":pnl_usd})
    pnl_data["total_pnl"]    = round(pnl_data["total_pnl"] + pnl_usd, 2)
    pnl_data["trade_count"] += 1
    _save_daily_pnl(pnl_data)
    icon = "✅" if pnl_usd >= 0 else "❌"
    print(f"[TRADE RESULT] {icon} {signal} P&L=${pnl_usd:+.2f} | Day total=${pnl_data['total_pnl']:+.2f}")
    return pnl_data

def get_daily_summary():
    pnl = _load_daily_pnl()
    blocked, reason, _ = _check_daily_limit()
    wins   = sum(1 for t in pnl["trades"] if t["pnl"] > 0)
    losses = sum(1 for t in pnl["trades"] if t["pnl"] < 0)
    return {"date":pnl["date"],"total_pnl":pnl["total_pnl"],"trade_count":pnl["trade_count"],
            "wins":wins,"losses":losses,"win_rate":round(wins/max(pnl["trade_count"],1)*100,1),
            "limit_usd":_DAILY_LOSS_LIMIT,"max_trades":_MAX_TRADES_PER_DAY,
            "blocked":blocked,"status":reason,"trades":pnl["trades"]}

_sr_flip_state = {"last_support": None, "alerted": False}

def check_sr_flip(price, support, resistance, atr):
    prev = _sr_flip_state["last_support"]
    if prev is not None and abs(support - prev) > 1.0:
        _sr_flip_state["alerted"] = False
    _sr_flip_state["last_support"] = support
    if price < support and not _sr_flip_state["alerted"]:
        _sr_flip_state["alerted"] = True
        msg = (
            f"⚠️ S/R FLIP ALERT\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Support ${support:.2f} broken\n"
            f"Now acting as RESISTANCE\n"
            f"Price  : ${price:.2f}\n"
            f"New est support: ~${support - atr:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⛔ Avoid BUY until price reclaims ${support:.2f}"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": msg},
                timeout=10
            )
            print(f"🚨 S/R flip alert sent — ${support:.2f} broken")
        except Exception as e:
            print(f"[SR FLIP] Telegram error: {e}")

class Brain2Analyzer:
    def analyze_trend(self, df):
        close=df["close"]; ema20=close.ewm(span=20,adjust=False).mean().iloc[-1]
        ema50=close.ewm(span=50,adjust=False).mean().iloc[-1]
        ema200=close.ewm(span=200,adjust=False).mean().iloc[-1]; price=close.iloc[-1]
        bs=sum([price>ema20,ema20>ema50,ema50>ema200])
        if bs>=3: d,s="BULLISH","STRONG" if bs==3 else "MODERATE"
        elif bs==0: d,s="BEARISH","STRONG"
        elif bs==1: d,s="BEARISH","MODERATE"
        else: d,s="NEUTRAL","CHOPPY"
        return {"direction":d,"strength":s,"ema20":round(ema20,2),"ema50":round(ema50,2),
                "ema200":round(ema200,2),"score":(bs/3)*10}

    def analyze_structure(self, df, lookback=20):
        highs=df["high"].values[-lookback:]; lows=df["low"].values[-lookback:]
        sh=[highs[i] for i in range(1,len(highs)-1) if highs[i]>highs[i-1] and highs[i]>highs[i+1]]
        sl=[lows[i]  for i in range(1,len(lows)-1)  if lows[i]<lows[i-1]  and lows[i]<lows[i+1]]
        structure,bos="RANGING",False
        if len(sh)>=2 and len(sl)>=2:
            if sh[-1]>sh[-2] and sl[-1]>sl[-2]: structure="UPTREND"
            elif sh[-1]<sh[-2] and sl[-1]<sl[-2]: structure="DOWNTREND"
            else: structure,bos="CHOPPY",True
        return {"structure":structure,"break_of_structure":bos,
                "swing_highs":[round(h,2) for h in sh[-3:]],"swing_lows":[round(l,2) for l in sl[-3:]],
                "score":8 if structure=="UPTREND" else 2 if structure=="DOWNTREND" else 5}

    def analyze_session(self, df):
        hour=datetime.now(timezone.utc).hour; active=[]
        if 7<=hour<16: active.append("LONDON")
        if 13<=hour<22: active.append("NEW_YORK")
        if 0<=hour<8:  active.append("ASIA")
        overlap=len(active)>1
        rr=(df["high"]-df["low"]).iloc[-1]; ar=(df["high"]-df["low"]).iloc[-20:].mean()
        return {"active_sessions":active,"overlap":overlap,
                "liquidity":"HIGH" if overlap else "MEDIUM" if active else "LOW",
                "candle_range_ratio":round(rr/ar if ar>0 else 1.0,2),
                "score":8 if overlap else 5 if active else 3}

    def analyze_momentum(self, df):
        close=df["close"]; delta=close.diff()
        gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
        rsi=(100-(100/(1+gain/(loss+1e-9)))).iloc[-1]
        macd=(close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean())
        hist=(macd-macd.ewm(span=9,adjust=False).mean())
        hist_now=hist.iloc[-1]; hist_prev=hist.iloc[-2]
        mc="BULLISH" if hist_now>0 else "BEARISH"
        rsi_series = 100-(100/(1+gain/(loss+1e-9)))
        rsi_slope = rsi_series.iloc[-1] - rsi_series.iloc[-5]
        if rsi>70: state="OVERBOUGHT"
        elif rsi<30: state="OVERSOLD"
        elif rsi>55 and mc=="BULLISH": state="BULLISH_MOMENTUM"
        elif rsi<45 and mc=="BEARISH": state="BEARISH_MOMENTUM"
        else: state="NEUTRAL"

        rocket = 0
        ema9  = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        price = close.iloc[-1]
        if rsi < 35: rocket += 25
        elif rsi < 45 and rsi_slope > 2: rocket += 15
        if hist_now > 0 and hist_now > hist_prev: rocket += 20
        if price > ema9.iloc[-1]: rocket += 15
        if ema9.iloc[-1] > ema21.iloc[-1]: rocket += 15
        if rsi_slope > 3: rocket += 15
        rocket = min(rocket, 100)
        waterfall = 0
        if rsi > 65: waterfall += 25
        elif rsi > 55 and rsi_slope < -2: waterfall += 15
        if hist_now < 0 and hist_now < hist_prev: waterfall += 20
        if price < ema9.iloc[-1]: waterfall += 15
        if ema9.iloc[-1] < ema21.iloc[-1]: waterfall += 15
        if rsi_slope < -3: waterfall += 15
        waterfall = min(waterfall, 100)

        if rocket >= 60:   score = 8
        elif rocket >= 40: score = 7
        elif waterfall >= 60: score = 2
        elif waterfall >= 40: score = 3
        elif state=="NEUTRAL": score = 5
        elif state=="OVERBOUGHT": score = 3
        else: score = 7
        return {"rsi":round(rsi,2),"macd_histogram":round(hist_now,4),"macd_cross":mc,
                "state":state,"rsi_slope":round(rsi_slope,2),
                "rocket_score":rocket,"waterfall_score":waterfall,
                "divergence_detected":False,"score":score}

    def analyze_volatility(self, df):
        h,l,c=df["high"],df["low"],df["close"]
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr=tr.rolling(14).mean().iloc[-1]; avg=tr.rolling(14).mean().iloc[-50:-1].mean()
        r=atr/avg if avg>0 else 1.0
        s="HIGH_VOLATILITY" if r>1.5 else "LOW_VOLATILITY" if r<0.7 else "NORMAL"
        return {"atr14":round(float(atr),4),"atr_ratio":round(float(r),2),"state":s,
                "suggested_sl_pips":round(float(atr),2),"suggested_tp_pips":round(float(atr*1.5),2),
                "score":7 if s=="NORMAL" else 4 if s=="HIGH_VOLATILITY" else 5}

    def analyze_sr_zones(self, df, lookback=50):
        highs=df["high"].values[-lookback:]; lows=df["low"].values[-lookback:]
        price=float(df["close"].iloc[-1])
        all_levels=sorted(set(
            [float(h) for i,h in enumerate(highs) if h==max(highs[max(0,i-3):i+4])]+
            [float(l) for i,l in enumerate(lows)  if l==min(lows[max(0,i-3):i+4])]))
        rs=[x for x in all_levels if x>price]; ss=[x for x in all_levels if x<price]
        nr=min(rs) if rs else price*1.01; ns=max(ss) if ss else price*0.99
        return {"nearest_resistance":round(nr,2),"nearest_support":round(ns,2),
                "dist_to_resistance_pct":round((nr-price)/price*100,3),
                "dist_to_support_pct":round((price-ns)/price*100,3),
                "at_key_zone":min((nr-price)/price*100,(price-ns)/price*100)<0.15,"score":5}

    def score_confluence(self, t, st, se, m, v):
        scores=[t["score"],st["score"],se["score"],m["score"],v["score"]]
        avg=sum(scores)/len(scores)
        if avg>=6.5: sig,conf="BUY","HIGH" if avg>=8 else "MEDIUM"
        elif avg<=3.5: sig,conf="SELL","HIGH" if avg<=2 else "MEDIUM"
        else: sig,conf="WAIT","LOW"
        return {"signal":sig,"confidence":conf,"confluence_score":round(avg,2),
                "module_scores":{"trend":t["score"],"structure":st["score"],
                "session":se["score"],"momentum":m["score"],"volatility":v["score"]}}

    def analyze(self, df):
        t=self.analyze_trend(df); st=self.analyze_structure(df)
        se=self.analyze_session(df); m=self.analyze_momentum(df)
        v=self.analyze_volatility(df); sr=self.analyze_sr_zones(df)
        c=self.score_confluence(t,st,se,m,v)
        return {"timestamp":datetime.now(timezone.utc).isoformat(),
                "price":round(float(df["close"].iloc[-1]),2),
                "signal":c["signal"],"confidence":c["confidence"],
                "confluence_score":c["confluence_score"],
                "modules":{"trend":t,"structure":st,"session":se,"momentum":m,"volatility":v,"sr_zones":sr},
                "module_scores":c["module_scores"]}

def build_features_for_brain1v2(df):
    d = df.copy()
    c = d["close"]; h = d["high"]; l = d["low"]; o = d["open"]
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()

    d["return_1"]  = c.pct_change(1)
    d["return_3"]  = c.pct_change(3)
    d["return_5"]  = c.pct_change(5)
    d["return_10"] = c.pct_change(10)

    for n in [5,10,20,50,100,200]:
        d[f"ema_{n}"]     = ema(c,n)
        d[f"c_vs_ema{n}"] = (c - ema(c,n)) / ema(c,n)
    for n in [10,20,50]:
        d[f"sma_{n}"]     = sma(c,n)
        d[f"c_vs_sma{n}"] = (c - sma(c,n)) / sma(c,n)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100/(1+gain/(loss+1e-9))
    for n, col in [(7,"rsi_7"),(21,"rsi_21")]:
        g  = delta.clip(lower=0).rolling(n).mean()
        ls = (-delta.clip(upper=0)).rolling(n).mean()
        d[col] = 100 - 100/(1+g/(ls+1e-9))
    d["rsi_div"] = d["rsi_14"] - d["rsi_14"].shift(5)

    d["macd"]        = ema(c,12) - ema(c,26)
    d["macd_signal"] = ema(d["macd"],9)
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    for n in [20,14]:
        mid  = sma(c,n); band = 2*std(c,n)
        d[f"bb_upper_{n}"] = mid+band
        d[f"bb_lower_{n}"] = mid-band
        d[f"bb_pct_{n}"]   = (c-(mid-band))/(2*band+1e-9)
        d[f"bb_width_{n}"] = (2*band)/(mid+1e-9)

    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["atr_7"]  = tr.rolling(7).mean()
    d["atr_21"] = tr.rolling(21).mean()

    low14=l.rolling(14).min(); high14=h.rolling(14).max()
    d["stoch_k"]  = 100*(c-low14)/(high14-low14+1e-9)
    d["stoch_d"]  = d["stoch_k"].rolling(3).mean()
    d["willr_14"] = -100*(high14-c)/(high14-low14+1e-9)

    tp2=(h+l+c)/3
    d["cci_20"] = (tp2-sma(tp2,20))/(0.015*std(tp2,20)+1e-9)

    plus_dm  = (h-h.shift()).clip(lower=0)
    minus_dm = (l.shift()-l).clip(lower=0)
    d["adx_14"] = (plus_dm.rolling(14).mean()-minus_dm.rolling(14).mean()).abs()/(tr.rolling(14).mean()+1e-9)*100

    if "volume" in d.columns and d["volume"].sum() > 0:
        d["volume_ratio"] = d["volume"]/(d["volume"].rolling(10).mean()+1e-9)
    else:
        d["volume_ratio"] = 1.0

    d["candle_body"] = (c-o).abs()/(h-l+1e-9)
    d["upper_wick"]  = (h-pd.concat([c,o],axis=1).max(axis=1))/(h-l+1e-9)
    d["lower_wick"]  = (pd.concat([c,o],axis=1).min(axis=1)-l)/(h-l+1e-9)
    d["candle_dir"]  = np.sign(c-o)

    d["hour"]     = df.index.hour
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24)
    d["dow"]      = df.index.dayofweek
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/5)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/5)

    d["mom_5"]  = c - c.shift(5)
    d["mom_10"] = c - c.shift(10)
    d["mom_20"] = c - c.shift(20)

    d["vol_ratio_5_20"] = std(c,5)/(std(c,20)+1e-9)
    d["trend_5_20"]     = (ema(c,5)-ema(c,20))/(ema(c,20)+1e-9)
    d["trend_10_50"]    = (ema(c,10)-ema(c,50))/(ema(c,50)+1e-9)
    d["pos_in_day"]     = (c-l)/(h-l+1e-9)
    d["gap"]            = (o-c.shift(1))/(c.shift(1)+1e-9)

    return d.dropna()

def send_telegram(msg_or_signal, price=None, score=None, b1_prob=None,
                  b2_score=None, atr=None, h4=None, tp=None, sl=None, tp2=None):
    if price is None:
        msg = msg_or_signal
    else:
        signal = msg_or_signal
        if signal not in ["BUY","SELL"]: return
        if tp is None or sl is None:
            tp = round(price + atr*1.5, 2) if signal=="BUY" else round(price - atr*1.5, 2)
            sl = round(price - atr,     2) if signal=="BUY" else round(price + atr,     2)
        if tp2 is None:
            tp2 = round(price + atr*2.5, 2) if signal=="BUY" else round(price - atr*2.5, 2)
        _tp_usd = round(abs(tp  - price) * 3, 2)
        _t2_usd = round(abs(tp2 - price) * 3, 2)
        _sl_usd = round(abs(sl  - price) * 3, 2)
        rr      = round(_tp_usd / _sl_usd, 2) if _sl_usd > 0 else 0
        icon    = "🟢" if signal=="BUY" else "🔴"
        h4_line = ""
        if h4:
            h4_icon = "📈" if h4.get("direction")=="BUY" else "📉" if h4.get("direction")=="SELL" else "➡️"
            h4_line = f"H4     : {h4_icon} {h4.get('direction','—')} | RSI {h4.get('rsi','—')}\n"
        msg = (
            f"{icon} {signal} XAU/USD\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Entry  : ${price:,.2f}\n"
            f"TP1    : ${tp:,.2f}  (+${round(abs(tp-price),2)})\n"
            f"TP2    : ${tp2:,.2f}  (+${round(abs(tp2-price),2)})\n"
            f"SL     : ${sl:,.2f}  (-${round(abs(sl-price),2)})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Score  : {round(score,2)}/10\n"
            f"Brain1 : {round(b1_prob,3)}\n"
            f"Brain2 : {round(b2_score,2)}/10\n"
            f"{h4_line}"
            f"ATR    : ${round(atr,2)}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Risk   : ${_sl_usd:.2f} | TP1: ${_tp_usd:.2f} | TP2: ${_t2_usd:.2f} | RR: {rr}:1\n"
            f"Open trade on Exness NOW!"
        )
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg},
            timeout=10
        )
        print("Telegram:", r.status_code)
    except Exception as e:
        print("Telegram error:", e)

def trailing_sl_check(direction, entry, current_high, current_low,
                      tp_level, sl_level, be_triggered):
    if be_triggered: return sl_level, True
    if direction == "BUY":
        half_tp = entry + (tp_level - entry) * 0.50
        if current_high >= half_tp: return entry + 0.10, True
    else:
        half_tp = entry - (entry - tp_level) * 0.50
        if current_low <= half_tp:  return entry - 0.10, True
    return sl_level, False

# ── Main analysis ─────────────────────────────────────────────────────────────
def run_analysis():
    session_quality, trade_allowed = _get_session_quality()
    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"\n🕐 Session: {now_utc} | Quality: {session_quality}")

    if not trade_allowed:
        _price = None
        try:
            import requests as _rq
            _k = get_td_key()
            _r = _rq.get("https://api.twelvedata.com/price",
                params={"symbol":"XAU/USD","apikey":_k}, timeout=10)
            _price = float(_r.json().get("price", 0)) or None
        except Exception:
            pass
        return _to_python({
            "signal":          "WAIT",
            "confidence":      0,
            "price":           _price,
            "session_blocked": True,
            "session_quality": session_quality,
            "session_trade_ok":False,
            "session_icon":    "🔴",
            "daily_pnl":       _load_daily_pnl()["total_pnl"],
            "trade_count":     _load_daily_pnl()["trade_count"],
            "health": {
                "data_feed":   {"ok":True,  "msg":"Off-hours — price only"},
                "brain1":      {"ok":True,  "msg":"Standby until 07:00 UTC"},
                "brain2":      {"ok":True,  "msg":"Standby until 07:00 UTC"},
                "brain3":      {"ok":True,  "msg":"Standby until 07:00 UTC"},
                "h4":          {"ok":True,  "msg":"Standby until 07:00 UTC"},
                "telegram":    {"ok":True,  "msg":"standby"},
                "daily_limit": {"ok":True,  "msg":"OK"},
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    day_blocked, day_reason, day_pnl = _check_daily_limit()
    if day_blocked:
        return {"signal":"WAIT","confidence":0,"price":None,
                "day_blocked":True,"day_reason":day_reason,
                "daily_pnl":day_pnl["total_pnl"],"trade_count":day_pnl["trade_count"],
                "timestamp":datetime.now(timezone.utc).isoformat()}

    health = {
        "data_feed":   {"ok":False,"msg":"not run"},
        "brain1":      {"ok":False,"msg":"not run"},
        "brain2":      {"ok":False,"msg":"not run"},
        "brain3":      {"ok":False,"msg":"not run"},
        "h4":          {"ok":False,"msg":"not run"},
        "telegram":    {"ok":True, "msg":"standby"},
        "daily_limit": {"ok":True, "msg":day_reason},
    }

    _b1_use_fallback = False
    try:
        with open("models/brain1_xgboost_v4.pkl","rb") as f:
            brain1_model = pickle.load(f)
        if not hasattr(brain1_model, "predict_proba"):
            raise ValueError(f"Loaded object is {type(brain1_model).__name__}, not XGBoost")
        with open("models/brain1_features_v4.json") as f:
            features = json.load(f)
        THRESHOLD = features.get("threshold", 0.60)
    except Exception as e:
        print(f"[B1] Load failed: {e} — using fallback")
        health["brain1"] = {"ok":False,"msg":f"Model failed: {e}"}
        _b1_use_fallback = True
        THRESHOLD = 0.60
        features  = {"features":[]}

    # [FIX-3] Declare global so key rotation persists
    global _td_key_index
    try:
        _td_df = None
        import requests as _req
        _all_keys = [k for k in [
            os.environ.get("TD_KEY",  "f3883b7831a540cda02cfafcfe77e082"),
            os.environ.get("TD_KEY2", "41c8cfdf490b4bf4a0d388e716a32453"),
            os.environ.get("TD_KEY3", "f58b0a482f1443e78fb23cf8975b44d9"),
        ] if k]
        for _ki, _active_key in enumerate(_all_keys):
            try:
                print(f"[DATA] Trying TD key {_ki+1}/{len(_all_keys)}...")
                _r = _req.get("https://api.twelvedata.com/time_series", params={
                    "symbol":"XAU/USD","interval":"1h","outputsize":500,
                    "apikey":_active_key,"timezone":"UTC","format":"JSON"
                }, timeout=30)
                _td = _r.json()
                if "code" in _td or ("message" in _td and "values" not in _td):
                    print(f"[DATA] TD key {_ki+1} error: {_td.get('message','unknown')} — trying next")
                    continue
                if "values" in _td and len(_td["values"]) > 50:
                    _td_df = pd.DataFrame(_td["values"])
                    _td_df["datetime"] = pd.to_datetime(_td_df["datetime"])
                    _td_df = _td_df.set_index("datetime").sort_index()
                    for _c in ["open","high","low","close"]:
                        if _c in _td_df.columns:
                            _td_df[_c] = pd.to_numeric(_td_df[_c], errors="coerce")
                    _td_df = _td_df[[c for c in ["open","high","low","close"] if c in _td_df.columns]].dropna()
                    if "volume" not in _td_df.columns:
                        _td_df["volume"] = 0
                    _td_key_index = _ki
                    print(f"[DATA] TD key {_ki+1} OK: {len(_td_df)} bars ✅")
                    break
            except Exception as _ke:
                print(f"[DATA] TD key {_ki+1} exception: {_ke}")
                continue

        if _td_df is not None and len(_td_df) > 50:
            df = _td_df
        else:
            print("[DATA] All TD keys exhausted — using yfinance emergency fallback")
            try:
                import yfinance as yf
                df = yf.download("GC=F", period="60d", interval="1h",
                                 auto_adjust=False, progress=False)
                df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
                df = df[["open","high","low","close"]].dropna()
                df.index = pd.to_datetime(df.index)
                try: df.index = df.index.tz_convert(None)
                except: pass
                df = df[df.index.dayofweek < 5]
                print(f"[DATA] yfinance fallback: {len(df)} bars ⚠️")
            except Exception as yfe:
                raise ValueError(f"All 3 TD keys failed + yfinance failed: {yfe}")

        if len(df) < 10: raise ValueError(f"Only {len(df)} rows")
        health["data_feed"] = {"ok":True,"msg":f"{len(df)} bars loaded"}
    except Exception as e:
        health["data_feed"] = {"ok":False,"msg":f"Feed error: {e}"}
        return {"signal":"ERROR","confidence":0,"price":None,"health":health,
                "timestamp":datetime.now(timezone.utc).isoformat()}

    try:
        df_live = build_features_for_brain1v2(df)
        if df_live is None or len(df_live) == 0:
            raise ValueError("Feature builder returned empty dataframe")
    except Exception as e:
        health["data_feed"] = {"ok":False,"msg":f"Feature build failed: {e}"}
        return {"signal":"ERROR","confidence":0,"price":None,"health":health,
                "timestamp":datetime.now(timezone.utc).isoformat()}

    # Brain 1
    try:
        if _b1_use_fallback:
            row = df_live.iloc[-1]
            _score = 5.0
            _rsi   = float(row.get("rsi_14", 50))
            _bb    = float(row.get("bb_pct_20", 0.5))
            _stoch = float(row.get("stoch_k", 50))
            _macdh = float(row.get("macd_hist", 0))
            if _rsi < 30: _score += 2.0
            elif _rsi < 40: _score += 1.0
            elif _rsi > 70: _score -= 2.0
            elif _rsi > 60: _score -= 1.0
            _ema5  = float(row.get("ema_5", 0))
            _ema20_val = float(row.get("ema_20", 0))
            if _ema5 and _ema20_val:
                _score += 1.0 if _ema5 > _ema20_val else -1.0
            _score += 0.5 if _macdh > 0 else -0.5
            if _bb < 0.1: _score += 1.5
            elif _bb > 0.9: _score -= 1.5
            if _stoch < 20: _score += 1.0
            elif _stoch > 80: _score -= 1.0
            _score  = max(0.0, min(10.0, _score))
            b1_buy  = _score / 10.0; b1_sell = 1.0 - b1_buy
            b1_dir  = "BUY" if b1_buy>=0.65 else "SELL" if b1_sell>=0.65 else "WAIT"
            b1_score = _score
            health["brain1"] = {"ok":True,"msg":f"FALLBACK prob={round(b1_buy,3)} threshold={THRESHOLD}"}
        else:
            missing = [f for f in features["features"] if f not in df_live.columns]
            if missing:
                raise ValueError(f"Missing features: {missing[:3]}{'...' if len(missing)>3 else ''}")
            X     = df_live[features["features"]].iloc[[-1]]
            prob  = brain1_model.predict_proba(X)[0]
            b1_buy = float(prob[1]); b1_sell = float(prob[0])
            b1_dir  = "BUY" if b1_buy>=THRESHOLD else "SELL" if b1_sell>=THRESHOLD else "WAIT"
            b1_score = b1_buy * 10
            health["brain1"] = {"ok":True,"msg":f"XGBoost prob={round(b1_buy,3)} threshold={THRESHOLD}"}
    except Exception as e:
        health["brain1"] = {"ok":False,"msg":f"Inference failed: {e}"}
        b1_buy=0.5; b1_sell=0.5; b1_dir="WAIT"; b1_score=5.0

    # Brain 2
    try:
        brain2  = Brain2Analyzer()
        report  = brain2.analyze(df_live)
        b2_sig  = report["signal"]
        b2_score = report["confluence_score"]
        m       = report["modules"]
        health["brain2"] = {"ok":True,"msg":f"score={b2_score} signal={b2_sig}"}
    except Exception as e:
        health["brain2"] = {"ok":False,"msg":f"Analysis failed: {e}"}
        b2_sig="WAIT"; b2_score=5.0
        m={"trend":{"direction":"UNKNOWN","strength":"UNKNOWN","ema20":None,"ema50":None},
           "structure":{"structure":"UNKNOWN"},
           "momentum":{"rsi":50,"state":"UNKNOWN","rocket_score":0,"waterfall_score":0},
           "volatility":{"atr14":15},
           "sr_zones":{"nearest_support":0,"nearest_resistance":0,
                       "dist_to_support_pct":0,"dist_to_resistance_pct":0}}
        report={"signal":"WAIT","confluence_score":5.0,"modules":m}

    # Signal fusion
    fused = (b1_score*0.4) + (b2_score*0.6)
    rocket    = m["momentum"].get("rocket_score", 0)
    waterfall = m["momentum"].get("waterfall_score", 0)

    # [FIX-4] B2 direction: BUY / SELL / WAIT
    _b2_up = b2_sig.upper()
    if any(x in _b2_up for x in ["BUY","BULL"]):
        b2_dir = "BUY"
    elif any(x in _b2_up for x in ["SELL","BEAR"]):
        b2_dir = "SELL"
    else:
        b2_dir = "WAIT"

    if b1_dir == b2_dir and b1_dir != "WAIT":
        fused += 0.5
        print(f"   ✅ B1+B2 aligned ({b1_dir}) → fused +0.5")

    # [FIX-1] The broken gate block that used undefined `sig` has been REMOVED.

    # H4 directional bonus (before final sig)
    try:
        _h4_pre = _get_h4_trend(df)
        _h4_dir = _h4_pre.get("direction","NEUTRAL")
        if _h4_dir == "BUY"  and b1_dir == "BUY"  and fused > 5.0:
            fused += 0.4
            print(f"   📈 H4 BUY aligned → fused +0.4")
        elif _h4_dir == "SELL" and b1_dir == "SELL" and fused < 5.0:
            fused -= 0.4
            print(f"   📉 H4 SELL aligned → fused -0.4")
    except Exception:
        pass

    # Decide signal from fused
    if   fused >= 6.5: sig, conf = "BUY",  "HIGH" if fused >= 8 else "MEDIUM"
    elif fused <= 3.5: sig, conf = "SELL", "HIGH" if fused <= 2 else "MEDIUM"
    else:              sig, conf = "WAIT", "LOW"

    # [FIX-2] SINGLE Rocket/Waterfall gate (was duplicated 3x, one broken)
    if sig == "BUY" and rocket < 50:
        print(f"   ⛔ Rocket={rocket} < 50 — BUY momentum not confirmed → WAIT")
        sig, conf = "WAIT", "LOW"
        fused = min(fused, 6.4)
    elif sig == "SELL" and waterfall < 50:
        print(f"   ⛔ Waterfall={waterfall} < 50 — SELL momentum not confirmed → WAIT")
        sig, conf = "WAIT", "LOW"
        fused = min(fused, 6.4)
    elif sig == "BUY":
        print(f"   🚀 Rocket={rocket} CONFIRMED → BUY")
    elif sig == "SELL":
        print(f"   💧 Waterfall={waterfall} CONFIRMED → SELL")

    price   = round(float(df_live["close"].iloc[-1]), 2)
    atr_raw = float(m["volatility"]["atr14"])
    atr     = round(atr_raw, 2)
    support    = float(m["sr_zones"]["nearest_support"])
    resistance = float(m["sr_zones"]["nearest_resistance"])
    ema20_val  = float(m["trend"]["ema20"]) if m["trend"]["ema20"] else price

    sr_gap = abs(resistance - support)
    tight_range = sr_gap < 10.0
    if tight_range:
        print(f"⚠️  Tight range: S/R gap ${sr_gap:.2f} < $10 → WAIT")
        sig   = "WAIT"
        fused = min(fused, 4.9)

    price_below_support = price < support
    price_below_ema20   = price < ema20_val
    sr_flip_active      = price_below_support

    if price_below_support and price_below_ema20 and not tight_range:
        print(f"⚠️  Price ${price:.2f} below support ${support:.2f} AND EMA20 ${ema20_val:.2f} → B2 -1.5")
        b2_score = round(b2_score - 1.5, 2)
        if b2_score < 5.0:
            sig   = "WAIT"
            fused = min(fused, 4.9)

    # H4 veto
    try:
        h4     = _get_h4_trend(df)
        h4_dir = h4["direction"]
        h4_veto = False
        health["h4"] = {"ok":True,"msg":f"{h4_dir} | {h4['reason']}"}
    except Exception as e:
        h4     = {"direction":"NEUTRAL","reason":f"error: {e}","ema20":None,"ema50":None,"rsi":None,"atr":None,"score":5}
        h4_dir = "NEUTRAL"; h4_veto = False
        health["h4"] = {"ok":False,"msg":f"H4 error: {e}"}

    sig_before_veto = sig
    if sig in ("BUY","SELL") and h4_dir != "NEUTRAL" and sig != h4_dir:
        h4_veto = True
        sig     = "WAIT"
        conf    = "LOW"
        print(f"   ⚠️  H4 VETO — {sig_before_veto} blocked (H4={h4_dir})")

    tp  = round(price + atr*1.5, 2) if sig=="BUY" else round(price - atr*1.5, 2)
    tp2 = round(price + atr*2.5, 2) if sig=="BUY" else round(price - atr*2.5, 2)
    sl  = round(price - atr,     2) if sig=="BUY" else round(price + atr,     2)

    # Brain 3 (Groq)
    sr_flip_note = "\n⚠️ NOTE: Price broke below support — possible S/R flip" if sr_flip_active else ""
    tight_note   = "\n⚠️ NOTE: Tight S/R range — low reward potential" if tight_range else ""
    prompt = (
        f"You are an expert XAU/USD gold trading analyst.\n\n"
        f"Price: ${price} | Signal: {sig} | Score: {round(fused,2)}/10\n"
        f"Brain1: {b1_dir} | Brain2: {b2_sig}\n"
        f"H4 Trend: {h4_dir} | {h4['reason']}\n"
        f"Trend: {m['trend']['direction']} {m['trend']['strength']}\n"
        f"Structure: {m['structure']['structure']}\n"
        f"RSI: {m['momentum']['rsi']} | {m['momentum']['state']}\n"
        f"ATR: {atr} | Support: {support} | Resistance: {resistance}\n"
        f"S/R gap: ${sr_gap:.2f}{sr_flip_note}{tight_note}\n\n"
        f"Give 5-line trading briefing. Be specific about WHY signal is {sig}."
    )
    import requests as req
    b3_text = None
    global _groq_model_index, _groq_key_index
    _total_attempts = len(_GROQ_KEYS) * len(GROQ_MODELS)
    for _attempt in range(_total_attempts):
        _key   = _GROQ_KEYS[_groq_key_index % len(_GROQ_KEYS)]
        _model = GROQ_MODELS[_groq_model_index % len(GROQ_MODELS)]
        try:
            resp = req.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {_key}","Content-Type":"application/json"},
                json={"model":_model,"max_tokens":150,
                      "messages":[{"role":"user","content":prompt}]},timeout=15)
            data = resp.json()
            err_msg = data.get("error",{}).get("message","") if "error" in data else ""
            if "rate_limit" in err_msg.lower() or "limit" in err_msg.lower() or "choices" not in data:
                print(f"[B3] key{_groq_key_index+1}/{_model} limited — rotating key")
                _groq_key_index  += 1
                if _groq_key_index % len(_GROQ_KEYS) == 0:
                    _groq_model_index += 1
                continue
            b3_text = data["choices"][0]["message"]["content"]
            health["brain3"] = {"ok":True,"msg":f"model={_model} key={_groq_key_index%len(_GROQ_KEYS)+1}/{len(_GROQ_KEYS)}"}
            break
        except Exception as e:
            print(f"[B3] {_model} error: {e}")
            _groq_key_index += 1
    if not b3_text:
        b3_text = (f"B3 unavailable — all {len(_GROQ_KEYS)} keys × {len(GROQ_MODELS)} models at limit. "
                   f"B1={round(b1_buy,3)} B2={b2_score}. Signal based on B1+B2 only.")
        health["brain3"] = {"ok":False,"msg":f"All {len(_GROQ_KEYS)} Groq keys rate limited"}

    check_sr_flip(price, support, resistance, atr)

    try:
        send_telegram(sig, price, fused, b1_buy, b2_score, atr, h4, tp=tp, sl=sl, tp2=tp2)
        health["telegram"] = {"ok":True,"msg":"alert sent" if sig in ("BUY","SELL") else "standby"}
    except Exception as e:
        health["telegram"] = {"ok":False,"msg":f"Telegram failed: {e}"}

    session_icon = "🟢" if "HIGH" in session_quality else "🟡" if session_quality=="MEDIUM" else "🔴"

    return _to_python({
        "signal":           sig,
        "confidence":       round(fused, 2),
        "price":            price,
        "brain1":           {"prediction":1 if b1_dir=="BUY" else 0,"probability":round(b1_buy,3)},
        "brain2":           {"score":b2_score,"details":{
                                "trend":m["trend"]["direction"],
                                "momentum":m["momentum"]["state"],
                                "support":support,"resistance":resistance,
                                "dist_support_pct":m["sr_zones"]["dist_to_support_pct"],
                                "dist_resistance_pct":m["sr_zones"]["dist_to_resistance_pct"],
                                "rocket_score":rocket,
                                "waterfall_score":waterfall,
                            }},
        "brain3":           {"verdict":sig,"reasoning":b3_text},
        "h4":               h4,
        "h4_veto":          h4_veto,
        "tp":               tp,
        "tp2":              tp2,
        "sl":               sl,
        "atr":              atr,
        "session":          session_quality,
        "session_blocked":  False,
        "session_trade_ok": True,
        "session_icon":     session_icon,
        "day_blocked":      False,
        "daily_pnl":        day_pnl["total_pnl"],
        "trade_count":      day_pnl["trade_count"],
        "health":           health,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "last_run":         datetime.now(timezone.utc).isoformat(),
        "improvements":     "ENABLED",
        "sr_flip_active":   sr_flip_active,
        "tight_range":      tight_range,
        "sr_gap":           round(sr_gap, 2),
        "rocket_score":     rocket,
        "waterfall_score":  waterfall,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 30 — ROCKET/WATERFALL MICRO-ENTRY SYSTEM (1-minute bars)
# ═══════════════════════════════════════════════════════════════════════════════

_rw_last_rocket    = 0
_rw_last_waterfall = 0
_rw_pre_alerted    = False
_rw_alerted        = False
_rw_alert_dir      = "WAIT"
_rw_alert_time     = 0.0
_rw_alert_price    = 0.0
_rw_1m_cache       = None
_rw_1m_cache_ts    = 0.0
_RW_CACHE_TTL      = 45
_RW_ENTRY_COOLDOWN = 600
_RW_PRE_THRESHOLD  = 40
_RW_ENTRY_THRESHOLD= 60

# [ADD-2] Aliases for main.py's mismatched names
_RW_COOLDOWN       = _RW_ENTRY_COOLDOWN
_rw_alert_ts       = _rw_alert_time
_rw_entry_alerted  = False
_rw_result         = None

def _fetch_1m_bars():
    """[ADD-1] 1-minute bars for maximum rocket/waterfall responsiveness."""
    global _rw_1m_cache, _rw_1m_cache_ts
    import time as _time_mod
    now = _time_mod.time()
    if _rw_1m_cache is not None and (now - _rw_1m_cache_ts) < _RW_CACHE_TTL:
        return _rw_1m_cache
    all_keys = [k for k in [
        os.environ.get("TD_KEY",  "f3883b7831a540cda02cfafcfe77e082"),
        os.environ.get("TD_KEY2", "41c8cfdf490b4bf4a0d388e716a32453"),
        os.environ.get("TD_KEY3", "f58b0a482f1443e78fb23cf8975b44d9"),
    ] if k]
    for key in all_keys:
        try:
            r = requests.get("https://api.twelvedata.com/time_series", params={
                "symbol": "XAU/USD", "interval": "1min", "outputsize": 120,
                "apikey": key, "timezone": "UTC", "format": "JSON"
            }, timeout=15)
            d = r.json()
            if "values" not in d:
                continue
            rows = []
            for v in reversed(d["values"]):
                try:
                    rows.append({
                        "open":  float(v["open"]),
                        "high":  float(v["high"]),
                        "low":   float(v["low"]),
                        "close": float(v["close"]),
                        "datetime": v["datetime"],
                    })
                except Exception:
                    pass
            if len(rows) < 30:
                continue
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            _rw_1m_cache    = df
            _rw_1m_cache_ts = now
            print(f"[RW] 1m bars loaded: {len(df)} via key {all_keys.index(key)+1}")
            return df
        except Exception as e:
            print(f"[RW] 1m fetch key error: {e}")
    return None

# Legacy alias
_fetch_5m_bars = _fetch_1m_bars

def _calc_rocket_waterfall(df, live_price=None):
    if df is None or len(df) < 20:
        return 0, 0
    close = df["close"].copy()
    high  = df["high"].copy()
    low   = df["low"].copy()
    if live_price and live_price > 100:
        close.iloc[-1] = live_price
        high.iloc[-1]  = max(high.iloc[-1], live_price)
        low.iloc[-1]   = min(low.iloc[-1],  live_price)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - (100 / (1 + gain / (loss + 1e-9))))
    rsi_now   = rsi.iloc[-1]
    rsi_prev  = rsi.iloc[-3]
    rsi_slope = rsi_now - rsi_prev

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig9  = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig9
    hist_now  = hist.iloc[-1]
    hist_prev = hist.iloc[-3]
    macd_rising  = hist_now > hist_prev and hist_now > 0
    macd_falling = hist_now < hist_prev and hist_now < 0

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    price_now = close.iloc[-1]
    bull_ema = (price_now > ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1])
    bear_ema = (price_now < ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1])
    price_above_ema9  = price_now > ema9.iloc[-1]
    price_below_ema9  = price_now < ema9.iloc[-1]

    tr   = pd.concat([high - low,
                      (high - close.shift()).abs(),
                      (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(14).mean().iloc[-1]
    body = abs(close.iloc[-1] - df["open"].iloc[-1])
    momentum_candle = body > atr * 0.3

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_width_now  = (bb_up.iloc[-1]  - bb_dn.iloc[-1])
    bb_width_prev = (bb_up.iloc[-5]  - bb_dn.iloc[-5])
    bb_expanding = bb_width_now > bb_width_prev * 1.05

    mom3 = close.iloc[-1] - close.iloc[-4]
    mom3_bull = mom3 > atr * 0.2
    mom3_bear = mom3 < -atr * 0.2

    rocket = 0
    if rsi_now < 30:                      rocket += 30
    elif rsi_now < 40 and rsi_slope > 2:  rocket += 22
    elif rsi_now < 50 and rsi_slope > 3:  rocket += 15
    elif rsi_now < 55 and rsi_slope > 1:  rocket +=  8
    if rsi_now > 70: rocket = min(rocket, 10)
    if macd_rising:
        rocket += 20
        if hist_now > abs(hist_prev) * 1.2: rocket += 5
    if bull_ema:          rocket += 20
    elif price_above_ema9: rocket += 10
    if bb_expanding and price_now > bb_mid.iloc[-1]: rocket += 10
    if mom3_bull: rocket += 10
    if momentum_candle and close.iloc[-1] > df["open"].iloc[-1]: rocket += 5
    rocket = min(int(rocket), 100)

    waterfall = 0
    if rsi_now > 70:                        waterfall += 30
    elif rsi_now > 60 and rsi_slope < -2:   waterfall += 22
    elif rsi_now > 50 and rsi_slope < -3:   waterfall += 15
    elif rsi_now > 45 and rsi_slope < -1:   waterfall +=  8
    if rsi_now < 30: waterfall = min(waterfall, 10)
    if macd_falling:
        waterfall += 20
        if hist_now < hist_prev * 1.2: waterfall += 5
    if bear_ema:           waterfall += 20
    elif price_below_ema9: waterfall += 10
    if bb_expanding and price_now < bb_mid.iloc[-1]: waterfall += 10
    if mom3_bear: waterfall += 10
    if momentum_candle and close.iloc[-1] < df["open"].iloc[-1]: waterfall += 5
    waterfall = min(int(waterfall), 100)

    return rocket, waterfall


def run_rocket_analysis(live_price=None):
    global _rw_last_rocket, _rw_last_waterfall

    session_quality, trade_allowed = _get_session_quality()
    if not trade_allowed:
        return {
            "rocket": 0, "waterfall": 0,
            "signal": "WAIT", "session_blocked": True,
            "msg": "Off-hours — standby",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    df1 = _fetch_1m_bars()
    if df1 is None:
        return {"rocket": 0, "waterfall": 0, "signal": "ERROR", "msg": "No 1m data",
                "timestamp": datetime.now(timezone.utc).isoformat()}

    rocket, waterfall = _calc_rocket_waterfall(df1, live_price)
    _rw_last_rocket    = rocket
    _rw_last_waterfall = waterfall

    price = live_price or float(df1["close"].iloc[-1])

    tr  = pd.concat([
        df1["high"] - df1["low"],
        (df1["high"] - df1["close"].shift()).abs(),
        (df1["low"]  - df1["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr_1m = float(tr.rolling(14).mean().iloc[-1])
    # Scale 1min ATR up to H1-equivalent for TP/SL (1m ATR ~= 1/5 of H1 ATR)
    atr    = max(atr_1m * 5, 8.0)

    if rocket >= _RW_ENTRY_THRESHOLD:
        signal = "ROCKET"
        entry  = round(price, 2)
        tp1    = round(price + atr * 1.2, 2)
        tp2    = round(price + atr * 2.0, 2)
        sl     = round(price - atr * 0.8, 2)
    elif waterfall >= _RW_ENTRY_THRESHOLD:
        signal = "WATERFALL"
        entry  = round(price, 2)
        tp1    = round(price - atr * 1.2, 2)
        tp2    = round(price - atr * 2.0, 2)
        sl     = round(price + atr * 0.8, 2)
    elif rocket >= _RW_PRE_THRESHOLD:
        signal = "ROCKET_BUILDING"
        entry = tp1 = tp2 = sl = 0
    elif waterfall >= _RW_PRE_THRESHOLD:
        signal = "WATERFALL_BUILDING"
        entry = tp1 = tp2 = sl = 0
    else:
        signal = "WAIT"
        entry = tp1 = tp2 = sl = 0

    delta = df1["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - (100 / (1 + gain / (loss + 1e-9)))).iloc[-1])

    return {
        "rocket":    rocket,
        "waterfall": waterfall,
        "signal":    signal,
        "price":     round(price, 2),
        "rsi":       round(rsi, 1),
        "atr":       round(atr, 2),
        "entry":     entry,
        "tp1":       tp1,
        "tp2":       tp2,
        "sl":        sl,
        "session":   session_quality,
        "session_blocked": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_time": datetime.now(timezone.utc).isoformat() if signal in ("ROCKET","WATERFALL") else None,
    }
