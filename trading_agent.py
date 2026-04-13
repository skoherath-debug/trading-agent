import pickle, json, requests, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import json as _json
import pandas as _pd

def calculate_atr_dynamic(df, period=14):
    """Wilder ATR."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = _pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_dynamic_tpsl(df, direction, entry_price):
    """Returns tp_usd, sl_usd, tp_level, sl_level using ATR config."""
    try:
        with open("/content/drive/MyDrive/trading_agent/dynamic_tpsl_config.json") as f:
            cfg = _json.load(f)
    except:
        return 15.0, 10.0, entry_price+5, entry_price-3.33

    atr_val = float(calculate_atr_dynamic(df, cfg["atr_period"]).iloc[-1])
    if _pd.isna(atr_val): atr_val = 16.78

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


# ── Phase 20: Session Filter ──────────────────────────────────────────────────
from datetime import time as _time

# ── numpy → pure Python converter (Phase 25 fix) ─────────────────────────────
def _to_python(obj):
    import numpy as _np
    if isinstance(obj, dict):   return {k: _to_python(v) for k,v in obj.items()}
    if isinstance(obj, list):   return [_to_python(v) for v in obj]
    if isinstance(obj, _np.integer):  return int(obj)
    if isinstance(obj, _np.floating): return float(obj)
    if isinstance(obj, _np.ndarray):  return obj.tolist()
    if isinstance(obj, _np.bool_):    return bool(obj)
    return obj
# ─────────────────────────────────────────────────────────────────────────────



_BLOCK_START = _time(21, 0)
_BLOCK_END   = _time(1,  0)

def _is_session_blocked(utc_dt=None):
    """Returns True during 21:00–01:00 UTC (low liquidity window)."""
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    t = utc_dt.time().replace(second=0, microsecond=0)
    return t >= _BLOCK_START or t < _BLOCK_END   # wraps midnight

def _get_session_quality(utc_dt=None):
    """Returns quality string and trade_allowed bool."""
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    if _is_session_blocked(utc_dt):
        return "BLOCKED", False
    h = utc_dt.hour
    london   = 7  <= h < 16
    new_york = 13 <= h < 21
    if london and new_york:
        return "HIGH (London/NY overlap)", True
    elif london or new_york:
        return "MEDIUM", True
    else:
        return "LOW (no major session)", False
# ─────────────────────────────────────────────────────────────────────────────


# ── Phase 21: H4 Trend Confirmation ──────────────────────────────────────────
def _get_h4_trend(df_h1: pd.DataFrame) -> dict:
    """
    Resample H1 OHLCV → H4 and compute trend direction.
    Returns dict with: direction, ema20, ema50, rsi, atr, score, reason
    direction: 'BUY' | 'SELL' | 'NEUTRAL'
    """
    try:
        df = df_h1.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Resample H1 → H4
        h4 = df.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum"
        }).dropna()

        if len(h4) < 55:
            return {"direction": "NEUTRAL", "reason": "insufficient H4 data",
                    "score": 5, "ema20": None, "ema50": None, "rsi": None, "atr": None}

        c = h4["close"]

        # EMAs
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()

        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1]

        # ATR
        tr   = pd.concat([
            h4["high"] - h4["low"],
            (h4["high"] - h4["close"].shift()).abs(),
            (h4["low"]  - h4["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr  = tr.rolling(14).mean().iloc[-1]

        price  = c.iloc[-1]
        e20    = ema20.iloc[-1]
        e50    = ema50.iloc[-1]
        e20_p  = ema20.iloc[-2]   # previous bar

        # Trend score (0–4 conditions)
        bull_pts = sum([
            price > e20,
            e20   > e50,
            e20   > e20_p,          # EMA20 rising
            rsi   > 50,
        ])

        if bull_pts >= 3:
            direction = "BUY"
            score     = 8 if bull_pts == 4 else 7
            reason    = f"H4 bullish ({bull_pts}/4 conditions)"
        elif bull_pts <= 1:
            direction = "SELL"
            score     = 2 if bull_pts == 0 else 3
            reason    = f"H4 bearish ({bull_pts}/4 conditions)"
        else:
            direction = "NEUTRAL"
            score     = 5
            reason    = f"H4 neutral ({bull_pts}/4 conditions)"

        return {
            "direction": direction,
            "score":     score,
            "reason":    reason,
            "ema20":     round(e20,   2),
            "ema50":     round(e50,   2),
            "rsi":       round(rsi,   2),
            "atr":       round(atr,   2),
            "bull_pts":  bull_pts,
            "candles":   len(h4),
        }

    except Exception as e:
        return {"direction": "NEUTRAL", "reason": f"H4 error: {e}",
                "score": 5, "ema20": None, "ema50": None, "rsi": None, "atr": None}
# ─────────────────────────────────────────────────────────────────────────────


# ── Phase 22: Daily Loss Limit ────────────────────────────────────────────────
_DAILY_PNL_PATH  = "/content/drive/MyDrive/trading_agent/daily_pnl.json"
_DAILY_LOSS_LIMIT  = 30.0   # USD — hard stop for the day
_MAX_TRADES_PER_DAY = 6     # max trades per UTC day

def _load_daily_pnl() -> dict:
    """Load today's P&L record. Resets automatically on new UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with open(_DAILY_PNL_PATH) as f:
            data = json.load(f)
        if data.get("date") != today:
            raise ValueError("stale")
        return data
    except:
        return {"date": today, "trades": [], "total_pnl": 0.0, "trade_count": 0}

def _save_daily_pnl(data: dict) -> None:
    try:
        with open(_DAILY_PNL_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[DAILY PNL] Save error: {e}")

def _check_daily_limit() -> tuple:
    """
    Returns (blocked: bool, reason: str, pnl: dict)
    Blocks if: total loss >= DAILY_LOSS_LIMIT  OR  trade_count >= MAX_TRADES_PER_DAY
    """
    pnl = _load_daily_pnl()
    total_loss = abs(min(pnl["total_pnl"], 0))   # only negative P&L counts
    count      = pnl["trade_count"]

    if total_loss >= _DAILY_LOSS_LIMIT:
        return True, f"Daily loss limit hit (${total_loss:.2f} / ${_DAILY_LOSS_LIMIT})", pnl
    if count >= _MAX_TRADES_PER_DAY:
        return True, f"Max trades reached ({count} / {_MAX_TRADES_PER_DAY})", pnl
    return False, f"OK — loss=${total_loss:.2f}/${_DAILY_LOSS_LIMIT} | trades={count}/{_MAX_TRADES_PER_DAY}", pnl

def record_trade_result(signal: str, entry: float, exit_price: float,
                        lot: float = 0.03, px_to_usd: float = 3.0) -> dict:
    """
    Call this after a trade closes (TP / SL / manual).
    Calculates P&L and appends to daily_pnl.json.

    pnl_usd = (exit - entry) * lot * 100 * px_to_usd  (for gold, 1pt = $0.03 per micro)
    Simplified: use raw price difference × px_to_usd × lot_multiplier
    """
    pts     = (exit_price - entry) if signal == "BUY" else (entry - exit_price)
    pnl_usd = round(pts * px_to_usd, 2)

    pnl_data = _load_daily_pnl()
    pnl_data["trades"].append({
        "time":   datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "entry":  entry,
        "exit":   exit_price,
        "pnl":    pnl_usd,
    })
    pnl_data["total_pnl"]    = round(pnl_data["total_pnl"] + pnl_usd, 2)
    pnl_data["trade_count"] += 1
    _save_daily_pnl(pnl_data)

    icon = "✅" if pnl_usd >= 0 else "❌"
    print(f"[TRADE RESULT] {icon} {signal} P&L=${pnl_usd:+.2f} | Day total=${pnl_data['total_pnl']:+.2f} | Trades={pnl_data['trade_count']}")
    return pnl_data

def get_daily_summary() -> dict:
    """Returns today's trading summary — use in dashboard."""
    pnl = _load_daily_pnl()
    blocked, reason, _ = _check_daily_limit()
    wins   = sum(1 for t in pnl["trades"] if t["pnl"] > 0)
    losses = sum(1 for t in pnl["trades"] if t["pnl"] < 0)
    return {
        "date":         pnl["date"],
        "total_pnl":    pnl["total_pnl"],
        "trade_count":  pnl["trade_count"],
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / max(pnl["trade_count"], 1) * 100, 1),
        "limit_usd":    _DAILY_LOSS_LIMIT,
        "max_trades":   _MAX_TRADES_PER_DAY,
        "blocked":      blocked,
        "status":       reason,
        "trades":       pnl["trades"],
    }
# ─────────────────────────────────────────────────────────────────────────────


def build_features(df):
    d = df.copy()
    d["return_1"]  = d["close"].pct_change(1)
    d["return_3"]  = d["close"].pct_change(3)
    d["return_5"]  = d["close"].pct_change(5)
    d["return_10"] = d["close"].pct_change(10)
    for s in [5,10,20,50,100,200]:
        d[f"ema_{s}"] = d["close"].ewm(span=s, adjust=False).mean()
        d[f"c_vs_ema{s}"] = (d["close"] - d[f"ema_{s}"]) / d["close"]
    for s in [10,20,50]:
        d[f"sma_{s}"] = d["close"].rolling(s).mean()
        d[f"c_vs_sma{s}"] = (d["close"] - d[f"sma_{s}"]) / d["close"]
    def rsi(series, n):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(n).mean()
        loss  = (-delta.clip(upper=0)).rolling(n).mean()
        return 100 - (100 / (1 + gain / loss))
    d["rsi_14"] = rsi(d["close"], 14)
    d["rsi_7"]  = rsi(d["close"], 7)
    d["rsi_21"] = rsi(d["close"], 21)
    d["rsi_div"] = d["rsi_14"] - d["rsi_14"].shift(5)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]
    bb20_mid = d["close"].rolling(20).mean()
    bb20_std = d["close"].rolling(20).std()
    d["bb_upper_20"] = bb20_mid + 2*bb20_std
    d["bb_lower_20"] = bb20_mid - 2*bb20_std
    d["bb_pct_20"]   = (d["close"] - d["bb_lower_20"]) / (d["bb_upper_20"] - d["bb_lower_20"] + 1e-9)
    d["bb_width_20"] = (bb20_std * 2) / bb20_mid
    bb14_mid = d["close"].rolling(14).mean()
    bb14_std = d["close"].rolling(14).std()
    d["bb_upper_14"] = bb14_mid + 2*bb14_std
    d["bb_lower_14"] = bb14_mid - 2*bb14_std
    d["bb_pct_14"]   = (d["close"] - d["bb_lower_14"]) / (d["bb_upper_14"] - d["bb_lower_14"] + 1e-9)
    d["bb_width_14"] = (bb14_std * 2) / bb14_mid
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"]  - d["close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["atr_7"]  = tr.rolling(7).mean()
    d["atr_21"] = tr.rolling(21).mean()
    low14  = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100*(d["close"]-low14)/(high14-low14+1e-9)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    d["willr_14"] = -100*(high14-d["close"])/(high14-low14+1e-9)
    tp = (d["high"]+d["low"]+d["close"])/3
    d["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015*tp.rolling(20).std()+1e-9)
    plus_dm  = d["high"].diff().clip(lower=0)
    minus_dm = (-d["low"].diff()).clip(lower=0)
    d["adx_14"] = 100*(plus_dm.rolling(14).sum()-minus_dm.rolling(14).sum()).abs()/(tr.rolling(14).sum()+1e-9)
    d["volume_ratio"]   = d["volume"]/(d["volume"].rolling(20).mean()+1e-9)
    d["vol_ratio_5_20"] = d["volume"].rolling(5).mean()/(d["volume"].rolling(20).mean()+1e-9)
    body = (d["close"]-d["open"]).abs()
    rng  = d["high"]-d["low"]+1e-9
    d["candle_body"]  = body/rng
    d["upper_wick"]   = (d["high"]-d[["close","open"]].max(axis=1))/rng
    d["lower_wick"]   = (d[["close","open"]].min(axis=1)-d["low"])/rng
    d["candle_dir"]   = (d["close"]>d["open"]).astype(int)
    d["sin_hour"] = np.sin(2*np.pi*d.index.hour/24)
    d["cos_hour"] = np.cos(2*np.pi*d.index.hour/24)
    d["sin_dow"]  = np.sin(2*np.pi*d.index.dayofweek/5)
    d["cos_dow"]  = np.cos(2*np.pi*d.index.dayofweek/5)
    d["hour"]     = d.index.hour
    d["dow"]      = d.index.dayofweek
    d["mom_5"]    = d["close"].pct_change(5)
    d["mom_10"]   = d["close"].pct_change(10)
    d["mom_20"]   = d["close"].pct_change(20)
    d["trend_5_20"]  = (d["ema_5"]  > d["ema_20"]).astype(int)
    d["trend_10_50"] = (d["ema_10"] > d["ema_50"]).astype(int)
    d["pos_in_day"]  = d.index.hour / 23.0
    d["gap"]         = (d["open"] - d["close"].shift()) / d["close"].shift()
    return d.dropna()


GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"
SYMBOL       = "GC=F"

class Brain2Analyzer:
    def analyze_trend(self, df):
        close=df["close"]; ema20=close.ewm(span=20,adjust=False).mean().iloc[-1]
        ema50=close.ewm(span=50,adjust=False).mean().iloc[-1]
        ema200=close.ewm(span=200,adjust=False).mean().iloc[-1]; price=close.iloc[-1]
        bs=sum([price>ema20,ema20>ema50,ema50>ema200])
        if bs==4: d,s="BULLISH","STRONG"
        elif bs==3: d,s="BULLISH","MODERATE"
        elif bs==0: d,s="BEARISH","STRONG"
        elif bs==1: d,s="BEARISH","MODERATE"
        else: d,s="NEUTRAL","CHOPPY"
        return {"direction":d,"strength":s,"ema20":round(ema20,2),"ema50":round(ema50,2),"ema200":round(ema200,2),"score":(bs/4)*10}

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
        if 0<=hour<8: active.append("ASIA")
        overlap=len(active)>1
        rr=(df["high"]-df["low"]).iloc[-1]; ar=(df["high"]-df["low"]).iloc[-20:].mean()
        return {"active_sessions":active,"overlap":overlap,
                "liquidity":"HIGH" if overlap else "MEDIUM" if active else "LOW",
                "candle_range_ratio":round(rr/ar if ar>0 else 1.0,2),
                "score":8 if overlap else 5 if active else 3}

    def analyze_momentum(self, df):
        close=df["close"]; delta=close.diff()
        gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
        rsi=(100-(100/(1+gain/loss))).iloc[-1]
        macd=(close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean())
        hist=(macd-macd.ewm(span=9,adjust=False).mean()).iloc[-1]
        mc="BULLISH" if hist>0 else "BEARISH"
        if rsi>70: state="OVERBOUGHT"
        elif rsi<30: state="OVERSOLD"
        elif rsi>55 and mc=="BULLISH": state="BULLISH_MOMENTUM"
        elif rsi<45 and mc=="BEARISH": state="BEARISH_MOMENTUM"
        else: state="NEUTRAL"
        return {"rsi":round(rsi,2),"macd_histogram":round(hist,4),"macd_cross":mc,"state":state,
                "divergence_detected":False,
                "score":8 if state=="BULLISH_MOMENTUM" else 2 if state=="BEARISH_MOMENTUM"
                        else 5 if state=="NEUTRAL" else 3 if state=="OVERBOUGHT" else 7}

    def analyze_volatility(self, df):
        h,l,c=df["high"],df["low"],df["close"]
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr=tr.rolling(14).mean().iloc[-1]; avg=tr.rolling(14).mean().iloc[-50:-1].mean()
        r=atr/avg if avg>0 else 1.0
        s="HIGH_VOLATILITY" if r>1.5 else "LOW_VOLATILITY" if r<0.7 else "NORMAL"
        return {"atr14":round(atr,4),"atr_ratio":round(r,2),"state":s,
                "suggested_sl_pips":round(atr,2),"suggested_tp_pips":round(atr*1.5,2),
                "score":7 if s=="NORMAL" else 4 if s=="HIGH_VOLATILITY" else 5}

    def analyze_sr_zones(self, df, lookback=50):
        highs=df["high"].values[-lookback:]; lows=df["low"].values[-lookback:]
        price=df["close"].iloc[-1]
        all_levels=sorted(set(
            [h for i,h in enumerate(highs) if h==max(highs[max(0,i-3):i+4])]+
            [l for i,l in enumerate(lows)  if l==min(lows[max(0,i-3):i+4])]))
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
                "price":round(df["close"].iloc[-1],2),
                "signal":c["signal"],"confidence":c["confidence"],
                "confluence_score":c["confluence_score"],
                "modules":{"trend":t,"structure":st,"session":se,"momentum":m,"volatility":v,"sr_zones":sr},
                "module_scores":c["module_scores"]}

def build_features(df):
    try:
        d=df.copy(); d.index=pd.to_datetime(d.index).tz_localize(None)
        delta=d["close"].diff()
        gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
        d["RSI"]=100-(100/(1+gain/loss))
        ema12=d["close"].ewm(span=12,adjust=False).mean(); ema26=d["close"].ewm(span=26,adjust=False).mean()
        d["MACD"]=ema12-ema26; d["MACD_signal"]=d["MACD"].ewm(span=9,adjust=False).mean()
        d["MACD_hist"]=d["MACD"]-d["MACD_signal"]
        bb_mid=d["close"].rolling(20).mean(); bb_std=d["close"].rolling(20).std()
        d["BB_width"]=(bb_std*2)/bb_mid
        d["BB_position"]=(d["close"]-(bb_mid-2*bb_std))/(4*bb_std)
        tr=pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift()).abs(),(d["low"]-d["close"].shift()).abs()],axis=1).max(axis=1)
        d["ATR"]=tr.rolling(14).mean(); d["ATR_ratio"]=d["ATR"]/tr.rolling(50).mean()
        ema9=d["close"].ewm(span=9,adjust=False).mean(); ema21=d["close"].ewm(span=21,adjust=False).mean()
        ema50=d["close"].ewm(span=50,adjust=False).mean(); ema200=d["close"].ewm(span=200,adjust=False).mean()
        d["EMA_9_above_21"]=(ema9>ema21).astype(int); d["EMA_21_above_50"]=(ema21>ema50).astype(int)
        d["EMA_50_above_200"]=(ema50>ema200).astype(int)
        d["Volume_ratio"]=d["volume"]/d["volume"].rolling(20).mean()
        d["Body_ratio"]=(d["close"]-d["open"]).abs()/(d["high"]-d["low"]+1e-9)
        plus_dm=d["high"].diff().clip(lower=0); minus_dm=(-d["low"].diff()).clip(lower=0)
        d["ADX"]=100*(plus_dm.rolling(14).sum()-minus_dm.rolling(14).sum()).abs()/(tr.rolling(14).sum()+1e-9)
        try:
            dxy=yf.download("DX-Y.NYB",period="2y",interval="1h",auto_adjust=False,progress=False)
            dxy.columns=[col[0].lower() if isinstance(col,tuple) else col.lower() for col in dxy.columns]
            dxy.index=pd.to_datetime(dxy.index).tz_localize(None)
            dxy["DXY_change"]=dxy["close"].pct_change()
            d=d.join(dxy[["DXY_change"]],how="left"); d["DXY_change"]=d["DXY_change"].fillna(0)
        except: d["DXY_change"]=0
        idx=d.index
        d["Hour"]=idx.hour
        d["Session_Asian"]=((idx.hour>=0)&(idx.hour<8)).astype(int)
        d["Session_London"]=((idx.hour>=7)&(idx.hour<16)).astype(int)
        d["Session_NY"]=((idx.hour>=13)&(idx.hour<22)).astype(int)
        d["Higher_high"]=(d["high"]>d["high"].rolling(20).max().shift()).astype(int)
        d["Lower_low"]=(d["low"]<d["low"].rolling(20).min().shift()).astype(int)
        d["Close_above_open"]=(d["close"]>d["open"]).astype(int)
        d["Dist_from_high20"]=(d["high"].rolling(20).max()-d["close"])/d["close"]
        d["Dist_from_low20"]=(d["close"]-d["low"].rolling(20).min())/d["close"]
        d["RSI_slope"]=d["RSI"]-d["RSI"].shift(5)
        d["MACD_cross"]=(d["MACD"]>d["MACD_signal"]).astype(int)
        d["Price_momentum"]=d["close"].pct_change(10)
        d["H4_RSI"]=d["RSI"].rolling(4).mean(); d["H4_ATR"]=d["ATR"].rolling(4).mean()
        d["H4_trend"]=(d["close"]>d["close"].shift(4)).astype(int)
        d["H4_mom"]=d["close"].pct_change(4); d["H4_BB_pos"]=d["BB_position"].rolling(4).mean()
        return d.dropna()
    except Exception as e:
        print(f"Error: {e}"); return None

def run_trading_agent():
    print("\n"+"="*50)
    print(f"  XAU/USD 3-Brain Trading Agent")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*50)
    with open("/content/drive/MyDrive/trading_agent/brain1_xgboost_v4.pkl","rb") as f:
        brain1_model=pickle.load(f)
    with open("/content/drive/MyDrive/trading_agent/brain1_features_v4.json","r") as f:
        features=json.load(f)
    THRESHOLD=features.get("threshold",0.74)
    print(f"✅ Brain 1 loaded")
    df=yf.download(SYMBOL,period="60d",interval="1h",auto_adjust=False,progress=False)
    df.columns=[col[0].lower() if isinstance(col,tuple) else col.lower() for col in df.columns]
    df=df[["open","high","low","close","volume"]].dropna()
    df=df[df.index.dayofweek<5]; df=df[df["volume"]>0]
    df_live=build_features_for_brain1v2(df.set_index('datetime') if 'datetime' in df.columns else df)
    print(f"✅ Live data: {len(df_live)} rows")
    X=df_live[features["features"]].iloc[[-1]]; prob=brain1_model.predict_proba(X)[0]
    b1_buy=prob[1]; b1_sell=prob[0]
    b1_dir="BUY" if b1_buy>=THRESHOLD else "SELL" if b1_sell>=THRESHOLD else "WAIT"
    b1_score=b1_buy*10
    brain2=Brain2Analyzer(); report=brain2.analyze(df_live)
    b2_sig=report["signal"]; b2_score=report["confluence_score"]; m=report["modules"]
    fused=(b1_score*0.4)+(b2_score*0.6)
    if b1_dir==b2_sig and b1_dir!="WAIT": fused+=0.5
    if fused>=6.5: sig,conf="BUY","HIGH" if fused>=8 else "MEDIUM"
    elif fused<=3.5: sig,conf="SELL","HIGH" if fused<=2 else "MEDIUM"
    else: sig,conf="WAIT","LOW"
    price=round(df_live["close"].iloc[-1],2)
    prompt=("You are an expert XAU/USD gold trading analyst.\n\n"
            "Price: $"+str(price)+"\nSignal: "+sig+" | Score: "+str(round(fused,2))+"/10\n"
            "Brain1: "+b1_dir+" | Brain2: "+b2_sig+"\n\n"
            "Trend: "+m["trend"]["direction"]+" "+m["trend"]["strength"]+"\n"
            "Structure: "+m["structure"]["structure"]+"\n"
            "RSI: "+str(m["momentum"]["rsi"])+" | "+m["momentum"]["state"]+"\n"
            "ATR: "+str(m["volatility"]["atr14"])+"\n"
            "Support: "+str(m["sr_zones"]["nearest_support"])+"\n"
            "Resistance: "+str(m["sr_zones"]["nearest_resistance"])+"\n\n"
            "Give 5-line trading briefing:\n1. Market condition now\n"
            "2. Why signal is "+sig+"\n3. What to watch next\n4. SL and TP levels")
    try:
        resp=requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
            json={"model":GROQ_MODEL,"max_tokens":300,"messages":[{"role":"user","content":prompt}]},timeout=15)
        data=resp.json()
        b3_text=data["choices"][0]["message"]["content"] if "choices" in data else str(data)
    except Exception as e:
        b3_text=f"Brain 3 unavailable: {e}"
    print("\n"+"="*50)
    print(f"  Price     : ${price}")
    print(f"  Brain 1   : {b1_dir:4} | {round(b1_score,1)}/10")
    print(f"  Brain 2   : {b2_sig:4} | {b2_score}/10")
    print(f"  FUSED     : {sig:4} | {round(fused,2)}/10 | {conf}")
    print("="*50)
    print(f"  Trend     : {m['trend']['direction']} {m['trend']['strength']}")
    print(f"  Structure : {m['structure']['structure']}")
    print(f"  RSI       : {m['momentum']['rsi']} | {m['momentum']['state']}")
    print(f"  Support   : {m['sr_zones']['nearest_support']}")
    print(f"  Resistance: {m['sr_zones']['nearest_resistance']}")
    print("="*50)
    print(b3_text)
    print("="*50)

    return {
        "signal": sig,
        "score": round(fused, 2),
        "price": price,
        "tp": get_dynamic_tpsl(d, sig, price)[2] if sig in ("BUY","SELL") else price,
        "sl": get_dynamic_tpsl(d, sig, price)[3] if sig in ("BUY","SELL") else price,
        "brain1_prob": round(b1_buy, 4),
        "brain2_score": b2_score,
        "brain3_view": b3_text[:120]
    }


# ── Brain 1 v2 (68 features) ──────────────────────
def build_features_v2(df):
    d = df.copy()

    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]            = 100 - (100 / (1 + gain / loss))
    d["RSI_slope"]      = d["RSI"] - d["RSI"].shift(5)
    d["RSI_slope2"]     = d["RSI"] - d["RSI"].shift(14)
    d["RSI_overbought"] = (d["RSI"] > 70).astype(int)
    d["RSI_oversold"]   = (d["RSI"] < 30).astype(int)

    low14  = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["Stoch_K"]    = 100 * (d["close"] - low14) / (high14 - low14 + 1e-9)
    d["Stoch_D"]    = d["Stoch_K"].rolling(3).mean()
    d["Stoch_cross"]= (d["Stoch_K"] > d["Stoch_D"]).astype(int)

    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["CCI"]        = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)
    d["Williams_R"] = -100 * (high14 - d["close"]) / (high14 - low14 + 1e-9)

    ema9   = d["close"].ewm(span=9,   adjust=False).mean()
    ema21  = d["close"].ewm(span=21,  adjust=False).mean()
    ema50  = d["close"].ewm(span=50,  adjust=False).mean()
    ema200 = d["close"].ewm(span=200, adjust=False).mean()
    d["EMA_9_above_21"]   = (ema9  > ema21 ).astype(int)
    d["EMA_21_above_50"]  = (ema21 > ema50 ).astype(int)
    d["EMA_50_above_200"] = (ema50 > ema200).astype(int)
    d["EMA_9_21_gap"]     = (ema9  - ema21)  / d["close"]
    d["EMA_21_50_gap"]    = (ema21 - ema50)  / d["close"]
    d["Price_vs_EMA50"]   = (d["close"] - ema50)  / d["close"]
    d["Price_vs_EMA200"]  = (d["close"] - ema200) / d["close"]

    plus_dm  = d["high"].diff().clip(lower=0)
    minus_dm = (-d["low"].diff()).clip(lower=0)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"]  - d["close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["ADX"]      = 100 * (plus_dm.rolling(14).sum() - minus_dm.rolling(14).sum()).abs() / (tr.rolling(14).sum() + 1e-9)
    d["Plus_DI"]  = 100 * plus_dm.rolling(14).sum()  / (tr.rolling(14).sum() + 1e-9)
    d["Minus_DI"] = 100 * minus_dm.rolling(14).sum() / (tr.rolling(14).sum() + 1e-9)
    d["DI_diff"]  = d["Plus_DI"] - d["Minus_DI"]

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["MACD"]            = ema12 - ema26
    d["MACD_signal"]     = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]       = d["MACD"] - d["MACD_signal"]
    d["MACD_cross"]      = (d["MACD"] > d["MACD_signal"]).astype(int)
    d["MACD_hist_slope"] = d["MACD_hist"] - d["MACD_hist"].shift(3)

    bb_mid   = d["close"].rolling(20).mean()
    bb_std   = d["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    d["BB_width"]       = (bb_std * 2) / bb_mid
    d["BB_position"]    = (d["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    d["BB_squeeze"]     = (d["BB_width"] < d["BB_width"].rolling(50).mean()).astype(int)
    d["Above_BB_upper"] = (d["close"] > bb_upper).astype(int)
    d["Below_BB_lower"] = (d["close"] < bb_lower).astype(int)

    d["ATR"]             = tr.rolling(14).mean()
    d["ATR_ratio"]       = d["ATR"] / (tr.rolling(50).mean() + 1e-9)
    d["ATR_pct"]         = d["ATR"] / d["close"]
    d["High_vol_regime"] = (d["ATR_ratio"] > 1.2).astype(int)
    d["Low_vol_regime"]  = (d["ATR_ratio"] < 0.8).astype(int)

    body        = (d["close"] - d["open"]).abs()
    candle      = d["high"] - d["low"] + 1e-9
    upper_wick  = d["high"] - d[["close","open"]].max(axis=1)
    lower_wick  = d[["close","open"]].min(axis=1) - d["low"]
    d["Body_ratio"]       = body / candle
    d["Upper_wick_ratio"] = upper_wick / candle
    d["Lower_wick_ratio"] = lower_wick / candle
    d["Close_above_open"] = (d["close"] > d["open"]).astype(int)
    d["Bull_pin"]  = ((lower_wick > 2*body) & (lower_wick > upper_wick)).astype(int)
    d["Bear_pin"]  = ((upper_wick > 2*body) & (upper_wick > lower_wick)).astype(int)
    prev_body = (d["close"].shift() - d["open"].shift()).abs()
    d["Bull_engulf"] = ((d["close"]>d["open"]) & (d["open"]<d["close"].shift()) & (d["close"]>d["open"].shift()) & (body>prev_body)).astype(int)
    d["Bear_engulf"] = ((d["close"]<d["open"]) & (d["open"]>d["close"].shift()) & (d["close"]<d["open"].shift()) & (body>prev_body)).astype(int)

    d["Higher_high"]      = (d["high"] > d["high"].rolling(20).max().shift()).astype(int)
    d["Lower_low"]        = (d["low"]  < d["low"].rolling(20).min().shift()).astype(int)
    d["Dist_from_high20"] = (d["high"].rolling(20).max() - d["close"]) / d["close"]
    d["Dist_from_low20"]  = (d["close"] - d["low"].rolling(20).min()) / d["close"]
    d["Near_resistance"]  = (d["Dist_from_high20"] < 0.002).astype(int)
    d["Near_support"]     = (d["Dist_from_low20"]  < 0.002).astype(int)

    d["Price_momentum"]  = d["close"].pct_change(10)
    d["Price_momentum3"] = d["close"].pct_change(3)
    d["Volume_ratio"]    = d["volume"] / (d["volume"].rolling(20).mean() + 1e-9)
    d["Volume_surge"]    = (d["Volume_ratio"] > 1.5).astype(int)

    idx = d.index
    d["Hour"]             = idx.hour
    d["Session_Asian"]    = ((idx.hour >= 0)  & (idx.hour < 8 )).astype(int)
    d["Session_London"]   = ((idx.hour >= 7)  & (idx.hour < 16)).astype(int)
    d["Session_NY"]       = ((idx.hour >= 13) & (idx.hour < 22)).astype(int)
    d["Session_overlap"]  = ((idx.hour >= 13) & (idx.hour < 16)).astype(int)
    d["Day_of_week"]      = idx.dayofweek

    d["H4_RSI"]    = d["RSI"].rolling(4).mean()
    d["H4_ATR"]    = d["ATR"].rolling(4).mean()
    d["H4_trend"]  = (d["close"] > d["close"].shift(4)).astype(int)
    d["H4_mom"]    = d["close"].pct_change(4)
    d["H4_BB_pos"] = d["BB_position"].rolling(4).mean()
    d["H4_ADX"]    = d["ADX"].rolling(4).mean()

    try:
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1h",
                          auto_adjust=False, progress=False)
        dxy.columns = [col[0].lower() if isinstance(col,tuple) else col.lower()
                       for col in dxy.columns]
        dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
        dxy["DXY_change"] = dxy["close"].pct_change()
        dxy["DXY_rsi"]    = 100-(100/(1+dxy["close"].diff().clip(lower=0).rolling(14).mean()/(-dxy["close"].diff().clip(upper=0)).rolling(14).mean()))
        d = d.join(dxy[["DXY_change","DXY_rsi"]], how="left")
        d["DXY_change"] = d["DXY_change"].fillna(0)
        d["DXY_rsi"]    = d["DXY_rsi"].fillna(50)
    except:
        d["DXY_change"] = 0
        d["DXY_rsi"]    = 50

    return d.dropna()


build_features = build_features_v2



def build_features_for_brain1v2(df):
    """Feature builder matching retrain_phase13.py column names exactly."""
    d = df.copy()
    c = d["close"]; h = d["high"]; l = d["low"]; o = d["open"]

    def ema(s,n): return s.ewm(span=n,adjust=False).mean()
    def sma(s,n): return s.rolling(n).mean()
    def std(s,n): return s.rolling(n).std()

    d["return_1"]  = c.pct_change(1)
    d["return_3"]  = c.pct_change(3)
    d["return_5"]  = c.pct_change(5)
    d["return_10"] = c.pct_change(10)

    for n in [5,10,20,50,100,200]:
        d[f"ema_{n}"]      = ema(c,n)
        d[f"c_vs_ema{n}"]  = (c - ema(c,n)) / ema(c,n)
    for n in [10,20,50]:
        d[f"sma_{n}"]      = sma(c,n)
        d[f"c_vs_sma{n}"]  = (c - sma(c,n)) / sma(c,n)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100/(1+gain/(loss+1e-9))
    for n,col in [(7,"rsi_7"),(21,"rsi_21")]:
        g = delta.clip(lower=0).rolling(n).mean()
        ls= (-delta.clip(upper=0)).rolling(n).mean()
        d[col] = 100 - 100/(1+g/(ls+1e-9))

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

    if "volume" in d.columns and d["volume"].sum()>0:
        d["volume_ratio"] = d["volume"]/(d["volume"].rolling(10).mean()+1e-9)
    else:
        d["volume_ratio"] = 1.0

    d["candle_body"] = (c-o).abs()/(h-l+1e-9)
    d["upper_wick"]  = (h-pd.concat([c,o],axis=1).max(axis=1))/(h-l+1e-9)
    d["lower_wick"]  = (pd.concat([c,o],axis=1).min(axis=1)-l)/(h-l+1e-9)
    d["candle_dir"]  = np.sign(c-o)

    d["hour"]    = df.index.hour
    d["sin_hour"]= np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"]= np.cos(2*np.pi*d["hour"]/24)
    d["dow"]     = df.index.dayofweek
    d["sin_dow"] = np.sin(2*np.pi*d["dow"]/5)
    d["cos_dow"] = np.cos(2*np.pi*d["dow"]/5)

    d["mom_5"]  = c - c.shift(5)
    d["mom_10"] = c - c.shift(10)
    d["mom_20"] = c - c.shift(20)

    d["vol_ratio_5_20"] = std(c,5)/(std(c,20)+1e-9)
    d["trend_5_20"]     = (ema(c,5)-ema(c,20))/(ema(c,20)+1e-9)
    d["trend_10_50"]    = (ema(c,10)-ema(c,50))/(ema(c,50)+1e-9)
    d["pos_in_day"]     = (c-l)/(h-l+1e-9)
    d["gap"]            = (o-c.shift(1))/(c.shift(1)+1e-9)
    d["rsi_div"]        = d["rsi_14"].diff(5)*np.sign(c.diff(5))*-1

    return d.dropna()


TELEGRAM_TOKEN = "8172828888:AAFWCvtCl1F-Kj5yOv_EFEB9vxL-ir-dD9I"
TELEGRAM_CHAT  = "7132630179"

def send_telegram(signal, price, score, b1_prob, b2_score, atr, h4=None):
    if signal not in ["BUY","SELL"]:
        return
    _tp_usd, _sl_usd, tp, sl = get_dynamic_tpsl(df, signal, price)
    icon = "🟢" if signal=="BUY" else "🔴"
    h4_line = ""
    if h4:
        h4_icon = "📈" if h4["direction"]=="BUY" else "📉" if h4["direction"]=="SELL" else "➡️"
        h4_line = f"H4     : {h4_icon} {h4['direction']} | RSI {h4['rsi']}\n"
    msg = (
        f"{icon} {signal} XAU/USD\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry  : ${price:,.2f}\n"
        f"TP     : ${tp:,.2f}  (+${round(abs(tp-price),2)})\n"
        f"SL     : ${sl:,.2f}  (-${round(abs(sl-price),2)})\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Score  : {round(score,2)}/10\n"
        f"Brain1 : {round(b1_prob,3)}\n"
        f"Brain2 : {round(b2_score,2)}/10\n"
        f"{h4_line}"
        f"ATR    : ${round(atr,2)}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Risk   : ${_sl_usd:.2f} | Reward: ${_tp_usd:.2f} | RR: {round(_tp_usd/_sl_usd,2)}:1"
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

def run_analysis():
    import pickle, json
    import requests as req
    import yfinance as yf

    # ── Phase 20: Session Filter ─────────────────────────────────────────────
    session_quality, trade_allowed = _get_session_quality()
    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"\n🕐 Session Check  : {now_utc}")
    print(f"   Quality        : {session_quality}")
    print(f"   Trade Allowed  : {'✅ YES' if trade_allowed else '🚫 NO — BLOCKED (21:00–01:00 UTC)'}")

    if not trade_allowed:
        print("\n⏸  Session blocked — no trade. Agent will resume after 01:00 UTC.")
        return {
            "signal":     "WAIT",
            "confidence": 0,
            "price":      None,
            "session_blocked": True,
            "session_quality": session_quality,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
    # ─────────────────────────────────────────────────────────────────────────

    # ── Phase 22: Daily Loss Limit ────────────────────────────────────────────
    day_blocked, day_reason, day_pnl = _check_daily_limit()
    print(f"\n💰 Daily P&L      : ${day_pnl['total_pnl']:+.2f} | Trades: {day_pnl['trade_count']}")
    print(f"   Limit Status   : {'🚫 BLOCKED — '+day_reason if day_blocked else '✅ '+day_reason}")

    if day_blocked:
        print(f"\n🛑 Trading halted for today. Resumes tomorrow 00:00 UTC.")
        return {
            "signal":       "WAIT",
            "confidence":   0,
            "price":        None,
            "day_blocked":  True,
            "day_reason":   day_reason,
            "daily_pnl":    day_pnl["total_pnl"],
            "trade_count":  day_pnl["trade_count"],
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }
    # ─────────────────────────────────────────────────────────────────────────

    # ── Health tracker — each component self-reports ok/error ────────────────
    health = {
        "data_feed":  {"ok": False, "msg": "not run"},
        "brain1":     {"ok": False, "msg": "not run"},
        "brain2":     {"ok": False, "msg": "not run"},
        "brain3":     {"ok": False, "msg": "not run"},
        "h4":         {"ok": False, "msg": "not run"},
        "telegram":   {"ok": True,  "msg": "standby"},
        "daily_limit":{"ok": True,  "msg": day_reason},
    }

    # Load model
    try:
        with open("/content/drive/MyDrive/trading_agent/brain1_xgboost_v4.pkl","rb") as f:
            brain1_model = pickle.load(f)
        with open("/content/drive/MyDrive/trading_agent/brain1_features_v4.json","r") as f:
            features = json.load(f)
        THRESHOLD = features.get("threshold", 0.74)
    except Exception as e:
        health["brain1"] = {"ok": False, "msg": f"Model load failed: {e}"}
        return {"signal":"ERROR","confidence":0,"price":None,"health":health,
                "timestamp":datetime.now(timezone.utc).isoformat()}

    # Data feed
    try:
        df = yf.download(SYMBOL, period="60d", interval="1h", auto_adjust=False, progress=False)
        df.columns = [col[0].lower() if isinstance(col,tuple) else col.lower() for col in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is None else pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index.dayofweek<5]; df = df[df["volume"]>0]
        if len(df) < 50:
            raise ValueError(f"Only {len(df)} rows — feed may be stale")
        health["data_feed"] = {"ok": True, "msg": f"{len(df)} bars loaded"}
    except Exception as e:
        health["data_feed"] = {"ok": False, "msg": f"yfinance error: {e}"}
        return {"signal":"ERROR","confidence":0,"price":None,"health":health,
                "timestamp":datetime.now(timezone.utc).isoformat()}

    # Feature engineering
    try:
        df_live = build_features_for_brain1v2(df)
        if df_live is None or len(df_live) == 0:
            raise ValueError("Feature builder returned empty dataframe")
    except Exception as e:
        health["data_feed"] = {"ok": False, "msg": f"Feature build failed: {e}"}
        return {"signal":"ERROR","confidence":0,"price":None,"health":health,
                "timestamp":datetime.now(timezone.utc).isoformat()}

    # Brain 1
    try:
        missing = [f for f in features["features"] if f not in df_live.columns]
        if missing:
            raise ValueError(f"Missing features: {missing[:3]}{'...' if len(missing)>3 else ''}")
        X = df_live[features["features"]].iloc[[-1]]
        prob = brain1_model.predict_proba(X)[0]
        b1_buy = prob[1]; b1_sell = prob[0]
        b1_dir = "BUY" if b1_buy>=THRESHOLD else "SELL" if b1_sell>=THRESHOLD else "WAIT"
        b1_score = b1_buy*10
        health["brain1"] = {"ok": True, "msg": f"prob={round(b1_buy,3)} threshold={THRESHOLD}"}
    except Exception as e:
        health["brain1"] = {"ok": False, "msg": f"Inference failed: {e}"}
        b1_buy=0.5; b1_sell=0.5; b1_dir="WAIT"; b1_score=5.0

    # Brain 2
    try:
        brain2 = Brain2Analyzer(); report = brain2.analyze(df_live)
        b2_sig = report["signal"]; b2_score = report["confluence_score"]; m = report["modules"]
        health["brain2"] = {"ok": True, "msg": f"score={b2_score} signal={b2_sig}"}
    except Exception as e:
        health["brain2"] = {"ok": False, "msg": f"Analysis failed: {e}"}
        b2_sig="WAIT"; b2_score=5.0
        m={"trend":{"direction":"UNKNOWN","strength":"UNKNOWN"},
           "structure":{"structure":"UNKNOWN"},
           "momentum":{"rsi":50,"state":"UNKNOWN"},
           "volatility":{"atr14":0},
           "sr_zones":{"nearest_support":0,"nearest_resistance":0}}
        report={"signal":"WAIT","confluence_score":5.0,"modules":m}
    # ─────────────────────────────────────────────────────────────────────────

    fused = (b1_score*0.4)+(b2_score*0.6)
    if b1_dir==b2_sig and b1_dir!="WAIT": fused += 0.5
    if fused>=6.5: sig,conf = "BUY","HIGH" if fused>=8 else "MEDIUM"
    elif fused<=3.5: sig,conf = "SELL","HIGH" if fused<=2 else "MEDIUM"
    else: sig,conf = "WAIT","LOW"

    # ── Phase 21: H4 Trend Confirmation ──────────────────────────────────────
    try:
        h4 = _get_h4_trend(df)
        h4_dir  = h4["direction"]
        h4_veto = False
        health["h4"] = {"ok": True, "msg": f"{h4_dir} | {h4['reason']}"}
    except Exception as e:
        h4 = {"direction":"NEUTRAL","reason":f"error: {e}","ema20":None,"ema50":None,"rsi":None,"atr":None,"score":5}
        h4_dir = "NEUTRAL"; h4_veto = False
        health["h4"] = {"ok": False, "msg": f"H4 resample failed: {e}"}

    if sig in ("BUY","SELL") and h4_dir != "NEUTRAL":
        if sig != h4_dir:
            h4_veto = True
            sig_before_veto = sig
            sig  = "WAIT"
            conf = "LOW"

    print(f"\n📊 H4 Confirmation : {h4_dir}  ({h4['reason']})")
    print(f"   H4 EMA20/50    : {h4['ema20']} / {h4['ema50']}")
    print(f"   H4 RSI         : {h4['rsi']}")
    if h4_veto:
        print(f"   ⚠️  H4 VETO — {sig_before_veto} signal blocked (H4={h4_dir})")
    else:
        print(f"   ✅ H4 aligned — signal passes")
    # ─────────────────────────────────────────────────────────────────────────

    price = round(df_live["close"].iloc[-1], 2)
    atr   = round(m["volatility"]["atr14"], 2)
    tp    = round(price + atr*1.5, 2) if sig=="BUY" else round(price - atr*1.5, 2)
    sl    = round(price - atr, 2)     if sig=="BUY" else round(price + atr, 2)

    # Brain 3
    prompt = ("You are an expert XAU/USD gold trading analyst.\n\n"
        "Price: $"+str(price)+"\nSignal: "+sig+" | Score: "+str(round(fused,2))+"/10\n"
        "Brain1: "+b1_dir+" | Brain2: "+b2_sig+"\n"
        "H4 Trend: "+h4_dir+" | "+h4["reason"]+"\n\n"
        "Trend: "+m["trend"]["direction"]+" "+m["trend"]["strength"]+"\n"
        "Structure: "+m["structure"]["structure"]+"\n"
        "RSI: "+str(m["momentum"]["rsi"])+" | "+m["momentum"]["state"]+"\n"
        "ATR: "+str(atr)+"\nSupport: "+str(m["sr_zones"]["nearest_support"])+"\n"
        "Resistance: "+str(m["sr_zones"]["nearest_resistance"])+"\n\n"
        "Give 5-line trading briefing.")
    try:
        resp = req.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
            json={"model":GROQ_MODEL,"max_tokens":300,"messages":[{"role":"user","content":prompt}]},timeout=15)
        data = resp.json()
        if "choices" not in data:
            raise ValueError(data.get("error", {}).get("message", str(data)))
        b3_text = data["choices"][0]["message"]["content"]
        health["brain3"] = {"ok": True, "msg": f"model={GROQ_MODEL}"}
    except Exception as e:
        b3_text = f"Brain 3 unavailable: {e}"
        health["brain3"] = {"ok": False, "msg": f"Groq API error: {e}"}

    try:
        send_telegram(sig, price, fused, b1_buy, b2_score, atr, h4)
        health["telegram"] = {"ok": True, "msg": "alert sent" if sig in ("BUY","SELL") else "standby"}
    except Exception as e:
        health["telegram"] = {"ok": False, "msg": f"Telegram failed: {e}"}

    return _to_python({
        "signal":       sig,
        "confidence":   round(fused, 2),
        "price":        price,
        "brain1":       {"prediction": 1 if b1_dir=="BUY" else 0, "probability": round(b1_buy, 3)},
        "brain2":       {"score": b2_score, "details": {"trend": m["trend"]["direction"], "momentum": m["momentum"]["state"]}},
        "brain3":       {"verdict": sig, "reasoning": b3_text},
        "h4":           h4,
        "h4_veto":      h4_veto,
        "tp":           tp,
        "sl":           sl,
        "atr":          atr,
        "session":      session_quality,
        "session_blocked": False,
        "day_blocked":  False,
        "daily_pnl":    day_pnl["total_pnl"],
        "trade_count":  day_pnl["trade_count"],
        "health":       health,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    })

def trailing_sl_check(direction, entry, current_high, current_low,
                      tp_level, sl_level, be_triggered):
    if be_triggered:
        return sl_level, True
    if direction == "BUY":
        half_tp = entry + (tp_level - entry) * 0.50
        if current_high >= half_tp:
            return entry + 0.10, True
    else:
        half_tp = entry - (entry - tp_level) * 0.50
        if current_low <= half_tp:
            return entry - 0.10, True
    return sl_level, False
