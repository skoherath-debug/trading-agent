"""
trade_coach.py — Single-trade manager for XAU/USD
═══════════════════════════════════════════════════════════════════════════════

Owns the full trade lifecycle from signal to exit. Replaces noisy
per-tick alerts with one-trade-at-a-time discipline:

  SCANNING → AWAITING_CONFIRMATION → TRADE_OPEN → TRADE_CLOSED → SCANNING

Design goals:
  • One active trade at a time (no signal spam)
  • Manual entry confirmation via Telegram YES/NO buttons
  • Mid-trade coaching: exit early if momentum reverses or SL is near
  • Dashboard-visible state so you always know "where am I"

Entry criteria (Balanced — 3-5 alerts/day):
  • Main Triple Brain signal = BUY or SELL
  • Fused confidence ≥ 7.0
  • Session = LONDON or NEW YORK (not Asian/off-hours)
  • No H4 veto
  • No daily loss limit hit
  • S/R gap ≥ $10 (not tight range)
  • At least 45 min since last trade closed (avoid revenge)

Exit criteria (while trade open):
  • TP2 hit → full profit close
  • SL hit → full loss close
  • RSI reverses hard (BUY trade: RSI > 75 & falling | SELL: RSI < 25 & rising)
  • Fused signal flips direction (BUY→SELL on next Phase B)
  • Trade age > 3 hours (time stop)
  • Price stagnant near entry for 90 min (dead trade)
"""

import os, json, time, requests
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
STATE_FILE           = "trade_state.json"
MIN_CONFIDENCE       = 7.0               # Balanced — ~3-5 alerts/day
MIN_TIME_BETWEEN_SEC = 45 * 60           # 45 min between trades
MAX_TRADE_AGE_SEC    = 3 * 60 * 60       # 3 hour time stop
STAGNANT_THRESHOLD   = 90 * 60           # 90 min of sideways = dead trade
STAGNANT_RANGE_USD   = 3.0               # if price moves <$3 in 90min → dead
DANGER_ZONE_RATIO    = 0.30              # SL warning when price within 30% of SL

# Progress update frequency — 30 second cadence for faster trade monitoring
PROGRESS_UPDATE_SEC  = 30                # every 30 sec (was 60)
# But only alert if price moved meaningfully vs last update — skip micro-wiggles
PROGRESS_MIN_MOVE    = 1.5               # USD — slightly lower than before (was 2.0) since 30s needs less move to matter
# Exception: always alert if big move since last update (regardless of direction)
PROGRESS_BIG_MOVE    = 4.0               # USD — "big" move threshold (was 5.0)

# State values
STATE_SCANNING     = "SCANNING"
STATE_AWAITING     = "AWAITING_CONFIRMATION"
STATE_OPEN         = "TRADE_OPEN"
STATE_CLOSED       = "TRADE_CLOSED"   # transient, moves to SCANNING

# Exit reasons
EXIT_TP2       = "TP2_HIT"
EXIT_TP1_MOVE  = "TP1_HIT"
EXIT_SL        = "SL_HIT"
EXIT_REVERSAL  = "MOMENTUM_REVERSED"
EXIT_FLIP      = "SIGNAL_FLIPPED"
EXIT_TIMEOUT   = "TIME_STOP"
EXIT_STAGNANT  = "STAGNANT"
EXIT_MANUAL    = "MANUAL"

# ── State persistence ─────────────────────────────────────────────────────────
def _default_state():
    return {
        "state":           STATE_SCANNING,
        "last_closed_ts":  0,        # unix timestamp of last closed trade
        # Signal details (populated when a signal fires)
        "direction":       None,     # "BUY" or "SELL"
        "entry":           None,
        "tp1":             None,
        "tp2":             None,
        "sl":              None,
        "confidence":      None,
        "signal_ts":       None,     # when the signal fired
        "signal_reason":   None,     # human-readable why
        # Active trade fields (populated on YES confirmation)
        "trade_open_ts":   None,
        "last_update_ts":  None,
        "last_update_price": None,   # price at last update alert (for big-move detection)
        "tp1_hit":         False,
        "sl_moved_to_be":  False,    # True after TP1 moves SL to entry
        "high_water":      None,     # best price seen (MFE tracking)
        "low_water":       None,
        # Awaiting confirmation fields
        "awaiting_since_ts": None,
        # Coaching alerts already sent this trade (dedupe)
        "alerts_sent":     [],
    }

def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        # Merge with defaults to handle new fields added later
        d = _default_state()
        d.update(s)
        return d
    except Exception:
        return _default_state()

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[COACH] Save error: {e}")

def reset_state_to_scanning():
    s = _default_state()
    s["last_closed_ts"] = time.time()
    save_state(s)
    return s

# ── Telegram with inline buttons ──────────────────────────────────────────────
def send_with_buttons(token, chat_id, text, buttons):
    """Send Telegram message with inline keyboard buttons.
    buttons = [[{"text":"YES","callback_data":"confirm_yes"}, ...]]"""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "reply_markup": {"inline_keyboard": buttons},
            },
            timeout=10,
        )
        return r.status_code == 200, r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"[COACH] Telegram button error: {e}")
        return False, None

def send_plain(token, chat_id, text):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[COACH] Telegram error: {e}")
        return False

# ── Entry gate — decides whether to alert on a new signal ─────────────────────
def should_alert_entry(analysis_result, state):
    """Return (ok: bool, reason: str).
    analysis_result is the dict returned by run_analysis()."""
    # Must be in scanning state — no active trade
    if state["state"] != STATE_SCANNING:
        return False, f"state={state['state']}"

    sig = analysis_result.get("signal", "WAIT")
    if sig not in ("BUY", "SELL"):
        return False, f"signal={sig}"

    conf = float(analysis_result.get("confidence", 0))
    if conf < MIN_CONFIDENCE:
        return False, f"confidence {conf} < {MIN_CONFIDENCE}"

    if analysis_result.get("h4_veto"):
        return False, "H4 veto"

    if analysis_result.get("session_blocked"):
        return False, "session blocked"

    if analysis_result.get("day_blocked"):
        return False, "daily limit"

    if analysis_result.get("tight_range"):
        return False, "tight range"

    # Time since last close
    since_last = time.time() - state.get("last_closed_ts", 0)
    if since_last < MIN_TIME_BETWEEN_SEC:
        mins_left = int((MIN_TIME_BETWEEN_SEC - since_last) / 60)
        return False, f"cooldown: {mins_left} min to go"

    # Session must be LONDON or NY
    sess = str(analysis_result.get("session", "")).upper()
    if "BLOCKED" in sess or "ASIAN" in sess or "OFF-HOURS" in sess:
        return False, "not london/NY"

    return True, "all checks passed"

# ── Format entry alert with YES/NO buttons ────────────────────────────────────
def format_entry_alert(result):
    sig    = result.get("signal", "WAIT")
    price  = result.get("price", 0)
    tp     = result.get("tp", 0)
    tp2    = result.get("tp2", 0)
    sl     = result.get("sl", 0)
    atr    = result.get("atr", 13)
    conf   = result.get("confidence", 0)
    b1_prob= (result.get("brain1") or {}).get("probability", 0)
    b2_scr = (result.get("brain2") or {}).get("score", 0)
    h4d    = (result.get("h4") or {}).get("direction", "—")
    sess   = result.get("session", "—")

    tp_usd  = round(abs(tp - price) * 3, 2)
    tp2_usd = round(abs(tp2 - price) * 3, 2)
    sl_usd  = round(abs(sl - price) * 3, 2)
    rr      = round(tp_usd / sl_usd, 1) if sl_usd > 0 else 0

    # Hold window estimate — ATR-based
    # Typical XAU trade covers 1-2 ATR in 30-120 min
    hold_min = max(int(abs(tp - price) / max(atr, 1) * 30), 30)
    hold_max = hold_min * 3

    icon = "📈" if sig == "BUY" else "📉"
    dir_word = "LONG" if sig == "BUY" else "SHORT"

    reason_parts = []
    b2_det = (result.get("brain2") or {}).get("details", {}) or {}
    if b2_det.get("trend"):    reason_parts.append(f"trend {b2_det['trend']}")
    if b2_det.get("momentum"): reason_parts.append(f"momentum {b2_det['momentum']}")
    if h4d in ("BUY","SELL"):  reason_parts.append(f"H4 {h4d}")
    reason = " • ".join(reason_parts) or "triple-brain confluence"

    msg = (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{icon} HIGH CONFIDENCE TRADE\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Direction : {sig} XAU/USD ({dir_word})\n"
        f"Entry     : ${price:,.2f}\n"
        f"TP1       : ${tp:,.2f}  (+${tp_usd})\n"
        f"TP2       : ${tp2:,.2f}  (+${tp2_usd})\n"
        f"SL        : ${sl:,.2f}  (-${sl_usd})\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Confidence : {conf:.1f}/10\n"
        f"RR        : 1:{rr}\n"
        f"Hold      : {hold_min}–{hold_max} min\n"
        f"Session   : {sess}\n"
        f"B1={b1_prob:.2f}  B2={b2_scr:.1f}/10  H4={h4d}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Why: {reason}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ DID YOU ENTER THIS TRADE?\n"
        f"Tap YES if you opened on Exness.\n"
        f"Tap NO if you skipped — bot resumes scanning."
    )
    return msg

def entry_confirmation_buttons():
    return [[
        {"text": "✅ YES, ENTERED", "callback_data": "trade_yes"},
        {"text": "❌ NO, SKIPPED",   "callback_data": "trade_no"},
    ]]

# ── When signal fires ─────────────────────────────────────────────────────────
def on_signal_fired(result, state, telegram_token, telegram_chat):
    """Called when a new BUY/SELL signal passes the entry gate.
    Moves state to AWAITING_CONFIRMATION, sends Telegram with buttons."""
    now = time.time()
    state.update({
        "state":            STATE_AWAITING,
        "direction":        result.get("signal"),
        "entry":            result.get("price"),
        "tp1":              result.get("tp"),
        "tp2":              result.get("tp2"),
        "sl":               result.get("sl"),
        "confidence":       result.get("confidence"),
        "signal_ts":        now,
        "awaiting_since_ts": now,
        "signal_reason":    result.get("signal") + " @ conf " + str(result.get("confidence")),
        "alerts_sent":      [],
    })
    save_state(state)

    text = format_entry_alert(result)
    buttons = entry_confirmation_buttons()
    ok, _ = send_with_buttons(telegram_token, telegram_chat, text, buttons)
    print(f"[COACH] Entry alert sent (buttons={ok}): {result.get('signal')} @ ${result.get('price')}")
    return state

# ── On YES button tap ─────────────────────────────────────────────────────────
def on_user_confirmed_yes(state, telegram_token, telegram_chat):
    """User tapped YES — mark trade open, start mid-trade coaching."""
    if state["state"] != STATE_AWAITING:
        return state
    now = time.time()
    state.update({
        "state":          STATE_OPEN,
        "trade_open_ts":  now,
        "last_update_ts": now,
        "last_update_price": state["entry"],
        "tp1_hit":        False,
        "sl_moved_to_be": False,
        "high_water":     state["entry"],
        "low_water":      state["entry"],
    })
    save_state(state)

    msg = (
        f"✅ TRADE TRACKING STARTED\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{state['direction']} @ ${state['entry']:.2f}\n"
        f"TP1: ${state['tp1']:.2f}  TP2: ${state['tp2']:.2f}\n"
        f"SL:  ${state['sl']:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"I'll watch the market and alert you when:\n"
        f"• TP1 hits → move SL to breakeven\n"
        f"• TP2 hits → close full profit\n"
        f"• Price nears SL → danger warning\n"
        f"• Momentum reverses → exit early\n"
        f"No other trades until this one closes."
    )
    send_plain(telegram_token, telegram_chat, msg)
    return state

# ── On NO button tap ──────────────────────────────────────────────────────────
def on_user_confirmed_no(state, telegram_token, telegram_chat):
    """User tapped NO — discard signal, resume scanning, no cooldown."""
    if state["state"] != STATE_AWAITING:
        return state
    direction = state.get("direction", "—")
    entry     = state.get("entry", 0)
    state = _default_state()
    state["last_closed_ts"] = 0   # NO skip → no cooldown, scan immediately
    save_state(state)
    send_plain(
        telegram_token, telegram_chat,
        f"❌ Signal skipped ({direction} @ ${entry:.2f}).\n"
        f"Scanning for next setup. No cooldown applied.",
    )
    return state

# ── Auto-timeout if user doesn't tap YES/NO within 5 minutes ──────────────────
AWAITING_TIMEOUT_SEC = 5 * 60

def check_awaiting_timeout(state, telegram_token, telegram_chat):
    """If user didn't confirm within 5 min, assume they didn't enter."""
    if state["state"] != STATE_AWAITING:
        return state
    elapsed = time.time() - state.get("awaiting_since_ts", 0)
    if elapsed > AWAITING_TIMEOUT_SEC:
        direction = state.get("direction", "—")
        state = _default_state()
        save_state(state)
        send_plain(
            telegram_token, telegram_chat,
            f"⏱️ Entry window expired ({direction}) — no confirmation in 5min.\n"
            f"Assumed skipped. Scanning for next setup.",
        )
    return state

# ── Mid-trade monitoring ──────────────────────────────────────────────────────
def _alert_once(state, key, token, chat, msg):
    """Send alert only if this specific alert hasn't been sent this trade."""
    if key in state.get("alerts_sent", []):
        return False
    state.setdefault("alerts_sent", []).append(key)
    save_state(state)
    return send_plain(token, chat, msg)

def check_trade_open(state, current_price, analysis_result, rw_result, token, chat):
    """Called every scheduler tick when state == TRADE_OPEN.
    Decides whether to alert user on price action, momentum shift, or exit."""
    if state["state"] != STATE_OPEN:
        return state

    direction = state["direction"]
    entry     = state["entry"]
    tp1       = state["tp1"]
    tp2       = state["tp2"]
    sl        = state["sl"]
    open_ts   = state["trade_open_ts"]
    age_sec   = time.time() - open_ts
    age_min   = int(age_sec / 60)

    # Track MFE / MAE
    if direction == "BUY":
        state["high_water"] = max(state.get("high_water", entry), current_price)
        state["low_water"]  = min(state.get("low_water",  entry), current_price)
    else:
        state["high_water"] = max(state.get("high_water", entry), current_price)
        state["low_water"]  = min(state.get("low_water",  entry), current_price)

    pnl_pts = (current_price - entry) if direction == "BUY" else (entry - current_price)
    pnl_usd = round(pnl_pts * 3, 2)

    # ── EXIT: TP2 HIT (full close) ────────────────────────────────────────
    tp2_hit = (direction == "BUY" and current_price >= tp2) or \
              (direction == "SELL" and current_price <= tp2)
    if tp2_hit:
        send_plain(token, chat, _build_close_msg(state, current_price, pnl_usd, age_min, EXIT_TP2))
        return reset_state_to_scanning()

    # ── EXIT: SL HIT (full close) ─────────────────────────────────────────
    sl_hit = (direction == "BUY" and current_price <= sl) or \
             (direction == "SELL" and current_price >= sl)
    if sl_hit:
        send_plain(token, chat, _build_close_msg(state, current_price, pnl_usd, age_min, EXIT_SL))
        return reset_state_to_scanning()

    # ── MILESTONE: TP1 HIT (partial / BE move) ────────────────────────────
    tp1_hit_now = (direction == "BUY" and current_price >= tp1) or \
                  (direction == "SELL" and current_price <= tp1)
    if tp1_hit_now and not state["tp1_hit"]:
        state["tp1_hit"]        = True
        state["sl_moved_to_be"] = True
        state["sl"]             = entry   # SL moves to breakeven
        save_state(state)
        _alert_once(state, "tp1_hit", token, chat,
            f"🎯 TP1 HIT — ${tp1:.2f}\n"
            f"Current: ${current_price:.2f}  (+${pnl_usd:+.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"ACTION: Move SL to breakeven ${entry:.2f}\n"
            f"Take partial profit (50%) optional.\n"
            f"Hold remainder for TP2 at ${tp2:.2f}."
        )
        return state

    # ── TIME STOP ─────────────────────────────────────────────────────────
    if age_sec > MAX_TRADE_AGE_SEC:
        send_plain(token, chat, _build_close_msg(state, current_price, pnl_usd, age_min, EXIT_TIMEOUT))
        return reset_state_to_scanning()

    # ── STAGNANT (90 min, tiny movement) ──────────────────────────────────
    total_move = abs(state["high_water"] - state["low_water"])
    if age_sec > STAGNANT_THRESHOLD and total_move < STAGNANT_RANGE_USD:
        send_plain(token, chat,
            f"😴 TRADE STAGNANT — {age_min} min\n"
            f"Range: ${total_move:.2f}  P&L: ${pnl_usd:+.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Market not moving. Consider manual close.\n"
            f"Reply /exit on Exness to clear for next setup."
        )
        return reset_state_to_scanning()

    # ── DANGER: approaching SL ────────────────────────────────────────────
    sl_dist = abs(current_price - sl)
    sl_total_dist = abs(entry - sl)
    if sl_total_dist > 0 and (sl_dist / sl_total_dist) < DANGER_ZONE_RATIO:
        _alert_once(state, "danger_sl", token, chat,
            f"⚠️ DANGER — SL APPROACHING\n"
            f"Price: ${current_price:.2f}  SL: ${sl:.2f}\n"
            f"Distance: ${sl_dist:.2f} (of ${sl_total_dist:.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"P&L: ${pnl_usd:+.2f}\n"
            f"Consider manual close if momentum continues against.\n"
            f"Bot will alert on SL hit."
        )

    # ── SIGNAL FLIP (Phase B re-ran and direction reversed) ───────────────
    if analysis_result:
        new_sig = analysis_result.get("signal", "WAIT")
        if new_sig in ("BUY", "SELL") and new_sig != direction:
            new_conf = float(analysis_result.get("confidence", 0))
            if new_conf >= 6.5:   # must be meaningful, not a weak flip
                send_plain(token, chat,
                    f"🔄 SIGNAL REVERSED\n"
                    f"Your trade: {direction}\n"
                    f"New signal: {new_sig} (conf {new_conf:.1f})\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"Current: ${current_price:.2f}  P&L: ${pnl_usd:+.2f}\n"
                    f"Consider closing manually before SL.\n"
                    f"Bot will continue tracking until TP/SL."
                )
                _alert_once(state, "signal_flip", token, chat, "")  # dedupe

    # ── MOMENTUM REVERSAL (RSI or Phase 30 waterfall/rocket opposite) ─────
    if rw_result:
        rsi = rw_result.get("rsi", 50)
        rocket    = rw_result.get("rocket", 0)
        waterfall = rw_result.get("waterfall", 0)

        # BUY trade: worry if waterfall ≥ 60
        if direction == "BUY" and waterfall >= 60:
            _alert_once(state, "reversal_buy", token, chat,
                f"⚠️ MOMENTUM REVERSING AGAINST YOU\n"
                f"Trade: BUY @ ${entry:.2f}\n"
                f"Waterfall score: {waterfall}/100  RSI: {rsi:.1f}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Current: ${current_price:.2f}  P&L: ${pnl_usd:+.2f}\n"
                f"Consider manual close. Bot still holds to TP/SL."
            )
        elif direction == "SELL" and rocket >= 60:
            _alert_once(state, "reversal_sell", token, chat,
                f"⚠️ MOMENTUM REVERSING AGAINST YOU\n"
                f"Trade: SELL @ ${entry:.2f}\n"
                f"Rocket score: {rocket}/100  RSI: {rsi:.1f}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Current: ${current_price:.2f}  P&L: ${pnl_usd:+.2f}\n"
                f"Consider manual close. Bot still holds to TP/SL."
            )

    # ── PERIODIC PROGRESS (every 1 min, but only on meaningful moves) ────
    last_upd    = state.get("last_update_ts", 0)
    last_upd_px = state.get("last_update_price", entry)
    time_since  = time.time() - last_upd
    move_since  = abs(current_price - last_upd_px)

    # Three conditions that trigger an update:
    #   1. 1 min passed AND price moved ≥ $2 (meaningful)
    #   2. Price moved ≥ $5 regardless of time (big move — always alert)
    #   3. First update ever for this trade at the 1-min mark
    should_update = (
        (time_since >= PROGRESS_UPDATE_SEC and move_since >= PROGRESS_MIN_MOVE) or
        (move_since >= PROGRESS_BIG_MOVE)
    )

    if should_update:
        state["last_update_ts"]    = time.time()
        state["last_update_price"] = current_price
        save_state(state)

        # Price direction since last update
        if current_price > last_upd_px + 0.1:
            move_arrow = "▲"
            move_txt   = f"+${move_since:.2f}"
        elif current_price < last_upd_px - 0.1:
            move_arrow = "▼"
            move_txt   = f"-${move_since:.2f}"
        else:
            move_arrow = "◆"
            move_txt   = "flat"

        # Big move flag
        big_flag = " 🔥" if move_since >= PROGRESS_BIG_MOVE else ""

        # Trade progress
        status = "🟢 profit" if pnl_usd > 0 else "🔴 loss" if pnl_usd < 0 else "⚪ flat"

        # Distance to TP / SL
        if direction == "BUY":
            dist_tp1 = tp1 - current_price
            dist_sl  = current_price - state["sl"]
        else:
            dist_tp1 = current_price - tp1
            dist_sl  = state["sl"] - current_price

        send_plain(token, chat,
            f"{move_arrow} UPDATE {age_min}m{big_flag}\n"
            f"${current_price:.2f}  ({move_txt} since last)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{direction} @ ${entry:.2f}\n"
            f"P&L: ${pnl_usd:+.2f}  {status}\n"
            f"TP1: ${tp1:.2f} ({dist_tp1:+.2f}) {'✅' if state['tp1_hit'] else '⏳'}\n"
            f"SL:  ${state['sl']:.2f} ({dist_sl:+.2f}) {'(BE)' if state['sl_moved_to_be'] else ''}"
        )

    save_state(state)
    return state

def _build_close_msg(state, exit_price, pnl_usd, age_min, reason):
    direction = state["direction"]
    entry     = state["entry"]
    emoji = "✅" if pnl_usd > 0 else "❌" if pnl_usd < -5 else "⚪"
    return (
        f"{emoji} TRADE CLOSED — {reason}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{direction}: ${entry:.2f} → ${exit_price:.2f}\n"
        f"P&L: ${pnl_usd:+.2f}\n"
        f"Duration: {age_min} min\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Scanner resumed. Next alert when setup appears.\n"
        f"45 min cooldown before next trade."
    )
