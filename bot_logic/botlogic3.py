import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import threading
import logging
from collections import defaultdict
from datetime import datetime

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- PER-USER BOT STORAGE ----------------
user_bots = defaultdict(dict)  # {username: {"thread":..., "running": True/False, "stop_event":...}}

# ---------------- MT5 LOGIN CONFIG ----------------
MT5_LOGIN = 5045838773
MT5_PASSWORD = "*kCo7qKz"
MT5_SERVER = "MetaQuotes-Demo"

# ---------------- BOT CONFIGURATION ----------------
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
FAST_EMA = 5
SLOW_EMA = 13
HTF_EMA = 21
RSI_PERIOD = 14
STOCH_K = 5
STOCH_D = 3
ATR_PERIOD = 14
ATR_THRESHOLD = 5
RISK_PERCENT = 1.0
STOPLOSS_PIPS = 20
TAKEPROFIT_PIPS = 40
CHECK_INTERVAL = 2
MAGIC = 202501
MAX_POSITIONS = 3
USE_TRAILING_STOP = True
TRAILING_DISTANCE = 15
CONFIDENCE_THRESHOLD = 0.5

# ---------------- GLOBAL TRADE VARIABLES ----------------
trade_stats = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
current_trends = defaultdict(lambda: "NEUTRAL")  # per-user trend

# ---------------- MT5 INITIALIZATION ----------------
def initialize_mt5():
    """Initialize MT5 connection (shared for all users)"""
    if not mt5.initialize():
        logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logger.error(f"❌ Login failed: {mt5.last_error()}")
        return False
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"❌ Failed to select {SYMBOL}")
        return False
    logger.info("✅ MT5 initialized successfully")
    return True

# ---------------- DATA & INDICATORS ----------------
def get_data(symbol, timeframe, n=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def add_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=FAST_EMA).mean()
    df["ema_slow"] = df["close"].ewm(span=SLOW_EMA).mean()
    
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=RSI_PERIOD-1, adjust=False).mean()
    ema_down = down.ewm(com=RSI_PERIOD-1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + ema_up / ema_down))
    
    low_min = df['low'].rolling(window=STOCH_K).min()
    high_max = df['high'].rolling(window=STOCH_K).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stoch_d'] = df['stoch_k'].rolling(window=STOCH_D).mean()
    
    high_low = df['high'] - df['low']
    high_prev = (df['high'] - df['close'].shift(1)).abs()
    low_prev = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

# ---------------- TREND ----------------
def get_trend_direction(user):
    df_htf = get_data(SYMBOL, mt5.TIMEFRAME_M15, 50)
    if df_htf is None:
        return current_trends[user]
    df_htf["ema_htf"] = df_htf["close"].ewm(span=HTF_EMA).mean()
    close = df_htf["close"].iloc[-1]
    ema = df_htf["ema_htf"].iloc[-1]
    new_trend = "NEUTRAL"
    if close > ema * 1.001:
        new_trend = "BULLISH"
    elif close < ema * 0.999:
        new_trend = "BEARISH"
    if new_trend != current_trends[user]:
        logger.info(f"[{user}] TREND CHANGE: {current_trends[user]} → {new_trend}")
        current_trends[user] = new_trend
    return current_trends[user]

# ---------------- SIGNAL & LOT ----------------
def calculate_signal_strength(signals):
    return sum(signals)/len(signals) if signals else 0.0

def calculate_lot(balance, risk_percent, sl_pips, confidence=1.0):
    risk_money = balance * (risk_percent/100) * confidence
    pip_value = 10
    lot = risk_money / (sl_pips * pip_value)
    return max(0.01, min(1.0, round(lot, 2)))

# ---------------- ORDER ----------------
def send_order(symbol, order_type, lot, sl, tp, confidence, signal_type):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("❌ No tick data")
        return None
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC,
        "comment": f"{signal_type}:{confidence:.2f}",
        "deviation": 20
    }
    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ {signal_type} EXECUTED lot={lot} price={price:.5f}")
        trade_stats['total_trades'] += 1
    else:
        logger.error(f"❌ {signal_type} FAILED: {result.retcode if result else 'No result'}")
    return result

def place_order(symbol, order_type, lot, sl, tp, confidence=1.0, signal_type="MANUAL"):
    order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type.lower()=="buy" else mt5.ORDER_TYPE_SELL
    return send_order(symbol, order_type_mt5, lot, sl, tp, confidence, signal_type)

# ---------------- TRAILING & CLOSE ----------------
def manage_trailing_stops(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions: return
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None: return
    point = symbol_info.point
    min_stop = max(symbol_info.trade_stops_level*point, point*10)
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None: continue
        entry = pos.price_open
        sl = pos.sl
        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = tick.bid - TRAILING_DISTANCE*point*10
            if (new_sl>sl or sl==0) and new_sl>=entry and (tick.bid-new_sl)>=min_stop:
                mt5.order_send({"action":mt5.TRADE_ACTION_SLTP,"position":pos.ticket,"sl":new_sl,"tp":pos.tp})
        else:
            new_sl = tick.ask + TRAILING_DISTANCE*point*10
            if (new_sl<sl or sl==0) and new_sl<=entry and (new_sl-tick.ask)>=min_stop:
                mt5.order_send({"action":mt5.TRADE_ACTION_SLTP,"position":pos.ticket,"sl":new_sl,"tp":pos.tp})

def close_opposite_positions(trend):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return
    for pos in positions:
        close_flag = (trend=="BEARISH" and pos.type==mt5.POSITION_TYPE_BUY) or (trend=="BULLISH" and pos.type==mt5.POSITION_TYPE_SELL)
        if close_flag:
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                mt5.order_send({
                    "action":mt5.TRADE_ACTION_DEAL,
                    "symbol":SYMBOL,
                    "volume":pos.volume,
                    "type":mt5.ORDER_TYPE_SELL if pos.type==mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position":pos.ticket,
                    "price":tick.bid if pos.type==mt5.POSITION_TYPE_BUY else tick.ask,
                    "magic":MAGIC,
                    "deviation":20
                })

# ---------------- BOT LOOP ----------------
def run_bot(user):
    stop_event = user_bots[user]["stop_event"]
    if not initialize_mt5():
        user_bots[user]["running"] = False
        return
    logger.info(f"[{user}] 🚀 Bot started on {SYMBOL} M5")
    while not stop_event.is_set():
        df = get_data(SYMBOL, TIMEFRAME)
        if df is None or len(df)<100:
            stop_event.wait(1)
            continue
        df = add_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        trend = get_trend_direction(user)
        
        # Signals
        ema_bull = prev.ema_fast <= prev.ema_slow and last.ema_fast>last.ema_slow
        ema_bear = prev.ema_fast >= prev.ema_slow and last.ema_fast<last.ema_slow
        rsi_bull = 25<last.rsi<60
        rsi_bear = 40<last.rsi<75
        stoch_bull = last.stoch_k>last.stoch_d and last.stoch_k<75
        stoch_bear = last.stoch_k<last.stoch_d and last.stoch_k>25
        macd_bull = last.macd>last.macd_signal
        macd_bear = last.macd<last.macd_signal
        atr_ok = last.atr>ATR_THRESHOLD
        
        bull_strength = calculate_signal_strength([ema_bull,rsi_bull,stoch_bull,macd_bull,atr_ok])
        bear_strength = calculate_signal_strength([ema_bear,rsi_bear,stoch_bear,macd_bear,atr_ok])
        buy_signal = bull_strength>=CONFIDENCE_THRESHOLD and trend in ["BULLISH","NEUTRAL"]
        sell_signal = bear_strength>=CONFIDENCE_THRESHOLD and trend in ["BEARISH","NEUTRAL"]
        
        logger.info(f"[{user}] 📊 Close={last.close:.5f} RSI={last.rsi:.1f} Trend={trend} Bull={bull_strength:.2f} Bear={bear_strength:.2f}")
        
        positions = mt5.positions_get(symbol=SYMBOL)
        current_pos = len(positions) if positions else 0
        acc = mt5.account_info()
        if acc:
            point = mt5.symbol_info(SYMBOL).point if mt5.symbol_info(SYMBOL) else 0.00001
            close_opposite_positions(trend)
            if USE_TRAILING_STOP and positions:
                manage_trailing_stops(SYMBOL)
            if buy_signal and current_pos<MAX_POSITIONS:
                sl = last.close - STOPLOSS_PIPS*point*10
                tp = last.close + TAKEPROFIT_PIPS*point*10
                lot = calculate_lot(acc.balance,RISK_PERCENT,STOPLOSS_PIPS,bull_strength)
                send_order(SYMBOL, mt5.ORDER_TYPE_BUY, lot, sl, tp, bull_strength, "BUY")
            elif sell_signal and current_pos<MAX_POSITIONS:
                sl = last.close + STOPLOSS_PIPS*point*10
                tp = last.close - TAKEPROFIT_PIPS*point*10
                lot = calculate_lot(acc.balance,RISK_PERCENT,STOPLOSS_PIPS,bear_strength)
                send_order(SYMBOL, mt5.ORDER_TYPE_SELL, lot, sl, tp, bear_strength, "SELL")
        stop_event.wait(CHECK_INTERVAL)
    user_bots[user]["running"] = False
    logger.info(f"[{user}] 🛑 Bot stopped")

# ---------------- BOT CONTROL (ORIGINAL NAMES) ----------------
def start_bot(user):
    if user in user_bots and user_bots[user].get("running"):
        return "Bot already running"
    stop_event = threading.Event()
    thread = threading.Thread(target=run_bot, args=(user,), daemon=True)
    user_bots[user] = {"thread": thread, "stop_event": stop_event, "running": True}
    thread.start()
    return "Bot started"

def stop_bot(user):
    if user in user_bots and user_bots[user].get("running"):
        user_bots[user]["stop_event"].set()
        return "Bot stopping..."
    return "Bot not running"

def bot_status(user):
    return user_bots[user]["running"] if user in user_bots else False

# ---------------- DASHBOARD HELPERS ----------------
def get_account_info():
    acc = mt5.account_info()
    if acc:
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free
        }
    return {}

def get_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    data = []
    if positions:
        for p in positions:
            data.append({
                "ticket": p.ticket,
                "type": "BUY" if p.type==mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit
            })
    return data
