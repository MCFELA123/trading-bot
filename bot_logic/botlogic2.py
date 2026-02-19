import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone
import threading

# ---------------- SETUP LOGGING ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- MT5 LOGIN CONFIGURATION ----------------
MT5_LOGIN = 100317612
MT5_PASSWORD = "*f7tNfQb"
MT5_SERVER = "MetaQuotes-Demo"

# ---------------- FIXED BOT CONFIGURATION ----------------
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
CONFIDENCE_THRESHOLD = 0.5  # Lower threshold for balanced signals

# Global trade stats
trade_stats = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
current_trend = "NEUTRAL"  # Track current trend direction

# Bot thread control
bot_thread = None
bot_running = False

# ---------------- ENHANCED INITIALIZATION ----------------
def initialize():
    logger.info("⏳ Initializing Enhanced MT5 Bot...")
    if not mt5.initialize():
        logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        logger.error(f"❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"❌ Failed to select {SYMBOL}")
        mt5.shutdown()
        return False
        
    logger.info("✅ Enhanced MT5 Bot initialized!")
    return True

# ---------------- TECHNICAL INDICATORS ----------------
def add_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=FAST_EMA).mean()
    df["ema_slow"] = df["close"].ewm(span=SLOW_EMA).mean()
    
    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=RSI_PERIOD-1, adjust=False).mean()
    ema_down = down.ewm(com=RSI_PERIOD-1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + ema_up / ema_down))
    
    # Stochastic
    low_min = df['low'].rolling(window=STOCH_K).min()
    high_max = df['high'].rolling(window=STOCH_K).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stoch_d'] = df['stoch_k'].rolling(window=STOCH_D).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_prev = (df['high'] - df['close'].shift(1)).abs()
    low_prev = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

# ---------------- DATA FETCHING ----------------
def get_data(symbol, timeframe, n=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# ---------------- TREND DETECTION ----------------
def get_trend_direction(symbol):
    global current_trend
    df_htf = get_data(symbol, mt5.TIMEFRAME_M15, 50)
    if df_htf is None:
        return current_trend
    
    df_htf["ema_htf"] = df_htf["close"].ewm(span=HTF_EMA).mean()
    current_close = df_htf["close"].iloc[-1]
    ema_value = df_htf["ema_htf"].iloc[-1]
    
    if current_close > ema_value * 1.001:
        new_trend = "BULLISH"
    elif current_close < ema_value * 0.999:
        new_trend = "BEARISH"
    else:
        new_trend = "NEUTRAL"
    
    if new_trend != current_trend:
        logger.info(f"🔄 TREND CHANGE: {current_trend} → {new_trend}")
        current_trend = new_trend
    
    return current_trend

# ---------------- SIGNAL STRENGTH ----------------
def calculate_signal_strength(signals):
    return sum(signals) / len(signals) if signals else 0.0

# ---------------- LOT SIZING ----------------
def calculate_lot(balance, risk_percent, sl_pips, confidence=1.0):
    risk_money = balance * (risk_percent / 100) * confidence
    pip_value_per_lot = 10
    lot = risk_money / (sl_pips * pip_value_per_lot)
    return max(0.01, min(1.0, round(lot, 2)))

# ---------------- ORDER MANAGEMENT ----------------
def send_order(symbol, order_type, lot, sl, tp, confidence, signal_type):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("❌ No tick data available")
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
    logger.info(f"Order sent: {signal_type} lot={lot} price={price:.5f} sl={sl:.5f} tp={tp:.5f}")
    return result

# ---------------- MANUAL ORDER ----------------
def place_order(symbol, order_type, lot, sl, tp, confidence, signal_type):
    if order_type.lower() == "buy":
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    else:
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    return send_order(symbol, order_type_mt5, lot, sl, tp, confidence, signal_type)

# ---------------- TRAILING STOP ----------------
def manage_trailing_stops(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return
    point = symbol_info.point
    min_stop_level = max(symbol_info.trade_stops_level * point, point * 10)

    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        entry_price = pos.price_open
        current_sl = pos.sl
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = tick.bid - TRAILING_DISTANCE * point * 10
            if (new_sl > current_sl or current_sl == 0) and new_sl >= entry_price and (tick.bid - new_sl) >= min_stop_level:
                modify_request = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": new_sl, "tp": pos.tp}
                mt5.order_send(modify_request)
        else:
            new_sl = tick.ask + TRAILING_DISTANCE * point * 10
            if (new_sl < current_sl or current_sl == 0) and new_sl <= entry_price and (new_sl - tick.ask) >= min_stop_level:
                modify_request = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": new_sl, "tp": pos.tp}
                mt5.order_send(modify_request)

# ---------------- CLOSE OPPOSITE ----------------
def close_opposite_positions(current_trend):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return
    
    for pos in positions:
        should_close = False
        if current_trend == "BEARISH" and pos.type == mt5.POSITION_TYPE_BUY:
            should_close = True
        elif current_trend == "BULLISH" and pos.type == mt5.POSITION_TYPE_SELL:
            should_close = True
        
        if should_close:
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask,
                    "magic": MAGIC,
                    "deviation": 20
                }
                mt5.order_send(close_request)



# ---------------- BOT LOGIC ----------------
def run_bot():
    global current_trend, bot_running
    if not initialize():
        return
        
    logger.info(f"🚀 FIXED Bot Started on {SYMBOL} - M5 Timeframe")
    
    try:
        while bot_running:
            df = get_data(SYMBOL, TIMEFRAME)
            if df is None or len(df) < 100:
                time.sleep(1)
                continue

            df = add_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            trend = get_trend_direction(SYMBOL)
            
            # Signals
            ema_bull = prev.ema_fast <= prev.ema_slow and last.ema_fast > last.ema_slow
            ema_bear = prev.ema_fast >= prev.ema_slow and last.ema_fast < last.ema_slow
            
            rsi_bull = last.rsi < 60 and last.rsi > 25
            rsi_bear = last.rsi > 40 and last.rsi < 75
            
            stoch_bull = last.stoch_k > last.stoch_d and last.stoch_k < 75
            stoch_bear = last.stoch_k < last.stoch_d and last.stoch_k > 25
            
            macd_bull = last.macd > last.macd_signal
            macd_bear = last.macd < last.macd_signal
            
            atr_ok = last.atr > ATR_THRESHOLD

            bull_signals = [ema_bull, rsi_bull, stoch_bull, macd_bull, atr_ok]
            bear_signals = [ema_bear, rsi_bear, stoch_bear, macd_bear, atr_ok]
            
            bull_strength = calculate_signal_strength(bull_signals)
            bear_strength = calculate_signal_strength(bear_signals)
            
            trend_ok_bull = trend in ["BULLISH", "NEUTRAL"]
            trend_ok_bear = trend in ["BEARISH", "NEUTRAL"]
            
            buy_signal = bull_strength >= CONFIDENCE_THRESHOLD and trend_ok_bull
            sell_signal = bear_strength >= CONFIDENCE_THRESHOLD and trend_ok_bear

            logger.info(f"📊 Close={last.close:.5f} RSI={last.rsi:.1f} Trend={trend} Bull={bull_strength:.2f} Bear={bear_strength:.2f}")

            positions = mt5.positions_get(symbol=SYMBOL)
            current_positions = len(positions) if positions else 0
            acc = mt5.account_info()
            
            if acc:
                symbol_info = mt5.symbol_info(SYMBOL)
                point = symbol_info.point if symbol_info else 0.00001
                
                close_opposite_positions(trend)
                
                if USE_TRAILING_STOP and positions:
                    manage_trailing_stops(SYMBOL)
                
                if buy_signal and current_positions < MAX_POSITIONS:
                    sl = last.close - STOPLOSS_PIPS * point * 10
                    tp = last.close + TAKEPROFIT_PIPS * point * 10
                    lot = calculate_lot(acc.balance, RISK_PERCENT, STOPLOSS_PIPS, bull_strength)
                    send_order(SYMBOL, mt5.ORDER_TYPE_BUY, lot, sl, tp, bull_strength, "BUY")
                
                elif sell_signal and current_positions < MAX_POSITIONS:
                    sl = last.close + STOPLOSS_PIPS * point * 10
                    tp = last.close - TAKEPROFIT_PIPS * point * 10
                    lot = calculate_lot(acc.balance, RISK_PERCENT, STOPLOSS_PIPS, bear_strength)
                    send_order(SYMBOL, mt5.ORDER_TYPE_SELL, lot, sl, tp, bear_strength, "SELL")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    finally:
        mt5.shutdown()
        logger.info("🔌 MT5 shutdown complete")

# ---------------- THREADING ----------------
def start_bot():
    global bot_thread, bot_running
    if bot_running:
        return "Bot already running"
    
    bot_running = True
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    return "Bot started"

def stop_bot():
    global bot_running
    if bot_running:
        bot_running = False
        return "Bot stopping..."
    return "Bot not running"
