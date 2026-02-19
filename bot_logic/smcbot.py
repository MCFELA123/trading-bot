import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger()

# ---------------- MT5 CONFIG ----------------
MT5_LOGIN = 100317612
MT5_PASSWORD = "*f7tNfQb"
MT5_SERVER = "MetaQuotes-Demo"

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
CHECK_INTERVAL = 2
MAGIC = 202501

# ---------------- RISK - CONSERVATIVE SETTINGS ----------------
RISK_PERCENT = 0.5  # 0.5% risk per trade - PROTECT CAPITAL!
STOPLOSS_PIPS = 25  # Wider SL for breathing room
TAKEPROFIT_PIPS = 50  # 1:2 RR ratio
MAX_POSITIONS = 1  # ONLY 1 position at a time!

# ---------------- INITIALIZE ----------------
def initialize():
    if not mt5.initialize():
        log.error("MT5 init failed")
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        log.error("Login failed")
        return False
    mt5.symbol_select(SYMBOL, True)
    log.info("✅ MT5 Connected")
    return True

# ---------------- DATA ----------------
def get_data(symbol, timeframe, bars=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# ---------------- TREND (HTF EMA) ----------------
def trend_bias(df):
    ema = df["close"].ewm(span=50).mean()
    if df["close"].iloc[-1] > ema.iloc[-1]:
        return "BULLISH"
    elif df["close"].iloc[-1] < ema.iloc[-1]:
        return "BEARISH"
    return "NEUTRAL"

# ---------------- LIQUIDITY GRAB ----------------
def liquidity_grab(df):
    high = df["high"]
    low = df["low"]

    sweep_high = high.iloc[-2] > high.iloc[-3] and high.iloc[-2] > high.iloc[-4]
    sweep_low = low.iloc[-2] < low.iloc[-3] and low.iloc[-2] < low.iloc[-4]

    return sweep_high, sweep_low

# ---------------- ORDER BLOCK ----------------
def order_block(df):
    last = df.iloc[-2]

    # Bullish OB = last down candle before impulse
    if last.close < last.open:
        return "BULLISH", last.low, last.high

    # Bearish OB = last up candle before dump
    if last.close > last.open:
        return "BEARISH", last.low, last.high

    return None, None, None

# ---------------- FAIR VALUE GAP ----------------
def fair_value_gap(df):
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    # Bullish FVG
    if c1.high < c3.low:
        return "BULLISH", c1.high, c3.low

    # Bearish FVG
    if c1.low > c3.high:
        return "BEARISH", c3.high, c1.low

    return None, None, None

# ---------------- LOT ----------------
def calc_lot(balance):
    risk_money = balance * (RISK_PERCENT / 100)
    lot = risk_money / (STOPLOSS_PIPS * 10)
    return max(0.01, round(lot, 2))

# ---------------- ORDER ----------------
def send_order(order_type, lot, sl, tp):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC,
        "deviation": 20,
        "comment": "SMC BOT"
    }

    res = mt5.order_send(req)
    log.info(f"📌 ORDER RESULT → {res.retcode}")
    return res

# ---------------- MAIN ----------------
def run_bot():
    if not initialize():
        return

    log.info("🚀 SMC BOT STARTED")

    while True:
        df = get_data(SYMBOL, TIMEFRAME)
        if df is None or len(df) < 100:
            time.sleep(1)
            continue

        trend = trend_bias(df)
        sweep_high, sweep_low = liquidity_grab(df)
        ob_type, ob_low, ob_high = order_block(df)
        fvg_type, fvg_low, fvg_high = fair_value_gap(df)

        price = df["close"].iloc[-1]
        acc = mt5.account_info()
        positions = mt5.positions_get(symbol=SYMBOL)
        count = len(positions) if positions else 0
        lot = calc_lot(acc.balance)
        point = mt5.symbol_info(SYMBOL).point

        log.info(f"Trend={trend} OB={ob_type} FVG={fvg_type} SweepH={sweep_high} SweepL={sweep_low}")

        # ---------------- BUY ----------------
        if (
            trend == "BULLISH"
            and sweep_low
            and (ob_type == "BULLISH" or fvg_type == "BULLISH")
            and count < MAX_POSITIONS
        ):
            sl = price - STOPLOSS_PIPS * point * 10
            tp = price + TAKEPROFIT_PIPS * point * 10
            send_order(mt5.ORDER_TYPE_BUY, lot, sl, tp)
            log.info("🟢 BUY (SMC CONFIRMED)")

        # ---------------- SELL ----------------
        if (
            trend == "BEARISH"
            and sweep_high
            and (ob_type == "BEARISH" or fvg_type == "BEARISH")
            and count < MAX_POSITIONS
        ):
            sl = price + STOPLOSS_PIPS * point * 10
            tp = price - TAKEPROFIT_PIPS * point * 10
            send_order(mt5.ORDER_TYPE_SELL, lot, sl, tp)
            log.info("🔴 SELL (SMC CONFIRMED)")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_bot()

