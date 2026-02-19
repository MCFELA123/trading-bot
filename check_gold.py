import MetaTrader5 as mt5
import pandas as pd
import numpy as np

mt5.initialize()

# Get same data as backtest
rates = mt5.copy_rates_from_pos('XAUUSDm', mt5.TIMEFRAME_M15, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Calculate ATR like the backtest does
tr_list = [0]
for i in range(1, len(df)):
    hl = df['high'].iloc[i] - df['low'].iloc[i]
    hc = abs(df['high'].iloc[i] - df['close'].iloc[i-1])
    lc = abs(df['low'].iloc[i] - df['close'].iloc[i-1])
    tr_list.append(max(hl, hc, lc))
df['tr'] = tr_list
df['atr'] = df['tr'].rolling(14).mean()

# Look at a few rows and simulate the signal
row = df.iloc[500]
print('Sample candle:')
print('Close:', row['close'])
print('ATR:', row['atr'])

pip_value = 0.1
atr = row['atr']
price = row['close']

# Simulate BUY signal
sl_price = price - (atr * 2.0)
tp_price = price + (atr * 4.0)

print('SL price:', sl_price)
print('TP price:', tp_price)

# Calculate pips
sl_pips = (price - sl_price) / pip_value
tp_pips = (tp_price - price) / pip_value
print('SL pips:', sl_pips)
print('TP pips:', tp_pips)

# If TP hits with 0.01 lot
lot = 0.01
pip_value_per_lot = 10.0
profit = tp_pips * lot * pip_value_per_lot
print('Profit if TP:', profit)

loss = sl_pips * lot * pip_value_per_lot
print('Loss if SL:', loss)

# 246 trades, 37% wins = ~91 wins
wins = 91
losses = 155

expected_total_profit = wins * profit - losses * loss
print('Expected total profit over 246 trades:', expected_total_profit)
