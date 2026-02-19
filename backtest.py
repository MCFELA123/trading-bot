"""
============================================================================
TRADING BOT BACKTESTING SYSTEM
============================================================================
Tests the trading strategy on historical data to validate performance.

IMPORTANT: Past performance does NOT guarantee future results!
Backtesting shows IF the strategy had an edge historically.
============================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

# Import indicators from botlogic
from botlogic import (
    calculate_advanced_indicators,
    trend_bias,
    liquidity_grab,
    order_block,
    fair_value_gap,
    check_market_structure,
    detect_market_regime,
    get_symbol_settings,
    SYMBOL_SETTINGS,
    MIN_SMC_SCORE,
    STOPLOSS_PIPS,
    TAKEPROFIT_PIPS,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    MIN_REWARD_RISK_RATIO,
)

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    'initial_balance': 1000.0,      # Starting balance
    'risk_percent': 1.0,            # Risk per trade
    'max_positions': 2,             # Max concurrent positions
    'spread_pips': 2.0,             # Simulated spread (conservative)
    'commission_per_lot': 7.0,      # Commission per lot (round turn)
    'slippage_pips': 0.5,           # Simulated slippage
    'use_fixed_lot': True,          # Use fixed lot for fair comparison
    'fixed_lot': 0.01,              # Fixed 0.01 lot (no compounding)
}

# Symbols to backtest
BACKTEST_SYMBOLS = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']

# Timeframe
BACKTEST_TF = mt5.TIMEFRAME_M15

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    def __init__(self, symbol, config=None):
        self.symbol = symbol
        self.config = config or BACKTEST_CONFIG.copy()
        self.balance = self.config['initial_balance']
        self.equity = self.balance
        self.starting_balance = self.balance
        
        # Trade tracking
        self.open_positions = []
        self.closed_trades = []
        self.trade_counter = 0
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'peak_balance': self.balance,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'trades_by_hour': defaultdict(int),
            'wins_by_hour': defaultdict(int),
            'trades_by_day': defaultdict(int),
            'wins_by_day': defaultdict(int),
        }
        
        # Symbol settings - strip broker suffix (m, .r, pro, etc.) to get base symbol
        base_symbol = symbol
        for suffix in ['m', '.r', 'pro']:
            if symbol.endswith(suffix):
                base_symbol = symbol[:-len(suffix)]
                break
        
        self.sym_settings = get_symbol_settings(base_symbol)
        self.pip_value = self.sym_settings.get('pip_value', 0.0001)
        self.point = self.pip_value / 10
        
    def calculate_lot_size(self, sl_pips):
        """Calculate lot size based on risk percentage"""
        
        # Use fixed lot if configured (no compounding - clearer backtest results)
        if self.config.get('use_fixed_lot', False):
            return self.config.get('fixed_lot', 0.01)
        
        risk_amount = self.balance * (self.config['risk_percent'] / 100)
        
        # Pip value per standard lot (1.0 lot)
        # For Gold: 1 pip = $0.10 move, 1 lot = 100 oz, so $10 per pip per lot
        # For Forex: ~$10 per pip per lot for most pairs
        if 'XAU' in self.symbol:
            pip_value_per_lot = 10.0  # $10 per pip per 1.0 lot for gold
        elif 'JPY' in self.symbol:
            pip_value_per_lot = 9.0  # ~$9 per pip per lot (depends on USDJPY rate)
        else:
            pip_value_per_lot = 10.0  # $10 per pip per standard lot
        
        if sl_pips <= 0:
            sl_pips = 20  # Default
            
        # lot = risk / (sl_pips * pip_value_per_lot)
        lot = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Cap lot sizes more realistically
        # For a $1000 account, max should be around 0.1 lots
        max_lot = min(0.1, self.balance / 5000)  # Scale max lot with balance
        lot = max(0.01, min(max_lot, round(lot, 2)))
        return lot
    
    def open_trade(self, direction, entry_price, sl, tp, timestamp, quality_score):
        """Open a simulated trade"""
        if len(self.open_positions) >= self.config['max_positions']:
            return None
        
        # Calculate lot size
        if direction == 'BUY':
            sl_pips = (entry_price - sl) / self.pip_value
        else:
            sl_pips = (sl - entry_price) / self.pip_value
        
        lot = self.calculate_lot_size(abs(sl_pips))
        
        # Apply spread (entry at worse price)
        spread = self.config['spread_pips'] * self.pip_value
        if direction == 'BUY':
            entry_price += spread / 2
        else:
            entry_price -= spread / 2
        
        # Apply slippage
        slippage = self.config['slippage_pips'] * self.pip_value
        if direction == 'BUY':
            entry_price += slippage
        else:
            entry_price -= slippage
        
        self.trade_counter += 1
        trade = {
            'ticket': self.trade_counter,
            'symbol': self.symbol,
            'direction': direction,
            'lot': lot,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'open_time': timestamp,
            'quality_score': quality_score,
        }
        
        self.open_positions.append(trade)
        return trade
    
    def check_and_close_trades(self, high, low, close, timestamp):
        """Check if any trades hit SL or TP"""
        closed = []
        
        for trade in self.open_positions[:]:  # Copy to avoid modification during iteration
            hit_sl = False
            hit_tp = False
            exit_price = close
            
            if trade['direction'] == 'BUY':
                # Check SL (low touched SL)
                if low <= trade['sl']:
                    hit_sl = True
                    exit_price = trade['sl']
                # Check TP (high touched TP)
                elif high >= trade['tp']:
                    hit_tp = True
                    exit_price = trade['tp']
            else:  # SELL
                # Check SL (high touched SL)
                if high >= trade['sl']:
                    hit_sl = True
                    exit_price = trade['sl']
                # Check TP (low touched TP)
                elif low <= trade['tp']:
                    hit_tp = True
                    exit_price = trade['tp']
            
            if hit_sl or hit_tp:
                self.close_trade(trade, exit_price, timestamp, 'SL' if hit_sl else 'TP')
                closed.append(trade)
        
        return closed
    
    def close_trade(self, trade, exit_price, timestamp, reason):
        """Close a trade and calculate P/L"""
        # Remove from open positions
        if trade in self.open_positions:
            self.open_positions.remove(trade)
        
        # Calculate profit/loss
        if trade['direction'] == 'BUY':
            pips = (exit_price - trade['entry_price']) / self.pip_value
        else:
            pips = (trade['entry_price'] - exit_price) / self.pip_value
        
        # Calculate dollar profit using proper pip values per lot
        # Gold: $10 per pip per lot (1 pip = $0.10 price move, 100 oz per lot)
        # JPY pairs: ~$9 per pip per lot
        # Other forex: ~$10 per pip per lot
        if 'XAU' in self.symbol:
            pip_value_per_lot = 10.0
        elif 'JPY' in self.symbol:
            pip_value_per_lot = 9.0
        else:
            pip_value_per_lot = 10.0
            
        profit = pips * trade['lot'] * pip_value_per_lot
        
        # Subtract commission
        commission = self.config['commission_per_lot'] * trade['lot']
        profit -= commission
        
        # Update balance
        self.balance += profit
        
        # Track peak for drawdown
        if self.balance > self.stats['peak_balance']:
            self.stats['peak_balance'] = self.balance
        
        # Calculate drawdown
        drawdown = self.stats['peak_balance'] - self.balance
        drawdown_pct = (drawdown / self.stats['peak_balance']) * 100 if self.stats['peak_balance'] > 0 else 0
        
        if drawdown > self.stats['max_drawdown']:
            self.stats['max_drawdown'] = drawdown
        if drawdown_pct > self.stats['max_drawdown_pct']:
            self.stats['max_drawdown_pct'] = drawdown_pct
        
        # Update statistics
        self.stats['total_trades'] += 1
        
        if profit > 0:
            self.stats['wins'] += 1
            self.stats['gross_profit'] += profit
            self.stats['consecutive_wins'] += 1
            self.stats['consecutive_losses'] = 0
            if self.stats['consecutive_wins'] > self.stats['max_consecutive_wins']:
                self.stats['max_consecutive_wins'] = self.stats['consecutive_wins']
        elif profit < 0:
            self.stats['losses'] += 1
            self.stats['gross_loss'] += abs(profit)
            self.stats['consecutive_losses'] += 1
            self.stats['consecutive_wins'] = 0
            if self.stats['consecutive_losses'] > self.stats['max_consecutive_losses']:
                self.stats['max_consecutive_losses'] = self.stats['consecutive_losses']
        else:
            self.stats['breakeven'] += 1
        
        # Track by time
        hour = timestamp.hour
        day = timestamp.strftime('%A')
        self.stats['trades_by_hour'][hour] += 1
        self.stats['trades_by_day'][day] += 1
        if profit > 0:
            self.stats['wins_by_hour'][hour] += 1
            self.stats['wins_by_day'][day] += 1
        
        # Record closed trade
        trade['exit_price'] = exit_price
        trade['close_time'] = timestamp
        trade['profit'] = profit
        trade['pips'] = pips
        trade['reason'] = reason
        self.closed_trades.append(trade)
        
        return profit
    
    def get_results(self):
        """Get backtest results summary"""
        total = self.stats['total_trades']
        if total == 0:
            return {'error': 'No trades executed'}
        
        win_rate = (self.stats['wins'] / total) * 100
        profit_factor = (self.stats['gross_profit'] / self.stats['gross_loss']) if self.stats['gross_loss'] > 0 else float('inf')
        net_profit = self.stats['gross_profit'] - self.stats['gross_loss']
        return_pct = ((self.balance - self.starting_balance) / self.starting_balance) * 100
        avg_win = self.stats['gross_profit'] / self.stats['wins'] if self.stats['wins'] > 0 else 0
        avg_loss = self.stats['gross_loss'] / self.stats['losses'] if self.stats['losses'] > 0 else 0
        
        # Expected value per trade
        expected_value = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        return {
            'symbol': self.symbol,
            'period_days': 0,  # Will be set by caller
            'total_trades': total,
            'wins': self.stats['wins'],
            'losses': self.stats['losses'],
            'breakeven': self.stats['breakeven'],
            'win_rate': round(win_rate, 2),
            'gross_profit': round(self.stats['gross_profit'], 2),
            'gross_loss': round(self.stats['gross_loss'], 2),
            'net_profit': round(net_profit, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Infinite',
            'return_pct': round(return_pct, 2),
            'starting_balance': self.starting_balance,
            'final_balance': round(self.balance, 2),
            'max_drawdown': round(self.stats['max_drawdown'], 2),
            'max_drawdown_pct': round(self.stats['max_drawdown_pct'], 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expected_value': round(expected_value, 2),
            'max_consecutive_wins': self.stats['max_consecutive_wins'],
            'max_consecutive_losses': self.stats['max_consecutive_losses'],
            'best_hour': max(self.stats['wins_by_hour'].items(), key=lambda x: x[1])[0] if self.stats['wins_by_hour'] else 'N/A',
            'best_day': max(self.stats['wins_by_day'].items(), key=lambda x: x[1])[0] if self.stats['wins_by_day'] else 'N/A',
        }


# ============================================================================
# STRATEGY SIMULATION (Replicates botlogic.py entry logic)
# ============================================================================

def simulate_strategy_signal_fast(df, index):
    """
    FAST version: Uses pre-calculated indicators.
    Returns: (direction, quality_score, sl_price, tp_price) or (None, 0, 0, 0)
    """
    if index < 100:
        return None, 0, 0, 0
    
    # Use pre-calculated values (no recalculation needed!)
    try:
        price = df['close'].iloc[index]
        ema_9 = df['ema_9'].iloc[index]
        ema_21 = df['ema_21'].iloc[index]
        ema_50 = df['ema_50'].iloc[index]
        rsi = df['rsi'].iloc[index]
        macd_hist = df['macd_hist'].iloc[index]
        atr = df['atr'].iloc[index]
        adx = df['adx'].iloc[index] if 'adx' in df.columns else 25
    except:
        return None, 0, 0, 0
    
    # Check for NaN values
    if pd.isna(ema_9) or pd.isna(rsi) or pd.isna(atr):
        return None, 0, 0, 0
    
    # Simple trend detection (last 20 bars)
    if index >= 20:
        recent_closes = df['close'].iloc[index-20:index+1].values
        trend_up = recent_closes[-1] > recent_closes[0]
        trend_down = recent_closes[-1] < recent_closes[0]
    else:
        trend_up = trend_down = False
    
    # Market regime from ADX
    is_trending = adx > 25 if not pd.isna(adx) else True
    
    # Calculate scores
    buy_score = 0
    sell_score = 0
    
    # Trend alignment (2 points)
    if trend_up and is_trending:
        buy_score += 2
    elif trend_down and is_trending:
        sell_score += 2
    
    # EMA alignment (2 points)
    if price > ema_9 > ema_21 > ema_50:
        buy_score += 2
    elif price < ema_9 < ema_21 < ema_50:
        sell_score += 2
    
    # MACD momentum (1 point)
    if macd_hist > 0:
        buy_score += 1
    elif macd_hist < 0:
        sell_score += 1
    
    # RSI confirmation (1 point)
    if 40 < rsi < 65:
        buy_score += 1
    if 35 < rsi < 60:
        sell_score += 1
    
    # Check for potential order block (simplified)
    if index >= 5:
        recent_highs = df['high'].iloc[index-5:index].values
        recent_lows = df['low'].iloc[index-5:index].values
        current_low = df['low'].iloc[index]
        current_high = df['high'].iloc[index]
        
        # Bullish OB: price swept recent lows then bounced
        if current_low < min(recent_lows) and price > df['open'].iloc[index]:
            buy_score += 1
        # Bearish OB: price swept recent highs then dropped
        if current_high > max(recent_highs) and price < df['open'].iloc[index]:
            sell_score += 1
    
    # Minimum score requirement
    MIN_SCORE = 4
    
    # Determine direction
    direction = None
    quality_score = 0
    
    if buy_score >= MIN_SCORE and buy_score > sell_score + 1:
        direction = 'BUY'
        quality_score = buy_score
    elif sell_score >= MIN_SCORE and sell_score > buy_score + 1:
        direction = 'SELL'
        quality_score = sell_score
    else:
        return None, 0, 0, 0
    
    # Skip ranging markets (low ADX)
    if adx < 20:
        return None, 0, 0, 0
    
    # Calculate SL/TP using ATR
    if atr <= 0 or pd.isna(atr):
        atr = price * 0.001
    
    if direction == 'BUY':
        sl_price = price - (atr * 2.0)
        tp_price = price + (atr * 4.0)
    else:
        sl_price = price + (atr * 2.0)
        tp_price = price - (atr * 4.0)
    
    return direction, quality_score, sl_price, tp_price


# ============================================================================
# MAIN BACKTEST FUNCTION
# ============================================================================

def run_backtest(symbol, days=180, timeframe=None, config=None):
    """
    Run backtest on historical data.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        days: Number of days of history to test
        timeframe: MT5 timeframe (default: M15)
        config: Backtest configuration dict
    
    Returns:
        dict: Backtest results
    """
    if timeframe is None:
        timeframe = BACKTEST_TF
    
    # Initialize MT5 if needed
    if not mt5.terminal_info():
        if not mt5.initialize():
            return {'error': 'Failed to initialize MT5'}
    
    # Check symbol exists
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        # Try with common suffixes
        for suffix in ['m', '.r', 'pro', '']:
            test_sym = symbol + suffix
            symbol_info = mt5.symbol_info(test_sym)
            if symbol_info:
                symbol = test_sym
                break
    
    if symbol_info is None:
        return {'error': f'Symbol {symbol} not found'}
    
    # Enable symbol
    mt5.symbol_select(symbol, True)
    
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"📊 Fetching {days} days of {symbol} data...")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) < 200:
        return {'error': f'Insufficient data for {symbol} (got {len(rates) if rates else 0} bars)'}
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"📈 Calculating indicators on {len(df)} candles...")
    
    # PRE-CALCULATE ALL INDICATORS ONCE (major speed improvement!)
    try:
        df = calculate_advanced_indicators(df)
    except Exception as e:
        print(f"⚠️ Indicator calculation error: {e}")
        # Calculate basic indicators manually
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # ADX (simplified)
        df['adx'] = 25  # Default
    
    print(f"🚀 Running simulation...")
    
    # Initialize engine
    engine = BacktestEngine(symbol, config)
    
    # Track signals to avoid overtrading
    last_signal_time = None
    min_bars_between_signals = 3  # Minimum 3 candles between signals
    
    # Run simulation bar by bar
    for i in range(100, len(df)):
        current_bar = df.iloc[i]
        timestamp = current_bar['time']
        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']
        
        # First, check existing positions for SL/TP
        engine.check_and_close_trades(high, low, close, timestamp)
        
        # Skip if we have max positions
        if len(engine.open_positions) >= engine.config['max_positions']:
            continue
        
        # Check for cooldown between signals
        if last_signal_time is not None:
            bars_since_signal = i - last_signal_time
            if bars_since_signal < min_bars_between_signals:
                continue
        
        # Get signal for this bar (FAST version - uses pre-calculated indicators)
        direction, quality_score, sl_price, tp_price = simulate_strategy_signal_fast(df, i)
        
        if direction is not None:
            # Open trade
            trade = engine.open_trade(direction, close, sl_price, tp_price, timestamp, quality_score)
            if trade:
                last_signal_time = i
    
    # Close any remaining positions at last price
    last_bar = df.iloc[-1]
    for trade in engine.open_positions[:]:
        engine.close_trade(trade, last_bar['close'], last_bar['time'], 'END')
    
    # Get results
    results = engine.get_results()
    results['period_days'] = days
    results['candles_tested'] = len(df)
    results['timeframe'] = 'M15'
    
    return results


def run_full_backtest(symbols=None, days=180):
    """
    Run backtest on multiple symbols.
    
    Args:
        symbols: List of symbols to test (default: BACKTEST_SYMBOLS)
        days: Days of history
    
    Returns:
        dict: Combined results for all symbols
    """
    if symbols is None:
        symbols = BACKTEST_SYMBOLS
    
    print("=" * 60)
    print("🔬 TRADING BOT BACKTEST SYSTEM")
    print("=" * 60)
    print(f"Testing {len(symbols)} symbols over {days} days")
    print(f"Starting balance: ${BACKTEST_CONFIG['initial_balance']}")
    print(f"Risk per trade: {BACKTEST_CONFIG['risk_percent']}%")
    print("=" * 60)
    
    all_results = {}
    combined_stats = {
        'total_trades': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_net_profit': 0,
        'symbols_tested': 0,
        'profitable_symbols': 0,
    }
    
    for symbol in symbols:
        print(f"\n🔄 Testing {symbol}...")
        result = run_backtest(symbol, days)
        all_results[symbol] = result
        
        if 'error' not in result:
            combined_stats['total_trades'] += result['total_trades']
            combined_stats['total_wins'] += result['wins']
            combined_stats['total_losses'] += result['losses']
            combined_stats['total_net_profit'] += result['net_profit']
            combined_stats['symbols_tested'] += 1
            if result['net_profit'] > 0:
                combined_stats['profitable_symbols'] += 1
            
            # Print summary
            emoji = "✅" if result['net_profit'] > 0 else "❌"
            print(f"{emoji} {symbol}: {result['total_trades']} trades, "
                  f"{result['win_rate']}% win rate, "
                  f"${result['net_profit']:+.2f} profit, "
                  f"Max DD: {result['max_drawdown_pct']:.1f}%")
    
    # Calculate combined metrics
    if combined_stats['total_trades'] > 0:
        combined_stats['overall_win_rate'] = round(
            (combined_stats['total_wins'] / combined_stats['total_trades']) * 100, 2
        )
    else:
        combined_stats['overall_win_rate'] = 0
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 COMBINED BACKTEST RESULTS")
    print("=" * 60)
    print(f"Symbols Tested: {combined_stats['symbols_tested']}")
    print(f"Profitable Symbols: {combined_stats['profitable_symbols']}")
    print(f"Total Trades: {combined_stats['total_trades']}")
    print(f"Overall Win Rate: {combined_stats['overall_win_rate']}%")
    print(f"Total Net Profit: ${combined_stats['total_net_profit']:+.2f}")
    print("=" * 60)
    
    # Verdict
    if combined_stats['total_net_profit'] > 0 and combined_stats['overall_win_rate'] >= 50:
        print("✅ VERDICT: Strategy shows POSITIVE expectancy")
    elif combined_stats['total_net_profit'] > 0:
        print("⚠️ VERDICT: Profitable but low win rate - relies on R:R")
    else:
        print("❌ VERDICT: Strategy shows NEGATIVE expectancy - needs optimization")
    
    # === MANUAL TUNING RECOMMENDATIONS ===
    print("\n" + "-" * 60)
    print("📝 PARAMETER RECOMMENDATIONS (edit botlogic.py)")
    print("-" * 60)
    
    # Find best/worst symbols
    profitable = []
    losing = []
    for sym, res in all_results.items():
        if 'error' not in res:
            if res['net_profit'] > 0:
                profitable.append((sym, res['net_profit'], res['win_rate']))
            else:
                losing.append((sym, res['net_profit'], res['win_rate']))
    
    # Symbol recommendations
    if losing:
        losing_syms = [s[0] for s in losing]
        print(f"\n1. DISABLE LOSING SYMBOLS: {', '.join(losing_syms)}")
        print("   Location: botlogic.py line ~90 (TRADING_SYMBOLS)")
        print(f"   Remove: {losing_syms}")
    
    if profitable:
        best = max(profitable, key=lambda x: x[1])
        print(f"\n2. BEST PERFORMER: {best[0]} (+${best[1]:.2f}, {best[2]}% win rate)")
        print("   Consider increasing lot size for this symbol")
    
    # Win rate recommendations
    if combined_stats['overall_win_rate'] < 35:
        print("\n3. LOW WIN RATE - Consider:")
        print("   - Increase MIN_QUALITY_SCORE from current value to 5 or 6")
        print("   - Location: botlogic.py search for 'MIN_QUALITY_SCORE'")
    
    # Drawdown recommendations  
    max_dd = 0
    max_dd_sym = ""
    for sym, res in all_results.items():
        if 'error' not in res and res.get('max_drawdown_pct', 0) > max_dd:
            max_dd = res['max_drawdown_pct']
            max_dd_sym = sym
    
    if max_dd > 30:
        print(f"\n4. HIGH DRAWDOWN ({max_dd:.1f}% on {max_dd_sym}) - Consider:")
        print("   - Reduce RISK_PERCENT (currently 1%) to 0.5%")
        print("   - Location: botlogic.py search for 'RISK_PERCENT'")
    
    print("-" * 60)
    
    return {
        'individual_results': all_results,
        'combined': combined_stats,
        'config': BACKTEST_CONFIG,
        'test_date': datetime.now().isoformat(),
    }


def save_backtest_report(results, filename=None):
    """Save backtest results to JSON file (single file, overwritten each time)"""
    if filename is None:
        filename = "backtest_report.json"
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    # Convert defaultdict to regular dict for JSON
    def convert_results(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {k: convert_results(v) for k, v in obj.items()}
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return obj
    
    clean_results = convert_results(results)
    
    with open(filepath, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    
    print(f"\n📁 Report saved to: {filepath}")
    return filepath


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    days = 180  # Default 6 months
    symbols = None  # Use defaults
    
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        symbols = sys.argv[2].split(',')
    
    # Run backtest
    results = run_full_backtest(symbols=symbols, days=days)
    
    # Save to single report file (overwrites previous)
    save_backtest_report(results)
    
    print("\n🏁 Backtest complete!")
