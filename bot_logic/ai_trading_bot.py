"""
ENHANCED TRADING BOT v3.0 - AI-Powered with Aggressive Profit Protection
=========================================================================

Key Improvements:
1. FIXED AI CONFIGURATION - OpenAI now works properly
2. AGGRESSIVE PROFIT PROTECTION - Locks in profits early
3. BETTER ENTRY VALIDATION - AI filters bad trades
4. MULTI-SYMBOL SUPPORT - Gold (XAUUSD) and more

Author: Trading Bot System
Date: February 2026
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import threading
import logging
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from openai import OpenAI

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================================================================
# ========================= OPENAI CONFIGURATION (FIXED!) ========================
# ================================================================================

# API Key - set via environment variable or use default
# IMPORTANT: Set OPENAI_API_KEY environment variable for production
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_client = None
AI_INITIALIZED = False

def get_openai_client():
    """
    Get or create OpenAI client - FIXED VERSION
    The previous version had faulty logic that prevented initialization
    """
    global openai_client, AI_INITIALIZED
    
    if openai_client is None:
        # Check if we have a valid API key
        if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20 and OPENAI_API_KEY.startswith('sk-'):
            try:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                AI_INITIALIZED = True
                logger.info("✅ OpenAI AI client initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI: {e}")
                openai_client = None
                AI_INITIALIZED = False
        else:
            logger.warning("⚠️ Invalid OpenAI API key format - AI features disabled")
            AI_INITIALIZED = False
    
    return openai_client


def is_ai_available():
    """Check if AI is properly configured and available"""
    client = get_openai_client()
    return client is not None


# ---------------- AI TRADE ANALYSIS STORAGE ----------------
ai_trade_history = defaultdict(list)
ai_learned_params = defaultdict(dict)

# Import trading log function
def log_trade(username, log_type, message, details=None):
    """Log trading activity to database"""
    try:
        from models import add_trading_log
        add_trading_log(username, log_type, message, details)
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")

# ---------------- PER-USER BOT STORAGE ----------------
user_bots = defaultdict(dict)
user_mt5_sessions = {}

# ---------------- DEFAULT MT5 LOGIN CONFIG ----------------
DEFAULT_MT5_LOGIN = 10009413572
DEFAULT_MT5_PASSWORD = "@3BhJfGr"
DEFAULT_MT5_SERVER = "MetaQuotes-Demo"

# ================================================================================
# ========================= BOT CONFIGURATION ====================================
# ================================================================================

# Supported trading symbols - GOLD IS PRIMARY
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
DEFAULT_SYMBOL = "XAUUSD"  # Gold

# Timeframe
TIMEFRAME = mt5.TIMEFRAME_M5  # Use M5 for more trading opportunities

# Risk Management - BALANCED for profit with protection
RISK_PERCENT = 1.0          # 1% risk per trade - proper risk management
STOPLOSS_PIPS = 20          # 20 pip stop loss
TAKEPROFIT_PIPS = 40        # 1:2 R:R ratio
MAX_POSITIONS = 3           # Allow 3 positions for scaling
MAX_DAILY_TRADES = 30       # Allow 30 trades per day for more opportunities
MAX_LOT_SIZE = 1.0          # Allow up to 1.0 lot

# Timing
CHECK_INTERVAL = 3          # Check every 3 seconds
MAGIC = 202502              # Magic number for bot trades

# ================================================================================
# ================== AGGRESSIVE PROFIT PROTECTION SYSTEM =========================
# ================================================================================

# These settings ensure profits are protected IMMEDIATELY
USE_PROFIT_PROTECTION = True
BREAKEVEN_TRIGGER_PIPS = 5        # Move to breakeven after 5 pips profit
BREAKEVEN_OFFSET_PIPS = 1         # Place SL 1 pip above entry
PROFIT_LOCK_START_PIPS = 8        # Start locking profit after 8 pips
PROFIT_LOCK_PERCENT = 40          # Lock 40% of max profit reached
TRAILING_START_PIPS = 12          # Start trailing after 12 pips
TRAILING_DISTANCE_PIPS = 8        # Trail 8 pips behind price

# Track maximum profit per position for profit locking
position_max_profits = defaultdict(float)

# ================================================================================
# ================== AI TRADING REQUIREMENTS =====================================
# ================================================================================

AI_ENABLED = True                   # Enable AI analysis
AI_ANALYSIS_EVERY_N_CYCLES = 2      # Analyze every 2 cycles for faster decisions
AI_MIN_CONFIDENCE_FOR_TRADE = 0.70  # 70% confidence - balanced
AI_MUST_APPROVE_TRADE = True        # AI must approve before entry
AI_ENTRY_QUALITY_REQUIRED = ["EXCELLENT", "GOOD"]  # Take GOOD and EXCELLENT entries

# ================================================================================
# ========================= GLOBAL TRACKING ======================================
# ================================================================================

trade_stats = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
current_trends = defaultdict(lambda: "NEUTRAL")
daily_trade_counts = defaultdict(int)

# ================================================================================
# ========================= OPENAI AI TRADING INTELLIGENCE =======================
# ================================================================================

def ai_analyze_market(df, symbol, user):
    """
    ENHANCED AI MARKET ANALYSIS
    Uses GPT-4o to analyze market conditions and provide trading recommendations.
    Only recommends trades with high probability setups.
    """
    client = get_openai_client()
    if client is None:
        logger.warning(f"[{user}] ⚠️ AI not available - check API key configuration")
        return {
            "recommendation": "HOLD", 
            "confidence": 0.3, 
            "reason": "AI not configured - please set OPENAI_API_KEY",
            "entry_quality": "POOR"
        }
    
    try:
        # Prepare comprehensive market data
        recent_data = df.tail(50)
        price_now = recent_data['close'].iloc[-1]
        price_5_ago = recent_data['close'].iloc[-5]
        price_20_ago = recent_data['close'].iloc[-20] if len(recent_data) >= 20 else price_now
        
        # Price levels
        high_20 = recent_data['high'].max()
        low_20 = recent_data['low'].min()
        range_20 = high_20 - low_20
        price_position = ((price_now - low_20) / range_20 * 100) if range_20 > 0 else 50
        
        # Multiple EMAs
        ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
        ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema_200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema_50
        
        # EMA alignment (all EMAs stacked = strong trend)
        ema_bullish_stack = ema_9 > ema_21 > ema_50
        ema_bearish_stack = ema_9 < ema_21 < ema_50
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = (macd - signal).iloc[-1]
        macd_trend = "BULLISH" if macd_histogram > 0 else "BEARISH"
        
        # ATR for volatility
        tr = pd.DataFrame()
        tr['hl'] = recent_data['high'] - recent_data['low']
        tr['hc'] = abs(recent_data['high'] - recent_data['close'].shift(1))
        tr['lc'] = abs(recent_data['low'] - recent_data['close'].shift(1))
        tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
        atr = tr['tr'].rolling(14).mean().iloc[-1]
        
        # Momentum (rate of change)
        momentum_5 = ((price_now - price_5_ago) / price_5_ago * 100)
        momentum_20 = ((price_now - price_20_ago) / price_20_ago * 100)
        
        # Volume analysis
        avg_volume = recent_data['tick_volume'].mean() if 'tick_volume' in recent_data.columns else 0
        recent_volume = recent_data['tick_volume'].iloc[-1] if 'tick_volume' in recent_data.columns else 0
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Candle pattern analysis
        last_candle = recent_data.iloc[-1]
        prev_candle = recent_data.iloc[-2]
        candle_body = abs(last_candle['close'] - last_candle['open'])
        candle_range = last_candle['high'] - last_candle['low']
        is_bullish_candle = last_candle['close'] > last_candle['open']
        is_bearish_candle = last_candle['close'] < last_candle['open']
        
        market_context = f"""
SYMBOL: {symbol}
CURRENT PRICE: {price_now:.2f}
TIMEFRAME: M5 (5-minute)

=== PRICE ACTION ===
20-Period High: {high_20:.2f}
20-Period Low: {low_20:.2f}
Price Position in Range: {price_position:.1f}% (0=at low, 100=at high)
5-Bar Momentum: {momentum_5:.3f}%
20-Bar Momentum: {momentum_20:.3f}%

=== MOVING AVERAGES ===
EMA 9: {ema_9:.2f} (Price {'ABOVE' if price_now > ema_9 else 'BELOW'})
EMA 21: {ema_21:.2f} (Price {'ABOVE' if price_now > ema_21 else 'BELOW'})
EMA 50: {ema_50:.2f} (Price {'ABOVE' if price_now > ema_50 else 'BELOW'})
EMA 200: {ema_200:.2f} (Price {'ABOVE' if price_now > ema_200 else 'BELOW'})
EMA Stack: {'BULLISH (9>21>50)' if ema_bullish_stack else 'BEARISH (9<21<50)' if ema_bearish_stack else 'MIXED/CHOPPY'}

=== INDICATORS ===
RSI (14): {rsi:.2f} {'(OVERBOUGHT)' if rsi > 70 else '(OVERSOLD)' if rsi < 30 else '(NEUTRAL)'}
MACD Histogram: {macd_histogram:.4f} ({macd_trend})
ATR (14): {atr:.2f}
Volume Ratio (vs avg): {volume_ratio:.2f}x

=== LAST CANDLE ===
Type: {'BULLISH' if is_bullish_candle else 'BEARISH' if is_bearish_candle else 'DOJI'}
Body Size: {candle_body:.2f}
Range: {candle_range:.2f}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an ELITE trading analyst AI specializing in Gold (XAUUSD) and Forex.
                    
YOUR MISSION: Identify ONLY high-probability trades that will PROTECT and GROW capital.

=== STRICT ENTRY CRITERIA (ALL must be met) ===

FOR BUY SIGNALS:
✓ EMA stack bullish (9>21>50) OR price above EMA 50 with momentum
✓ RSI between 40-65 (not overbought)
✓ MACD histogram positive or turning positive
✓ Price near 20-period low (0-30% range) OR breaking above resistance
✓ Volume above average

FOR SELL SIGNALS:
✓ EMA stack bearish (9<21<50) OR price below EMA 50 with momentum
✓ RSI between 35-60 (not oversold)  
✓ MACD histogram negative or turning negative
✓ Price near 20-period high (70-100% range) OR breaking below support
✓ Volume above average

=== REJECT TRADES WHEN ===
✗ Price in middle of range (40-60%)
✗ EMAs are flat or crossing frequently
✗ RSI in neutral zone (45-55)
✗ Low volume/momentum
✗ Conflicting signals between indicators

=== FOR GOLD (XAUUSD) SPECIFICALLY ===
- Gold is volatile - require stronger confirmation
- Respect key psychological levels (1900, 1950, 2000, etc.)
- Consider USD strength movements
- Use wider stops (20-30 pips)

Respond ONLY with valid JSON:
{
    "recommendation": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reason": "Brief explanation (max 40 words)",
    "entry_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR",
    "suggested_sl_pips": number (15-40),
    "suggested_tp_pips": number (25-80),
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "market_condition": "TRENDING" or "RANGING" or "VOLATILE"
}"""
                },
                {
                    "role": "user",
                    "content": f"Analyze this market data and provide your recommendation:\n{market_context}"
                }
            ],
            temperature=0.15,  # Low temperature for consistency
            max_tokens=350
        )
        
        # Parse response
        content = response.choices[0].message.content
        # Clean JSON if needed
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content.strip())
        
        # Log the analysis
        quality = result.get('entry_quality', 'N/A')
        conf = result.get('confidence', 0)
        rec = result.get('recommendation', 'HOLD')
        reason = result.get('reason', 'No reason provided')
        
        logger.info(f"[{user}] 🤖 AI: {rec} | Quality: {quality} | Confidence: {conf:.0%}")
        logger.info(f"[{user}] 📊 Reason: {reason}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"[{user}] AI JSON parse error: {e}")
        return {"recommendation": "HOLD", "confidence": 0.3, "reason": "AI response parse error", "entry_quality": "POOR"}
    except Exception as e:
        logger.error(f"[{user}] AI analysis error: {e}")
        return {"recommendation": "HOLD", "confidence": 0.3, "reason": f"AI error: {str(e)[:50]}", "entry_quality": "POOR"}


def ai_validate_trade_signal(df, signal_type, smc_score, user):
    """
    AI validates a trading signal BEFORE execution.
    This is the final gate - be STRICT to protect capital.
    """
    client = get_openai_client()
    if client is None:
        # Without AI, be conservative
        return smc_score >= 3, 0.8
    
    try:
        recent_data = df.tail(30)
        price = recent_data['close'].iloc[-1]
        
        # Calculate key metrics
        atr = (recent_data['high'] - recent_data['low']).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Momentum
        momentum = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5] * 100
        
        # Recent candles analysis
        bullish_count = sum(1 for i in range(-5, 0) if recent_data['close'].iloc[i] > recent_data['open'].iloc[i])
        
        signal_context = f"""
PROPOSED TRADE: {signal_type}
SMC Score: {smc_score}/4
Current Price: {price:.2f}
RSI: {rsi:.2f}
5-Bar Momentum: {momentum:.3f}%
Recent Bullish Candles: {bullish_count}/5
ATR: {atr:.2f}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a STRICT trade filter AI. Your job is to PROTECT CAPITAL by rejecting weak trades.

APPROVE only when:
- For BUY: RSI < 65, positive momentum, majority bullish candles
- For SELL: RSI > 35, negative momentum, majority bearish candles
- SMC Score >= 2

REJECT when:
- Conflicting momentum and signal direction
- RSI at extremes against signal direction
- SMC Score < 2

When in doubt, REJECT. Better to miss a trade than lose money.

Respond ONLY with JSON:
{
    "approved": true or false,
    "confidence_multiplier": 0.5 to 1.2,
    "reason": "max 25 words"
}"""
                },
                {
                    "role": "user",
                    "content": f"Should we execute this trade?\n{signal_context}"
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content.strip())
        
        approved = result.get('approved', False)
        conf_mult = result.get('confidence_multiplier', 1.0)
        reason = result.get('reason', 'No reason')
        
        status = '✅ APPROVED' if approved else '❌ REJECTED'
        logger.info(f"[{user}] 🔍 Validation: {status} | {reason}")
        
        return approved, conf_mult
        
    except Exception as e:
        logger.error(f"[{user}] AI validation error: {e}")
        # On error, be conservative - only approve strong signals
        return smc_score >= 3, 0.8


def ai_study_trade_results(user, trade_data):
    """AI learns from completed trades to improve future performance"""
    client = get_openai_client()
    if client is None:
        return
    
    ai_trade_history[user].append(trade_data)
    
    # Analyze after every 5 trades
    if len(ai_trade_history[user]) % 5 != 0:
        return
    
    try:
        recent_trades = ai_trade_history[user][-20:]
        wins = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        losses = len(recent_trades) - wins
        total_profit = sum(t.get('profit', 0) for t in recent_trades)
        
        avg_win = np.mean([t['profit'] for t in recent_trades if t.get('profit', 0) > 0]) if wins > 0 else 0
        avg_loss = np.mean([abs(t['profit']) for t in recent_trades if t.get('profit', 0) < 0]) if losses > 0 else 0
        
        trade_summary = f"""
Win Rate: {(wins/len(recent_trades)*100):.1f}%
Total P/L: ${total_profit:.2f}
Avg Win: ${avg_win:.2f}
Avg Loss: ${avg_loss:.2f}
R:R Ratio: {(avg_win/avg_loss if avg_loss > 0 else 0):.2f}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Analyze trading performance and suggest optimizations.
Focus on CAPITAL PRESERVATION first.

Respond with JSON:
{
    "suggested_sl_pips": 10-35,
    "suggested_tp_pips": 20-70,
    "suggested_risk_percent": 0.3-1.5,
    "min_smc_score_for_entry": 2-3,
    "insights": "max 40 words",
    "strategy_adjustment": "specific recommendation"
}"""
                },
                {
                    "role": "user",
                    "content": f"Analyze:\n{trade_summary}"
                }
            ],
            temperature=0.3,
            max_tokens=250
        )
        
        content = response.choices[0].message.content
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content.strip())
        
        ai_learned_params[user] = {
            'sl_pips': result.get('suggested_sl_pips', STOPLOSS_PIPS),
            'tp_pips': result.get('suggested_tp_pips', TAKEPROFIT_PIPS),
            'risk_percent': result.get('suggested_risk_percent', RISK_PERCENT),
            'min_score': result.get('min_smc_score_for_entry', 2),
            'last_updated': datetime.now().isoformat(),
            'insights': result.get('insights', ''),
            'strategy_adjustment': result.get('strategy_adjustment', '')
        }
        
        logger.info(f"[{user}] 🧠 AI Learning: {result.get('insights', 'Updated parameters')}")
        
    except Exception as e:
        logger.error(f"[{user}] AI learning error: {e}")


def get_ai_optimized_params(user):
    """Get AI-optimized parameters for the user"""
    if user in ai_learned_params and ai_learned_params[user]:
        params = ai_learned_params[user]
        return {
            'sl_pips': params.get('sl_pips', STOPLOSS_PIPS),
            'tp_pips': params.get('tp_pips', TAKEPROFIT_PIPS),
            'risk_percent': params.get('risk_percent', RISK_PERCENT),
            'min_score': params.get('min_score', 2)
        }
    return {
        'sl_pips': STOPLOSS_PIPS,
        'tp_pips': TAKEPROFIT_PIPS,
        'risk_percent': RISK_PERCENT,
        'min_score': 2
    }


def get_ai_insights(user):
    """Get AI insights for dashboard"""
    if user in ai_learned_params and ai_learned_params[user]:
        return {
            'has_insights': True,
            'insights': ai_learned_params[user].get('insights', 'Learning...'),
            'strategy_adjustment': ai_learned_params[user].get('strategy_adjustment', ''),
            'last_updated': ai_learned_params[user].get('last_updated', ''),
            'optimized_params': {
                'sl_pips': ai_learned_params[user].get('sl_pips', STOPLOSS_PIPS),
                'tp_pips': ai_learned_params[user].get('tp_pips', TAKEPROFIT_PIPS),
                'risk_percent': ai_learned_params[user].get('risk_percent', RISK_PERCENT)
            }
        }
    return {
        'has_insights': False,
        'insights': 'AI is learning from your trades.',
        'strategy_adjustment': 'Execute more trades for AI optimization.',
        'optimized_params': {}
    }


def get_ai_status():
    """Check AI configuration status"""
    client = get_openai_client()
    if client:
        return {
            "configured": True,
            "status": "ACTIVE",
            "model": "gpt-4o",
            "message": "AI is active and analyzing markets"
        }
    return {
        "configured": False,
        "status": "INACTIVE",
        "model": None,
        "message": "AI not configured - check OPENAI_API_KEY"
    }


# ================================================================================
# ========================= MT5 INITIALIZATION ===================================
# ================================================================================

def initialize_mt5(login=None, password=None, server=None):
    """Initialize MT5 connection"""
    mt5_login = login or DEFAULT_MT5_LOGIN
    mt5_password = password or DEFAULT_MT5_PASSWORD
    mt5_server = server or DEFAULT_MT5_SERVER
    
    if not mt5.initialize():
        error_code, error_msg = mt5.last_error()
        if error_code == -6:
            logger.error("❌ MT5 terminal not running. Please start MetaTrader 5.")
        else:
            logger.error(f"❌ MT5 init failed: {error_msg}")
        return False
    
    if not mt5.login(mt5_login, password=mt5_password, server=mt5_server):
        error_code, error_msg = mt5.last_error()
        logger.error(f"❌ Login failed: {error_msg}")
        mt5.shutdown()
        return False
    
    # Enable symbols
    for symbol in SYMBOLS:
        mt5.symbol_select(symbol, True)
    
    logger.info(f"✅ MT5 initialized for account {mt5_login}")
    return True


def test_mt5_connection(login, password, server):
    """Test MT5 connection"""
    try:
        if not mt5.initialize():
            error_code, error_msg = mt5.last_error()
            if error_code == -6:
                return False, "MT5 terminal not running."
            return False, f"MT5 init failed: {error_msg}"
        
        if not mt5.login(int(login), password=password, server=server):
            error_code, error_msg = mt5.last_error()
            mt5.shutdown()
            return False, f"Login failed: {error_msg}"
        
        acc = mt5.account_info()
        if acc:
            mt5.shutdown()
            return True, f"Connected to account {acc.login}"
        
        mt5.shutdown()
        return False, "Could not get account info"
    except Exception as e:
        return False, f"Error: {str(e)}"


# ================================================================================
# ========================= DATA FETCHING ========================================
# ================================================================================

def get_data(symbol, timeframe, n=300):
    """Get market data"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# ================================================================================
# ========================= SMC STRATEGY FUNCTIONS ===============================
# ================================================================================

def trend_bias(df):
    """Determine trend with multiple EMA confirmation"""
    ema20 = df["close"].ewm(span=20).mean()
    ema50 = df["close"].ewm(span=50).mean()
    close = df["close"].iloc[-1]
    
    if close > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        return "BULLISH"
    elif close < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        return "BEARISH"
    return "NEUTRAL"


def liquidity_grab(df):
    """Detect liquidity sweeps"""
    high = df["high"].tail(10)
    low = df["low"].tail(10)
    
    sweep_high = high.iloc[-2] > high.iloc[-5:-2].max() if len(high) >= 5 else False
    sweep_low = low.iloc[-2] < low.iloc[-5:-2].min() if len(low) >= 5 else False
    
    return sweep_high, sweep_low


def order_block(df):
    """Identify order blocks"""
    if len(df) < 5:
        return None, None, None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if prev.close < prev.open and last.close > last.open:
        return "BULLISH", prev.low, prev.high
    if prev.close > prev.open and last.close < last.open:
        return "BEARISH", prev.low, prev.high
    
    return None, None, None


def fair_value_gap(df):
    """Detect FVG"""
    if len(df) < 3:
        return None, None, None
    
    c1 = df.iloc[-3]
    c3 = df.iloc[-1]
    
    if c1.high < c3.low and (c3.low - c1.high) > 0.5:
        return "BULLISH", c1.high, c3.low
    if c1.low > c3.high and (c1.low - c3.high) > 0.5:
        return "BEARISH", c3.high, c1.low
    
    return None, None, None


def check_market_structure(df):
    """Check for BOS"""
    if len(df) < 20:
        return False, False
    
    highs = df["high"].tail(20)
    lows = df["low"].tail(20)
    
    bullish_bos = highs.iloc[-1] > highs.iloc[-10:-1].max()
    bearish_bos = lows.iloc[-1] < lows.iloc[-10:-1].min()
    
    return bullish_bos, bearish_bos


# ================================================================================
# ========================= LOT CALCULATION ======================================
# ================================================================================

def calculate_lot(balance, risk_percent, sl_pips, symbol="XAUUSD"):
    """Calculate lot size with safety limits"""
    risk_money = balance * (risk_percent / 100)
    
    # Pip value varies by symbol
    if symbol == "XAUUSD":
        pip_value = 10
    elif "JPY" in symbol:
        pip_value = 1000
    else:
        pip_value = 10
    
    lot = risk_money / (sl_pips * pip_value)
    return max(0.01, min(MAX_LOT_SIZE, round(lot, 2)))


# ================================================================================
# ========================= ORDER EXECUTION ======================================
# ================================================================================

def send_order(symbol, order_type, lot, sl, tp, signal_type):
    """Send order to MT5"""
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
        "comment": signal_type,
        "deviation": 20
    }
    
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ {signal_type} executed: {lot} lots @ {price:.5f}")
        trade_stats['total_trades'] += 1
        return result
    else:
        error = result.retcode if result else 'No result'
        logger.error(f"❌ Order failed: {error}")
        return None


def place_order(symbol, order_type, lot, sl, tp, confidence=1.0, signal_type="MANUAL"):
    """Place order wrapper"""
    mt5_type = mt5.ORDER_TYPE_BUY if order_type.lower() == "buy" else mt5.ORDER_TYPE_SELL
    return send_order(symbol, mt5_type, lot, sl, tp, signal_type)


# ================================================================================
# ================= AGGRESSIVE PROFIT PROTECTION (KEY FEATURE!) =================
# ================================================================================

def manage_profit_protection(symbol, user):
    """
    AGGRESSIVE PROFIT PROTECTION SYSTEM
    
    This is the KEY feature you requested - protect profits immediately!
    
    1. BREAKEVEN: Move SL to breakeven after small profit
    2. PROFIT LOCK: Lock in percentage of maximum profit reached
    3. TRAILING: Tight trailing stop when in strong profit
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return
    
    point = symbol_info.point
    min_stop = max(symbol_info.trade_stops_level * point, point * 10)
    pip_size = point * 10  # 1 pip = 10 points for gold
    
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue
        
        entry = pos.price_open
        current_sl = pos.sl
        current_profit_pips = 0
        
        # Calculate current profit in pips
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_price = tick.bid
            current_profit_pips = (current_price - entry) / pip_size
        else:
            current_price = tick.ask
            current_profit_pips = (entry - current_price) / pip_size
        
        # Track maximum profit for this position
        position_max_profits[pos.ticket] = max(
            position_max_profits.get(pos.ticket, 0), 
            current_profit_pips
        )
        max_profit_pips = position_max_profits[pos.ticket]
        
        new_sl = None
        protection_type = None
        
        # ========== LEVEL 1: BREAKEVEN ==========
        # After BREAKEVEN_TRIGGER_PIPS profit, move SL to breakeven + offset
        if current_profit_pips >= BREAKEVEN_TRIGGER_PIPS:
            if pos.type == mt5.POSITION_TYPE_BUY:
                be_sl = entry + (BREAKEVEN_OFFSET_PIPS * pip_size)
                if current_sl < be_sl or current_sl == 0:
                    new_sl = be_sl
                    protection_type = "BREAKEVEN"
            else:
                be_sl = entry - (BREAKEVEN_OFFSET_PIPS * pip_size)
                if current_sl > be_sl or current_sl == 0:
                    new_sl = be_sl
                    protection_type = "BREAKEVEN"
        
        # ========== LEVEL 2: PROFIT LOCK ==========
        # After PROFIT_LOCK_START_PIPS, lock in PROFIT_LOCK_PERCENT of max profit
        if max_profit_pips >= PROFIT_LOCK_START_PIPS:
            lock_pips = max_profit_pips * (PROFIT_LOCK_PERCENT / 100)
            
            if pos.type == mt5.POSITION_TYPE_BUY:
                locked_sl = entry + (lock_pips * pip_size)
                if new_sl is None or locked_sl > new_sl:
                    new_sl = locked_sl
                    protection_type = f"LOCK {lock_pips:.1f}p"
            else:
                locked_sl = entry - (lock_pips * pip_size)
                if new_sl is None or locked_sl < new_sl:
                    new_sl = locked_sl
                    protection_type = f"LOCK {lock_pips:.1f}p"
        
        # ========== LEVEL 3: TIGHT TRAILING ==========
        # After TRAILING_START_PIPS, use tight trailing stop
        if current_profit_pips >= TRAILING_START_PIPS:
            trail_distance = TRAILING_DISTANCE_PIPS * pip_size
            
            if pos.type == mt5.POSITION_TYPE_BUY:
                trailing_sl = current_price - trail_distance
                if new_sl is None or trailing_sl > new_sl:
                    new_sl = trailing_sl
                    protection_type = "TRAILING"
            else:
                trailing_sl = current_price + trail_distance
                if new_sl is None or trailing_sl < new_sl:
                    new_sl = trailing_sl
                    protection_type = "TRAILING"
        
        # Apply new SL if valid
        if new_sl is not None and new_sl != current_sl:
            # Validate stop distance
            valid = False
            if pos.type == mt5.POSITION_TYPE_BUY:
                if (current_price - new_sl) >= min_stop and new_sl > current_sl:
                    valid = True
            else:
                if (new_sl - current_price) >= min_stop and (new_sl < current_sl or current_sl == 0):
                    valid = True
            
            if valid:
                result = mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp
                })
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[{user}] 🔒 {protection_type}: Position {pos.ticket} SL → {new_sl:.5f} (Profit: {current_profit_pips:.1f}p, Max: {max_profit_pips:.1f}p)")


def close_opposite_positions(symbol, trend):
    """Close positions against trend"""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    for pos in positions:
        should_close = False
        if trend == "BEARISH" and pos.type == mt5.POSITION_TYPE_BUY:
            should_close = True
        elif trend == "BULLISH" and pos.type == mt5.POSITION_TYPE_SELL:
            should_close = True
        
        if should_close:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "magic": MAGIC,
                    "deviation": 20
                })


# ================================================================================
# ========================= MAIN BOT LOOP ========================================
# ================================================================================

def run_bot(user, symbol=DEFAULT_SYMBOL):
    """Main trading bot loop with AI enhancement"""
    stop_event = user_bots[user]["stop_event"]
    
    # Get MT5 credentials
    from models import get_user_mt5_credentials
    creds = get_user_mt5_credentials(user)
    
    if creds:
        login = creds['login']
        password = creds['password']
        server = creds['server']
    else:
        login = DEFAULT_MT5_LOGIN
        password = DEFAULT_MT5_PASSWORD
        server = DEFAULT_MT5_SERVER
        logger.warning(f"[{user}] Using default MT5 credentials")
    
    if not initialize_mt5(login, password, server):
        user_bots[user]["running"] = False
        log_trade(user, 'error', 'MT5 init failed', {})
        return
    
    user_mt5_sessions[user] = True
    
    # Check AI status
    ai_status = get_ai_status()
    if ai_status['configured']:
        logger.info(f"[{user}] 🤖 AI ACTIVE - Enhanced trading enabled")
    else:
        logger.warning(f"[{user}] ⚠️ AI NOT CONFIGURED - Using technical analysis only")
    
    logger.info(f"[{user}] 🚀 Bot started on {symbol} M5 with AGGRESSIVE PROFIT PROTECTION")
    log_trade(user, 'bot', f'Bot started on {symbol}', {'ai_enabled': ai_status['configured']})
    
    prev_positions = {}
    ai_analysis_counter = 0
    last_ai_recommendation = None
    
    # Reset daily trade count
    today = datetime.now().date()
    daily_trade_counts[user] = 0
    
    while not stop_event.is_set():
        try:
            # Check if new day
            if datetime.now().date() != today:
                today = datetime.now().date()
                daily_trade_counts[user] = 0
            
            # Check daily trade limit
            if daily_trade_counts[user] >= MAX_DAILY_TRADES:
                logger.info(f"[{user}] 📊 Daily trade limit reached ({MAX_DAILY_TRADES}). Waiting...")
                stop_event.wait(60)
                continue
            
            df = get_data(symbol, TIMEFRAME)
            if df is None or len(df) < 100:
                stop_event.wait(2)
                continue
            
            # Technical analysis
            trend = trend_bias(df)
            sweep_high, sweep_low = liquidity_grab(df)
            ob_type, ob_low, ob_high = order_block(df)
            fvg_type, fvg_low, fvg_high = fair_value_gap(df)
            bullish_bos, bearish_bos = check_market_structure(df)
            
            price = df["close"].iloc[-1]
            current_trends[user] = trend
            
            # Calculate SMC scores
            buy_score = 0
            sell_score = 0
            
            if trend == "BULLISH":
                buy_score += 1
            elif trend == "BEARISH":
                sell_score += 1
            
            if sweep_low:
                buy_score += 1
            if sweep_high:
                sell_score += 1
            
            if ob_type == "BULLISH" or fvg_type == "BULLISH":
                buy_score += 1
            if ob_type == "BEARISH" or fvg_type == "BEARISH":
                sell_score += 1
            
            if bullish_bos:
                buy_score += 1
            if bearish_bos:
                sell_score += 1
            
            # ========== AI ANALYSIS ==========
            ai_analysis_counter += 1
            if ai_analysis_counter >= AI_ANALYSIS_EVERY_N_CYCLES and AI_ENABLED:
                ai_analysis_counter = 0
                last_ai_recommendation = ai_analyze_market(df, symbol, user)
                
                # Boost scores if AI agrees
                if last_ai_recommendation:
                    conf = last_ai_recommendation.get('confidence', 0)
                    rec = last_ai_recommendation.get('recommendation', 'HOLD')
                    
                    if conf >= AI_MIN_CONFIDENCE_FOR_TRADE:
                        if rec == 'BUY':
                            buy_score += 1
                        elif rec == 'SELL':
                            sell_score += 1
            
            logger.debug(f"[{user}] {symbol}: {price:.2f} | Trend: {trend} | Buy: {buy_score} | Sell: {sell_score}")
            
            positions = mt5.positions_get(symbol=symbol)
            current_pos = len(positions) if positions else 0
            acc = mt5.account_info()
            
            # Track closed positions for AI learning
            current_tickets = {p.ticket for p in positions} if positions else set()
            for ticket, pos_data in list(prev_positions.items()):
                if ticket not in current_tickets:
                    trade_result = {
                        'ticket': ticket,
                        'type': pos_data['type'],
                        'profit': pos_data.get('last_profit', 0),
                        'smc_score': pos_data.get('score', 0)
                    }
                    ai_study_trade_results(user, trade_result)
                    
                    if ticket in position_max_profits:
                        del position_max_profits[ticket]
                    del prev_positions[ticket]
            
            # Update position tracking
            if positions:
                for p in positions:
                    prev_positions[p.ticket] = {
                        'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'price': p.price_open,
                        'last_profit': p.profit,
                        'score': buy_score if p.type == mt5.POSITION_TYPE_BUY else sell_score,
                        'trend': trend
                    }
            
            if acc:
                point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.00001
                pip_size = point * 10
                
                # ========== PROFIT PROTECTION (RUNS EVERY CYCLE) ==========
                if USE_PROFIT_PROTECTION and positions:
                    manage_profit_protection(symbol, user)
                
                # Get AI parameters
                ai_params = get_ai_optimized_params(user)
                sl_pips = ai_params['sl_pips']
                tp_pips = ai_params['tp_pips']
                risk_pct = ai_params['risk_percent']
                min_score = ai_params['min_score']
                
                lot = calculate_lot(acc.balance, risk_pct, sl_pips, symbol)
                
                # ========== BUY SIGNAL ==========
                buy_conditions = (
                    (trend == "BULLISH" and (sweep_low or ob_type == "BULLISH" or fvg_type == "BULLISH")) or
                    (sweep_low and (ob_type == "BULLISH" or fvg_type == "BULLISH" or bullish_bos))
                )
                
                if buy_conditions and current_pos < MAX_POSITIONS and buy_score >= min_score:
                    # AI validation
                    ai_approved, conf_mult = ai_validate_trade_signal(df, "BUY", buy_score, user)
                    
                    # Check AI recommendation alignment
                    ai_aligned = False
                    if last_ai_recommendation:
                        rec = last_ai_recommendation.get('recommendation', 'HOLD')
                        conf = last_ai_recommendation.get('confidence', 0)
                        quality = last_ai_recommendation.get('entry_quality', 'POOR')
                        
                        ai_aligned = (
                            rec == 'BUY' and 
                            conf >= AI_MIN_CONFIDENCE_FOR_TRADE and
                            quality in AI_ENTRY_QUALITY_REQUIRED
                        )
                    
                    should_trade = ai_approved and (ai_aligned or not AI_MUST_APPROVE_TRADE)
                    
                    if should_trade:
                        adjusted_lot = max(0.01, min(MAX_LOT_SIZE, round(lot * conf_mult, 2)))
                        
                        sl = price - sl_pips * pip_size
                        tp = price + tp_pips * pip_size
                        
                        # Use AI suggested levels if high confidence
                        if last_ai_recommendation and last_ai_recommendation.get('confidence', 0) > 0.75:
                            ai_sl = last_ai_recommendation.get('suggested_sl_pips')
                            ai_tp = last_ai_recommendation.get('suggested_tp_pips')
                            if ai_sl:
                                sl = price - ai_sl * pip_size
                            if ai_tp:
                                tp = price + ai_tp * pip_size
                        
                        result = send_order(symbol, mt5.ORDER_TYPE_BUY, adjusted_lot, sl, tp, f"AI_BUY_{buy_score}")
                        
                        if result:
                            daily_trade_counts[user] += 1
                            logger.info(f"[{user}] 🟢 BUY {symbol} @ {price:.2f} | Lot: {adjusted_lot} | SL: {sl:.2f} | TP: {tp:.2f}")
                            log_trade(user, 'trade', f'BUY {symbol} @ {price:.2f}', {
                                'type': 'BUY', 'lot': adjusted_lot, 'sl': sl, 'tp': tp, 'score': buy_score
                            })
                    else:
                        logger.debug(f"[{user}] BUY signal rejected by AI")
                
                # ========== SELL SIGNAL ==========
                sell_conditions = (
                    (trend == "BEARISH" and (sweep_high or ob_type == "BEARISH" or fvg_type == "BEARISH")) or
                    (sweep_high and (ob_type == "BEARISH" or fvg_type == "BEARISH" or bearish_bos))
                )
                
                if sell_conditions and current_pos < MAX_POSITIONS and sell_score >= min_score:
                    ai_approved, conf_mult = ai_validate_trade_signal(df, "SELL", sell_score, user)
                    
                    ai_aligned = False
                    if last_ai_recommendation:
                        rec = last_ai_recommendation.get('recommendation', 'HOLD')
                        conf = last_ai_recommendation.get('confidence', 0)
                        quality = last_ai_recommendation.get('entry_quality', 'POOR')
                        
                        ai_aligned = (
                            rec == 'SELL' and 
                            conf >= AI_MIN_CONFIDENCE_FOR_TRADE and
                            quality in AI_ENTRY_QUALITY_REQUIRED
                        )
                    
                    should_trade = ai_approved and (ai_aligned or not AI_MUST_APPROVE_TRADE)
                    
                    if should_trade:
                        adjusted_lot = max(0.01, min(MAX_LOT_SIZE, round(lot * conf_mult, 2)))
                        
                        sl = price + sl_pips * pip_size
                        tp = price - tp_pips * pip_size
                        
                        if last_ai_recommendation and last_ai_recommendation.get('confidence', 0) > 0.75:
                            ai_sl = last_ai_recommendation.get('suggested_sl_pips')
                            ai_tp = last_ai_recommendation.get('suggested_tp_pips')
                            if ai_sl:
                                sl = price + ai_sl * pip_size
                            if ai_tp:
                                tp = price - ai_tp * pip_size
                        
                        result = send_order(symbol, mt5.ORDER_TYPE_SELL, adjusted_lot, sl, tp, f"AI_SELL_{sell_score}")
                        
                        if result:
                            daily_trade_counts[user] += 1
                            logger.info(f"[{user}] 🔴 SELL {symbol} @ {price:.2f} | Lot: {adjusted_lot} | SL: {sl:.2f} | TP: {tp:.2f}")
                            log_trade(user, 'trade', f'SELL {symbol} @ {price:.2f}', {
                                'type': 'SELL', 'lot': adjusted_lot, 'sl': sl, 'tp': tp, 'score': sell_score
                            })
                    else:
                        logger.debug(f"[{user}] SELL signal rejected by AI")
            
            stop_event.wait(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"[{user}] Bot error: {e}")
            stop_event.wait(5)
    
    user_bots[user]["running"] = False
    logger.info(f"[{user}] 🛑 Bot stopped")
    log_trade(user, 'bot', 'Bot stopped', {'symbol': symbol})


# ================================================================================
# ========================= BOT CONTROL ==========================================
# ================================================================================

def start_bot(user, symbol=DEFAULT_SYMBOL):
    """Start trading bot for user"""
    if user in user_bots and user_bots[user].get("running"):
        return "Bot already running"
    
    stop_event = threading.Event()
    thread = threading.Thread(target=run_bot, args=(user, symbol), daemon=True)
    user_bots[user] = {"thread": thread, "stop_event": stop_event, "running": True, "symbol": symbol}
    thread.start()
    
    return f"Bot started on {symbol} with AI enhancement"


def stop_bot(user):
    """Stop trading bot for user"""
    if user in user_bots and user_bots[user].get("running"):
        user_bots[user]["stop_event"].set()
        return "Bot stopping..."
    return "Bot not running"


def bot_status(user):
    """Get bot status"""
    return user_bots[user]["running"] if user in user_bots else False


# ================================================================================
# ========================= DASHBOARD HELPERS ====================================
# ================================================================================

def get_account_info(user=None):
    """Get MT5 account info"""
    if not mt5.terminal_info():
        if user:
            from models import get_user_mt5_credentials
            creds = get_user_mt5_credentials(user)
            if creds:
                initialize_mt5(creds['login'], creds['password'], creds['server'])
            else:
                initialize_mt5()
        else:
            initialize_mt5()
    
    acc = mt5.account_info()
    if acc:
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free
        }
    return {}


def get_positions(user=None, symbol=None):
    """Get open positions"""
    if not mt5.terminal_info():
        if user:
            from models import get_user_mt5_credentials
            creds = get_user_mt5_credentials(user)
            if creds:
                initialize_mt5(creds['login'], creds['password'], creds['server'])
            else:
                initialize_mt5()
        else:
            initialize_mt5()
    
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    
    data = []
    if positions:
        for p in positions:
            data.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit
            })
    return data


# Test AI on import
logger.info("=" * 60)
logger.info("TRADING BOT v3.0 - AI-Enhanced with Profit Protection")
logger.info("=" * 60)
ai_check = get_ai_status()
if ai_check['configured']:
    logger.info("✅ AI Status: ACTIVE")
else:
    logger.warning("⚠️ AI Status: NOT CONFIGURED - Set OPENAI_API_KEY")
logger.info("=" * 60)
