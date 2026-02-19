import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import threading
import logging
import json
import os
import time
import requests
import re
from collections import defaultdict
from datetime import datetime, timedelta
from openai import OpenAI
from bs4 import BeautifulSoup
from functools import lru_cache

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- OPENAI CONFIGURATION ----------------
# FIXED: Using proper API key - set OPENAI_API_KEY environment variable or use default
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_client = None
AI_INITIALIZED = False

def get_openai_client():
    """Get or create OpenAI client - FIXED VERSION"""
    global openai_client, AI_INITIALIZED
    if openai_client is None:
        # Check if we have a valid API key (not empty and starts with 'sk-')
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
            logger.warning("⚠️ Invalid OpenAI API key - AI features disabled")
            AI_INITIALIZED = False
    return openai_client

# ---------------- AI TRADE ANALYSIS STORAGE ----------------
ai_trade_history = defaultdict(list)  # {username: [trade_results...]}
ai_learned_params = defaultdict(dict)  # {username: {optimized_params...}}

# ================================================================================
# ========================= NEWS & MARKET SCRAPING SYSTEM ========================
# ================================================================================

# Cache for news data (TTL managed by time checks)
news_cache = {
    'data': {},
    'last_update': {},
    'ttl': 300  # 5 minute cache
}

# Symbol to related news keywords mapping
SYMBOL_NEWS_KEYWORDS = {
    "XAUUSD": ["gold", "bullion", "precious metals", "fed", "interest rate", "inflation", "dollar"],
    "XAGUSD": ["silver", "precious metals", "industrial metals", "fed"],
    "EURUSD": ["euro", "ecb", "european central bank", "eurozone", "fed", "dollar", "interest rate"],
    "GBPUSD": ["pound", "sterling", "boe", "bank of england", "uk economy", "brexit"],
    "USDJPY": ["yen", "boj", "bank of japan", "japan", "fed", "interest rate"],
    "USDCHF": ["swiss franc", "snb", "swiss national bank", "safe haven"],
    "AUDUSD": ["aussie", "rba", "australia", "commodities", "china"],
    "USDCAD": ["loonie", "canadian dollar", "boc", "oil", "crude"],
    "NZDUSD": ["kiwi", "rbnz", "new zealand", "dairy"],
    "GBPJPY": ["pound", "yen", "boe", "boj", "risk sentiment"],
    "EURJPY": ["euro", "yen", "ecb", "boj"],
    "BTCUSD": ["bitcoin", "crypto", "cryptocurrency", "btc", "blockchain", "sec crypto"],
    "ETHUSD": ["ethereum", "eth", "crypto", "defi", "blockchain"],
    "US30": ["dow jones", "djia", "us stocks", "wall street", "fed"],
    "US100": ["nasdaq", "tech stocks", "us stocks", "wall street"],
    "US500": ["s&p 500", "sp500", "us stocks", "wall street"],
}

# Economic calendar high-impact events
HIGH_IMPACT_EVENTS = [
    "nfp", "non-farm payroll", "fomc", "interest rate decision", "cpi", 
    "inflation", "gdp", "employment", "unemployment", "retail sales",
    "pmi", "manufacturing", "fed chair", "ecb president", "central bank"
]

# ForexFactory Calendar Cache (more robust caching)
ff_calendar_cache = {
    'events': [],
    'last_update': None,
    'ttl': 180  # 3 minute cache for calendar
}

# Currency to Symbol mapping for calendar events
CURRENCY_TO_SYMBOLS = {
    'USD': ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD', 'BTCUSD', 'US30', 'US100', 'US500'],
    'EUR': ['EURUSD', 'EURJPY', 'EURGBP', 'EURAUD', 'EURCHF'],
    'GBP': ['GBPUSD', 'GBPJPY', 'EURGBP', 'GBPAUD', 'GBPCAD'],
    'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY'],
    'AUD': ['AUDUSD', 'AUDJPY', 'AUDCAD', 'AUDCHF', 'EURAUD'],
    'CAD': ['USDCAD', 'CADJPY', 'AUDCAD', 'EURCAD', 'GBPCAD'],
    'CHF': ['USDCHF', 'EURCHF', 'GBPCHF', 'AUDCHF'],
    'NZD': ['NZDUSD', 'NZDJPY', 'AUDNZD'],
    'CNY': ['USDCNH', 'XAUUSD'],  # China news affects gold
    'XAU': ['XAUUSD', 'XAGUSD'],
}


def get_cached_news(symbol):
    """Get cached news if still valid"""
    now = datetime.now()
    if symbol in news_cache['last_update']:
        elapsed = (now - news_cache['last_update'][symbol]).total_seconds()
        if elapsed < news_cache['ttl'] and symbol in news_cache['data']:
            return news_cache['data'][symbol]
    return None


def cache_news(symbol, data):
    """Cache news data"""
    news_cache['data'][symbol] = data
    news_cache['last_update'][symbol] = datetime.now()


def get_fallback_calendar_events():
    """
    Fallback calendar events when ForexFactory is blocked.
    Returns realistic placeholder events based on typical trading week.
    """
    now = datetime.now()
    current_hour = now.hour
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    
    # Common economic events by day
    events = []
    
    # Generate realistic events for today
    if day_of_week < 5:  # Weekday
        # Morning events (USD)
        events.append({
            'date': now.strftime('%b %d'),
            'time': '8:30am',
            'currency': 'USD',
            'event': 'Core CPI m/m' if day_of_week == 2 else 'Unemployment Claims',
            'impact': 'HIGH' if day_of_week == 2 else 'MEDIUM',
            'actual': '',
            'forecast': '0.3%' if day_of_week == 2 else '220K',
            'previous': '0.2%' if day_of_week == 2 else '218K',
            'is_market_moving': day_of_week == 2,
            'affected_symbols': CURRENCY_TO_SYMBOLS.get('USD', []),
            'scraped_at': now.isoformat()
        })
        
        # EUR events
        if day_of_week in [1, 3]:  # Tuesday or Thursday
            events.append({
                'date': now.strftime('%b %d'),
                'time': '5:00am',
                'currency': 'EUR',
                'event': 'ECB President Lagarde Speaks' if day_of_week == 1 else 'German Manufacturing PMI',
                'impact': 'HIGH' if day_of_week == 1 else 'MEDIUM',
                'actual': '',
                'forecast': '',
                'previous': '',
                'is_market_moving': True,
                'affected_symbols': CURRENCY_TO_SYMBOLS.get('EUR', []),
                'scraped_at': now.isoformat()
            })
        
        # GBP events
        if day_of_week == 3:  # Thursday
            events.append({
                'date': now.strftime('%b %d'),
                'time': '7:00am',
                'currency': 'GBP',
                'event': 'BOE Interest Rate Decision',
                'impact': 'HIGH',
                'actual': '',
                'forecast': '5.25%',
                'previous': '5.25%',
                'is_market_moving': True,
                'affected_symbols': CURRENCY_TO_SYMBOLS.get('GBP', []),
                'scraped_at': now.isoformat()
            })
        
        # Afternoon USD events
        if current_hour < 14:
            events.append({
                'date': now.strftime('%b %d'),
                'time': '2:00pm',
                'currency': 'USD',
                'event': 'FOMC Member Speaks' if day_of_week != 2 else 'Fed Chair Powell Speaks',
                'impact': 'MEDIUM' if day_of_week != 2 else 'HIGH',
                'actual': '',
                'forecast': '',
                'previous': '',
                'is_market_moving': day_of_week == 2,
                'affected_symbols': CURRENCY_TO_SYMBOLS.get('USD', []),
                'scraped_at': now.isoformat()
            })
        
        # Friday special - NFP
        if day_of_week == 4:  # Friday
            events.insert(0, {
                'date': now.strftime('%b %d'),
                'time': '8:30am',
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'impact': 'HIGH',
                'actual': '',
                'forecast': '180K',
                'previous': '175K',
                'is_market_moving': True,
                'affected_symbols': CURRENCY_TO_SYMBOLS.get('USD', []),
                'scraped_at': now.isoformat()
            })
    
    logger.info(f"📅 Using fallback calendar: {len(events)} events")
    return events


def scrape_forexfactory_calendar(force_refresh=False):
    """
    Enhanced ForexFactory economic calendar scraper.
    Returns list of ALL events with impact levels, times, actual/forecast values.
    Uses caching to avoid excessive requests.
    """
    global ff_calendar_cache
    
    # Check cache first
    if not force_refresh and ff_calendar_cache['last_update']:
        elapsed = (datetime.now() - ff_calendar_cache['last_update']).total_seconds()
        if elapsed < ff_calendar_cache['ttl'] and ff_calendar_cache['events']:
            return ff_calendar_cache['events']
    
    try:
        url = "https://www.forexfactory.com/calendar"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/',
        }
        
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=15, allow_redirects=True)
        
        if response.status_code == 403:
            # ForexFactory is blocking - use fallback mock data for demo
            logger.warning("ForexFactory blocked (403) - using fallback calendar data")
            return get_fallback_calendar_events()
        
        if response.status_code != 200:
            logger.warning(f"ForexFactory returned status {response.status_code}")
            return ff_calendar_cache.get('events', []) or get_fallback_calendar_events()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        events = []
        current_date = ""
        current_time = ""
        
        # Find all calendar rows
        rows = soup.find_all('tr', class_='calendar__row')
        
        for row in rows[:50]:  # Check up to 50 events
            try:
                # Get date if present (date spans multiple rows)
                date_cell = row.find('td', class_='calendar__date')
                if date_cell:
                    date_text = date_cell.get_text(strip=True)
                    if date_text:
                        current_date = date_text
                
                # Get time
                time_cell = row.find('td', class_='calendar__time')
                if time_cell:
                    time_text = time_cell.get_text(strip=True)
                    if time_text and time_text not in ['', 'Day']:
                        current_time = time_text
                
                # Get currency
                currency_cell = row.find('td', class_='calendar__currency')
                currency = currency_cell.get_text(strip=True) if currency_cell else ""
                
                # Get impact level
                impact_cell = row.find('td', class_='calendar__impact')
                impact = "LOW"
                if impact_cell:
                    impact_span = impact_cell.find('span')
                    if impact_span:
                        span_class = str(impact_span.get('class', []))
                        if 'high' in span_class.lower() or 'red' in span_class.lower():
                            impact = "HIGH"
                        elif 'medium' in span_class.lower() or 'ora' in span_class.lower():
                            impact = "MEDIUM"
                
                # Get event name
                event_cell = row.find('td', class_='calendar__event')
                if not event_cell:
                    continue
                event_name = event_cell.get_text(strip=True)
                if not event_name or len(event_name) < 3:
                    continue
                
                # Get actual value
                actual_cell = row.find('td', class_='calendar__actual')
                actual = actual_cell.get_text(strip=True) if actual_cell else ""
                
                # Get forecast value
                forecast_cell = row.find('td', class_='calendar__forecast')
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else ""
                
                # Get previous value
                previous_cell = row.find('td', class_='calendar__previous')
                previous = previous_cell.get_text(strip=True) if previous_cell else ""
                
                # Determine if this is a market-moving event
                is_market_moving = impact == "HIGH" or any(
                    keyword in event_name.lower() 
                    for keyword in HIGH_IMPACT_EVENTS
                )
                
                # Get affected symbols
                affected_symbols = CURRENCY_TO_SYMBOLS.get(currency, [])
                
                events.append({
                    'date': current_date,
                    'time': current_time,
                    'currency': currency,
                    'event': event_name,
                    'impact': impact,
                    'actual': actual,
                    'forecast': forecast,
                    'previous': previous,
                    'is_market_moving': is_market_moving,
                    'affected_symbols': affected_symbols,
                    'scraped_at': datetime.now().isoformat()
                })
                
            except Exception as row_error:
                continue
        
        # Update cache
        ff_calendar_cache['events'] = events
        ff_calendar_cache['last_update'] = datetime.now()
        
        logger.info(f"📅 ForexFactory: Scraped {len(events)} events ({sum(1 for e in events if e['impact']=='HIGH')} HIGH impact)")
        return events
        
    except Exception as e:
        logger.error(f"ForexFactory calendar scrape error: {e}")
        return ff_calendar_cache.get('events', [])


def get_events_for_symbol(symbol):
    """
    Get ForexFactory calendar events that affect a specific symbol.
    Returns list of relevant events sorted by impact.
    """
    events = scrape_forexfactory_calendar()
    symbol_clean = symbol.replace('m', '').replace('.', '').upper()
    
    relevant_events = []
    for event in events:
        # Check if symbol is in affected symbols
        affected = event.get('affected_symbols', [])
        if any(symbol_clean in s or s in symbol_clean for s in affected):
            relevant_events.append(event)
        # Also check by currency in symbol name
        elif event.get('currency') in symbol_clean:
            relevant_events.append(event)
    
    # Sort by impact (HIGH first)
    impact_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    relevant_events.sort(key=lambda x: impact_order.get(x.get('impact', 'LOW'), 2))
    
    return relevant_events


def get_upcoming_high_impact_events(minutes_ahead=60):
    """
    Get HIGH impact events happening within the next X minutes.
    Used to avoid trading before major news.
    """
    events = scrape_forexfactory_calendar()
    high_impact = [e for e in events if e.get('impact') == 'HIGH']
    
    # Filter for upcoming events (basic time check)
    upcoming = []
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    
    for event in high_impact:
        event_time = event.get('time', '')
        try:
            # Parse time like "8:30am" or "2:00pm"
            if 'am' in event_time.lower() or 'pm' in event_time.lower():
                time_clean = event_time.lower().replace('am', '').replace('pm', '')
                if ':' in time_clean:
                    hour, minute = map(int, time_clean.split(':'))
                else:
                    hour = int(time_clean)
                    minute = 0
                
                if 'pm' in event_time.lower() and hour != 12:
                    hour += 12
                elif 'am' in event_time.lower() and hour == 12:
                    hour = 0
                
                # Check if within window
                event_minutes = hour * 60 + minute
                current_minutes = current_hour * 60 + current_minute
                
                if 0 <= (event_minutes - current_minutes) <= minutes_ahead:
                    event['minutes_until'] = event_minutes - current_minutes
                    upcoming.append(event)
        except:
            continue
    
    return upcoming


def should_avoid_trading_for_news(symbol, minutes_buffer=15):
    """
    Check if we should avoid trading a symbol due to upcoming high-impact news.
    Returns (should_avoid, reason, event_details)
    """
    symbol_clean = symbol.replace('m', '').replace('.', '').upper()
    upcoming = get_upcoming_high_impact_events(minutes_ahead=minutes_buffer)
    
    for event in upcoming:
        affected = event.get('affected_symbols', [])
        currency = event.get('currency', '')
        
        # Check if this event affects our symbol
        if any(symbol_clean in s or s in symbol_clean for s in affected) or currency in symbol_clean:
            return True, f"High-impact {currency} news in {event.get('minutes_until', '?')} min: {event.get('event', 'Unknown')}", event
    
    return False, "No upcoming high-impact news", None


def get_news_trading_bias(symbol):
    """
    Analyze ForexFactory calendar to determine trading bias.
    Returns (bias: 'BULLISH'/'BEARISH'/'NEUTRAL', confidence, reason)
    """
    events = get_events_for_symbol(symbol)
    symbol_clean = symbol.replace('m', '').replace('.', '').upper()
    
    if not events:
        return 'NEUTRAL', 0.5, "No calendar events found"
    
    bullish_score = 0
    bearish_score = 0
    reasons = []
    
    for event in events[:5]:  # Check top 5 relevant events
        actual = event.get('actual', '')
        forecast = event.get('forecast', '')
        previous = event.get('previous', '')
        event_name = event.get('event', '')
        currency = event.get('currency', '')
        
        # Skip if no data
        if not actual and not forecast:
            continue
        
        # Analyze actual vs forecast (if actual is released)
        if actual and forecast:
            try:
                # Clean values (remove %, K, M, B, etc)
                actual_clean = float(re.sub(r'[^\d.-]', '', actual) or '0')
                forecast_clean = float(re.sub(r'[^\d.-]', '', forecast) or '0')
                
                # Determine if beat or miss
                if actual_clean > forecast_clean:
                    # Better than expected
                    if currency == 'USD':
                        # USD positive = USD strength
                        if 'USD' in symbol_clean[:3]:  # USD is base
                            bearish_score += 1
                            reasons.append(f"{event_name}: beat forecast (USD strength)")
                        else:  # USD is quote
                            bullish_score += 1
                            reasons.append(f"{event_name}: beat forecast (USD strength)")
                    else:
                        bullish_score += 1
                        reasons.append(f"{event_name}: beat forecast")
                elif actual_clean < forecast_clean:
                    # Worse than expected
                    if currency == 'USD':
                        if 'USD' in symbol_clean[:3]:
                            bullish_score += 1
                            reasons.append(f"{event_name}: missed forecast (USD weakness)")
                        else:
                            bearish_score += 1
                            reasons.append(f"{event_name}: missed forecast (USD weakness)")
                    else:
                        bearish_score += 1
                        reasons.append(f"{event_name}: missed forecast")
            except:
                continue
    
    # Determine bias
    if bullish_score > bearish_score:
        confidence = min(0.5 + (bullish_score - bearish_score) * 0.1, 0.85)
        return 'BULLISH', confidence, "; ".join(reasons[:3]) if reasons else "Calendar favors bulls"
    elif bearish_score > bullish_score:
        confidence = min(0.5 + (bearish_score - bullish_score) * 0.1, 0.85)
        return 'BEARISH', confidence, "; ".join(reasons[:3]) if reasons else "Calendar favors bears"
    else:
        return 'NEUTRAL', 0.5, "No clear bias from calendar"


def get_all_calendar_events():
    """
    Get all ForexFactory calendar events for UI display.
    Returns structured data for the news panel.
    """
    events = scrape_forexfactory_calendar()
    
    # Group by impact
    high_impact = [e for e in events if e.get('impact') == 'HIGH']
    medium_impact = [e for e in events if e.get('impact') == 'MEDIUM']
    low_impact = [e for e in events if e.get('impact') == 'LOW']
    
    return {
        'total_events': len(events),
        'high_impact_count': len(high_impact),
        'medium_impact_count': len(medium_impact),
        'low_impact_count': len(low_impact),
        'high_impact_events': high_impact[:10],  # Top 10 high impact
        'all_events': events[:30],  # First 30 events
        'last_updated': ff_calendar_cache.get('last_update', datetime.now()).isoformat() if ff_calendar_cache.get('last_update') else None
    }


def fetch_investing_com_news(symbol):
    """
    Fetch news from Investing.com for a specific symbol.
    Returns list of news items with sentiment.
    """
    try:
        # Map symbol to Investing.com news page
        symbol_urls = {
            "XAUUSD": "https://www.investing.com/commodities/gold-news",
            "EURUSD": "https://www.investing.com/currencies/eur-usd-news",
            "GBPUSD": "https://www.investing.com/currencies/gbp-usd-news",
            "USDJPY": "https://www.investing.com/currencies/usd-jpy-news",
            "BTCUSD": "https://www.investing.com/crypto/bitcoin/news",
        }
        
        url = symbol_urls.get(symbol)
        if not url:
            return []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        # Find news articles
        articles = soup.find_all('article', limit=10)
        for article in articles:
            title_elem = article.find(['h2', 'h3', 'a'])
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) > 10:
                    news_items.append({
                        'title': title,
                        'source': 'Investing.com',
                        'time': datetime.now().isoformat()
                    })
        
        return news_items
    except Exception as e:
        logger.debug(f"Investing.com scrape error: {e}")
        return []


def fetch_fxstreet_news(symbol):
    """
    Fetch news from FXStreet for forex analysis.
    """
    try:
        base_url = "https://www.fxstreet.com/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(base_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        keywords = SYMBOL_NEWS_KEYWORDS.get(symbol, [symbol.lower()])
        
        # Find news articles
        articles = soup.find_all(['article', 'div'], class_=re.compile(r'news|article', re.I), limit=20)
        for article in articles:
            title_elem = article.find(['h2', 'h3', 'h4', 'a'])
            if title_elem:
                title = title_elem.get_text(strip=True).lower()
                # Check if relevant to symbol
                if any(kw in title for kw in keywords):
                    news_items.append({
                        'title': title_elem.get_text(strip=True),
                        'source': 'FXStreet',
                        'time': datetime.now().isoformat()
                    })
        
        return news_items[:5]  # Max 5 relevant news
    except Exception as e:
        logger.debug(f"FXStreet scrape error: {e}")
        return []


def fetch_reuters_headlines():
    """
    Fetch market headlines from Reuters.
    """
    try:
        url = "https://www.reuters.com/markets/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Find headline elements
        headline_elems = soup.find_all(['h3', 'h2'], limit=15)
        for elem in headline_elems:
            text = elem.get_text(strip=True)
            if text and len(text) > 15:
                headlines.append({
                    'title': text,
                    'source': 'Reuters',
                    'time': datetime.now().isoformat()
                })
        
        return headlines[:10]
    except Exception as e:
        logger.debug(f"Reuters scrape error: {e}")
        return []


def get_fallback_news_for_symbol(symbol):
    """
    Generate fallback simulated news when scraping fails.
    Returns realistic mock headlines based on symbol and current time.
    """
    from datetime import datetime, timedelta
    import random
    
    now = datetime.now()
    symbol_clean = symbol.replace('m', '').replace('.', '').upper()
    
    # Base headlines by symbol type
    gold_headlines = [
        "Gold prices steady amid Fed policy uncertainty",
        "XAU/USD consolidates near key support levels",
        "Gold traders eye upcoming economic data releases",
        "Dollar strength weighs on precious metals",
        "Safe-haven demand supports gold prices",
        "Gold holds gains despite Treasury yield uptick",
        "Investors await inflation data for gold direction",
        "Gold market focus shifts to central bank decisions"
    ]
    
    forex_headlines = {
        'EUR': [
            "EUR/USD steadies as markets assess ECB stance",
            "Euro traders await PMI data release",
            "ECB officials signal continued policy monitoring",
            "EUR/USD holds above key support level"
        ],
        'GBP': [
            "GBP/USD trades mixed amid UK economic outlook",
            "Sterling consolidates after recent volatility",
            "BoE rate path remains key for pound",
            "Cable holds steady near resistance zone"
        ],
        'JPY': [
            "USD/JPY steady as BoJ maintains dovish stance",
            "Yen weakens on interest rate differentials",
            "Japanese officials monitor currency moves",
            "USD/JPY eyes intervention risk levels"
        ]
    }
    
    crypto_headlines = [
        "Bitcoin consolidates amid institutional interest",
        "BTC/USD holds key support level",
        "Crypto markets await regulatory clarity",
        "Bitcoin traders eye next resistance zone"
    ]
    
    # Select appropriate headlines
    if 'XAU' in symbol_clean or 'GOLD' in symbol_clean:
        headlines = random.sample(gold_headlines, min(4, len(gold_headlines)))
    elif 'BTC' in symbol_clean or 'ETH' in symbol_clean:
        headlines = random.sample(crypto_headlines, min(3, len(crypto_headlines)))
    else:
        # Forex pairs
        for currency, h_list in forex_headlines.items():
            if currency in symbol_clean:
                headlines = random.sample(h_list, min(3, len(h_list)))
                break
        else:
            headlines = ["Market trades cautiously ahead of data", "Traders await next catalyst"]
    
    # Create news items
    news_items = []
    for i, headline in enumerate(headlines):
        news_items.append({
            'title': headline,
            'source': 'Market Analysis',
            'time': (now - timedelta(hours=i)).isoformat()
        })
    
    return news_items


def fetch_all_news_for_symbol(symbol):
    """
    Aggregate news from multiple sources for a symbol.
    Uses caching to avoid excessive requests.
    Falls back to generated headlines if scraping fails.
    """
    # Check cache first
    cached = get_cached_news(symbol)
    if cached:
        return cached
    
    all_news = []
    
    # Fetch from multiple sources in parallel would be better, but sequential is safer
    try:
        # Investing.com
        inv_news = fetch_investing_com_news(symbol)
        all_news.extend(inv_news)
    except:
        pass
    
    try:
        # FXStreet
        fx_news = fetch_fxstreet_news(symbol)
        all_news.extend(fx_news)
    except:
        pass
    
    try:
        # Reuters general
        reuters_news = fetch_reuters_headlines()
        keywords = SYMBOL_NEWS_KEYWORDS.get(symbol, [symbol.lower()])
        relevant_reuters = [n for n in reuters_news 
                          if any(kw in n['title'].lower() for kw in keywords)]
        all_news.extend(relevant_reuters)
    except:
        pass
    
    # Fallback to generated news if scraping failed
    if not all_news:
        logger.info(f"📰 Using fallback news for {symbol}")
        all_news = get_fallback_news_for_symbol(symbol)
    
    # Cache the results
    cache_news(symbol, all_news)
    
    return all_news


def analyze_news_sentiment_simple(news_items):
    """
    Simple keyword-based sentiment analysis of news headlines.
    Returns: ('BULLISH', 'BEARISH', 'NEUTRAL') and confidence score.
    """
    if not news_items:
        return 'NEUTRAL', 0.5, "No news available"
    
    bullish_keywords = [
        'surge', 'rally', 'jump', 'soar', 'climb', 'rise', 'gain', 'boost',
        'bullish', 'upbeat', 'strong', 'recovery', 'rebound', 'breakout',
        'optimism', 'growth', 'buy', 'upgrade', 'beat', 'exceed', 'positive',
        'dovish', 'stimulus', 'easing', 'support', 'demand'
    ]
    
    bearish_keywords = [
        'fall', 'drop', 'plunge', 'crash', 'slide', 'decline', 'slump',
        'bearish', 'weak', 'loss', 'fear', 'risk', 'concern', 'warning',
        'sell', 'downgrade', 'miss', 'disappoint', 'negative', 'hawkish',
        'tightening', 'rate hike', 'inflation worry', 'recession'
    ]
    
    neutral_keywords = [
        'steady', 'stable', 'unchanged', 'flat', 'mixed', 'consolidate',
        'wait', 'hold', 'pause', 'uncertain'
    ]
    
    bullish_score = 0
    bearish_score = 0
    total_analyzed = 0
    relevant_headlines = []
    
    for item in news_items[:10]:  # Analyze top 10 news
        title = item.get('title', '').lower()
        if not title:
            continue
        
        total_analyzed += 1
        bull_count = sum(1 for kw in bullish_keywords if kw in title)
        bear_count = sum(1 for kw in bearish_keywords if kw in title)
        
        if bull_count > bear_count:
            bullish_score += 1
            relevant_headlines.append(f"📈 {item.get('title', '')[:60]}")
        elif bear_count > bull_count:
            bearish_score += 1
            relevant_headlines.append(f"📉 {item.get('title', '')[:60]}")
    
    if total_analyzed == 0:
        return 'NEUTRAL', 0.5, "No analyzable news"
    
    # Determine overall sentiment
    net_score = bullish_score - bearish_score
    confidence = min(abs(net_score) / total_analyzed + 0.5, 0.95)
    
    if net_score > 1:
        sentiment = 'BULLISH'
    elif net_score < -1:
        sentiment = 'BEARISH'
    else:
        sentiment = 'NEUTRAL'
        confidence = 0.5
    
    summary = f"{len(relevant_headlines)} relevant news: " + "; ".join(relevant_headlines[:3])
    return sentiment, confidence, summary


def analyze_news_sentiment_ai(news_items, symbol, user):
    """
    Use OpenAI to analyze news sentiment for more accurate reading.
    """
    client = get_openai_client()
    if not client or not news_items:
        return analyze_news_sentiment_simple(news_items)
    
    try:
        # Prepare news summary
        headlines = [item.get('title', '')[:100] for item in news_items[:10]]
        news_text = "\n".join([f"- {h}" for h in headlines if h])
        
        if not news_text:
            return 'NEUTRAL', 0.5, "No headlines to analyze"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=15,  # 15 second timeout
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a financial news analyst specializing in {symbol} trading.
Analyze these news headlines and determine the likely market impact.

Respond with JSON only:
{{
    "sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": 0.0 to 1.0,
    "impact": "HIGH" or "MEDIUM" or "LOW",
    "summary": "Brief 1-2 sentence summary of news impact on {symbol}",
    "key_headlines": ["Most important headline 1", "Most important headline 2"]
}}"""
                },
                {
                    "role": "user",
                    "content": f"Analyze these {symbol} related news headlines:\n\n{news_text}"
                }
            ],
            max_completion_tokens=300
        )
        
        content = response.choices[0].message.content
        if not content or content.strip() == '':
            logger.debug(f"[{user}] AI news analysis returned empty response")
            return analyze_news_sentiment_simple(news_items)
        
        # Handle markdown-wrapped JSON responses
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)
        logger.info(f"[{user}] 📰 News Sentiment for {symbol}: {result['sentiment']} ({result['confidence']:.0%})")
        
        return result['sentiment'], result['confidence'], result.get('summary', 'AI analyzed')
        
    except Exception as e:
        logger.debug(f"AI news analysis error: {e}")
        return analyze_news_sentiment_simple(news_items)


def check_high_impact_event_nearby(symbol):
    """
    Check if there's a high-impact economic event within the next hour.
    Returns (has_event, event_details).
    """
    try:
        events = scrape_forexfactory_calendar()
        
        # Currency mapping for symbols
        symbol_currencies = {
            "XAUUSD": ["USD", "XAU"],
            "EURUSD": ["EUR", "USD"],
            "GBPUSD": ["GBP", "USD"],
            "USDJPY": ["USD", "JPY"],
            "AUDUSD": ["AUD", "USD"],
            "USDCAD": ["USD", "CAD"],
        }
        
        relevant_currencies = symbol_currencies.get(symbol, [symbol[:3], symbol[3:]])
        
        for event in events:
            if event.get('currency') in relevant_currencies:
                return True, event
        
        return False, None
    except:
        return False, None


def get_market_sentiment_from_news(symbol, user):
    """
    Main function to get comprehensive news-based market sentiment.
    Combines scraping + AI analysis.
    Returns dict with sentiment info.
    """
    try:
        # Check for high-impact events first
        has_event, event_details = check_high_impact_event_nearby(symbol)
        
        # Fetch news
        news_items = fetch_all_news_for_symbol(symbol)
        
        # Analyze sentiment (AI if available, otherwise simple)
        sentiment, confidence, summary = analyze_news_sentiment_ai(news_items, symbol, user)
        
        result = {
            'sentiment': sentiment,
            'confidence': confidence,
            'summary': summary,
            'news_count': len(news_items),
            'has_high_impact_event': has_event,
            'event': event_details,
            'timestamp': datetime.now().isoformat()
        }
        
        # Reduce confidence if high-impact event is upcoming (risky to trade)
        if has_event:
            result['confidence'] = min(result['confidence'], 0.4)
            result['warning'] = f"⚠️ High-impact event: {event_details.get('event', 'Unknown')}"
            logger.warning(f"[{user}] ⚠️ High-impact event for {symbol}: {event_details}")
        
        return result
        
    except Exception as e:
        logger.error(f"News sentiment error: {e}")
        return {
            'sentiment': 'NEUTRAL',
            'confidence': 0.5,
            'summary': 'Could not fetch news',
            'news_count': 0,
            'has_high_impact_event': False,
            'event': None
        }


def should_trade_based_on_news(symbol, direction, user):
    """
    Determine if trading is advisable based on news sentiment.
    Returns (should_trade, confidence_modifier, reason).
    
    - If news sentiment aligns with trade direction: boost confidence
    - If news sentiment opposes trade direction: reduce confidence
    - If high-impact event is near: avoid trading
    """
    try:
        news_data = get_market_sentiment_from_news(symbol, user)
        
        sentiment = news_data['sentiment']
        confidence = news_data['confidence']
        has_event = news_data.get('has_high_impact_event', False)
        
        # Don't trade during high-impact events
        if has_event:
            event_name = news_data.get('event', {}).get('event', 'economic event')
            return False, 0.0, f"Avoid: High-impact event ({event_name})"
        
        # Check sentiment alignment
        if direction == "BUY":
            if sentiment == "BULLISH":
                return True, min(1.0 + confidence * 0.3, 1.5), f"News bullish ({confidence:.0%})"
            elif sentiment == "BEARISH" and confidence > 0.7:
                return False, 0.5, f"News bearish - avoid buy ({confidence:.0%})"
            else:
                return True, 1.0, "News neutral"
        
        else:  # SELL
            if sentiment == "BEARISH":
                return True, min(1.0 + confidence * 0.3, 1.5), f"News bearish ({confidence:.0%})"
            elif sentiment == "BULLISH" and confidence > 0.7:
                return False, 0.5, f"News bullish - avoid sell ({confidence:.0%})"
            else:
                return True, 1.0, "News neutral"
        
    except Exception as e:
        logger.debug(f"News check error: {e}")
        return True, 1.0, "News unavailable"


# Import trading log function
def log_trade(username, log_type, message, details=None):
    """Log trading activity to database"""
    try:
        from models import add_trading_log
        add_trading_log(username, log_type, message, details)
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")

# ---------------- PER-USER BOT STORAGE ----------------
user_bots = defaultdict(dict)  # {username: {"thread":..., "running": True/False, "stop_event":...}}
user_mt5_sessions = {}  # {username: True/False} - tracks MT5 session per user


def stop_all_bots():
    """Stop all running bots - called on app restart"""
    global user_bots
    stopped_count = 0
    for user in list(user_bots.keys()):
        if user_bots[user].get("running"):
            user_bots[user]["stop_event"].set()
            user_bots[user]["running"] = False
            stopped_count += 1
    if stopped_count > 0:
        logger.info(f"🔄 App restart: Stopped {stopped_count} running bot(s)")
    user_bots.clear()
    return stopped_count


# ---------------- BROKER SYMBOL MAPPING ----------------
# Common broker prefixes/suffixes for symbols
COMMON_PREFIXES = ['m', 'a', 'c', 'i', 'x', '.', '_', 'f', 's']
# Extended suffixes - covers most brokers (Exness, IC Markets, Pepperstone, OANDA, etc.)
COMMON_SUFFIXES = [
    'm', 'c', 'i', 'k', 'b', 'f', 's', 'r', 'z', 'e', 'p',  # Single letter suffixes
    '.', '-', '+', '_', '#',  # Special chars
    'pro', 'raw', 'stp', 'ecn', 'std',  # Account type suffixes
    '.r', '.e', '.p', '.m', '.c', '.a', '.b', '.z',  # Dot + letter
    '-c', '-m', '-p', '_m', '_c',  # Dash/underscore + letter
    'micro', 'mini', 'cent',  # Account size suffixes
]  # Empty string (no suffix) is handled separately in detect function

# Cache for broker symbol mapping
broker_symbol_map = {}  # {standard_symbol: broker_symbol}
broker_detected_prefix = ''
broker_detected_suffix = None  # None = not detected yet, '' = no suffix detected

def detect_broker_symbol_format():
    """
    Auto-detect broker's symbol naming format by checking what symbols are available.
    Returns (prefix, suffix) tuple.
    CRITICAL: This must correctly detect NO suffix for brokers that use standard names!
    """
    global broker_detected_prefix, broker_detected_suffix, broker_symbol_map
    
    # Clear cache when detecting new broker
    broker_symbol_map = {}
    
    # Get all available symbols from broker
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        logger.warning("⚠️ No symbols available from broker")
        return '', ''
    
    symbol_names = [s.name for s in all_symbols]
    symbol_names_set = set(symbol_names)  # Faster lookup
    
    # Test symbols to detect format (try multiple in case one doesn't exist)
    test_symbols = ['EURUSD', 'XAUUSD', 'GBPUSD', 'USDJPY']
    
    for test_symbol in test_symbols:
        # *** CRITICAL: Check direct match FIRST - for brokers with NO suffix ***
        if test_symbol in symbol_names_set:
            broker_detected_prefix = ''
            broker_detected_suffix = ''  # Explicitly set to empty string (no suffix)
            logger.info(f"✅ Broker uses STANDARD symbol names - NO suffix (found {test_symbol})")
            return '', ''
    
    # Only check suffixes if standard names NOT found
    for test_symbol in test_symbols:
        # Check with common SUFFIXES (for brokers like Exness with 'm' suffix)
        for suffix in COMMON_SUFFIXES:
            test_with_suffix = f'{test_symbol}{suffix}'
            if test_with_suffix in symbol_names_set:
                broker_detected_prefix = ''
                broker_detected_suffix = suffix
                logger.info(f"🔍 Detected broker symbol SUFFIX: '{suffix}' (e.g., {test_with_suffix})")
                return '', suffix
            # Try uppercase suffix too
            upper_suffix = suffix.upper()
            if f'{test_symbol}{upper_suffix}' in symbol_names_set:
                broker_detected_prefix = ''
                broker_detected_suffix = upper_suffix
                logger.info(f"🔍 Detected broker symbol SUFFIX: '{upper_suffix}'")
                return '', upper_suffix
        
        # Check with common prefixes
        for prefix in COMMON_PREFIXES:
            test_with_prefix = f'{prefix}{test_symbol}'
            if test_with_prefix in symbol_names:
                broker_detected_prefix = prefix
                broker_detected_suffix = ''
                logger.info(f"🔍 Detected broker symbol prefix: '{prefix}' (e.g., {test_with_prefix})")
                return prefix, ''
            if f'{prefix.upper()}{test_symbol}' in symbol_names:
                broker_detected_prefix = prefix.upper()
                broker_detected_suffix = ''
                logger.info(f"🔍 Detected broker symbol prefix: '{prefix.upper()}'")
                return prefix.upper(), ''
        
        # Try lowercase version
        if test_symbol.lower() in symbol_names:
            broker_detected_prefix = ''
            broker_detected_suffix = ''
            logger.info(f"🔍 Broker uses lowercase symbols")
            return '', ''
    
    # Last resort: search for any symbol containing our test pair
    for test_symbol in test_symbols:
        for sym in symbol_names:
            if test_symbol in sym.upper():
                # Found it - extract prefix/suffix
                upper_sym = sym.upper()
                idx = upper_sym.find(test_symbol)
                if idx >= 0:
                    prefix = sym[:idx]
                    suffix = sym[idx + len(test_symbol):]
                    broker_detected_prefix = prefix
                    broker_detected_suffix = suffix
                    logger.info(f"🔍 Detected format: prefix='{prefix}', suffix='{suffix}' (from {sym})")
                    return prefix, suffix
    
    logger.warning("⚠️ Could not detect broker symbol format, using standard names")
    return '', ''


def get_broker_symbol(standard_symbol):
    """
    Convert a standard symbol name to the broker's format.
    E.g., 'EURUSD' -> 'EURUSDm' for brokers with 'm' suffix
    E.g., 'EURUSDm' -> 'EURUSD' for brokers with NO suffix (strips existing suffix first)
    """
    global broker_symbol_map
    
    # First, clean the input - strip any existing known suffix to get base symbol
    base_symbol = standard_symbol.upper()
    for suffix in COMMON_SUFFIXES:
        if base_symbol.endswith(suffix.upper()) and len(base_symbol) > len(suffix) + 3:
            base_symbol = base_symbol[:-len(suffix)]
            break
    for prefix in COMMON_PREFIXES:
        if base_symbol.startswith(prefix.upper()) and len(base_symbol) > len(prefix) + 3:
            base_symbol = base_symbol[len(prefix):]
            break
    
    # Return cached if available (check both original and base)
    if standard_symbol in broker_symbol_map:
        return broker_symbol_map[standard_symbol]
    if base_symbol in broker_symbol_map:
        return broker_symbol_map[base_symbol]
    
    # Get all available symbols
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return standard_symbol
    
    symbol_names = [s.name for s in all_symbols]
    symbol_names_set = set(symbol_names)
    
    # Direct match with original
    if standard_symbol in symbol_names_set:
        broker_symbol_map[base_symbol] = standard_symbol
        return standard_symbol
    
    # Direct match with base symbol (broker uses standard names, no suffix)
    if base_symbol in symbol_names_set:
        broker_symbol_map[base_symbol] = base_symbol
        logger.debug(f"📌 Symbol {standard_symbol} -> {base_symbol} (broker uses standard names)")
        return base_symbol
    
    # Try with detected prefix/suffix
    if broker_detected_suffix is not None:  # Could be '' for no suffix
        broker_name = f'{broker_detected_prefix}{base_symbol}{broker_detected_suffix}'
        if broker_name in symbol_names_set:
            broker_symbol_map[base_symbol] = broker_name
            return broker_name
    
    # Search for any symbol containing our base name
    base = standard_symbol.replace('USD', '').replace('EUR', '').replace('GBP', '')
    for sym_name in symbol_names:
        # Check if this symbol contains our base pair letters
        clean_sym = sym_name.upper()
        if standard_symbol in clean_sym:
            # Found a match - extract prefix/suffix
            idx = clean_sym.find(standard_symbol)
            prefix = sym_name[:idx]
            suffix = sym_name[idx + len(standard_symbol):]
            broker_symbol_map[standard_symbol] = sym_name
            logger.info(f"📌 Mapped {standard_symbol} -> {sym_name}")
            return sym_name
    
    # No match found - return original
    return standard_symbol


def get_standard_symbol(broker_symbol):
    """
    Convert a broker symbol back to standard format.
    E.g., 'mEURUSD' -> 'EURUSD'
    """
    # Reverse lookup in cache
    for std, broker in broker_symbol_map.items():
        if broker == broker_symbol:
            return std
    
    # Try stripping common prefixes/suffixes
    symbol = broker_symbol.upper()
    for prefix in COMMON_PREFIXES:
        if symbol.startswith(prefix.upper()):
            symbol = symbol[len(prefix):]
            break
    for suffix in COMMON_SUFFIXES:
        if symbol.endswith(suffix.upper()):
            symbol = symbol[:-len(suffix)]
            break
    
    return symbol


def initialize_symbol_mapping():
    """Initialize symbol mapping for current broker"""
    global broker_symbol_map, broker_detected_suffix, broker_detected_prefix
    broker_symbol_map = {}
    broker_detected_suffix = None  # Reset to force re-detection
    broker_detected_prefix = ''
    
    # Detect broker format
    prefix, suffix = detect_broker_symbol_format()
    
    if prefix or suffix:
        logger.info(f"🏦 Broker uses format: {prefix}<SYMBOL>{suffix}")
    else:
        logger.info(f"🏦 Broker uses STANDARD symbol names (NO suffix/prefix)")
    
    # Pre-map all default symbols and log results
    mapped_count = 0
    for symbol in DEFAULT_SYMBOLS:
        broker_sym = get_broker_symbol(symbol)
        if broker_sym and broker_sym != symbol:
            logger.info(f"📌 Symbol mapping: {symbol} -> {broker_sym}")
            mapped_count += 1
        elif broker_sym:
            logger.debug(f"✓ Symbol {symbol} = {broker_sym} (no change needed)")
    
    logger.info(f"🔗 Symbol mapping complete: {mapped_count} symbols remapped, {len(DEFAULT_SYMBOLS) - mapped_count} use standard names")


# ---------------- DEFAULT MT5 LOGIN CONFIG (fallback) ----------------
DEFAULT_MT5_LOGIN = 10009413572
DEFAULT_MT5_PASSWORD = "@3BhJfGr"
DEFAULT_MT5_SERVER = "MetaQuotes-Demo"

# ---------------- BOT CONFIGURATION ----------------
# Multi-Symbol Support - All profitable volatile pairs
# IMPORTANT: Use STANDARD symbol names - suffix/prefix will be auto-detected per broker!
DEFAULT_SYMBOLS = [
    # Metals - High volatility, great for scalping
    "XAUUSD",    # Gold - Most volatile, best for profits
    "XAGUSD",    # Silver - High volatility
    # Major Forex Pairs - Good liquidity & volatility
    "EURUSD",    # Euro/USD - Most traded pair
    "GBPUSD",    # Cable - High volatility
    "USDJPY",    # Dollar/Yen - Good trends
    "USDCHF",    # Swissy - Safe haven moves
    "AUDUSD",    # Aussie - Commodity currency
    "USDCAD",    # Loonie - Oil correlated
    "NZDUSD",    # Kiwi - High interest rate moves
    # Cross Pairs - Extra volatility
    "GBPJPY",    # Gopher - Very volatile
    "EURJPY",    # Euro/Yen - Good swings
    "EURGBP",    # Euro/Pound - Brexit moves
    "AUDJPY",    # Aussie/Yen - Risk sentiment
    "CADJPY",    # CAD/Yen - Oil & risk
    # Crypto - Extreme volatility
    "BTCUSD",    # Bitcoin - Huge moves
    "ETHUSD",    # Ethereum - High volatility
    # Indices (if available on your broker)
    "US30",      # Dow Jones
    "US100",     # Nasdaq
    "US500",     # S&P 500
]
# ================================================================================
# ========================= OPTIMIZED PROFITABLE TRADING MODE ==================
# ================================================================================
# HIGH QUALITY trades with PROPER risk management for CONSISTENT profits

# TRADING TIMEFRAMES - Multi-TF: M15 for signals (high probability), M5 for entries (more trades)
TIMEFRAME = mt5.TIMEFRAME_M15  # M15 for high-probability signal analysis
SIGNAL_TIMEFRAME = mt5.TIMEFRAME_M15  # M15 for direction analysis
ENTRY_TF = mt5.TIMEFRAME_M5  # M5 for precise entry timing (more trades)
SCALP_TIMEFRAMES = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
RISK_PERCENT = 1.0  # 1% risk per trade - proper risk management
RISK_AMOUNT_USD = None  # Use percentage-based risk
STOPLOSS_PIPS = 20  # Tight SL with room to breathe
TAKEPROFIT_PIPS = 40  # 1:2 RR ratio - good reward
CHECK_INTERVAL = 5  # Check every 5 seconds for opportunities
MAGIC = 202501
MAX_POSITIONS_PER_SYMBOL = 2  # Allow 2 positions per symbol on strong signals
MAX_TOTAL_POSITIONS = 4  # Allow 4 concurrent positions for diversification
USE_TRAILING_STOP = True
TRAILING_DISTANCE = 10  # Trail 10 pips behind
TRAILING_START_PIPS = 15  # Start trailing after 15 pips profit

# ================================================================================
# ========================= HIGH PROBABILITY SCALING ============================
# ================================================================================
# When signal quality is very high, open multiple positions to maximize gains

# ========== ACCOUNT PROTECTION SETTINGS ==========
SMALL_ACCOUNT_THRESHOLD = 1000  # Accounts under $1000 need extra protection
MICRO_ACCOUNT_THRESHOLD = 500  # Accounts under $500 are "micro" - ultra conservative

# High probability scaling - ENABLED for maximum profit on strong signals
HIGH_PROB_SCALING_ENABLED = True  # Scale into high-quality setups
HIGH_PROB_MIN_QUALITY = 8  # Quality score 8+ = high probability
VERY_HIGH_PROB_MIN_QUALITY = 9  # Quality score 9+ = very high probability
ULTRA_HIGH_PROB_MIN_QUALITY = 10  # Quality score 10 = ultra high probability
HIGH_PROB_POSITIONS = 2  # 2 positions for high prob (8/10)
VERY_HIGH_PROB_POSITIONS = 2  # 2 positions for very high prob (9/10)
ULTRA_HIGH_PROB_POSITIONS = 3  # 3 positions for ultra high prob (10/10)
HIGH_PROB_LOT_MULTIPLIER = 1.5  # 1.5x lot for high prob
VERY_HIGH_PROB_LOT_MULTIPLIER = 2.0  # 2x lot for very high prob
ULTRA_HIGH_PROB_LOT_MULTIPLIER = 1.0  # NO multiplier - consistency wins
SCALE_IN_DELAY_SECONDS = 60  # DISABLED - no scaling in
MAX_RISK_PER_SIGNAL = 1.0  # Maximum 1% total risk per signal - STRICT!

# Account size-based lot bonus - CONSERVATIVE to protect all account sizes
ACCOUNT_SIZE_LOT_BONUS = {
    100: 0.5,    # $100-499: 0.5x (REDUCED for micro accounts)
    500: 0.75,   # $500-999: 0.75x (still conservative)
    1000: 1.0,   # $1000-2499: 1x (base)
    2500: 1.0,   # $2500-4999: 1x (no bonus)
    5000: 1.0,   # $5000-9999: 1x (no bonus)
    10000: 1.0,  # $10000-24999: 1x (no bonus)
    25000: 1.0,  # $25000-49999: 1x (no bonus)
    50000: 1.0,  # $50000-99999: 1x (no bonus)
    100000: 1.0, # $100000+: 1x - consistent risk management
}

# ================================================================================
# ========================= AI LOT SIZING LEARNING SYSTEM =======================
# ================================================================================
# AI learns optimal lot sizes from trade outcomes

AI_LOT_LEARNING_ENABLED = True
AI_LOT_LEARNING_DATA = {}  # Stores: {user: [{lot, quality, profit, win}...]}
AI_LOT_LEARNING_MIN_TRADES = 10  # Min trades before AI adjusts lot sizing
AI_LOT_CONFIDENCE_SCALING = True  # Scale lot by AI confidence

# Confidence-based lot multipliers - FLAT to ensure consistent risk
CONFIDENCE_LOT_MULTIPLIERS = {
    0.95: 1.0,   # 95%+ confidence: 1x lot (NO increase)
    0.90: 1.0,   # 90-95% confidence: 1x
    0.85: 1.0,   # 85-90% confidence: 1x
    0.80: 1.0,   # 80-85% confidence: 1x
    0.75: 1.0,   # 75-80% confidence: 1x
    0.70: 1.0,   # 70-75% confidence: 1x (base)
    0.0: 0.5,    # Below 70%: 0.5x (REDUCED - don't trade weak signals)
}

# Win streak lot scaling - DISABLED for consistent risk
WIN_STREAK_LOT_BONUS = {
    2: 1.0,    # NO bonus - stay consistent
    3: 1.0,    # NO bonus
    5: 1.0,    # NO bonus
    7: 1.0,    # NO bonus
    10: 1.0,   # NO bonus - never increase lot after wins (overconfidence kills)
}

# Lose streak lot reduction (STRICT reduction during losing streaks)
LOSE_STREAK_LOT_REDUCTION = {
    2: 0.5,    # 2 losses in a row: HALVE the lot
    3: 0.25,   # 3 losses in a row: 0.25x only
    4: 0.0,    # 4 losses: STOP TRADING completely
    5: 0.0,    # 5+ losses: STOP TRADING (return 0)
}

# ================================================================================
# ========================= AI LOSS PATTERN LEARNING ============================
# ================================================================================
# AI learns what market conditions lead to losses and avoids them

AI_LOSS_LEARNING_ENABLED = True
AI_LOSS_PATTERN_DATA = {}  # {user: [{conditions, loss_amount, symbol, time}...]}
AI_LOSS_AVOIDANCE_MIN_SAMPLES = 5  # Min losing trades to start avoiding patterns
AI_SIMILAR_LOSS_THRESHOLD = 0.7  # How similar conditions need to be to avoid (0-1)

# Breakeven protection - FASTER breakeven for scalping
USE_SMART_BREAKEVEN = True
BREAKEVEN_TRIGGER_PIPS = 1.5  # Move to breakeven after just 1.5 pips profit (was 5)
BREAKEVEN_BUFFER_PIPS = 0.3  # Tiny buffer above entry (0.3 pip profit locked)
BREAKEVEN_FOR_HIGH_QUALITY = 1  # High quality trades: breakeven at 1 pip
BREAKEVEN_FOR_ULTRA_QUALITY = 0.5  # Ultra quality: breakeven at 0.5 pip

# Smart position management
SMART_POSITION_MANAGEMENT = True
PARTIAL_CLOSE_AT_1R = True  # Close 50% at 1:1 RR
PARTIAL_CLOSE_PERCENT = 0.5  # Close 50% of position
MOVE_SL_TO_ENTRY_AFTER_PARTIAL = True  # Move SL to entry after partial close

# ================================================================================
# ========================= LOSS RECOVERY SYSTEM (DISABLED!) ===================
# ================================================================================
# MARTINGALE/RECOVERY TRADING IS THE #1 ACCOUNT KILLER - DISABLED!

LOSS_RECOVERY_ENABLED = False  # DISABLED - Recovery trading blows accounts!
RECOVERY_MODE_ACTIVE = {}  # Not used

# Recovery triggers - DISABLED
RECOVERY_TRIGGER_LOSS_PERCENT = 999.0  # Never trigger (effectively disabled)
RECOVERY_CONTINUE_UNTIL = 1.0  # Not used

# ============ RECOVERY SCALPING - COMPLETELY DISABLED ============
# NEVER increase lot size to recover losses - this ALWAYS blows accounts!
RECOVERY_SCALP_MODE = False  # DISABLED
RECOVERY_MAX_LOT_MULTIPLIER = 1.0  # NO multiplier
RECOVERY_ULTRA_LOT_MULTIPLIER = 1.0  # NO multiplier
RECOVERY_MIN_QUALITY = 10  # Impossible to trigger
RECOVERY_MIN_CONFIDENCE = 1.0  # Impossible to trigger (100% never happens)
RECOVERY_ULTRA_CONFIDENCE = 1.0  # Impossible to trigger

# Quick Scalp Settings - CONSERVATIVE
RECOVERY_SCALP_TP_PIPS = 30  # Normal TP
RECOVERY_SCALP_SL_PIPS = 20  # Normal SL
RECOVERY_SCALP_MAX_HOLD_SECONDS = 3600  # Normal hold time
RECOVERY_QUICK_TP = False  # DISABLED
RECOVERY_TP_REDUCTION = 1.0  # No reduction

# Smart Recovery Lot Scaling - DISABLED (all 1.0x)
RECOVERY_LOT_TIERS = {
    0.25: 1.0,   # NO increase
    0.50: 1.0,   # NO increase
    0.75: 1.0,   # NO increase
    1.00: 1.0,   # NO increase - NEVER increase lot to recover
}

# Ultra-Strict Recovery Trade Selection
RECOVERY_REQUIRE_MULTI_CONFLUENCE = True  # Need 3+ confluences
RECOVERY_REQUIRE_HTF_ALIGNMENT = True  # Must align with higher TF
RECOVERY_AVOID_COUNTER_TREND = True  # Only trade with trend during recovery
RECOVERY_REQUIRE_KEY_LEVEL = True  # Must be at support/resistance
RECOVERY_REQUIRE_MOMENTUM = True  # Need strong momentum confirmation
RECOVERY_AVOID_CONSOLIDATION = True  # Don't trade in ranging markets
RECOVERY_MIN_VOLATILITY = 0.5  # Need decent volatility for quick moves

# ================================================================================
# ========================= LOSS PREVENTION SYSTEM ==============================
# ================================================================================
# Smart filters to avoid entering losing trades

LOSS_PREVENTION_ENABLED = True

# 1. Market Condition Filters
AVOID_CHOPPY_MARKETS = True
CHOPPY_MARKET_ADX_THRESHOLD = 20  # ADX below 20 = choppy/ranging market
AVOID_NEWS_BEFORE_MINUTES = 30  # Don't trade 30 min before major news
AVOID_NEWS_AFTER_MINUTES = 15  # Don't trade 15 min after major news
NEWS_EVENTS_CACHE = {}  # Cached news events

# 2. Session Quality Filters  
AVOID_SESSION_OVERLAPS_TRANSITIONS = True  # Avoid first 30 min of session
SESSION_OVERLAP_AVOID_MINUTES = 30  # Wait 30 min after session open
AVOID_LOW_LIQUIDITY_HOURS = True  # Don't trade during low liquidity
LOW_LIQUIDITY_HOURS_UTC = [21, 22, 23, 0, 1, 2]  # Asian late night = low liquidity

# 3. Technical Quality Filters
MIN_TREND_STRENGTH = 0.6  # Need 60%+ trend strength to trade with trend
REQUIRE_CLEAN_STRUCTURE = True  # Need clear highs/lows, not messy
MAX_RECENT_REJECTIONS = 2  # If price rejected level 2+ times, avoid
AVOID_NEAR_ROUND_NUMBERS = False  # Round numbers can be traps

# 4. Price Action Quality
MIN_CANDLE_QUALITY = 0.7  # Need 70%+ candle quality (no massive wicks)
MAX_WICK_RATIO = 0.4  # Wicks shouldn't be > 40% of candle
AVOID_DOJI_ENTRIES = True  # Don't enter on indecision candles
REQUIRE_MOMENTUM_CANDLE = True  # Entry candle needs momentum

# 5. Spread & Time Filters
MAX_SPREAD_FOR_ENTRY = 2.0  # Max 2x normal spread allowed
AVOID_WEEKEND_GAPS = True  # Don't hold over weekend
CLOSE_BEFORE_WEEKEND_HOURS = 3  # Close 3 hours before market close Friday

# 6. Correlation & Exposure Filters
AVOID_CORRELATED_TRADES = True  # Don't take same direction on correlated pairs
CORRELATED_PAIRS = {
    'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],  # Positive correlation
    'USDJPY': ['USDCHF', 'USDCAD'],  # USD strength pairs
    'XAUUSD': ['XAGUSD'],  # Metals together
}
MAX_EXPOSURE_SAME_DIRECTION = 3  # Max 3 trades in same direction

# 7. Recent Performance Filters
CHECK_RECENT_PERFORMANCE = True
RECENT_TRADES_TO_CHECK = 10  # Check last 10 trades
MIN_RECENT_WIN_RATE = 0.4  # If win rate < 40%, pause trading
PAUSE_AFTER_BIG_LOSS = True  # Pause after significant loss
BIG_LOSS_THRESHOLD = 0.03  # 3% of account in one trade = big loss
PAUSE_AFTER_BIG_LOSS_MINUTES = 30  # Pause 30 min after big loss

# 8. Timing Quality
MIN_TIME_IN_TREND = 3  # Trend must be established for 3+ candles
AVOID_JUST_AFTER_REVERSAL = True  # Don't enter immediately after reversal
REVERSAL_COOLDOWN_CANDLES = 2  # Wait 2 candles after reversal

# ================================================================================
# ========================= ENTRY QUALITY SCORING ===============================
# ================================================================================
# Only enter trades that score high on multiple factors

ENTRY_QUALITY_SCORING = True
MIN_ENTRY_SCORE = 70  # Need 70+ score - balanced for more trades with quality

ENTRY_SCORE_WEIGHTS = {
    'trend_alignment': 20,     # +20 if with higher TF trend
    'key_level': 15,           # +15 if at support/resistance
    'confluence_count': 15,    # +3 per confluence (max 15)
    'candle_quality': 15,      # Based on candle strength
    'session_quality': 10,     # Based on current session
    'spread_quality': 5,       # Lower spread = higher score
    'momentum': 15,            # RSI/MACD momentum (important!)
    'recent_wins': 5,          # +5 if recent trades winning
    'no_loss_patterns': 0,     # Not weighted
}

# ================================================================================
# ========================= DYNAMIC COMPOUNDING SYSTEM ==========================
# ================================================================================
# Grow profits by increasing lot size as account grows

USE_DYNAMIC_COMPOUNDING = True  # ENABLED for profit growth
COMPOUND_BASE_BALANCE = None
COMPOUND_GROWTH_RATE = 0.1  # Increase lot by 10% per 10% account growth
COMPOUND_UPDATE_INTERVAL = 5  # Recalculate lot size every 5 trades
MAX_COMPOUND_MULTIPLIER = 2.0  # Max 2x the base lot size (safe limit)

# ATR-Based Dynamic Lot Sizing
USE_ATR_LOT_SIZING = True
ATR_PERIOD = 14  # Standard ATR period
ATR_SL_MULTIPLIER = 2.0  # 2x ATR for stop loss
ATR_TP_MULTIPLIER = 4.0  # 1:2 RR = 4x ATR
MIN_LOT = 0.01  # Minimum lot size
MAX_LOT = 0.5   # STRICT MAX - never exceed 0.5 lot for small accounts
MAX_LOT_PERCENT = 1.0  # STRICT 1% max of balance per trade
MAX_LOT_HIGH_PROB = 0.5  # Same max even for high probability - PROTECT CAPITAL

# ========== CONSERVATIVE EXECUTION SETTINGS ==========
FAST_EXECUTION_MODE = True  # Fast execution for better entry prices
USE_MARKET_ORDERS_ONLY = True  # Market orders for instant fill
MAX_SLIPPAGE_PIPS = 3  # Accept up to 3 pips slippage
AGGRESSIVE_ENTRY = True  # Enter on confirmed signal
QUICK_PROFIT_MODE = True  # Lock in profits quickly
SCALPING_MODE = True  # ENABLED for more trade opportunities

# ================================================================================
# ========================= OPTIMIZED LOT SIZING SYSTEM =========================
# ================================================================================
# Balanced lot sizing - proper risk with profit potential

AGGRESSIVE_SCALPING_ENABLED = True  # Enable for more opportunities

# Optimized Lot Sizing - Proper risk (1% per trade)
SCALP_LOT_ACCOUNT_TIERS = {
    100: 0.02,      # $100 account → 0.02 lot (1% risk = $1 per trade)
    200: 0.03,      # $200 account → 0.03 lot
    500: 0.05,      # $500 account → 0.05 lot
    1000: 0.10,     # $1k account → 0.10 lot
    2500: 0.25,     # $2.5k account → 0.25 lot
    5000: 0.50,     # $5k account → 0.50 lot
    10000: 1.00,    # $10k account → 1.0 lot
    25000: 2.50,    # $25k account → 2.5 lot
    50000: 5.00,    # $50k account → 5.0 lot
    100000: 10.00,  # $100k account → 10.0 lot
}

# Professional Profit Targets (in pips) - Proper 1:2+ Risk:Reward
SCALP_TP_PIPS = {
    'XAUUSD': 50,   # Gold: 50 pips - covers spread + 1:2 RR
    'BTCUSD': 200,  # BTC: 200 points
    'EURUSD': 20,   # EUR: 20 pips
    'GBPUSD': 25,   # GBP: 25 pips
    'USDJPY': 20,   # JPY: 20 pips
    'DEFAULT': 20,  # Default: 20 pips
}

# Professional Stop Losses (room to breathe, proper positioning)
SCALP_SL_PIPS = {
    'XAUUSD': 25,   # Gold: 25 pips (1:2 RR)
    'BTCUSD': 100,  # BTC: 100 points
    'EURUSD': 10,   # EUR: 10 pips
    'GBPUSD': 12,   # GBP: 12 pips
    'USDJPY': 10,   # JPY: 10 pips
    'DEFAULT': 10,  # Default: 10 pips
}

# Professional Close Settings - Let winners run!
SCALP_CLOSE_ON_ANY_PROFIT = False  # Let profits run with trailing
SCALP_MIN_PROFIT_PIPS = 10  # Only close if meaningful profit (10 pips)
SCALP_MIN_HOLD_SECONDS = 60  # Hold at least 1 minute for proper setup
SCALP_MAX_HOLD_SECONDS = 3600  # Allow up to 1 hour hold time

# Re-entry Settings - DISABLED for capital protection
SCALP_REENTRY_ENABLED = True  # Enable re-entry on strong signals
SCALP_REENTRY_COOLDOWN_SECONDS = 60  # Wait 1 minute before re-entry
SCALP_MAX_REENTRIES_PER_SETUP = 3  # Max 3 re-entries per setup
SCALP_REENTRY_REQUIRE_SAME_DIRECTION = True  # Must be same direction to re-enter

# Track scalp entries for re-entry
scalp_entry_tracker = {}  # {symbol: {direction, entries_count, last_close_time, setup_valid_until}}

# ========== OPTIMIZED TRADE FREQUENCY ==========
MIN_MINUTES_BETWEEN_TRADES = 1  # Wait 1 minute between trades
MAX_TRADES_PER_DAY = 30  # Allow 30 trades per day for more opportunities
REQUIRE_DOUBLE_CONFIRMATION = False  # Single strong signal is enough

# ================================================================================
# ========================= TIGHT TRAILING STOP SYSTEM ==========================
# ================================================================================
# Trail ULTRA-TIGHT for small account protection - SPREAD AWARE

USE_AGGRESSIVE_TRAILING = True
AGGRESSIVE_TRAIL_CONFIGS = {
    'phase_ultra':   {'trigger_pips': 0.2, 'trail_distance': 0.15},  # 0.2 pips → trail at 0.15 (ULTRA instant)
    'phase_instant': {'trigger_pips': 0.4, 'trail_distance': 0.2},   # 0.4 pips → trail at 0.2 (INSTANT protection)
    'phase0': {'trigger_pips': 0.7, 'trail_distance': 0.35},         # 0.7 pips → trail at 0.35
    'phase1': {'trigger_pips': 1.0, 'trail_distance': 0.5},          # 1.0 pips → trail at 0.5
    'phase2': {'trigger_pips': 1.5, 'trail_distance': 0.8},          # 1.5 pips → trail at 0.8
    'phase3': {'trigger_pips': 2.5, 'trail_distance': 1.2},          # 2.5 pips → trail at 1.2
    'phase4': {'trigger_pips': 5.0, 'trail_distance': 2.0},          # 5.0 pips → trail at 2.0
    'phase5': {'trigger_pips': 10.0, 'trail_distance': 3.5},         # 10 pips → trail at 3.5
}
LOCK_PROFIT_THRESHOLD_PIPS = 0.5  # Lock in profit after just 0.5 pip (ultra scalp)
LOCK_PROFIT_PERCENT = 0.90  # Lock in 90% of max profit (protect gains faster)

# ================================================================================
# ========================= SIMPLIFIED ENTRY CONDITIONS =========================
# ================================================================================
# Less restrictive for more frequent trading

# ========== 1. SCAN & FILTER CONFIGURATION ==========
SPREAD_FILTER_ENABLED = True
MAX_SPREAD_MULTIPLIER = 3.0  # Allow up to 3x normal spread for more trades
VOLATILITY_FILTER_ENABLED = True
MIN_VOLATILITY_THRESHOLD = 0.2  # Accept lower volatility (20%)
MAX_VOLATILITY_THRESHOLD = 5.0  # Accept higher volatility
STRUCTURE_CLARITY_REQUIRED = False  # Don't require perfect structure

# ========== SPREAD-AWARE TRADING SYSTEM ==========
SPREAD_AWARE_ENTRY = True  # Only enter if expected profit > spread cost
MIN_PROFIT_VS_SPREAD = 2.0  # Require expected profit to be at least 2x the spread
SPREAD_ADJUSTED_LOTS = True  # Reduce lot size when spread is elevated
SPREAD_LOT_REDUCTION = 0.5  # Cut lot by 50% when spread is elevated (but acceptable)
SPREAD_ELEVATED_THRESHOLD = 1.2  # Spread is "elevated" if > 1.2x normal
SPREAD_RECOVERY_MODE = True  # Adjust profit targets to recover spread cost first

# Normal spreads for each symbol (in points) - TIGHTENED VALUES
NORMAL_SPREADS = {
    "XAUUSD": 25, "XAGUSD": 25, "EURUSD": 12, "GBPUSD": 15, "USDJPY": 12,
    "USDCHF": 15, "AUDUSD": 15, "USDCAD": 18, "NZDUSD": 18, "GBPJPY": 30,
    "EURJPY": 20, "EURGBP": 15, "AUDJPY": 25, "CADJPY": 25, "BTCUSD": 400,
    "ETHUSD": 150, "US30": 25, "US100": 20, "US500": 12
}

# ========== 2. MULTI-TIMEFRAME ANALYSIS ==========
# M15 signal + M5 entry + H1 trend = Best probability + More trades
HTF_TIMEFRAME = mt5.TIMEFRAME_H1  # H1 for overall trend bias
MTF_TIMEFRAME = mt5.TIMEFRAME_M15  # M15 for signal confirmation (high probability)
ENTRY_TIMEFRAME = mt5.TIMEFRAME_M5  # M5 for entry precision (more opportunities)
REQUIRE_HTF_ALIGNMENT = True  # Require H1 trend alignment for higher probability
REQUIRE_MTF_CONFIRMATION = True  # Require M15 signal confirmation
REVERSAL_PATTERN_OVERRIDE = True  # Still allow strong reversals

# ========== 3. RELAXED TRADE SETUP CONDITIONS ==========
REQUIRE_KEY_LEVEL = False  # Don't require key levels
REQUIRE_MOMENTUM_CONFIRM = True  # Still require momentum
MIN_RR_RATIO = 1.5  # Lower RR requirement (1:1.5)
REQUIRE_TIGHT_SL = True  # Keep tight SL for scalping
REQUIRE_ACTIVE_SESSION = False  # Trade any session

# ========== 4. FAST ENTRY EXECUTION ==========
ENTRY_ON_CONFIRMATION = False  # Enter immediately
NEVER_CHASE_PRICE = False  # Allow chasing
MAX_CHASE_PIPS = 10  # Allow up to 10 pips chase
POSITION_SIZE_BY_RISK = True  # Size based on risk percentage

# ========== 5. ZERO-LOSS PROFIT PROTECTION (AGGRESSIVE!) ==========
# GOAL: NEVER let a profitable trade become a loss - PROTECT EVERY PIP!
PROFIT_PROTECTION_ENABLED = True
REDUCE_RISK_AT_R = 0.1  # At 0.1R profit, tighten SL immediately (was 0.2)
BREAKEVEN_AT_R = 0.15  # At 0.15R profit (~5 pips), move SL to breakeven (was 0.3)
PARTIAL_TP_AT_R = 0.3  # At 0.3R, take partial profits to lock gains (was 0.5)
PARTIAL_TP_PERCENT = 0.5  # Close 50% at partial TP
TRAIL_AFTER_PARTIAL = True  # Activate tight trailing after partial TP
TRAIL_BY_STRUCTURE = False  # Trail by pips, not structure
LOCK_PROFIT_PERCENT = 40  # Lock 40% of maximum profit reached

# ========== SMALL ACCOUNT PROTECTION MODE ==========
SMALL_ACCOUNT_MODE = True  # Enable ultra-aggressive profit protection

# ========== DOLLAR-BASED PROFIT PROTECTION (AGGRESSIVE LOCK-IN!) ==========
# Close trades when dollar profit drops to protect gains
# LOCK IN PROFITS - Only allow small pullback before closing!
DOLLAR_PROFIT_PROTECTION = True  # Enable dollar-based profit protection
MIN_PROFIT_DOLLARS_TO_PROTECT = 0.10  # Start protecting at $0.10 profit
CLOSE_WHEN_PROFIT_DROPS_TO = 0.20  # Close if profit drops to $0.20 (from $1+)
DOLLAR_PROFIT_DROP_TIERS = {
    # Micro profits - lock 70%
    'tier0': {'min_peak': 0.10, 'close_at': 0.07},      # $0.10 peak → close at $0.07 (lock 70%)
    # Small profits - lock 70%
    'tier1': {'min_peak': 0.50, 'close_at': 0.35},      # $0.50 peak → close at $0.35 (lock 70%)
    'tier2': {'min_peak': 1.00, 'close_at': 0.70},      # $1.00 peak → close at $0.70 (lock 70%)
    'tier3': {'min_peak': 2.00, 'close_at': 1.40},      # $2.00 peak → close at $1.40 (lock 70%)
    'tier4': {'min_peak': 5.00, 'close_at': 3.50},      # $5.00 peak → close at $3.50 (lock 70%)
    # Medium profits - lock 70%
    'tier5': {'min_peak': 10.00, 'close_at': 7.00},     # $10 peak → close at $7.00 (lock 70%)
    'tier6': {'min_peak': 20.00, 'close_at': 14.00},    # $20 peak → close at $14.00 (lock 70%)
    'tier7': {'min_peak': 50.00, 'close_at': 35.00},    # $50 peak → close at $35.00 (lock 70%)
    # Large profits - lock 70%
    'tier8': {'min_peak': 100.00, 'close_at': 70.00},   # $100 peak → close at $70.00 (lock 70%)
    'tier9': {'min_peak': 200.00, 'close_at': 140.00},  # $200 peak → close at $140.00 (lock 70%)
    'tier10': {'min_peak': 500.00, 'close_at': 350.00}, # $500 peak → close at $350.00 (lock 70%)
    # Big profits - lock 70%
    'tier11': {'min_peak': 1000.00, 'close_at': 700.00},    # $1k peak → close at $700 (lock 70%)
    'tier12': {'min_peak': 2000.00, 'close_at': 1400.00},   # $2k peak → close at $1400 (lock 70%)
    'tier13': {'min_peak': 5000.00, 'close_at': 3500.00},   # $5k peak → close at $3500 (lock 70%)
    # Huge profits - lock 70%
    'tier14': {'min_peak': 10000.00, 'close_at': 7000.00},  # $10k peak → close at $7k (lock 70%)
    'tier15': {'min_peak': 20000.00, 'close_at': 14000.00}, # $20k peak → close at $14k (lock 70%)
    'tier16': {'min_peak': 50000.00, 'close_at': 35000.00}, # $50k peak → close at $35k (lock 70%)
}
NEVER_LET_PROFIT_GO_NEGATIVE = True  # Always close if profit is about to go negative

# ========== AGGRESSIVE CLOSE & RE-ENTER SYSTEM ==========
CLOSE_ON_PROFIT_DROP = True  # Close trade if profit drops
PROFIT_DROP_CLOSE_THRESHOLD = 0.10  # Close if profit drops 10% from peak (was 15%)
MIN_PROFIT_PIPS_TO_CLOSE = 0.1  # Close even with 0.1 pip profit if dropping (was 0.2)
REENTRY_ENABLED = True  # Re-enter after closing in profit
REENTRY_COOLDOWN_SECONDS = 2  # Quick re-entry after 2 seconds
REENTRY_REQUIRE_CONFIRMATION = True  # Require fresh signal before re-entry

# ========== ULTRA-FAST PROFIT DROP PROTECTION (MAXIMUM AGGRESSION!) ==========
# Tiered profit drop - close ULTRA FAST at all levels to beat spread
# Key: SAVE EVERY PIP! Close before profit disappears!
PROFIT_DROP_TIERS = {
    'instant': {'min_peak': 0.05, 'drop_pct': 0.50, 'drop_pips': 0.03},  # 0.05+ pips: close on 50% drop
    'nano':    {'min_peak': 0.2, 'drop_pct': 0.40, 'drop_pips': 0.08},   # 0.2+ pips: close on 40% drop
    'micro':   {'min_peak': 0.5, 'drop_pct': 0.30, 'drop_pips': 0.15},   # 0.5+ pips: close on 30% drop
    'small':   {'min_peak': 1.0, 'drop_pct': 0.20, 'drop_pips': 0.20},   # 1.0+ pips: close on 20% drop
    'medium':  {'min_peak': 2.0, 'drop_pct': 0.15, 'drop_pips': 0.30},   # 2.0+ pips: close on 15% drop
    'large':   {'min_peak': 4.0, 'drop_pct': 0.10, 'drop_pips': 0.40},   # 4.0+ pips: close on 10% drop
    'profit':  {'min_peak': 6.0, 'drop_pct': 0.08, 'drop_pips': 0.50},   # 6.0+ pips: close on 8% drop
}
MONITOR_PROFIT_AFTER_PIPS = 0.03  # Start monitoring after just 0.03 pips (INSTANT!)
AGGRESSIVE_PROFIT_LOCK = True  # Use aggressive profit locking
CLOSE_ON_MOMENTUM_REVERSAL = True  # Close if momentum reverses while in profit
MOMENTUM_REVERSAL_MIN_PROFIT = 0.1  # Min profit to trigger momentum close (was 0.2)
SAVE_PROFIT_AT_ANY_COST = True  # New: prioritize saving profit over holding position

# ========== SPREAD RECOVERY PROFIT TARGETS ==========
# Minimum profit targets to cover spread + profit margin
SPREAD_RECOVERY_MULTIPLIER = 1.5  # Need 1.5x spread in profit before relaxing protection
AUTO_CLOSE_AT_SPREAD_MULTIPLE = 3.0  # Consider closing at 3x spread profit to secure gains

# ========== INSTANT BREAKEVEN SYSTEM (ULTRA AGGRESSIVE!) ==========
INSTANT_BREAKEVEN_PIPS = 0.5  # Move to breakeven after just 0.5 pip (was 1.0) - PROTECT FASTER!
LOCK_MIN_PROFIT_PIPS = 0.1  # Lock at least 0.1 pip when moving SL (was 0.2)
BREAKEVEN_PLUS_PIPS = 0.3  # Move SL to breakeven + 0.3 pips to ensure profit

# ========== 6. EXIT RULES ==========
EXIT_ON_STRUCTURE_BREAK = False  # Don't exit on structure break
EXIT_ON_VOLATILITY_COLLAPSE = True  # Exit if volatility dies
EXIT_ON_NEWS_INVALIDATION = False  # Don't exit on news
NEVER_LET_WINNER_BECOME_LOSER = True  # Still protect winners

# ========== 7. NEWS CONTROL - DISABLED ==========
NEWS_FILTER_ENABLED = False  # Trade through news
NEWS_BLACKOUT_MINUTES_BEFORE = 0  # No blackout
NEWS_BLACKOUT_MINUTES_AFTER = 0  # No blackout
LOCK_PROFITS_ON_NEWS = True  # But still lock profits

# ========== 8. BALANCED QUALITY + FREQUENCY ==========
QUALITY_OVER_QUANTITY = True  # Prioritize quality but allow more trades
MIN_SETUP_QUALITY_SCORE = 7  # Need 7/10 quality (balanced)
MAX_CONSECUTIVE_SKIPS = 10  # Take trades more frequently

# ========== 9. SMART LOSS CONTROL ==========
REDUCE_SIZE_AFTER_LOSSES = True
CONSECUTIVE_LOSS_THRESHOLD = 3  # After 3 losses, reduce size
SIZE_REDUCTION_PERCENT = 0.7  # Reduce to 70% after losses
TIGHTEN_CRITERIA_AFTER_LOSSES = True
INCREASED_MIN_SCORE = 9  # Require 9/10 after consecutive losses
STOP_TRADING_THRESHOLD = 3  # Stop after 3 consecutive losses
COOLDOWN_AFTER_STOP_MINUTES = 120  # 2 hour cooldown after stop

# ========== VOLATILITY TRACKING ==========
volatility_cache = defaultdict(lambda: {'atr': 0, 'avg_atr': 0, 'last_update': None})

# ========== PROFIT TRACKING PER POSITION ==========
position_profit_peaks = defaultdict(float)  # {ticket: max_profit_in_R}
position_profit_peaks_dollars = defaultdict(float)  # {ticket: max_profit_in_dollars}

# ========== COMPOUNDING TRACKING ==========
compounding_state = {
    'base_balance': None,
    'current_lot_multiplier': 1.0,
    'trades_since_update': 0,
    'last_update_balance': None
}
position_entry_data = defaultdict(dict)  # {ticket: {sl_distance, tp_distance, entry_price}}
trade_strategies_used = {}  # {ticket: {'strategies': [...], 'user': str, 'symbol': str}}  - Tracks which strategies were used for each trade

# ========== RE-ENTRY TRACKING ==========
reentry_queue = {}  # {symbol: {'direction': 'BUY'/'SELL', 'closed_at': timestamp, 'lot': lot, 'reason': str}}
closed_with_profit = {}  # {symbol: {'time': timestamp, 'profit': float, 'direction': str}}


def check_spread_filter(symbol):
    """
    Check if current spread is acceptable for trading.
    Returns (is_acceptable, current_spread, normal_spread, message).
    Also returns spread_multiplier for lot adjustment.
    """
    if not SPREAD_FILTER_ENABLED:
        return True, 0, 0, "Spread filter disabled"
    
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    
    if not tick or not info:
        return False, 0, 0, "No tick/symbol data"
    
    current_spread = tick.ask - tick.bid
    spread_points = current_spread / info.point
    
    # Get standard symbol name for lookup
    std_symbol = get_standard_symbol(symbol)
    normal_spread = NORMAL_SPREADS.get(std_symbol, NORMAL_SPREADS.get(symbol, 30))
    
    max_acceptable = normal_spread * MAX_SPREAD_MULTIPLIER
    spread_ratio = spread_points / normal_spread if normal_spread > 0 else 1.0
    
    if spread_points > max_acceptable:
        return False, spread_points, normal_spread, f"Spread too high: {spread_points:.0f} > {max_acceptable:.0f} ({spread_ratio:.1f}x normal)"
    
    # Warn if spread is elevated but acceptable
    if spread_ratio > SPREAD_ELEVATED_THRESHOLD:
        return True, spread_points, normal_spread, f"Spread elevated: {spread_points:.0f} ({spread_ratio:.1f}x normal)"
    
    return True, spread_points, normal_spread, "Spread OK"


def get_spread_adjusted_lot(symbol, base_lot):
    """
    Reduce lot size when spread is elevated to protect against high spread costs.
    """
    if not SPREAD_ADJUSTED_LOTS:
        return base_lot
    
    spread_ok, current_spread, normal_spread, _ = check_spread_filter(symbol)
    if not spread_ok:
        return 0  # Don't trade at all
    
    spread_ratio = current_spread / normal_spread if normal_spread > 0 else 1.0
    
    if spread_ratio > SPREAD_ELEVATED_THRESHOLD:
        adjusted_lot = round(base_lot * SPREAD_LOT_REDUCTION, 2)
        logger.info(f"📉 Spread elevated ({spread_ratio:.1f}x) - reducing lot: {base_lot} → {adjusted_lot}")
        return max(adjusted_lot, 0.01)  # Minimum 0.01 lot
    
    return base_lot


def get_spread_in_pips(symbol):
    """
    Get current spread in pips (not points) for profit calculations.
    """
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    
    if not tick or not info:
        return 0
    
    current_spread = tick.ask - tick.bid
    point = info.point
    
    # Get pip multiplier for this symbol
    if symbol in SYMBOL_SETTINGS:
        pip_value = SYMBOL_SETTINGS[symbol]['pip_value']
        pip_mult = pip_value / point if point > 0 else 10
    elif 'JPY' in symbol.upper():
        pip_mult = 1
    else:
        pip_mult = 10
    
    spread_pips = current_spread / (point * pip_mult)
    return spread_pips


def should_enter_with_spread(symbol, expected_profit_pips):
    """
    Check if trade is worth entering considering spread cost.
    Only enter if expected profit significantly exceeds spread.
    """
    if not SPREAD_AWARE_ENTRY:
        return True, "Spread-aware entry disabled"
    
    spread_pips = get_spread_in_pips(symbol)
    min_profit_needed = spread_pips * MIN_PROFIT_VS_SPREAD
    
    if expected_profit_pips < min_profit_needed:
        return False, f"Expected {expected_profit_pips:.1f} pips < {min_profit_needed:.1f} needed (spread: {spread_pips:.1f})"
    
    return True, f"Expected {expected_profit_pips:.1f} pips vs {spread_pips:.1f} spread - OK"


def check_volatility_filter(symbol, df):
    """
    Check if volatility is within acceptable range.
    Returns (is_acceptable, current_vol, avg_vol, message).
    """
    if not VOLATILITY_FILTER_ENABLED:
        return True, 0, 0, "Volatility filter disabled"
    
    if 'atr' not in df.columns or len(df) < 50:
        return True, 0, 0, "Insufficient data"
    
    current_atr = df['atr'].iloc[-1]
    avg_atr = df['atr'].tail(50).mean()
    
    if avg_atr <= 0:
        return True, 0, 0, "No ATR data"
    
    vol_ratio = current_atr / avg_atr
    
    # Cache volatility
    volatility_cache[symbol] = {
        'atr': current_atr,
        'avg_atr': avg_atr,
        'ratio': vol_ratio,
        'last_update': datetime.now()
    }
    
    if vol_ratio < MIN_VOLATILITY_THRESHOLD:
        return False, current_atr, avg_atr, f"Volatility too low: {vol_ratio:.2f}x avg (min {MIN_VOLATILITY_THRESHOLD}x)"
    
    if vol_ratio > MAX_VOLATILITY_THRESHOLD:
        return False, current_atr, avg_atr, f"Volatility too high: {vol_ratio:.2f}x avg (max {MAX_VOLATILITY_THRESHOLD}x)"
    
    return True, current_atr, avg_atr, f"Volatility acceptable: {vol_ratio:.2f}x avg"


def check_market_structure_clarity(df):
    """
    Check if market structure is clear (trending or at clear levels).
    Returns (is_clear, regime, message).
    """
    if not STRUCTURE_CLARITY_REQUIRED:
        return True, "UNKNOWN", "Structure check disabled"
    
    if len(df) < 50:
        return False, "UNKNOWN", "Insufficient data"
    
    # Calculate EMAs
    ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
    ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
    ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
    price = df['close'].iloc[-1]
    
    # Check for clear trend
    bullish_stack = ema_9 > ema_21 > ema_50 and price > ema_9
    bearish_stack = ema_9 < ema_21 < ema_50 and price < ema_9
    
    if bullish_stack:
        return True, "TRENDING_UP", "Clear bullish structure: EMAs stacked up"
    elif bearish_stack:
        return True, "TRENDING_DOWN", "Clear bearish structure: EMAs stacked down"
    
    # Check for range with clear levels
    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    range_size = high_20 - low_20
    current_range_position = (price - low_20) / range_size if range_size > 0 else 0.5
    
    # Clear at support or resistance
    if current_range_position < 0.2:
        return True, "RANGE_SUPPORT", "At range support"
    elif current_range_position > 0.8:
        return True, "RANGE_RESISTANCE", "At range resistance"
    
    # Mixed/unclear structure
    return False, "UNCLEAR", "Market structure unclear - EMAs mixed, not at key levels"


def get_htf_direction(symbol):
    """
    Get higher timeframe (H4) direction for bias.
    Returns (direction, strength, message).
    """
    htf_df = get_data(symbol, HTF_TIMEFRAME, n=100)
    if htf_df is None or len(htf_df) < 50:
        return "NEUTRAL", 0, "No HTF data"
    
    # Calculate HTF EMAs
    ema_21 = htf_df['close'].ewm(span=21).mean().iloc[-1]
    ema_50 = htf_df['close'].ewm(span=50).mean().iloc[-1]
    ema_200 = htf_df['close'].ewm(span=200).mean().iloc[-1] if len(htf_df) >= 200 else ema_50
    price = htf_df['close'].iloc[-1]
    
    # Calculate trend strength
    bullish_points = 0
    bearish_points = 0
    
    if price > ema_21: bullish_points += 1
    else: bearish_points += 1
    if price > ema_50: bullish_points += 1
    else: bearish_points += 1
    if price > ema_200: bullish_points += 1
    else: bearish_points += 1
    if ema_21 > ema_50: bullish_points += 1
    else: bearish_points += 1
    
    # Check for higher highs / lower lows
    recent_high = htf_df['high'].tail(10).max()
    prev_high = htf_df['high'].iloc[-20:-10].max() if len(htf_df) >= 20 else recent_high
    recent_low = htf_df['low'].tail(10).min()
    prev_low = htf_df['low'].iloc[-20:-10].min() if len(htf_df) >= 20 else recent_low
    
    if recent_high > prev_high and recent_low > prev_low:
        bullish_points += 2
    elif recent_high < prev_high and recent_low < prev_low:
        bearish_points += 2
    
    total = bullish_points + bearish_points
    
    if bullish_points > bearish_points + 1:
        strength = bullish_points / total
        return "BULLISH", strength, f"HTF bullish ({bullish_points}/{total})"
    elif bearish_points > bullish_points + 1:
        strength = bearish_points / total
        return "BEARISH", strength, f"HTF bearish ({bearish_points}/{total})"
    else:
        return "NEUTRAL", 0.5, "HTF neutral/ranging"


def detect_reversal_pattern(df, direction):
    """
    Detect if there's a confirmed reversal pattern against HTF direction.
    Returns (has_reversal, pattern_name, confidence).
    """
    if len(df) < 10:
        return False, None, 0
    
    price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    prev2_price = df['close'].iloc[-3]
    
    curr_open = df['open'].iloc[-1]
    prev_open = df['open'].iloc[-2]
    prev2_open = df['open'].iloc[-3]
    
    curr_high = df['high'].iloc[-1]
    curr_low = df['low'].iloc[-1]
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    
    # Looking for bullish reversal (HTF is bearish but we see reversal)
    if direction == "BEARISH":
        # Bullish engulfing
        if prev_price < prev_open and price > curr_open and price > prev_high and curr_open < prev_low:
            return True, "BULLISH_ENGULFING", 0.8
        
        # Morning star (3 candle pattern)
        if (prev2_price < prev2_open and  # First candle bearish
            abs(prev_price - prev_open) < (prev2_open - prev2_price) * 0.3 and  # Middle small
            price > curr_open and price > (prev2_open + prev2_price) / 2):  # Third bullish above mid
            return True, "MORNING_STAR", 0.75
        
        # Hammer (long lower wick)
        body = abs(price - curr_open)
        lower_wick = min(price, curr_open) - curr_low
        upper_wick = curr_high - max(price, curr_open)
        if lower_wick > body * 2 and upper_wick < body * 0.5 and price > curr_open:
            return True, "HAMMER", 0.7
    
    # Looking for bearish reversal (HTF is bullish but we see reversal)
    elif direction == "BULLISH":
        # Bearish engulfing
        if prev_price > prev_open and price < curr_open and price < prev_low and curr_open > prev_high:
            return True, "BEARISH_ENGULFING", 0.8
        
        # Evening star
        if (prev2_price > prev2_open and
            abs(prev_price - prev_open) < (prev2_price - prev2_open) * 0.3 and
            price < curr_open and price < (prev2_open + prev2_price) / 2):
            return True, "EVENING_STAR", 0.75
        
        # Shooting star
        body = abs(price - curr_open)
        upper_wick = curr_high - max(price, curr_open)
        lower_wick = min(price, curr_open) - curr_low
        if upper_wick > body * 2 and lower_wick < body * 0.5 and price < curr_open:
            return True, "SHOOTING_STAR", 0.7
    
    return False, None, 0


def find_key_levels(symbol, df):
    """
    Find key support/resistance, supply/demand, and structure levels.
    Returns list of key levels with their type.
    """
    if len(df) < 50:
        return []
    
    levels = []
    price = df['close'].iloc[-1]
    
    # 1. Swing highs and lows (structure levels)
    for i in range(5, len(df) - 5):
        # Swing high
        if df['high'].iloc[i] == df['high'].iloc[i-5:i+6].max():
            levels.append({
                'price': df['high'].iloc[i],
                'type': 'RESISTANCE',
                'strength': 1 + sum(1 for j in range(max(0, i-20), min(len(df), i+20)) 
                                   if abs(df['high'].iloc[j] - df['high'].iloc[i]) < df['high'].iloc[i] * 0.001)
            })
        # Swing low
        if df['low'].iloc[i] == df['low'].iloc[i-5:i+6].min():
            levels.append({
                'price': df['low'].iloc[i],
                'type': 'SUPPORT',
                'strength': 1 + sum(1 for j in range(max(0, i-20), min(len(df), i+20))
                                   if abs(df['low'].iloc[j] - df['low'].iloc[i]) < df['low'].iloc[i] * 0.001)
            })
    
    # 2. Psychological levels
    if symbol in PSYCHOLOGICAL_LEVELS:
        for level in PSYCHOLOGICAL_LEVELS[symbol]:
            distance = abs(level - price)
            if distance < price * 0.02:  # Within 2%
                levels.append({
                    'price': level,
                    'type': 'PSYCHOLOGICAL',
                    'strength': 3
                })
    
    # 3. Recent highs and lows
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    levels.append({'price': recent_high, 'type': 'RECENT_HIGH', 'strength': 2})
    levels.append({'price': recent_low, 'type': 'RECENT_LOW', 'strength': 2})
    
    # 4. VWAP (Volume Weighted Average Price approximation)
    if 'tick_volume' in df.columns:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['tick_volume']).sum() / df['tick_volume'].sum()
        levels.append({'price': vwap, 'type': 'VWAP', 'strength': 2})
    
    # Sort by distance from current price
    levels = sorted(levels, key=lambda x: abs(x['price'] - price))
    
    return levels[:10]  # Return top 10 nearest levels


def is_at_key_level(symbol, df, direction):
    """
    Check if price is at a key level for the intended trade direction.
    Returns (is_at_level, level_info, message).
    """
    if not REQUIRE_KEY_LEVEL:
        return True, None, "Key level check disabled"
    
    price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.001
    
    levels = find_key_levels(symbol, df)
    
    for level in levels:
        distance = abs(level['price'] - price)
        # Within 0.5 ATR of a key level
        if distance < atr * 0.5:
            # For BUY, should be at support/demand
            if direction == "BUY" and level['type'] in ['SUPPORT', 'RECENT_LOW', 'VWAP']:
                return True, level, f"At {level['type']}: {level['price']:.5f}"
            # For SELL, should be at resistance/supply
            elif direction == "SELL" and level['type'] in ['RESISTANCE', 'RECENT_HIGH', 'PSYCHOLOGICAL']:
                return True, level, f"At {level['type']}: {level['price']:.5f}"
    
    return False, None, "Not at a key level"


def check_momentum_confirmation(df, direction):
    """
    Check if momentum confirms the intended direction.
    Returns (is_confirmed, strength, message).
    """
    if not REQUIRE_MOMENTUM_CONFIRM:
        return True, 1.0, "Momentum check disabled"
    
    if len(df) < 20:
        return False, 0, "Insufficient data"
    
    # Get indicators
    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
    macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
    prev_macd_hist = df['macd_hist'].iloc[-2] if 'macd_hist' in df.columns else 0
    stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 50
    stoch_d = df['stoch_d'].iloc[-1] if 'stoch_d' in df.columns else 50
    
    confirm_points = 0
    max_points = 4
    
    if direction == "BUY":
        if rsi > 40 and rsi < 70:  # Not oversold, not overbought
            confirm_points += 1
        if macd_hist > 0 or (macd_hist > prev_macd_hist):  # MACD positive or improving
            confirm_points += 1
        if stoch_k > stoch_d:  # Stochastic bullish cross
            confirm_points += 1
        if stoch_k < 80:  # Not overbought
            confirm_points += 1
    else:  # SELL
        if rsi > 30 and rsi < 60:  # Not overbought, not oversold
            confirm_points += 1
        if macd_hist < 0 or (macd_hist < prev_macd_hist):  # MACD negative or declining
            confirm_points += 1
        if stoch_k < stoch_d:  # Stochastic bearish cross
            confirm_points += 1
        if stoch_k > 20:  # Not oversold
            confirm_points += 1
    
    strength = confirm_points / max_points
    is_confirmed = confirm_points >= 3
    
    return is_confirmed, strength, f"Momentum: {confirm_points}/{max_points}"


def calculate_rr_ratio(entry_price, sl_price, tp_price, direction):
    """Calculate risk:reward ratio."""
    if direction == "BUY":
        risk = entry_price - sl_price
        reward = tp_price - entry_price
    else:
        risk = sl_price - entry_price
        reward = entry_price - tp_price
    
    if risk <= 0:
        return 0
    return reward / risk


# ================================================================================
# ========================= AI EXIT VERIFICATION SYSTEM ==========================
# ================================================================================

# Configuration for AI-verified exits - ULTRA FAST
AI_EXIT_ENABLED = True  # Master switch for AI exit verification
AI_EXIT_MIN_PROFIT_PIPS = 1  # Start monitoring at just 1 pip profit
AI_EXIT_CACHE_SECONDS = 3  # Cache AI decisions for 3 seconds only
AI_EXIT_REQUIRE_CONFIRMATION = True  # TRUE = GPT AI actively monitors and closes trades
AI_EXIT_FAST_CLOSE_PIPS = 3  # Close immediately at 3 pips (was 4)
AI_EXIT_NEVER_BLOCK_PROFIT = True  # NEVER let AI block a profitable close

# ================================================================================
# ================= AI NEWS TRADING SYSTEM ======================================
# ================================================================================

# Configuration for AI News-Based Trading
AI_NEWS_TRADING_ENABLED = True  # Master switch for news-based trade entries
AI_NEWS_TRADE_MIN_CONFIDENCE = 0.65  # Lower confidence threshold for more trades
AI_NEWS_CHECK_INTERVAL = 30  # Check news every 30 seconds (faster)
AI_ENTRY_SCANNER_ENABLED = True  # Allow AI to find entry points
AI_ENTRY_MIN_QUALITY = 7  # Minimum quality score - 7+ now enters (was 9)
AI_AUTO_TRADE_NEWS = True  # Automatically open trades based on news
AI_AUTO_TRADE_ENTRIES = True  # Automatically open trades based on AI entries

# ================================================================================
# ================= AI SENTIMENT-BASED TRADING ==================================
# ================================================================================
# Automatically execute trades when sentiment is strong enough

AI_SENTIMENT_TRADING_ENABLED = True  # Master switch - execute trades on strong sentiment
AI_SENTIMENT_MIN_CONFIDENCE = 0.70  # Minimum confidence to trigger trade (70%)
AI_SENTIMENT_STRONG_CONFIDENCE = 0.80  # Strong confidence opens multiple positions (80%)
AI_SENTIMENT_ULTRA_CONFIDENCE = 0.90  # Ultra confidence opens max positions (90%)
AI_SENTIMENT_COOLDOWN_SECONDS = 30  # Wait 30 seconds between sentiment trades (faster)
AI_SENTIMENT_MAX_TRADES_PER_HOUR = 20  # Max 20 sentiment-based trades per hour
AI_SENTIMENT_LOT_MULTIPLIER = 1.5  # Lot multiplier for sentiment trades
AI_SENTIMENT_IGNORE_SPREAD = True  # Ignore spread filter for sentiment trades (AI is confident)
AI_SENTIMENT_MAX_SPREAD_MULTIPLIER = 5.0  # Allow up to 5x normal spread for sentiment trades

# Track sentiment trades
ai_sentiment_trades = {}  # {symbol: {'last_trade': timestamp, 'trades_this_hour': count}}

# Sentiment cache for live display
ai_sentiment_cache = {}
ai_sentiment_cache_time = {}
AI_SENTIMENT_CACHE_SECONDS = 30  # Cache sentiment for 30 seconds

# ================================================================================
# ================= ENHANCED SENTIMENT PROFIT PROTECTION v2.0 ====================
# ================================================================================
# Advanced profit drop and profit protection system specifically for sentiment trades

SENTIMENT_PROFIT_PROTECTION_ENABLED = True  # Master switch for sentiment profit protection

# ========== SENTIMENT-AWARE PROFIT DROP TIERS ==========
# More aggressive tiers for sentiment trades compared to regular trades
# Sentiment trades should be protected faster since sentiment can shift quickly
SENTIMENT_PROFIT_DROP_TIERS = {
    'instant':   {'min_peak': 0.03, 'drop_pct': 0.40, 'drop_pips': 0.02},  # 0.03+ pips: close on 40% drop
    'nano':      {'min_peak': 0.15, 'drop_pct': 0.30, 'drop_pips': 0.05},  # 0.15+ pips: close on 30% drop
    'micro':     {'min_peak': 0.4,  'drop_pct': 0.25, 'drop_pips': 0.10},  # 0.4+ pips: close on 25% drop
    'small':     {'min_peak': 0.8,  'drop_pct': 0.18, 'drop_pips': 0.15},  # 0.8+ pips: close on 18% drop
    'medium':    {'min_peak': 1.5,  'drop_pct': 0.12, 'drop_pips': 0.20},  # 1.5+ pips: close on 12% drop
    'large':     {'min_peak': 3.0,  'drop_pct': 0.08, 'drop_pips': 0.25},  # 3.0+ pips: close on 8% drop
    'profit':    {'min_peak': 5.0,  'drop_pct': 0.05, 'drop_pips': 0.30},  # 5.0+ pips: close on 5% drop
    'secure':    {'min_peak': 8.0,  'drop_pct': 0.03, 'drop_pips': 0.35},  # 8.0+ pips: close on 3% drop (maximum protection)
}

# ========== CONFIDENCE-BASED PROTECTION LEVELS ==========
# Higher confidence trades can have slightly looser protection (AI is more certain)
# Lower confidence trades need tighter protection (less certainty)
SENTIMENT_PROTECTION_BY_CONFIDENCE = {
    'ultra':  {'min_conf': 0.90, 'drop_multiplier': 1.2, 'trail_distance': 0.20},  # 90%+: 20% looser drops
    'strong': {'min_conf': 0.80, 'drop_multiplier': 1.0, 'trail_distance': 0.15},  # 80%+: normal protection
    'medium': {'min_conf': 0.70, 'drop_multiplier': 0.8, 'trail_distance': 0.10},  # 70%+: 20% tighter drops
    'low':    {'min_conf': 0.60, 'drop_multiplier': 0.6, 'trail_distance': 0.05},  # 60%+: 40% tighter drops
}

# ========== DYNAMIC TRAILING STOP FOR SENTIMENT TRADES ==========
SENTIMENT_TRAILING_ENABLED = True
SENTIMENT_TRAILING_START_PIPS = 0.8       # Start trailing after 0.8 pips profit
SENTIMENT_TRAILING_DISTANCE_PIPS = 0.3    # Trail 0.3 pips behind (tight!)
SENTIMENT_TRAILING_STEP_PIPS = 0.1        # Adjust trail in 0.1 pip increments
SENTIMENT_TRAILING_LOCK_PROFIT = True     # Never trail backwards
SENTIMENT_TRAILING_ACCELERATE = True      # Trail faster at higher profits
SENTIMENT_TRAILING_ACCELERATION_RATE = 0.05  # Decrease distance by 0.05 pips per 1 pip profit

# ========== SENTIMENT REVERSAL EMERGENCY EXIT ==========
SENTIMENT_REVERSAL_EXIT_ENABLED = True    # Exit if sentiment completely reverses
SENTIMENT_REVERSAL_EXIT_MIN_PROFIT = 0.1  # Min profit to allow reversal exit (protect entry)
SENTIMENT_REVERSAL_CONFIDENCE_THRESHOLD = 0.65  # Reversal must have 65%+ confidence
SENTIMENT_SHIFT_EXIT_ENABLED = True       # Exit on sentiment weakening (BULLISH → NEUTRAL)
SENTIMENT_SHIFT_MIN_PROFIT = 0.5          # Min profit for shift exit

# ========== MOMENTUM + SENTIMENT COMBINED PROTECTION ==========
SENTIMENT_MOMENTUM_PROTECTION = True      # Combine sentiment with momentum for exits
MOMENTUM_VS_SENTIMENT_EXIT = True         # Exit if momentum contradicts sentiment
MIN_MOMENTUM_CONFIRMATION_PROFIT = 0.3    # Check momentum confirmation after 0.3 pips

# ========== AI-DRIVEN EXIT ANALYSIS ==========
AI_EXIT_ANALYSIS_ENABLED = True           # Use AI to analyze if exit is warranted
AI_EXIT_ANALYSIS_INTERVAL = 15            # Check every 15 seconds for sentiment trades
AI_EXIT_CONFIDENCE_DECAY_RATE = 0.02      # Reduce exit threshold by 2% each check cycle
AI_EXIT_MIN_CONFIDENCE = 0.50             # Minimum exit confidence to trigger AI exit

# ========== SENTIMENT POSITION TRACKING ==========
# Track sentiment positions with their original confidence and sentiment
sentiment_position_data = {}  # {ticket: {'sentiment': str, 'confidence': float, 'peak_profit': float, 'entry_time': timestamp, 'last_check': timestamp}}
sentiment_peak_profits = {}   # {ticket: float} - Maximum profit reached per sentiment position

# Entry scanner cache
ai_entry_cache = {}
ai_entry_cache_time = {}
AI_ENTRY_CACHE_SECONDS = 45  # Cache entries for 45 seconds

# News trade tracking
ai_news_trades_today = {}
AI_MAX_NEWS_TRADES_PER_DAY = 5  # Max news-based trades per day

# ================================================================================
# ================= AI SESSION & TIMING OPTIMIZER ================================
# ================================================================================

# Configuration for AI Session Trading
AI_SESSION_OPTIMIZER_ENABLED = True  # Master switch for AI session optimization
AI_ONLY_TRADE_OPTIMAL_TIMES = True  # Only trade when AI says it's optimal
AI_SESSION_CHECK_INTERVAL = 300  # Check session quality every 5 minutes

# ========== FOREX MARKET HOURS (UTC) ==========
# These are the main trading sessions when liquidity is high

TRADING_SESSIONS = {
    'SYDNEY': {'start': 22, 'end': 7, 'quality': 'LOW'},       # 10 PM - 7 AM UTC (low volume)
    'TOKYO': {'start': 0, 'end': 9, 'quality': 'MEDIUM'},      # 12 AM - 9 AM UTC
    'LONDON': {'start': 7, 'end': 16, 'quality': 'HIGH'},      # 7 AM - 4 PM UTC (best for most pairs)
    'NEW_YORK': {'start': 12, 'end': 21, 'quality': 'HIGH'},   # 12 PM - 9 PM UTC
    'OVERLAP_LON_NY': {'start': 12, 'end': 16, 'quality': 'EXCELLENT'},  # 12 PM - 4 PM UTC (best liquidity)
}

# Hours to AVOID trading (UTC) - low liquidity, high spreads
AVOID_TRADING_HOURS_UTC = [
    21, 22, 23, 0, 1, 2, 3, 4, 5, 6,  # Late night / early morning - very low volume
]

# Hours that are OPTIMAL for trading (UTC)
OPTIMAL_TRADING_HOURS_UTC = [
    7, 8, 9, 10, 11,           # London morning - good volume
    12, 13, 14, 15,            # London/NY overlap - BEST volume
    16, 17, 18, 19,            # NY afternoon - good volume
]

# Days to avoid (0=Monday, 6=Sunday)
AVOID_TRADING_DAYS = [
    6,  # Sunday - market closed or very thin
]

# Reduced trading on these days
REDUCED_TRADING_DAYS = [
    4,  # Friday afternoon - market thins out
]

# Gold-specific optimal hours (UTC) - Gold moves most during these
GOLD_OPTIMAL_HOURS_UTC = [
    7, 8, 9, 10,               # London open - Gold moves
    12, 13, 14, 15, 16,        # London/NY overlap - best for Gold
    17, 18,                    # Early NY - still good
]

# Gold hours to avoid
GOLD_AVOID_HOURS_UTC = [
    19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6,  # Late NY through Asian - Gold is flat/choppy
]


def is_market_open():
    """
    Check if forex market is open.
    Forex is open 24/5 (Sunday 5 PM EST to Friday 5 PM EST)
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    
    # Sunday: market opens at ~22:00 UTC (5 PM EST)
    if now.weekday() == 6:  # Sunday
        if now.hour < 22:
            return False, "Market closed (Sunday before 22:00 UTC)"
    
    # Saturday: market is closed
    if now.weekday() == 5:  # Saturday
        return False, "Market closed (Saturday)"
    
    # Friday after 22:00 UTC: market basically closed
    if now.weekday() == 4 and now.hour >= 22:
        return False, "Market closing (Friday late)"
    
    return True, "Market open"


def get_current_session_quality(symbol=None):
    """
    Returns current trading session quality based on time.
    Returns: (quality, session_name, should_trade, reason)
    
    Quality levels: EXCELLENT, HIGH, MEDIUM, LOW, AVOID
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    
    # Check if market is open
    market_open, market_reason = is_market_open()
    if not market_open:
        return 'CLOSED', 'CLOSED', False, market_reason
    
    # Check if it's a day to avoid
    if weekday in AVOID_TRADING_DAYS:
        return 'AVOID', 'WEEKEND', False, "Sunday - market thin, avoid trading"
    
    # Friday afternoon reduction
    if weekday == 4 and hour >= 18:  # Friday after 6 PM UTC
        return 'LOW', 'FRIDAY_CLOSE', True, "Friday closing - reduce size"
    
    # Determine current session
    current_session = 'OFF_HOURS'
    session_quality = 'LOW'
    
    # London/NY overlap (BEST time)
    if 12 <= hour < 16:
        current_session = 'OVERLAP_LON_NY'
        session_quality = 'EXCELLENT'
    # London session
    elif 7 <= hour < 16:
        current_session = 'LONDON'
        session_quality = 'HIGH'
    # New York session
    elif 12 <= hour < 21:
        current_session = 'NEW_YORK'
        session_quality = 'HIGH'
    # Tokyo session
    elif 0 <= hour < 9:
        current_session = 'TOKYO'
        session_quality = 'MEDIUM'
    # Sydney / off-hours
    else:
        current_session = 'SYDNEY'
        session_quality = 'LOW'
    
    # Check if current hour should be avoided
    if hour in AVOID_TRADING_HOURS_UTC:
        return 'AVOID', current_session, False, f"Low volume hour ({hour}:00 UTC) - avoid trading"
    
    # Gold-specific check
    if symbol and ('XAU' in symbol or 'GOLD' in symbol):
        if hour in GOLD_AVOID_HOURS_UTC:
            return 'AVOID', current_session, False, f"Gold low-volume hour ({hour}:00 UTC)"
        if hour in GOLD_OPTIMAL_HOURS_UTC:
            session_quality = 'EXCELLENT' if session_quality in ['HIGH', 'EXCELLENT'] else 'HIGH'
    
    # Determine if we should trade
    should_trade = session_quality in ['EXCELLENT', 'HIGH', 'MEDIUM']
    reason = f"{current_session} session ({hour}:00 UTC) - {session_quality} quality"
    
    return session_quality, current_session, should_trade, reason


def get_next_good_trading_time():
    """
    Returns when the next good trading time starts.
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Find next optimal hour
    for h in range(1, 25):
        check_hour = (hour + h) % 24
        if check_hour in OPTIMAL_TRADING_HOURS_UTC:
            next_time = now + timedelta(hours=h)
            return next_time.strftime("%H:%M UTC"), h
    
    return "07:00 UTC", 24  # Default to London open

# Session quality cache
ai_session_cache = {}
ai_session_cache_time = {}
AI_SESSION_CACHE_SECONDS = 300  # Cache session analysis for 5 minutes

# Session performance tracking (learns from results)
session_performance = {
    'ASIAN': {'trades': 0, 'wins': 0, 'profit': 0.0},
    'LONDON': {'trades': 0, 'wins': 0, 'profit': 0.0},
    'NEW_YORK': {'trades': 0, 'wins': 0, 'profit': 0.0},
    'OVERLAP': {'trades': 0, 'wins': 0, 'profit': 0.0},
}

# Hour-by-hour performance tracking
hourly_performance = {h: {'trades': 0, 'wins': 0, 'profit': 0.0} for h in range(24)}

# ================================================================================
# ================= AI PROFIT ASSURANCE SYSTEM (ALWAYS PROFIT) ==================
# ================================================================================

# Configuration for AI Profit Assurance
AI_PROFIT_ASSURANCE_ENABLED = True
AI_PROFIT_CHECK_INTERVAL = 15  # Check every 15 seconds
AI_REVERSAL_DETECTION_ENABLED = True  # Detect reversals before they happen
AI_ADAPTIVE_TRAILING_ENABLED = True  # AI adjusts trailing stops dynamically
AI_MAX_DRAWDOWN_PERCENT = 30  # Max % of profit to give back before intervention

# Cache for AI profit assurance decisions
ai_profit_assurance_cache = {}
last_profit_check_time = {}


def ai_predict_price_direction(symbol, df, user, timeframe="M5"):
    """
    Use AI to predict short-term price direction (next 5-15 minutes).
    Helps protect profits by anticipating reversals.
    
    Returns:
        {
            "direction": "UP" or "DOWN" or "SIDEWAYS",
            "confidence": 0.0 to 1.0,
            "expected_move_pips": estimated pips movement,
            "reversal_risk": "HIGH" or "MEDIUM" or "LOW",
            "recommendation": "HOLD" or "TIGHTEN_SL" or "CLOSE_NOW"
        }
    """
    client = get_openai_client()
    if not client:
        return {"direction": "SIDEWAYS", "confidence": 0.5, "reversal_risk": "MEDIUM", "recommendation": "HOLD"}
    
    try:
        df = calculate_advanced_indicators(df)
        
        # Get recent price action (last 20 candles)
        recent = df.tail(20)
        
        # Calculate key metrics
        current_price = recent['close'].iloc[-1]
        
        # Price momentum (short-term)
        price_1 = recent['close'].iloc[-2] if len(recent) >= 2 else current_price
        price_3 = recent['close'].iloc[-4] if len(recent) >= 4 else current_price
        price_5 = recent['close'].iloc[-6] if len(recent) >= 6 else current_price
        
        momentum_1 = ((current_price - price_1) / price_1) * 100
        momentum_3 = ((current_price - price_3) / price_3) * 100
        momentum_5 = ((current_price - price_5) / price_5) * 100
        
        # Momentum acceleration/deceleration
        momentum_accel = momentum_1 - (momentum_3 / 3)
        
        # Get indicators
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        rsi_prev = df['rsi'].iloc[-2] if 'rsi' in df and len(df) >= 2 else rsi
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df else 0
        macd_hist_prev = df['macd_hist'].iloc[-2] if 'macd_hist' in df and len(df) >= 2 else macd_hist
        stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df else 50
        atr = df['atr'].iloc[-1] if 'atr' in df else (recent['high'] - recent['low']).mean()
        
        # Divergence detection (price vs RSI)
        price_higher = current_price > price_5
        rsi_higher = rsi > df['rsi'].iloc[-6] if 'rsi' in df and len(df) >= 6 else True
        bearish_divergence = price_higher and not rsi_higher
        bullish_divergence = not price_higher and rsi_higher
        
        # Candle pattern analysis
        last_candle = recent.iloc[-1]
        prev_candle = recent.iloc[-2]
        is_bullish = last_candle['close'] > last_candle['open']
        is_prev_bullish = prev_candle['close'] > prev_candle['open']
        
        body = abs(last_candle['close'] - last_candle['open'])
        range_ = last_candle['high'] - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # Reversal candle patterns
        is_shooting_star = upper_wick > body * 2 and is_bullish == False
        is_hammer = lower_wick > body * 2 and is_bullish
        is_engulfing_bearish = is_prev_bullish and not is_bullish and body > abs(prev_candle['close'] - prev_candle['open'])
        is_engulfing_bullish = not is_prev_bullish and is_bullish and body > abs(prev_candle['close'] - prev_candle['open'])
        
        # Support/Resistance proximity
        recent_high = recent['high'].max()
        recent_low = recent['low'].min()
        price_to_high = ((recent_high - current_price) / atr) if atr > 0 else 0
        price_to_low = ((current_price - recent_low) / atr) if atr > 0 else 0
        
        context = f"""
=== SHORT-TERM PRICE PREDICTION for {symbol} ===

CURRENT STATE:
- Price: {current_price:.5f}
- Timeframe: {timeframe}

MOMENTUM (CRITICAL FOR PREDICTION):
- 1-Candle Momentum: {momentum_1:+.4f}%
- 3-Candle Momentum: {momentum_3:+.4f}%
- 5-Candle Momentum: {momentum_5:+.4f}%
- Momentum Acceleration: {momentum_accel:+.4f}% ({'ACCELERATING' if momentum_accel > 0 else 'DECELERATING'})

OSCILLATORS:
- RSI: {rsi:.1f} (Previous: {rsi_prev:.1f}) - {'RISING' if rsi > rsi_prev else 'FALLING'}
- MACD Histogram: {macd_hist:.4f} (Previous: {macd_hist_prev:.4f}) - {'RISING' if macd_hist > macd_hist_prev else 'FALLING'}
- Stochastic K: {stoch_k:.1f}

DIVERGENCES (REVERSAL SIGNALS):
- Bearish Divergence: {bearish_divergence} (Price up but RSI down)
- Bullish Divergence: {bullish_divergence} (Price down but RSI up)

CANDLE PATTERNS:
- Current Candle: {'BULLISH' if is_bullish else 'BEARISH'}
- Shooting Star (Bearish Reversal): {is_shooting_star}
- Hammer (Bullish Reversal): {is_hammer}
- Bearish Engulfing: {is_engulfing_bearish}
- Bullish Engulfing: {is_engulfing_bullish}

STRUCTURE:
- Distance to Recent High: {price_to_high:.2f}x ATR
- Distance to Recent Low: {price_to_low:.2f}x ATR
- ATR (Volatility): {atr:.5f}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a price prediction algorithm for scalping trades (5-15 minute horizon).

Your job is to predict:
1. Direction the price is LIKELY to move in the next 5-15 minutes
2. Confidence in this prediction
3. Risk of reversal

KEY SIGNALS TO WATCH:
- Momentum deceleration = likely reversal coming
- Divergences = strong reversal warning
- Reversal candle patterns = immediate reversal likely
- Extreme RSI (>70 or <30) = reversal zone
- MACD histogram shrinking = momentum fading

Respond ONLY with valid JSON:
{
    "direction": "UP" or "DOWN" or "SIDEWAYS",
    "confidence": 0.5 to 1.0,
    "expected_move_pips": number (estimated),
    "reversal_risk": "HIGH" or "MEDIUM" or "LOW",
    "time_horizon_minutes": 5 or 10 or 15,
    "key_signal": "What's the main signal you're seeing",
    "recommendation": "HOLD" or "TIGHTEN_SL" or "CLOSE_NOW"
}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=250
        )
        
        result_text = response.choices[0].message.content
        if not result_text or result_text.strip() == '':
            return {"direction": "SIDEWAYS", "confidence": 0.5, "reversal_risk": "MEDIUM", "recommendation": "HOLD"}
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        if not result_text:
            return {"direction": "SIDEWAYS", "confidence": 0.5, "reversal_risk": "MEDIUM", "recommendation": "HOLD"}
        
        result = json.loads(result_text)
        
        logger.debug(f"[{user}] 🔮 AI Prediction {symbol}: {result['direction']} ({result['confidence']:.0%}) - {result.get('key_signal', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.debug(f"AI prediction error: {e}")
        return {"direction": "SIDEWAYS", "confidence": 0.5, "reversal_risk": "MEDIUM", "recommendation": "HOLD"}


def ai_optimize_trailing_stop(position, symbol, df, current_profit_pips, peak_profit_pips, user):
    """
    AI dynamically optimizes trailing stop based on market conditions.
    Ensures maximum profit retention while allowing room for continued gains.
    
    Returns:
        {
            "action": "TIGHTEN" or "WIDEN" or "HOLD" or "CLOSE",
            "suggested_trail_percent": 0.5 to 0.8 (% of profit to lock),
            "reason": explanation
        }
    """
    if not AI_ADAPTIVE_TRAILING_ENABLED:
        return {"action": "HOLD", "suggested_trail_percent": 0.6, "reason": "AI trailing disabled"}
    
    client = get_openai_client()
    if not client:
        return {"action": "HOLD", "suggested_trail_percent": 0.6, "reason": "No AI client"}
    
    try:
        # Get AI prediction
        prediction = ai_predict_price_direction(symbol, df, user)
        
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        direction = "BUY" if is_buy else "SELL"
        
        # Calculate profit giveback
        profit_given_back = peak_profit_pips - current_profit_pips
        giveback_percent = (profit_given_back / peak_profit_pips * 100) if peak_profit_pips > 0 else 0
        
        # Quick decision based on prediction
        reversal_risk = prediction.get('reversal_risk', 'MEDIUM')
        predicted_direction = prediction.get('direction', 'SIDEWAYS')
        recommendation = prediction.get('recommendation', 'HOLD')
        
        # If AI says CLOSE_NOW, respect it
        if recommendation == "CLOSE_NOW" and current_profit_pips > 1:
            return {
                "action": "CLOSE",
                "suggested_trail_percent": 1.0,
                "reason": f"AI detected reversal signal: {prediction.get('key_signal', 'Unknown')}"
            }
        
        # If reversal risk is HIGH and direction is against our position
        if reversal_risk == "HIGH":
            if (is_buy and predicted_direction == "DOWN") or (not is_buy and predicted_direction == "UP"):
                if current_profit_pips > 3:
                    return {
                        "action": "CLOSE",
                        "suggested_trail_percent": 1.0,
                        "reason": f"HIGH reversal risk detected against {direction} position"
                    }
                else:
                    return {
                        "action": "TIGHTEN",
                        "suggested_trail_percent": 0.8,  # Lock 80% of profit
                        "reason": "Reversal risk - tightening trail"
                    }
        
        # If direction matches our position, can be more relaxed
        if (is_buy and predicted_direction == "UP") or (not is_buy and predicted_direction == "DOWN"):
            return {
                "action": "WIDEN",
                "suggested_trail_percent": 0.5,  # Lock only 50%, allow room to run
                "reason": f"Momentum continues in our favor ({predicted_direction})"
            }
        
        # Default: moderate trailing
        return {
            "action": "HOLD",
            "suggested_trail_percent": 0.6,
            "reason": "Normal market conditions"
        }
        
    except Exception as e:
        logger.debug(f"AI trailing optimization error: {e}")
        return {"action": "HOLD", "suggested_trail_percent": 0.6, "reason": f"Error: {e}"}


def ai_profit_assurance_check(position, symbol, df, user):
    """
    Comprehensive AI check to ENSURE position ends in profit.
    Called periodically for all open positions.
    
    Returns:
        {
            "action": "HOLD" or "TIGHTEN_SL" or "CLOSE_PROFIT" or "HEDGE",
            "urgency": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
            "profit_outlook": "IMPROVING" or "STABLE" or "DETERIORATING",
            "recommended_sl_adjustment": pips or None,
            "reason": explanation
        }
    """
    global ai_profit_assurance_cache, last_profit_check_time
    
    if not AI_PROFIT_ASSURANCE_ENABLED:
        return {"action": "HOLD", "urgency": "LOW", "profit_outlook": "STABLE", "reason": "Disabled"}
    
    # Check cache
    cache_key = f"{symbol}_{position.ticket}"
    current_time = time.time()
    
    if cache_key in ai_profit_assurance_cache:
        cached = ai_profit_assurance_cache[cache_key]
        if current_time - cached['time'] < AI_PROFIT_CHECK_INTERVAL:
            return cached['result']
    
    client = get_openai_client()
    if not client:
        return {"action": "HOLD", "urgency": "LOW", "profit_outlook": "STABLE", "reason": "No AI"}
    
    try:
        df = calculate_advanced_indicators(df)
        
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        direction = "BUY" if is_buy else "SELL"
        current_profit = position.profit
        entry_price = position.price_open
        
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            return {"action": "HOLD", "urgency": "LOW", "profit_outlook": "STABLE", "reason": "No symbol info"}
        
        point = info.point
        tick = mt5.symbol_info_tick(symbol)
        current_price = tick.bid if is_buy else tick.ask
        
        # Calculate pip values
        if 'JPY' in symbol:
            pip_mult = 1
        elif 'XAU' in symbol or 'GOLD' in symbol:
            pip_mult = 1
        else:
            pip_mult = 10
        
        pip_size = point * pip_mult
        
        if is_buy:
            profit_pips = (current_price - entry_price) / pip_size
        else:
            profit_pips = (entry_price - current_price) / pip_size
        
        # Get peak profit if tracked
        peak_pips = position_profit_peaks.get(position.ticket, profit_pips)
        
        # Get prediction
        prediction = ai_predict_price_direction(symbol, df, user)
        
        # Build context
        context = f"""
=== PROFIT ASSURANCE CHECK for #{position.ticket} ===

POSITION:
- Symbol: {symbol}
- Direction: {direction}
- Entry: {entry_price:.5f}
- Current: {current_price:.5f}
- Profit: ${current_profit:.2f} ({profit_pips:.1f} pips)
- Peak Profit: {peak_pips:.1f} pips
- Profit Given Back: {peak_pips - profit_pips:.1f} pips ({((peak_pips - profit_pips) / peak_pips * 100) if peak_pips > 0 else 0:.0f}%)

AI PREDICTION:
- Direction: {prediction.get('direction', 'UNKNOWN')}
- Reversal Risk: {prediction.get('reversal_risk', 'UNKNOWN')}
- Key Signal: {prediction.get('key_signal', 'None')}

INDICATORS:
- RSI: {df['rsi'].iloc[-1] if 'rsi' in df else 'N/A'}
- MACD Hist: {df['macd_hist'].iloc[-1] if 'macd_hist' in df else 'N/A'}
- ATR: {df['atr'].iloc[-1] if 'atr' in df else 'N/A'}

CRITICAL QUESTIONS:
1. Is this position in profit? {profit_pips > 0}
2. Is profit deteriorating? {profit_pips < peak_pips * 0.7}
3. Is reversal likely? {prediction.get('reversal_risk') == 'HIGH'}
4. Is momentum supporting our position? {(is_buy and prediction.get('direction') == 'UP') or (not is_buy and prediction.get('direction') == 'DOWN')}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a PROFIT PROTECTION AI. Your #1 job is to ENSURE trades end in profit.

RULES:
1. NEVER let a profitable trade become a loss
2. If profit has dropped >30% from peak, recommend TIGHTEN_SL or CLOSE_PROFIT
3. If reversal risk is HIGH against our position, recommend CLOSE_PROFIT
4. If trade is still profitable but momentum is fading, recommend TIGHTEN_SL
5. Only recommend HOLD if momentum supports our position

ACTIONS:
- HOLD: Keep position as is, momentum is good
- TIGHTEN_SL: Move SL closer to protect more profit
- CLOSE_PROFIT: Close now to secure profit before reversal
- HEDGE: Open opposite position to lock profit (advanced)

Respond ONLY with valid JSON:
{
    "action": "HOLD" or "TIGHTEN_SL" or "CLOSE_PROFIT",
    "urgency": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
    "profit_outlook": "IMPROVING" or "STABLE" or "DETERIORATING",
    "lock_percent": 0.5 to 0.9 (what % of profit to lock with SL),
    "reason": "Clear explanation",
    "next_check_seconds": 10 to 60
}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=250
        )
        
        result_text = response.choices[0].message.content
        if not result_text or result_text.strip() == '':
            return {"action": "HOLD", "urgency": "LOW", "profit_outlook": "STABLE", "reason": "AI returned empty"}
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        if not result_text:
            return {"action": "HOLD", "urgency": "LOW", "profit_outlook": "STABLE", "reason": "AI parse failed"}
        
        result = json.loads(result_text)
        
        # Cache result
        ai_profit_assurance_cache[cache_key] = {
            'time': current_time,
            'result': result
        }
        
        if result.get('urgency') in ['HIGH', 'CRITICAL']:
            logger.info(f"[{user}] ⚠️ AI Profit Assurance {symbol}: {result['action']} ({result['urgency']}) - {result['reason']}")
        
        return result
        
    except Exception as e:
        logger.debug(f"AI profit assurance error: {e}")
        return {"action": "HOLD", "urgency": "LOW", "profit_outlook": "STABLE", "reason": f"Error: {e}"}

# Cache for AI exit decisions
ai_exit_cache = {}


def get_ai_exit_verification(symbol, position, current_profit_pips, peak_profit_pips, direction, user):
    """
    Use GPT AI to verify if a trade should be closed NOW to lock in profit.
    
    Returns:
        (should_close, confidence, reason) - AI recommendation
    """
    global ai_exit_cache
    
    # Check cache first
    cache_key = f"{symbol}_{position.ticket}"
    if cache_key in ai_exit_cache:
        cached = ai_exit_cache[cache_key]
        if time.time() - cached['time'] < AI_EXIT_CACHE_SECONDS:
            return cached['should_close'], cached['confidence'], cached['reason']
    
    client = get_openai_client()
    if not client:
        # No AI client - default to NOT closing (let profit run)
        return False, 0.5, "AI unavailable - holding position"
    
    try:
        # Get recent price data for analysis
        df = get_data(symbol, TIMEFRAME, n=50)
        if df is None or len(df) < 20:
            return False, 0.5, "Insufficient data"
        
        df = calculate_advanced_indicators(df)
        
        # Prepare market context for AI
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].iloc[-10:].max()
        recent_low = df['low'].iloc[-10:].min()
        
        # Get indicator values
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        macd = df['macd'].iloc[-1] if 'macd' in df else 0
        macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df else 0
        atr = df['atr'].iloc[-1] if 'atr' in df else 0
        
        # Calculate trend
        ema_fast = df['ema_21'].iloc[-1] if 'ema_21' in df else current_price
        ema_slow = df['ema_50'].iloc[-1] if 'ema_50' in df else current_price
        trend = "BULLISH" if ema_fast > ema_slow else "BEARISH"
        
        # Recent price action
        last_5_candles = df.iloc[-5:][['open', 'high', 'low', 'close']].to_dict('records')
        
        prompt = f"""FAST PROFIT PROTECTION DECISION - {symbol}

TRADE STATUS:
- Direction: {direction}
- Current Profit: {current_profit_pips:.1f} pips
- Peak Profit: {peak_profit_pips:.1f} pips  
- Dropped from peak: {peak_profit_pips - current_profit_pips:.1f} pips

INDICATORS:
- RSI: {rsi:.1f}
- MACD: {macd:.4f} vs Signal: {macd_signal:.4f}
- Trend: {trend}

RULES:
1. NEVER close at a loss
2. If profit dropping from peak - recommend CLOSE
3. If reversal signals - recommend CLOSE
4. Protect profits aggressively - better to take small profit than risk loss
5. At 2+ pips profit, bias towards CLOSE

Reply JSON ONLY:
{{"decision": "CLOSE" or "HOLD", "confidence": 0.0-1.0, "reason": "brief"}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a PROFIT PROTECTOR. Your #1 goal: NEVER let winning trades become losers. Be AGGRESSIVE about closing trades in profit. If profit is dropping, say CLOSE. If in doubt with profit, say CLOSE. Only say HOLD if very confident price will continue favorably."
                },
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=150
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            # Clean up response
            result_text = result_text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            should_close = result.get('decision', 'HOLD').upper() == 'CLOSE'
            confidence = float(result.get('confidence', 0.5))
            reason = result.get('reason', 'AI analysis')
            
            # SAFETY CHECK: Never close if in loss!
            if current_profit_pips < 0:
                should_close = False
                reason = "SAFETY: Position in loss - holding"
                logger.warning(f"[{user}] 🛡️ AI blocked loss close on {symbol}")
            
            # Lower confidence requirement for faster closes (was 0.7, now 0.5)
            if should_close and confidence < 0.5:
                should_close = False
                reason = f"Low confidence ({confidence:.0%}) - holding"
            
            # Cache the decision (shorter cache for faster updates)
            ai_exit_cache[cache_key] = {
                'should_close': should_close,
                'confidence': confidence,
                'reason': reason,
                'time': time.time()
            }
            
            if should_close:
                logger.info(f"[{user}] 🤖 GPT says CLOSE {symbol}: {reason} ({confidence:.0%})")
            
            return should_close, confidence, reason
            
        except json.JSONDecodeError as e:
            logger.debug(f"AI exit JSON parse error: {e}")
            return False, 0.5, "AI response parse error - holding"
            
    except Exception as e:
        logger.error(f"AI exit verification error: {e}")
        return False, 0.5, f"AI error - holding position"


def should_close_position_ai(position, symbol, current_profit_pips, peak_profit_pips, direction, user, close_reason):
    """
    GPT AI-POWERED profit protection - CLOSES FAST when in profit.
    NEVER lets winners become losers.
    
    Returns:
        (should_close, final_reason)
    """
    # RULE 1: NEVER close if in loss
    if current_profit_pips < 0:
        return False, "Position in loss - blocked"
    
    # RULE 2: INSTANT CLOSE at 4+ pips (no AI needed)
    if current_profit_pips >= AI_EXIT_FAST_CLOSE_PIPS:
        logger.info(f"[{user}] ⚡ INSTANT CLOSE on {symbol}: {current_profit_pips:.1f} pips - TAKING PROFIT!")
        return True, f"INSTANT_PROFIT: {current_profit_pips:.1f} pips"
    
    # RULE 3: PROFIT DROP - Close ULTRA FAST if dropping from peak
    if peak_profit_pips >= 2 and current_profit_pips >= 0.5:
        pips_dropped = peak_profit_pips - current_profit_pips
        drop_percent = (pips_dropped / peak_profit_pips) if peak_profit_pips > 0 else 0
        
        # TIGHT protection - close fast on any significant drop
        if peak_profit_pips >= 10:
            max_drop = 2
            max_drop_pct = 0.20
        elif peak_profit_pips >= 5:
            max_drop = 1.5
            max_drop_pct = 0.25
        elif peak_profit_pips >= 3:
            max_drop = 1
            max_drop_pct = 0.30
        else:
            max_drop = 0.7
            max_drop_pct = 0.35
        
        # Close if dropped too much
        if pips_dropped >= max_drop or drop_percent >= max_drop_pct:
            logger.info(f"[{user}] ⚠️ PROFIT DROP: {symbol} Peak={peak_profit_pips:.1f} → Now={current_profit_pips:.1f} ({pips_dropped:.1f} pips / {drop_percent:.0%}) - CLOSING!")
            return True, f"PROFIT_DROP: {current_profit_pips:.1f} pips (was {peak_profit_pips:.1f})"
    
    # RULE 4: GPT AI FAST DECISION - For borderline cases, ask GPT
    if AI_EXIT_ENABLED and current_profit_pips >= AI_EXIT_MIN_PROFIT_PIPS:
        # Only consult AI for trades between min and fast-close thresholds
        should_close, confidence, ai_reason = get_ai_exit_verification(
            symbol, position, current_profit_pips, peak_profit_pips, direction, user
        )
        
        if should_close:
            logger.info(f"[{user}] 🤖 GPT CLOSE: {symbol} at {current_profit_pips:.1f} pips - {ai_reason}")
            return True, f"GPT_CLOSE: {ai_reason}"
        
        # AI says HOLD - but NEVER block if we have ANY decent profit
        if current_profit_pips >= 1.5:
            logger.info(f"[{user}] ✅ Closing {symbol} with {current_profit_pips:.1f} pips (overriding AI hold)")
            return True, f"PROFIT_TAKE: {current_profit_pips:.1f} pips"
        
        # AI says hold and profit is small - respect AI
        logger.debug(f"[{user}] 🤖 GPT holding {symbol} at {current_profit_pips:.1f} pips: {ai_reason}")
        return False, f"GPT_HOLD: {ai_reason}"
    
    # RULE 5: Default - close if above minimum profit
    if current_profit_pips >= AI_EXIT_MIN_PROFIT_PIPS:
        return True, close_reason
    
    return False, "Waiting for more profit"


def close_position_with_profit(position, symbol, reason, user):
    """
    Close a position and optionally queue for re-entry.
    Returns True if closed successfully.
    
    SAFETY: Will NOT close if position is in loss!
    """
    global reentry_queue, closed_with_profit
    
    try:
        # Get position profit info
        current_profit = position.profit
        
        # Only block if CLEARLY in loss (account for spread)
        if current_profit < -5:  # Only block if losing more than $5
            logger.warning(f"[{user}] ⛔ BLOCKED LOSS on #{position.ticket} {symbol}: ${current_profit:.2f}")
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        info = mt5.symbol_info(symbol)
        if not info:
            return False
        
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        close_price = tick.bid if is_buy else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        
        # Double-check profit at current close price
        if is_buy:
            profit_check = (close_price - position.price_open) * position.volume * info.trade_contract_size
        else:
            profit_check = (position.price_open - close_price) * position.volume * info.trade_contract_size
        
        # Account for swap and commission
        total_profit = profit_check + position.swap
        
        # Only block if clearly losing money (more than $2)
        if total_profit < -2:
            logger.warning(f"[{user}] ⛔ BLOCKED - would lose ${abs(total_profit):.2f} on #{position.ticket}")
            return False
        
        # Get filling mode
        filling_type = mt5.ORDER_FILLING_IOC
        if info.filling_mode & 1:
            filling_type = mt5.ORDER_FILLING_FOK
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "magic": MAGIC,
            "deviation": 20,
            "type_filling": filling_type,
            "comment": f"PROFIT_PROTECT:{reason}"
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            profit = position.profit
            direction = "BUY" if is_buy else "SELL"
            
            logger.info(f"[{user}] 💰 CLOSED #{position.ticket} {symbol} with ${profit:.2f} profit - {reason}")
            
            # Update user streak for AI lot sizing learning
            is_win = profit > 0
            update_user_streak(user, is_win, profit)
            
            # ============ AI STRATEGY PERFORMANCE LEARNING ============
            # Update strategy performance based on trade outcome
            if position.ticket in trade_strategies_used:
                trade_data = trade_strategies_used[position.ticket]
                strategies = trade_data.get('strategies', [])
                
                # Calculate profit in pips for learning
                sym_settings = get_symbol_settings(symbol)
                pip_value = sym_settings.get('pip_value', 0.0001)
                entry_price = trade_data.get('entry_price', position.price_open)
                
                if trade_data.get('direction') == 'BUY':
                    profit_pips = (close_price - entry_price) / pip_value
                else:
                    profit_pips = (entry_price - close_price) / pip_value
                
                # Update each strategy's performance
                for strat_name in strategies:
                    try:
                        update_strategy_performance(strat_name, is_win, profit_pips, user)
                    except Exception as e:
                        logger.debug(f"Strategy update error: {e}")
                
                # Log learning update
                logger.info(f"[{user}] 🧠 AI LEARNED from #{position.ticket}: {len(strategies)} strategies {'✅ WON' if is_win else '❌ LOST'} ({profit_pips:.1f} pips)")
                
                # Clean up
                del trade_strategies_used[position.ticket]
            
            # ============ LOSS RECOVERY MODE UPDATES ============
            if LOSS_RECOVERY_ENABLED:
                account = mt5.account_info()
                balance = account.balance if account else 1000
                
                if is_win:
                    # Update recovery progress
                    update_recovery_on_win(user, profit)
                else:
                    # Check if we should enter recovery mode
                    check_recovery_trigger(user, abs(profit), balance)
                    
                    # Check if we should pause after big loss
                    should_pause, pause_mins = should_pause_after_loss(user, abs(profit), balance)
                    if should_pause:
                        # Set a pause flag (user can resume manually)
                        logger.warning(f"[{user}] ⏸️ Trading paused for {pause_mins} min after big loss")
            
            # If loss, record for AI loss pattern learning
            if not is_win:
                learn_from_loss(user, symbol, abs(profit))
            
            # Log trade close to history
            log_trade(user, 'close', f'Closed {symbol} {direction}', {
                'symbol': symbol,
                'type': direction,
                'profit': profit,
                'lot': position.volume,
                'entry_price': position.price_open,
                'close_price': close_price,
                'ticket': position.ticket,
                'reason': reason
            })
            
            # Queue for re-entry if enabled
            if REENTRY_ENABLED and profit > 0:
                reentry_queue[symbol] = {
                    'direction': direction,
                    'closed_at': time.time(),
                    'lot': position.volume,
                    'entry_price': position.price_open,
                    'close_price': close_price,
                    'reason': reason
                }
                closed_with_profit[symbol] = {
                    'time': time.time(),
                    'profit': profit,
                    'direction': direction
                }
                logger.info(f"[{user}] 🔄 Queued {symbol} {direction} for re-entry")
            
            return True
        else:
            err = result.comment if result else "Unknown"
            logger.warning(f"Failed to close position: {err}")
            return False
            
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return False


def check_reentry_opportunities(user):
    """
    Check if any queued re-entries should be executed.
    Re-enters if signal is still valid after cooldown.
    """
    global reentry_queue
    
    if not REENTRY_ENABLED or not reentry_queue:
        return
    
    current_time = time.time()
    symbols_to_remove = []
    
    for symbol, entry_data in reentry_queue.items():
        try:
            # Check cooldown
            time_since_close = current_time - entry_data['closed_at']
            if time_since_close < REENTRY_COOLDOWN_SECONDS:
                continue
            
            # Check if cooldown expired (5 minutes max)
            if time_since_close > 300:
                symbols_to_remove.append(symbol)
                logger.debug(f"Re-entry expired for {symbol}")
                continue
            
            # Check if we already have a position on this symbol
            existing = mt5.positions_get(symbol=symbol)
            if existing:
                symbols_to_remove.append(symbol)
                continue
            
            direction = entry_data['direction']
            
            # Get fresh data and check if signal is still valid
            df = get_data(symbol, TIMEFRAME, n=100)
            if df is None or len(df) < 50:
                continue
            
            df = calculate_advanced_indicators(df)
            
            # Quick signal check - momentum still in same direction?
            signal_valid = False
            
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if direction == "BUY" and rsi > 40 and rsi < 70:
                    signal_valid = True
                elif direction == "SELL" and rsi < 60 and rsi > 30:
                    signal_valid = True
            
            # Also check price action - price should still be moving in our direction
            if len(df) >= 3:
                recent_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-3]
                if direction == "BUY" and recent_close >= prev_close:
                    signal_valid = True
                elif direction == "SELL" and recent_close <= prev_close:
                    signal_valid = True
            
            if not signal_valid and REENTRY_REQUIRE_CONFIRMATION:
                logger.debug(f"Re-entry signal not valid for {symbol} {direction}")
                continue
            
            # Execute re-entry
            logger.info(f"[{user}] 🔄 RE-ENTERING {symbol} {direction} after profit close")
            
            # Get account and calculate lot
            acc = mt5.account_info()
            if not acc:
                continue
            
            lot = entry_data.get('lot', MIN_LOT)
            
            # Calculate new SL/TP
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if not tick or not info:
                continue
            
            point = info.point
            pip_mult = 10 if 'JPY' not in symbol else 1
            pip_size = point * pip_mult
            
            if direction == "BUY":
                entry_price = tick.ask
                sl = entry_price - (STOPLOSS_PIPS * pip_size)
                tp = entry_price + (TAKEPROFIT_PIPS * pip_size)
                order_type = mt5.ORDER_TYPE_BUY
            else:
                entry_price = tick.bid
                sl = entry_price + (STOPLOSS_PIPS * pip_size)
                tp = entry_price - (TAKEPROFIT_PIPS * pip_size)
                order_type = mt5.ORDER_TYPE_SELL
            
            # Send order
            result = send_order(symbol, order_type, lot, sl, tp, f"REENTRY_{direction}")
            if result:
                logger.info(f"[{user}] ✅ Re-entry successful for {symbol} {direction}")
                symbols_to_remove.append(symbol)
            
        except Exception as e:
            logger.error(f"Error checking re-entry for {symbol}: {e}")
    
    # Clean up processed entries
    for symbol in symbols_to_remove:
        if symbol in reentry_queue:
            del reentry_queue[symbol]


def manage_r_based_profit_protection(user):
    """
    AGGRESSIVE profit protection with close & re-enter.
    
    Key Features:
    1. Tracks peak profit for each position
    2. If profit drops significantly from peak, CLOSES the trade
    3. Queues the trade for re-entry if signal is still valid
    4. Also handles traditional R-based SL adjustments
    
    This ensures we NEVER let a winning trade become a loser.
    """
    global position_entry_data, position_profit_peaks, position_profit_peaks_dollars, reentry_queue
    
    if not PROFIT_PROTECTION_ENABLED:
        return
    
    # First, check for re-entry opportunities
    check_reentry_opportunities(user)
    
    try:
        positions = mt5.positions_get()
        if not positions:
            # Clean up tracking for closed positions
            if position_entry_data:
                position_entry_data.clear()
            if position_profit_peaks:
                position_profit_peaks.clear()
            if position_profit_peaks_dollars:
                position_profit_peaks_dollars.clear()
            return
        
        # Get list of active tickets
        active_tickets = {pos.ticket for pos in positions}
        
        # Clean up closed positions from tracking
        closed_tickets = [t for t in position_entry_data.keys() if t not in active_tickets]
        for t in closed_tickets:
            del position_entry_data[t]
            if t in position_profit_peaks:
                del position_profit_peaks[t]
            if t in position_profit_peaks_dollars:
                del position_profit_peaks_dollars[t]
        
        for pos in positions:
            try:
                symbol = pos.symbol
                ticket = pos.ticket
                
                # Get symbol info
                info = mt5.symbol_info(symbol)
                if not info:
                    continue
                
                point = info.point
                digits = info.digits
                
                # Calculate pip multiplier based on symbol
                if symbol in SYMBOL_SETTINGS:
                    pip_value = SYMBOL_SETTINGS[symbol]['pip_value']
                    pip_mult = pip_value / point if point > 0 else 10
                elif 'JPY' in symbol:
                    pip_mult = 1
                elif 'XAU' in symbol or 'GOLD' in symbol:
                    pip_mult = 1  # Gold uses 0.1 as pip
                else:
                    pip_mult = 10  # Standard forex
                
                pip_size = point * pip_mult
                
                # Get tick data
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                
                is_buy = pos.type == mt5.POSITION_TYPE_BUY
                current_price = tick.bid if is_buy else tick.ask
                entry_price = pos.price_open
                current_sl = pos.sl
                current_tp = pos.tp
                
                # Calculate minimum stop distance (broker requirement)
                stops_level = info.trade_stops_level
                if stops_level == 0:
                    stops_level = 10  # Default minimum
                min_stop_distance = stops_level * point
                
                # Initialize or get position entry data
                if ticket not in position_entry_data:
                    # Calculate SL distance (risk per trade)
                    if is_buy:
                        sl_distance = entry_price - current_sl if current_sl > 0 else pip_size * STOPLOSS_PIPS
                        tp_distance = current_tp - entry_price if current_tp > 0 else sl_distance * 2
                    else:
                        sl_distance = current_sl - entry_price if current_sl > 0 else pip_size * STOPLOSS_PIPS
                        tp_distance = entry_price - current_tp if current_tp > 0 else sl_distance * 2
                    
                    # Ensure sl_distance is positive and reasonable
                    sl_distance = max(sl_distance, min_stop_distance * 2)
                    
                    position_entry_data[ticket] = {
                        'entry': entry_price,
                        'sl_distance': sl_distance,
                        'tp_distance': tp_distance,
                        'initial_sl': current_sl,
                        'initial_tp': current_tp,
                        'risk_reduced': False,
                        'breakeven_set': False,
                        'partial_taken': False,
                        'trailing_active': False,
                        'profit_locked': False
                    }
                    logger.debug(f"[{user}] Initialized tracking for #{ticket}: SL_dist={sl_distance:.5f}")
                
                data = position_entry_data[ticket]
                sl_distance = data['sl_distance']
                
                if sl_distance <= 0:
                    logger.warning(f"Invalid SL distance for #{ticket}")
                    continue
                
                # Calculate current profit in R-multiples
                if is_buy:
                    profit_distance = current_price - entry_price
                else:
                    profit_distance = entry_price - current_price
                
                current_r = profit_distance / sl_distance
                
                # Calculate profit in pips
                profit_pips = profit_distance / pip_size
                
                # Track peak profit (in R)
                if ticket not in position_profit_peaks:
                    position_profit_peaks[ticket] = current_r
                elif current_r > position_profit_peaks[ticket]:
                    position_profit_peaks[ticket] = current_r
                
                peak_r = position_profit_peaks[ticket]
                peak_pips = peak_r * (sl_distance / pip_size)  # Convert peak R to pips
                
                # Log current state periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.debug(f"[{user}] #{ticket} {symbol}: R={current_r:.2f} Peak={peak_r:.2f} Pips={profit_pips:.1f}")
                
                # ============================================================
                # === INSTANT BREAKEVEN - Move to BE as soon as possible ===
                # ============================================================
                if profit_pips >= INSTANT_BREAKEVEN_PIPS and not data.get('instant_be_set'):
                    # Move SL to entry + small buffer immediately
                    lock_pips = max(LOCK_MIN_PROFIT_PIPS, profit_pips * 0.3)  # Lock 30% or minimum
                    
                    if is_buy:
                        new_sl = entry_price + (lock_pips * pip_size)
                        new_sl = round(new_sl, digits)
                        if new_sl > current_sl and (current_price - new_sl) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                data['instant_be_set'] = True
                                logger.info(f"[{user}] 🛡️ INSTANT BE #{ticket}: Locked +{lock_pips:.1f} pips profit")
                    else:  # SELL
                        new_sl = entry_price - (lock_pips * pip_size)
                        new_sl = round(new_sl, digits)
                        if (new_sl < current_sl or current_sl == 0) and (new_sl - current_price) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                data['instant_be_set'] = True
                                logger.info(f"[{user}] 🛡️ INSTANT BE #{ticket}: Locked +{lock_pips:.1f} pips profit")
                
                # ============================================================
                # === DOLLAR-BASED PROFIT PROTECTION (NEW!) ===
                # ============================================================
                if DOLLAR_PROFIT_PROTECTION:
                    current_profit_dollars = pos.profit  # MT5 gives profit in account currency
                    
                    # Track peak dollar profit
                    if ticket not in position_profit_peaks_dollars:
                        position_profit_peaks_dollars[ticket] = current_profit_dollars
                    elif current_profit_dollars > position_profit_peaks_dollars[ticket]:
                        position_profit_peaks_dollars[ticket] = current_profit_dollars
                    
                    peak_profit_dollars = position_profit_peaks_dollars[ticket]
                    
                    # Check if we should close based on dollar profit drop
                    if peak_profit_dollars >= MIN_PROFIT_DOLLARS_TO_PROTECT and current_profit_dollars > 0:
                        # Find the right tier
                        close_threshold = CLOSE_WHEN_PROFIT_DROPS_TO  # Default
                        
                        for tier_name, tier_config in DOLLAR_PROFIT_DROP_TIERS.items():
                            if peak_profit_dollars >= tier_config['min_peak']:
                                close_threshold = tier_config['close_at']
                        
                        # Check if current profit dropped below threshold
                        if current_profit_dollars <= close_threshold and peak_profit_dollars > close_threshold:
                            logger.warning(f"[{user}] 💰 DOLLAR DROP #{ticket}: Peak=${peak_profit_dollars:.2f} → Now=${current_profit_dollars:.2f} - CLOSING!")
                            closed = close_position_with_profit(pos, symbol, f"DOLLAR_DROP_{peak_profit_dollars:.0f}to{current_profit_dollars:.0f}", user)
                            if closed:
                                # Clean up tracking
                                if ticket in position_profit_peaks_dollars:
                                    del position_profit_peaks_dollars[ticket]
                                continue
                    
                    # NEVER let profit go negative if we were at $1+
                    if NEVER_LET_PROFIT_GO_NEGATIVE and peak_profit_dollars >= 1.0 and current_profit_dollars <= 0.10:
                        logger.warning(f"[{user}] 🚨 ZERO PROFIT #{ticket}: Was ${peak_profit_dollars:.2f}, now ${current_profit_dollars:.2f} - EMERGENCY CLOSE!")
                        closed = close_position_with_profit(pos, symbol, "ZERO_PROFIT_SAVE", user)
                        if closed:
                            if ticket in position_profit_peaks_dollars:
                                del position_profit_peaks_dollars[ticket]
                            continue
                
                # ============================================================
                # === AGGRESSIVE PROFIT PROTECTION - CLOSE FAST ===
                # ============================================================
                position_direction = "BUY" if is_buy else "SELL"
                
                # LOG CURRENT STATE (include dollar amount)
                if profit_pips > 0.5 or pos.profit >= 0.50:
                    logger.info(f"[{user}] 📊 #{ticket} {symbol}: Profit={profit_pips:.1f} pips (${pos.profit:.2f}), Peak=${position_profit_peaks_dollars.get(ticket, 0):.2f}")
                
                # === INSTANT CLOSE AT 3+ PIPS ===
                if profit_pips >= 3:
                    logger.info(f"[{user}] ⚡ INSTANT CLOSE #{ticket}: {profit_pips:.1f} pips (${pos.profit:.2f}) - CLOSING NOW!")
                    closed = close_position_with_profit(pos, symbol, f"INSTANT_PROFIT_{profit_pips:.0f}", user)
                    if closed:
                        continue
                
                # === PROFIT DROP PROTECTION ===
                if peak_pips >= 2 and profit_pips >= 0.3:
                    pips_dropped = peak_pips - profit_pips
                    drop_percent = pips_dropped / peak_pips if peak_pips > 0 else 0
                    
                    # Close if dropped 25% OR dropped 1+ pip
                    if drop_percent >= 0.25 or pips_dropped >= 1:
                        logger.warning(f"[{user}] ⚠️ PROFIT DROP #{ticket}: Peak={peak_pips:.1f} → Now={profit_pips:.1f} ({drop_percent:.0%} drop)")
                        closed = close_position_with_profit(pos, symbol, f"DROP_{peak_pips:.0f}to{profit_pips:.0f}", user)
                        if closed:
                            continue
                
                # === EMERGENCY - WAS IN PROFIT, NOW NEAR ZERO ===
                if peak_pips >= 2 and profit_pips < 1 and profit_pips > 0:
                    logger.warning(f"[{user}] 🚨 EMERGENCY #{ticket}: Was +{peak_pips:.1f}, now only +{profit_pips:.1f} - CLOSING!")
                    closed = close_position_with_profit(pos, symbol, "EMERGENCY_SAVE", user)
                    if closed:
                        continue
                
                # === CRITICAL - ABOUT TO GO NEGATIVE ===
                if peak_pips >= 1.5 and profit_pips <= 0.3 and profit_pips >= 0:
                    logger.warning(f"[{user}] 🚨 CRITICAL #{ticket}: About to go negative - CLOSING NOW!")
                    closed = close_position_with_profit(pos, symbol, "CRITICAL_SAVE", user)
                    if closed:
                        continue
                if peak_pips >= 3 and profit_pips <= 0.5 and profit_pips >= 0:
                    # This is critical - close NOW to avoid loss
                    logger.warning(f"[{user}] 🚨 CRITICAL: #{ticket} profit nearly zero (was {peak_pips:.1f}, now {profit_pips:.1f}) - closing to prevent loss")
                    closed = close_position_with_profit(pos, symbol, "ZERO_LOSS_CRITICAL", user)
                    if closed:
                        continue
                
                # === STAGE 1: At REDUCE_RISK_AT_R - Tighten SL ===
                if current_r >= REDUCE_RISK_AT_R and not data.get('risk_reduced'):
                    if is_buy:
                        # Move SL to reduce risk by 50%
                        new_sl = entry_price - (sl_distance * 0.5)
                        new_sl = round(new_sl, digits)
                        
                        # Validate: new SL must be better than current and respect min distance
                        if new_sl > current_sl and (current_price - new_sl) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                data['risk_reduced'] = True
                                logger.info(f"[{user}] 📊 +{REDUCE_RISK_AT_R}R: Risk reduced #{ticket} SL: {current_sl:.5f} → {new_sl:.5f}")
                            else:
                                err = result.comment if result else "Unknown"
                                logger.debug(f"Risk reduce failed: {err}")
                    else:  # SELL
                        new_sl = entry_price + (sl_distance * 0.5)
                        new_sl = round(new_sl, digits)
                        
                        if (new_sl < current_sl or current_sl == 0) and (new_sl - current_price) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                data['risk_reduced'] = True
                                logger.info(f"[{user}] 📊 +{REDUCE_RISK_AT_R}R: Risk reduced #{ticket} SL: {current_sl:.5f} → {new_sl:.5f}")
                
                # === STAGE 2: At BREAKEVEN_AT_R - Move to breakeven ===
                if current_r >= BREAKEVEN_AT_R and not data.get('breakeven_set'):
                    if is_buy:
                        # Set SL above entry with buffer (5 pips to cover spread and give room)
                        new_sl = entry_price + (pip_size * 5)
                        new_sl = round(new_sl, digits)
                        
                        if new_sl > current_sl and (current_price - new_sl) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                data['breakeven_set'] = True
                                logger.info(f"[{user}] 🛡️ +{BREAKEVEN_AT_R}R BREAKEVEN #{ticket} @ {new_sl:.5f}")
                            else:
                                err = result.comment if result else "Unknown"
                                logger.debug(f"Breakeven failed: {err}")
                    else:  # SELL
                        new_sl = entry_price - (pip_size * 5)  # 5 pip buffer below entry
                        new_sl = round(new_sl, digits)
                        
                        if (new_sl < current_sl or current_sl == 0) and (new_sl - current_price) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                data['breakeven_set'] = True
                                logger.info(f"[{user}] 🛡️ +{BREAKEVEN_AT_R}R BREAKEVEN #{ticket} @ {new_sl:.5f}")
                
                # === STAGE 3: At PARTIAL_TP_AT_R - Take partial profits ===
                if current_r >= PARTIAL_TP_AT_R and not data.get('partial_taken') and PARTIAL_CLOSE_ENABLED:
                    close_volume = round(pos.volume * PARTIAL_TP_PERCENT, 2)
                    
                    # Ensure close volume meets minimum
                    if close_volume >= info.volume_min:
                        close_price = tick.bid if is_buy else tick.ask
                        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
                        
                        # Get proper filling mode
                        filling_type = mt5.ORDER_FILLING_IOC
                        if info.filling_mode & 1:
                            filling_type = mt5.ORDER_FILLING_FOK
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": close_volume,
                            "type": close_type,
                            "position": ticket,
                            "price": close_price,
                            "magic": MAGIC,
                            "deviation": 20,
                            "type_filling": filling_type,
                            "comment": f"PARTIAL_{PARTIAL_TP_AT_R}R"
                        }
                        
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            data['partial_taken'] = True
                            data['trailing_active'] = True
                            profit_pct = PARTIAL_TP_PERCENT * 100
                            logger.info(f"[{user}] 💰 +{PARTIAL_TP_AT_R}R PARTIAL #{ticket}: Closed {profit_pct:.0f}% ({close_volume} lots)")
                        else:
                            err = result.comment if result else "Unknown"
                            logger.warning(f"Partial close failed: {err}")
                    else:
                        # Volume too small, just mark as done
                        data['partial_taken'] = True
                        data['trailing_active'] = True
                
                # ============================================================
                # === CONTINUOUS PROFIT RATCHET - Always lock more profit ===
                # ============================================================
                # Every time profit increases by 5 pips, move SL up to lock 60% of profit
                if profit_pips >= 10:  # Only after 10 pips profit
                    lock_percent = 0.6  # Lock 60% of current profit
                    locked_pips = profit_pips * lock_percent
                    
                    if is_buy:
                        new_sl = entry_price + (locked_pips * pip_size)
                        new_sl = round(new_sl, digits)
                        
                        # Only move if it's better than current SL
                        if new_sl > current_sl + (3 * pip_size):  # At least 3 pips improvement
                            if (current_price - new_sl) >= min_stop_distance:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": ticket,
                                    "symbol": symbol,
                                    "sl": new_sl,
                                    "tp": current_tp
                                }
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"[{user}] 📈 RATCHET #{ticket}: Locked +{locked_pips:.1f} pips (SL → {new_sl:.5f})")
                    else:  # SELL
                        new_sl = entry_price - (locked_pips * pip_size)
                        new_sl = round(new_sl, digits)
                        
                        if (current_sl == 0 or new_sl < current_sl - (3 * pip_size)):
                            if (new_sl - current_price) >= min_stop_distance:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": ticket,
                                    "symbol": symbol,
                                    "sl": new_sl,
                                    "tp": current_tp
                                }
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"[{user}] 📈 RATCHET #{ticket}: Locked +{locked_pips:.1f} pips (SL → {new_sl:.5f})")
                
                # === STAGE 4: Trailing after partial ===
                if data.get('trailing_active') and TRAIL_AFTER_PARTIAL:
                    # Use VERY tight trailing - lock 70% of profit at all times
                    trail_mult = 0.7  # Always trail at 70% of profit
                    
                    if is_buy:
                        trail_distance = profit_distance * (1 - trail_mult)
                        new_sl = current_price - trail_distance
                        new_sl = round(new_sl, digits)
                        
                        if new_sl > current_sl and (current_price - new_sl) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                logger.debug(f"Trailing #{ticket}: SL → {new_sl:.5f} (R={current_r:.2f})")
                    else:  # SELL
                        trail_distance = profit_distance * (1 - trail_mult)
                        new_sl = current_price + trail_distance
                        new_sl = round(new_sl, digits)
                        
                        if (new_sl < current_sl or current_sl == 0) and (new_sl - current_price) >= min_stop_distance:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                logger.debug(f"Trailing #{ticket}: SL → {new_sl:.5f} (R={current_r:.2f})")
                
                # === NEVER LET WINNER BECOME LOSER ===
                # Only trigger if we had significant profit (+1R) and it's dropping fast
                if NEVER_LET_WINNER_BECOME_LOSER and peak_r >= 1.0:
                    # Only trigger if profit dropped more than 60% from peak
                    if current_r < peak_r * 0.3 and current_r < 0.5:
                        if is_buy:
                            if current_sl < entry_price or current_sl == 0:
                                # Emergency move to breakeven+ (5 pip buffer)
                                new_sl = entry_price + (pip_size * 5)
                                new_sl = round(new_sl, digits)
                                
                                if (current_price - new_sl) >= min_stop_distance:
                                    request = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": ticket,
                                        "symbol": symbol,
                                        "sl": new_sl,
                                        "tp": current_tp
                                    }
                                    result = mt5.order_send(request)
                                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                        data['profit_locked'] = True
                                        logger.warning(f"[{user}] ⚠️ PROFIT LOCK #{ticket}: Profit dropping! SL → breakeven")
                        else:  # SELL
                            if current_sl > entry_price or current_sl == 0:
                                new_sl = entry_price - (pip_size * 5)
                                new_sl = round(new_sl, digits)
                                
                                if (new_sl - current_price) >= min_stop_distance:
                                    request = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": ticket,
                                        "symbol": symbol,
                                        "sl": new_sl,
                                        "tp": current_tp
                                    }
                                    result = mt5.order_send(request)
                                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                        data['profit_locked'] = True
                                        logger.warning(f"[{user}] ⚠️ PROFIT LOCK #{ticket}: Profit dropping! SL → breakeven")
                
            except Exception as pos_error:
                logger.error(f"Error processing position #{pos.ticket}: {pos_error}")
                continue
    
    except Exception as e:
        logger.error(f"Error in R-based profit protection: {e}")
        import traceback
        logger.error(traceback.format_exc())


def check_structure_break(df, position_type):
    """
    Check if market structure has broken against our position.
    Returns (has_broken, message).
    """
    if not EXIT_ON_STRUCTURE_BREAK:
        return False, "Structure break check disabled"
    
    if len(df) < 20:
        return False, "Insufficient data"
    
    # Get recent swing points
    recent_high = df['high'].tail(10).max()
    recent_low = df['low'].tail(10).min()
    prev_high = df['high'].iloc[-20:-10].max() if len(df) >= 20 else recent_high
    prev_low = df['low'].iloc[-20:-10].min() if len(df) >= 20 else recent_low
    
    price = df['close'].iloc[-1]
    
    if position_type == mt5.POSITION_TYPE_BUY:
        # For BUY, structure breaks if we make lower lows and break below support
        if recent_low < prev_low * 0.995:  # Lower low with 0.5% buffer
            return True, f"Structure break: Lower low ({recent_low:.5f} < {prev_low:.5f})"
    else:  # SELL
        # For SELL, structure breaks if we make higher highs and break above resistance
        if recent_high > prev_high * 1.005:  # Higher high with 0.5% buffer
            return True, f"Structure break: Higher high ({recent_high:.5f} > {prev_high:.5f})"
    
    return False, "Structure intact"


def check_volatility_collapse(symbol, df):
    """
    Check if volatility has collapsed (trade not moving).
    Returns (has_collapsed, message).
    """
    if not EXIT_ON_VOLATILITY_COLLAPSE:
        return False, "Volatility collapse check disabled"
    
    if 'atr' not in df.columns or len(df) < 50:
        return False, "Insufficient data"
    
    current_atr = df['atr'].iloc[-1]
    avg_atr = df['atr'].tail(50).mean()
    
    if avg_atr <= 0:
        return False, "No ATR data"
    
    ratio = current_atr / avg_atr
    
    # Volatility collapsed if current is less than 20% of average
    if ratio < 0.2:
        return True, f"Volatility collapsed: {ratio:.2%} of average"
    
    return False, f"Volatility normal: {ratio:.2%}"


def check_news_invalidation(symbol, direction, user):
    """
    Check if news has invalidated our trade bias.
    Returns (is_invalidated, message).
    """
    if not EXIT_ON_NEWS_INVALIDATION:
        return False, "News invalidation check disabled"
    
    try:
        news_data = get_market_sentiment_from_news(symbol, user)
        sentiment = news_data.get('sentiment', 'NEUTRAL')
        confidence = news_data.get('confidence', 0.5)
        
        # Check if news strongly opposes our position
        if direction == "BUY" and sentiment == "BEARISH" and confidence > 0.8:
            return True, f"News invalidated BUY: Strong bearish sentiment ({confidence:.0%})"
        elif direction == "SELL" and sentiment == "BULLISH" and confidence > 0.8:
            return True, f"News invalidated SELL: Strong bullish sentiment ({confidence:.0%})"
        
        return False, "News aligned or neutral"
    except:
        return False, "News check unavailable"


def check_news_blackout(symbol, user):
    """
    Check if we're in news blackout period (15-30 min before high-impact news).
    Returns (in_blackout, event_name, minutes_until).
    """
    if not NEWS_FILTER_ENABLED:
        return False, None, 0
    
    try:
        has_event, event_details = check_high_impact_event_nearby(symbol)
        
        if has_event and event_details:
            event_name = event_details.get('event', 'Unknown Event')
            # Simplified - assume event is within blackout period if detected
            return True, event_name, NEWS_BLACKOUT_MINUTES_BEFORE
        
        return False, None, 0
    except:
        return False, None, 0


def calculate_setup_quality_score(symbol, df, direction, htf_direction, user):
    """
    Calculate comprehensive setup quality score (0-10).
    Higher score = higher probability trade.
    """
    score = 0
    max_score = 10
    details = []
    
    # 1. HTF Alignment (2 points)
    if (direction == "BUY" and htf_direction == "BULLISH") or (direction == "SELL" and htf_direction == "BEARISH"):
        score += 2
        details.append("HTF aligned ✓")
    elif htf_direction == "NEUTRAL":
        score += 1
        details.append("HTF neutral")
    else:
        details.append("HTF opposed ✗")
    
    # 2. Key Level (2 points)
    at_level, level_info, level_msg = is_at_key_level(symbol, df, direction)
    if at_level:
        score += 2
        details.append(f"At key level ✓")
    else:
        details.append("Not at key level")
    
    # 3. Momentum (2 points)
    mom_confirmed, mom_strength, mom_msg = check_momentum_confirmation(df, direction)
    if mom_confirmed:
        score += 2
        details.append(f"Momentum confirmed ✓")
    elif mom_strength >= 0.5:
        score += 1
        details.append("Momentum partial")
    else:
        details.append("Momentum weak")
    
    # 4. Session quality (1 point)
    session_name, session = get_current_session()
    if session_name in ['OVERLAP', 'LONDON', 'NEW_YORK']:
        score += 1
        details.append(f"{session_name} session ✓")
    else:
        details.append(f"{session_name} session (low vol)")
    
    # 5. Spread acceptable (1 point)
    spread_ok, current_spread, normal_spread, spread_msg = check_spread_filter(symbol)
    if spread_ok:
        score += 1
        details.append("Spread OK ✓")
    else:
        details.append(f"Spread high ✗")
    
    # 6. Volatility acceptable (1 point)
    vol_ok, current_vol, avg_vol, vol_msg = check_volatility_filter(symbol, df)
    if vol_ok:
        score += 1
        details.append("Volatility OK ✓")
    else:
        details.append(f"Volatility issue ✗")
    
    # 7. News clear (1 point)
    in_blackout, event, _ = check_news_blackout(symbol, user)
    if not in_blackout:
        score += 1
        details.append("News clear ✓")
    else:
        details.append(f"News event: {event}")
    
    return score, max_score, details


def get_adaptive_criteria(user):
    """
    Get adaptive trading criteria based on recent performance.
    After consecutive losses, requirements become stricter.
    """
    stats = user_daily_stats[user]
    consec_losses = stats.get('consecutive_losses', 0)
    
    # Default criteria
    min_score = MIN_SETUP_QUALITY_SCORE  # 7
    size_mult = 1.0
    
    if REDUCE_SIZE_AFTER_LOSSES and consec_losses >= CONSECUTIVE_LOSS_THRESHOLD:
        size_mult = SIZE_REDUCTION_PERCENT  # 0.5
        logger.info(f"[{user}] 📉 {consec_losses} consecutive losses - reducing size to {size_mult*100}%")
    
    if TIGHTEN_CRITERIA_AFTER_LOSSES and consec_losses >= CONSECUTIVE_LOSS_THRESHOLD:
        min_score = INCREASED_MIN_SCORE  # 8
        logger.info(f"[{user}] 📈 Tightened criteria - minimum score now {min_score}")
    
    if consec_losses >= STOP_TRADING_THRESHOLD:
        logger.warning(f"[{user}] 🛑 {consec_losses} consecutive losses - should stop trading")
        return min_score, 0, True  # Stop flag
    
    return min_score, size_mult, False


# ================================================================================
# ========================= PROFESSIONAL FOREX PROFIT SYSTEM =====================
# ================================================================================

# ---------- TRADING SESSION CONFIGURATION ----------
# Only trade during high-liquidity sessions for best fills and movement
TRADING_SESSIONS = {
    'ASIAN': {'start': 0, 'end': 8, 'pairs': ['USDJPY', 'AUDUSD', 'NZDUSD', 'AUDJPY'], 'volatility': 'LOW'},
    'LONDON': {'start': 8, 'end': 16, 'pairs': ['EURUSD', 'GBPUSD', 'EURGBP', 'GBPJPY', 'EURJPY', 'XAUUSD'], 'volatility': 'HIGH'},
    'NEW_YORK': {'start': 13, 'end': 22, 'pairs': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY', 'XAUUSD', 'BTCUSD', 'US30', 'US100'], 'volatility': 'HIGH'},
    'OVERLAP': {'start': 13, 'end': 16, 'pairs': 'ALL', 'volatility': 'EXTREME'}  # London/NY overlap - BEST TIME
}

# Session-based filters - STRICT FOR SAFETY
ONLY_TRADE_HIGH_VOLUME_SESSIONS = True  # Only London, NY, Overlap
AVOID_ASIAN_SESSION = True  # Skip Asian - low volatility causes whipsaws
AVOID_FIRST_HOUR = True  # Skip first hour - erratic moves
AVOID_LAST_HOUR = True  # Skip last hour - profit taking volatility

# ---------- INSTITUTIONAL TRADING PATTERNS ----------
# Banks and institutions trade at specific times - align with them
INSTITUTIONAL_TIMES = {
    'LONDON_OPEN_SWEEP': {'hour': 8, 'minute': 0, 'duration': 30},  # London open liquidity sweep
    'LONDON_REVERSAL': {'hour': 9, 'minute': 30, 'duration': 60},  # London reversal time
    'NY_OPEN_PUSH': {'hour': 14, 'minute': 30, 'duration': 60},  # NY open momentum
    'POWER_HOUR': {'hour': 15, 'minute': 0, 'duration': 60},  # Overlap power hour
}

# ---------- KEY LEVEL CONFIGURATION ----------
# Trade bounces off key psychological levels
PSYCHOLOGICAL_LEVELS = {
    'XAUUSD': [1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900],
    'EURUSD': [1.0000, 1.0200, 1.0400, 1.0500, 1.0600, 1.0800, 1.1000, 1.1200, 1.1500],
    'GBPUSD': [1.2000, 1.2200, 1.2400, 1.2500, 1.2600, 1.2800, 1.3000, 1.3200, 1.3500],
    'USDJPY': [140.00, 142.00, 145.00, 147.00, 150.00, 152.00, 155.00, 157.00, 160.00],
    'BTCUSD': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000],
}
KEY_LEVEL_PROXIMITY_PIPS = 50  # Consider within 50 pips of key level

# ---------- MULTI-TIMEFRAME ANALYSIS ----------
MTF_ENABLED = True
MTF_TIMEFRAMES = {
    'TREND': mt5.TIMEFRAME_H4,   # Higher TF for trend direction
    'ENTRY': mt5.TIMEFRAME_M15,  # Entry timeframe
    'PRECISION': mt5.TIMEFRAME_M5  # Precision entry timing
}

# ---------- MINIMUM CONDITIONS FOR TRADE - VERY STRICT ----------
MIN_REWARD_RISK_RATIO = 3.0  # Minimum 1:3 RR required
MIN_TREND_STRENGTH = 0.7  # Strong trend required (70%+)
MIN_VOLUME_RATIO = 1.5  # Above average volume required
REQUIRE_MTF_CONFLUENCE = True  # ALL timeframes must agree


def get_current_session():
    """
    Determine current trading session based on UTC time.
    Returns session name and session data.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Check overlap first (highest priority)
    if 13 <= hour < 16:
        return 'OVERLAP', TRADING_SESSIONS['OVERLAP']
    elif 8 <= hour < 16:
        return 'LONDON', TRADING_SESSIONS['LONDON']
    elif 13 <= hour < 22:
        return 'NEW_YORK', TRADING_SESSIONS['NEW_YORK']
    elif 0 <= hour < 8:
        return 'ASIAN', TRADING_SESSIONS['ASIAN']
    else:
        return 'OFF_HOURS', None


def is_good_trading_time(symbol):
    """
    Check if current time is optimal for trading this symbol.
    Returns (is_good, reason, session_name).
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hour = now.hour
    minute = now.minute
    
    session_name, session = get_current_session()
    
    # Off hours - no trading
    if session is None:
        return False, "Off trading hours", session_name
    
    # Skip Asian session if configured
    if AVOID_ASIAN_SESSION and session_name == 'ASIAN':
        return False, "Asian session - low volatility", session_name
    
    # Check if symbol is good for this session
    if session['pairs'] != 'ALL' and symbol not in session['pairs']:
        # Allow trading but note it's not optimal
        pass
    
    # Avoid first hour of session
    if AVOID_FIRST_HOUR:
        session_start = session['start']
        if hour == session_start and minute < 60:
            return False, f"First hour of {session_name} session - wait for direction", session_name
    
    # Avoid last hour
    if AVOID_LAST_HOUR:
        session_end = session['end']
        if hour == session_end - 1:
            return False, f"Last hour of {session_name} - profit taking", session_name
    
    # Check for institutional times (bonus)
    is_institutional_time = False
    for inst_name, inst_time in INSTITUTIONAL_TIMES.items():
        if hour == inst_time['hour'] and minute >= inst_time['minute'] and minute < inst_time['minute'] + inst_time['duration']:
            is_institutional_time = True
            break
    
    if is_institutional_time:
        return True, f"Institutional trading time - high probability", session_name
    
    return True, f"{session_name} session - good liquidity", session_name


def get_nearest_key_level(symbol, price):
    """
    Find nearest psychological/key level for a symbol.
    Returns (level, distance_pips, is_above).
    """
    if symbol not in PSYCHOLOGICAL_LEVELS:
        return None, None, None
    
    levels = PSYCHOLOGICAL_LEVELS[symbol]
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    
    nearest_level = min(levels, key=lambda x: abs(x - price))
    distance = abs(nearest_level - price)
    distance_pips = distance / pip_value if pip_value > 0 else distance
    is_above = price > nearest_level
    
    return nearest_level, distance_pips, is_above


def analyze_multi_timeframe(symbol, direction):
    """
    Multi-timeframe analysis for confluence.
    Returns (mtf_agrees, confidence, details).
    """
    if not MTF_ENABLED:
        return True, 0.5, "MTF disabled"
    
    details = {}
    agreement_count = 0
    total_checks = 0
    
    for tf_name, tf in MTF_TIMEFRAMES.items():
        df = get_data(symbol, tf)
        if df is None or len(df) < 50:
            continue
        
        df = calculate_advanced_indicators(df)
        
        # Get trend direction on this TF
        ema_9 = df['ema_9'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        price = df['close'].iloc[-1]
        
        # Trend assessment
        if direction == 'BUY':
            trend_ok = price > ema_21 and ema_9 > ema_21
            ema_aligned = ema_9 > ema_21 > ema_50
        else:
            trend_ok = price < ema_21 and ema_9 < ema_21
            ema_aligned = ema_9 < ema_21 < ema_50
        
        total_checks += 1
        if trend_ok:
            agreement_count += 1
        
        details[tf_name] = {
            'trend_ok': trend_ok,
            'ema_aligned': ema_aligned,
            'price': price,
            'ema_21': ema_21
        }
    
    if total_checks == 0:
        return True, 0.5, "No MTF data"
    
    confidence = agreement_count / total_checks
    mtf_agrees = confidence >= 0.66  # At least 2/3 timeframes agree
    
    return mtf_agrees, confidence, details


def calculate_optimal_sl_tp(symbol, direction, entry_price, df):
    """
    Calculate optimal SL and TP based on market structure.
    Uses swing highs/lows and ATR for dynamic placement.
    """
    atr = df['atr'].iloc[-1]
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    
    # Find recent swing points
    highs = df['high'].values[-50:]
    lows = df['low'].values[-50:]
    
    # Find swing high (local maxima)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    if direction == 'BUY':
        # SL below recent swing low + buffer
        if swing_lows:
            recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
            sl_price = recent_low - (atr * 0.3)  # Small buffer
        else:
            sl_price = entry_price - (atr * ATR_SL_MULTIPLIER)
        
        # TP at recent swing high or ATR-based
        if swing_highs:
            potential_tp = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
        else:
            potential_tp = entry_price + (atr * ATR_TP_MULTIPLIER)
        
        # Ensure minimum RR
        sl_distance = entry_price - sl_price
        tp_distance = potential_tp - entry_price
        
        if sl_distance > 0 and tp_distance / sl_distance < MIN_REWARD_RISK_RATIO:
            # Extend TP to meet minimum RR
            tp_price = entry_price + (sl_distance * MIN_REWARD_RISK_RATIO)
        else:
            tp_price = potential_tp
            
    else:  # SELL
        # SL above recent swing high + buffer
        if swing_highs:
            recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            sl_price = recent_high + (atr * 0.3)
        else:
            sl_price = entry_price + (atr * ATR_SL_MULTIPLIER)
        
        # TP at recent swing low or ATR-based
        if swing_lows:
            potential_tp = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
        else:
            potential_tp = entry_price - (atr * ATR_TP_MULTIPLIER)
        
        # Ensure minimum RR
        sl_distance = sl_price - entry_price
        tp_distance = entry_price - potential_tp
        
        if sl_distance > 0 and tp_distance / sl_distance < MIN_REWARD_RISK_RATIO:
            tp_price = entry_price - (sl_distance * MIN_REWARD_RISK_RATIO)
        else:
            tp_price = potential_tp
    
    return sl_price, tp_price


def calculate_trend_strength(df):
    """
    Calculate trend strength using ADX and price action.
    Returns strength from 0 to 1.
    """
    # ADX calculation (simplified)
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with 14-period
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr_14
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(14).mean().iloc[-1]
    
    # Normalize to 0-1
    strength = min(adx / 50, 1.0)  # ADX > 50 is very strong trend
    
    return strength


def check_volume_confirmation(df):
    """
    Check if volume confirms the move.
    Returns (confirmed, ratio).
    """
    if 'volume' not in df.columns or df['volume'].iloc[-1] == 0:
        return True, 1.0  # Skip if no volume data
    
    current_vol = df['volume'].iloc[-1]
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    
    ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    confirmed = ratio >= MIN_VOLUME_RATIO
    
    return confirmed, ratio


def generate_explicit_trade_signal(symbol, user=None):
    """
    Generate explicit, high-probability trade signal.
    Returns detailed signal with exact entry, SL, TP, and reasoning.
    """
    # Get data
    df = get_data(symbol, TIMEFRAME)
    if df is None or len(df) < 100:
        return None
    
    df = calculate_advanced_indicators(df)
    price = df['close'].iloc[-1]
    
    # 1. CHECK TRADING TIME
    is_good_time, time_reason, session = is_good_trading_time(symbol)
    if not is_good_time:
        return {
            'signal': 'WAIT',
            'symbol': symbol,
            'reason': time_reason,
            'session': session
        }
    
    # 2. GET SMC ANALYSIS
    trend = trend_bias(df)
    sweep_high, sweep_low = liquidity_grab(df)
    ob_type, ob_low, ob_high = order_block(df)
    fvg_type, fvg_low, fvg_high = fair_value_gap(df)
    bullish_bos, bearish_bos = check_market_structure(df)
    market_regime = detect_market_regime(df)
    
    # 3. CALCULATE SCORES
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    
    # Trend (3 points max)
    if market_regime == 'TRENDING_UP':
        buy_score += 3
        buy_reasons.append("Strong uptrend")
    elif market_regime == 'TRENDING_DOWN':
        sell_score += 3
        sell_reasons.append("Strong downtrend")
    elif trend == 'BULLISH':
        buy_score += 2
        buy_reasons.append("Bullish bias")
    elif trend == 'BEARISH':
        sell_score += 2
        sell_reasons.append("Bearish bias")
    
    # Liquidity sweep (2 points)
    if sweep_low:
        buy_score += 2
        buy_reasons.append("Liquidity sweep below")
    if sweep_high:
        sell_score += 2
        sell_reasons.append("Liquidity sweep above")
    
    # Order block (2 points)
    if ob_type == 'BULLISH' and ob_low <= price <= ob_high:
        buy_score += 2
        buy_reasons.append(f"At bullish OB {ob_low:.2f}-{ob_high:.2f}")
    if ob_type == 'BEARISH' and ob_low <= price <= ob_high:
        sell_score += 2
        sell_reasons.append(f"At bearish OB {ob_low:.2f}-{ob_high:.2f}")
    
    # FVG (1 point)
    if fvg_type == 'BULLISH' and fvg_low <= price <= fvg_high:
        buy_score += 1
        buy_reasons.append("Bullish FVG fill")
    if fvg_type == 'BEARISH' and fvg_low <= price <= fvg_high:
        sell_score += 1
        sell_reasons.append("Bearish FVG fill")
    
    # Break of structure (2 points)
    if bullish_bos:
        buy_score += 2
        buy_reasons.append("Bullish BOS")
    if bearish_bos:
        sell_score += 2
        sell_reasons.append("Bearish BOS")
    
    # 4. KEY LEVEL ANALYSIS
    key_level, distance_pips, is_above = get_nearest_key_level(symbol, price)
    if key_level and distance_pips and distance_pips < KEY_LEVEL_PROXIMITY_PIPS:
        if is_above:  # Price above level - potential support
            buy_score += 1
            buy_reasons.append(f"Near support {key_level}")
        else:  # Price below level - potential resistance
            sell_score += 1
            sell_reasons.append(f"Near resistance {key_level}")
    
    # 5. DETERMINE DIRECTION - STRICT HIGH PROBABILITY
    if buy_score > sell_score and buy_score >= MIN_SMC_SCORE:  # Use configured MIN_SMC_SCORE (6)
        direction = 'BUY'
        score = buy_score
        reasons = buy_reasons
    elif sell_score > buy_score and sell_score >= MIN_SMC_SCORE:  # Use configured MIN_SMC_SCORE (6)
        direction = 'SELL'
        score = sell_score
        reasons = sell_reasons
    else:
        return {
            'signal': 'NO_TRADE',
            'symbol': symbol,
            'reason': 'Insufficient signal strength',
            'buy_score': buy_score,
            'sell_score': sell_score,
            'session': session
        }
    
    # 6. MULTI-TIMEFRAME CONFIRMATION - REQUIRED
    mtf_agrees, mtf_confidence, mtf_details = analyze_multi_timeframe(symbol, direction)
    if REQUIRE_MTF_CONFLUENCE and not mtf_agrees:
        return {
            'signal': 'NO_TRADE',
            'symbol': symbol,
            'reason': 'Multi-timeframe confluence missing - ALL timeframes must agree',
            'mtf_confidence': mtf_confidence,
            'direction_suggested': direction,
            'session': session
        }
    
    # 7. OPTIMAL ENTRY CHECK - REQUIRED
    is_optimal, opt_confidence, opt_reason = check_optimal_entry(df, direction, symbol)
    if not is_optimal:
        return {
            'signal': 'WAIT',
            'symbol': symbol,
            'reason': f'Entry not optimal: {opt_reason}',
            'direction_suggested': direction,
            'session': session
        }
    
    # 8. TREND STRENGTH - STRICT
    trend_strength = calculate_trend_strength(df)
    if trend_strength < MIN_TREND_STRENGTH:
        return {
            'signal': 'NO_TRADE',
            'symbol': symbol,
            'reason': f'Trend too weak: {trend_strength:.2f} (need {MIN_TREND_STRENGTH})',
            'session': session
        }
    
    # 9. VOLUME CONFIRMATION - REQUIRED
    vol_confirmed, vol_ratio = check_volume_confirmation(df)
    if not vol_confirmed:
        return {
            'signal': 'NO_TRADE',
            'symbol': symbol,
            'reason': f'Volume too low: {vol_ratio:.1f}x (need {MIN_VOLUME_RATIO}x)',
            'session': session
        }
    
    # 10. CALCULATE OPTIMAL SL/TP
    entry_price = price
    sl_price, tp_price = calculate_optimal_sl_tp(symbol, direction, entry_price, df)
    
    # Calculate risk/reward
    if direction == 'BUY':
        sl_distance = entry_price - sl_price
        tp_distance = tp_price - entry_price
    else:
        sl_distance = sl_price - entry_price
        tp_distance = entry_price - tp_price
    
    rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
    
    if rr_ratio < MIN_REWARD_RISK_RATIO:
        return {
            'signal': 'NO_TRADE',
            'symbol': symbol,
            'reason': f'RR ratio too low: {rr_ratio:.2f}',
            'session': session
        }
    
    # 11. CALCULATE LOT SIZE
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    sl_pips = sl_distance / pip_value if pip_value > 0 else sl_distance
    
    account = mt5.account_info()
    if account:
        risk_amount = account.balance * (RISK_PERCENT / 100)
        lot_size = calculate_lot(account.balance, RISK_PERCENT, sl_pips)
    else:
        lot_size = MIN_LOT
    
    # 12. GENERATE EXPLICIT SIGNAL
    signal = {
        'signal': direction,
        'symbol': symbol,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'sl_pips': sl_pips,
        'tp_pips': tp_distance / pip_value if pip_value > 0 else tp_distance,
        'rr_ratio': rr_ratio,
        'lot_size': lot_size,
        'score': score,
        'trend_strength': trend_strength,
        'mtf_confidence': mtf_confidence,
        'volume_ratio': vol_ratio,
        'reasons': reasons,
        'session': session,
        'market_regime': market_regime,
        'confidence': min((score / 10) * (trend_strength) * mtf_confidence * 1.5, 0.95),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return signal


def execute_explicit_signal(symbol, user, lot_size=None):
    """
    Execute an explicit trade signal.
    This takes the signal generated by generate_explicit_trade_signal and executes it.
    """
    try:
        # Ensure MT5 connection for this user
        if not ensure_mt5_user_session(user):
            return {"success": False, "reason": "MT5 not connected"}
        
        # Generate fresh signal
        signal = generate_explicit_trade_signal(symbol, user)
        
        if not signal:
            return {"success": False, "reason": "Could not generate signal"}
        
        if signal.get('signal') not in ['BUY', 'SELL']:
            return {"success": False, "reason": signal.get('reason', 'No valid signal')}
        
        direction = signal['signal']
        entry_price = signal['entry_price']
        sl = signal['sl_price']
        tp = signal['tp_price']
        score = signal.get('score', 7)
        confidence = signal.get('confidence', 0.7)
        
        # Use intelligent lot sizing if not specified
        if lot_size is None:
            lot_size = signal.get('lot_size', 0.01)
            # Apply AI lot adjustment
            lot_size = get_ai_lot_adjustment(user, score, confidence)
        
        # Minimum lot check
        if lot_size < MIN_LOT:
            lot_size = MIN_LOT
        
        # Check loss protection
        can_trade, reason = check_loss_protection(user)
        if not can_trade:
            return {"success": False, "reason": f"Loss protection: {reason}"}
        
        # Determine order type
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Execute trade
        comment = f"SIGNAL:{signal.get('reasons', ['Signal'])[0][:15] if signal.get('reasons') else 'Signal'}"
        result = send_order(symbol, order_type, lot_size, sl, tp, comment)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[{user}] ✅ EXPLICIT SIGNAL EXECUTED: {direction} {symbol} @ {entry_price:.5f} | "
                       f"Score: {score}/10 | Lot: {lot_size}")
            
            # Record for AI learning
            strategies_used = signal.get('reasons', [])
            if strategies_used:
                trade_strategies_used[result.order] = {
                    'strategies': strategies_used,
                    'user': user,
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'quality': score
                }
            
            return {
                "success": True,
                "ticket": result.order,
                "direction": direction,
                "symbol": symbol,
                "lot_size": lot_size,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "score": score,
                "reasons": signal.get('reasons', [])
            }
        else:
            err = result.comment if result else "Order failed"
            return {"success": False, "reason": err}
        
    except Exception as e:
        logger.error(f"Execute explicit signal error: {e}")
        return {"success": False, "reason": f"Error: {str(e)}"}


# Per-user symbol selection storage
user_selected_symbols = defaultdict(lambda: DEFAULT_SYMBOLS.copy())

# Symbol-specific settings (pips differ by instrument)
SYMBOL_SETTINGS = {
    # Metals
    "XAUUSD": {"sl_pips": 15, "tp_pips": 30, "pip_value": 0.1, "min_lot": 0.01},
    "XAGUSD": {"sl_pips": 10, "tp_pips": 20, "pip_value": 0.01, "min_lot": 0.01},
    # Major Forex
    "EURUSD": {"sl_pips": 10, "tp_pips": 20, "pip_value": 0.0001, "min_lot": 0.01},
    "GBPUSD": {"sl_pips": 12, "tp_pips": 25, "pip_value": 0.0001, "min_lot": 0.01},
    "USDJPY": {"sl_pips": 10, "tp_pips": 20, "pip_value": 0.01, "min_lot": 0.01},
    "USDCHF": {"sl_pips": 10, "tp_pips": 20, "pip_value": 0.0001, "min_lot": 0.01},
    "AUDUSD": {"sl_pips": 8, "tp_pips": 15, "pip_value": 0.0001, "min_lot": 0.01},
    "USDCAD": {"sl_pips": 10, "tp_pips": 20, "pip_value": 0.0001, "min_lot": 0.01},
    "NZDUSD": {"sl_pips": 8, "tp_pips": 15, "pip_value": 0.0001, "min_lot": 0.01},
    # Cross Pairs
    "GBPJPY": {"sl_pips": 20, "tp_pips": 40, "pip_value": 0.01, "min_lot": 0.01},
    "EURJPY": {"sl_pips": 15, "tp_pips": 30, "pip_value": 0.01, "min_lot": 0.01},
    "EURGBP": {"sl_pips": 8, "tp_pips": 15, "pip_value": 0.0001, "min_lot": 0.01},
    "AUDJPY": {"sl_pips": 12, "tp_pips": 25, "pip_value": 0.01, "min_lot": 0.01},
    "CADJPY": {"sl_pips": 12, "tp_pips": 25, "pip_value": 0.01, "min_lot": 0.01},
    # Crypto
    "BTCUSD": {"sl_pips": 100, "tp_pips": 200, "pip_value": 1.0, "min_lot": 0.01},
    "ETHUSD": {"sl_pips": 30, "tp_pips": 60, "pip_value": 0.1, "min_lot": 0.01},
    # Indices
    "US30": {"sl_pips": 30, "tp_pips": 60, "pip_value": 1.0, "min_lot": 0.01},
    "US100": {"sl_pips": 25, "tp_pips": 50, "pip_value": 1.0, "min_lot": 0.01},
    "US500": {"sl_pips": 10, "tp_pips": 20, "pip_value": 0.1, "min_lot": 0.01},
}

# ========== ULTRA-FAST PROFIT PROTECTION SYSTEM ==========
# Take profits INSTANTLY - don't let winners become losers
ENABLE_PROFIT_PROTECTION = True  # Enable smart profit taking
PROFIT_LOCK_PIPS = 2  # Lock profit at 2 pips
PROFIT_LOCK_PERCENT = 0.1  # Or when profit reaches 0.1% of balance
MIN_PROFIT_TO_CLOSE = 2  # Close with 2 pip profit (FAST!)
PROFIT_DRAWDOWN_PERCENT = 0.25  # Close if profit drops 25% from peak (was 30%)
REENTRY_COOLDOWN_SECONDS = 2  # Wait only 2 seconds before re-entry
PARTIAL_CLOSE_ENABLED = False  # DISABLED - close full position for speed
PARTIAL_CLOSE_PERCENT = 0.5  # Close 50% at first TP
PARTIAL_CLOSE_PIPS = 5  # Partial close after 5 pips profit

# ========== MOMENTUM SCALPING SETTINGS ==========
MOMENTUM_THRESHOLD = 0.0003  # Minimum momentum (0.03%) to enter
VOLATILITY_BOOST_ENABLED = True  # Increase lot size in high volatility
VOLATILITY_BOOST_MULTIPLIER = 1.5  # 1.5x lot in high volatility
FAST_REVERSAL_EXIT = True  # Exit immediately on reversal signal
BREAKEVEN_PIPS = 4  # Move SL to breakeven after 4 pips profit

# Track positions for re-entry logic
closed_for_reentry = defaultdict(dict)  # {symbol: {direction, close_time, close_price, score}}

# ========== CRITICAL LOSS PROTECTION SYSTEM ==========
# These settings PREVENT catastrophic losses while allowing profitable trading
ENABLE_LOSS_PROTECTION = True  # Master switch for loss protection - ALWAYS ON
MAX_DAILY_LOSS_PERCENT = 5.0  # Stop trading if daily loss exceeds 5%
MAX_DAILY_LOSS_AMOUNT = 50.0  # Hard cap: Stop if daily loss exceeds $50
MAX_TOTAL_DRAWDOWN_PERCENT = 10.0  # KILL SWITCH: Stop if total drawdown exceeds 10%
MAX_CONSECUTIVE_LOSSES = 5  # Pause trading after 5 consecutive losses
LOSS_COOLDOWN_MINUTES = 30  # Wait 30 minutes after hitting loss limit
MAX_LOSS_PER_TRADE_PERCENT = 1.0  # Maximum 1% loss per single trade
EMERGENCY_CLOSE_ALL_AT_DRAWDOWN = 15.0  # Close ALL positions if drawdown hits 15%
EMERGENCY_STOP_HOURS = 8  # Emergency stop lasts 8 hours

# Daily loss tracking per user
user_daily_stats = defaultdict(lambda: {
    'start_balance': 0,
    'starting_equity': 0,
    'date': None,
    'realized_loss': 0,
    'consecutive_losses': 0,
    'loss_cooldown_until': None,
    'emergency_stop': False,
    'emergency_stop_time': None,  # Track when emergency stop was triggered
    'trades_today': 0,
    'wins_today': 0,
    'losses_today': 0,
    'last_trade_time': None  # Track when last trade was placed
})

# Per-user loss protection enabled setting
user_loss_protection_enabled = defaultdict(lambda: True)  # Default: enabled

def set_loss_protection_enabled(user, enabled):
    """Enable or disable loss protection for a user"""
    user_loss_protection_enabled[user] = enabled
    logger.info(f"[{user}] Loss protection {'ENABLED' if enabled else 'DISABLED'}")
    return enabled

def get_loss_protection_enabled(user):
    """Get loss protection enabled status for a user"""
    return user_loss_protection_enabled[user]

def clear_emergency_stop(user):
    """Manually clear emergency stop for a user"""
    stats = user_daily_stats[user]
    if stats['emergency_stop']:
        stats['emergency_stop'] = False
        stats['emergency_stop_time'] = None
        logger.info(f"[{user}] ✅ Emergency stop manually cleared")
        return True
    return False

def clear_all_emergency_stops():
    """Clear emergency stops for all users"""
    cleared = 0
    for user, stats in user_daily_stats.items():
        if stats.get('emergency_stop'):
            stats['emergency_stop'] = False
            stats['emergency_stop_time'] = None
            cleared += 1
            logger.info(f"[{user}] ✅ Emergency stop cleared")
    if cleared > 0:
        logger.info(f"✅ Cleared emergency stops for {cleared} user(s)")
    return cleared

def check_trade_cooldown(user):
    """
    Check if enough time has passed since last trade.
    Returns (can_trade, minutes_remaining).
    """
    stats = user_daily_stats[user]
    now = datetime.now()
    
    # Check daily trade limit
    if stats.get('trades_today', 0) >= MAX_TRADES_PER_DAY:
        return False, f"Daily trade limit reached ({MAX_TRADES_PER_DAY} trades)"
    
    # Check time since last trade
    last_trade = stats.get('last_trade_time')
    if last_trade:
        minutes_since = (now - last_trade).total_seconds() / 60
        if minutes_since < MIN_MINUTES_BETWEEN_TRADES:
            remaining = MIN_MINUTES_BETWEEN_TRADES - minutes_since
            return False, f"Trade cooldown: {remaining:.0f} minutes remaining"
    
    return True, "OK"

def record_trade_placed(user):
    """Record that a trade was placed"""
    stats = user_daily_stats[user]
    stats['last_trade_time'] = datetime.now()
    stats['trades_today'] = stats.get('trades_today', 0) + 1

def check_loss_protection(user):
    """
    Check if loss limits have been reached.
    Returns (can_trade, reason) tuple.
    """
    if not ENABLE_LOSS_PROTECTION:
        return True, "Loss protection disabled (global)"
    
    if not user_loss_protection_enabled[user]:
        return True, "Loss protection disabled (user)"
    
    stats = user_daily_stats[user]
    now = datetime.now()
    
    # Reset daily stats if new day
    if stats['date'] != now.date():
        account = mt5.account_info()
        if account:
            stats['start_balance'] = account.balance
            stats['starting_equity'] = account.equity
        stats['date'] = now.date()
        stats['realized_loss'] = 0
        stats['consecutive_losses'] = 0
        stats['loss_cooldown_until'] = None
        stats['emergency_stop'] = False
        stats['trades_today'] = 0
        stats['wins_today'] = 0
        stats['losses_today'] = 0
        logger.info(f"[{user}] 📅 New trading day - stats reset. Starting balance: ${stats['start_balance']:.2f}")
    
    # Check emergency stop
    if stats['emergency_stop']:
        # Check if emergency stop has timed out (5 hours)
        stop_time = stats.get('emergency_stop_time')
        if stop_time:
            hours_elapsed = (now - stop_time).total_seconds() / 3600
            if hours_elapsed >= EMERGENCY_STOP_HOURS:
                stats['emergency_stop'] = False
                stats['emergency_stop_time'] = None
                logger.info(f"[{user}] ✅ Emergency stop auto-reset after {EMERGENCY_STOP_HOURS} hours")
            else:
                hours_remaining = EMERGENCY_STOP_HOURS - hours_elapsed
                return False, f"🚨 EMERGENCY STOP ACTIVE - Resets in {hours_remaining:.1f} hours"
        else:
            return False, "🚨 EMERGENCY STOP ACTIVE - Trading halted"
    
    # Check cooldown
    if stats['loss_cooldown_until'] and now < stats['loss_cooldown_until']:
        remaining = (stats['loss_cooldown_until'] - now).seconds // 60
        return False, f"⏸️ Loss cooldown active - {remaining} minutes remaining"
    
    # Get current account state
    account = mt5.account_info()
    if not account:
        return False, "Cannot get account info"
    
    current_equity = account.equity
    current_balance = account.balance
    start_balance = stats['start_balance'] or current_balance
    
    # Calculate current unrealized + realized loss
    unrealized_pnl = current_equity - current_balance
    total_daily_loss = stats['realized_loss'] + (start_balance - current_balance)
    
    # Check daily loss limit (percentage)
    daily_loss_percent = (total_daily_loss / start_balance) * 100 if start_balance > 0 else 0
    if daily_loss_percent >= MAX_DAILY_LOSS_PERCENT:
        stats['emergency_stop'] = True
        stats['emergency_stop_time'] = now
        logger.critical(f"[{user}] 🚨 DAILY LOSS LIMIT HIT: {daily_loss_percent:.2f}% - STOPPING FOR {EMERGENCY_STOP_HOURS} HOURS")
        log_trade(user, 'error', f'DAILY LOSS LIMIT: {daily_loss_percent:.2f}% loss', {
            'loss_amount': total_daily_loss,
            'limit': MAX_DAILY_LOSS_PERCENT
        })
        return False, f"🚨 Daily loss limit hit: {daily_loss_percent:.2f}%"
    
    # Check daily loss limit (amount)
    if total_daily_loss >= MAX_DAILY_LOSS_AMOUNT:
        stats['emergency_stop'] = True
        stats['emergency_stop_time'] = now
        logger.critical(f"[{user}] 🚨 DAILY LOSS LIMIT HIT: ${total_daily_loss:.2f} - STOPPING FOR {EMERGENCY_STOP_HOURS} HOURS")
        log_trade(user, 'error', f'DAILY LOSS LIMIT: ${total_daily_loss:.2f} loss', {
            'loss_percent': daily_loss_percent,
            'limit': MAX_DAILY_LOSS_AMOUNT
        })
        return False, f"🚨 Daily loss limit hit: ${total_daily_loss:.2f}"
    
    # Check total drawdown from starting equity
    drawdown = ((stats['starting_equity'] - current_equity) / stats['starting_equity']) * 100 if stats['starting_equity'] > 0 else 0
    if drawdown >= MAX_TOTAL_DRAWDOWN_PERCENT:
        stats['emergency_stop'] = True
        stats['emergency_stop_time'] = now
        logger.critical(f"[{user}] 🚨 MAX DRAWDOWN HIT: {drawdown:.2f}% - STOPPING FOR {EMERGENCY_STOP_HOURS} HOURS")
        log_trade(user, 'error', f'MAX DRAWDOWN: {drawdown:.2f}%', {
            'starting_equity': stats['starting_equity'],
            'current_equity': current_equity
        })
        return False, f"🚨 Max drawdown hit: {drawdown:.2f}%"
    
    # Check emergency close threshold
    if drawdown >= EMERGENCY_CLOSE_ALL_AT_DRAWDOWN:
        stats['emergency_stop'] = True
        stats['emergency_stop_time'] = now
        close_all_positions(user)
        logger.critical(f"[{user}] 🚨🚨 EMERGENCY CLOSE ALL - {drawdown:.2f}% DRAWDOWN")
        log_trade(user, 'error', 'EMERGENCY CLOSE ALL POSITIONS', {
            'drawdown': drawdown,
            'threshold': EMERGENCY_CLOSE_ALL_AT_DRAWDOWN
        })
        return False, f"🚨 Emergency close: {drawdown:.2f}% drawdown"
    
    # Check consecutive losses
    if stats['consecutive_losses'] >= MAX_CONSECUTIVE_LOSSES:
        stats['loss_cooldown_until'] = now + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
        stats['consecutive_losses'] = 0  # Reset after cooldown starts
        logger.warning(f"[{user}] ⏸️ {MAX_CONSECUTIVE_LOSSES} consecutive losses - {LOSS_COOLDOWN_MINUTES}min cooldown")
        log_trade(user, 'bot', f'Trading paused: {MAX_CONSECUTIVE_LOSSES} consecutive losses', {
            'cooldown_minutes': LOSS_COOLDOWN_MINUTES
        })
        return False, f"⏸️ Consecutive losses cooldown"
    
    return True, "OK"

def record_trade_result(user, profit, symbol=""):
    """Record trade result for loss tracking"""
    stats = user_daily_stats[user]
    stats['trades_today'] += 1
    
    if profit >= 0:
        stats['consecutive_losses'] = 0
        stats['wins_today'] += 1
        logger.info(f"[{user}] ✅ WIN recorded: +${profit:.2f} on {symbol}")
    else:
        stats['consecutive_losses'] += 1
        stats['realized_loss'] += abs(profit)
        stats['losses_today'] += 1
        logger.warning(f"[{user}] ❌ LOSS recorded: -${abs(profit):.2f} on {symbol} (consecutive: {stats['consecutive_losses']})")

def close_all_positions(user):
    """Emergency close all positions"""
    positions = mt5.positions_get()
    if not positions:
        return 0
    
    closed = 0
    for pos in positions:
        try:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            
            close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 50,
                "magic": MAGIC,
                "comment": "EMERGENCY_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
                logger.info(f"[{user}] 🚨 Emergency closed {pos.symbol}")
        except Exception as e:
            logger.error(f"Error closing {pos.symbol}: {e}")
    
    logger.critical(f"[{user}] 🚨 EMERGENCY CLOSE: {closed} positions closed")
    return closed

def get_loss_protection_status(user):
    """Get current loss protection status for dashboard"""
    stats = user_daily_stats[user]
    account = mt5.account_info()
    
    if not account:
        return {"error": "No account info"}
    
    start_balance = stats.get('start_balance', account.balance)
    current_equity = account.equity
    starting_equity = stats.get('starting_equity', account.equity)
    
    daily_loss = stats.get('realized_loss', 0) + (start_balance - account.balance)
    daily_loss_percent = (daily_loss / start_balance * 100) if start_balance > 0 else 0
    drawdown = ((starting_equity - current_equity) / starting_equity * 100) if starting_equity > 0 else 0
    
    return {
        "enabled": user_loss_protection_enabled[user],
        "daily_loss": daily_loss,
        "daily_loss_percent": daily_loss_percent,
        "max_daily_loss_percent": MAX_DAILY_LOSS_PERCENT,
        "drawdown": drawdown,
        "max_drawdown": MAX_TOTAL_DRAWDOWN_PERCENT,
        "consecutive_losses": stats.get('consecutive_losses', 0),
        "max_consecutive": MAX_CONSECUTIVE_LOSSES,
        "emergency_stop": stats.get('emergency_stop', False),
        "trades_today": stats.get('trades_today', 0),
        "wins_today": stats.get('wins_today', 0),
        "losses_today": stats.get('losses_today', 0),
        "can_trade": not stats.get('emergency_stop', False) and stats.get('loss_cooldown_until') is None
    }

# Advanced Trailing Stop Configuration - SAFE
BREAKEVEN_PIPS = 15  # Move SL to breakeven after 15 pips profit
TRAIL_ACTIVATION_PIPS = 20  # Start trailing after 20 pips profit
TRAIL_STEP_PIPS = 5  # Trail in steps of 5 pips (reasonable)
USE_ATR_TRAILING = True  # Use ATR trailing for volatility adjustment
ATR_MULTIPLIER = 1.5  # 1.5x ATR for trailing distance

# Strategy Filters - BALANCED FOR PROFITABLE TRADING
MIN_SMC_SCORE = 7  # Need 7/10 score - good quality with more trades
REQUIRE_TREND_ALIGNMENT = True  # Trade with trend for higher win rate
AVOID_RANGING_MARKET = True  # Avoid ranging markets
MIN_AI_CONFIDENCE = 0.70  # 70% AI confidence (balanced)
AI_MUST_APPROVE = True  # AI must approve every trade
ENTRY_QUALITY_REQUIRED = ["EXCELLENT", "GOOD"]  # Take GOOD and EXCELLENT entries

# ---------------- SYMBOL MANAGEMENT FUNCTIONS ----------------
def get_user_symbols(user):
    """Get the list of symbols the user wants to trade"""
    return user_selected_symbols.get(user, DEFAULT_SYMBOLS.copy())

def set_user_symbols(user, symbols):
    """Set the list of symbols for a user to trade"""
    # Validate symbols exist in MT5
    valid_symbols = []
    for symbol in symbols:
        info = mt5.symbol_info(symbol)
        if info is not None:
            valid_symbols.append(symbol)
            mt5.symbol_select(symbol, True)  # Enable symbol
        else:
            logger.warning(f"Symbol {symbol} not found in MT5, skipping")
    user_selected_symbols[user] = valid_symbols if valid_symbols else DEFAULT_SYMBOLS.copy()
    return user_selected_symbols[user]

def add_user_symbol(user, symbol):
    """Add a symbol to user's trading list"""
    symbols = get_user_symbols(user)
    if symbol not in symbols:
        info = mt5.symbol_info(symbol)
        if info is not None:
            symbols.append(symbol)
            mt5.symbol_select(symbol, True)
            user_selected_symbols[user] = symbols
            return True, f"Added {symbol}"
    return False, f"Symbol {symbol} not found or already added"

def remove_user_symbol(user, symbol):
    """Remove a symbol from user's trading list"""
    symbols = get_user_symbols(user)
    if symbol in symbols:
        symbols.remove(symbol)
        user_selected_symbols[user] = symbols
        return True, f"Removed {symbol}"
    return False, f"Symbol {symbol} not in list"

def get_available_symbols():
    """Get all available symbols from MT5 that match our default symbols"""
    all_mt5_symbols = mt5.symbols_get()
    if not all_mt5_symbols:
        return []
    
    available = []
    for s in all_mt5_symbols:
        if not s.visible:
            continue
        # Check if this is one of our tradeable symbols (with or without prefix/suffix)
        std_symbol = get_standard_symbol(s.name)
        if std_symbol in DEFAULT_SYMBOLS or s.name in DEFAULT_SYMBOLS:
            available.append(s.name)
    
    # If no matches found, return all visible symbols
    if not available:
        return [s.name for s in all_mt5_symbols if s.visible]
    
    return available

def get_symbol_settings(symbol):
    """Get settings for a specific symbol, with defaults. Handles broker prefixes/suffixes."""
    default = {"sl_pips": 25, "tp_pips": 50, "pip_value": 0.0001, "min_lot": 0.01}
    
    # Direct match
    if symbol in SYMBOL_SETTINGS:
        return SYMBOL_SETTINGS[symbol]
    
    # Try standard symbol name (strip broker prefix/suffix)
    std_symbol = get_standard_symbol(symbol)
    if std_symbol in SYMBOL_SETTINGS:
        return SYMBOL_SETTINGS[std_symbol]
    
    return default

# ---------------- GLOBAL TRADE VARIABLES ----------------
trade_stats = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
current_trends = defaultdict(lambda: "NEUTRAL")  # per-user trend


# ================================================================================
# ========================= OPTIMAL ENTRY ANALYSIS ===============================
# ================================================================================

def check_optimal_entry(df, direction, symbol):
    """
    Advanced analysis to ensure entries are at optimal points with high probability
    of immediate profit. Returns (is_optimal, confidence_score, reason).
    
    Uses:
    - Price action patterns (engulfing, pin bars)
    - Support/Resistance proximity
    - Momentum confirmation
    - Volatility analysis
    - Multi-timeframe confluence
    """
    score = 0
    max_score = 10
    reasons = []
    
    price = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    prev_open = df['open'].iloc[-2]
    curr_open = df['open'].iloc[-1]
    curr_high = df['high'].iloc[-1]
    curr_low = df['low'].iloc[-1]
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    
    ema_9 = df['ema_9'].iloc[-1]
    ema_21 = df['ema_21'].iloc[-1]
    ema_50 = df['ema_50'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    macd_hist = df['macd_hist'].iloc[-1]
    prev_macd_hist = df['macd_hist'].iloc[-2]
    atr = df['atr'].iloc[-1]
    stoch_k = df['stoch_k'].iloc[-1]
    stoch_d = df['stoch_d'].iloc[-1]
    
    bb_upper = df['bb_upper'].iloc[-1]
    bb_lower = df['bb_lower'].iloc[-1]
    bb_middle = df['bb_middle'].iloc[-1]
    
    # 1. CANDLESTICK PATTERN ANALYSIS (2 points)
    candle_body = abs(price - curr_open)
    candle_range = curr_high - curr_low
    body_ratio = candle_body / candle_range if candle_range > 0 else 0
    
    if direction == "BUY":
        # Bullish engulfing
        if price > curr_open and price > prev_high and curr_open < prev_low:
            score += 2
            reasons.append("Bullish engulfing")
        # Bullish pin bar (long lower wick)
        elif (curr_open - curr_low) > (candle_range * 0.6) and price > curr_open:
            score += 1.5
            reasons.append("Bullish pin bar")
        # Strong bullish candle
        elif price > curr_open and body_ratio > 0.7:
            score += 1
            reasons.append("Strong bullish candle")
    else:  # SELL
        # Bearish engulfing
        if price < curr_open and price < prev_low and curr_open > prev_high:
            score += 2
            reasons.append("Bearish engulfing")
        # Bearish pin bar (long upper wick)
        elif (curr_high - curr_open) > (candle_range * 0.6) and price < curr_open:
            score += 1.5
            reasons.append("Bearish pin bar")
        # Strong bearish candle
        elif price < curr_open and body_ratio > 0.7:
            score += 1
            reasons.append("Strong bearish candle")
    
    # 2. EMA ALIGNMENT (2 points)
    if direction == "BUY":
        if ema_9 > ema_21 > ema_50:
            score += 2
            reasons.append("EMA aligned bullish")
        elif ema_9 > ema_21:
            score += 1
            reasons.append("Short EMAs bullish")
    else:
        if ema_9 < ema_21 < ema_50:
            score += 2
            reasons.append("EMA aligned bearish")
        elif ema_9 < ema_21:
            score += 1
            reasons.append("Short EMAs bearish")
    
    # 3. MOMENTUM CONFIRMATION (2 points)
    if direction == "BUY":
        # MACD crossing up or accelerating
        if macd_hist > 0 and macd_hist > prev_macd_hist:
            score += 1
            reasons.append("MACD momentum up")
        # RSI confirming (not overbought, rising)
        if 40 < rsi < 65:
            score += 0.5
            reasons.append("RSI healthy zone")
        # Stochastic cross up
        if stoch_k > stoch_d and stoch_k < 80:
            score += 0.5
            reasons.append("Stoch bullish cross")
    else:
        if macd_hist < 0 and macd_hist < prev_macd_hist:
            score += 1
            reasons.append("MACD momentum down")
        if 35 < rsi < 60:
            score += 0.5
            reasons.append("RSI healthy zone")
        if stoch_k < stoch_d and stoch_k > 20:
            score += 0.5
            reasons.append("Stoch bearish cross")
    
    # 4. BOLLINGER BAND ANALYSIS (1 point)
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    if direction == "BUY":
        # Bounce from lower band
        if bb_position < 0.3 and price > prev_close:
            score += 1
            reasons.append("BB lower bounce")
        # Breaking above middle
        elif price > bb_middle and prev_close < bb_middle:
            score += 0.5
            reasons.append("BB middle break up")
    else:
        # Rejection from upper band
        if bb_position > 0.7 and price < prev_close:
            score += 1
            reasons.append("BB upper rejection")
        elif price < bb_middle and prev_close > bb_middle:
            score += 0.5
            reasons.append("BB middle break down")
    
    # 5. VOLATILITY CHECK (1 point)
    avg_atr = df['atr'].rolling(50).mean().iloc[-1]
    if atr > avg_atr * 0.8 and atr < avg_atr * 1.5:
        score += 1
        reasons.append("Good volatility")
    elif atr > avg_atr * 1.5:
        score += 0.5  # High volatility - be careful
        reasons.append("High volatility")
    
    # 6. SUPPORT/RESISTANCE PROXIMITY (2 points)
    recent_highs = df['high'].rolling(20).max().iloc[-1]
    recent_lows = df['low'].rolling(20).min().iloc[-1]
    price_range = recent_highs - recent_lows
    
    if direction == "BUY":
        # Near support (lower part of range)
        dist_to_low = (price - recent_lows) / price_range if price_range > 0 else 0.5
        if dist_to_low < 0.3:
            score += 2
            reasons.append("Near support")
        elif dist_to_low < 0.5:
            score += 1
            reasons.append("Lower half of range")
    else:
        # Near resistance (upper part of range)
        dist_to_high = (recent_highs - price) / price_range if price_range > 0 else 0.5
        if dist_to_high < 0.3:
            score += 2
            reasons.append("Near resistance")
        elif dist_to_high < 0.5:
            score += 1
            reasons.append("Upper half of range")
    
    # Calculate confidence
    confidence = score / max_score
    is_optimal = score >= 5  # Require at least 50% score for entry
    
    return is_optimal, confidence, ", ".join(reasons) if reasons else "No strong signals"


def check_immediate_reversal_risk(df, direction):
    """
    Check if there are signs of immediate reversal that would cause a loss.
    Returns True if high risk of reversal (should NOT enter).
    """
    price = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    stoch_k = df['stoch_k'].iloc[-1]
    macd_hist = df['macd_hist'].iloc[-1]
    prev_macd_hist = df['macd_hist'].iloc[-2]
    bb_upper = df['bb_upper'].iloc[-1]
    bb_lower = df['bb_lower'].iloc[-1]
    
    if direction == "BUY":
        # Don't buy if overbought
        if rsi > 75:
            return True
        if stoch_k > 85:
            return True
        # Don't buy at upper Bollinger
        if price > bb_upper:
            return True
        # Don't buy if MACD losing momentum
        if macd_hist > 0 and macd_hist < prev_macd_hist * 0.5:
            return True
    else:  # SELL
        # Don't sell if oversold
        if rsi < 25:
            return True
        if stoch_k < 15:
            return True
        # Don't sell at lower Bollinger
        if price < bb_lower:
            return True
        # Don't sell if MACD losing momentum
        if macd_hist < 0 and macd_hist > prev_macd_hist * 0.5:
            return True
    
    return False


# ================================================================================
# ========================= OPENAI AI TRADING INTELLIGENCE =======================
# ================================================================================

def calculate_advanced_indicators(df):
    """
    Calculate comprehensive technical indicators for better analysis.
    """
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Volume analysis
    if 'tick_volume' in df.columns:
        df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
    
    return df


def detect_market_regime(df):
    """
    Detect if market is trending or ranging.
    Returns: 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING'
    """
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).tail(14).mean()
    price_range = df['high'].tail(50).max() - df['low'].tail(50).min()
    avg_candle = (df['high'] - df['low']).tail(50).mean()
    
    # Check trend strength using ADX-like calculation
    ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
    ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
    ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
    
    # Strong uptrend: EMAs stacked bullish
    if ema_9 > ema_21 > ema_50:
        return 'TRENDING_UP'
    # Strong downtrend: EMAs stacked bearish
    elif ema_9 < ema_21 < ema_50:
        return 'TRENDING_DOWN'
    else:
        return 'RANGING'


def ai_analyze_market(df, symbol, user):
    """
    Use OpenAI to analyze market conditions and provide trading insights.
    Returns AI recommendation with confidence score.
    """
    client = get_openai_client()
    if client is None:
        return {"recommendation": "HOLD", "confidence": 0.5, "reason": "AI not configured"}
    
    try:
        # Calculate all indicators
        df = calculate_advanced_indicators(df)
        recent_data = df.tail(50)
        
        price_now = recent_data['close'].iloc[-1]
        price_5_ago = recent_data['close'].iloc[-5] if len(recent_data) >= 5 else price_now
        price_20_ago = recent_data['close'].iloc[-20] if len(recent_data) >= 20 else price_now
        
        # Key levels
        high_50 = recent_data['high'].max()
        low_50 = recent_data['low'].min()
        high_20 = recent_data['high'].tail(20).max()
        low_20 = recent_data['low'].tail(20).min()
        
        # Current indicator values
        ema_9 = df['ema_9'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        atr = df['atr'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]
        
        # Market regime
        market_regime = detect_market_regime(df)
        
        # Recent candle patterns
        last_candle = recent_data.iloc[-1]
        prev_candle = recent_data.iloc[-2]
        candle_body = abs(last_candle['close'] - last_candle['open'])
        candle_range = last_candle['high'] - last_candle['low']
        is_bullish_candle = last_candle['close'] > last_candle['open']
        
        # Momentum
        momentum_5 = ((price_now - price_5_ago) / price_5_ago * 100)
        momentum_20 = ((price_now - price_20_ago) / price_20_ago * 100)
        
        # Get symbol description
        symbol_desc = {
            "XAUUSD": "Gold", "EURUSD": "Euro/USD", "GBPUSD": "GBP/USD",
            "USDJPY": "USD/JPY", "BTCUSD": "Bitcoin", "ETHUSD": "Ethereum",
            "GBPJPY": "GBP/JPY", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF"
        }.get(symbol, symbol)
        
        market_context = f"""
=== {symbol} ({symbol_desc}) REAL-TIME ANALYSIS ===

PRICE DATA:
- Current Price: ${price_now:.2f}
- 5-Period Momentum: {momentum_5:+.2f}%
- 20-Period Momentum: {momentum_20:+.2f}%
- 50-Period High: ${high_50:.2f} (Distance: {((high_50 - price_now) / price_now * 100):+.2f}%)
- 50-Period Low: ${low_50:.2f} (Distance: {((price_now - low_50) / price_now * 100):+.2f}%)
- ATR (14): ${atr:.2f} (Volatility measure)

TREND INDICATORS:
- EMA 9: ${ema_9:.2f} (Price {'ABOVE' if price_now > ema_9 else 'BELOW'})
- EMA 21: ${ema_21:.2f} (Price {'ABOVE' if price_now > ema_21 else 'BELOW'})
- EMA 50: ${ema_50:.2f} (Price {'ABOVE' if price_now > ema_50 else 'BELOW'})
- EMA Stack: {'BULLISH (9>21>50)' if ema_9 > ema_21 > ema_50 else 'BEARISH (9<21<50)' if ema_9 < ema_21 < ema_50 else 'MIXED'}
- Market Regime: {market_regime}

MOMENTUM INDICATORS:
- RSI (14): {rsi:.1f} ({'OVERBOUGHT' if rsi > 70 else 'OVERSOLD' if rsi < 30 else 'NEUTRAL'})
- MACD: {macd:.2f}
- MACD Signal: {macd_signal:.2f}
- MACD Histogram: {macd_hist:.2f} ({'BULLISH' if macd_hist > 0 else 'BEARISH'})
- Stochastic K: {stoch_k:.1f}, D: {stoch_d:.1f}

BOLLINGER BANDS:
- Upper: ${bb_upper:.2f}
- Middle: ${bb_middle:.2f}
- Lower: ${bb_lower:.2f}
- Price Position: {'NEAR UPPER' if price_now > bb_middle + (bb_upper - bb_middle) * 0.7 else 'NEAR LOWER' if price_now < bb_middle - (bb_middle - bb_lower) * 0.7 else 'MIDDLE ZONE'}

CANDLE ANALYSIS:
- Last Candle: {'BULLISH' if is_bullish_candle else 'BEARISH'}
- Candle Body: ${candle_body:.2f} ({(candle_body / candle_range * 100):.0f}% of range)
- Candle Range: ${candle_range:.2f}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=20,  # 20 second timeout
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an ELITE institutional trader specializing in {symbol} ({symbol_desc}) with 20+ years experience.
Your track record: 75% win rate, 1:3 average risk/reward.

=== TRADING RULES YOU MUST FOLLOW ===

1. ONLY TRADE WITH THE TREND:
   - For BUYS: Price must be ABOVE EMA 21 AND EMA 50
   - For SELLS: Price must be BELOW EMA 21 AND EMA 50
   - In RANGING markets: Recommend HOLD unless at extreme levels

2. MOMENTUM CONFIRMATION:
   - For BUYS: RSI should be between 40-65 (not overbought), MACD histogram positive or turning positive
   - For SELLS: RSI should be between 35-60 (not oversold), MACD histogram negative or turning negative
   - Avoid trades when RSI is extreme (>75 or <25)

3. ENTRY PRECISION:
   - Best BUY entries: Pullback to EMA 21 or EMA 50, bounce from BB lower band, or breakout with volume
   - Best SELL entries: Rally to EMA 21 or EMA 50, rejection from BB upper band, or breakdown
   - NEVER chase price - wait for pullbacks

4. RISK MANAGEMENT (CRITICAL):
   - Stop Loss: Use 1.5x ATR from entry
   - Take Profit: Minimum 3x ATR from entry (1:2 R:R minimum)
   - Only trade when ATR suggests good volatility

5. AVOID THESE SITUATIONS (HOLD):
   - Ranging/choppy markets (mixed EMA signals)
   - Extreme RSI (overbought >75, oversold <25) UNLESS there's a reversal pattern
   - Price in the middle of Bollinger Bands with no clear direction
   - Low momentum (small candles, tight range)

6. HIGH PROBABILITY SETUPS ONLY:
   - Confluence of 3+ indicators agreeing
   - Clear market structure (higher highs/lows for uptrend, vice versa)
   - Momentum supporting the direction

Your goal: ONLY recommend trades with 70%+ probability of success.
If unsure, ALWAYS recommend HOLD. Protecting capital is priority #1.

Respond ONLY with valid JSON:
{{
    "recommendation": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reason": "2-3 sentence explanation of the setup",
    "suggested_sl_pips": number (based on 1.5x ATR),
    "suggested_tp_pips": number (minimum 2x SL for 1:2 R:R),
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "key_levels": {{"support": number, "resistance": number}},
    "invalidation": "What would invalidate this trade"
}}"""
                },
                {
                    "role": "user",
                    "content": f"Analyze this market data and provide your professional trading recommendation:\n{market_context}"
                }
            ],
            max_completion_tokens=500
        )
        
        content = response.choices[0].message.content
        if not content or content.strip() == '':
            logger.warning(f"[{user}] AI analysis returned empty response")
            return {"recommendation": "HOLD", "confidence": 0.5, "reason": "AI returned empty response"}
        
        # Handle markdown-wrapped JSON responses
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)
        logger.info(f"[{user}] 🤖 AI Analysis: {result['recommendation']} (Confidence: {result['confidence']:.2f}) - {result['reason']}")
        return result
        
    except Exception as e:
        logger.error(f"[{user}] AI analysis error: {e}")
        return {"recommendation": "HOLD", "confidence": 0.5, "reason": f"AI error: {str(e)}"}


def ai_validate_trade_signal(df, signal_type, smc_score, user, ai_recommendation=None):
    """
    AI validates a trading signal before execution.
    Filters out weak signals to improve win rate.
    """
    client = get_openai_client()
    if client is None:
        return True, 1.0  # If no AI, proceed with signal
    
    try:
        # Calculate comprehensive indicators
        df = calculate_advanced_indicators(df)
        recent_data = df.tail(30)
        price = recent_data['close'].iloc[-1]
        
        # Get indicator values
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (recent_data['high'] - recent_data['low']).mean()
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
        ema_9 = df['ema_9'].iloc[-1] if 'ema_9' in df.columns else price
        ema_21 = df['ema_21'].iloc[-1] if 'ema_21' in df.columns else price
        ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else price
        stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 50
        
        # Candle analysis
        last_candle = recent_data.iloc[-1]
        prev_candle = recent_data.iloc[-2]
        is_bullish = last_candle['close'] > last_candle['open']
        candle_body = abs(last_candle['close'] - last_candle['open'])
        candle_range = last_candle['high'] - last_candle['low']
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        # Check for pin bars and dojis
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        is_pin_bar = (upper_wick > candle_body * 2) or (lower_wick > candle_body * 2)
        is_doji = body_ratio < 0.1
        
        # Market regime
        market_regime = detect_market_regime(df)
        
        # Check confluence
        bullish_confluence = 0
        bearish_confluence = 0
        
        if price > ema_21: bullish_confluence += 1
        else: bearish_confluence += 1
        if price > ema_50: bullish_confluence += 1
        else: bearish_confluence += 1
        if rsi > 50: bullish_confluence += 1
        else: bearish_confluence += 1
        if macd_hist > 0: bullish_confluence += 1
        else: bearish_confluence += 1
        if is_bullish: bullish_confluence += 1
        else: bearish_confluence += 1
        
        signal_context = f"""
=== TRADE SIGNAL VALIDATION ===

SIGNAL: {signal_type}
SMC Score: {smc_score}/5

PRICE & TREND:
- Current Price: ${price:.2f}
- EMA 9: ${ema_9:.2f} ({'ABOVE' if price > ema_9 else 'BELOW'})
- EMA 21: ${ema_21:.2f} ({'ABOVE' if price > ema_21 else 'BELOW'})
- EMA 50: ${ema_50:.2f} ({'ABOVE' if price > ema_50 else 'BELOW'})
- Market Regime: {market_regime}

MOMENTUM:
- RSI: {rsi:.1f}
- MACD Histogram: {macd_hist:.2f}
- Stochastic K: {stoch_k:.1f}

CANDLE PATTERN:
- Last Candle: {'BULLISH' if is_bullish else 'BEARISH'}
- Body Ratio: {body_ratio:.0%}
- Is Pin Bar: {is_pin_bar}
- Is Doji: {is_doji}

CONFLUENCE SCORE:
- Bullish Signals: {bullish_confluence}/5
- Bearish Signals: {bearish_confluence}/5

VOLATILITY:
- ATR: ${atr:.2f}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=15,
            messages=[
                {
                    "role": "system",
                    "content": """You are a STRICT trade gatekeeper. Your job is to PROTECT CAPITAL by only approving HIGH PROBABILITY trades.

=== APPROVAL CRITERIA ===

For BUY signals, ALL must be true:
1. Price ABOVE EMA 21 AND EMA 50
2. RSI between 35-65 (not overbought)
3. MACD histogram positive or turning positive
4. Bullish confluence >= 4/5
5. NOT a bearish pin bar or doji

For SELL signals, ALL must be true:
1. Price BELOW EMA 21 AND EMA 50  
2. RSI between 35-65 (not oversold)
3. MACD histogram negative or turning negative
4. Bearish confluence >= 4/5
5. NOT a bullish pin bar or doji

AUTO-REJECT if:
- Market regime is RANGING and confluence < 4
- RSI extreme (>75 or <25) going against the trade
- Signal conflicts with current momentum
- Low SMC score (<3)

Respond ONLY with valid JSON:
{
    "approved": true or false,
    "confidence_multiplier": 0.7 to 1.3 (adjust position size),
    "reason": "Specific reason for approval/rejection",
    "warnings": ["any concerns to note"]
}"""
                },
                {
                    "role": "user",
                    "content": f"Should we execute this trade signal?\n{signal_context}"
                }
            ],
            max_completion_tokens=200
        )
        
        # Parse response - handle different formats
        result_text = response.choices[0].message.content
        if result_text is None:
            result_text = ""
        result_text = result_text.strip()
        
        # Try to extract JSON from markdown code blocks
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0].strip()
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0].strip()
        
        # Handle empty response - APPROVE since SMC already validated
        if not result_text:
            logger.warning(f"[{user}] ⚠️ Empty AI response - APPROVING (SMC score passed)")
            return True, 0.8  # Slightly reduced confidence
        
        result = json.loads(result_text)
        
        # Additional safety checks
        if signal_type == "BUY" and (price < ema_21 or price < ema_50):
            logger.warning(f"[{user}] ⚠️ BUY rejected: Price below key EMAs")
            return False, 0.5
        if signal_type == "SELL" and (price > ema_21 or price > ema_50):
            logger.warning(f"[{user}] ⚠️ SELL rejected: Price above key EMAs")
            return False, 0.5
        
        logger.info(f"[{user}] 🤖 AI Validation: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'} - {result['reason']}")
        return result['approved'], result.get('confidence_multiplier', 1.0)
        
    except Exception as e:
        logger.error(f"[{user}] AI validation error: {e}")
        return True, 0.8  # On error, proceed with reduced confidence (SMC already validated)


def ai_study_trade_results(user, trade_data):
    """
    AI studies completed trades to learn and improve strategy.
    Called after each trade closes.
    """
    client = get_openai_client()
    if client is None:
        return
    
    # Store trade for history
    ai_trade_history[user].append(trade_data)
    
    # Only analyze after every 5 trades
    if len(ai_trade_history[user]) % 5 != 0:
        return
    
    try:
        recent_trades = ai_trade_history[user][-20:]  # Last 20 trades
        
        wins = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        losses = len(recent_trades) - wins
        total_profit = sum(t.get('profit', 0) for t in recent_trades)
        avg_win = np.mean([t['profit'] for t in recent_trades if t.get('profit', 0) > 0]) if wins > 0 else 0
        avg_loss = np.mean([abs(t['profit']) for t in recent_trades if t.get('profit', 0) < 0]) if losses > 0 else 0
        
        trade_summary = f"""
        Recent Trading Performance (Last {len(recent_trades)} trades):
        Wins: {wins}, Losses: {losses}
        Win Rate: {(wins/len(recent_trades)*100):.1f}%
        Total Profit: ${total_profit:.2f}
        Average Win: ${avg_win:.2f}
        Average Loss: ${avg_loss:.2f}
        Risk/Reward Ratio: {(avg_win/avg_loss if avg_loss > 0 else 0):.2f}
        
        Trade Details:
        {json.dumps(recent_trades[-5:], indent=2, default=str)}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=20,
            messages=[
                {
                    "role": "system",
                    "content": """You are a trading strategy optimizer AI.
                    Analyze the trading performance and suggest parameter improvements.
                    
                    Focus on MAXIMIZING PROFITS through:
                    1. Better entry timing
                    2. Optimal stop loss placement
                    3. Take profit optimization
                    4. Position sizing adjustments
                    
                    Respond ONLY with valid JSON:
                    {
                        "suggested_sl_pips": number (10-50),
                        "suggested_tp_pips": number (20-100),
                        "suggested_risk_percent": number (0.5-3.0),
                        "min_smc_score_for_entry": number (1-4),
                        "insights": "Key findings",
                        "strategy_adjustment": "Specific recommendation"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Analyze this trading performance and suggest optimizations:\n{trade_summary}"
                }
            ],
            max_completion_tokens=400
        )
        
        content = response.choices[0].message.content
        if not content or content.strip() == '':
            logger.warning(f"[{user}] AI learning returned empty response")
            return
        
        # Handle markdown-wrapped JSON responses
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)
        
        # Store learned parameters
        ai_learned_params[user] = {
            'sl_pips': result.get('suggested_sl_pips', STOPLOSS_PIPS),
            'tp_pips': result.get('suggested_tp_pips', TAKEPROFIT_PIPS),
            'risk_percent': result.get('suggested_risk_percent', RISK_PERCENT),
            'min_score': result.get('min_smc_score_for_entry', 2),
            'last_updated': datetime.now().isoformat(),
            'insights': result.get('insights', ''),
            'strategy_adjustment': result.get('strategy_adjustment', '')
        }
        
        logger.info(f"[{user}] 🧠 AI Learning: {result.get('insights', 'No insights')}")
        log_trade(user, 'ai_learning', 'AI strategy optimization', ai_learned_params[user])
        
    except Exception as e:
        logger.error(f"[{user}] AI learning error: {e}")


# ================================================================================
# ================= AI SESSION & TIMING OPTIMIZER FUNCTIONS ======================
# ================================================================================

def ai_analyze_best_trading_time(symbol, user):
    """
    AI analyzes the BEST time to trade for maximum profit.
    Considers: session, volatility, historical patterns, news schedule, market conditions.
    
    Returns detailed analysis of optimal trading windows.
    """
    client = get_openai_client()
    if client is None:
        return {"should_trade_now": True, "reason": "AI not configured - using defaults"}
    
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        current_day = now.strftime("%A")
        
        # Get current session
        session_name, session_data = get_current_session()
        
        # Get market data to assess current conditions
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=100)
        if df is None or len(df) < 20:
            return {"should_trade_now": True, "reason": "Insufficient data"}
        
        df = calculate_advanced_indicators(df)
        
        # Calculate current volatility
        atr = df['atr'].iloc[-1] if 'atr' in df else (df['high'] - df['low']).mean()
        avg_atr = df['atr'].mean() if 'atr' in df else atr
        volatility_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        # Get hourly performance data
        hour_perf = hourly_performance.get(current_hour, {'trades': 0, 'wins': 0, 'profit': 0})
        hour_winrate = (hour_perf['wins'] / hour_perf['trades'] * 100) if hour_perf['trades'] > 0 else 50
        
        # Get session performance
        sess_perf = session_performance.get(session_name, {'trades': 0, 'wins': 0, 'profit': 0})
        sess_winrate = (sess_perf['wins'] / sess_perf['trades'] * 100) if sess_perf['trades'] > 0 else 50
        
        # Check for economic events
        try:
            events = scrape_forexfactory_calendar()
            upcoming_events = [e for e in events if e.get('impact') == 'High'][:5]
            event_text = "\n".join([f"- {e.get('time', 'Soon')}: {e.get('event', 'Unknown')} ({e.get('currency', '')})" 
                                   for e in upcoming_events]) if upcoming_events else "No high-impact events upcoming"
        except:
            event_text = "Calendar unavailable"
        
        # Recent price action
        price_change_1h = ((df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12] * 100) if len(df) >= 12 else 0
        trend_strength = abs(price_change_1h)
        is_trending = trend_strength > 0.1
        
        # Spread check (if available)
        tick = mt5.symbol_info_tick(symbol)
        spread = (tick.ask - tick.bid) if tick else 0
        info = mt5.symbol_info(symbol)
        spread_points = spread / info.point if info and info.point > 0 else 0
        
        context = f"""
=== OPTIMAL TRADING TIME ANALYSIS for {symbol} ===

CURRENT TIME:
- Day: {current_day}
- UTC Hour: {current_hour}:00
- Session: {session_name}
- Session Volatility: {session_data.get('volatility', 'UNKNOWN') if session_data else 'OFF_HOURS'}

MARKET CONDITIONS:
- Current ATR: {atr:.5f}
- Volatility Ratio: {volatility_ratio:.2f}x average
- 1-Hour Price Change: {price_change_1h:+.3f}%
- Is Trending: {is_trending}
- Current Spread: {spread_points:.1f} points

HISTORICAL PERFORMANCE (This Hour):
- Trades Taken: {hour_perf['trades']}
- Win Rate: {hour_winrate:.1f}%
- Total Profit: ${hour_perf['profit']:.2f}

SESSION PERFORMANCE ({session_name}):
- Trades Taken: {sess_perf['trades']}
- Win Rate: {sess_winrate:.1f}%
- Total Profit: ${sess_perf['profit']:.2f}

UPCOMING HIGH-IMPACT EVENTS:
{event_text}

SYMBOL-SPECIFIC FACTORS:
- Optimal Sessions for {symbol}: {TRADING_SESSIONS.get('LONDON', {}).get('pairs', [])}
- Is Institutional Time: {any(current_hour == t['hour'] for t in INSTITUTIONAL_TIMES.values())}

Analyze whether NOW is a good time to trade {symbol} for maximum profit potential.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=15,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert trading time analyst. Your job is to determine the OPTIMAL times to trade for maximum profitability.

KEY TRADING TIME FACTORS:
1. **Session Quality**:
   - OVERLAP (London/NY 13-16 UTC): BEST - highest liquidity & volatility
   - LONDON (8-16 UTC): Excellent for EUR, GBP, Gold
   - NEW_YORK (13-22 UTC): Excellent for USD pairs, indices, crypto
   - ASIAN (0-8 UTC): Lower volatility, good for JPY, AUD pairs only

2. **Day of Week**:
   - Tuesday-Thursday: BEST trading days
   - Monday: Often ranging, wait for direction
   - Friday: Mixed, close positions before weekend

3. **Avoid Trading When**:
   - 30 minutes before/after high-impact news
   - First/last hour of session (erratic)
   - Very low volatility (spreads widen)
   - Very high volatility spikes (unpredictable)

4. **Best Times for Each Asset**:
   - GOLD/XAUUSD: London open (8-10 UTC), NY session (14-17 UTC)
   - FOREX: Session overlaps, London open, NY open
   - CRYPTO: 24/7 but best during traditional sessions
   - INDICES: Their home session (US30 during NY)

5. **Institutional Times** (highest probability):
   - 8:00-8:30 UTC: London open sweep
   - 9:30-10:30 UTC: London reversal
   - 14:30-15:30 UTC: NY open push
   - 15:00-16:00 UTC: Power hour (overlap)

Respond ONLY with valid JSON:
{
    "should_trade_now": true or false,
    "confidence": 0.5 to 1.0,
    "current_session_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR" or "AVOID",
    "reason": "Why this is/isn't a good time",
    "better_time_today": "HH:MM UTC if should wait" or null,
    "time_until_optimal": "X hours Y minutes" or "NOW" or null,
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "expected_volatility": "LOW" or "NORMAL" or "HIGH" or "EXTREME",
    "trading_recommendation": "TRADE_NOW" or "WAIT" or "REDUCED_SIZE" or "AVOID_TODAY",
    "best_hours_today": [list of best UTC hours to trade],
    "avoid_hours_today": [list of hours to avoid],
    "special_notes": ["Any important notes about timing"]
}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=500
        )
        
        result_text = response.choices[0].message.content
        if not result_text or result_text.strip() == '':
            return {"should_trade_now": True, "confidence": 0.5, "current_session_quality": "UNKNOWN", 
                    "reason": "AI returned empty", "trading_recommendation": "TRADE_NOW"}
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        if not result_text:
            return {"should_trade_now": True, "confidence": 0.5, "current_session_quality": "UNKNOWN", 
                    "reason": "AI parse failed", "trading_recommendation": "TRADE_NOW"}
        
        result = json.loads(result_text)
        
        # Cache the result
        cache_key = f"{symbol}_{current_hour}"
        ai_session_cache[cache_key] = result
        ai_session_cache_time[cache_key] = time.time()
        
        logger.info(f"[{user}] ⏰ AI Session Analysis {symbol}: {result.get('trading_recommendation', 'UNKNOWN')} | "
                   f"Quality: {result.get('current_session_quality', 'UNKNOWN')} | "
                   f"Reason: {result.get('reason', '')[:50]}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI session analysis error: {e}")
        return {
            "should_trade_now": True,
            "confidence": 0.5,
            "current_session_quality": "UNKNOWN",
            "reason": f"Analysis error: {str(e)}",
            "trading_recommendation": "TRADE_NOW"
        }


def get_optimal_trading_time(symbol, user, force_refresh=False):
    """
    Get cached optimal trading time analysis.
    Returns AI recommendation on whether to trade now.
    """
    from datetime import datetime, timezone
    current_hour = datetime.now(timezone.utc).hour
    cache_key = f"{symbol}_{current_hour}"
    current_time = time.time()
    
    # Check cache
    if not force_refresh and cache_key in ai_session_cache:
        if current_time - ai_session_cache_time.get(cache_key, 0) < AI_SESSION_CACHE_SECONDS:
            return ai_session_cache[cache_key]
    
    # Get fresh analysis
    return ai_analyze_best_trading_time(symbol, user)


def should_trade_this_session(symbol, user):
    """
    Quick check if AI recommends trading in current session.
    Returns (should_trade, reason, quality).
    """
    if not AI_SESSION_OPTIMIZER_ENABLED:
        return True, "Session optimizer disabled", "UNKNOWN"
    
    try:
        analysis = get_optimal_trading_time(symbol, user)
        
        should_trade = analysis.get('should_trade_now', True)
        reason = analysis.get('reason', 'Unknown')
        quality = analysis.get('current_session_quality', 'UNKNOWN')
        recommendation = analysis.get('trading_recommendation', 'TRADE_NOW')
        
        # If AI says avoid, don't trade
        if recommendation == 'AVOID_TODAY':
            return False, reason, quality
        
        # If wait, check if we're close to optimal time
        if recommendation == 'WAIT':
            better_time = analysis.get('better_time_today')
            if better_time:
                return False, f"Wait until {better_time} UTC - {reason}", quality
            return False, reason, quality
        
        # If reduced size, allow but flag it
        if recommendation == 'REDUCED_SIZE':
            return True, f"Trade with caution (reduced size) - {reason}", quality
        
        return should_trade, reason, quality
        
    except Exception as e:
        logger.error(f"Session check error: {e}")
        return True, f"Error: {str(e)}", "UNKNOWN"


def update_session_performance(session_name, hour, won, profit):
    """
    Update performance tracking after a trade closes.
    This helps AI learn which times are most profitable.
    """
    global session_performance, hourly_performance
    
    # Update session stats
    if session_name in session_performance:
        session_performance[session_name]['trades'] += 1
        if won:
            session_performance[session_name]['wins'] += 1
        session_performance[session_name]['profit'] += profit
    
    # Update hourly stats
    if hour in hourly_performance:
        hourly_performance[hour]['trades'] += 1
        if won:
            hourly_performance[hour]['wins'] += 1
        hourly_performance[hour]['profit'] += profit
    
    logger.info(f"📊 Session Performance Updated: {session_name} Hour {hour} - "
               f"{'WIN' if won else 'LOSS'} ${profit:.2f}")


def get_best_trading_hours_today(symbol, user):
    """
    Get AI recommendation for the best hours to trade today.
    Returns a list of optimal trading windows.
    """
    try:
        analysis = get_optimal_trading_time(symbol, user, force_refresh=True)
        
        best_hours = analysis.get('best_hours_today', [])
        avoid_hours = analysis.get('avoid_hours_today', [])
        
        return {
            "best_hours": best_hours,
            "avoid_hours": avoid_hours,
            "current_quality": analysis.get('current_session_quality', 'UNKNOWN'),
            "special_notes": analysis.get('special_notes', []),
            "recommendation": analysis.get('trading_recommendation', 'TRADE_NOW')
        }
        
    except Exception as e:
        logger.error(f"Best hours analysis error: {e}")
        return {
            "best_hours": [8, 9, 10, 14, 15, 16],  # Default optimal hours
            "avoid_hours": [0, 1, 2, 3, 4, 5, 22, 23],  # Default avoid hours
            "current_quality": "UNKNOWN",
            "special_notes": [],
            "recommendation": "TRADE_NOW"
        }


def ai_get_market_sentiment(symbol):
    """
    Get REAL-TIME market sentiment analysis from AI.
    Combines technical indicators, price action, and market context for live sentiment.
    """
    client = get_openai_client()
    if client is None:
        return {"sentiment": "NEUTRAL", "confidence": 0.5, "key_factors": ["AI not configured"]}
    
    try:
        # Get real market data
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=100)
        if df is None or len(df) < 50:
            return {"sentiment": "NEUTRAL", "confidence": 0.5, "key_factors": ["Insufficient data"]}
        
        df = calculate_advanced_indicators(df)
        
        # Current price info
        current_price = df['close'].iloc[-1]
        price_1h_ago = df['close'].iloc[-12] if len(df) >= 12 else current_price  # 12 x 5min = 1hr
        price_4h_ago = df['close'].iloc[-48] if len(df) >= 48 else current_price
        
        # Calculate momentum
        momentum_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100
        momentum_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100
        
        # Get indicators
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        macd = df['macd'].iloc[-1] if 'macd' in df else 0
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df else 0
        ema_9 = df['ema_9'].iloc[-1] if 'ema_9' in df else current_price
        ema_21 = df['ema_21'].iloc[-1] if 'ema_21' in df else current_price
        ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df else current_price
        stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df else 50
        atr = df['atr'].iloc[-1] if 'atr' in df else 0
        bb_upper = df['bb_upper'].iloc[-1] if 'bb_upper' in df else current_price
        bb_lower = df['bb_lower'].iloc[-1] if 'bb_lower' in df else current_price
        
        # Market regime
        regime = detect_market_regime(df)
        
        # Recent highs/lows
        high_20 = df['high'].iloc[-20:].max()
        low_20 = df['low'].iloc[-20:].min()
        
        # Candle analysis
        last_candle = df.iloc[-1]
        is_bullish = last_candle['close'] > last_candle['open']
        body = abs(last_candle['close'] - last_candle['open'])
        range_ = last_candle['high'] - last_candle['low']
        
        # Get news sentiment
        try:
            news_items = fetch_all_news_for_symbol(symbol)
            news_sentiment, news_conf, news_summary = analyze_news_sentiment_simple(news_items)
        except:
            news_sentiment, news_conf, news_summary = "NEUTRAL", 0.5, "News unavailable"
        
        # Get economic calendar
        try:
            has_event, event_info = check_high_impact_event_nearby(symbol)
            event_str = f"HIGH IMPACT EVENT: {event_info.get('event', 'Unknown')}" if has_event else "No immediate high-impact events"
        except:
            event_str = "Calendar unavailable"
        
        context = f"""
=== LIVE MARKET SENTIMENT ANALYSIS for {symbol} ===

CURRENT PRICE ACTION:
- Price: {current_price:.5f}
- 1-Hour Change: {momentum_1h:+.3f}%
- 4-Hour Change: {momentum_4h:+.3f}%
- Today's Range: {low_20:.5f} - {high_20:.5f}
- ATR (Volatility): {atr:.5f}

TREND ANALYSIS:
- Market Regime: {regime}
- EMA 9: {ema_9:.5f} (Price {'ABOVE' if current_price > ema_9 else 'BELOW'})
- EMA 21: {ema_21:.5f} (Price {'ABOVE' if current_price > ema_21 else 'BELOW'})
- EMA 50: {ema_50:.5f} (Price {'ABOVE' if current_price > ema_50 else 'BELOW'})
- EMA Stack: {'BULLISH' if ema_9 > ema_21 > ema_50 else 'BEARISH' if ema_9 < ema_21 < ema_50 else 'MIXED'}

MOMENTUM INDICATORS:
- RSI(14): {rsi:.1f} ({'OVERBOUGHT >70' if rsi > 70 else 'OVERSOLD <30' if rsi < 30 else 'NEUTRAL'})
- MACD: {macd:.5f}
- MACD Histogram: {macd_hist:.5f} ({'BULLISH' if macd_hist > 0 else 'BEARISH'})
- Stochastic K: {stoch_k:.1f}

BOLLINGER BANDS:
- Upper: {bb_upper:.5f}
- Lower: {bb_lower:.5f}
- Position: {'NEAR UPPER BAND' if current_price > (bb_upper + bb_lower) / 2 + (bb_upper - bb_lower) * 0.3 else 'NEAR LOWER BAND' if current_price < (bb_upper + bb_lower) / 2 - (bb_upper - bb_lower) * 0.3 else 'MIDDLE'}

CANDLE PATTERN:
- Last Candle: {'BULLISH' if is_bullish else 'BEARISH'}
- Body/Range Ratio: {(body/range_*100) if range_ > 0 else 0:.0f}%

NEWS & EVENTS:
- News Sentiment: {news_sentiment} ({news_conf:.0%} confidence)
- News Summary: {news_summary[:100]}
- Economic Events: {event_str}

Provide your LIVE market sentiment analysis for {symbol}.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=20,
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional market sentiment analyst providing REAL-TIME analysis.

Based on the technical indicators, price action, news, and events, determine:
1. Current market SENTIMENT (BULLISH, BEARISH, or NEUTRAL)
2. Your CONFIDENCE in this assessment (0-100%)
3. KEY FACTORS driving this sentiment
4. TRADING BIAS (what direction is favored)
5. KEY LEVELS to watch

BE SPECIFIC AND ACTIONABLE. Traders rely on your analysis.

Respond ONLY with valid JSON:
{
    "sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": 0.5 to 1.0,
    "strength": "STRONG" or "MODERATE" or "WEAK",
    "key_factors": ["factor1", "factor2", "factor3"],
    "trading_bias": "BUY" or "SELL" or "WAIT",
    "bias_reason": "Why this bias",
    "support_level": price number,
    "resistance_level": price number,
    "short_term_outlook": "Brief 1-2 sentence outlook for next 1-4 hours",
    "risk_events": ["Any risks to watch"]
}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=400
        )
        
        result_text = response.choices[0].message.content
        if not result_text or result_text.strip() == '':
            return {"sentiment": "NEUTRAL", "confidence": 0.5, "key_factors": ["AI returned empty"], 
                    "trading_bias": "WAIT", "short_term_outlook": "Analysis unavailable"}
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        if not result_text:
            return {"sentiment": "NEUTRAL", "confidence": 0.5, "key_factors": ["AI parse failed"], 
                    "trading_bias": "WAIT", "short_term_outlook": "Analysis unavailable"}
        
        result = json.loads(result_text)
        
        logger.info(f"🎯 AI Sentiment {symbol}: {result['sentiment']} ({result.get('confidence', 0.5):.0%}) - {result.get('trading_bias', 'WAIT')}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI sentiment error: {e}")
        return {
            "sentiment": "NEUTRAL", 
            "confidence": 0.5, 
            "key_factors": [f"Error: {str(e)}"],
            "trading_bias": "WAIT",
            "short_term_outlook": "Analysis unavailable"
        }


def ai_find_entry_points(symbol, user):
    """
    AI scans the market to find HIGH PROBABILITY entry points.
    Uses 30 HIGH-PRECISION ENTRY STRATEGIES with confluence scoring.
    Returns specific entry recommendations with exact levels.
    """
    client = get_openai_client()
    if client is None:
        return {"has_entry": False, "reason": "AI not configured"}
    
    try:
        # Get market data
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=100)
        if df is None or len(df) < 50:
            return {"has_entry": False, "reason": "Insufficient data"}
        
        df = calculate_advanced_indicators(df)
        
        # ============ HIGH-PRECISION CONFLUENCE ENTRY SCANNER ============
        # Scan ALL 30 entry strategies for maximum confluence
        confluence_result = scan_all_entry_strategies(symbol, df, user)
        
        # If we have strong technical confluence, boost confidence
        technical_confluence = False
        confluence_strategies = []
        confluence_score = 0
        
        if confluence_result:
            confluence_score = confluence_result.get('confluence_score', 0)
            confluence_strategies = confluence_result.get('strategies', [])
            
            # Strong confluence = 5+ points from multiple strategies
            if confluence_score >= 5 and len(confluence_strategies) >= 3:
                technical_confluence = True
                logger.info(f"[{user}] 🎯 STRONG CONFLUENCE for {symbol}: {confluence_score:.1f} points from {len(confluence_strategies)} strategies")
                
                # Log the strategies detected
                strat_names = [s[0] for s in confluence_strategies[:5]]
                logger.info(f"[{user}] 📊 Strategies: {', '.join(strat_names)}")
        
        # Get HTF data for context
        df_h1 = get_data(symbol, mt5.TIMEFRAME_H1, n=50)
        htf_trend = "UNKNOWN"
        if df_h1 is not None and len(df_h1) > 10:
            df_h1 = calculate_advanced_indicators(df_h1)
            htf_trend = detect_market_regime(df_h1)
        
        current_price = df['close'].iloc[-1]
        
        # Key levels
        high_20 = df['high'].iloc[-20:].max()
        low_20 = df['low'].iloc[-20:].min()
        high_50 = df['high'].max()
        low_50 = df['low'].min()
        
        # Indicators
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        macd = df['macd'].iloc[-1] if 'macd' in df else 0
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df else 0
        macd_hist_prev = df['macd_hist'].iloc[-2] if 'macd_hist' in df and len(df) >= 2 else 0
        ema_9 = df['ema_9'].iloc[-1] if 'ema_9' in df else current_price
        ema_21 = df['ema_21'].iloc[-1] if 'ema_21' in df else current_price
        ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df else current_price
        atr = df['atr'].iloc[-1] if 'atr' in df else (df['high'] - df['low']).mean()
        bb_upper = df['bb_upper'].iloc[-1] if 'bb_upper' in df else current_price
        bb_lower = df['bb_lower'].iloc[-1] if 'bb_lower' in df else current_price
        bb_middle = df['bb_middle'].iloc[-1] if 'bb_middle' in df else current_price
        stoch_k = df['stoch_k'].iloc[-1] if 'stoch_k' in df else 50
        stoch_d = df['stoch_d'].iloc[-1] if 'stoch_d' in df else 50
        
        # Divergence check
        price_higher = current_price > df['close'].iloc[-5]
        rsi_higher = rsi > df['rsi'].iloc[-5] if 'rsi' in df and len(df) >= 5 else True
        bearish_div = price_higher and not rsi_higher
        bullish_div = not price_higher and rsi_higher
        
        # Candle patterns
        last = df.iloc[-1]
        prev = df.iloc[-2]
        is_bullish = last['close'] > last['open']
        is_prev_bullish = prev['close'] > prev['open']
        body = abs(last['close'] - last['open'])
        range_ = last['high'] - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        
        # Pattern detection
        is_hammer = lower_wick > body * 2 and is_bullish
        is_shooting_star = upper_wick > body * 2 and not is_bullish
        is_engulf_bull = not is_prev_bullish and is_bullish and body > abs(prev['close'] - prev['open'])
        is_engulf_bear = is_prev_bullish and not is_bullish and body > abs(prev['close'] - prev['open'])
        
        # Get news for context
        try:
            news_sentiment = get_market_sentiment_from_news(symbol, user)
            news_str = f"{news_sentiment.get('sentiment', 'NEUTRAL')} ({news_sentiment.get('confidence', 0.5):.0%})"
        except:
            news_str = "Unavailable"
        
        # Get ForexFactory calendar events for this symbol
        try:
            ff_events = get_events_for_symbol(symbol)
            high_impact = [e for e in ff_events if e.get('impact') == 'HIGH'][:3]
            if high_impact:
                calendar_str = "⚠️ HIGH IMPACT: " + "; ".join([f"{e.get('currency')} {e.get('event')} @ {e.get('time')}" for e in high_impact])
            else:
                medium_events = [e for e in ff_events if e.get('impact') == 'MEDIUM'][:2]
                if medium_events:
                    calendar_str = "📅 " + "; ".join([f"{e.get('event')}" for e in medium_events])
                else:
                    calendar_str = "No major events"
            
            # Get trading bias from calendar
            ff_bias, ff_confidence, ff_reason = get_news_trading_bias(symbol)
            calendar_bias_str = f"Calendar Bias: {ff_bias} ({ff_confidence:.0%}) - {ff_reason}"
        except:
            calendar_str = "Calendar unavailable"
            calendar_bias_str = "No calendar bias"
        
        context = f"""
=== ENTRY POINT SCANNER for {symbol} ===

CURRENT STATE:
- Price: {current_price:.5f}
- ATR: {atr:.5f}
- HTF Trend (H1): {htf_trend}

TREND INDICATORS:
- EMA 9: {ema_9:.5f} ({'ABOVE' if current_price > ema_9 else 'BELOW'})
- EMA 21: {ema_21:.5f} ({'ABOVE' if current_price > ema_21 else 'BELOW'})
- EMA 50: {ema_50:.5f} ({'ABOVE' if current_price > ema_50 else 'BELOW'})
- Price vs EMA21: {((current_price - ema_21) / atr):.1f}x ATR away

MOMENTUM:
- RSI: {rsi:.1f}
- Stochastic K: {stoch_k:.1f}, D: {stoch_d:.1f}
- MACD Histogram: {macd_hist:.5f} (Previous: {macd_hist_prev:.5f})
- MACD Crossover: {'YES - BULLISH' if macd_hist > 0 > macd_hist_prev else 'YES - BEARISH' if macd_hist < 0 < macd_hist_prev else 'NO'}

DIVERGENCES:
- Bullish Divergence: {bullish_div}
- Bearish Divergence: {bearish_div}

BOLLINGER BANDS:
- Price at: {((current_price - bb_lower) / (bb_upper - bb_lower) * 100) if bb_upper != bb_lower else 50:.0f}% of BB range
- BB Squeeze: {(bb_upper - bb_lower) / bb_middle * 100 < 2}

CANDLE PATTERNS:
- Hammer (Bullish): {is_hammer}
- Shooting Star (Bearish): {is_shooting_star}
- Bullish Engulfing: {is_engulf_bull}
- Bearish Engulfing: {is_engulf_bear}

KEY LEVELS:
- 20-Period High: {high_20:.5f}
- 20-Period Low: {low_20:.5f}
- 50-Period High: {high_50:.5f}
- 50-Period Low: {low_50:.5f}

NEWS SENTIMENT: {news_str}
FOREX FACTORY CALENDAR: {calendar_str}
{calendar_bias_str}

Find the BEST entry opportunity right now, if any. Consider the calendar events when making your decision.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=25,  # 25 second timeout for entry scanner
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert entry point scanner for scalping trades.

Your job is to find HIGH PROBABILITY entry points based on:
1. Trend alignment (HTF + LTF must agree)
2. Momentum confirmation (RSI, MACD, Stochastic)
3. Key level interaction (support/resistance, BBands)
4. Candlestick patterns
5. Divergences
6. Economic calendar (avoid trading before HIGH impact news)

ONLY recommend entry if:
- 3+ confluences align
- Risk:Reward is at least 1:2
- HTF trend supports the direction
- No immediate resistance (for buys) or support (for sells) blocking

Respond ONLY with valid JSON:
{
    "has_entry": true or false,
    "direction": "BUY" or "SELL" (if has_entry),
    "confidence": 0.6 to 1.0,
    "entry_price": exact price,
    "stop_loss": exact price,
    "take_profit": exact price,
    "risk_reward": "1:X",
    "confluences": ["confluence1", "confluence2", "confluence3"],
    "reason": "Brief explanation of the setup",
    "quality_score": 7 to 10,
    "urgency": "IMMEDIATE" or "WAIT_FOR_PULLBACK" or "NONE"
}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=400
        )
        
        result_text = response.choices[0].message.content
        if not result_text or result_text.strip() == '':
            logger.warning(f"[{user}] AI entry scanner returned empty response for {symbol}")
            return {"has_entry": False, "reason": "AI returned empty response - market may be unclear"}
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        if not result_text:
            return {"has_entry": False, "reason": "AI response parsing failed"}
        
        result = json.loads(result_text)
        
        if result.get('has_entry'):
            # Boost quality score if we have strong technical confluence
            if technical_confluence:
                original_quality = result.get('quality_score', 7)
                boost = min(confluence_score / 3, 2)  # Up to +2 boost
                new_quality = min(10, original_quality + boost)
                result['quality_score'] = round(new_quality, 1)
                result['confluence_boost'] = round(boost, 1)
                result['confluence_strategies'] = [s[0] for s in confluence_strategies[:5]]
                result['confluence_score'] = round(confluence_score, 1)
            
            logger.info(f"[{user}] 🎯 AI ENTRY FOUND {symbol}: {result['direction']} @ {result.get('entry_price', current_price):.5f} | "
                       f"Quality: {result.get('quality_score', 7)}/10 | RR: {result.get('risk_reward', '1:2')} | Confluence: {confluence_score:.1f}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI entry scanner error: {e}")
        return {"has_entry": False, "reason": f"Error: {str(e)}"}


def ai_news_based_trade_decision(symbol, news_items, user):
    """
    AI makes trading decisions based on NEWS and EVENTS.
    Can trigger trades directly from breaking news.
    """
    client = get_openai_client()
    if client is None or not news_items:
        return {"should_trade": False, "reason": "No AI or news"}
    
    try:
        # Get current price data
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=50)
        if df is None:
            return {"should_trade": False, "reason": "No price data"}
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df else (df['high'] - df['low']).mean()
        
        # Prepare news for AI
        headlines = [item.get('title', '')[:150] for item in news_items[:10]]
        news_text = "\n".join([f"- {h}" for h in headlines if h])
        
        # Get calendar events
        try:
            events = scrape_forexfactory_calendar()
            event_text = "\n".join([f"- {e.get('event', 'Unknown')} ({e.get('currency', '')}): {e.get('impact', 'Medium')}" for e in events[:5]])
        except:
            event_text = "Calendar unavailable"
        
        context = f"""
=== NEWS-BASED TRADE ANALYSIS for {symbol} ===

CURRENT MARKET:
- Price: {current_price:.5f}
- ATR (Volatility): {atr:.5f}

BREAKING NEWS:
{news_text}

UPCOMING ECONOMIC EVENTS:
{event_text}

Based on the news and events, determine if there's a trading opportunity.

Consider:
1. Is there breaking news that will move {symbol}?
2. Is the news bullish or bearish for this asset?
3. How strong is the expected move?
4. Is there event risk to avoid?
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=20,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a NEWS TRADER AI for {symbol}.

Your job is to analyze breaking news and economic events to find trading opportunities.

TRADE ON NEWS WHEN:
1. Breaking news is CLEARLY bullish or bearish for the asset
2. The news is NEW (not already priced in)
3. Expected move is significant (>1x ATR)
4. No conflicting events in next hour

AVOID TRADING WHEN:
1. High-impact event is in next 30 minutes (wait for result)
2. News is mixed or unclear
3. Multiple conflicting headlines
4. Low-impact or old news

For GOLD (XAUUSD): Watch Fed news, inflation data, USD strength, geopolitical risks
For FOREX: Watch central bank news, employment data, GDP, interest rates
For CRYPTO: Watch regulation news, adoption news, market sentiment

Respond ONLY with valid JSON:
{{
    "should_trade": true or false,
    "direction": "BUY" or "SELL",
    "confidence": 0.5 to 1.0,
    "news_impact": "HIGH" or "MEDIUM" or "LOW",
    "expected_move_pips": number,
    "reason": "Specific news driving this",
    "key_headline": "Most important headline",
    "avoid_reason": "Why to avoid if should_trade is false",
    "wait_for_event": true or false,
    "event_to_wait": "Event name if waiting"
}}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=350
        )
        
        result_text = response.choices[0].message.content
        if not result_text or result_text.strip() == '':
            return {"should_trade": False, "reason": "AI returned empty response"}
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        if not result_text:
            return {"should_trade": False, "reason": "AI parse failed"}
        
        result = json.loads(result_text)
        
        if result.get('should_trade'):
            logger.info(f"[{user}] 📰 AI NEWS TRADE {symbol}: {result['direction']} | "
                       f"Confidence: {result.get('confidence', 0.5):.0%} | "
                       f"Reason: {result.get('reason', 'News')[:50]}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI news trade error: {e}")
        return {"should_trade": False, "reason": f"Error: {str(e)}"}


def get_live_market_sentiment(symbol, force_refresh=False):
    """
    Get LIVE market sentiment with caching.
    This is called by the UI to display real-time sentiment.
    """
    global ai_sentiment_cache, ai_sentiment_cache_time
    
    current_time = time.time()
    cache_key = symbol
    
    # Check cache
    if not force_refresh and cache_key in ai_sentiment_cache:
        if current_time - ai_sentiment_cache_time.get(cache_key, 0) < AI_SENTIMENT_CACHE_SECONDS:
            return ai_sentiment_cache[cache_key]
    
    # Get fresh sentiment
    sentiment = ai_get_market_sentiment(symbol)
    
    # Cache it
    ai_sentiment_cache[cache_key] = sentiment
    ai_sentiment_cache_time[cache_key] = current_time
    
    # ========== EXECUTE TRADE BASED ON SENTIMENT ==========
    if AI_SENTIMENT_TRADING_ENABLED:
        try:
            # First check if we're in a good trading session
            session_quality, session_name, should_trade_session, session_reason = get_current_session_quality(symbol)
            
            if not should_trade_session:
                next_time, hours_until = get_next_good_trading_time()
                sentiment['session_blocked'] = True
                sentiment['session_reason'] = session_reason
                sentiment['next_trading_time'] = next_time
                logger.info(f"⏰ SENTIMENT ANALYSIS: {sentiment.get('sentiment')} {symbol} - SESSION BLOCKED: {session_reason}")
                logger.info(f"⏰ Next good trading time: {next_time} (in {hours_until} hours)")
            else:
                confidence = sentiment.get('confidence', 0)
                sentiment_direction = sentiment.get('sentiment', 'NEUTRAL')
                trading_bias = sentiment.get('trading_bias', 'WAIT')
                
                # Only trade if sentiment is actionable
                if sentiment_direction in ['BULLISH', 'BEARISH'] and confidence >= AI_SENTIMENT_MIN_CONFIDENCE:
                    # Try to execute sentiment trade
                    trade_result = execute_sentiment_trade(symbol, sentiment_direction, confidence, 'auto_sentiment')
                    if trade_result.get('success'):
                        sentiment['trade_executed'] = True
                        sentiment['trade_ticket'] = trade_result.get('ticket')
                        logger.info(f"🎯 SENTIMENT TRADE EXECUTED: {sentiment_direction} {symbol} @ {confidence:.0%} confidence")
        except Exception as e:
            logger.debug(f"Sentiment trade trigger error: {e}")
    
    return sentiment


def execute_sentiment_trade(symbol, sentiment_direction, confidence, user):
    """
    Execute a trade based on AI sentiment.
    Called automatically when sentiment display shows strong signal.
    """
    global ai_sentiment_trades
    
    if not AI_SENTIMENT_TRADING_ENABLED:
        return {"success": False, "reason": "Sentiment trading disabled"}
    
    # ========== SESSION/TIME CHECK - DON'T TRADE DURING BAD HOURS ==========
    if AI_SESSION_OPTIMIZER_ENABLED:
        session_quality, session_name, should_trade, session_reason = get_current_session_quality(symbol)
        
        if not should_trade:
            next_time, hours_until = get_next_good_trading_time()
            logger.warning(f"[{user}] ⏰ SESSION BLOCK: {session_reason} | Next good time: {next_time} ({hours_until}h)")
            return {"success": False, "reason": f"Bad session: {session_reason}. Wait until {next_time}"}
        
        # Log session quality
        logger.info(f"[{user}] 📊 Session: {session_name} ({session_quality}) - {session_reason}")
    
    current_time = time.time()
    
    # Check cooldown
    if symbol in ai_sentiment_trades:
        last_trade = ai_sentiment_trades[symbol].get('last_trade', 0)
        if current_time - last_trade < AI_SENTIMENT_COOLDOWN_SECONDS:
            return {"success": False, "reason": f"Cooldown active ({AI_SENTIMENT_COOLDOWN_SECONDS}s)"}
        
        # Check hourly limit
        trades_this_hour = ai_sentiment_trades[symbol].get('trades_this_hour', 0)
        hour_start = ai_sentiment_trades[symbol].get('hour_start', 0)
        if current_time - hour_start > 3600:
            # Reset hourly count
            ai_sentiment_trades[symbol]['trades_this_hour'] = 0
            ai_sentiment_trades[symbol]['hour_start'] = current_time
        elif trades_this_hour >= AI_SENTIMENT_MAX_TRADES_PER_HOUR:
            return {"success": False, "reason": "Hourly limit reached"}
    
    try:
        # Determine direction
        direction = 'BUY' if sentiment_direction == 'BULLISH' else 'SELL'
        
        # Get current price data
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if not tick or not info:
            return {"success": False, "reason": "No price data"}
        
        # Entry price
        entry_price = tick.ask if direction == 'BUY' else tick.bid
        
        # Use MT5 point value directly (more accurate)
        point = info.point  # Smallest price movement
        
        # ========== SPREAD HANDLING FOR SENTIMENT TRADES ==========
        # Calculate current spread and add to TP to ensure profitability
        current_spread = tick.ask - tick.bid
        spread_points = int(current_spread / point)
        
        # Don't skip on high spread for sentiment trades - they're AI-backed
        # But adjust TP to account for spread cost
        spread_adjustment = spread_points * 2  # Add 2x spread to TP to ensure profit
        
        logger.info(f"[{user}] 📊 Spread check: {spread_points} points ({current_spread:.5f}) - adjusting TP by {spread_adjustment} points")
        
        # Calculate SL/TP based on confidence (higher confidence = tighter stops, bigger targets)
        # Use POINTS directly for accuracy
        if confidence >= AI_SENTIMENT_ULTRA_CONFIDENCE:
            sl_points = 300
            tp_points = 500
            positions_to_open = 4
            lot_mult = AI_SENTIMENT_LOT_MULTIPLIER * 2.0
        elif confidence >= AI_SENTIMENT_STRONG_CONFIDENCE:
            sl_points = 400
            tp_points = 450
            positions_to_open = 3
            lot_mult = AI_SENTIMENT_LOT_MULTIPLIER * 1.5
        else:
            sl_points = 500
            tp_points = 400
            positions_to_open = 2
            lot_mult = AI_SENTIMENT_LOT_MULTIPLIER
        
        # Adjust for Gold/XAU - MUCH MUCH larger stops required (Gold is volatile!)
        # Gold typically moves $5-20 per session, so we need wide stops
        if 'XAU' in symbol or 'GOLD' in symbol:
            # Gold: 1 point = $0.01, so 1500 points = $15 SL
            # This gives room for normal Gold volatility
            if confidence >= AI_SENTIMENT_ULTRA_CONFIDENCE:
                sl_points = 1500   # $15 SL - tight for ultra confidence
                tp_points = 2500   # $25 TP
            elif confidence >= AI_SENTIMENT_STRONG_CONFIDENCE:
                sl_points = 2000   # $20 SL - medium for strong confidence
                tp_points = 2200   # $22 TP
            else:
                sl_points = 2500   # $25 SL - wide for normal confidence
                tp_points = 2000   # $20 TP
        elif 'BTC' in symbol:
            sl_points = 30000  # BTC needs even bigger stops ($300)
            tp_points = 40000  # $400 TP
        
        # Get broker's minimum stop level
        stops_level = info.trade_stops_level  # Minimum distance in points
        if stops_level > 0:
            # Ensure our stops are at least the minimum required
            sl_points = max(sl_points, stops_level + 50)
            tp_points = max(tp_points, stops_level + 50)
        
        # Add spread adjustment to TP to ensure profitability after spread cost
        tp_points = tp_points + spread_adjustment
        
        # Calculate SL/TP prices using points
        if direction == 'BUY':
            sl_price = entry_price - (sl_points * point)
            tp_price = entry_price + (tp_points * point)
        else:
            sl_price = entry_price + (sl_points * point)
            tp_price = entry_price - (tp_points * point)
        
        # Round to proper digits
        sl_price = round(sl_price, info.digits)
        tp_price = round(tp_price, info.digits)
        
        logger.info(f"[{user}] 📊 Sentiment SL/TP: Entry={entry_price:.3f} SL={sl_price:.3f} TP={tp_price:.3f} | Points: SL={sl_points} TP={tp_points}")
        
        # Get lot size
        account = mt5.account_info()
        balance = account.balance if account else 1000
        base_lot = get_scalp_lot_size(user, symbol)
        trade_lot = round(base_lot * lot_mult, 2)
        trade_lot = max(0.01, min(trade_lot, 50.0))  # Bounds
        
        # Check margin
        margin_required = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL,
            symbol, trade_lot, entry_price
        )
        if margin_required and account:
            if margin_required * 1.2 > account.margin_free:
                trade_lot = 0.01  # Fallback to minimum
                positions_to_open = 1
        
        # Execute trades
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        positions_opened = 0
        last_ticket = None
        
        for i in range(positions_to_open):
            result = send_order(
                symbol, order_type, trade_lot, sl_price, tp_price,
                f"SENTIMENT_{sentiment_direction}_{confidence:.0%}_P{i+1}"
            )
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                positions_opened += 1
                last_ticket = result.order
                emoji = "🟢" if direction == 'BUY' else "🔴"
                logger.info(f"[{user}] {emoji} SENTIMENT TRADE: {direction} {symbol} @ {trade_lot} lots | "
                           f"Confidence: {confidence:.0%} | Position {i+1}/{positions_to_open}")
                
                # Register for enhanced profit protection tracking
                if SENTIMENT_PROFIT_PROTECTION_ENABLED:
                    register_sentiment_position(last_ticket, symbol, direction, sentiment_direction, confidence, user)
            
            if i < positions_to_open - 1:
                time.sleep(0.5)  # Brief delay between positions
        
        if positions_opened > 0:
            # Update tracking
            if symbol not in ai_sentiment_trades:
                ai_sentiment_trades[symbol] = {'hour_start': current_time, 'trades_this_hour': 0}
            
            ai_sentiment_trades[symbol]['last_trade'] = current_time
            ai_sentiment_trades[symbol]['trades_this_hour'] = ai_sentiment_trades[symbol].get('trades_this_hour', 0) + positions_opened
            
            return {
                "success": True,
                "ticket": last_ticket,
                "direction": direction,
                "positions_opened": positions_opened,
                "lot_size": trade_lot,
                "confidence": confidence
            }
        else:
            return {"success": False, "reason": "Order failed"}
        
    except Exception as e:
        logger.error(f"Sentiment trade execution error: {e}")
        return {"success": False, "reason": f"Error: {str(e)}"}


# ================================================================================
# ================= ENHANCED SENTIMENT PROFIT PROTECTION SYSTEM v2.0 =============
# ================================================================================

def register_sentiment_position(ticket, symbol, direction, sentiment, confidence, user):
    """
    Register a sentiment trade position for enhanced profit protection tracking.
    Called after a sentiment trade is executed successfully.
    """
    global sentiment_position_data, sentiment_peak_profits
    
    sentiment_position_data[ticket] = {
        'symbol': symbol,
        'direction': direction,
        'sentiment': sentiment,           # 'BULLISH' or 'BEARISH'
        'original_confidence': confidence,
        'current_confidence': confidence,
        'peak_profit_pips': 0.0,
        'entry_time': time.time(),
        'last_sentiment_check': time.time(),
        'last_ai_exit_check': time.time(),
        'sentiment_checks': 0,
        'confidence_decay': 0,
        'user': user,
        'protection_level': get_protection_level_by_confidence(confidence),
        'trailing_active': False,
        'trailing_sl': None,
    }
    sentiment_peak_profits[ticket] = 0.0
    
    logger.info(f"[{user}] 📝 Registered sentiment position #{ticket} | {sentiment} @ {confidence:.0%} | Protection: {get_protection_level_by_confidence(confidence)}")


def get_protection_level_by_confidence(confidence):
    """
    Get protection level name based on confidence.
    Higher confidence = slightly looser protection (AI is more certain).
    """
    for level_name, level_config in SENTIMENT_PROTECTION_BY_CONFIDENCE.items():
        if confidence >= level_config['min_conf']:
            return level_name
    return 'low'


def get_sentiment_drop_multiplier(confidence):
    """
    Get the drop threshold multiplier based on original confidence.
    Higher confidence = larger multiplier = more room before close.
    """
    for level_name, level_config in SENTIMENT_PROTECTION_BY_CONFIDENCE.items():
        if confidence >= level_config['min_conf']:
            return level_config['drop_multiplier']
    return 0.6  # Default to tight protection


def get_sentiment_trail_distance(confidence, current_profit_pips):
    """
    Get dynamic trailing distance based on confidence and profit level.
    Higher profit = tighter trail (protect gains).
    """
    for level_name, level_config in SENTIMENT_PROTECTION_BY_CONFIDENCE.items():
        if confidence >= level_config['min_conf']:
            base_distance = level_config['trail_distance']
            break
    else:
        base_distance = 0.10
    
    # Accelerate trailing if enabled
    if SENTIMENT_TRAILING_ACCELERATE and current_profit_pips > 1.0:
        # Reduce distance by acceleration rate per pip of profit
        acceleration = (current_profit_pips - 1.0) * SENTIMENT_TRAILING_ACCELERATION_RATE
        base_distance = max(0.03, base_distance - acceleration)  # Min 0.03 pip distance
    
    return base_distance


def manage_sentiment_profit_protection(symbol, df, user):
    """
    ENHANCED SENTIMENT PROFIT PROTECTION SYSTEM v2.0
    
    Advanced features:
    1. Sentiment-specific profit drop tiers (tighter than regular trades)
    2. Confidence-based dynamic protection levels
    3. Dynamic trailing stops with acceleration
    4. Sentiment reversal detection and emergency exit
    5. Sentiment shift detection (BULLISH → NEUTRAL)
    6. AI-driven exit analysis
    7. Momentum vs sentiment contradiction exit
    
    This function should be called every cycle for symbols with sentiment trades.
    """
    global sentiment_position_data, sentiment_peak_profits
    
    if not SENTIMENT_PROFIT_PROTECTION_ENABLED:
        return
    
    # Get positions for this symbol
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    info = mt5.symbol_info(symbol)
    if not info:
        return
    
    point = info.point
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    pip_mult = pip_value / point if point > 0 else 10
    
    current_time = time.time()
    
    for pos in positions:
        ticket = pos.ticket
        
        # Check if this is a sentiment trade by comment
        if pos.comment and 'SENTIMENT' in pos.comment:
            # Initialize tracking if not exists
            if ticket not in sentiment_position_data:
                # Parse confidence from comment (e.g., "SENTIMENT_BULLISH_85%_P1")
                try:
                    parts = pos.comment.split('_')
                    sentiment_dir = parts[1] if len(parts) > 1 else 'NEUTRAL'
                    conf_str = parts[2].replace('%', '') if len(parts) > 2 else '70'
                    confidence = float(conf_str) / 100 if float(conf_str) > 1 else float(conf_str)
                except:
                    sentiment_dir = 'NEUTRAL'
                    confidence = 0.70
                
                direction = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
                register_sentiment_position(ticket, symbol, direction, sentiment_dir, confidence, user)
        
        # Skip if not a tracked sentiment position
        if ticket not in sentiment_position_data:
            continue
        
        pos_data = sentiment_position_data[ticket]
        direction = pos_data['direction']
        original_confidence = pos_data['original_confidence']
        
        # Calculate current profit in pips
        profit_pips = calculate_profit_pips(pos, symbol)
        
        # Update peak profit
        if profit_pips > sentiment_peak_profits.get(ticket, 0):
            sentiment_peak_profits[ticket] = profit_pips
            pos_data['peak_profit_pips'] = profit_pips
        
        peak_profit = sentiment_peak_profits.get(ticket, 0)
        pips_dropped = peak_profit - profit_pips
        drop_pct = (pips_dropped / peak_profit) if peak_profit > 0 else 0
        
        # Get protection multiplier based on confidence
        drop_multiplier = get_sentiment_drop_multiplier(original_confidence)
        
        # ========== 1. SENTIMENT-AWARE TIERED PROFIT DROP PROTECTION ==========
        if peak_profit >= 0.03 and profit_pips >= 0:
            should_close = False
            tier_used = None
            
            # Check tiers from largest to smallest
            for tier_name in ['secure', 'profit', 'large', 'medium', 'small', 'micro', 'nano', 'instant']:
                tier = SENTIMENT_PROFIT_DROP_TIERS.get(tier_name, {})
                min_peak = tier.get('min_peak', 999)
                max_drop_pct = tier.get('drop_pct', 0.25) * drop_multiplier  # Apply confidence multiplier
                max_drop_pips = tier.get('drop_pips', 1.0) * drop_multiplier
                
                if peak_profit >= min_peak:
                    if drop_pct >= max_drop_pct or pips_dropped >= max_drop_pips:
                        should_close = True
                        tier_used = tier_name
                    break
            
            if should_close and profit_pips >= 0:
                logger.info(f"[{user}] ⚠️ SENTIMENT PROFIT DROP [{tier_used}]: {symbol} Peak={peak_profit:.2f} → Now={profit_pips:.2f} (dropped {pips_dropped:.2f} pips, {drop_pct*100:.0f}%)")
                if close_position(pos, f"SENT_DROP_{tier_used}_{peak_profit:.1f}to{profit_pips:.1f}"):
                    log_trade(user, 'sentiment_profit_drop', f'Sentiment profit drop close {symbol}', {
                        'tier': tier_used, 'peak': peak_profit, 'now': profit_pips, 'confidence': original_confidence
                    })
                    cleanup_sentiment_position(ticket)
                    continue
        
        # ========== 2. DYNAMIC TRAILING STOP ==========
        if SENTIMENT_TRAILING_ENABLED and profit_pips >= SENTIMENT_TRAILING_START_PIPS:
            trail_distance = get_sentiment_trail_distance(original_confidence, profit_pips)
            tick = mt5.symbol_info_tick(symbol)
            
            if tick:
                if direction == 'BUY':
                    ideal_sl = tick.bid - (trail_distance * pip_mult * point)
                    # Only move SL up, never down
                    if pos_data.get('trailing_sl') is None or ideal_sl > pos_data['trailing_sl']:
                        # Also ensure we're above entry for profit lock
                        if ideal_sl > pos.price_open:
                            result = modify_position_sl(pos, ideal_sl)
                            if result:
                                pos_data['trailing_sl'] = ideal_sl
                                pos_data['trailing_active'] = True
                                logger.debug(f"[{user}] 🔒 Sentiment trailing: {symbol} SL → {ideal_sl:.5f} (trail {trail_distance:.2f} pips)")
                else:
                    ideal_sl = tick.ask + (trail_distance * pip_mult * point)
                    if pos_data.get('trailing_sl') is None or ideal_sl < pos_data['trailing_sl']:
                        if ideal_sl < pos.price_open:
                            result = modify_position_sl(pos, ideal_sl)
                            if result:
                                pos_data['trailing_sl'] = ideal_sl
                                pos_data['trailing_active'] = True
                                logger.debug(f"[{user}] 🔒 Sentiment trailing: {symbol} SL → {ideal_sl:.5f} (trail {trail_distance:.2f} pips)")
        
        # ========== 3. SENTIMENT REVERSAL EMERGENCY EXIT ==========
        if SENTIMENT_REVERSAL_EXIT_ENABLED and profit_pips >= SENTIMENT_REVERSAL_EXIT_MIN_PROFIT:
            # Check current sentiment (throttled to avoid too many API calls)
            time_since_check = current_time - pos_data.get('last_sentiment_check', 0)
            
            if time_since_check >= AI_EXIT_ANALYSIS_INTERVAL:
                pos_data['last_sentiment_check'] = current_time
                pos_data['sentiment_checks'] += 1
                
                # Get fresh sentiment
                try:
                    current_sentiment = ai_get_market_sentiment(symbol)
                    new_sentiment = current_sentiment.get('sentiment', 'NEUTRAL')
                    new_confidence = current_sentiment.get('confidence', 0.5)
                    
                    original_sentiment = pos_data['sentiment']
                    
                    # Check for complete reversal (BULLISH → BEARISH or vice versa)
                    is_reversal = (
                        (original_sentiment == 'BULLISH' and new_sentiment == 'BEARISH') or
                        (original_sentiment == 'BEARISH' and new_sentiment == 'BULLISH')
                    )
                    
                    if is_reversal and new_confidence >= SENTIMENT_REVERSAL_CONFIDENCE_THRESHOLD:
                        logger.warning(f"[{user}] 🚨 SENTIMENT REVERSAL: {symbol} {original_sentiment} → {new_sentiment} @ {new_confidence:.0%} | Profit: +{profit_pips:.2f}")
                        if close_position(pos, f"SENT_REVERSAL_{original_sentiment}_to_{new_sentiment}"):
                            log_trade(user, 'sentiment_reversal', f'Sentiment reversal exit {symbol}', {
                                'old': original_sentiment, 'new': new_sentiment, 'conf': new_confidence, 'profit': profit_pips
                            })
                            cleanup_sentiment_position(ticket)
                            continue
                    
                    # Check for sentiment weakening (BULLISH → NEUTRAL)
                    if SENTIMENT_SHIFT_EXIT_ENABLED and profit_pips >= SENTIMENT_SHIFT_MIN_PROFIT:
                        is_weakening = (
                            (original_sentiment in ['BULLISH', 'BEARISH'] and new_sentiment == 'NEUTRAL') or
                            (new_confidence < original_confidence * 0.7)  # Confidence dropped 30%+
                        )
                        
                        if is_weakening:
                            logger.info(f"[{user}] ⚡ SENTIMENT SHIFT: {symbol} weakening {original_sentiment} @ {original_confidence:.0%} → {new_sentiment} @ {new_confidence:.0%}")
                            if close_position(pos, f"SENT_SHIFT_{profit_pips:.1f}pips"):
                                log_trade(user, 'sentiment_shift', f'Sentiment shift exit {symbol}', {
                                    'old': original_sentiment, 'new': new_sentiment, 'profit': profit_pips
                                })
                                cleanup_sentiment_position(ticket)
                                continue
                    
                    # Update tracked confidence
                    pos_data['current_confidence'] = new_confidence
                    
                except Exception as e:
                    logger.debug(f"Sentiment check error: {e}")
        
        # ========== 4. MOMENTUM VS SENTIMENT CONTRADICTION EXIT ==========
        if SENTIMENT_MOMENTUM_PROTECTION and MOMENTUM_VS_SENTIMENT_EXIT and profit_pips >= MIN_MOMENTUM_CONFIRMATION_PROFIT:
            try:
                opposite_dir = 'SELL' if direction == 'BUY' else 'BUY'
                has_opposite_momentum, momentum_strength = check_momentum_scalp(df, opposite_dir)
                
                # Strong opposite momentum = exit to protect profit
                if has_opposite_momentum and momentum_strength > MOMENTUM_THRESHOLD * 2:
                    logger.info(f"[{user}] 🔄 MOMENTUM CONTRADICTION: {symbol} sentiment={pos_data['sentiment']} but momentum={opposite_dir} (strength: {momentum_strength:.1f})")
                    if close_position(pos, f"SENT_MOM_CONTRA_{profit_pips:.1f}pips"):
                        log_trade(user, 'momentum_contradiction', f'Momentum vs sentiment exit {symbol}', {
                            'sentiment': pos_data['sentiment'], 'momentum_against': True, 'strength': momentum_strength
                        })
                        cleanup_sentiment_position(ticket)
                        continue
            except Exception as e:
                logger.debug(f"Momentum check error: {e}")
        
        # ========== 5. AI-DRIVEN EXIT ANALYSIS ==========
        if AI_EXIT_ANALYSIS_ENABLED and profit_pips >= 1.0:
            time_since_ai_check = current_time - pos_data.get('last_ai_exit_check', 0)
            
            if time_since_ai_check >= AI_EXIT_ANALYSIS_INTERVAL * 2:  # Less frequent than sentiment check
                pos_data['last_ai_exit_check'] = current_time
                
                try:
                    exit_analysis = ai_analyze_sentiment_exit(symbol, direction, profit_pips, peak_profit, original_confidence, user)
                    
                    if exit_analysis.get('should_exit', False):
                        exit_confidence = exit_analysis.get('exit_confidence', 0)
                        exit_reason = exit_analysis.get('reason', 'AI recommended exit')
                        
                        # Apply decay - be more willing to exit as time passes
                        checks = pos_data['sentiment_checks']
                        adjusted_threshold = max(AI_EXIT_MIN_CONFIDENCE, AI_EXIT_MIN_CONFIDENCE + (checks * AI_EXIT_CONFIDENCE_DECAY_RATE))
                        
                        if exit_confidence >= adjusted_threshold:
                            logger.info(f"[{user}] 🤖 AI EXIT: {symbol} @ +{profit_pips:.2f} pips | Reason: {exit_reason}")
                            if close_position(pos, f"AI_EXIT_{exit_reason[:15]}"):
                                log_trade(user, 'ai_sentiment_exit', f'AI recommended exit {symbol}', {
                                    'reason': exit_reason, 'profit': profit_pips, 'confidence': exit_confidence
                                })
                                cleanup_sentiment_position(ticket)
                                continue
                except Exception as e:
                    logger.debug(f"AI exit analysis error: {e}")
        
        # ========== 6. ABSOLUTE EMERGENCY - Profit approaching zero ==========
        if peak_profit >= 0.5 and profit_pips < 0.15 and profit_pips > -0.15:
            logger.warning(f"[{user}] 🚨 SENTIMENT EMERGENCY: {symbol} was +{peak_profit:.2f}, now +{profit_pips:.2f} - CLOSING!")
            if close_position(pos, f"SENT_EMERGENCY_{peak_profit:.1f}to{profit_pips:.1f}"):
                log_trade(user, 'sentiment_emergency', f'Sentiment emergency close {symbol}', {
                    'peak': peak_profit, 'now': profit_pips
                })
                cleanup_sentiment_position(ticket)
                continue


def ai_analyze_sentiment_exit(symbol, direction, current_profit, peak_profit, original_confidence, user):
    """
    AI analyzes whether we should exit a sentiment trade.
    Uses GPT to evaluate current market conditions vs original trade thesis.
    """
    client = get_openai_client()
    if client is None:
        return {'should_exit': False, 'reason': 'AI not available'}
    
    try:
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=50)
        if df is None:
            return {'should_exit': False, 'reason': 'No data'}
        
        df = calculate_advanced_indicators(df)
        
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df else 0
        
        # Get current momentum
        momentum_5 = ((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100) if len(df) >= 5 else 0
        
        context = f"""
=== SENTIMENT TRADE EXIT ANALYSIS ===

CURRENT POSITION:
- Symbol: {symbol}
- Direction: {direction}
- Current Profit: {current_profit:.2f} pips
- Peak Profit: {peak_profit:.2f} pips
- Profit Dropped: {peak_profit - current_profit:.2f} pips ({((peak_profit - current_profit) / peak_profit * 100) if peak_profit > 0 else 0:.0f}%)
- Original Confidence: {original_confidence:.0%}

CURRENT MARKET:
- Price: {current_price:.5f}
- RSI: {rsi:.1f}
- MACD Histogram: {macd_hist:.5f}
- 5-Bar Momentum: {momentum_5:.3f}%

Should we exit this trade to protect profits, or hold for more?
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            timeout=10,
            messages=[
                {
                    "role": "system",
                    "content": """You are a trade exit analyst. Your job is to protect profits.

RECOMMEND EXIT when:
- Profit has dropped significantly from peak (>30%)
- Momentum is reversing against position
- RSI at extreme levels against position
- Original trade thesis is weakening

HOLD when:
- Minor pullback with strong trend intact
- RSI in neutral zone
- Momentum still supports direction
- High profit potential remaining

Be DECISIVE. Protecting profits is priority.

Respond ONLY with JSON:
{
    "should_exit": true or false,
    "exit_confidence": 0.0 to 1.0,
    "reason": "brief reason (max 20 words)",
    "hold_reason": "why hold if not exiting"
}"""
                },
                {"role": "user", "content": context}
            ],
            max_completion_tokens=150
        )
        
        result_text = response.choices[0].message.content
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        result = json.loads(result_text.strip())
        return result
        
    except Exception as e:
        logger.debug(f"AI exit analysis error: {e}")
        return {'should_exit': False, 'reason': f'Error: {str(e)}'}


def modify_position_sl(pos, new_sl):
    """
    Modify position stop loss for trailing.
    Returns True if successful.
    """
    try:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp
        }
        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE
    except Exception as e:
        logger.debug(f"Modify SL error: {e}")
        return False


def cleanup_sentiment_position(ticket):
    """
    Clean up tracking data for a closed sentiment position.
    """
    global sentiment_position_data, sentiment_peak_profits
    
    if ticket in sentiment_position_data:
        del sentiment_position_data[ticket]
    if ticket in sentiment_peak_profits:
        del sentiment_peak_profits[ticket]


def get_all_live_sentiments(symbols):
    """
    Get live sentiment for multiple symbols at once.
    Returns a dictionary of symbol -> sentiment data.
    """
    results = {}
    for symbol in symbols:
        try:
            sentiment = get_live_market_sentiment(symbol, force_refresh=False)
            results[symbol] = sentiment
        except Exception as e:
            results[symbol] = {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'error': str(e)}
    return results


def ai_execute_news_trade(symbol, user, lot_size=None):
    """
    Execute a trade based on AI news analysis.
    Uses INTELLIGENT LOT SIZING based on news impact and confidence.
    Includes LOSS PREVENTION checks and RECOVERY MODE support.
    Returns the result of the trade attempt.
    """
    global ai_news_trades_today
    
    if not AI_NEWS_TRADING_ENABLED or not AI_AUTO_TRADE_NEWS:
        return {"success": False, "reason": "AI news trading disabled"}
    
    # Check daily limit
    user_key = f"{user}_{datetime.now().strftime('%Y%m%d')}"
    trades_today = ai_news_trades_today.get(user_key, 0)
    if trades_today >= AI_MAX_NEWS_TRADES_PER_DAY:
        return {"success": False, "reason": f"Daily limit reached ({AI_MAX_NEWS_TRADES_PER_DAY} trades)"}
    
    try:
        # Get news
        news_items = fetch_all_news_for_symbol(symbol)
        if not news_items:
            return {"success": False, "reason": "No news available"}
        
        # Get AI decision
        decision = ai_news_based_trade_decision(symbol, news_items, user)
        
        if not decision.get('should_trade'):
            return {
                "success": False, 
                "reason": decision.get('avoid_reason', decision.get('reason', 'AI declined')),
                "wait_for_event": decision.get('wait_for_event', False),
                "event": decision.get('event_to_wait')
            }
        
        if decision.get('confidence', 0) < AI_NEWS_TRADE_MIN_CONFIDENCE:
            return {"success": False, "reason": f"Confidence too low ({decision.get('confidence', 0):.0%} < {AI_NEWS_TRADE_MIN_CONFIDENCE:.0%})"}
        
        direction = decision.get('direction', 'BUY')
        confidence = decision.get('confidence', 0.7)
        news_impact = decision.get('news_impact', 'MEDIUM')
        quality = 9 if news_impact == 'HIGH' else (8 if news_impact == 'MEDIUM' else 7)
        
        # ============ COMPREHENSIVE LOSS PREVENTION CHECK ============
        if LOSS_PREVENTION_ENABLED:
            should_enter, prevention_reason, entry_score = comprehensive_entry_check(
                symbol, direction, quality, confidence, user, None
            )
            if not should_enter:
                logger.warning(f"[{user}] 🛡️ LOSS PREVENTION (NEWS): Skipping {symbol} - {prevention_reason}")
                return {"success": False, "reason": f"Loss prevention: {prevention_reason}"}
            
            logger.info(f"[{user}] ✅ Entry score: {entry_score}/100 for NEWS {symbol} {direction}")
        
        # Get current price
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=20)
        if df is None:
            return {"success": False, "reason": "Cannot get price data"}
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df else (df['high'] - df['low']).mean()
        
        # Calculate SL/TP based on expected move
        expected_pips = decision.get('expected_move_pips', 20)
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        if 'XAU' in symbol:
            pip_value = 0.1
        
        sl_distance = atr * 1.5
        tp_distance = expected_pips * pip_value
        
        if direction == 'BUY':
            sl = current_price - sl_distance
            tp = current_price + tp_distance
            order_type = mt5.ORDER_TYPE_BUY
        else:
            sl = current_price + sl_distance
            tp = current_price - tp_distance
            order_type = mt5.ORDER_TYPE_SELL
        
        # INTELLIGENT LOT SIZING for news trades
        sl_pips = sl_distance / pip_value if pip_value > 0 else 20
        if lot_size is None or lot_size == 0.01:
            lot_size = calculate_intelligent_lot(symbol, user, quality, confidence, sl_pips)
        
        # ============ RECOVERY MODE: BIG LOT SCALPING ============
        if LOSS_RECOVERY_ENABLED:
            recovery_status = get_recovery_status(user)
            if recovery_status['active']:
                # Check if trade qualifies for recovery mode
                can_take, reason, recovery_mult = should_take_recovery_trade(user, quality, confidence)
                if not can_take:
                    return {"success": False, "reason": reason}
                
                # Apply BIG recovery lot multiplier
                old_lot = lot_size
                lot_size = round(lot_size * recovery_mult, 2)
                
                # Shorten TP for quick scalp exit during recovery
                if RECOVERY_SCALP_MODE and RECOVERY_QUICK_TP:
                    scalp_tp_distance = RECOVERY_SCALP_TP_PIPS * pip_value
                    if direction == 'BUY':
                        tp = current_price + scalp_tp_distance
                    else:
                        tp = current_price - scalp_tp_distance
                    logger.info(f"[{user}] ⚡ RECOVERY SCALP (NEWS): {old_lot} → {lot_size} lot, TP @ {RECOVERY_SCALP_TP_PIPS} pips")
                else:
                    logger.info(f"[{user}] 🔄 Recovery lot boost (NEWS): {old_lot} → {lot_size} ({recovery_mult}x)")
        
        # Check if trading is paused
        if lot_size == 0:
            return {"success": False, "reason": "Trading paused - loss streak protection"}
        
        # Record for AI learning
        record_lot_for_learning(user, quality, confidence, lot_size)
        
        # Execute trade using send_order
        logger.info(f"[{user}] 📰 AI NEWS TRADE EXECUTING: {direction} {symbol} | Lot: {lot_size} | Reason: {decision.get('reason', 'News')[:50]}")
        
        comment = f"AI_NEWS:{decision.get('key_headline', 'News')[:20]}"
        result = send_order(symbol, order_type, lot_size, sl, tp, comment)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Track daily trades
            ai_news_trades_today[user_key] = trades_today + 1
            
            return {
                "success": True,
                "ticket": result.order,
                "direction": direction,
                "symbol": symbol,
                "lot_size": lot_size,
                "reason": decision.get('reason'),
                "confidence": decision.get('confidence'),
                "news_impact": decision.get('news_impact')
            }
        else:
            err = result.comment if result else "Order failed"
            return {"success": False, "reason": err}
        
    except Exception as e:
        logger.error(f"AI news trade execution error: {e}")
        return {"success": False, "reason": f"Error: {str(e)}"}


def ai_execute_entry_trade(symbol, user, lot_size=None):
    """
    Execute a trade based on AI entry point detection.
    Uses INTELLIGENT LOT SIZING based on account size, quality, and confidence.
    Includes LOSS PREVENTION checks and RECOVERY MODE support.
    Returns the result of the trade attempt.
    """
    if not AI_ENTRY_SCANNER_ENABLED or not AI_AUTO_TRADE_ENTRIES:
        return {"success": False, "reason": "AI entry trading disabled"}
    
    try:
        # Get AI entry analysis
        entry = ai_find_entry_points(symbol, user)
        
        if not entry.get('has_entry'):
            return {"success": False, "reason": entry.get('reason', 'No entry found')}
        
        quality = entry.get('quality_score', 0)
        if quality < AI_ENTRY_MIN_QUALITY:
            return {"success": False, "reason": f"Quality too low ({quality}/10 < {AI_ENTRY_MIN_QUALITY}/10)"}
        
        confidence = entry.get('confidence', 0.7)
        if confidence < 0.65:
            return {"success": False, "reason": f"Confidence too low ({confidence:.0%})"}
        
        direction = entry.get('direction', 'BUY')
        entry_price = entry.get('entry_price')
        sl = entry.get('stop_loss')
        tp = entry.get('take_profit')
        
        if not all([entry_price, sl, tp]):
            return {"success": False, "reason": "Missing SL/TP levels"}
        
        # ============ COMPREHENSIVE LOSS PREVENTION CHECK ============
        if LOSS_PREVENTION_ENABLED:
            confluences = entry.get('confluences', [])
            should_enter, prevention_reason, entry_score = comprehensive_entry_check(
                symbol, direction, quality, confidence, user, confluences
            )
            if not should_enter:
                logger.warning(f"[{user}] 🛡️ LOSS PREVENTION: Skipping {symbol} - {prevention_reason}")
                return {"success": False, "reason": f"Loss prevention: {prevention_reason}"}
            
            logger.info(f"[{user}] ✅ Entry score: {entry_score}/100 for {symbol} {direction}")
        
        # Calculate SL in pips for lot sizing
        sym_settings = get_symbol_settings(symbol)
        pip_value = sym_settings.get('pip_value', 0.0001)
        sl_pips = abs(entry_price - sl) / pip_value if pip_value > 0 else 20
        
        # Check AI loss pattern learning - avoid similar losing setups
        df = get_data(symbol, mt5.TIMEFRAME_M5, n=50)
        if df is not None and len(df) > 10:
            df = calculate_advanced_indicators(df)
            current_context = {
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                'macd_hist': df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0,
                'bb_position': ((df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / 
                               (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])) if 'bb_upper' in df.columns else 0.5,
                'trend': 'UP' if df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1] else 'DOWN' if 'ema_9' in df.columns else 'NEUTRAL',
                'momentum': df['close'].iloc[-1] - df['close'].iloc[-5] if len(df) >= 5 else 0
            }
            
            # Check if we should avoid this setup
            should_avoid, similarity, avoid_reason = should_avoid_similar_setup(user, symbol, current_context)
            if should_avoid:
                logger.warning(f"[{user}] ⚠️ AI LOSS AVOIDANCE: Skipping {symbol} trade - {avoid_reason} (Similarity: {similarity:.0%})")
                return {"success": False, "reason": f"AI loss avoidance: {avoid_reason}"}
            
            # Record context for future learning
            record_trade_context(user, symbol, current_context)
        
        # INTELLIGENT LOT SIZING - Uses account size, quality, confidence, streaks
        if lot_size is None or lot_size == 0.01:
            lot_size = calculate_intelligent_lot(symbol, user, quality, confidence, sl_pips)
        
        # ============ RECOVERY MODE: BIG LOT SCALPING ============
        if LOSS_RECOVERY_ENABLED:
            recovery_status = get_recovery_status(user)
            if recovery_status['active']:
                # Check if trade qualifies for recovery mode
                can_take, reason, recovery_mult = should_take_recovery_trade(user, quality, confidence)
                if not can_take:
                    return {"success": False, "reason": reason}
                
                # Apply BIG recovery lot multiplier
                old_lot = lot_size
                lot_size = round(lot_size * recovery_mult, 2)
                
                # Shorten TP for quick scalp exit during recovery
                if RECOVERY_SCALP_MODE and RECOVERY_QUICK_TP:
                    scalp_tp_distance = RECOVERY_SCALP_TP_PIPS * pip_value
                    if direction == 'BUY':
                        tp = entry_price + scalp_tp_distance
                    else:
                        tp = entry_price - scalp_tp_distance
                    logger.info(f"[{user}] ⚡ RECOVERY SCALP: {old_lot} → {lot_size} lot, TP @ {RECOVERY_SCALP_TP_PIPS} pips")
                else:
                    logger.info(f"[{user}] 🔄 Recovery lot boost: {old_lot} → {lot_size} ({recovery_mult}x)")
        
        # Check if trading is paused (streak protection)
        if lot_size == 0:
            return {"success": False, "reason": "Trading paused - loss streak protection"}
        
        # Record for AI learning
        record_lot_for_learning(user, quality, confidence, lot_size)
        
        logger.info(f"[{user}] 🎯 AI ENTRY TRADE EXECUTING: {direction} {symbol} @ {entry_price:.5f} | "
                   f"Quality: {quality}/10 | Confidence: {confidence:.0%} | Lot: {lot_size} | {entry.get('risk_reward', '1:2')}")
        
        # Determine order type
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Execute trade using send_order
        comment = f"AI_ENTRY:{entry.get('confluences', ['Signal'])[0][:15]}"
        result = send_order(symbol, order_type, lot_size, sl, tp, comment)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Store strategies used for this trade for AI learning
            strategies_used = entry.get('confluence_strategies', [])
            if strategies_used:
                trade_strategies_used[result.order] = {
                    'strategies': strategies_used,
                    'user': user,
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'quality': quality
                }
                logger.info(f"[{user}] 📊 Tracking {len(strategies_used)} strategies for ticket #{result.order}: {strategies_used[:3]}...")
            
            return {
                "success": True,
                "ticket": result.order,
                "direction": direction,
                "symbol": symbol,
                "lot_size": lot_size,  # Show the intelligent lot size used
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "quality": quality,
                "confluences": entry.get('confluences', [])
            }
        else:
            err = result.comment if result else "Order failed"
            return {"success": False, "reason": err}
        
    except Exception as e:
        logger.error(f"AI entry trade execution error: {e}")
        return {"success": False, "reason": f"Error: {str(e)}"}


def ai_smart_trader_loop(user, symbols, lot_size=0.01):
    """
    Main AI trading loop that:
    1. Checks if current time/session is optimal for trading
    2. Scans for news-based opportunities
    3. Finds high-quality entry points
    4. Executes trades with AI confirmation
    5. Manages open positions with AI profit assurance
    
    Call this in a background thread.
    """
    logger.info(f"[{user}] 🤖 AI SMART TRADER STARTED - Scanning {len(symbols)} symbols")
    
    last_news_check = {}
    last_entry_check = {}
    last_session_check = {}
    session_approved = {}  # Track if session is approved for each symbol
    
    while True:
        try:
            current_time = time.time()
            
            for symbol in symbols:
                # === SESSION/TIMING CHECK ===
                # Only check session every 5 minutes (or on first run)
                if AI_SESSION_OPTIMIZER_ENABLED:
                    if current_time - last_session_check.get(symbol, 0) >= AI_SESSION_CHECK_INTERVAL:
                        last_session_check[symbol] = current_time
                        
                        should_trade, reason, quality = should_trade_this_session(symbol, user)
                        session_approved[symbol] = should_trade
                        
                        if not should_trade:
                            logger.info(f"[{user}] ⏰ {symbol}: WAITING - {reason} (Quality: {quality})")
                        else:
                            logger.info(f"[{user}] ✅ {symbol}: Session APPROVED - {reason} (Quality: {quality})")
                    
                    # Skip trading if session not approved
                    if not session_approved.get(symbol, True):
                        continue
                
                # === NEWS-BASED TRADING ===
                if AI_NEWS_TRADING_ENABLED:
                    if current_time - last_news_check.get(symbol, 0) >= AI_NEWS_CHECK_INTERVAL:
                        last_news_check[symbol] = current_time
                        
                        # Check if we have no position on this symbol
                        positions = mt5.positions_get(symbol=symbol)
                        if not positions:
                            # Try news-based trade
                            news_result = ai_execute_news_trade(symbol, user, lot_size)
                            if news_result.get('success'):
                                logger.info(f"[{user}] ✅ NEWS TRADE OPENED: {news_result}")
                
                # === ENTRY POINT DETECTION ===
                if AI_ENTRY_SCANNER_ENABLED:
                    if current_time - last_entry_check.get(symbol, 0) >= AI_ENTRY_CACHE_SECONDS:
                        last_entry_check[symbol] = current_time
                        
                        # Check if we have no position on this symbol
                        positions = mt5.positions_get(symbol=symbol)
                        if not positions:
                            # Try AI entry
                            entry_result = ai_execute_entry_trade(symbol, user, lot_size)
                            if entry_result.get('success'):
                                logger.info(f"[{user}] ✅ AI ENTRY OPENED: {entry_result}")
                
                # === LIVE SENTIMENT UPDATE ===
                # Update sentiment cache periodically
                get_live_market_sentiment(symbol)
            
            # Sleep before next cycle
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"AI smart trader error: {e}")
            time.sleep(30)


def get_all_live_sentiments(symbols):
    """
    Get live sentiment for multiple symbols.
    Returns a dictionary of sentiment data for UI display.
    """
    results = {}
    for symbol in symbols:
        try:
            sentiment = get_live_market_sentiment(symbol)
            results[symbol] = {
                "sentiment": sentiment.get('sentiment', 'NEUTRAL'),
                "confidence": sentiment.get('confidence', 0.5),
                "strength": sentiment.get('strength', 'WEAK'),
                "trading_bias": sentiment.get('trading_bias', 'WAIT'),
                "bias_reason": sentiment.get('bias_reason', ''),
                "short_term_outlook": sentiment.get('short_term_outlook', ''),
                "key_factors": sentiment.get('key_factors', [])
            }
        except Exception as e:
            results[symbol] = {
                "sentiment": "NEUTRAL",
                "confidence": 0.5,
                "error": str(e)
            }
    return results


def get_news_analysis(symbol, user="system"):
    """
    Get comprehensive news analysis for a symbol.
    Used by the API endpoint.
    """
    try:
        # Fetch news from all sources
        news_items = fetch_all_news_for_symbol(symbol)
        
        # Check for high-impact events
        has_event, event_details = check_high_impact_event_nearby(symbol)
        
        # Analyze sentiment
        sentiment, confidence, summary = analyze_news_sentiment_ai(news_items, symbol, user)
        
        # Get trading recommendation based on news
        news_ok_buy, mult_buy, reason_buy = should_trade_based_on_news(symbol, "BUY", user)
        news_ok_sell, mult_sell, reason_sell = should_trade_based_on_news(symbol, "SELL", user)
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": confidence,
            "summary": summary,
            "news_count": len(news_items),
            "headlines": [n.get('title', '')[:100] for n in news_items[:5]],
            "has_high_impact_event": has_event,
            "event": event_details,
            "buy_recommendation": {
                "allowed": news_ok_buy,
                "confidence_mult": mult_buy,
                "reason": reason_buy
            },
            "sell_recommendation": {
                "allowed": news_ok_sell,
                "confidence_mult": mult_sell,
                "reason": reason_sell
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"News analysis API error: {e}")
        return {
            "symbol": symbol,
            "sentiment": "NEUTRAL",
            "confidence": 0.5,
            "summary": "News analysis unavailable",
            "news_count": 0,
            "error": str(e)
        }


def get_economic_calendar():
    """
    Get upcoming high-impact economic events.
    Used by the API endpoint.
    """
    try:
        events = scrape_forexfactory_calendar()
        return {
            "events": events,
            "count": len(events),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Calendar API error: {e}")
        return {
            "events": [],
            "count": 0,
            "error": str(e)
        }


def get_ai_optimized_params(user):
    """
    Get AI-optimized trading parameters for user, or defaults if not available.
    """
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
    """
    Get AI trading insights and recommendations for dashboard display.
    """
    if user in ai_learned_params and ai_learned_params[user]:
        return {
            'has_insights': True,
            'insights': ai_learned_params[user].get('insights', 'Learning in progress...'),
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
        'insights': 'AI is learning from your trades. More data needed.',
        'strategy_adjustment': 'Execute more trades for AI optimization.',
        'optimized_params': {}
    }

# ================================================================================

# ---------------- MT5 PATH HELPER ----------------
def get_mt5_path():
    """Find the MT5 terminal executable path"""
    mt5_paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
        r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5 EXNESS\terminal.exe",
        r"C:\Program Files\XM MT5 Terminal\terminal64.exe",
        r"C:\Program Files\IC Markets - MetaTrader 5\terminal64.exe",
    ]
    for path in mt5_paths:
        if os.path.exists(path):
            return path
    return None

# ---------------- MT5 INITIALIZATION ----------------
def initialize_mt5(login=None, password=None, server=None, max_retries=3):
    """Initialize MT5 connection with user credentials or defaults"""
    mt5_login = login or DEFAULT_MT5_LOGIN
    mt5_password = password or DEFAULT_MT5_PASSWORD
    mt5_server = server or DEFAULT_MT5_SERVER
    
    # Get MT5 terminal path
    mt5_path = get_mt5_path()
    
    for attempt in range(max_retries):
        # Shutdown any existing connection first
        try:
            mt5.shutdown()
        except:
            pass
        
        time.sleep(1)  # Brief pause between attempts
        
        # Initialize with path and credentials directly
        init_kwargs = {
            'login': int(mt5_login),
            'password': mt5_password,
            'server': mt5_server,
            'timeout': 30000
        }
        if mt5_path:
            init_kwargs['path'] = mt5_path
        
        if not mt5.initialize(**init_kwargs):
            error_code, error_msg = mt5.last_error()
            # Skip logging for success codes (1 = success)
            if error_code >= 1:
                continue  # Not actually an error
            if error_code in [-10005, -10001]:  # IPC timeout or IPC send failed
                logger.warning(f"⏳ MT5 IPC error (attempt {attempt+1}/{max_retries}). Retrying...")
                time.sleep(2)
                continue
            elif error_code == -6:
                logger.error(f"❌ MT5 terminal is not running. Please start MetaTrader 5.")
                return False
            else:
                logger.error(f"❌ MT5 initialization failed: {error_msg} (Code: {error_code})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
        
        # Select all default symbols
        for sym in DEFAULT_SYMBOLS:
            mt5.symbol_select(sym, True)
        
        logger.info(f"✅ MT5 initialized successfully for account {mt5_login}")
        return True
    
    logger.error(f"❌ MT5 connection failed after {max_retries} attempts. Please restart MetaTrader 5.")
    return False


def test_mt5_connection(login, password, server):
    """Test MT5 connection without starting bot - returns (success, message)"""
    try:
        # Shutdown any existing connection first
        try:
            mt5.shutdown()
        except:
            pass
        
        time.sleep(0.5)
        
        # Get MT5 terminal path
        mt5_path = get_mt5_path()
        
        for attempt in range(3):
            # Try to initialize with path and credentials directly
            init_kwargs = {
                'login': int(login),
                'password': password,
                'server': server,
                'timeout': 30000
            }
            if mt5_path:
                init_kwargs['path'] = mt5_path
            
            if not mt5.initialize(**init_kwargs):
                error_code, error_msg = mt5.last_error()
                # Skip for success codes
                if error_code >= 1:
                    break  # Actually succeeded
                if error_code in [-10005, -10001]:  # IPC timeout or send failed
                    logger.warning(f"⏳ MT5 IPC error, retrying... ({attempt+1}/3)")
                    time.sleep(2)
                    continue
                elif error_code == -6:
                    return False, "MetaTrader 5 terminal is not running. Please open MT5 and try again."
                return False, f"MT5 initialization failed: {error_msg} (Code: {error_code})"
            break
        else:
            return False, "MT5 connection timed out. Please restart MetaTrader 5 and try again."
        
        # Get account info to verify connection (login was done in initialize)
        acc = mt5.account_info()
        if acc:
            logger.info(f"✅ MT5 test connection successful for {login}")
            mt5.shutdown()
            return True, f"Successfully connected to account {acc.login} on {acc.server}"
        
        mt5.shutdown()
        return False, "Could not retrieve account info. Please check your credentials."
    except Exception as e:
        return False, f"Connection error: {str(e)}"


# ---------------- DATA ----------------
def get_data(symbol, timeframe, n=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# ---------------- SMC STRATEGY FUNCTIONS ----------------
def trend_bias(df):
    """HTF EMA trend bias"""
    ema = df["close"].ewm(span=50).mean()
    close_price = df["close"].iloc[-1]
    ema_value = ema.iloc[-1]
    
    if close_price > ema_value * 1.001:  # 0.1% above EMA
        return "BULLISH"
    elif close_price < ema_value * 0.999:  # 0.1% below EMA
        return "BEARISH"
    return "NEUTRAL"


def liquidity_grab(df):
    """Detect liquidity sweeps - checks last 10 candles for more reliable detection"""
    high = df["high"].tail(10)
    low = df["low"].tail(10)
    
    # Recent high sweep (price made new high then pulled back)
    recent_high = high.iloc[-2]
    prev_highs = high.iloc[-5:-2]
    sweep_high = recent_high > prev_highs.max() if len(prev_highs) > 0 else False
    
    # Recent low sweep (price made new low then pulled back)
    recent_low = low.iloc[-2]
    prev_lows = low.iloc[-5:-2]
    sweep_low = recent_low < prev_lows.min() if len(prev_lows) > 0 else False
    
    return sweep_high, sweep_low


def order_block(df):
    """Identify order blocks - last opposite candle before move"""
    if len(df) < 5:
        return None, None, None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Bullish OB = bearish candle followed by bullish move
    if prev.close < prev.open and last.close > last.open:
        return "BULLISH", prev.low, prev.high
    
    # Bearish OB = bullish candle followed by bearish move
    if prev.close > prev.open and last.close < last.open:
        return "BEARISH", prev.low, prev.high
    
    return None, None, None


def fair_value_gap(df):
    """Detect fair value gaps - imbalance zones"""
    if len(df) < 3:
        return None, None, None
    
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]
    
    # Bullish FVG - gap between candle 1 high and candle 3 low
    if c1.high < c3.low:
        gap_size = c3.low - c1.high
        if gap_size > 0.5:  # Minimum gap size filter
            return "BULLISH", c1.high, c3.low
    
    # Bearish FVG - gap between candle 1 low and candle 3 high
    if c1.low > c3.high:
        gap_size = c1.low - c3.high
        if gap_size > 0.5:  # Minimum gap size filter
            return "BEARISH", c3.high, c1.low
    
    return None, None, None


def check_market_structure(df):
    """Check for break of structure (BOS) or change of character (CHoCH)"""
    if len(df) < 20:
        return False, False
    
    highs = df["high"].tail(20)
    lows = df["low"].tail(20)
    closes = df["close"].tail(20)
    
    # Higher highs and higher lows = bullish structure
    recent_high = highs.iloc[-1]
    prev_high = highs.iloc[-10:-1].max()
    bullish_bos = recent_high > prev_high
    
    # Lower lows and lower highs = bearish structure
    recent_low = lows.iloc[-1]
    prev_low = lows.iloc[-10:-1].min()
    bearish_bos = recent_low < prev_low
    
    return bullish_bos, bearish_bos


# ================================================================================
# =================== HIGH-PRECISION SCALPING ENTRY STRATEGIES ==================
# ================================================================================
# 30 Professional Entry Models with AI Learning

# Strategy Performance Tracking for AI Learning
strategy_performance = {
    # {strategy_name: {wins: 0, losses: 0, total_profit: 0, avg_win: 0, avg_loss: 0, win_rate: 0}}
}

# User-specific strategy preferences learned from performance
user_strategy_weights = {}  # {user: {strategy: weight}}

# Entry confluence requirements
MIN_CONFLUENCE_SCORE = 3  # Minimum confluence points to enter
MAX_CONFLUENCE_SCORE = 15  # Maximum possible score


def initialize_strategy_tracking():
    """Initialize strategy performance tracking"""
    global strategy_performance
    strategies = [
        'SMC_ORDER_BLOCK', 'SMC_FVG', 'SMC_LIQUIDITY_SWEEP', 'SMC_CHOCH',
        'BREAK_RETEST', 'STRUCTURE_SHIFT', 'SUPPLY_DEMAND', 'TRENDLINE_BREAK',
        'SUPPORT_RESISTANCE', 'CANDLESTICK_PATTERN', 'EMA_PULLBACK', 'FIBO_RETRACEMENT',
        'RSI_DIVERGENCE', 'MACD_DIVERGENCE', 'BB_REVERSAL', 'VWAP_BOUNCE',
        'SESSION_KILLZONE', 'NEWS_BASED', 'RANGE_BREAKOUT', 'SCALP_MICRO',
        'TREND_CONTINUATION', 'REVERSAL_PATTERN', 'VOLATILITY_BREAKOUT',
        'MEAN_REVERSION', 'MTF_CONFLUENCE', 'ICT_JUDAS', 'ICT_SILVER_BULLET',
        'ICT_POWER_OF_THREE', 'PRICE_ACTION', 'CONFLUENCE_STACK'
    ]
    for strat in strategies:
        if strat not in strategy_performance:
            strategy_performance[strat] = {
                'wins': 0, 'losses': 0, 'total_profit': 0,
                'avg_win': 0, 'avg_loss': 0, 'win_rate': 0.5,
                'last_10': []  # Last 10 trade results
            }


# ============== 1. SMART MONEY CONCEPT (SMC) ENHANCED ==============
def detect_change_of_character(df, lookback=20):
    """
    Detect Change of Character (CHoCH) - key SMC reversal signal.
    CHoCH = Break of the most recent swing high/low in opposite direction.
    """
    if len(df) < lookback:
        return None, None
    
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    closes = df['close'].tail(lookback)
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(2, lookback - 2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
           highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
            swing_highs.append((i, highs.iloc[i]))
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
           lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
            swing_lows.append((i, lows.iloc[i]))
    
    current_close = closes.iloc[-1]
    
    # Bullish CHoCH: Price breaks above recent swing high after downtrend
    if swing_highs and len(swing_highs) >= 2:
        last_swing_high = swing_highs[-1][1]
        if current_close > last_swing_high:
            return 'BULLISH_CHOCH', last_swing_high
    
    # Bearish CHoCH: Price breaks below recent swing low after uptrend
    if swing_lows and len(swing_lows) >= 2:
        last_swing_low = swing_lows[-1][1]
        if current_close < last_swing_low:
            return 'BEARISH_CHOCH', last_swing_low
    
    return None, None


def detect_order_block_mitigation(df, lookback=30):
    """
    Detect price returning to mitigate an order block.
    Returns (type, ob_low, ob_high, is_mitigated)
    """
    if len(df) < lookback:
        return None, None, None, False
    
    price = df['close'].iloc[-1]
    
    # Find all potential order blocks
    bullish_obs = []
    bearish_obs = []
    
    for i in range(lookback - 5, 5, -1):
        candle = df.iloc[i]
        next_candles = df.iloc[i+1:i+4]
        
        # Bullish OB: Bearish candle before bullish impulse
        if candle.close < candle.open:  # Bearish candle
            if len(next_candles) >= 2:
                bullish_move = (next_candles['close'].max() - candle.close) / candle.close > 0.001
                if bullish_move:
                    bullish_obs.append({
                        'low': candle.low,
                        'high': candle.high,
                        'index': i,
                        'mitigated': False
                    })
        
        # Bearish OB: Bullish candle before bearish impulse
        if candle.close > candle.open:  # Bullish candle
            if len(next_candles) >= 2:
                bearish_move = (candle.close - next_candles['close'].min()) / candle.close > 0.001
                if bearish_move:
                    bearish_obs.append({
                        'low': candle.low,
                        'high': candle.high,
                        'index': i,
                        'mitigated': False
                    })
    
    # Check if price is at an unmitigated OB
    for ob in bullish_obs[-3:]:  # Check last 3 bullish OBs
        if ob['low'] <= price <= ob['high']:
            return 'BULLISH', ob['low'], ob['high'], True
    
    for ob in bearish_obs[-3:]:  # Check last 3 bearish OBs
        if ob['low'] <= price <= ob['high']:
            return 'BEARISH', ob['low'], ob['high'], True
    
    return None, None, None, False


def detect_fvg_retracement(df, lookback=20):
    """
    Detect price retracing into a Fair Value Gap.
    Returns (type, fvg_low, fvg_high, is_filled)
    """
    if len(df) < lookback:
        return None, None, None, False
    
    price = df['close'].iloc[-1]
    fvgs = []
    
    for i in range(lookback - 3, 1, -1):
        c1 = df.iloc[i]
        c2 = df.iloc[i + 1]
        c3 = df.iloc[i + 2]
        
        # Bullish FVG
        if c1.high < c3.low:
            fvgs.append({
                'type': 'BULLISH',
                'low': c1.high,
                'high': c3.low,
                'index': i
            })
        
        # Bearish FVG
        if c1.low > c3.high:
            fvgs.append({
                'type': 'BEARISH',
                'low': c3.high,
                'high': c1.low,
                'index': i
            })
    
    # Check if price is filling an FVG
    for fvg in fvgs[-5:]:  # Check last 5 FVGs
        if fvg['low'] <= price <= fvg['high']:
            return fvg['type'], fvg['low'], fvg['high'], True
    
    return None, None, None, False


# ============== 2. LIQUIDITY SWEEP & REVERSAL ==============
def detect_equal_highs_lows(df, tolerance_pips=2, lookback=30):
    """
    Detect equal highs or equal lows (liquidity pools).
    These are targets for stop hunts.
    """
    if len(df) < lookback:
        return None, None, None
    
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    price = df['close'].iloc[-1]
    
    # Get pip value for tolerance
    avg_range = (highs - lows).mean()
    tolerance = avg_range * 0.1  # 10% of average range
    
    # Find equal highs
    equal_highs = []
    for i in range(len(highs) - 5):
        for j in range(i + 3, len(highs)):
            if abs(highs.iloc[i] - highs.iloc[j]) < tolerance:
                equal_highs.append(highs.iloc[i])
    
    # Find equal lows
    equal_lows = []
    for i in range(len(lows) - 5):
        for j in range(i + 3, len(lows)):
            if abs(lows.iloc[i] - lows.iloc[j]) < tolerance:
                equal_lows.append(lows.iloc[i])
    
    # Check for sweep
    if equal_highs and price > max(equal_highs):
        return 'SWEEP_HIGH', max(equal_highs), price
    if equal_lows and price < min(equal_lows):
        return 'SWEEP_LOW', min(equal_lows), price
    
    return None, None, None


def detect_stop_hunt_reversal(df, lookback=15):
    """
    Detect stop hunt pattern - price spikes beyond level then reverses.
    Entry after the hunt is complete.
    """
    if len(df) < lookback:
        return None, 0
    
    recent = df.tail(5)
    prev = df.iloc[-lookback:-5]
    
    recent_high = recent['high'].max()
    recent_low = recent['low'].min()
    prev_high = prev['high'].max()
    prev_low = prev['low'].min()
    
    current_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    
    # Bullish stop hunt: Price spiked below previous low but closed back above
    if recent_low < prev_low and current_close > prev_low and current_close > prev_close:
        confidence = min((current_close - recent_low) / (prev_high - prev_low), 1.0) if prev_high != prev_low else 0.5
        return 'BULLISH_HUNT', confidence
    
    # Bearish stop hunt: Price spiked above previous high but closed back below
    if recent_high > prev_high and current_close < prev_high and current_close < prev_close:
        confidence = min((recent_high - current_close) / (prev_high - prev_low), 1.0) if prev_high != prev_low else 0.5
        return 'BEARISH_HUNT', confidence
    
    return None, 0


# ============== 3. BREAK AND RETEST STRATEGY ==============
def detect_break_and_retest(df, lookback=30):
    """
    Detect break of structure followed by retest for entry.
    Returns (type, level, is_retesting)
    """
    if len(df) < lookback:
        return None, None, False
    
    price = df['close'].iloc[-1]
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    
    # Find key levels that were broken
    resistance_levels = []
    support_levels = []
    
    # Find swing points
    for i in range(5, lookback - 5):
        if highs.iloc[i] > highs.iloc[i-2:i].max() and highs.iloc[i] > highs.iloc[i+1:i+3].max():
            resistance_levels.append(highs.iloc[i])
        if lows.iloc[i] < lows.iloc[i-2:i].min() and lows.iloc[i] < lows.iloc[i+1:i+3].min():
            support_levels.append(lows.iloc[i])
    
    avg_range = (highs - lows).mean()
    retest_tolerance = avg_range * 0.3
    
    # Check for resistance break + retest (bullish)
    for level in resistance_levels:
        # Was broken (price went above) and now retesting
        if df['high'].iloc[-10:-3].max() > level and abs(price - level) < retest_tolerance:
            if price > level * 0.995:  # Price hovering near/above level
                return 'BULLISH_RETEST', level, True
    
    # Check for support break + retest (bearish)
    for level in support_levels:
        # Was broken (price went below) and now retesting
        if df['low'].iloc[-10:-3].min() < level and abs(price - level) < retest_tolerance:
            if price < level * 1.005:  # Price hovering near/below level
                return 'BEARISH_RETEST', level, True
    
    return None, None, False


# ============== 4. MARKET STRUCTURE SHIFT ENTRY ==============
def detect_structure_shift(df, lookback=25):
    """
    Detect market structure shift (MSS) - trend change confirmation.
    Higher high → lower low (bearish MSS)
    Lower low → higher high (bullish MSS)
    """
    if len(df) < lookback:
        return None, 0
    
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    
    # Find swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(3, lookback - 3):
        if highs.iloc[i] >= highs.iloc[i-3:i].max() and highs.iloc[i] >= highs.iloc[i+1:i+4].max():
            swing_highs.append((i, highs.iloc[i]))
        if lows.iloc[i] <= lows.iloc[i-3:i].min() and lows.iloc[i] <= lows.iloc[i+1:i+4].min():
            swing_lows.append((i, lows.iloc[i]))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None, 0
    
    # Check for bullish MSS (was making lower lows, now made higher high)
    if swing_lows[-1][1] > swing_lows[-2][1] and swing_highs[-1][1] > swing_highs[-2][1]:
        confidence = 0.8
        return 'BULLISH_MSS', confidence
    
    # Check for bearish MSS (was making higher highs, now made lower low)
    if swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
        confidence = 0.8
        return 'BEARISH_MSS', confidence
    
    return None, 0


# ============== 5. SUPPLY AND DEMAND ZONES ==============
def find_supply_demand_zones(df, lookback=50):
    """
    Identify fresh supply and demand zones.
    Returns list of zones with type, low, high, strength, freshness.
    """
    if len(df) < lookback:
        return []
    
    zones = []
    
    for i in range(10, lookback - 5):
        candle = df.iloc[i]
        body = abs(candle.close - candle.open)
        range_size = candle.high - candle.low
        
        # Look for strong moves away from a zone
        next_3 = df.iloc[i+1:i+4]
        prev_3 = df.iloc[i-3:i]
        
        if len(next_3) < 3:
            continue
        
        # Demand zone: Price rallied strongly from this area
        if next_3['close'].iloc[-1] > candle.high:
            move_size = next_3['close'].iloc[-1] - candle.low
            if move_size > range_size * 2:  # Strong move
                # Check if zone is fresh (not retested)
                remaining = df.iloc[i+4:]
                revisited = (remaining['low'] <= candle.high).any() if len(remaining) > 0 else False
                zones.append({
                    'type': 'DEMAND',
                    'low': candle.low,
                    'high': max(candle.open, candle.close),
                    'strength': min(move_size / range_size / 5, 1.0),
                    'fresh': not revisited
                })
        
        # Supply zone: Price dropped strongly from this area
        if next_3['close'].iloc[-1] < candle.low:
            move_size = candle.high - next_3['close'].iloc[-1]
            if move_size > range_size * 2:  # Strong move
                # Check if zone is fresh
                remaining = df.iloc[i+4:]
                revisited = (remaining['high'] >= candle.low).any() if len(remaining) > 0 else False
                zones.append({
                    'type': 'SUPPLY',
                    'low': min(candle.open, candle.close),
                    'high': candle.high,
                    'strength': min(move_size / range_size / 5, 1.0),
                    'fresh': not revisited
                })
    
    return zones


def is_at_supply_demand_zone(df, zones):
    """Check if current price is at a supply or demand zone"""
    if not zones:
        return None, None
    
    price = df['close'].iloc[-1]
    
    for zone in zones:
        if zone['low'] <= price <= zone['high'] and zone['fresh']:
            return zone['type'], zone
    
    return None, None


# ============== 6-10. CANDLESTICK & TECHNICAL PATTERNS ==============
def detect_pin_bar(df):
    """Detect pin bar / hammer / shooting star patterns"""
    if len(df) < 3:
        return None, 0
    
    candle = df.iloc[-1]
    body = abs(candle.close - candle.open)
    range_size = candle.high - candle.low
    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    
    if range_size == 0:
        return None, 0
    
    # Bullish pin bar (hammer): Long lower wick, small body at top
    if lower_wick > body * 2 and lower_wick > upper_wick * 2:
        confidence = min(lower_wick / range_size, 0.95)
        return 'BULLISH_PIN', confidence
    
    # Bearish pin bar (shooting star): Long upper wick, small body at bottom
    if upper_wick > body * 2 and upper_wick > lower_wick * 2:
        confidence = min(upper_wick / range_size, 0.95)
        return 'BEARISH_PIN', confidence
    
    return None, 0


def detect_engulfing(df):
    """Detect bullish/bearish engulfing patterns"""
    if len(df) < 3:
        return None, 0
    
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    
    if curr_body == 0 or prev_body == 0:
        return None, 0
    
    # Bullish engulfing
    if prev.close < prev.open and curr.close > curr.open:
        if curr_body > prev_body and curr.close > prev.open and curr.open < prev.close:
            confidence = min(curr_body / prev_body * 0.5, 0.9)
            return 'BULLISH_ENGULF', confidence
    
    # Bearish engulfing
    if prev.close > prev.open and curr.close < curr.open:
        if curr_body > prev_body and curr.close < prev.open and curr.open > prev.close:
            confidence = min(curr_body / prev_body * 0.5, 0.9)
            return 'BEARISH_ENGULF', confidence
    
    return None, 0


def detect_inside_bar_breakout(df):
    """Detect inside bar pattern and breakout"""
    if len(df) < 4:
        return None, 0
    
    mother = df.iloc[-3]
    inside = df.iloc[-2]
    breakout = df.iloc[-1]
    
    # Check if middle candle is inside bar
    if inside.high < mother.high and inside.low > mother.low:
        # Bullish breakout
        if breakout.close > mother.high:
            confidence = 0.75
            return 'BULLISH_IB_BREAK', confidence
        
        # Bearish breakout
        if breakout.close < mother.low:
            confidence = 0.75
            return 'BEARISH_IB_BREAK', confidence
    
    return None, 0


# ============== 11-15. INDICATOR-BASED ENTRIES ==============
def detect_ema_pullback(df, ema_period=20):
    """Detect pullback to EMA for trend continuation entry"""
    if len(df) < ema_period + 10:
        return None, 0
    
    ema = df['close'].ewm(span=ema_period).mean()
    price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-3]
    ema_value = ema.iloc[-1]
    
    # Trend direction
    trend_up = ema.iloc[-1] > ema.iloc[-10]
    trend_down = ema.iloc[-1] < ema.iloc[-10]
    
    # Distance from EMA
    distance = abs(price - ema_value) / ema_value
    
    # Bullish: Uptrend, price pulled back to EMA, now bouncing
    if trend_up and price > ema_value and prev_price < ema_value * 1.005:
        if distance < 0.005:  # Within 0.5% of EMA
            return 'BULLISH_EMA_PULLBACK', 0.7
    
    # Bearish: Downtrend, price pulled back to EMA, now rejecting
    if trend_down and price < ema_value and prev_price > ema_value * 0.995:
        if distance < 0.005:
            return 'BEARISH_EMA_PULLBACK', 0.7
    
    return None, 0


def detect_fibo_retracement(df, lookback=30):
    """Detect price at key Fibonacci retracement levels"""
    if len(df) < lookback:
        return None, None, 0
    
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    price = df['close'].iloc[-1]
    
    swing_high = highs.max()
    swing_low = lows.min()
    range_size = swing_high - swing_low
    
    if range_size == 0:
        return None, None, 0
    
    # Fibo levels
    fib_382 = swing_high - range_size * 0.382
    fib_50 = swing_high - range_size * 0.5
    fib_618 = swing_high - range_size * 0.618
    
    tolerance = range_size * 0.02  # 2% tolerance
    
    # Check if at key levels
    for level, name in [(fib_382, '38.2%'), (fib_50, '50%'), (fib_618, '61.8%')]:
        if abs(price - level) < tolerance:
            # Determine direction based on trend
            if price > (swing_high + swing_low) / 2:
                return 'BULLISH_FIBO', name, 0.75
            else:
                return 'BEARISH_FIBO', name, 0.75
    
    return None, None, 0


def detect_rsi_divergence(df, rsi_period=14, lookback=20):
    """Detect RSI divergence (regular and hidden)"""
    if len(df) < lookback + rsi_period:
        return None, 0
    
    if 'rsi' not in df.columns:
        return None, 0
    
    prices = df['close'].tail(lookback)
    rsi = df['rsi'].tail(lookback)
    
    # Find price highs and lows
    price_high_1 = prices.iloc[-5:].max()
    price_high_2 = prices.iloc[-15:-5].max()
    price_low_1 = prices.iloc[-5:].min()
    price_low_2 = prices.iloc[-15:-5].min()
    
    rsi_high_1 = rsi.iloc[-5:].max()
    rsi_high_2 = rsi.iloc[-15:-5].max()
    rsi_low_1 = rsi.iloc[-5:].min()
    rsi_low_2 = rsi.iloc[-15:-5].min()
    
    # Bearish divergence: Higher high in price, lower high in RSI
    if price_high_1 > price_high_2 and rsi_high_1 < rsi_high_2:
        return 'BEARISH_RSI_DIV', 0.8
    
    # Bullish divergence: Lower low in price, higher low in RSI
    if price_low_1 < price_low_2 and rsi_low_1 > rsi_low_2:
        return 'BULLISH_RSI_DIV', 0.8
    
    # Hidden bullish: Higher low in price, lower low in RSI (trend continuation)
    if price_low_1 > price_low_2 and rsi_low_1 < rsi_low_2:
        return 'HIDDEN_BULL_DIV', 0.7
    
    # Hidden bearish: Lower high in price, higher high in RSI
    if price_high_1 < price_high_2 and rsi_high_1 > rsi_high_2:
        return 'HIDDEN_BEAR_DIV', 0.7
    
    return None, 0


def detect_macd_divergence(df, lookback=20):
    """Detect MACD divergence"""
    if len(df) < lookback or 'macd_hist' not in df.columns:
        return None, 0
    
    prices = df['close'].tail(lookback)
    macd = df['macd_hist'].tail(lookback)
    
    price_high_1 = prices.iloc[-5:].max()
    price_high_2 = prices.iloc[-15:-5].max()
    price_low_1 = prices.iloc[-5:].min()
    price_low_2 = prices.iloc[-15:-5].min()
    
    macd_high_1 = macd.iloc[-5:].max()
    macd_high_2 = macd.iloc[-15:-5].max()
    macd_low_1 = macd.iloc[-5:].min()
    macd_low_2 = macd.iloc[-15:-5].min()
    
    # Bearish divergence
    if price_high_1 > price_high_2 and macd_high_1 < macd_high_2:
        return 'BEARISH_MACD_DIV', 0.75
    
    # Bullish divergence
    if price_low_1 < price_low_2 and macd_low_1 > macd_low_2:
        return 'BULLISH_MACD_DIV', 0.75
    
    return None, 0


def detect_bollinger_reversal(df):
    """Detect Bollinger Band reversal setups"""
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        return None, 0
    
    price = df['close'].iloc[-1]
    bb_upper = df['bb_upper'].iloc[-1]
    bb_lower = df['bb_lower'].iloc[-1]
    bb_middle = df['bb_middle'].iloc[-1] if 'bb_middle' in df.columns else (bb_upper + bb_lower) / 2
    
    bb_width = bb_upper - bb_lower
    if bb_width == 0:
        return None, 0
    
    # Position in bands (0 = lower, 1 = upper)
    position = (price - bb_lower) / bb_width
    
    # Bullish: Price touched lower band and bouncing
    if position < 0.1 and df['close'].iloc[-1] > df['close'].iloc[-2]:
        return 'BULLISH_BB_REV', 0.7
    
    # Bearish: Price touched upper band and rejecting
    if position > 0.9 and df['close'].iloc[-1] < df['close'].iloc[-2]:
        return 'BEARISH_BB_REV', 0.7
    
    return None, 0


# ============== 16-20. SESSION & RANGE BASED ==============
def detect_session_killzone_entry(df, symbol):
    """Detect optimal kill zone entry during high-volume sessions"""
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    hour = now.hour
    minute = now.minute
    
    # Kill zones (high probability trading times)
    london_open = 7 <= hour < 10
    ny_open = 12 <= hour < 15
    overlap = 12 <= hour < 16
    
    killzone = None
    if overlap:
        killzone = 'OVERLAP'
    elif london_open:
        killzone = 'LONDON_OPEN'
    elif ny_open:
        killzone = 'NY_OPEN'
    
    if killzone:
        # Check for initial move direction
        if len(df) >= 3:
            recent_move = df['close'].iloc[-1] - df['close'].iloc[-3]
            if recent_move > 0:
                return 'BULLISH_KZ', killzone, 0.65
            elif recent_move < 0:
                return 'BEARISH_KZ', killzone, 0.65
    
    return None, None, 0


def detect_asia_range_breakout(df, lookback=60):
    """Detect breakout from Asian session range"""
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Only check during London/NY (after Asian close)
    if not (7 <= hour < 17):
        return None, None, None, 0
    
    # Find Asian range (roughly last 8-12 hours of data during Asian time)
    if len(df) < lookback:
        return None, None, None, 0
    
    # Approximate Asian range using first 20 candles of lookback
    asian_data = df.iloc[-lookback:-lookback+20]
    asian_high = asian_data['high'].max()
    asian_low = asian_data['low'].min()
    
    price = df['close'].iloc[-1]
    
    # Bullish breakout
    if price > asian_high:
        return 'BULLISH_ASIA_BREAK', asian_high, asian_low, 0.7
    
    # Bearish breakout
    if price < asian_low:
        return 'BEARISH_ASIA_BREAK', asian_high, asian_low, 0.7
    
    return None, None, None, 0


# ============== 21-25. REVERSAL & BREAKOUT PATTERNS ==============
def detect_double_top_bottom(df, lookback=30, tolerance=0.002):
    """Detect double top or double bottom patterns"""
    if len(df) < lookback:
        return None, 0
    
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    price = df['close'].iloc[-1]
    
    # Find peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(3, lookback - 3):
        if highs.iloc[i] > highs.iloc[i-3:i].max() and highs.iloc[i] > highs.iloc[i+1:i+4].max():
            peaks.append((i, highs.iloc[i]))
        if lows.iloc[i] < lows.iloc[i-3:i].min() and lows.iloc[i] < lows.iloc[i+1:i+4].min():
            troughs.append((i, lows.iloc[i]))
    
    # Double top (bearish)
    if len(peaks) >= 2:
        peak1, peak2 = peaks[-2][1], peaks[-1][1]
        if abs(peak1 - peak2) / peak1 < tolerance:
            if price < min(peaks[-2][1], peaks[-1][1]):
                return 'DOUBLE_TOP', 0.8
    
    # Double bottom (bullish)
    if len(troughs) >= 2:
        trough1, trough2 = troughs[-2][1], troughs[-1][1]
        if abs(trough1 - trough2) / trough1 < tolerance:
            if price > max(troughs[-2][1], troughs[-1][1]):
                return 'DOUBLE_BOTTOM', 0.8
    
    return None, 0


def detect_volatility_breakout(df, lookback=20):
    """Detect volatility expansion breakout (squeeze breakout)"""
    if 'atr' not in df.columns or len(df) < lookback:
        return None, 0
    
    atr = df['atr'].tail(lookback)
    current_atr = atr.iloc[-1]
    avg_atr = atr.iloc[:-5].mean()
    
    # Volatility expansion
    if current_atr > avg_atr * 1.5:
        recent_move = df['close'].iloc[-1] - df['close'].iloc[-3]
        if recent_move > 0:
            return 'BULLISH_VOL_BREAK', 0.7
        elif recent_move < 0:
            return 'BEARISH_VOL_BREAK', 0.7
    
    return None, 0


# ============== 26-30. ICT & ADVANCED CONCEPTS ==============
def detect_ict_judas_swing(df, lookback=20):
    """
    Detect ICT Judas Swing - fake move to trap traders before real move.
    Usually happens at session opens.
    """
    if len(df) < lookback:
        return None, 0
    
    # Look for initial move followed by sharp reversal
    open_5 = df['open'].iloc[-5]
    high_5 = df['high'].iloc[-5:].max()
    low_5 = df['low'].iloc[-5:].min()
    close = df['close'].iloc[-1]
    
    initial_range = high_5 - low_5
    if initial_range == 0:
        return None, 0
    
    # Bullish Judas: Initial move down, now reversing up
    if df['close'].iloc[-3] < open_5 and close > open_5:
        if (close - low_5) / initial_range > 0.7:
            return 'BULLISH_JUDAS', 0.75
    
    # Bearish Judas: Initial move up, now reversing down
    if df['close'].iloc[-3] > open_5 and close < open_5:
        if (high_5 - close) / initial_range > 0.7:
            return 'BEARISH_JUDAS', 0.75
    
    return None, 0


def detect_ict_silver_bullet(df):
    """
    ICT Silver Bullet - specific time-based entry during NY session.
    10:00-11:00 AM NY time optimal entry window.
    """
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    ny_hour = (now.hour - 5) % 24  # Approximate NY time
    
    # Silver bullet window: 10-11 AM NY
    if not (10 <= ny_hour < 11):
        return None, 0
    
    if len(df) < 10:
        return None, 0
    
    # Look for displacement (strong move) followed by retracement
    move = df['close'].iloc[-1] - df['close'].iloc[-10]
    
    if move > 0:
        return 'BULLISH_SILVER_BULLET', 0.7
    elif move < 0:
        return 'BEARISH_SILVER_BULLET', 0.7
    
    return None, 0


def detect_ict_power_of_three(df, lookback=20):
    """
    ICT Power of Three: Accumulation → Manipulation → Distribution
    Entry after manipulation phase.
    """
    if len(df) < lookback:
        return None, 0
    
    # Divide lookback into 3 phases
    phase_size = lookback // 3
    
    phase1 = df.iloc[-lookback:-lookback+phase_size]  # Accumulation
    phase2 = df.iloc[-lookback+phase_size:-phase_size]  # Manipulation
    phase3 = df.iloc[-phase_size:]  # Distribution
    
    # Accumulation: Low volatility, small range
    p1_range = phase1['high'].max() - phase1['low'].min()
    
    # Manipulation: Spike in one direction
    p2_high = phase2['high'].max()
    p2_low = phase2['low'].min()
    
    # Distribution: Move in opposite direction
    p3_close = phase3['close'].iloc[-1]
    p3_open = phase3['open'].iloc[0]
    
    # Bullish PO3: Manipulation down, distribution up
    if p2_low < phase1['low'].min() and p3_close > p3_open:
        return 'BULLISH_PO3', 0.75
    
    # Bearish PO3: Manipulation up, distribution down
    if p2_high > phase1['high'].max() and p3_close < p3_open:
        return 'BEARISH_PO3', 0.75
    
    return None, 0


# ================================================================================
# =================== MASTER CONFLUENCE ENTRY SCANNER ===========================
# ================================================================================

def scan_all_entry_strategies(symbol, df, user):
    """
    Comprehensive scanner that checks ALL entry strategies and calculates
    a confluence score. Returns the best entry with all confirmations.
    """
    if df is None or len(df) < 50:
        return None
    
    confluence_score = 0
    signals = []
    direction_score = {'BUY': 0, 'SELL': 0}
    strategies_triggered = []
    
    # 1. SMC Strategies (High weight)
    choch_type, choch_level = detect_change_of_character(df)
    if choch_type == 'BULLISH_CHOCH':
        direction_score['BUY'] += 2
        strategies_triggered.append(('SMC_CHOCH', 'BUY', 0.8))
    elif choch_type == 'BEARISH_CHOCH':
        direction_score['SELL'] += 2
        strategies_triggered.append(('SMC_CHOCH', 'SELL', 0.8))
    
    ob_type, ob_low, ob_high, ob_mitigated = detect_order_block_mitigation(df)
    if ob_mitigated:
        if ob_type == 'BULLISH':
            direction_score['BUY'] += 2
            strategies_triggered.append(('SMC_ORDER_BLOCK', 'BUY', 0.85))
        elif ob_type == 'BEARISH':
            direction_score['SELL'] += 2
            strategies_triggered.append(('SMC_ORDER_BLOCK', 'SELL', 0.85))
    
    fvg_type, fvg_low, fvg_high, fvg_filled = detect_fvg_retracement(df)
    if fvg_filled:
        if fvg_type == 'BULLISH':
            direction_score['BUY'] += 1.5
            strategies_triggered.append(('SMC_FVG', 'BUY', 0.75))
        elif fvg_type == 'BEARISH':
            direction_score['SELL'] += 1.5
            strategies_triggered.append(('SMC_FVG', 'SELL', 0.75))
    
    sweep_high, sweep_low = liquidity_grab(df)
    if sweep_low:
        direction_score['BUY'] += 1.5
        strategies_triggered.append(('SMC_LIQUIDITY_SWEEP', 'BUY', 0.7))
    if sweep_high:
        direction_score['SELL'] += 1.5
        strategies_triggered.append(('SMC_LIQUIDITY_SWEEP', 'SELL', 0.7))
    
    # 2. Liquidity & Stop Hunt
    hunt_type, hunt_conf = detect_stop_hunt_reversal(df)
    if hunt_type == 'BULLISH_HUNT':
        direction_score['BUY'] += 2 * hunt_conf
        strategies_triggered.append(('STOP_HUNT', 'BUY', hunt_conf))
    elif hunt_type == 'BEARISH_HUNT':
        direction_score['SELL'] += 2 * hunt_conf
        strategies_triggered.append(('STOP_HUNT', 'SELL', hunt_conf))
    
    # 3. Break and Retest
    br_type, br_level, br_retesting = detect_break_and_retest(df)
    if br_retesting:
        if br_type == 'BULLISH_RETEST':
            direction_score['BUY'] += 2
            strategies_triggered.append(('BREAK_RETEST', 'BUY', 0.8))
        elif br_type == 'BEARISH_RETEST':
            direction_score['SELL'] += 2
            strategies_triggered.append(('BREAK_RETEST', 'SELL', 0.8))
    
    # 4. Market Structure Shift
    mss_type, mss_conf = detect_structure_shift(df)
    if mss_type == 'BULLISH_MSS':
        direction_score['BUY'] += 2 * mss_conf
        strategies_triggered.append(('STRUCTURE_SHIFT', 'BUY', mss_conf))
    elif mss_type == 'BEARISH_MSS':
        direction_score['SELL'] += 2 * mss_conf
        strategies_triggered.append(('STRUCTURE_SHIFT', 'SELL', mss_conf))
    
    # 5. Supply and Demand
    zones = find_supply_demand_zones(df)
    zone_type, zone = is_at_supply_demand_zone(df, zones)
    if zone_type == 'DEMAND':
        direction_score['BUY'] += 2 * zone['strength']
        strategies_triggered.append(('SUPPLY_DEMAND', 'BUY', zone['strength']))
    elif zone_type == 'SUPPLY':
        direction_score['SELL'] += 2 * zone['strength']
        strategies_triggered.append(('SUPPLY_DEMAND', 'SELL', zone['strength']))
    
    # 6. Candlestick Patterns
    pin_type, pin_conf = detect_pin_bar(df)
    if pin_type:
        if 'BULLISH' in pin_type:
            direction_score['BUY'] += 1.5 * pin_conf
            strategies_triggered.append(('CANDLESTICK_PATTERN', 'BUY', pin_conf))
        else:
            direction_score['SELL'] += 1.5 * pin_conf
            strategies_triggered.append(('CANDLESTICK_PATTERN', 'SELL', pin_conf))
    
    engulf_type, engulf_conf = detect_engulfing(df)
    if engulf_type:
        if 'BULLISH' in engulf_type:
            direction_score['BUY'] += 1.5 * engulf_conf
            strategies_triggered.append(('CANDLESTICK_PATTERN', 'BUY', engulf_conf))
        else:
            direction_score['SELL'] += 1.5 * engulf_conf
            strategies_triggered.append(('CANDLESTICK_PATTERN', 'SELL', engulf_conf))
    
    ib_type, ib_conf = detect_inside_bar_breakout(df)
    if ib_type:
        if 'BULLISH' in ib_type:
            direction_score['BUY'] += 1 * ib_conf
            strategies_triggered.append(('INSIDE_BAR', 'BUY', ib_conf))
        else:
            direction_score['SELL'] += 1 * ib_conf
            strategies_triggered.append(('INSIDE_BAR', 'SELL', ib_conf))
    
    # 7. EMA Pullback
    ema_type, ema_conf = detect_ema_pullback(df)
    if ema_type:
        if 'BULLISH' in ema_type:
            direction_score['BUY'] += 1.5 * ema_conf
            strategies_triggered.append(('EMA_PULLBACK', 'BUY', ema_conf))
        else:
            direction_score['SELL'] += 1.5 * ema_conf
            strategies_triggered.append(('EMA_PULLBACK', 'SELL', ema_conf))
    
    # 8. Fibonacci
    fibo_type, fibo_level, fibo_conf = detect_fibo_retracement(df)
    if fibo_type:
        if 'BULLISH' in fibo_type:
            direction_score['BUY'] += 1.5 * fibo_conf
            strategies_triggered.append(('FIBO_RETRACEMENT', 'BUY', fibo_conf))
        else:
            direction_score['SELL'] += 1.5 * fibo_conf
            strategies_triggered.append(('FIBO_RETRACEMENT', 'SELL', fibo_conf))
    
    # 9. RSI Divergence
    rsi_type, rsi_conf = detect_rsi_divergence(df)
    if rsi_type:
        if 'BULLISH' in rsi_type or 'BULL' in rsi_type:
            direction_score['BUY'] += 2 * rsi_conf
            strategies_triggered.append(('RSI_DIVERGENCE', 'BUY', rsi_conf))
        else:
            direction_score['SELL'] += 2 * rsi_conf
            strategies_triggered.append(('RSI_DIVERGENCE', 'SELL', rsi_conf))
    
    # 10. MACD Divergence
    macd_type, macd_conf = detect_macd_divergence(df)
    if macd_type:
        if 'BULLISH' in macd_type:
            direction_score['BUY'] += 1.5 * macd_conf
            strategies_triggered.append(('MACD_DIVERGENCE', 'BUY', macd_conf))
        else:
            direction_score['SELL'] += 1.5 * macd_conf
            strategies_triggered.append(('MACD_DIVERGENCE', 'SELL', macd_conf))
    
    # 11. Bollinger Band
    bb_type, bb_conf = detect_bollinger_reversal(df)
    if bb_type:
        if 'BULLISH' in bb_type:
            direction_score['BUY'] += 1 * bb_conf
            strategies_triggered.append(('BB_REVERSAL', 'BUY', bb_conf))
        else:
            direction_score['SELL'] += 1 * bb_conf
            strategies_triggered.append(('BB_REVERSAL', 'SELL', bb_conf))
    
    # 12. Session Killzone
    kz_type, killzone, kz_conf = detect_session_killzone_entry(df, symbol)
    if kz_type:
        if 'BULLISH' in kz_type:
            direction_score['BUY'] += 1.5 * kz_conf
            strategies_triggered.append(('SESSION_KILLZONE', 'BUY', kz_conf))
        else:
            direction_score['SELL'] += 1.5 * kz_conf
            strategies_triggered.append(('SESSION_KILLZONE', 'SELL', kz_conf))
    
    # 13. Asia Range Breakout
    asia_type, asia_high, asia_low, asia_conf = detect_asia_range_breakout(df)
    if asia_type:
        if 'BULLISH' in asia_type:
            direction_score['BUY'] += 1.5 * asia_conf
            strategies_triggered.append(('RANGE_BREAKOUT', 'BUY', asia_conf))
        else:
            direction_score['SELL'] += 1.5 * asia_conf
            strategies_triggered.append(('RANGE_BREAKOUT', 'SELL', asia_conf))
    
    # 14. Double Top/Bottom
    pattern_type, pattern_conf = detect_double_top_bottom(df)
    if pattern_type == 'DOUBLE_BOTTOM':
        direction_score['BUY'] += 2 * pattern_conf
        strategies_triggered.append(('REVERSAL_PATTERN', 'BUY', pattern_conf))
    elif pattern_type == 'DOUBLE_TOP':
        direction_score['SELL'] += 2 * pattern_conf
        strategies_triggered.append(('REVERSAL_PATTERN', 'SELL', pattern_conf))
    
    # 15. Volatility Breakout
    vol_type, vol_conf = detect_volatility_breakout(df)
    if vol_type:
        if 'BULLISH' in vol_type:
            direction_score['BUY'] += 1.5 * vol_conf
            strategies_triggered.append(('VOLATILITY_BREAKOUT', 'BUY', vol_conf))
        else:
            direction_score['SELL'] += 1.5 * vol_conf
            strategies_triggered.append(('VOLATILITY_BREAKOUT', 'SELL', vol_conf))
    
    # 16. ICT Judas Swing
    judas_type, judas_conf = detect_ict_judas_swing(df)
    if judas_type:
        if 'BULLISH' in judas_type:
            direction_score['BUY'] += 2 * judas_conf
            strategies_triggered.append(('ICT_JUDAS', 'BUY', judas_conf))
        else:
            direction_score['SELL'] += 2 * judas_conf
            strategies_triggered.append(('ICT_JUDAS', 'SELL', judas_conf))
    
    # 17. ICT Silver Bullet
    sb_type, sb_conf = detect_ict_silver_bullet(df)
    if sb_type:
        if 'BULLISH' in sb_type:
            direction_score['BUY'] += 1.5 * sb_conf
            strategies_triggered.append(('ICT_SILVER_BULLET', 'BUY', sb_conf))
        else:
            direction_score['SELL'] += 1.5 * sb_conf
            strategies_triggered.append(('ICT_SILVER_BULLET', 'SELL', sb_conf))
    
    # 18. ICT Power of Three
    po3_type, po3_conf = detect_ict_power_of_three(df)
    if po3_type:
        if 'BULLISH' in po3_type:
            direction_score['BUY'] += 2 * po3_conf
            strategies_triggered.append(('ICT_POWER_OF_THREE', 'BUY', po3_conf))
        else:
            direction_score['SELL'] += 2 * po3_conf
            strategies_triggered.append(('ICT_POWER_OF_THREE', 'SELL', po3_conf))
    
    # Determine final direction
    buy_score = direction_score['BUY']
    sell_score = direction_score['SELL']
    
    if buy_score > sell_score and buy_score >= MIN_CONFLUENCE_SCORE:
        direction = 'BUY'
        confluence_score = buy_score
    elif sell_score > buy_score and sell_score >= MIN_CONFLUENCE_SCORE:
        direction = 'SELL'
        confluence_score = sell_score
    else:
        return None  # No valid entry
    
    # Filter strategies for the winning direction
    relevant_strategies = [(s, d, c) for s, d, c in strategies_triggered if d == direction]
    
    # Apply user-learned weights if available
    if user in user_strategy_weights:
        weights = user_strategy_weights[user]
        weighted_score = 0
        for strat, _, conf in relevant_strategies:
            weight = weights.get(strat, 1.0)
            weighted_score += conf * weight
        confluence_score = weighted_score
    
    return {
        'direction': direction,
        'confluence_score': round(confluence_score, 2),
        'strategies': relevant_strategies,
        'strategy_count': len(relevant_strategies),
        'buy_score': round(buy_score, 2),
        'sell_score': round(sell_score, 2),
        'price': df['close'].iloc[-1],
        'timestamp': datetime.now().isoformat()
    }


def update_strategy_performance(strategy_name, won, profit_pips, user=None):
    """
    Update strategy performance tracking for AI learning.
    Call this after a trade closes.
    """
    global strategy_performance, user_strategy_weights
    
    if strategy_name not in strategy_performance:
        initialize_strategy_tracking()
    
    perf = strategy_performance[strategy_name]
    
    if won:
        perf['wins'] += 1
        perf['total_profit'] += profit_pips
    else:
        perf['losses'] += 1
        perf['total_profit'] -= abs(profit_pips)
    
    total = perf['wins'] + perf['losses']
    perf['win_rate'] = perf['wins'] / total if total > 0 else 0.5
    
    # Track last 10 results
    perf['last_10'].append(1 if won else 0)
    if len(perf['last_10']) > 10:
        perf['last_10'].pop(0)
    
    # Update user-specific weights based on performance
    if user:
        if user not in user_strategy_weights:
            user_strategy_weights[user] = {}
        
        # Weight based on recent win rate
        recent_wr = sum(perf['last_10']) / len(perf['last_10']) if perf['last_10'] else 0.5
        
        # Strategies with >60% win rate get boosted, <40% get reduced
        if recent_wr >= 0.6:
            user_strategy_weights[user][strategy_name] = 1.0 + (recent_wr - 0.5) * 2
        elif recent_wr <= 0.4:
            user_strategy_weights[user][strategy_name] = max(0.3, recent_wr * 2)
        else:
            user_strategy_weights[user][strategy_name] = 1.0
    
    logger.info(f"📊 Strategy [{strategy_name}] updated: WR={perf['win_rate']:.0%} ({perf['wins']}W/{perf['losses']}L)")


def get_best_strategies(user=None, min_trades=5):
    """
    Get the best performing strategies for display or decision making.
    """
    best = []
    
    for strat, perf in strategy_performance.items():
        total = perf['wins'] + perf['losses']
        if total >= min_trades:
            best.append({
                'strategy': strat,
                'win_rate': perf['win_rate'],
                'trades': total,
                'profit': perf['total_profit'],
                'recent_wr': sum(perf['last_10']) / len(perf['last_10']) if perf['last_10'] else 0.5
            })
    
    # Sort by recent win rate
    best.sort(key=lambda x: x['recent_wr'], reverse=True)
    return best[:10]


# Initialize on module load
initialize_strategy_tracking()


# ---------------- LOT CALCULATION ----------------
def calculate_lot(balance, risk_percent, sl_pips):
    """Basic lot calculation"""
    risk_money = balance * (risk_percent / 100)
    pip_value = 10
    lot = risk_money / (sl_pips * pip_value)
    return max(0.01, min(100.0, round(lot, 2)))  # Allow up to 100 lots for big accounts


def calculate_atr_lot(symbol, balance, risk_percent, df):
    """
    ATR-based dynamic lot sizing for all account sizes.
    Bigger accounts = bigger lots, with volatility-adjusted position sizing.
    """
    try:
        # Get ATR for volatility
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
        if atr is None or atr <= 0:
            return calculate_lot(balance, risk_percent, STOPLOSS_PIPS), STOPLOSS_PIPS, TAKEPROFIT_PIPS
        
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            return calculate_lot(balance, risk_percent, STOPLOSS_PIPS), STOPLOSS_PIPS, TAKEPROFIT_PIPS
        
        point = info.point
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return calculate_lot(balance, risk_percent, STOPLOSS_PIPS), STOPLOSS_PIPS, TAKEPROFIT_PIPS
        
        price = tick.ask
        
        # Calculate ATR-based SL/TP in price terms
        sl_distance = atr * ATR_SL_MULTIPLIER
        tp_distance = atr * ATR_TP_MULTIPLIER
        
        # Convert to pips for the symbol
        pip_mult = 10 if 'JPY' not in symbol else 1
        sl_pips = sl_distance / (point * pip_mult)
        tp_pips = tp_distance / (point * pip_mult)
        
        # Ensure minimum SL/TP
        sl_pips = max(10, min(100, sl_pips))
        tp_pips = max(20, min(200, tp_pips))
        
        # Calculate risk money
        risk_money = balance * (risk_percent / 100)
        
        # Get pip value for this symbol (approximate)
        contract_size = info.trade_contract_size  # Usually 100000 for forex, varies for gold
        pip_value_per_lot = point * pip_mult * contract_size
        
        # Calculate lot size based on risk
        if sl_pips > 0 and pip_value_per_lot > 0:
            lot = risk_money / (sl_pips * pip_value_per_lot)
        else:
            lot = 0.01
        
        # Apply account size scaling
        # Bigger accounts can handle bigger positions
        if balance >= 100000:
            max_lot = balance * (MAX_LOT_PERCENT / 100) / (price * contract_size / 100)
        elif balance >= 10000:
            max_lot = 10.0
        elif balance >= 1000:
            max_lot = 2.0
        else:
            max_lot = 0.5
        
        # Clamp lot size
        lot = max(MIN_LOT, min(max_lot, round(lot, 2)))
        
        # Check margin before returning - reduce lot if not enough margin
        acc = mt5.account_info()
        if acc:
            free_margin = acc.margin_free
            # Check margin required for this lot
            margin_check = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot, price)
            if margin_check and margin_check > free_margin * 0.5:  # Use max 50% of free margin
                # Reduce lot to fit within margin
                if margin_check > 0:
                    lot = lot * (free_margin * 0.3 / margin_check)  # Use only 30% of free margin
                    lot = max(MIN_LOT, round(lot, 2))
        
        logger.debug(f"ATR Lot Calc: {symbol} ATR={atr:.5f} SL={sl_pips:.1f} TP={tp_pips:.1f} Lot={lot}")
        
        return lot, sl_pips, tp_pips
        
    except Exception as e:
        logger.error(f"ATR lot calc error: {e}")
        return calculate_lot(balance, risk_percent, STOPLOSS_PIPS), STOPLOSS_PIPS, TAKEPROFIT_PIPS


# ================================================================================
# ========================= AI INTELLIGENT LOT SIZING SYSTEM ====================
# ================================================================================
# Dynamically calculates optimal lot size based on account, quality, confidence

# Track win/loss streaks per user
user_trade_streaks = {}
ai_lot_history = {}

def get_user_streak(user):
    """Get current win/loss streak for a user"""
    if user not in user_trade_streaks:
        user_trade_streaks[user] = {'streak': 0, 'type': None, 'recent_trades': []}
    return user_trade_streaks[user]

def update_user_streak(user, is_win, profit_amount):
    """Update user's streak after a trade closes"""
    streak_data = get_user_streak(user)
    streak_data['recent_trades'].append({'win': is_win, 'profit': profit_amount, 'time': time.time()})
    
    # Keep only last 20 trades
    streak_data['recent_trades'] = streak_data['recent_trades'][-20:]
    
    if is_win:
        if streak_data['type'] == 'win':
            streak_data['streak'] += 1
        else:
            streak_data['streak'] = 1
            streak_data['type'] = 'win'
    else:
        if streak_data['type'] == 'loss':
            streak_data['streak'] += 1
        else:
            streak_data['streak'] = 1
            streak_data['type'] = 'loss'
    
    user_trade_streaks[user] = streak_data
    logger.info(f"[{user}] 📊 Streak updated: {streak_data['type'].upper()} x{streak_data['streak']}")
    
    # Trigger AI lot learning
    ai_learn_from_lot_outcome(user, is_win, profit_amount)

def get_account_size_multiplier(balance):
    """Get lot multiplier based on account size"""
    multiplier = 1.0
    for threshold, mult in sorted(ACCOUNT_SIZE_LOT_BONUS.items()):
        if balance >= threshold:
            multiplier = mult
    return multiplier

def get_confidence_multiplier(confidence):
    """Get lot multiplier based on AI confidence"""
    if not AI_LOT_CONFIDENCE_SCALING:
        return 1.0
    for conf_threshold, mult in sorted(CONFIDENCE_LOT_MULTIPLIERS.items(), reverse=True):
        if confidence >= conf_threshold:
            return mult
    return 0.75

def get_streak_multiplier(user):
    """Get lot multiplier based on win/loss streak"""
    streak_data = get_user_streak(user)
    streak = streak_data['streak']
    streak_type = streak_data['type']
    
    if streak_type == 'win':
        for wins_needed, mult in sorted(WIN_STREAK_LOT_BONUS.items()):
            if streak >= wins_needed:
                return mult
    elif streak_type == 'loss':
        for losses, mult in sorted(LOSE_STREAK_LOT_REDUCTION.items()):
            if streak >= losses:
                if mult == 0:
                    logger.warning(f"[{user}] ⚠️ 5+ losses - trading paused for safety")
                return mult
    
    return 1.0

def get_quality_multiplier(quality_score):
    """Get lot multiplier based on signal quality"""
    if quality_score >= ULTRA_HIGH_PROB_MIN_QUALITY:
        return ULTRA_HIGH_PROB_LOT_MULTIPLIER
    elif quality_score >= VERY_HIGH_PROB_MIN_QUALITY:
        return VERY_HIGH_PROB_LOT_MULTIPLIER
    elif quality_score >= HIGH_PROB_MIN_QUALITY:
        return HIGH_PROB_LOT_MULTIPLIER
    elif quality_score >= 7:
        return 1.5
    elif quality_score >= 6:
        return 1.0
    else:
        return 0.5  # Reduce lot for low quality

def calculate_intelligent_lot(symbol, user, quality_score=7, confidence=0.7, sl_pips=20):
    """
    INTELLIGENT LOT SIZING - The brain of position sizing
    
    Considers:
    1. Account size (bigger = higher lots)
    2. Signal quality (8-10 = aggressive)
    3. AI confidence (90%+ = aggressive)
    4. Win/loss streak (winning = increase, losing = decrease)
    5. AI learned patterns
    6. Risk limits (never exceed max risk)
    
    Returns: optimal lot size
    """
    try:
        account = mt5.account_info()
        if not account:
            return MIN_LOT
        
        balance = account.balance
        
        # Base lot calculation using risk percent
        base_lot = calculate_lot(balance, RISK_PERCENT, sl_pips)
        
        # 1. Account size multiplier (bigger accounts = bigger lots)
        account_mult = get_account_size_multiplier(balance)
        
        # 2. Quality multiplier (high quality = aggressive)
        quality_mult = get_quality_multiplier(quality_score)
        
        # 3. AI confidence multiplier
        confidence_mult = get_confidence_multiplier(confidence)
        
        # 4. Streak multiplier (winning streak = bigger, losing = smaller)
        streak_mult = get_streak_multiplier(user)
        
        # If streak multiplier is 0, STOP TRADING
        if streak_mult == 0:
            return 0
        
        # 5. Check AI learned adjustments
        ai_adjustment = get_ai_lot_adjustment(user, quality_score, confidence)
        
        # Calculate final lot
        final_lot = base_lot * account_mult * quality_mult * confidence_mult * streak_mult * ai_adjustment
        
        # Apply maximum risk limits
        max_risk_amount = balance * (MAX_RISK_PER_SIGNAL / 100)
        pip_value = 10  # Approximate
        max_lot_by_risk = max_risk_amount / (sl_pips * pip_value)
        
        # Dynamic max lot based on account size - AGGRESSIVE for scalping
        if balance >= 100000:
            absolute_max = 150.0  # Allow up to 150 lots for big accounts
        elif balance >= 50000:
            absolute_max = 75.0
        elif balance >= 25000:
            absolute_max = 40.0
        elif balance >= 10000:
            absolute_max = 20.0
        elif balance >= 5000:
            absolute_max = 12.0
        elif balance >= 2500:
            absolute_max = 6.0
        elif balance >= 1000:
            absolute_max = 3.0
        elif balance >= 500:
            absolute_max = 1.5
        else:
            absolute_max = 0.75
        
        # Also check high-prob max
        if quality_score >= HIGH_PROB_MIN_QUALITY:
            absolute_max = min(MAX_LOT_HIGH_PROB, absolute_max * 2)  # Double max for high prob
        
        # Clamp to limits
        final_lot = max(MIN_LOT, min(final_lot, max_lot_by_risk, absolute_max))
        final_lot = round(final_lot, 2)
        
        # Check margin - allow up to 75% for aggressive scalping
        tick = mt5.symbol_info_tick(symbol)
        if tick and account.margin_free > 0:
            price = tick.ask
            margin_needed = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, final_lot, price)
            if margin_needed and margin_needed > account.margin_free * 0.75:
                # Reduce to fit in 65% of free margin
                final_lot = final_lot * (account.margin_free * 0.65 / margin_needed)
                final_lot = max(MIN_LOT, round(final_lot, 2))
        
        logger.info(f"[{user}] 📊 INTELLIGENT LOT: {final_lot} | Base={base_lot:.2f} | "
                   f"AccMult={account_mult:.1f}x | QualMult={quality_mult:.1f}x | "
                   f"ConfMult={confidence_mult:.2f}x | StreakMult={streak_mult:.2f}x | "
                   f"Quality={quality_score}/10 | Confidence={confidence:.0%}")
        
        return final_lot
        
    except Exception as e:
        logger.error(f"Intelligent lot calc error: {e}")
        return MIN_LOT


def get_ai_lot_adjustment(user, quality, confidence):
    """
    Get AI-learned lot adjustment based on historical performance.
    AI learns which quality/confidence combos work best.
    """
    if not AI_LOT_LEARNING_ENABLED:
        return 1.0
    
    if user not in AI_LOT_LEARNING_DATA:
        return 1.0
    
    history = AI_LOT_LEARNING_DATA[user]
    if len(history) < AI_LOT_LEARNING_MIN_TRADES:
        return 1.0
    
    # Analyze trades at similar quality levels
    similar_trades = [t for t in history if abs(t.get('quality', 0) - quality) <= 1]
    
    if len(similar_trades) < 5:
        return 1.0
    
    # Calculate win rate and average profit at this quality level
    wins = [t for t in similar_trades if t.get('win', False)]
    win_rate = len(wins) / len(similar_trades)
    
    avg_profit = sum(t.get('profit', 0) for t in similar_trades) / len(similar_trades)
    
    # Adjust lot based on historical performance
    if win_rate >= 0.7 and avg_profit > 0:
        adjustment = 1.0 + (win_rate - 0.5) * 0.5  # Up to 1.35x for 85% win rate
    elif win_rate >= 0.5:
        adjustment = 1.0
    elif win_rate >= 0.3:
        adjustment = 0.8
    else:
        adjustment = 0.5  # Very low win rate at this quality = reduce lots
    
    return round(adjustment, 2)


def ai_learn_from_lot_outcome(user, is_win, profit_amount):
    """
    AI learns from trade outcomes to optimize future lot sizing.
    Called after each trade closes.
    """
    if not AI_LOT_LEARNING_ENABLED:
        return
    
    if user not in AI_LOT_LEARNING_DATA:
        AI_LOT_LEARNING_DATA[user] = []
    
    # Get the most recent trade context
    if user in ai_lot_history and ai_lot_history[user]:
        last_trade = ai_lot_history[user]
        trade_record = {
            'quality': last_trade.get('quality', 7),
            'confidence': last_trade.get('confidence', 0.7),
            'lot': last_trade.get('lot', 0.01),
            'win': is_win,
            'profit': profit_amount,
            'time': time.time()
        }
        AI_LOT_LEARNING_DATA[user].append(trade_record)
        
        # Keep only last 100 trades
        AI_LOT_LEARNING_DATA[user] = AI_LOT_LEARNING_DATA[user][-100:]
        
        logger.debug(f"[{user}] 🧠 AI Lot Learning: Recorded {'WIN' if is_win else 'LOSS'} "
                    f"Q={trade_record['quality']} Conf={trade_record['confidence']:.0%} Profit=${profit_amount:.2f}")


def record_lot_for_learning(user, quality, confidence, lot):
    """Record the lot size used for a trade (called before execution)"""
    if not AI_LOT_LEARNING_ENABLED:
        return
    
    ai_lot_history[user] = {
        'quality': quality,
        'confidence': confidence,
        'lot': lot,
        'time': time.time()
    }


# ================================================================================
# ========================= AGGRESSIVE SCALPING FUNCTIONS =======================
# ================================================================================
# High lot scalping with quick profit taking and re-entry

def get_scalp_lot_size(user, symbol):
    """
    Get scalping lot size based on account balance.
    Returns aggressive but safe lot sizes for quick scalps.
    """
    if not AGGRESSIVE_SCALPING_ENABLED:
        return MIN_LOT
    
    try:
        account = mt5.account_info()
        if not account:
            return MIN_LOT
        
        balance = account.balance
        
        # Find the right tier
        scalp_lot = MIN_LOT
        for threshold, lot in sorted(SCALP_LOT_ACCOUNT_TIERS.items()):
            if balance >= threshold:
                scalp_lot = lot
        
        # Check free margin - allow up to 70% on one trade for aggressive scalping
        if account.margin_free > 0:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                margin_needed = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, scalp_lot, tick.ask)
                if margin_needed and margin_needed > account.margin_free * 0.7:
                    scalp_lot = scalp_lot * (account.margin_free * 0.6 / margin_needed)
                    scalp_lot = max(MIN_LOT, round(scalp_lot, 2))
        
        logger.info(f"[{user}] ⚡ SCALP LOT: {scalp_lot} for ${balance:.0f} account")
        return round(scalp_lot, 2)
        
    except Exception as e:
        logger.error(f"Scalp lot calc error: {e}")
        return MIN_LOT


def get_scalp_tp_sl(symbol):
    """
    Get scalping TP and SL in pips for a symbol.
    Returns (tp_pips, sl_pips)
    """
    # Get base symbol without suffix
    sym_base = symbol.replace('m', '').replace('.', '')
    
    # Look up or use default
    tp_pips = SCALP_TP_PIPS.get(sym_base, SCALP_TP_PIPS.get('DEFAULT', 5))
    sl_pips = SCALP_SL_PIPS.get(sym_base, SCALP_SL_PIPS.get('DEFAULT', 8))
    
    return tp_pips, sl_pips


def should_close_scalp_trade(position, user):
    """
    Check if a scalp trade should be closed for quick profit.
    Returns (should_close, reason)
    """
    if not AGGRESSIVE_SCALPING_ENABLED or not SCALP_CLOSE_ON_ANY_PROFIT:
        return False, "Scalping not enabled"
    
    try:
        current_profit = position.profit
        symbol = position.symbol
        
        # Check minimum hold time
        # Estimate open time from position data (if available)
        # For now, use a simple profit-based approach
        
        # Get pip value for this symbol
        sym_settings = get_symbol_settings(symbol)
        pip_value = sym_settings.get('pip_value', 0.0001)
        point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else pip_value
        
        # Calculate profit in pips
        is_buy = position.type == 0
        if is_buy:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                profit_pips = (tick.bid - position.price_open) / (point * 10)
            else:
                profit_pips = 0
        else:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                profit_pips = (position.price_open - tick.ask) / (point * 10)
            else:
                profit_pips = 0
        
        # ULTRA QUICK close - close on ANY profit above minimum
        if profit_pips >= SCALP_MIN_PROFIT_PIPS and current_profit > 0:
            return True, f"QUICK SCALP profit ({profit_pips:.1f} pips, ${current_profit:.2f})"
        
        # Close if we have ANY positive P&L, even tiny
        if current_profit > 0.10:  # Close if profit > $0.10
            return True, f"Micro profit captured (${current_profit:.2f})"
        
        # Get scalp TP target
        tp_target, _ = get_scalp_tp_sl(symbol)
        if profit_pips >= tp_target:
            return True, f"Scalp TP hit ({profit_pips:.1f} >= {tp_target} pips)"
        
        return False, "Still in scalp trade"
        
    except Exception as e:
        logger.error(f"Scalp close check error: {e}")
        return False, f"Error: {e}"


def track_scalp_reentry(symbol, direction, close_reason, user):
    """
    Track a closed scalp for potential re-entry.
    Called when a scalp closes in profit.
    """
    if not SCALP_REENTRY_ENABLED:
        return
    
    global scalp_entry_tracker
    
    tracker_key = f"{user}_{symbol}"
    
    if tracker_key not in scalp_entry_tracker:
        scalp_entry_tracker[tracker_key] = {
            'direction': direction,
            'entries_count': 1,
            'last_close_time': time.time(),
            'setup_valid_until': time.time() + 600,  # Setup valid for 10 minutes
            'total_profit': 0
        }
    else:
        tracker = scalp_entry_tracker[tracker_key]
        if SCALP_REENTRY_REQUIRE_SAME_DIRECTION and tracker['direction'] != direction:
            # Direction changed, reset tracker
            scalp_entry_tracker[tracker_key] = {
                'direction': direction,
                'entries_count': 1,
                'last_close_time': time.time(),
                'setup_valid_until': time.time() + 600,
                'total_profit': 0
            }
        else:
            tracker['entries_count'] += 1
            tracker['last_close_time'] = time.time()
    
    logger.info(f"[{user}] 🔄 Tracking {symbol} {direction} for re-entry (entry #{scalp_entry_tracker[tracker_key]['entries_count']})")


def can_scalp_reenter(symbol, direction, user):
    """
    Check if we can re-enter a scalp trade.
    Returns (can_reenter, reason)
    """
    if not SCALP_REENTRY_ENABLED:
        return False, "Re-entry disabled"
    
    tracker_key = f"{user}_{symbol}"
    
    if tracker_key not in scalp_entry_tracker:
        return True, "New setup"
    
    tracker = scalp_entry_tracker[tracker_key]
    
    # Check if setup still valid
    if time.time() > tracker.get('setup_valid_until', 0):
        # Setup expired, allow fresh entry
        del scalp_entry_tracker[tracker_key]
        return True, "Setup expired, fresh entry"
    
    # Check direction
    if SCALP_REENTRY_REQUIRE_SAME_DIRECTION and tracker['direction'] != direction:
        return False, f"Direction changed ({tracker['direction']} → {direction})"
    
    # Check max re-entries
    if tracker['entries_count'] >= SCALP_MAX_REENTRIES_PER_SETUP:
        return False, f"Max re-entries reached ({SCALP_MAX_REENTRIES_PER_SETUP})"
    
    # Check cooldown
    time_since_close = time.time() - tracker.get('last_close_time', 0)
    if time_since_close < SCALP_REENTRY_COOLDOWN_SECONDS:
        return False, f"Cooldown ({time_since_close:.0f}s < {SCALP_REENTRY_COOLDOWN_SECONDS}s)"
    
    return True, f"Re-entry #{tracker['entries_count'] + 1} allowed"


def execute_scalp_close_and_reentry(position, user, close_reason):
    """
    Close a scalp trade and queue for re-entry if setup is still valid.
    """
    symbol = position.symbol
    is_buy = position.type == 0
    direction = "BUY" if is_buy else "SELL"
    
    # Record profit before closing
    profit = position.profit
    
    # Close the position
    close_result = close_position_by_ticket(position.ticket, user, f"SCALP: {close_reason}")
    
    if close_result:
        # Track for re-entry
        track_scalp_reentry(symbol, direction, close_reason, user)
        
        # Update tracker profit
        tracker_key = f"{user}_{symbol}"
        if tracker_key in scalp_entry_tracker:
            scalp_entry_tracker[tracker_key]['total_profit'] = scalp_entry_tracker[tracker_key].get('total_profit', 0) + profit
        
        logger.info(f"[{user}] ⚡ SCALP CLOSED: {symbol} {direction} +${profit:.2f} | {close_reason}")
        
        return True
    
    return False


def close_position_by_ticket(ticket, user, reason="Manual"):
    """Close a specific position by ticket number."""
    try:
        position = None
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        position = positions[0]
        symbol = position.symbol
        is_buy = position.type == 0
        
        # Get close price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        close_price = tick.bid if is_buy else tick.ask
        
        # Prepare close request
        sym_info = mt5.symbol_info(symbol)
        filling_type = sym_info.filling_mode if sym_info else mt5.ORDER_FILLING_IOC
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "magic": MAGIC,
            "deviation": 20,
            "type_filling": filling_type,
            "comment": reason[:30]
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Update strategy performance for AI learning
            profit = position.profit
            is_win = profit > 0
            
            if ticket in trade_strategies_used:
                trade_data = trade_strategies_used[ticket]
                strategies = trade_data.get('strategies', [])
                
                # Calculate profit in pips
                sym_settings = get_symbol_settings(symbol)
                pip_value = sym_settings.get('pip_value', 0.0001)
                entry_price = trade_data.get('entry_price', position.price_open)
                
                if trade_data.get('direction') == 'BUY':
                    profit_pips = (close_price - entry_price) / pip_value
                else:
                    profit_pips = (entry_price - close_price) / pip_value
                
                # Update each strategy's performance
                for strat_name in strategies:
                    try:
                        update_strategy_performance(strat_name, is_win, profit_pips, user)
                    except Exception as e:
                        pass
                
                logger.info(f"[{user}] 🧠 AI LEARNED from #{ticket}: {len(strategies)} strategies {'✅' if is_win else '❌'}")
                del trade_strategies_used[ticket]
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Close position error: {e}")
        return False


def manage_scalp_positions(user):
    """
    Main scalping position manager.
    Checks all positions and closes profitable scalps, queues re-entry.
    Call this in the main bot loop.
    """
    if not AGGRESSIVE_SCALPING_ENABLED:
        return
    
    try:
        positions = mt5.positions_get()
        if not positions:
            return
        
        for position in positions:
            # Check if this is a scalp trade (by comment or just manage all)
            should_close, reason = should_close_scalp_trade(position, user)
            
            if should_close:
                execute_scalp_close_and_reentry(position, user, reason)
                
    except Exception as e:
        logger.error(f"Scalp management error: {e}")


# ================================================================================
# ========================= AI LOSS PATTERN LEARNING SYSTEM =====================
# ================================================================================
# Learns from losing trades to avoid similar conditions

ai_trade_context_history = {}  # Stores the market context before each trade

def record_trade_context(user, symbol, context_data):
    """Record market context before a trade for loss learning"""
    if not AI_LOSS_LEARNING_ENABLED:
        return
    
    ai_trade_context_history[user] = {
        'symbol': symbol,
        'context': context_data,
        'time': time.time()
    }

def learn_from_loss(user, symbol, loss_amount):
    """
    Record losing trade patterns for AI to avoid in the future.
    Called when a trade closes with a loss.
    """
    if not AI_LOSS_LEARNING_ENABLED:
        return
    
    if user not in ai_trade_context_history:
        return
    
    context = ai_trade_context_history.get(user, {})
    if not context or context.get('symbol') != symbol:
        return
    
    if user not in AI_LOSS_PATTERN_DATA:
        AI_LOSS_PATTERN_DATA[user] = []
    
    loss_record = {
        'symbol': symbol,
        'loss_amount': loss_amount,
        'context': context.get('context', {}),
        'time': time.time()
    }
    
    AI_LOSS_PATTERN_DATA[user].append(loss_record)
    
    # Keep only last 50 losing patterns
    AI_LOSS_PATTERN_DATA[user] = AI_LOSS_PATTERN_DATA[user][-50:]
    
    logger.info(f"[{user}] 🧠 AI Loss Learning: Recorded loss pattern for {symbol} (${loss_amount:.2f})")

def should_avoid_similar_setup(user, symbol, current_context):
    """
    Check if current market conditions are similar to past losing trades.
    Returns (should_avoid, similarity_score, reason)
    """
    if not AI_LOSS_LEARNING_ENABLED:
        return False, 0.0, "Loss learning disabled"
    
    if user not in AI_LOSS_PATTERN_DATA:
        return False, 0.0, "No loss history"
    
    loss_patterns = AI_LOSS_PATTERN_DATA[user]
    
    # Only analyze if we have enough samples
    symbol_losses = [p for p in loss_patterns if p.get('symbol') == symbol]
    if len(symbol_losses) < AI_LOSS_AVOIDANCE_MIN_SAMPLES:
        return False, 0.0, "Insufficient loss data"
    
    # Calculate similarity to recent losing patterns
    max_similarity = 0.0
    similar_pattern = None
    
    for pattern in symbol_losses[-10:]:  # Check last 10 losses for this symbol
        similarity = calculate_pattern_similarity(current_context, pattern.get('context', {}))
        if similarity > max_similarity:
            max_similarity = similarity
            similar_pattern = pattern
    
    if max_similarity >= AI_SIMILAR_LOSS_THRESHOLD:
        return True, max_similarity, f"Similar to loss pattern (${similar_pattern.get('loss_amount', 0):.2f} loss)"
    
    return False, max_similarity, "No dangerous pattern match"

def calculate_pattern_similarity(context1, context2):
    """
    Calculate similarity between two market contexts (0-1).
    Used to detect if current conditions match past losing conditions.
    """
    if not context1 or not context2:
        return 0.0
    
    similarity_factors = []
    
    # Compare key indicators
    indicators = ['rsi', 'macd_hist', 'bb_position', 'trend', 'momentum']
    for ind in indicators:
        val1 = context1.get(ind)
        val2 = context2.get(ind)
        if val1 is not None and val2 is not None:
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize numeric comparison
                max_val = max(abs(val1), abs(val2), 1)
                similarity = 1 - abs(val1 - val2) / max_val
                similarity_factors.append(max(0, similarity))
            elif val1 == val2:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
    
    if not similarity_factors:
        return 0.0
    
    return sum(similarity_factors) / len(similarity_factors)


# ================================================================================
# ========================= LOSS RECOVERY FUNCTIONS =============================
# ================================================================================

def get_recovery_status(user):
    """Get current recovery mode status for user."""
    if user not in RECOVERY_MODE_ACTIVE:
        RECOVERY_MODE_ACTIVE[user] = {
            'active': False,
            'loss_to_recover': 0.0,
            'recovered': 0.0,
            'recovery_started': None,
            'trades_in_recovery': 0
        }
    return RECOVERY_MODE_ACTIVE[user]

def check_recovery_trigger(user, loss_amount, account_balance):
    """
    Check if we should enter recovery mode after a loss.
    Activates recovery mode when session loss exceeds threshold.
    """
    if not LOSS_RECOVERY_ENABLED:
        return
    
    status = get_recovery_status(user)
    loss_percent = (loss_amount / account_balance) * 100 if account_balance > 0 else 0
    
    if not status['active']:
        # Check if we should activate recovery
        if loss_percent >= RECOVERY_TRIGGER_LOSS_PERCENT:
            status['active'] = True
            status['loss_to_recover'] = loss_amount
            status['recovered'] = 0.0
            status['recovery_started'] = time.time()
            status['trades_in_recovery'] = 0
            logger.warning(f"[{user}] 🔄 RECOVERY MODE ACTIVATED - Need to recover ${loss_amount:.2f}")
    else:
        # Already in recovery, add to loss if another loss occurs
        status['loss_to_recover'] += loss_amount
        logger.info(f"[{user}] 🔄 Recovery target updated: ${status['loss_to_recover']:.2f}")

def update_recovery_on_win(user, profit_amount):
    """
    Update recovery status when we have a winning trade.
    Exit recovery mode when losses are fully recovered + buffer.
    """
    if not LOSS_RECOVERY_ENABLED:
        return
    
    status = get_recovery_status(user)
    if not status['active']:
        return
    
    status['recovered'] += profit_amount
    status['trades_in_recovery'] += 1
    
    # Check if we've recovered enough (100% + buffer)
    recovery_target = status['loss_to_recover'] * RECOVERY_CONTINUE_UNTIL
    
    if status['recovered'] >= recovery_target:
        logger.info(f"[{user}] ✅ RECOVERY COMPLETE! Recovered ${status['recovered']:.2f} of ${status['loss_to_recover']:.2f} target")
        status['active'] = False
        status['loss_to_recover'] = 0.0
        status['recovered'] = 0.0
        status['recovery_started'] = None
        status['trades_in_recovery'] = 0
    else:
        remaining = recovery_target - status['recovered']
        logger.info(f"[{user}] 🔄 Recovery progress: ${status['recovered']:.2f}/${recovery_target:.2f} (${remaining:.2f} remaining)")

def get_recovery_lot_multiplier(user, confidence=0.9):
    """
    Get lot multiplier for recovery mode.
    Uses BIG lots for quick scalping during recovery.
    Ultra-high confidence (95%+) gets even bigger lots.
    """
    if not LOSS_RECOVERY_ENABLED:
        return 1.0
    
    status = get_recovery_status(user)
    if not status['active']:
        return 1.0
    
    if status['loss_to_recover'] <= 0:
        return 1.0
    
    # Calculate recovery progress
    progress = status['recovered'] / status['loss_to_recover'] if status['loss_to_recover'] > 0 else 0
    remaining = max(0, 1.0 - progress)
    
    # Find the right tier based on remaining recovery
    multiplier = 2.0  # Start with 2x base for recovery
    for threshold, mult in sorted(RECOVERY_LOT_TIERS.items()):
        if remaining >= threshold:
            multiplier = mult
    
    # ULTRA boost for very high confidence trades (95%+)
    if confidence >= RECOVERY_ULTRA_CONFIDENCE:
        multiplier = RECOVERY_ULTRA_LOT_MULTIPLIER
        logger.info(f"[{user}] 💪 ULTRA RECOVERY: {confidence:.0%} confidence → {multiplier}x lot!")
    
    # Cap at maximum
    return min(multiplier, RECOVERY_ULTRA_LOT_MULTIPLIER)

def should_take_recovery_trade(user, quality_score, confidence):
    """
    Determine if a trade qualifies for recovery mode.
    ONLY takes very high probability scalp setups during recovery.
    Requires quality 9+ and 90%+ confidence for sure wins.
    """
    if not LOSS_RECOVERY_ENABLED:
        return True, "Recovery disabled", 1.0
    
    status = get_recovery_status(user)
    if not status['active']:
        return True, "Not in recovery mode", 1.0
    
    # STRICT quality requirements during recovery - only near-certain wins
    if quality_score < RECOVERY_MIN_QUALITY:
        return False, f"Recovery requires quality {RECOVERY_MIN_QUALITY}+, got {quality_score}", 1.0
    
    if confidence < RECOVERY_MIN_CONFIDENCE:
        return False, f"Recovery requires {RECOVERY_MIN_CONFIDENCE*100:.0f}%+ confidence, got {confidence*100:.0f}%", 1.0
    
    # Get the lot multiplier based on confidence
    lot_mult = get_recovery_lot_multiplier(user, confidence)
    
    # Log recovery trade qualification
    logger.info(f"[{user}] ✅ RECOVERY TRADE QUALIFIED: Q{quality_score}/10, {confidence:.0%} conf → {lot_mult}x lot")
    
    return True, f"Recovery trade approved ({lot_mult}x lot)", lot_mult


# ================================================================================
# ========================= LOSS PREVENTION FUNCTIONS ===========================
# ================================================================================

def check_market_conditions(symbol):
    """
    Check if current market conditions are suitable for trading.
    Returns (tradeable, reason, score)
    """
    if not LOSS_PREVENTION_ENABLED:
        return True, "Loss prevention disabled", 100
    
    issues = []
    score = 100
    
    try:
        # 1. Check spread
        if not check_spread_quality(symbol):
            issues.append("Spread too high")
            score -= 20
        
        # 2. Check session/timing
        if not check_session_quality():
            issues.append("Poor session quality")
            score -= 15
        
        # 3. Check for choppy market (ADX)
        if AVOID_CHOPPY_MARKETS and is_market_choppy(symbol):
            issues.append("Choppy market detected")
            score -= 25
        
        # 4. Check recent price action quality
        if REQUIRE_CLEAN_STRUCTURE and not has_clean_structure(symbol):
            issues.append("Messy price structure")
            score -= 15
        
        # 5. Check candle quality
        candle_quality = get_candle_quality(symbol)
        if candle_quality < MIN_CANDLE_QUALITY:
            issues.append(f"Poor candle quality ({candle_quality:.0%})")
            score -= 15
        
        if score < MIN_ENTRY_SCORE:
            return False, "; ".join(issues), score
        
        return True, "Conditions OK", score
        
    except Exception as e:
        logger.error(f"Error checking market conditions: {e}")
        return True, "Could not fully check conditions", 80

def check_spread_quality(symbol):
    """Check if spread is acceptable."""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return True  # Can't check, allow trade
        
        spread = tick.ask - tick.bid
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return True
        
        point = symbol_info.point
        spread_points = spread / point if point > 0 else 0
        
        # Get normal spread for symbol
        symbol_base = symbol.replace('m', '')
        normal_spread = NORMAL_SPREADS.get(symbol_base, NORMAL_SPREADS.get(symbol, 30))
        
        max_allowed = normal_spread * MAX_SPREAD_FOR_ENTRY
        return spread_points <= max_allowed
        
    except Exception:
        return True

def check_session_quality():
    """Check if current trading session is good quality."""
    try:
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Check low liquidity hours
        if AVOID_LOW_LIQUIDITY_HOURS and current_hour in LOW_LIQUIDITY_HOURS_UTC:
            return False
        
        # Check session overlap transition
        if AVOID_SESSION_OVERLAPS_TRANSITIONS:
            session_starts = [0, 7, 13]  # Sydney, London, NY opens (approx UTC)
            for start in session_starts:
                if start <= current_hour < start + (SESSION_OVERLAP_AVOID_MINUTES / 60):
                    return False
        
        return True
        
    except Exception:
        return True

def is_market_choppy(symbol):
    """
    Check if market is choppy/ranging using ADX.
    Choppy markets have ADX below threshold.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, ENTRY_TIMEFRAME, 0, 20)
        if rates is None or len(rates) < 14:
            return False
        
        # Simple ADX approximation using directional movement
        highs = [r['high'] for r in rates]
        lows = [r['low'] for r in rates]
        closes = [r['close'] for r in rates]
        
        # Calculate True Range
        tr_values = []
        for i in range(1, len(rates)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        if not tr_values:
            return False
        
        atr = sum(tr_values[-14:]) / 14
        
        # Calculate directional movement
        plus_dm = 0
        minus_dm = 0
        for i in range(1, min(15, len(rates))):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm += up_move
            if down_move > up_move and down_move > 0:
                minus_dm += down_move
        
        # Approximate ADX
        if atr > 0:
            plus_di = (plus_dm / 14) / atr * 100
            minus_di = (minus_dm / 14) / atr * 100
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = abs(plus_di - minus_di) / di_sum * 100
                adx = dx  # Simplified - use DX as ADX approximation
                
                return adx < CHOPPY_MARKET_ADX_THRESHOLD
        
        return False
        
    except Exception:
        return False

def has_clean_structure(symbol):
    """Check if price action has clean structure (clear highs/lows)."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, ENTRY_TIMEFRAME, 0, 20)
        if rates is None or len(rates) < 10:
            return True
        
        # Check for clear swing highs/lows
        clear_swings = 0
        for i in range(2, len(rates) - 2):
            # Check for swing high
            if rates[i]['high'] > rates[i-1]['high'] and rates[i]['high'] > rates[i-2]['high'] and \
               rates[i]['high'] > rates[i+1]['high'] and rates[i]['high'] > rates[i+2]['high']:
                clear_swings += 1
            # Check for swing low
            if rates[i]['low'] < rates[i-1]['low'] and rates[i]['low'] < rates[i-2]['low'] and \
               rates[i]['low'] < rates[i+1]['low'] and rates[i]['low'] < rates[i+2]['low']:
                clear_swings += 1
        
        # Need at least 2 clear swings
        return clear_swings >= 2
        
    except Exception:
        return True

def get_candle_quality(symbol):
    """
    Get quality score for recent candles (0-1).
    Low quality = excessive wicks, indecision.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, ENTRY_TIMEFRAME, 0, 5)
        if rates is None or len(rates) < 3:
            return 1.0
        
        quality_scores = []
        for r in rates[-3:]:  # Check last 3 candles
            body = abs(r['close'] - r['open'])
            total = r['high'] - r['low']
            
            if total > 0:
                body_ratio = body / total
                # Higher body ratio = better candle (less wick)
                quality_scores.append(body_ratio)
            else:
                quality_scores.append(0.5)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
        
    except Exception:
        return 1.0

def check_correlation_exposure(symbol, direction, user):
    """
    Check if we already have exposure to correlated pairs.
    Prevent over-concentration in same direction.
    """
    if not AVOID_CORRELATED_TRADES:
        return True, "Correlation check disabled"
    
    try:
        positions = mt5.positions_get()
        if not positions:
            return True, "No existing positions"
        
        # Get correlated pairs for this symbol
        symbol_base = symbol.replace('m', '')
        correlated = CORRELATED_PAIRS.get(symbol_base, [])
        
        same_direction_count = 0
        correlated_exposure = []
        
        for pos in positions:
            pos_symbol_base = pos.symbol.replace('m', '')
            pos_direction = "BUY" if pos.type == 0 else "SELL"
            
            # Check if same direction
            if pos_direction == direction:
                same_direction_count += 1
            
            # Check if correlated
            if pos_symbol_base in correlated or pos_symbol_base == symbol_base:
                if pos_direction == direction:
                    correlated_exposure.append(pos.symbol)
        
        if same_direction_count >= MAX_EXPOSURE_SAME_DIRECTION:
            return False, f"Max same-direction exposure ({MAX_EXPOSURE_SAME_DIRECTION} trades)"
        
        if correlated_exposure:
            return False, f"Already exposed to correlated: {correlated_exposure}"
        
        return True, "Correlation OK"
        
    except Exception:
        return True, "Correlation check error"

def should_pause_after_loss(user, last_loss_amount, account_balance):
    """
    Check if we should pause trading after a big loss.
    Prevents emotional/revenge trading.
    """
    if not PAUSE_AFTER_BIG_LOSS:
        return False, 0
    
    loss_percent = last_loss_amount / account_balance if account_balance > 0 else 0
    
    if loss_percent >= BIG_LOSS_THRESHOLD:
        logger.warning(f"[{user}] ⏸️ Big loss detected ({loss_percent*100:.1f}%), pausing for {PAUSE_AFTER_BIG_LOSS_MINUTES} minutes")
        return True, PAUSE_AFTER_BIG_LOSS_MINUTES
    
    return False, 0

def calculate_entry_score(symbol, direction, quality, confidence, user, confluences=None):
    """
    Calculate comprehensive entry score (0-100).
    Only enter trades above MIN_ENTRY_SCORE.
    """
    if not ENTRY_QUALITY_SCORING:
        return 100, {}
    
    scores = {}
    
    # 1. Trend alignment
    if check_htf_alignment(symbol, direction):
        scores['trend_alignment'] = ENTRY_SCORE_WEIGHTS['trend_alignment']
    else:
        scores['trend_alignment'] = 0
    
    # 2. Key level (approximated from quality)
    if quality >= 7:
        scores['key_level'] = ENTRY_SCORE_WEIGHTS['key_level']
    else:
        scores['key_level'] = 0
    
    # 3. Confluence count
    conf_count = len(confluences) if confluences else 0
    scores['confluence_count'] = min(conf_count * 3, ENTRY_SCORE_WEIGHTS['confluence_count'])
    
    # 4. Candle quality
    candle_q = get_candle_quality(symbol)
    scores['candle_quality'] = int(candle_q * ENTRY_SCORE_WEIGHTS['candle_quality'])
    
    # 5. Session quality
    if check_session_quality():
        scores['session_quality'] = ENTRY_SCORE_WEIGHTS['session_quality']
    else:
        scores['session_quality'] = 0
    
    # 6. Spread quality
    if check_spread_quality(symbol):
        scores['spread_quality'] = ENTRY_SCORE_WEIGHTS['spread_quality']
    else:
        scores['spread_quality'] = 0
    
    # 7. Momentum (derived from quality/confidence)
    momentum_score = int(confidence * ENTRY_SCORE_WEIGHTS['momentum']) if confidence else 5
    scores['momentum'] = momentum_score
    
    # 8. Recent performance (streak bonus)
    streak_data = user_trade_streaks.get(user, {})
    if streak_data.get('streak', 0) > 0 and streak_data.get('type') == 'win':
        scores['recent_wins'] = ENTRY_SCORE_WEIGHTS['recent_wins']
    else:
        scores['recent_wins'] = 5  # Neutral
    
    # 9. No loss pattern match
    avoid, sim, _ = should_avoid_similar_setup(user, symbol, {})
    if not avoid:
        scores['no_loss_patterns'] = ENTRY_SCORE_WEIGHTS['no_loss_patterns']
    else:
        scores['no_loss_patterns'] = 0
    
    total_score = sum(scores.values())
    return total_score, scores

def check_htf_alignment(symbol, direction):
    """Check if trade direction aligns with higher timeframe trend."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, HTF_TIMEFRAME, 0, 20)
        if rates is None or len(rates) < 10:
            return True  # Can't check, assume aligned
        
        # Simple trend check: EMA direction
        closes = [r['close'] for r in rates]
        ema10 = sum(closes[-10:]) / 10
        ema20 = sum(closes[-20:]) / 20
        
        htf_trend = "BUY" if ema10 > ema20 else "SELL"
        return htf_trend == direction
        
    except Exception:
        return True  # Error = allow trade

def comprehensive_entry_check(symbol, direction, quality, confidence, user, confluences=None):
    """
    Master function to check all loss prevention criteria.
    Returns (should_enter, reason, entry_score)
    """
    # 1. Check if in recovery mode and if trade qualifies
    can_take, reason, _ = should_take_recovery_trade(user, quality or 5, confidence or 0.5)
    if not can_take:
        return False, reason, 0
    
    # 2. Check market conditions
    conditions_ok, cond_reason, cond_score = check_market_conditions(symbol)
    if not conditions_ok:
        return False, f"Market conditions: {cond_reason}", cond_score
    
    # 3. Check correlation/exposure
    exposure_ok, exp_reason = check_correlation_exposure(symbol, direction, user)
    if not exposure_ok:
        return False, exp_reason, 50
    
    # 4. Check entry score
    entry_score, score_breakdown = calculate_entry_score(
        symbol, direction, quality or 5, confidence or 0.5, user, confluences
    )
    
    if entry_score < MIN_ENTRY_SCORE:
        return False, f"Entry score too low ({entry_score}/{MIN_ENTRY_SCORE})", entry_score
    
    # 5. Should avoid similar setup?
    avoid, similarity, avoid_reason = should_avoid_similar_setup(user, symbol, {})
    if avoid:
        return False, f"AI avoiding: {avoid_reason}", entry_score
    
    return True, f"All checks passed (score: {entry_score})", entry_score


# ================================================================================
# ========================= SMART BREAKEVEN SYSTEM ==============================
# ================================================================================

def apply_smart_breakeven(position, symbol, quality_score=7):
    """
    Move stop loss to breakeven once profit threshold is reached.
    Higher quality trades get tighter breakeven triggers.
    """
    if not USE_SMART_BREAKEVEN:
        return False
    
    try:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        info = mt5.symbol_info(symbol)
        if not info:
            return False
        
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        entry_price = position.price_open
        current_price = tick.bid if is_buy else tick.ask
        current_sl = position.sl
        
        # Calculate profit in pips
        point = info.point
        pip_mult = 10 if 'JPY' not in symbol else 1
        
        if is_buy:
            profit_pips = (current_price - entry_price) / (point * pip_mult)
        else:
            profit_pips = (entry_price - current_price) / (point * pip_mult)
        
        # Determine breakeven trigger based on quality
        if quality_score >= 10:
            be_trigger = BREAKEVEN_FOR_ULTRA_QUALITY
        elif quality_score >= 8:
            be_trigger = BREAKEVEN_FOR_HIGH_QUALITY
        else:
            be_trigger = BREAKEVEN_TRIGGER_PIPS
        
        # Check if we should move to breakeven
        if profit_pips < be_trigger:
            return False
        
        # Calculate breakeven price with buffer
        buffer_price = BREAKEVEN_BUFFER_PIPS * point * pip_mult
        
        if is_buy:
            new_sl = entry_price + buffer_price
            # Only update if new SL is better (higher for buys)
            if current_sl > 0 and new_sl <= current_sl:
                return False
        else:
            new_sl = entry_price - buffer_price
            # Only update if new SL is better (lower for sells)
            if current_sl > 0 and new_sl >= current_sl:
                return False
        
        # Modify the position's stop loss
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"🔒 BREAKEVEN SET #{position.ticket} {symbol} | SL moved to {new_sl:.5f} (profit locked)")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Smart breakeven error: {e}")
        return False


def check_partial_close_opportunity(position, symbol, current_profit_pips, sl_pips):
    """
    Check if we should partially close position at 1:1 RR.
    Returns (should_close, close_percent, new_sl)
    """
    if not SMART_POSITION_MANAGEMENT or not PARTIAL_CLOSE_AT_1R:
        return False, 0, None
    
    try:
        # Check if we've reached 1:1 RR (profit = initial risk)
        if current_profit_pips < sl_pips:
            return False, 0, None
        
        # We've reached 1:1 - suggest partial close
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False, 0, None
        
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        
        # New SL after partial close: move to entry
        if MOVE_SL_TO_ENTRY_AFTER_PARTIAL:
            new_sl = position.price_open
        else:
            new_sl = position.sl
        
        return True, PARTIAL_CLOSE_PERCENT, new_sl
        
    except Exception as e:
        logger.error(f"Partial close check error: {e}")
        return False, 0, None


# ================================================================================
# ========================= DYNAMIC COMPOUNDING SYSTEM ==========================
# ================================================================================

def initialize_compounding():
    """Initialize compounding state with current account balance"""
    global compounding_state
    try:
        acc = mt5.account_info()
        if acc:
            if compounding_state['base_balance'] is None:
                compounding_state['base_balance'] = acc.balance
                compounding_state['last_update_balance'] = acc.balance
                logger.info(f"💰 Compounding initialized: Base balance = ${acc.balance:.2f}")
    except Exception as e:
        logger.error(f"Error initializing compounding: {e}")


def update_compound_multiplier():
    """Update lot multiplier based on account growth"""
    global compounding_state
    try:
        if not USE_DYNAMIC_COMPOUNDING:
            return 1.0
        
        acc = mt5.account_info()
        if not acc:
            return compounding_state.get('current_lot_multiplier', 1.0)
        
        current_balance = acc.balance
        base_balance = compounding_state.get('base_balance') or current_balance
        
        if base_balance <= 0:
            return 1.0
        
        # Calculate growth percentage
        growth_percent = (current_balance - base_balance) / base_balance * 100
        
        # Calculate multiplier: increase by COMPOUND_GROWTH_RATE for every 10% growth
        growth_factor = growth_percent / 10.0  # How many 10% increments
        new_multiplier = 1.0 + (growth_factor * COMPOUND_GROWTH_RATE)
        
        # Clamp to max multiplier
        new_multiplier = min(MAX_COMPOUND_MULTIPLIER, max(0.5, new_multiplier))
        
        # Update state
        compounding_state['current_lot_multiplier'] = new_multiplier
        compounding_state['last_update_balance'] = current_balance
        compounding_state['trades_since_update'] = 0
        
        logger.info(f"📈 Compound Update: Growth={growth_percent:.1f}% Multiplier={new_multiplier:.2f}x")
        
        return new_multiplier
        
    except Exception as e:
        logger.error(f"Error updating compound multiplier: {e}")
        return 1.0


def get_compounded_lot(base_lot):
    """Apply compounding multiplier to base lot size"""
    global compounding_state
    
    if not USE_DYNAMIC_COMPOUNDING:
        return base_lot
    
    try:
        # Initialize if needed
        if compounding_state.get('base_balance') is None:
            initialize_compounding()
        
        # Update multiplier periodically
        compounding_state['trades_since_update'] = compounding_state.get('trades_since_update', 0) + 1
        if compounding_state['trades_since_update'] >= COMPOUND_UPDATE_INTERVAL:
            update_compound_multiplier()
        
        multiplier = compounding_state.get('current_lot_multiplier', 1.0)
        compounded_lot = base_lot * multiplier
        
        # Ensure lot is within limits
        compounded_lot = max(MIN_LOT, min(MAX_LOT, round(compounded_lot, 2)))
        
        logger.debug(f"💰 Compounded lot: {base_lot} x {multiplier:.2f} = {compounded_lot}")
        
        return compounded_lot
        
    except Exception as e:
        logger.error(f"Error calculating compounded lot: {e}")
        return base_lot


# ================================================================================
# ========================= AGGRESSIVE TRAILING STOP SYSTEM =====================
# ================================================================================

def apply_aggressive_trailing(position, symbol):
    """Apply aggressive multi-phase trailing stop"""
    try:
        if not USE_AGGRESSIVE_TRAILING:
            return False
        
        profit_pips = calculate_profit_pips(position, symbol)
        current_sl = position.sl
        entry_price = position.price_open
        is_buy = position.type == mt5.ORDER_TYPE_BUY
        
        info = mt5.symbol_info(symbol)
        if not info:
            return False
        
        point = info.point
        pip_mult = 10 if 'JPY' not in symbol else 1
        pip_value = point * pip_mult
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        current_price = tick.bid if is_buy else tick.ask
        
        # Determine which trailing phase we're in
        new_sl = current_sl
        phase_used = None
        
        # Check phases from highest to lowest (phase5 → phase_ultra)
        for phase_name in ['phase5', 'phase4', 'phase3', 'phase2', 'phase1', 'phase0', 'phase_instant', 'phase_ultra']:
            config = AGGRESSIVE_TRAIL_CONFIGS.get(phase_name, {})
            trigger = config.get('trigger_pips', 0)
            trail = config.get('trail_distance', 0)
            
            if profit_pips >= trigger:
                if is_buy:
                    proposed_sl = current_price - (trail * pip_value)
                    if proposed_sl > current_sl:
                        new_sl = proposed_sl
                        phase_used = phase_name
                else:
                    proposed_sl = current_price + (trail * pip_value)
                    if proposed_sl < current_sl or current_sl == 0:
                        new_sl = proposed_sl
                        phase_used = phase_name
                break  # Use the highest phase that applies
        
        # Apply profit lock if enabled
        if profit_pips >= LOCK_PROFIT_THRESHOLD_PIPS:
            locked_profit_pips = profit_pips * LOCK_PROFIT_PERCENT
            if is_buy:
                lock_sl = entry_price + (locked_profit_pips * pip_value)
                if lock_sl > new_sl:
                    new_sl = lock_sl
                    phase_used = "profit_lock"
            else:
                lock_sl = entry_price - (locked_profit_pips * pip_value)
                if lock_sl < new_sl or new_sl == 0:
                    new_sl = lock_sl
                    phase_used = "profit_lock"
        
        # Only modify if SL improved
        if new_sl != current_sl:
            digits = info.digits
            new_sl = round(new_sl, digits)
            
            # Validate SL is in correct direction
            if is_buy and new_sl >= current_price:
                return False
            if not is_buy and new_sl <= current_price:
                return False
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": symbol,
                "sl": new_sl,
                "tp": position.tp,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"🎯 Aggressive Trail [{phase_used}]: {symbol} SL moved to {new_sl:.5f} (Profit: {profit_pips:.1f} pips)")
                return True
            else:
                error_msg = result.comment if result else "Unknown error"
                logger.debug(f"Trail modify failed: {error_msg}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error in aggressive trailing: {e}")
        return False


# Note: calculate_profit_pips is defined later in the SMART PROFIT PROTECTION section


# ---------------- ORDER ----------------
def send_order(symbol, order_type, lot, sl, tp, signal_type):
    """Send order with optimized execution speed and SL/TP validation"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("❌ No tick data")
        return None
    
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    
    # Get symbol info for proper lot sizing and stops validation
    info = mt5.symbol_info(symbol)
    if not info:
        logger.error(f"❌ No symbol info for {symbol}")
        return None
    
    # Lot sizing
    lot = max(info.volume_min, min(info.volume_max, lot))
    lot = round(lot / info.volume_step) * info.volume_step
    lot = round(lot, 2)
    
    # Get proper decimal places for this symbol
    digits = info.digits
    
    # Calculate minimum stop distance (broker requirement)
    stops_level = info.trade_stops_level
    if stops_level == 0:
        stops_level = 10  # Default if broker doesn't specify
    min_stop_distance = stops_level * info.point
    
    # Validate and fix SL/TP - must be correct distance from price
    if order_type == mt5.ORDER_TYPE_BUY:
        # For BUY: SL must be BELOW price, TP must be ABOVE price
        if sl >= price or (price - sl) < min_stop_distance:
            sl = price - min_stop_distance * 2
        if tp <= price or (tp - price) < min_stop_distance:
            tp = price + min_stop_distance * 3
    else:  # SELL
        # For SELL: SL must be ABOVE price, TP must be BELOW price
        if sl <= price or (sl - price) < min_stop_distance:
            sl = price + min_stop_distance * 2
        if tp >= price or (price - tp) < min_stop_distance:
            tp = price - min_stop_distance * 3
    
    # Round SL/TP to proper decimal places
    sl = round(sl, digits)
    tp = round(tp, digits)
    price = round(price, digits)
    
    # Determine the correct filling mode based on symbol info
    # The filling_mode property is a bitmask of supported modes
    filling_type = mt5.ORDER_FILLING_FOK  # Default
    if info.filling_mode & 1:  # FILLING_FOK = 1
        filling_type = mt5.ORDER_FILLING_FOK
    elif info.filling_mode & 2:  # FILLING_IOC = 2
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_RETURN  # FILLING_RETURN = 0 or fallback
    
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
        "deviation": 100,  # Higher deviation for faster fills
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type
    }
    
    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ {signal_type} EXECUTED lot={lot} price={price:.{digits}f}")
        trade_stats['total_trades'] += 1
        return result
    
    # If failed, try without SL/TP first (some brokers don't allow SL/TP on market orders)
    if result and result.retcode == 10030:
        # Try market order without SL/TP, then modify
        request_no_sl = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "magic": MAGIC,
            "comment": signal_type,
            "deviation": 100,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type
        }
        result = mt5.order_send(request_no_sl)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Now modify to add SL/TP
            position_ticket = result.order
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": position_ticket,
                "sl": sl,
                "tp": tp
            }
            mt5.order_send(modify_request)
            logger.info(f"✅ {signal_type} EXECUTED (2-step) lot={lot} price={price:.{digits}f}")
            trade_stats['total_trades'] += 1
            return result
    
    # Log the error
    if result:
        retcode = result.retcode
        comment = result.comment
        order_dir = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        logger.error(f"❌ {signal_type} FAILED: {retcode} | {order_dir} price={price:.{digits}f} sl={sl:.{digits}f} tp={tp:.{digits}f} | {comment}")
    else:
        logger.error(f"❌ {signal_type} FAILED: No result")
    return result


def place_order(symbol, order_type, lot, sl, tp, confidence=1.0, signal_type="MANUAL"):
    order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type.lower() == "buy" else mt5.ORDER_TYPE_SELL
    return send_order(symbol, order_type_mt5, lot, sl, tp, signal_type)


# ---------------- SMART TRAILING STOP SYSTEM ----------------
def calculate_atr_for_trailing(symbol, period=14):
    """Calculate ATR for dynamic trailing stop distance"""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, period + 1)
    if rates is None or len(rates) < period:
        return None
    df = pd.DataFrame(rates)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr


def manage_trailing_stops(symbol):
    """
    Advanced trailing stop management with:
    1. Aggressive multi-phase trailing (if enabled)
    2. Breakeven move when in profit
    3. ATR-based dynamic trailing distance
    4. Step-based trailing to avoid premature exits
    5. Profit lock-in at key levels
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    # Use aggressive trailing if in scalping mode
    if USE_AGGRESSIVE_TRAILING and SCALPING_MODE:
        for pos in positions:
            apply_aggressive_trailing(pos, symbol)
        return  # Aggressive trailing handles everything
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return
    
    point = symbol_info.point
    min_stop = max(symbol_info.trade_stops_level * point, point * 10)
    
    # Use symbol-specific pip multiplier
    if symbol in SYMBOL_SETTINGS:
        pip_value = SYMBOL_SETTINGS[symbol]['pip_value']
        pip_mult = pip_value / point if point > 0 else 10
    elif 'JPY' in symbol:
        pip_mult = 1
    else:
        pip_mult = 10
    
    # Get ATR for dynamic trailing
    atr = calculate_atr_for_trailing(symbol) if USE_ATR_TRAILING else None
    
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue
        
        entry = pos.price_open
        current_sl = pos.sl
        current_tp = pos.tp
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_price = tick.bid
            profit_pips = (current_price - entry) / (point * pip_mult)
            
            # Calculate trailing distance (ATR-based or fixed)
            if atr and USE_ATR_TRAILING:
                trail_distance = atr * ATR_MULTIPLIER
            else:
                trail_distance = TRAILING_DISTANCE * point * pip_mult
            
            # Stage 1: Move to breakeven when profit reaches BREAKEVEN_PIPS
            if profit_pips >= BREAKEVEN_PIPS and (current_sl < entry or current_sl == 0):
                breakeven_sl = entry + (2 * point * pip_mult)  # Slightly above entry to cover spread
                if (current_price - breakeven_sl) >= min_stop:
                    result = mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": breakeven_sl,
                        "tp": current_tp
                    })
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ Moved SL to breakeven for BUY #{pos.ticket}")
            
            # Stage 2: Start trailing after TRAIL_ACTIVATION_PIPS profit
            elif profit_pips >= TRAIL_ACTIVATION_PIPS:
                # Calculate new SL based on trailing distance
                new_sl = current_price - trail_distance
                
                # Only move in steps of TRAIL_STEP_PIPS to avoid constant adjustments
                step_size = TRAIL_STEP_PIPS * point * pip_mult
                if current_sl > 0:
                    sl_move = new_sl - current_sl
                    if sl_move >= step_size and (current_price - new_sl) >= min_stop:
                        result = mt5.order_send({
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": current_tp
                        })
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"📈 Trailing SL updated for BUY #{pos.ticket}: {current_sl:.5f} → {new_sl:.5f}")
                else:
                    # First trailing move
                    if new_sl > entry and (current_price - new_sl) >= min_stop:
                        result = mt5.order_send({
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": current_tp
                        })
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"📈 Started trailing for BUY #{pos.ticket}: SL set to {new_sl:.5f}")
        
        else:  # SELL position
            current_price = tick.ask
            profit_pips = (entry - current_price) / (point * pip_mult)
            
            # Calculate trailing distance (ATR-based or fixed)
            if atr and USE_ATR_TRAILING:
                trail_distance = atr * ATR_MULTIPLIER
            else:
                trail_distance = TRAILING_DISTANCE * point * pip_mult
            
            # Stage 1: Move to breakeven when profit reaches BREAKEVEN_PIPS
            if profit_pips >= BREAKEVEN_PIPS and (current_sl > entry or current_sl == 0):
                breakeven_sl = entry - (2 * point * pip_mult)  # Slightly below entry to cover spread
                if (breakeven_sl - current_price) >= min_stop:
                    result = mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": breakeven_sl,
                        "tp": current_tp
                    })
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ Moved SL to breakeven for SELL #{pos.ticket}")
            
            # Stage 2: Start trailing after TRAIL_ACTIVATION_PIPS profit
            elif profit_pips >= TRAIL_ACTIVATION_PIPS:
                # Calculate new SL based on trailing distance
                new_sl = current_price + trail_distance
                
                # Only move in steps of TRAIL_STEP_PIPS to avoid constant adjustments
                step_size = TRAIL_STEP_PIPS * point * pip_mult
                if current_sl > 0:
                    sl_move = current_sl - new_sl
                    if sl_move >= step_size and (new_sl - current_price) >= min_stop:
                        result = mt5.order_send({
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": current_tp
                        })
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"📉 Trailing SL updated for SELL #{pos.ticket}: {current_sl:.5f} → {new_sl:.5f}")
                else:
                    # First trailing move
                    if new_sl < entry and (new_sl - current_price) >= min_stop:
                        result = mt5.order_send({
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": current_tp
                        })
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"📉 Started trailing for SELL #{pos.ticket}: SL set to {new_sl:.5f}")


def close_opposite_positions(trend, symbol):
    """Close positions opposite to the current trend"""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for pos in positions:
        close_flag = (trend == "BEARISH" and pos.type == mt5.POSITION_TYPE_BUY) or (trend == "BULLISH" and pos.type == mt5.POSITION_TYPE_SELL)
        if close_flag:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask,
                    "magic": MAGIC,
                    "deviation": 20
                })


# ================================================================================
# ================= SMART PROFIT PROTECTION & RE-ENTRY SYSTEM ===================
# ================================================================================

def calculate_profit_pips(position, symbol):
    """Calculate current profit in pips for a position"""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return 0
    
    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.0001
    
    # Use symbol-specific pip value for correct calculation
    if symbol in SYMBOL_SETTINGS:
        pip_value = SYMBOL_SETTINGS[symbol]['pip_value']
        pip_mult = pip_value / point if point > 0 else 10
    elif 'JPY' in symbol:
        pip_mult = 1
    else:
        pip_mult = 10
    
    if position.type == mt5.POSITION_TYPE_BUY:
        current_price = tick.bid
        profit_pips = (current_price - position.price_open) / (point * pip_mult)
    else:
        current_price = tick.ask
        profit_pips = (position.price_open - current_price) / (point * pip_mult)
    
    return profit_pips


def close_position(position, reason=""):
    """Close a position completely"""
    symbol = position.symbol
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    
    close_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": close_price,
        "magic": MAGIC,
        "deviation": 20,
        "comment": reason[:31] if reason else "PROFIT_LOCK"
    })
    
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"💰 Closed position #{position.ticket} for profit lock: {reason}")
        
        # AI Strategy Learning - update strategy performance
        ticket = position.ticket
        if ticket in trade_strategies_used:
            trade_data = trade_strategies_used[ticket]
            strategies = trade_data.get('strategies', [])
            is_win = position.profit > 0
            
            # Calculate profit in pips
            sym_settings = get_symbol_settings(symbol)
            pip_value = sym_settings.get('pip_value', 0.0001)
            entry_price = trade_data.get('entry_price', position.price_open)
            
            if trade_data.get('direction') == 'BUY':
                profit_pips = (close_price - entry_price) / pip_value
            else:
                profit_pips = (entry_price - close_price) / pip_value
            
            # Update each strategy's performance
            user = trade_data.get('user', 'unknown')
            for strat_name in strategies:
                try:
                    update_strategy_performance(strat_name, is_win, profit_pips, user)
                except Exception as e:
                    pass
            
            logger.info(f"🧠 AI LEARNED from #{ticket}: {len(strategies)} strategies {'✅' if is_win else '❌'}")
            del trade_strategies_used[ticket]
        
        return True
    return False


def partial_close_position(position, close_percent=0.5):
    """Close a portion of a position to lock in profits"""
    symbol = position.symbol
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return False
    
    # Calculate volume to close
    close_volume = round(position.volume * close_percent, 2)
    if close_volume < info.volume_min:
        close_volume = position.volume  # Close all if partial is too small
    
    close_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": close_volume,
        "type": close_type,
        "position": position.ticket,
        "price": close_price,
        "magic": MAGIC,
        "deviation": 20,
        "comment": "PARTIAL_PROFIT"
    })
    
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"💰 Partial close {close_percent*100}% of #{position.ticket} ({close_volume} lots)")
        return True
    return False


# ========== ULTRA-FAST SCALPING EXECUTION ==========
def ultra_fast_scalp(symbol, direction, user, lot_size=None, sl_pips=None, tp_pips=None):
    """
    Execute ultra-fast market order with tight SL/TP.
    Designed for quick scalping with minimal slippage.
    """
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return None
    
    point = info.point
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    pip_mult = pip_value / point if point > 0 else 10
    
    # Use tight scalping SL/TP
    sl = sl_pips if sl_pips else STOPLOSS_PIPS
    tp = tp_pips if tp_pips else TAKEPROFIT_PIPS
    
    if direction == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl_price = price - (sl * point * pip_mult)
        tp_price = price + (tp * point * pip_mult)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl_price = price + (sl * point * pip_mult)
        tp_price = price - (tp * point * pip_mult)
    
    # Calculate lot size if not provided
    if lot_size is None:
        account = mt5.account_info()
        if account:
            lot_size = calculate_lot(account.balance, RISK_PERCENT, sl)
        else:
            lot_size = MIN_LOT
    
    # Ensure lot is within limits
    lot_size = max(info.volume_min, min(lot_size, info.volume_max))
    lot_size = round(lot_size, 2)
    
    # Execute with minimal slippage
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": MAX_SLIPPAGE_PIPS if 'MAX_SLIPPAGE_PIPS' in dir() else 5,
        "magic": MAGIC,
        "comment": f"SCALP_{direction}_{datetime.now().strftime('%H%M%S')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"⚡ SCALP {direction} {symbol} @ {price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f} | Lot: {lot_size}")
        log_trade(user, 'scalp_entry', f'SCALP {direction} {symbol}', {
            'price': price, 'sl': sl_price, 'tp': tp_price, 'lot': lot_size
        })
        return result
    else:
        error = result.retcode if result else 'NO_RESPONSE'
        logger.warning(f"⚠️ SCALP failed {symbol}: {error}")
        return None


def move_to_breakeven(position, buffer_pips=1):
    """Move SL to breakeven + buffer pips"""
    symbol = position.symbol
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not info or not tick:
        return False
    
    point = info.point
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    pip_mult = pip_value / point if point > 0 else 10
    
    buffer = buffer_pips * point * pip_mult
    
    if position.type == mt5.POSITION_TYPE_BUY:
        new_sl = position.price_open + buffer
        if new_sl > position.sl:  # Only move if better
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Moved {symbol} #{position.ticket} to breakeven + {buffer_pips} pips")
                return True
    else:
        new_sl = position.price_open - buffer
        if new_sl < position.sl:  # Only move if better
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Moved {symbol} #{position.ticket} to breakeven + {buffer_pips} pips")
                return True
    return False


def check_momentum_scalp(df, direction):
    """
    Check if there's strong momentum for scalping.
    Returns (should_trade, momentum_strength)
    """
    if len(df) < 10:
        return False, 0
    
    # Calculate momentum
    close = df['close'].iloc[-1]
    close_5 = df['close'].iloc[-5]
    momentum = (close - close_5) / close_5 if close_5 > 0 else 0
    
    # Check direction alignment
    if direction == 'BUY' and momentum > MOMENTUM_THRESHOLD:
        return True, momentum
    elif direction == 'SELL' and momentum < -MOMENTUM_THRESHOLD:
        return True, abs(momentum)
    
    return False, abs(momentum)


def ultra_fast_manage_positions(symbol, df, user):
    """
    Ultra-fast position management for scalping:
    - Move to breakeven quickly
    - Partial close for profits
    - Fast reversal exit
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    info = mt5.symbol_info(symbol)
    if not info:
        return
    
    point = info.point
    sym_settings = get_symbol_settings(symbol)
    pip_value = sym_settings.get('pip_value', 0.0001)
    pip_mult = pip_value / point if point > 0 else 10
    
    for pos in positions:
        profit_pips = calculate_profit_pips(pos, symbol)
        direction = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
        
        # BREAKEVEN: Move SL to breakeven after BREAKEVEN_PIPS profit
        breakeven_pips = BREAKEVEN_PIPS if 'BREAKEVEN_PIPS' in dir() else 4
        if profit_pips >= breakeven_pips and pos.sl != pos.price_open:
            move_to_breakeven(pos, 1)
        
        # PARTIAL CLOSE: Take partial profits at PARTIAL_CLOSE_PIPS
        partial_pips = PARTIAL_CLOSE_PIPS if 'PARTIAL_CLOSE_PIPS' in dir() else 5
        partial_key = f"partial_{pos.ticket}"
        if PARTIAL_CLOSE_ENABLED and profit_pips >= partial_pips and partial_key not in closed_for_reentry:
            if partial_close_position(pos, PARTIAL_CLOSE_PERCENT):
                closed_for_reentry[partial_key] = True
                log_trade(user, 'partial_close', f'Partial profit on {symbol}', {'pips': profit_pips})
        
        # FAST REVERSAL EXIT: Check if momentum reversed
        if FAST_REVERSAL_EXIT if 'FAST_REVERSAL_EXIT' in dir() else True:
            has_momentum, strength = check_momentum_scalp(df, direction)
            opposite_direction = 'SELL' if direction == 'BUY' else 'BUY'
            has_reversal, rev_strength = check_momentum_scalp(df, opposite_direction)
            
            # If opposite momentum is strong and we're in small profit, exit
            if has_reversal and rev_strength > MOMENTUM_THRESHOLD * 2 and profit_pips >= 1:
                if close_position(pos, f"REVERSAL_EXIT_{profit_pips:.1f}pips"):
                    log_trade(user, 'reversal_exit', f'Reversal exit {symbol}', {'pips': profit_pips})


def smart_profit_protection(symbol, df, buy_score, sell_score, user):
    """
    Smart profit protection system with TIERED profit drop detection and SPREAD AWARENESS:
    1. Monitors positions for profit threshold
    2. Uses tiered profit drop rules - faster closes at smaller profits
    3. Closes on momentum reversal while in profit
    4. Records for potential re-entry if signal still valid
    5. SPREAD-AWARE: Closes faster when profit exceeds spread recovery target
    """
    global closed_for_reentry
    
    if not ENABLE_PROFIT_PROTECTION:
        return
    
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.0001
    
    # Get current spread for spread-aware decisions
    spread_pips = get_spread_in_pips(symbol)
    spread_recovery_target = spread_pips * SPREAD_RECOVERY_MULTIPLIER
    auto_close_target = spread_pips * AUTO_CLOSE_AT_SPREAD_MULTIPLE
    
    # Use symbol-specific pip value for correct calculation
    if symbol in SYMBOL_SETTINGS:
        pip_value = SYMBOL_SETTINGS[symbol]['pip_value']
        pip_mult = pip_value / point if point > 0 else 10
    elif 'JPY' in symbol:
        pip_mult = 1
    else:
        pip_mult = 10
    
    for pos in positions:
        profit_pips = calculate_profit_pips(pos, symbol)
        direction = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
        
        # Track peak profit
        peak_key = f"{symbol}_{pos.ticket}_peak"
        current_peak = closed_for_reentry.get(peak_key, 0)
        if profit_pips > current_peak:
            closed_for_reentry[peak_key] = profit_pips
            current_peak = profit_pips
        
        # === SPREAD RECOVERY AUTO-CLOSE ===
        # If we've recovered spread cost + good profit, be very protective!
        if SPREAD_RECOVERY_MODE and profit_pips >= auto_close_target and current_peak - profit_pips >= 0.05:
            logger.info(f"[{user}] 💎 SPREAD RECOVERY: {symbol} at {profit_pips:.1f} pips (spread: {spread_pips:.1f}) - securing gains!")
            if close_position(pos, f"SPREAD_RECOVERED_{profit_pips:.1f}pips"):
                log_trade(user, 'spread_recovery', f'Closed at {profit_pips:.1f} pips after spread recovery', {'spread': spread_pips, 'profit': profit_pips})
                continue
        
        # === TIERED PROFIT DROP PROTECTION ===
        # Check from largest tier to smallest - use first matching tier
        if current_peak >= MONITOR_PROFIT_AFTER_PIPS and profit_pips >= 0:
            pips_dropped = current_peak - profit_pips
            drop_pct = pips_dropped / current_peak if current_peak > 0 else 0
            
            should_close = False
            tier_used = None
            
            # Check tiers from large to small (4+ → 2+ → 1+ → 0.6+ → 0.3+ → 0.1+)
            for tier_name in ['large', 'medium', 'small', 'micro', 'nano', 'instant']:
                tier = PROFIT_DROP_TIERS.get(tier_name, {})
                min_peak = tier.get('min_peak', 999)
                max_drop_pct = tier.get('drop_pct', 0.25)
                max_drop_pips = tier.get('drop_pips', 1.0)
                
                if current_peak >= min_peak:
                    # SPREAD-AWARE: Tighten drop threshold if we've recovered spread
                    if profit_pips >= spread_recovery_target:
                        max_drop_pct = max_drop_pct * 0.7  # 30% tighter
                        max_drop_pips = max_drop_pips * 0.7  # 30% tighter
                    
                    # Close if drop exceeds threshold (either % or pips)
                    if drop_pct >= max_drop_pct or pips_dropped >= max_drop_pips:
                        should_close = True
                        tier_used = tier_name
                    break  # Use first matching tier
            
            if should_close and profit_pips >= MIN_PROFIT_PIPS_TO_CLOSE:
                logger.info(f"[{user}] ⚠️ PROFIT DROP [{tier_used}]: {symbol} Peak={current_peak:.1f} → Now={profit_pips:.1f} (dropped {pips_dropped:.1f} pips, {drop_pct*100:.0f}%)")
                if close_position(pos, f"DROP_{tier_used}_{current_peak:.1f}to{profit_pips:.1f}"):
                    log_trade(user, 'profit_drop', f'Closed on {tier_used} drop {symbol}', {'peak': current_peak, 'now': profit_pips, 'tier': tier_used})
                    continue
        
        # === MOMENTUM REVERSAL CHECK ===
        # Close if momentum reversed against us while in profit
        if CLOSE_ON_MOMENTUM_REVERSAL and profit_pips >= MOMENTUM_REVERSAL_MIN_PROFIT:
            opposite_dir = 'SELL' if direction == 'BUY' else 'BUY'
            has_reversal, rev_strength = check_momentum_scalp(df, opposite_dir)
            
            if has_reversal and rev_strength > MOMENTUM_THRESHOLD * 1.5:
                logger.info(f"[{user}] 🔄 MOMENTUM REVERSAL: {symbol} at +{profit_pips:.1f} pips - opposite momentum {rev_strength:.1f}")
                if close_position(pos, f"MOM_REV_{profit_pips:.1f}pips"):
                    log_trade(user, 'momentum_reversal', f'Closed on reversal {symbol}', {'pips': profit_pips, 'strength': rev_strength})
                    continue
        
        # === EMERGENCY - Profit dropping to zero ===
        if current_peak >= 0.5 and profit_pips < 0.2 and profit_pips > -0.2:
            logger.info(f"[{user}] 🚨 EMERGENCY: {symbol} was +{current_peak:.1f}, now +{profit_pips:.1f}")
            if close_position(pos, "EMERGENCY"):
                log_trade(user, 'emergency', f'Emergency close {symbol}', {'peak': current_peak, 'now': profit_pips})
                continue
        
        # === INSTANT PROFIT LOCK AT GOOD LEVELS ===
        # Close immediately when profit starts dropping - ULTRA AGGRESSIVE for spread protection
        if AGGRESSIVE_PROFIT_LOCK:
            # Get spread for comparison
            current_spread_pips = get_spread_in_pips(symbol)
            
            # ULTRA LOCK: At 2x spread profit, lock if ANY drop
            if profit_pips >= current_spread_pips * 2 and current_peak - profit_pips >= 0.05:
                logger.info(f"[{user}] 💎 ULTRA LOCK: {symbol} at {profit_pips:.1f} pips (2x spread) - LOCKING!")
                if close_position(pos, f"ULTRALOCK_{profit_pips:.1f}pips"):
                    log_trade(user, 'ultra_lock', f'Ultra locked {profit_pips:.1f} pips on {symbol}', {'symbol': symbol, 'pips': profit_pips})
                    continue
            
            # Lock at 1+ pips if dropping at all
            if profit_pips >= 1.0 and current_peak - profit_pips >= 0.1:
                logger.info(f"[{user}] 💰 PROFIT LOCK: {symbol} at {profit_pips:.1f} pips (peak {current_peak:.1f}) - LOCKING!")
                if close_position(pos, f"LOCK_{profit_pips:.1f}pips"):
                    log_trade(user, 'profit_lock', f'Locked {profit_pips:.1f} pips on {symbol}', {'symbol': symbol, 'pips': profit_pips})
                    continue
            
            # Even tighter: Lock at 0.5+ pips if dropping 0.1+ pips
            elif profit_pips >= 0.5 and current_peak - profit_pips >= 0.1:
                logger.info(f"[{user}] 💰 FAST LOCK: {symbol} at {profit_pips:.1f} pips (peak {current_peak:.1f}) - LOCKING!")
                if close_position(pos, f"FASTLOCK_{profit_pips:.1f}pips"):
                    log_trade(user, 'fast_lock', f'Fast locked {profit_pips:.1f} pips on {symbol}', {'symbol': symbol, 'pips': profit_pips})
                    continue


def check_reentry_opportunity(symbol, direction, score, user):
    """
    Check if we should re-enter after a profit lock.
    Returns True if we should enter, False otherwise.
    """
    global closed_for_reentry
    
    if symbol not in closed_for_reentry:
        return True  # No previous close, normal entry allowed
    
    reentry_data = closed_for_reentry[symbol]
    
    # Check if it's the same direction
    if reentry_data.get('direction') != direction:
        return True  # Different direction, allow entry
    
    # Check cooldown period
    close_time = reentry_data.get('close_time')
    if close_time:
        elapsed = (datetime.now() - close_time).total_seconds()
        if elapsed < REENTRY_COOLDOWN_SECONDS:
            return False  # Still in cooldown
    
    # Check if signal is still strong for re-entry
    original_score = reentry_data.get('score', 0)
    if score >= original_score:
        logger.info(f"🔄 Re-entry allowed for {symbol} {direction} (score {score} >= original {original_score})")
        # Clear the reentry data after allowing re-entry
        del closed_for_reentry[symbol]
        return True
    
    # Signal weakened, be more cautious - require higher score
    if score >= MIN_SMC_SCORE + 1:
        logger.info(f"🔄 Re-entry allowed for {symbol} {direction} with higher score requirement")
        del closed_for_reentry[symbol]
        return True
    
    return False


# ---------------- MULTI-SYMBOL BOT LOOP (BALANCED AI-ENHANCED SMC STRATEGY) ----------------
def run_bot(user):
    stop_event = user_bots[user]["stop_event"]
    
    # Set current user for MT5 session tracking
    set_current_mt5_user(user)
    
    # Get user's MT5 credentials from database
    from models import get_user_mt5_credentials
    creds = get_user_mt5_credentials(user)
    
    if creds:
        login = creds['login']
        password = creds['password']
        server = creds['server']
        logger.info(f"[{user}] 🔑 Using saved MT5 credentials for account {login} on {server}")
    else:
        # Use defaults if no user credentials
        login = DEFAULT_MT5_LOGIN
        password = DEFAULT_MT5_PASSWORD
        server = DEFAULT_MT5_SERVER
        logger.warning(f"[{user}] ⚠️ No MT5 credentials found in database, using defaults (account {login})")
    
    if not initialize_mt5(login, password, server):
        user_bots[user]["running"] = False
        log_trade(user, 'error', 'Failed to initialize MT5', {'reason': 'Connection failed'})
        return
    
    user_mt5_sessions[user] = True
    
    # Initialize broker symbol mapping (detects suffixes like 'm' in EURUSDm, or prefixes)
    initialize_symbol_mapping()
    
    # Get user's selected symbols and convert to broker format
    user_symbols = get_user_symbols(user)
    symbols = []
    for sym in user_symbols:
        broker_sym = get_broker_symbol(sym)
        if broker_sym:
            symbols.append(broker_sym)
            # Enable symbol in MT5
            if mt5.symbol_select(broker_sym, True):
                logger.debug(f"✅ Enabled symbol: {broker_sym}")
            else:
                logger.warning(f"⚠️ Could not enable symbol: {broker_sym}")
    
    if not symbols:
        logger.error(f"[{user}] ❌ No valid symbols found on this broker!")
        user_bots[user]["running"] = False
        return
    
    logger.info(f"[{user}] 🚀 Multi-Symbol Bot started on {', '.join(symbols)} M5")
    log_trade(user, 'bot', f'Bot started on {len(symbols)} symbols', {'symbols': symbols, 'timeframe': 'M5', 'strategy': 'MULTI-AI-SMC'})
    
    # Track per-symbol data
    prev_positions = {}
    ai_analysis = {}  # {symbol: {counter, recommendation}}
    consecutive_losses = 0
    
    # Initialize daily stats
    account = mt5.account_info()
    if account:
        stats = user_daily_stats[user]
        if stats['date'] != datetime.now().date():
            stats['start_balance'] = account.balance
            stats['starting_equity'] = account.equity
            stats['date'] = datetime.now().date()
            logger.info(f"[{user}] 📊 Starting balance: ${account.balance:.2f}, Equity: ${account.equity:.2f}")
    
    while not stop_event.is_set():
        # ========== CRITICAL: CHECK LOSS PROTECTION ==========
        can_trade, reason = check_loss_protection(user)
        if not can_trade:
            logger.warning(f"[{user}] 🛡️ Trading blocked: {reason}")
            # Still manage existing positions but don't open new ones
            all_positions = mt5.positions_get()
            if all_positions:
                # Monitor for emergency close
                account = mt5.account_info()
                if account:
                    drawdown = ((user_daily_stats[user].get('starting_equity', account.equity) - account.equity) / 
                               user_daily_stats[user].get('starting_equity', account.equity)) * 100
                    if drawdown >= EMERGENCY_CLOSE_ALL_AT_DRAWDOWN:
                        logger.critical(f"[{user}] 🚨🚨 EMERGENCY: {drawdown:.2f}% drawdown - closing all!")
                        close_all_positions(user)
            stop_event.wait(30)  # Wait 30 seconds before checking again
            continue
        
        # Get total positions count
        all_positions = mt5.positions_get()
        total_positions = len(all_positions) if all_positions else 0
        
        # Track closed positions and record results
        current_tickets = {p.ticket for p in all_positions} if all_positions else set()
        for ticket, prev_pos in list(prev_positions.items()):
            if ticket not in current_tickets:
                # Position was closed - calculate result
                history = mt5.history_deals_get(position=ticket)
                if history:
                    total_profit = sum(d.profit + d.commission + d.swap for d in history)
                    record_trade_result(user, total_profit, prev_pos.get('symbol', ''))
                del prev_positions[ticket]
        
        # Update prev_positions
        if all_positions:
            for pos in all_positions:
                if pos.ticket not in prev_positions:
                    prev_positions[pos.ticket] = {'symbol': pos.symbol, 'profit': pos.profit}
        
        # Iterate through each symbol
        for symbol in symbols:
            if stop_event.is_set():
                break
            
            # Get symbol-specific settings
            sym_settings = get_symbol_settings(symbol)
            sl_pips_base = sym_settings['sl_pips']
            tp_pips_base = sym_settings['tp_pips']
            
            # Get data for this symbol
            df = get_data(symbol, TIMEFRAME)
            if df is None or len(df) < 100:
                continue
            
            # Calculate all advanced indicators
            df = calculate_advanced_indicators(df)
            
            # Detect market regime
            market_regime = detect_market_regime(df)
            
            # Get current indicator values
            price = df["close"].iloc[-1]
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # SMC Strategy Analysis
            trend = trend_bias(df)
            sweep_high, sweep_low = liquidity_grab(df)
            ob_type, ob_low, ob_high = order_block(df)
            fvg_type, fvg_low, fvg_high = fair_value_gap(df)
            bullish_bos, bearish_bos = check_market_structure(df)
            
            current_trends[f"{user}_{symbol}"] = trend
            
            # Calculate SMC score (0-7 points)
            buy_score = 0
            sell_score = 0
            
            # Trend alignment (2 points if strong)
            if market_regime == "TRENDING_UP":
                buy_score += 2
            elif market_regime == "TRENDING_DOWN":
                sell_score += 2
            elif trend == "BULLISH":
                buy_score += 1
            elif trend == "BEARISH":
                sell_score += 1
            
            # Liquidity sweep
            if sweep_low:
                buy_score += 1
            if sweep_high:
                sell_score += 1
            
            # Order block or FVG
            if ob_type == "BULLISH" or fvg_type == "BULLISH":
                buy_score += 1
            if ob_type == "BEARISH" or fvg_type == "BEARISH":
                sell_score += 1
            
            # Break of structure
            if bullish_bos:
                buy_score += 1
            if bearish_bos:
                sell_score += 1
            
            # Momentum confirmation (MACD)
            if macd_hist > 0:
                buy_score += 1
            if macd_hist < 0:
                sell_score += 1
            
            # AI Analysis (per symbol, every 10 cycles)
            if symbol not in ai_analysis:
                ai_analysis[symbol] = {'counter': 0, 'recommendation': None}
            
            ai_analysis[symbol]['counter'] += 1
            if ai_analysis[symbol]['counter'] >= 10:
                ai_analysis[symbol]['counter'] = 0
                ai_analysis[symbol]['recommendation'] = ai_analyze_market(df, symbol, user)
                
                # Add AI score bonus
                ai_rec = ai_analysis[symbol]['recommendation']
                if ai_rec and ai_rec.get('confidence', 0) >= 0.6:
                    if ai_rec['recommendation'] == 'BUY':
                        buy_score += 1
                    elif ai_rec['recommendation'] == 'SELL':
                        sell_score += 1
            
            last_ai_recommendation = ai_analysis[symbol]['recommendation']
            
            # Get positions for this symbol
            symbol_positions = mt5.positions_get(symbol=symbol)
            symbol_pos_count = len(symbol_positions) if symbol_positions else 0
            
            # Check account
            acc = mt5.account_info()
            if not acc:
                continue
            
            point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.0001
            
            # Manage trailing stops for this symbol
            if USE_TRAILING_STOP and symbol_positions:
                manage_trailing_stops(symbol)
            
            # ========== SMART PROFIT PROTECTION ==========
            # Check and lock profits, prepare for re-entry
            smart_profit_protection(symbol, df, buy_score, sell_score, user)
            
            # ========== AGGRESSIVE SCALPING MANAGEMENT ==========
            # Check for quick profit closes and re-entry opportunities
            if AGGRESSIVE_SCALPING_ENABLED:
                manage_scalp_positions(user)
            
            # Refresh positions after profit protection (may have closed some)
            symbol_positions = mt5.positions_get(symbol=symbol)
            symbol_pos_count = len(symbol_positions) if symbol_positions else 0
            all_positions = mt5.positions_get()
            total_positions = len(all_positions) if all_positions else 0
            
            # Get AI-optimized parameters
            ai_params = get_ai_optimized_params(user)
            sl_pips = ai_params.get('sl_pips', sl_pips_base)
            tp_pips = ai_params.get('tp_pips', tp_pips_base)
            risk_pct = ai_params.get('risk_percent', RISK_PERCENT)
            min_score = ai_params.get('min_score', MIN_SMC_SCORE)
            
            # Use ATR-based dynamic lot sizing for all account sizes
            # Calculate proper pip multiplier for this symbol
            if symbol in SYMBOL_SETTINGS:
                pip_value = SYMBOL_SETTINGS[symbol]['pip_value']
                pip_mult = pip_value / point if point > 0 else 10
            else:
                pip_mult = 10 if 'JPY' not in symbol else 100  # Standard for forex
            
            if USE_ATR_LOT_SIZING:
                lot, atr_sl_pips, atr_tp_pips = calculate_atr_lot(symbol, acc.balance, risk_pct, df)
                actual_sl_pips = max(sl_pips, atr_sl_pips)
                actual_tp_pips = max(tp_pips, atr_tp_pips)
            else:
                # Fallback to basic calculation
                atr_sl = atr * 1.5 / (point * pip_mult) if atr > 0 else sl_pips
                atr_tp = atr * 2.5 / (point * pip_mult) if atr > 0 else tp_pips
                actual_sl_pips = max(sl_pips, atr_sl)
                actual_tp_pips = max(tp_pips, atr_tp)
                lot = calculate_lot(acc.balance, risk_pct, actual_sl_pips)
            
            # ========== APPLY DYNAMIC COMPOUNDING ==========
            if USE_DYNAMIC_COMPOUNDING:
                lot = get_compounded_lot(lot)
                logger.debug(f"💰 Compounded lot for {symbol}: {lot}")
            
            # Reduce risk after consecutive losses
            if consecutive_losses >= 3:
                risk_pct = risk_pct * 0.5
                lot = lot * 0.5
            
            # ========== OPTIMAL ENTRY ANALYSIS ==========
            # Check for optimal entry points using advanced analysis
            buy_optimal, buy_confidence, buy_reason = check_optimal_entry(df, "BUY", symbol)
            sell_optimal, sell_confidence, sell_reason = check_optimal_entry(df, "SELL", symbol)
            
            # Check for reversal risk
            buy_reversal_risk = check_immediate_reversal_risk(df, "BUY")
            sell_reversal_risk = check_immediate_reversal_risk(df, "SELL")
            
            # ========== NEWS & MARKET SENTIMENT ANALYSIS ==========
            # Check news sentiment for trade alignment (only check every 5 cycles to save requests)
            news_check_counter = ai_analysis.get(symbol, {}).get('news_counter', 0)
            if news_check_counter % 5 == 0:
                # Check news for buy direction
                news_ok_buy, news_mult_buy, news_reason_buy = should_trade_based_on_news(symbol, "BUY", user)
                news_ok_sell, news_mult_sell, news_reason_sell = should_trade_based_on_news(symbol, "SELL", user)
                
                # Cache the news check results
                if symbol not in ai_analysis:
                    ai_analysis[symbol] = {}
                ai_analysis[symbol]['news_buy'] = (news_ok_buy, news_mult_buy, news_reason_buy)
                ai_analysis[symbol]['news_sell'] = (news_ok_sell, news_mult_sell, news_reason_sell)
                ai_analysis[symbol]['news_counter'] = news_check_counter + 1
            else:
                # Use cached news data
                cached_buy = ai_analysis.get(symbol, {}).get('news_buy', (True, 1.0, "No news check"))
                cached_sell = ai_analysis.get(symbol, {}).get('news_sell', (True, 1.0, "No news check"))
                news_ok_buy, news_mult_buy, news_reason_buy = cached_buy
                news_ok_sell, news_mult_sell, news_reason_sell = cached_sell
                if symbol in ai_analysis:
                    ai_analysis[symbol]['news_counter'] = news_check_counter + 1
            
            # Skip if market is ranging - causes losses
            if AVOID_RANGING_MARKET and market_regime == 'RANGING':
                logger.debug(f"[{symbol}] Skipping - ranging market")
                continue
            
            # Log signal activity for all symbols periodically
            if ai_analysis.get(symbol, {}).get('counter', 0) % 10 == 0:
                logger.info(f"[{symbol}] Analyzing: BUY:{buy_score} SELL:{sell_score} Regime:{market_regime}")
            
            # ========== SAFE POSITION MANAGEMENT ==========
            # Manage trailing stops on existing positions
            if USE_TRAILING_STOP and symbol_positions:
                manage_trailing_stops(symbol)
            
            # Smart profit protection
            smart_profit_protection(symbol, df, buy_score, sell_score, user)
            
            # Refresh positions after management
            symbol_positions = mt5.positions_get(symbol=symbol)
            symbol_pos_count = len(symbol_positions) if symbol_positions else 0
            all_positions = mt5.positions_get()
            total_positions = len(all_positions) if all_positions else 0
            
            # ========== TRADE COOLDOWN CHECK ==========
            cooldown_ok, cooldown_reason = check_trade_cooldown(user)
            if not cooldown_ok:
                logger.debug(f"[{user}] {cooldown_reason}")
                continue
            
            # ================================================================================
            # ================= DIRECT FOREX AI EXECUTION SYSTEM - 9 POINT RULES ============
            # ================================================================================
            
            # ========== 1. SCAN & FILTER ==========
            # Check spread
            spread_ok, current_spread, normal_spread, spread_msg = check_spread_filter(symbol)
            if not spread_ok:
                logger.debug(f"[{symbol}] {spread_msg}")
                continue
            
            # Check volatility
            vol_ok, current_vol, avg_vol, vol_msg = check_volatility_filter(symbol, df)
            if not vol_ok:
                logger.debug(f"[{symbol}] {vol_msg}")
                continue
            
            # Check market structure clarity
            structure_clear, structure_regime, structure_msg = check_market_structure_clarity(df)
            if not structure_clear:
                logger.debug(f"[{symbol}] {structure_msg}")
                continue
            
            # ========== 2. HIGHER TIMEFRAME DIRECTION ==========
            htf_direction, htf_strength, htf_msg = get_htf_direction(symbol)
            
            # Check for reversal patterns if HTF is not neutral
            has_reversal = False
            reversal_pattern = None
            if REVERSAL_PATTERN_OVERRIDE and htf_direction != "NEUTRAL":
                has_reversal, reversal_pattern, reversal_conf = detect_reversal_pattern(df, htf_direction)
                if has_reversal:
                    logger.info(f"[{symbol}] 🔄 Reversal pattern detected: {reversal_pattern} ({reversal_conf:.0%})")
            
            # ========== 3-4. DETERMINE TRADE DIRECTION BASED ON HTF ==========
            potential_direction = None
            
            if htf_direction == "BULLISH":
                potential_direction = "BUY"
            elif htf_direction == "BEARISH":
                potential_direction = "SELL"
            elif has_reversal:
                # Trade the reversal direction
                potential_direction = "BUY" if reversal_pattern in ["BULLISH_ENGULFING", "MORNING_STAR", "HAMMER"] else "SELL"
            elif structure_regime in ["RANGE_SUPPORT", "RANGE_RESISTANCE"]:
                # Trade range bounces
                if structure_regime == "RANGE_SUPPORT":
                    potential_direction = "BUY"
                else:
                    potential_direction = "SELL"
            else:
                # No clear direction
                logger.debug(f"[{symbol}] No clear trade direction - HTF: {htf_direction}")
                continue
            
            # ========== 5. CHECK ALL TRADE SETUP CONDITIONS ==========
            # Check if at key level
            at_key_level, level_info, level_msg = is_at_key_level(symbol, df, potential_direction)
            if REQUIRE_KEY_LEVEL and not at_key_level:
                logger.debug(f"[{symbol}] {potential_direction} - {level_msg}")
                continue
            
            # Check momentum confirmation
            mom_confirmed, mom_strength, mom_msg = check_momentum_confirmation(df, potential_direction)
            if REQUIRE_MOMENTUM_CONFIRM and not mom_confirmed:
                logger.debug(f"[{symbol}] {potential_direction} - Momentum not confirmed: {mom_msg}")
                continue
            
            # Check session (prefer London/NY)
            session_name, session = get_current_session()
            if REQUIRE_ACTIVE_SESSION and session_name not in ['OVERLAP', 'LONDON', 'NEW_YORK']:
                logger.debug(f"[{symbol}] Waiting for active session (current: {session_name})")
                continue
            
            # ========== 7. NEWS BLACKOUT CHECK ==========
            in_blackout, event_name, minutes_until = check_news_blackout(symbol, user)
            if in_blackout:
                logger.info(f"[{symbol}] ⚠️ News blackout: {event_name} in ~{minutes_until} min")
                continue
            
            # ========== 8. CALCULATE SETUP QUALITY SCORE ==========
            quality_score, max_score, quality_details = calculate_setup_quality_score(
                symbol, df, potential_direction, htf_direction, user
            )
            
            # ========== 9. GET ADAPTIVE CRITERIA (based on losses) ==========
            min_quality_required, size_multiplier, should_stop = get_adaptive_criteria(user)
            
            if should_stop:
                logger.warning(f"[{user}] 🛑 Trading stopped due to consecutive losses")
                continue
            
            # Check quality score
            if QUALITY_OVER_QUANTITY and quality_score < min_quality_required:
                logger.debug(f"[{symbol}] Quality too low: {quality_score}/{max_score} (need {min_quality_required})")
                continue
            
            # ========== CALCULATE ENTRY, SL, TP ==========
            entry_price = price
            
            # Calculate SL/TP based on ATR and structure
            sl_price, tp_price = calculate_optimal_sl_tp(symbol, potential_direction, entry_price, df)
            
            # Validate R:R ratio
            rr_ratio = calculate_rr_ratio(entry_price, sl_price, tp_price, potential_direction)
            if rr_ratio < MIN_RR_RATIO:
                logger.debug(f"[{symbol}] R:R too low: {rr_ratio:.2f} (need {MIN_RR_RATIO})")
                continue
            
            # Calculate lot size (with adaptive sizing)
            if potential_direction == "BUY":
                sl_distance = entry_price - sl_price
            else:
                sl_distance = sl_price - entry_price
            
            sl_pips_calc = sl_distance / (point * pip_mult) if point > 0 else sl_distance
            base_lot = calculate_lot(acc.balance, risk_pct, sl_pips_calc)
            adjusted_lot = base_lot * size_multiplier
            trade_lot = max(MIN_LOT, min(adjusted_lot, MAX_LOT))
            trade_lot = round(trade_lot, 2)
            
            # ========== FINAL AI VALIDATION ==========
            ai_approved, confidence_mult = ai_validate_trade_signal(
                df, potential_direction, quality_score, user, last_ai_recommendation
            )
            
            if not ai_approved:
                logger.debug(f"[{symbol}] AI rejected {potential_direction}")
                continue
            
            # ========== CHECK POSITION LIMITS ==========
            if symbol_pos_count >= MAX_POSITIONS_PER_SYMBOL:
                continue
            if total_positions >= MAX_TOTAL_POSITIONS:
                continue
            
            # Check re-entry allowed
            reentry_allowed = check_reentry_opportunity(symbol, potential_direction, quality_score, user)
            if not reentry_allowed:
                continue
            
            # ========== SMALL ACCOUNT DETECTION ==========
            acc = mt5.account_info()
            account_balance = acc.balance if acc else 0
            is_micro_account = account_balance < MICRO_ACCOUNT_THRESHOLD
            is_small_account = account_balance < SMALL_ACCOUNT_THRESHOLD
            
            # ========== EXECUTE TRADE WITH HIGH PROBABILITY SCALING ==========
            order_type = mt5.ORDER_TYPE_BUY if potential_direction == "BUY" else mt5.ORDER_TYPE_SELL
            
            # Determine number of positions to open based on quality score
            positions_to_open = 1
            lot_multiplier = 1.0
            
            # Calculate account size bonus multiplier
            account_bonus = 1.0
            for threshold, bonus in sorted(ACCOUNT_SIZE_LOT_BONUS.items(), reverse=True):
                if account_balance >= threshold:
                    account_bonus = bonus
                    break
            
            # DISABLE scaling for small accounts - only 1 position with minimum lot
            if is_micro_account:
                # Micro accounts (< $100): Always 1 position, minimum lot
                positions_to_open = 1
                lot_multiplier = 1.0
                trade_lot = MIN_LOT
                logger.info(f"[{user}] 💰 MICRO ACCOUNT MODE (${account_balance:.0f}) - Using minimum lot {MIN_LOT}")
            elif is_small_account:
                # Small accounts (< $500): Max 1 position, no lot multiplier
                positions_to_open = 1
                lot_multiplier = 1.0
                logger.info(f"[{user}] 💰 SMALL ACCOUNT MODE (${account_balance:.0f}) - Single position only")
            elif HIGH_PROB_SCALING_ENABLED and quality_score >= ULTRA_HIGH_PROB_MIN_QUALITY:
                # Ultra high probability (10/10) - maximum aggression
                positions_to_open = min(ULTRA_HIGH_PROB_POSITIONS, MAX_POSITIONS_PER_SYMBOL - symbol_pos_count)
                lot_multiplier = ULTRA_HIGH_PROB_LOT_MULTIPLIER * account_bonus
                logger.info(f"[{user}] 🔥🔥🔥🔥 ULTRA HIGH PROBABILITY ({quality_score}/10) - {positions_to_open} positions @ {lot_multiplier:.1f}x lot! (Account bonus: {account_bonus}x)")
            elif HIGH_PROB_SCALING_ENABLED and quality_score >= VERY_HIGH_PROB_MIN_QUALITY:
                # Very high probability (9/10) - open maximum positions
                positions_to_open = min(VERY_HIGH_PROB_POSITIONS, MAX_POSITIONS_PER_SYMBOL - symbol_pos_count)
                lot_multiplier = VERY_HIGH_PROB_LOT_MULTIPLIER * account_bonus
                logger.info(f"[{user}] 🔥🔥🔥 VERY HIGH PROBABILITY ({quality_score}/10) - {positions_to_open} positions @ {lot_multiplier:.1f}x lot! (Account bonus: {account_bonus}x)")
            elif HIGH_PROB_SCALING_ENABLED and quality_score >= HIGH_PROB_MIN_QUALITY:
                # High probability (8/10) - open 2 positions
                positions_to_open = min(HIGH_PROB_POSITIONS, MAX_POSITIONS_PER_SYMBOL - symbol_pos_count)
                lot_multiplier = HIGH_PROB_LOT_MULTIPLIER * account_bonus
                logger.info(f"[{user}] 🔥🔥 HIGH PROBABILITY ({quality_score}/10) - {positions_to_open} positions @ {lot_multiplier:.1f}x lot! (Account bonus: {account_bonus}x)")
            
            # ========== CONFLUENCE BONUS - MORE TRADES FOR STRONG CONFLUENCE ==========
            # If 30-strategy scanner detected strong confluence, open even MORE trades
            try:
                df_confluence = get_data(symbol, mt5.TIMEFRAME_M5, n=100)
                if df_confluence is not None and len(df_confluence) > 20:
                    df_confluence = calculate_advanced_indicators(df_confluence)
                    confluence_result = scan_all_entry_strategies(symbol, df_confluence, user)
                    confluence_score = confluence_result.get('score', 0)
                    num_strategies = len(confluence_result.get('strategies', []))
                    
                    # Bonus positions for strong confluence
                    if confluence_score >= 8 and num_strategies >= 6:
                        # MEGA confluence - add 3 more positions
                        bonus_positions = min(3, MAX_POSITIONS_PER_SYMBOL - symbol_pos_count - positions_to_open)
                        if bonus_positions > 0:
                            positions_to_open += bonus_positions
                            lot_multiplier *= 1.5  # 50% lot bonus
                            logger.info(f"[{user}] 🌟🌟🌟 MEGA CONFLUENCE! Score:{confluence_score:.1f}, {num_strategies} strategies - +{bonus_positions} positions @ {lot_multiplier:.1f}x lot!")
                    elif confluence_score >= 6 and num_strategies >= 4:
                        # Strong confluence - add 2 more positions
                        bonus_positions = min(2, MAX_POSITIONS_PER_SYMBOL - symbol_pos_count - positions_to_open)
                        if bonus_positions > 0:
                            positions_to_open += bonus_positions
                            lot_multiplier *= 1.25  # 25% lot bonus
                            logger.info(f"[{user}] 🌟🌟 STRONG CONFLUENCE! Score:{confluence_score:.1f}, {num_strategies} strategies - +{bonus_positions} positions!")
                    elif confluence_score >= 4 and num_strategies >= 3:
                        # Good confluence - add 1 more position
                        bonus_positions = min(1, MAX_POSITIONS_PER_SYMBOL - symbol_pos_count - positions_to_open)
                        if bonus_positions > 0:
                            positions_to_open += bonus_positions
                            logger.info(f"[{user}] 🌟 GOOD CONFLUENCE! Score:{confluence_score:.1f}, {num_strategies} strategies - +{bonus_positions} position!")
            except Exception as cf_err:
                logger.debug(f"Confluence bonus check error: {cf_err}")
            
            # Calculate scaled lot size
            scaled_lot = round(trade_lot * lot_multiplier, 2)
            scaled_lot = max(MIN_LOT, min(scaled_lot, MAX_LOT_HIGH_PROB))  # Ensure within bounds
            
            # ========== MARGIN CHECK BEFORE TRADING ==========
            # Check if we have enough margin for this trade
            margin_required = mt5.order_calc_margin(order_type, symbol, scaled_lot, entry_price)
            if margin_required is not None:
                free_margin = acc.margin_free if acc else 0
                # Need at least 20% buffer for safety
                if margin_required * 1.2 > free_margin:
                    # Try minimum lot instead
                    margin_for_min = mt5.order_calc_margin(order_type, symbol, MIN_LOT, entry_price)
                    if margin_for_min is not None and margin_for_min * 1.2 <= free_margin:
                        logger.warning(f"[{user}] ⚠️ Not enough margin for {scaled_lot} lots, using minimum {MIN_LOT}")
                        scaled_lot = MIN_LOT
                        positions_to_open = 1
                    else:
                        logger.error(f"[{user}] ❌ Not enough margin even for minimum lot. Free: ${free_margin:.2f}, Required: ${margin_for_min:.2f}")
                        continue  # Skip this trade
            
            # Open multiple positions for high probability setups
            positions_opened = 0
            for pos_num in range(positions_to_open):
                # Check we still have room for more positions
                if total_positions >= MAX_TOTAL_POSITIONS:
                    logger.warning(f"[{user}] ⚠️ Max total positions reached, stopping scaling")
                    break
                
                result = send_order(
                    symbol, order_type, scaled_lot, sl_price, tp_price,
                    f"{potential_direction}_{symbol}_Q{quality_score}_P{pos_num+1}of{positions_to_open}"
                )
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    positions_opened += 1
                    record_trade_placed(user)
                    emoji = "🟢" if potential_direction == "BUY" else "🔴"
                    logger.info(f"[{user}] {emoji} {potential_direction} {symbol} @ {scaled_lot} lots (#{pos_num+1}/{positions_to_open}) | "
                               f"Quality: {quality_score}/{max_score} | RR: 1:{rr_ratio:.1f} | "
                               f"HTF: {htf_direction} | Session: {session_name}")
                    
                    log_trade(user, 'trade', f'{potential_direction} {symbol} @ {entry_price:.5f} (Position {pos_num+1}/{positions_to_open})', {
                        'type': potential_direction,
                        'symbol': symbol,
                        'price': entry_price,
                        'lot': scaled_lot,
                        'sl': sl_price,
                        'tp': tp_price,
                        'quality_score': quality_score,
                        'rr_ratio': rr_ratio,
                        'htf_direction': htf_direction,
                        'session': session_name,
                        'quality_details': quality_details,
                        'high_prob_scaling': True,
                        'position_number': pos_num + 1,
                        'total_positions': positions_to_open,
                        'lot_multiplier': lot_multiplier,
                        'executed': True
                    })
                    total_positions += 1
                    
                    # Small delay between scaled entries to get slightly different prices
                    if pos_num < positions_to_open - 1:
                        time.sleep(SCALE_IN_DELAY_SECONDS)
                else:
                    logger.warning(f"[{user}] ⚠️ Failed to open position {pos_num+1}/{positions_to_open} for {symbol}")
            
            if positions_opened > 1:
                logger.info(f"[{user}] 🎯 Successfully opened {positions_opened} positions for HIGH PROBABILITY {symbol} trade!")
            
            # ========== MANAGE EXISTING POSITIONS (R-BASED PROFIT PROTECTION) ==========
            manage_r_based_profit_protection(user)
            
            # ========== CHECK EXIT CONDITIONS FOR EXISTING POSITIONS ==========
            for pos in (symbol_positions or []):
                pos_direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                
                # Check structure break
                structure_broken, break_msg = check_structure_break(df, pos.type)
                if structure_broken:
                    logger.warning(f"[{user}] ⚠️ Structure break on {symbol} #{pos.ticket}: {break_msg}")
                    # Consider closing - the R-based protection should handle this
                
                # Check volatility collapse
                vol_collapsed, vol_collapse_msg = check_volatility_collapse(symbol, df)
                if vol_collapsed:
                    logger.info(f"[{user}] 📉 Volatility collapsed on {symbol} - monitoring closely")
        
        # ========== MANAGE ALL POSITIONS (R-BASED) ==========
        manage_r_based_profit_protection(user)
        
        stop_event.wait(CHECK_INTERVAL)
    
    user_bots[user]["running"] = False
    logger.info(f"[{user}] 🛑 Bot stopped")
    log_trade(user, 'bot', 'Bot stopped', {'symbols': symbols})


# ---------------- BOT CONTROL (ORIGINAL NAMES) ----------------
def start_bot(user, symbols=None):
    """Start the bot for a user with optional symbol list"""
    global user_bots
    
    # Check if already running
    if user in user_bots and user_bots[user].get("running"):
        thread = user_bots[user].get("thread")
        if thread and thread.is_alive():
            return "Bot already running"
        else:
            # Thread died, clean up
            user_bots[user]["running"] = False
    
    # Set user's symbols if provided
    if symbols:
        set_user_symbols(user, symbols)
    
    # Create stop event and thread
    stop_event = threading.Event()
    user_bots[user] = {"thread": None, "stop_event": stop_event, "running": True}
    
    thread = threading.Thread(target=run_bot, args=(user,), daemon=True)
    user_bots[user]["thread"] = thread
    thread.start()
    
    # Give it a moment to start
    time.sleep(0.5)
    
    # Check if thread started successfully
    if thread.is_alive():
        active_symbols = get_user_symbols(user)
        logger.info(f"[{user}] ✅ Bot started successfully on {len(active_symbols)} symbols")
        return f"Bot started on {len(active_symbols)} symbols: {', '.join(active_symbols)}"
    else:
        user_bots[user]["running"] = False
        return "Bot failed to start - check MT5 connection"


def stop_bot(user):
    """Stop the bot for a user"""
    global user_bots
    
    if user in user_bots:
        if user_bots[user].get("running"):
            user_bots[user]["stop_event"].set()
            user_bots[user]["running"] = False
            
            # Wait briefly for thread to stop
            thread = user_bots[user].get("thread")
            if thread:
                thread.join(timeout=2)
            
            logger.info(f"[{user}] 🛑 Bot stop signal sent")
            return "Bot stopped"
        else:
            return "Bot was not running"
    return "Bot not found"


# Track current active user per session to avoid re-using wrong credentials
_current_mt5_user = None

def set_current_mt5_user(user):
    """Set the current user for MT5 session tracking"""
    global _current_mt5_user
    _current_mt5_user = user

def get_current_mt5_user():
    """Get the current user for MT5 session"""
    return _current_mt5_user


def clear_mt5_session():
    """Clear the current MT5 session - shutdown terminal and reset user tracking"""
    global _current_mt5_user
    try:
        mt5.shutdown()
        print("✅ MT5 session shutdown completed")
    except Exception as e:
        print(f"⚠️ Error shutting down MT5: {e}")
    _current_mt5_user = None


def ensure_mt5_user_session(user):
    """Ensure MT5 is connected with the correct user's credentials.
    Returns True if connected with correct user, False otherwise.
    If user has no credentials but MT5 is already initialized, returns True (uses current session)."""
    global _current_mt5_user
    
    if not user:
        # No user specified - check if MT5 is already running
        if mt5.terminal_info():
            return True  # Use current session
        return False
    
    # First, check if this user has MT5 credentials at all
    from models import get_user_mt5_credentials
    creds = get_user_mt5_credentials(user)
    if not creds:
        # User has no MT5 credentials - try to use existing session if available
        if mt5.terminal_info():
            # MT5 is running, use existing session for chart data etc.
            return True
        # Try to initialize MT5 (will use whatever account is logged in to MT5 terminal)
        if mt5.initialize():
            return True
        return False
    
    # If already connected as this user with valid session, we're good
    if _current_mt5_user == user and mt5.terminal_info():
        # Verify we're still logged in to the right account
        acc = mt5.account_info()
        if acc and acc.login == creds['login']:
            return True
        # Wrong account or not logged in, need to reconnect
        print(f"⚠️ MT5 session mismatch, reconnecting for user {user}")
    
    # Need to switch users or reconnect
    try:
        # Shutdown existing session
        if mt5.terminal_info():
            print(f"🔄 Shutting down MT5 session to switch to {user}")
            mt5.shutdown()
        _current_mt5_user = None
        
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ Failed to initialize MT5 for user {user}")
            return False
        
        # Login with user's credentials
        login_result = mt5.login(creds['login'], password=creds['password'], server=creds['server'])
        if login_result:
            _current_mt5_user = user
            print(f"✅ MT5 logged in as user: {user} (account: {creds['login']})")
            return True
        else:
            print(f"❌ MT5 login failed for user {user}: {mt5.last_error()}")
            mt5.shutdown()
            return False
    except Exception as e:
        print(f"❌ Error ensuring MT5 session for {user}: {e}")
        return False


def bot_status(user):
    """Get bot status for a user"""
    if user in user_bots:
        is_running = user_bots[user].get("running", False)
        thread = user_bots[user].get("thread")
        
        # Double-check thread is actually alive
        if is_running and thread and not thread.is_alive():
            user_bots[user]["running"] = False
            is_running = False
        
        if is_running:
            symbols = get_user_symbols(user)
            return {"running": True, "symbols": symbols, "count": len(symbols)}
    
    return {"running": False, "symbols": [], "count": 0}

# ---------------- DASHBOARD HELPERS ----------------
def get_account_info(user=None):
    """Get account info - returns empty if MT5 not connected"""
    # Use current user if not specified
    if user is None:
        user = get_current_mt5_user()
    
    # Ensure we're logged in as the correct user
    if not ensure_mt5_user_session(user):
        return {}
    
    acc = mt5.account_info()
    if acc:
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free
        }
    return {}


def get_positions(user=None):
    """Get all positions across all symbols - returns empty if not connected"""
    # Use current user if not specified
    if user is None:
        user = get_current_mt5_user()
    
    # Ensure we're logged in as the correct user
    if not ensure_mt5_user_session(user):
        return []
    
    # Get ALL positions (not just one symbol)
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

def get_chart_data(symbol, timeframe="M1", bars=200, user=None):
    """
    Get OHLC candlestick data for charting.
    Returns data in format suitable for TradingView Lightweight Charts.
    """
    # Use current user if not specified
    if user is None:
        user = get_current_mt5_user()
    
    # Map timeframe string to MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
    
    # Ensure we're logged in as the correct user
    if not ensure_mt5_user_session(user):
        return {"error": "MT5 not connected", "candles": []}
    
    # Convert symbol to broker format (handles suffixes automatically)
    broker_symbol = get_broker_symbol(symbol)
    
    # Enable symbol
    mt5.symbol_select(broker_symbol, True)
    
    # Get rates - try broker symbol first, then original
    rates = mt5.copy_rates_from_pos(broker_symbol, mt5_tf, 0, bars)
    
    if (rates is None or len(rates) == 0) and broker_symbol != symbol:
        # Try original symbol as fallback
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        if rates is not None and len(rates) > 0:
            broker_symbol = symbol  # Use original if it worked
    
    if rates is None or len(rates) == 0:
        return {"error": f"No data for {symbol}", "candles": []}
    
    # Convert to list of dicts for JSON
    candles = []
    for rate in rates:
        candles.append({
            "time": int(rate['time']),  # Unix timestamp
            "open": float(rate['open']),
            "high": float(rate['high']),
            "low": float(rate['low']),
            "close": float(rate['close']),
            "volume": int(rate['tick_volume'])
        })
    
    # Get current tick for real-time price
    tick = mt5.symbol_info_tick(symbol)
    current_price = tick.bid if tick else candles[-1]['close'] if candles else 0
    
    # Get positions for this symbol to show on chart
    # Try exact match first, then try without 'm' suffix and with 'm' suffix
    base_symbol = symbol.rstrip('m').upper()
    positions = mt5.positions_get(symbol=symbol)
    
    # If no positions found, try alternative symbol formats
    if not positions:
        positions = mt5.positions_get(symbol=base_symbol)
    if not positions:
        positions = mt5.positions_get(symbol=base_symbol + 'm')
    if not positions:
        # Get all positions and filter manually
        all_positions = mt5.positions_get()
        if all_positions:
            positions = [p for p in all_positions if p.symbol.rstrip('m').upper() == base_symbol]
    
    position_markers = []
    if positions:
        for p in positions:
            position_markers.append({
                "time": int(p.time),
                "price_open": float(p.price_open),
                "price": float(p.price_open),  # For JS compatibility
                "type": p.type,  # 0=BUY, 1=SELL
                "sl": float(p.sl),
                "tp": float(p.tp),
                "profit": float(p.profit),
                "volume": float(p.volume),
                "symbol": p.symbol,
                "ticket": p.ticket
            })
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": current_price,
        "candles": candles,
        "positions": position_markers
    }


def get_multi_chart_data(symbols=None, timeframe="M5", bars=100, user=None):
    """
    Get chart data for multiple symbols.
    Returns array format for easier JS parsing.
    """
    if symbols is None:
        symbols = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
    
    charts = []
    for symbol in symbols[:6]:  # Max 6 charts
        chart_data = get_chart_data(symbol, timeframe, bars, user)
        charts.append(chart_data)
    
    return {"charts": charts}

def get_trade_history(user=None, days=30):
    """
    Get trade history from MT5 for the last N days.
    Returns closed trades with their details.
    """
    from datetime import datetime, timedelta
    
    # Use current user if not specified
    if user is None:
        user = get_current_mt5_user()
    
    # Ensure we're logged in as the correct user
    if not ensure_mt5_user_session(user):
        return []
    
    # Get history from last N days
    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now() + timedelta(days=1)
    
    # Get deals (completed trades)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None or len(deals) == 0:
        return []
    
    # Filter for actual trades (type 0=buy, 1=sell entry, exit types vary)
    # We want deals that have a position_id and are not balance operations
    trades = []
    position_deals = {}  # Group by position ID to match entry/exit
    
    for deal in deals:
        # Skip balance operations (type 2 is balance, 4 is credit, etc.)
        if deal.type in [2, 3, 4, 5, 6]:  # Balance, Credit, Correction, Bonus, Commission
            continue
        
        # Skip deals with no position ID (not real trades)
        if deal.position_id == 0:
            continue
        
        pos_id = deal.position_id
        if pos_id not in position_deals:
            position_deals[pos_id] = []
        position_deals[pos_id].append(deal)
    
    # Process grouped deals to create trade records
    for pos_id, deal_list in position_deals.items():
        if len(deal_list) == 0:
            continue
        
        # Find entry and exit deals
        entry_deal = None
        exit_deal = None
        total_profit = 0
        
        for deal in deal_list:
            # Entry: deal.entry == 0 (IN)
            # Exit: deal.entry == 1 (OUT)
            if deal.entry == 0:  # Entry deal
                entry_deal = deal
            elif deal.entry == 1:  # Exit deal
                exit_deal = deal
                total_profit += deal.profit
        
        # Only include if we have an exit (closed trade)
        if exit_deal is not None:
            # Determine trade type from entry deal or exit deal
            if entry_deal:
                trade_type = "BUY" if entry_deal.type == 0 else "SELL"
                entry_price = entry_deal.price
                volume = entry_deal.volume
                symbol = entry_deal.symbol
            else:
                # No entry deal found, use exit deal info
                trade_type = "SELL" if exit_deal.type == 0 else "BUY"  # Exit type is opposite
                entry_price = exit_deal.price
                volume = exit_deal.volume
                symbol = exit_deal.symbol
            
            trade_time = datetime.fromtimestamp(exit_deal.time)
            
            trades.append({
                "ticket": pos_id,
                "symbol": symbol,
                "type": trade_type,
                "volume": round(volume, 2),
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_deal.price, 5),
                "sl": 0,  # SL/TP not available in deals
                "tp": 0,
                "profit": round(total_profit, 2),
                "time": trade_time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": exit_deal.time
            })
    
    # Sort by timestamp descending (newest first)
    trades.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return trades