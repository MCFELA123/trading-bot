import sys
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime

# Force unbuffered output with UTF-8 encoding (for emojis on Windows)
try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    # Fallback for older Python versions
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# Configure logging BEFORE importing botlogic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
# Also set root logger level
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mail import Mail, Message
from dotenv import load_dotenv
from config import Config

# Now import botlogic AFTER logging is configured
from botlogic import (start_bot, stop_bot, stop_all_bots, bot_status, get_account_info, get_positions, 
                      test_mt5_connection, get_ai_insights, ai_get_market_sentiment, 
                      get_ai_optimized_params, get_user_symbols, set_user_symbols,
                      add_user_symbol, remove_user_symbol, get_available_symbols,
                      DEFAULT_SYMBOLS, SYMBOL_SETTINGS, get_news_analysis, get_economic_calendar,
                      get_chart_data, get_multi_chart_data, get_loss_protection_status,
                      generate_explicit_trade_signal, execute_explicit_signal, get_current_session,
                      set_loss_protection_enabled, get_loss_protection_enabled,
                      get_live_market_sentiment, get_all_live_sentiments,
                      ai_find_entry_points, ai_execute_news_trade, ai_execute_entry_trade,
                      get_optimal_trading_time, get_best_trading_hours_today, should_trade_this_session,
                      clear_all_emergency_stops, clear_emergency_stop, clear_mt5_session,
                      get_trade_history)
from models import (init_db, add_user, verify_user, get_user_by_username, 
                    create_default_admin, update_mt5_credentials, get_user_mt5_credentials, disconnect_mt5,
                    get_trading_logs, clear_trading_logs, add_trading_log,
                    generate_verification_code, store_pending_verification, verify_code_and_create_user,
                    get_pending_verification, resend_verification_code, get_user_by_email,
                    store_password_reset, verify_reset_code, reset_user_password, resend_reset_code,
                    change_user_password, store_password_change_otp, verify_password_change_otp,
                    complete_password_change_with_otp, resend_password_change_otp)

# Load environment variables
load_dotenv()

print("=" * 50, flush=True)
print("🚀 Starting Trading Bot Application...", flush=True)
print("=" * 50, flush=True)

app = Flask(__name__)
# Generate new secret key on each restart - this invalidates all sessions and requires re-login
app.secret_key = secrets.token_hex(32)
print("✅ New session secret key generated", flush=True)

# Email Configuration
app.config['MAIL_SERVER'] = Config.MAIL_SERVER
app.config['MAIL_PORT'] = Config.MAIL_PORT
app.config['MAIL_USE_TLS'] = Config.MAIL_USE_TLS
app.config['MAIL_USE_SSL'] = Config.MAIL_USE_SSL
app.config['MAIL_USERNAME'] = Config.MAIL_USERNAME
app.config['MAIL_PASSWORD'] = Config.MAIL_PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = Config.MAIL_DEFAULT_SENDER
print(f"✅ Email configured: {Config.MAIL_SERVER}:{Config.MAIL_PORT}", flush=True)

mail = Mail(app)

# ---------------- BACKGROUND AUTO-SCAN MANAGER ----------------
# Stores background scan threads and state per user
auto_scan_threads = {}  # {username: thread}
auto_scan_running = {}  # {username: bool}
auto_scan_status = {}   # {username: {symbol, last_scan, last_entry, etc}}
auto_scan_lock = threading.Lock()

# Use standard symbol names - bot will auto-detect broker suffix
AUTO_SCAN_SYMBOLS = ['XAUUSD', 'BTCUSD', 'EURUSD', 'GBPUSD', 'XAGUSD']
AUTO_SCAN_INTERVAL = 15  # seconds
MIN_QUALITY_SCORE = 7

def background_auto_scan(username):
    """Background thread that continuously scans for entry points"""
    global auto_scan_running, auto_scan_status
    
    symbol_index = 0
    logger.info(f"🔍 Background auto-scan started for user: {username}")
    
    while auto_scan_running.get(username, False):
        try:
            symbol = AUTO_SCAN_SYMBOLS[symbol_index]
            symbol_index = (symbol_index + 1) % len(AUTO_SCAN_SYMBOLS)
            
            with auto_scan_lock:
                auto_scan_status[username] = {
                    'scanning': True,
                    'current_symbol': symbol,
                    'last_scan_time': datetime.now().isoformat(),
                    'status': f'Scanning {symbol}...'
                }
            
            # Call the AI entry finding function
            entry = ai_find_entry_points(symbol, username)
            
            with auto_scan_lock:
                if entry.get('has_entry', False):
                    quality = entry.get('quality_score', 0)
                    if quality >= MIN_QUALITY_SCORE:
                        # Auto-execute the trade
                        result = ai_execute_entry_trade(symbol, username, None)
                        auto_scan_status[username] = {
                            'scanning': True,
                            'current_symbol': symbol,
                            'last_scan_time': datetime.now().isoformat(),
                            'status': f'Trade executed on {symbol}!' if result.get('success') else f'Trade failed: {result.get("reason")}',
                            'last_entry': {
                                'symbol': symbol,
                                'direction': entry.get('direction'),
                                'quality': quality,
                                'executed': result.get('success', False),
                                'ticket': result.get('ticket'),
                                'time': datetime.now().isoformat()
                            }
                        }
                        logger.info(f"🎯 Auto-scan executed trade on {symbol} for {username}: {result}")
                    else:
                        auto_scan_status[username] = {
                            'scanning': True,
                            'current_symbol': symbol,
                            'last_scan_time': datetime.now().isoformat(),
                            'status': f'{symbol}: Entry found but quality {quality}/10 < {MIN_QUALITY_SCORE}'
                        }
                else:
                    auto_scan_status[username] = {
                        'scanning': True,
                        'current_symbol': symbol,
                        'last_scan_time': datetime.now().isoformat(),
                        'status': f'{symbol}: {entry.get("reason", "No setup")}'
                    }
            
            # Wait before next scan
            for _ in range(AUTO_SCAN_INTERVAL):
                if not auto_scan_running.get(username, False):
                    break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Auto-scan error for {username}: {e}")
            with auto_scan_lock:
                auto_scan_status[username] = {
                    'scanning': True,
                    'current_symbol': symbol,
                    'last_scan_time': datetime.now().isoformat(),
                    'status': f'Error: {str(e)}'
                }
            time.sleep(5)
    
    logger.info(f"🛑 Background auto-scan stopped for user: {username}")
    with auto_scan_lock:
        auto_scan_status[username] = {
            'scanning': False,
            'status': 'Auto-scan stopped'
        }

def start_background_scan(username):
    """Start background auto-scan for a user"""
    global auto_scan_threads, auto_scan_running
    
    with auto_scan_lock:
        # Stop existing thread if running
        if auto_scan_running.get(username, False):
            return True  # Already running
        
        auto_scan_running[username] = True
        auto_scan_status[username] = {
            'scanning': True,
            'status': 'Starting auto-scan...'
        }
    
    thread = threading.Thread(target=background_auto_scan, args=(username,), daemon=True)
    thread.start()
    auto_scan_threads[username] = thread
    return True

def stop_background_scan(username):
    """Stop background auto-scan for a user"""
    global auto_scan_running
    
    with auto_scan_lock:
        auto_scan_running[username] = False
        auto_scan_status[username] = {
            'scanning': False,
            'status': 'Auto-scan stopped'
        }
    return True

def get_scan_status(username):
    """Get current scan status for a user"""
    with auto_scan_lock:
        return auto_scan_status.get(username, {'scanning': False, 'status': 'Not started'})

# ---------------- BACKGROUND SIGNAL AUTO-EXECUTE MANAGER ----------------
# Auto-executes explicit trade signals in the background
signal_auto_execute_threads = {}  # {username: thread}
signal_auto_execute_running = {}  # {username: bool}
signal_auto_execute_status = {}   # {username: {status, last_signal, etc}}
signal_auto_execute_lock = threading.Lock()

SIGNAL_CHECK_INTERVAL = 15  # Check for signals every 15 seconds
SIGNAL_MIN_SCORE = 7  # Minimum score to auto-execute
# Use standard symbol names - bot will auto-detect broker suffix
SIGNAL_SYMBOLS = ['XAUUSD', 'XAGUSD', 'BTCUSD', 'EURUSD', 'GBPUSD', 'USDJPY']

def background_signal_auto_execute(username):
    """Background thread that auto-executes explicit trade signals"""
    global signal_auto_execute_running, signal_auto_execute_status
    
    executed_signals = set()  # Track executed to avoid duplicates
    logger.info(f"🎯 Background signal auto-execute started for user: {username}")
    
    while signal_auto_execute_running.get(username, False):
        try:
            with signal_auto_execute_lock:
                signal_auto_execute_status[username] = {
                    'running': True,
                    'status': 'Scanning for signals...',
                    'last_check': datetime.now().isoformat()
                }
            
            # Check each symbol for signals
            for symbol in SIGNAL_SYMBOLS:
                if not signal_auto_execute_running.get(username, False):
                    break
                    
                signal = generate_explicit_trade_signal(symbol, username)
                
                if signal and signal.get('signal') in ['BUY', 'SELL']:
                    score = signal.get('score', 0)
                    signal_key = f"{symbol}_{signal['signal']}_{signal.get('time', '')}"
                    
                    if score >= SIGNAL_MIN_SCORE and signal_key not in executed_signals:
                        # Execute the signal
                        result = execute_explicit_signal(symbol, username, None)
                        executed_signals.add(signal_key)
                        
                        with signal_auto_execute_lock:
                            signal_auto_execute_status[username] = {
                                'running': True,
                                'status': f"Executed {signal['signal']} {symbol}!" if result.get('success') else f"Failed: {result.get('reason')}",
                                'last_check': datetime.now().isoformat(),
                                'last_signal': {
                                    'symbol': symbol,
                                    'direction': signal['signal'],
                                    'score': score,
                                    'executed': result.get('success', False),
                                    'ticket': result.get('ticket'),
                                    'time': datetime.now().isoformat()
                                }
                            }
                        
                        if result.get('success'):
                            logger.info(f"🎯 Signal auto-executed: {signal['signal']} {symbol} for {username} - Ticket #{result.get('ticket')}")
                        else:
                            logger.warning(f"⚠️ Signal execution failed for {username}: {result.get('reason')}")
            
            # Clean up old executed signals (keep last 50)
            if len(executed_signals) > 50:
                executed_signals = set(list(executed_signals)[-50:])
            
            # Wait before next check
            for _ in range(SIGNAL_CHECK_INTERVAL):
                if not signal_auto_execute_running.get(username, False):
                    break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Signal auto-execute error for {username}: {e}")
            with signal_auto_execute_lock:
                signal_auto_execute_status[username] = {
                    'running': True,
                    'status': f'Error: {str(e)}',
                    'last_check': datetime.now().isoformat()
                }
            time.sleep(5)
    
    logger.info(f"🛑 Background signal auto-execute stopped for user: {username}")
    with signal_auto_execute_lock:
        signal_auto_execute_status[username] = {
            'running': False,
            'status': 'Auto-execute stopped'
        }

def start_signal_auto_execute(username):
    """Start background signal auto-execute for a user"""
    global signal_auto_execute_threads, signal_auto_execute_running
    
    with signal_auto_execute_lock:
        if signal_auto_execute_running.get(username, False):
            return True  # Already running
        
        signal_auto_execute_running[username] = True
        signal_auto_execute_status[username] = {
            'running': True,
            'status': 'Starting auto-execute...'
        }
    
    thread = threading.Thread(target=background_signal_auto_execute, args=(username,), daemon=True)
    thread.start()
    signal_auto_execute_threads[username] = thread
    return True

def stop_signal_auto_execute(username):
    """Stop background signal auto-execute for a user"""
    global signal_auto_execute_running
    
    with signal_auto_execute_lock:
        signal_auto_execute_running[username] = False
        signal_auto_execute_status[username] = {
            'running': False,
            'status': 'Auto-execute stopped'
        }
    return True

def get_signal_auto_execute_status(username):
    """Get current signal auto-execute status for a user"""
    with signal_auto_execute_lock:
        return signal_auto_execute_status.get(username, {'running': False, 'status': 'Not started'})

# ---------------- STOP ALL BOTS ON RESTART ----------------
print("🛑 Stopping any running bots from previous session...", flush=True)
stop_all_bots()  # Clear any running bots from previous session
print("✅ All bots stopped", flush=True)

# ---------------- CLEAR EMERGENCY STOPS ----------------
print("🚨 Clearing any emergency stops...", flush=True)
cleared = clear_all_emergency_stops()
if cleared > 0:
    print(f"✅ Cleared {cleared} emergency stop(s)", flush=True)
else:
    print("✅ No emergency stops to clear", flush=True)

# ---------------- INITIALIZE MONGODB ----------------
db_connected = init_db()
if db_connected:
    create_default_admin()  # Create admin/admin123 if not exists
    print("✅ Database initialized", flush=True)
else:
    print("⚠️ Database not available - some features will be limited", flush=True)

# ---------------- HELPERS ----------------
def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def send_verification_email(email, code, username):
    """Send verification code email to user"""
    # Development mode - skip email and show code directly
    if Config.EMAIL_DEV_MODE:
        print(f"📧 [DEV MODE] Verification code for {email}: {code}")
        return True, None, code  # Return code for dev mode display
    
    # Check if email is configured
    if not Config.MAIL_USERNAME or not Config.MAIL_PASSWORD:
        print(f"⚠️ Email not configured. Code for {email}: {code}")
        return True, None, code  # Return code when email not configured
    
    try:
        msg = Message(
            subject='🔐 TradingBot - Verify Your Email',
            recipients=[email]
        )
        msg.html = f'''
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #0a1628 0%, #1a2332 100%); padding: 30px; border-radius: 12px 12px 0 0; text-align: center;">
                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">🤖 TradingBot</h1>
            </div>
            <div style="background: #ffffff; padding: 40px 30px; border: 1px solid #e2e8f0; border-top: none;">
                <h2 style="color: #0a1628; margin-top: 0;">Welcome, {username}! 👋</h2>
                <p style="color: #5b6b8b; font-size: 16px; line-height: 1.6;">
                    Thank you for signing up for TradingBot. To complete your registration, please enter the verification code below:
                </p>
                <div style="background: #f7fafc; border: 2px dashed #0052ff; border-radius: 12px; padding: 25px; text-align: center; margin: 30px 0;">
                    <span style="font-size: 36px; font-weight: 700; letter-spacing: 8px; color: #0052ff;">{code}</span>
                </div>
                <p style="color: #5b6b8b; font-size: 14px;">
                    ⏰ This code will expire in <strong>10 minutes</strong>.
                </p>
                <p style="color: #5b6b8b; font-size: 14px;">
                    If you didn't request this verification, please ignore this email.
                </p>
            </div>
            <div style="background: #f7fafc; padding: 20px 30px; border-radius: 0 0 12px 12px; text-align: center; border: 1px solid #e2e8f0; border-top: none;">
                <p style="color: #718096; font-size: 12px; margin: 0;">
                    © 2026 TradingBot. Automated Trading Made Simple.
                </p>
            </div>
        </div>
        '''
        msg.body = f'''
TradingBot - Email Verification

Welcome, {username}!

Your verification code is: {code}

This code will expire in 10 minutes.

If you didn't request this verification, please ignore this email.
        '''
        mail.send(msg)
        return True, None, None  # Email sent successfully
    except Exception as e:
        print(f"❌ Failed to send verification email: {e}")
        return False, str(e), code  # Return code on failure so user can still proceed

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_or_email = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username_or_email or not password:
            return render_template("login.html", error="Please fill in all fields")
        
        user = verify_user(username_or_email, password)
        if user:
            # Clear any previous user's MT5 session completely
            clear_mt5_session()
            session.clear()
            
            # Set new user session
            session["user"] = user.username
            session["user_id"] = user.id
            session["role"] = user.role
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username/email or password")
    return render_template("login.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        # Validation
        if not username or not email or not password or not confirm_password:
            return render_template("signup.html", error="Please fill in all fields")
        
        if len(username) < 3:
            return render_template("signup.html", error="Username must be at least 3 characters")
        
        if len(username) > 20:
            return render_template("signup.html", error="Username must be less than 20 characters")
        
        if not username.isalnum():
            return render_template("signup.html", error="Username can only contain letters and numbers")
        
        if not is_valid_email(email):
            return render_template("signup.html", error="Please enter a valid email address")
        
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters")
        
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")
        
        # Check if username already exists
        if get_user_by_username(username):
            return render_template("signup.html", error="Username already exists")
        
        # Check if email already exists
        if get_user_by_email(email):
            return render_template("signup.html", error="Email already exists")
        
        # Generate verification code
        code = generate_verification_code()
        
        # Store pending verification
        store_pending_verification(username, email, password, code)
        
        # Send verification email
        success, error_msg, fallback_code = send_verification_email(email, code, username)
        
        # Store email in session for verification page
        session['pending_email'] = email
        session['pending_username'] = username
        
        # Show code on verification page if:
        # 1. DEV MODE is enabled, OR
        # 2. Email sending FAILED (so user can still proceed)
        session.pop('dev_mode_code', None)
        if fallback_code:  # Always show code if available (dev mode or email failed)
            session['dev_mode_code'] = fallback_code
        
        return redirect(url_for("verify_email"))
    
    return render_template("signup.html")

# ---------------- EMAIL VERIFICATION ----------------
@app.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    email = session.get('pending_email')
    username = session.get('pending_username')
    dev_code = session.get('dev_mode_code')  # For dev mode or when email fails
    
    if not email:
        return redirect(url_for("signup"))
    
    if request.method == "POST":
        code = request.form.get("code", "").strip()
        
        if not code:
            return render_template("verify_email.html", error="Please enter the verification code", email=email, dev_code=dev_code)
        
        if len(code) != 6 or not code.isdigit():
            return render_template("verify_email.html", error="Invalid code format. Please enter the 6-digit code.", email=email, dev_code=dev_code)
        
        # Verify code and create user
        user, error = verify_code_and_create_user(email, code)
        
        if error:
            return render_template("verify_email.html", error=error, email=email, dev_code=dev_code)
        
        if user:
            # Clear any previous MT5 session
            clear_mt5_session()
            
            # Clear pending session data
            session.pop('pending_email', None)
            session.pop('pending_username', None)
            session.pop('dev_mode_code', None)
            
            # Auto-login after verification (don't set MT5 user until they connect)
            session["user"] = user.username
            session["user_id"] = user.id
            session["role"] = user.role
            return redirect(url_for("index"))
        
        return render_template("verify_email.html", error="Verification failed. Please try again.", email=email, dev_code=dev_code)
    
    return render_template("verify_email.html", email=email, username=username, dev_code=dev_code)

# ---------------- RESEND VERIFICATION CODE ----------------
@app.route("/resend-code", methods=["POST"])
def resend_code():
    email = session.get('pending_email')
    username = session.get('pending_username')
    
    if not email:
        return jsonify({"success": False, "error": "No pending verification found"})
    
    # Generate new code
    new_code, error = resend_verification_code(email)
    
    if error:
        return jsonify({"success": False, "error": error})
    
    # Send new verification email
    success, error_msg, fallback_code = send_verification_email(email, new_code, username or "User")
    
    # Update session with new code if available
    if fallback_code:
        session['dev_mode_code'] = fallback_code
    
    if success:
        return jsonify({"success": True, "message": "Verification code sent successfully", "dev_code": fallback_code})
    else:
        # Still return success with code so user can proceed
        return jsonify({"success": True, "message": "Email failed but code is shown below", "dev_code": fallback_code})

# ---------------- FORGOT PASSWORD ----------------
def send_password_reset_email(email, code, username):
    """Send password reset code email to user"""
    # Development mode - skip email and show code directly
    if Config.EMAIL_DEV_MODE:
        print(f"📧 [DEV MODE] Password reset code for {email}: {code}")
        return True, None, code
    
    # Check if email is configured
    if not Config.MAIL_USERNAME or not Config.MAIL_PASSWORD:
        print(f"⚠️ Email not configured. Reset code for {email}: {code}")
        return True, None, code
    
    try:
        msg = Message(
            subject='🔑 TradingBot - Reset Your Password',
            recipients=[email]
        )
        msg.html = f'''
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #0a1628 0%, #1a2332 100%); padding: 30px; border-radius: 12px 12px 0 0; text-align: center;">
                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">🤖 TradingBot</h1>
            </div>
            <div style="background: #ffffff; padding: 40px 30px; border: 1px solid #e2e8f0; border-top: none;">
                <h2 style="color: #0a1628; margin-top: 0;">Password Reset Request 🔐</h2>
                <p style="color: #5b6b8b; font-size: 16px; line-height: 1.6;">
                    Hi {username}, we received a request to reset your password. Use the code below to set a new password:
                </p>
                <div style="background: #f7fafc; border: 2px dashed #ff6b35; border-radius: 12px; padding: 25px; text-align: center; margin: 30px 0;">
                    <span style="font-size: 36px; font-weight: 700; letter-spacing: 8px; color: #ff6b35;">{code}</span>
                </div>
                <p style="color: #5b6b8b; font-size: 14px;">
                    ⏰ This code will expire in <strong>15 minutes</strong>.
                </p>
                <p style="color: #5b6b8b; font-size: 14px;">
                    If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.
                </p>
            </div>
            <div style="background: #f7fafc; padding: 20px 30px; border-radius: 0 0 12px 12px; text-align: center; border: 1px solid #e2e8f0; border-top: none;">
                <p style="color: #718096; font-size: 12px; margin: 0;">
                    © 2026 TradingBot. Automated Trading Made Simple.
                </p>
            </div>
        </div>
        '''
        msg.body = f'''
TradingBot - Password Reset

Hi {username},

We received a request to reset your password.

Your password reset code is: {code}

This code will expire in 15 minutes.

If you didn't request a password reset, you can safely ignore this email.
        '''
        mail.send(msg)
        return True, None, None
    except Exception as e:
        print(f"❌ Failed to send password reset email: {e}")
        return False, str(e), code

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        
        if not email:
            return render_template("forgot_password.html", error="Please enter your email address")
        
        if not is_valid_email(email):
            return render_template("forgot_password.html", error="Please enter a valid email address")
        
        # Check if user exists
        user = get_user_by_email(email)
        if not user:
            # Don't reveal if email exists for security, but still show success
            return render_template("forgot_password.html", 
                success="If an account with that email exists, we've sent a reset code.")
        
        # Generate reset code
        code = generate_verification_code()
        
        # Store reset request
        store_password_reset(email, code)
        
        # Send reset email
        success, error_msg, fallback_code = send_password_reset_email(email, code, user.username)
        
        # Store email in session
        session['reset_email'] = email
        
        # Show code if available (dev mode or email failed)
        session.pop('dev_mode_reset_code', None)
        if fallback_code:
            session['dev_mode_reset_code'] = fallback_code
        
        return redirect(url_for("reset_password"))
    
    return render_template("forgot_password.html")

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    email = session.get('reset_email')
    dev_code = session.get('dev_mode_reset_code')
    
    if not email:
        return redirect(url_for("forgot_password"))
    
    if request.method == "POST":
        code = request.form.get("code", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not code:
            return render_template("reset_password.html", error="Please enter the reset code", email=email, dev_code=dev_code)
        
        if len(code) != 6 or not code.isdigit():
            return render_template("reset_password.html", error="Invalid code format", email=email, dev_code=dev_code)
        
        if not new_password:
            return render_template("reset_password.html", error="Please enter a new password", email=email, dev_code=dev_code)
        
        if len(new_password) < 6:
            return render_template("reset_password.html", error="Password must be at least 6 characters", email=email, dev_code=dev_code)
        
        if new_password != confirm_password:
            return render_template("reset_password.html", error="Passwords do not match", email=email, dev_code=dev_code)
        
        # Reset password
        success, error = reset_user_password(email, code, new_password)
        
        if error:
            return render_template("reset_password.html", error=error, email=email, dev_code=dev_code)
        
        # Clear session
        session.pop('reset_email', None)
        session.pop('dev_mode_reset_code', None)
        
        # Redirect to login with success message
        return render_template("login.html", success="Password reset successful! Please sign in with your new password.")
    
    return render_template("reset_password.html", email=email, dev_code=dev_code)

@app.route("/resend-reset-code", methods=["POST"])
def resend_reset_code_route():
    email = session.get('reset_email')
    
    if not email:
        return jsonify({"success": False, "error": "No reset request found"})
    
    # Get user for username
    user = get_user_by_email(email)
    if not user:
        return jsonify({"success": False, "error": "User not found"})
    
    # Generate new code
    new_code, error = resend_reset_code(email)
    
    if error:
        return jsonify({"success": False, "error": error})
    
    # Send new reset email
    success, error_msg, fallback_code = send_password_reset_email(email, new_code, user.username)
    
    # Show code if available (dev mode or email failed)
    session.pop('dev_mode_reset_code', None)
    if fallback_code:
        session['dev_mode_reset_code'] = fallback_code
    
    if success:
        return jsonify({"success": True, "message": "Reset code sent successfully", "dev_code": fallback_code})
    else:
        # Still return success with code if email failed but code was generated
        return jsonify({"success": True, "message": "Email failed but code is shown below", "dev_code": fallback_code})

# ---------------- REGISTER (alias for signup) ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    return redirect(url_for("signup"))

@app.route("/logout")
def logout():
    username = session.get("user")
    if username:
        # Stop any running bots for this user
        stop_bot(username)
    # Clear MT5 session and user tracking
    clear_mt5_session()
    session.clear()
    return redirect(url_for("login"))

# ---------------- DASHBOARD ----------------
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    username = session["user"]
    status = "Running" if bot_status(username) else "Stopped"
    return render_template("index.html", status=status, username=username)

# ---------------- BOT CONTROL ----------------
@app.route("/start_bot")
def start():
    username = session.get("user")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    msg = start_bot(username)
    status = bot_status(username)
    return jsonify({"status": "running" if status.get("running") else "stopped", "message": msg, "running": status.get("running", False)})

@app.route("/stop_bot")
def stop():
    username = session.get("user")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    msg = stop_bot(username)
    status = bot_status(username)
    return jsonify({"status": "stopped" if not status.get("running") else "running", "message": msg, "running": status.get("running", False)})

@app.route("/status")
def status():
    username = session.get("user")
    if not username:
        return jsonify({"running": False})
    status_info = bot_status(username)
    return jsonify({"running": status_info.get("running", False), "symbols": status_info.get("symbols", []), "count": status_info.get("count", 0)})

# ---------------- ACCOUNT & POSITIONS ----------------
@app.route("/account")
def account_info():
    username = session.get("user")
    if not username:
        return jsonify({})
    try:
        return jsonify(get_account_info(username))
    except Exception as e:
        print(f"⚠️ Error in /account: {str(e)[:100]}")
        return jsonify({"error": "Connection issue, retrying..."}), 503

@app.route("/positions")
def positions_info():
    username = session.get("user")
    if not username:
        return jsonify([])
    try:
        return jsonify(get_positions(username))
    except Exception as e:
        print(f"⚠️ Error in /positions: {str(e)[:100]}")
        return jsonify([]), 503

@app.route("/api/trade_history")
def trade_history():
    """Get trade history for the logged-in user"""
    username = session.get("user")
    if not username:
        return jsonify([])
    
    days = request.args.get('days', 30, type=int)
    trades = get_trade_history(username, days)
    return jsonify(trades)

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run backtest on strategy"""
    username = session.get("user")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        from backtest import run_backtest, save_backtest_report
        
        data = request.get_json() or {}
        symbol = data.get('symbol', 'XAUUSD')
        days = data.get('days', 180)
        
        # Validate inputs
        days = min(365, max(30, int(days)))
        
        print(f"🔬 Running backtest for {symbol} ({days} days)...")
        result = run_backtest(symbol, days)
        
        if 'error' not in result:
            # Save report
            save_backtest_report({'symbol': symbol, 'result': result})
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ Backtest error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/backtest/full", methods=["POST"])
def api_backtest_full():
    """Run full backtest on all symbols"""
    username = session.get("user")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        from backtest import run_full_backtest, save_backtest_report
        
        data = request.get_json() or {}
        days = data.get('days', 180)
        symbols = data.get('symbols', None)  # None = use defaults
        
        # Validate inputs
        days = min(365, max(30, int(days)))
        
        print(f"🔬 Running full backtest ({days} days)...")
        results = run_full_backtest(symbols=symbols, days=days)
        
        # Save report
        save_backtest_report(results)
        
        return jsonify(results)
    except Exception as e:
        print(f"❌ Backtest error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ---------------- PROFILE ----------------
@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    user = get_user_by_username(session["user"])
    if not user:
        session.clear()
        return redirect(url_for("login"))
    
    return render_template("profile.html", user=user)

@app.route("/api/change_password", methods=["POST"])
def api_change_password():
    """Change the logged-in user's password"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    
    username = session["user"]
    data = request.get_json() or {}
    
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")
    
    if not current_password or not new_password or not confirm_password:
        return jsonify({"success": False, "error": "All fields are required"})
    
    if new_password != confirm_password:
        return jsonify({"success": False, "error": "New passwords do not match"})
    
    success, error = change_user_password(username, current_password, new_password)
    
    if success:
        return jsonify({"success": True, "message": "Password changed successfully"})
    else:
        return jsonify({"success": False, "error": error})


@app.route("/api/request_password_change_otp", methods=["POST"])
def api_request_password_change_otp():
    """Request OTP for password change - verifies current password first"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    
    username = session["user"]
    data = request.get_json() or {}
    current_password = data.get("current_password", "")
    
    if not current_password:
        return jsonify({"success": False, "error": "Current password is required"})
    
    # Get user to verify current password
    user = get_user_by_username(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"})
    
    # Verify current password
    if not user.check_password(current_password):
        return jsonify({"success": False, "error": "Current password is incorrect"})
    
    # Generate OTP
    code = generate_verification_code()
    store_password_change_otp(username, code)
    
    # Send OTP email
    success, error_msg, fallback_code = send_password_change_otp_email(user.email, code, username)
    
    response_data = {"success": True, "message": "OTP sent to your email"}
    if fallback_code:
        response_data["dev_code"] = fallback_code
    
    return jsonify(response_data)


@app.route("/api/verify_password_change_otp", methods=["POST"])
def api_verify_password_change_otp():
    """Verify OTP and change password"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    
    username = session["user"]
    data = request.get_json() or {}
    
    code = data.get("code", "").strip()
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")
    
    if not code:
        return jsonify({"success": False, "error": "OTP code is required"})
    
    if len(code) != 6 or not code.isdigit():
        return jsonify({"success": False, "error": "Invalid OTP format"})
    
    if not new_password:
        return jsonify({"success": False, "error": "New password is required"})
    
    if len(new_password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters"})
    
    if new_password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match"})
    
    # Complete password change with OTP verification
    success, error = complete_password_change_with_otp(username, code, new_password)
    
    if success:
        return jsonify({"success": True, "message": "Password changed successfully"})
    else:
        return jsonify({"success": False, "error": error})


@app.route("/api/resend_password_change_otp", methods=["POST"])
def api_resend_password_change_otp():
    """Resend password change OTP"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    
    username = session["user"]
    user = get_user_by_username(username)
    
    if not user:
        return jsonify({"success": False, "error": "User not found"})
    
    # Generate new code
    new_code, error = resend_password_change_otp(username)
    
    if error:
        return jsonify({"success": False, "error": error})
    
    # Send new OTP email
    success, error_msg, fallback_code = send_password_change_otp_email(user.email, new_code, username)
    
    response_data = {"success": True, "message": "New OTP sent to your email"}
    if fallback_code:
        response_data["dev_code"] = fallback_code
    
    return jsonify(response_data)


def send_password_change_otp_email(email, code, username):
    """Send password change OTP email to user"""
    # Development mode - skip email and show code directly
    if Config.EMAIL_DEV_MODE:
        print(f"📧 [DEV MODE] Password change OTP for {email}: {code}")
        return True, None, code  # Return code for dev mode display
    
    # Check if email is configured
    if not Config.MAIL_USERNAME or not Config.MAIL_PASSWORD:
        print(f"⚠️ Email not configured. Password change OTP for {email}: {code}")
        return True, None, code  # Return code when email not configured
    
    try:
        msg = Message(
            subject='🔐 TradingBot - Password Change Verification',
            recipients=[email]
        )
        msg.html = f'''
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #0a1628 0%, #1a2332 100%); padding: 30px; border-radius: 12px 12px 0 0; text-align: center;">
                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">🤖 TradingBot</h1>
            </div>
            <div style="background: #ffffff; padding: 40px 30px; border: 1px solid #e2e8f0; border-top: none;">
                <h2 style="color: #0a1628; margin-top: 0;">Password Change Request 🔑</h2>
                <p style="color: #5b6b8b; font-size: 16px; line-height: 1.6;">
                    Hi {username}, you've requested to change your password. Use the code below to verify this action:
                </p>
                <div style="background: #f7fafc; border: 2px dashed #3b82f6; border-radius: 12px; padding: 25px; text-align: center; margin: 30px 0;">
                    <span style="font-size: 36px; font-weight: 700; letter-spacing: 8px; color: #3b82f6;">{code}</span>
                </div>
                <p style="color: #5b6b8b; font-size: 14px;">
                    ⏰ This code will expire in <strong>15 minutes</strong>.
                </p>
                <p style="color: #5b6b8b; font-size: 14px;">
                    If you didn't request this password change, please secure your account immediately and change your password.
                </p>
            </div>
            <div style="background: #f7fafc; padding: 20px 30px; border-radius: 0 0 12px 12px; text-align: center; border: 1px solid #e2e8f0; border-top: none;">
                <p style="color: #718096; font-size: 12px; margin: 0;">
                    © 2026 TradingBot. Automated Trading Made Simple.
                </p>
            </div>
        </div>
        '''
        msg.body = f'''
TradingBot - Password Change Verification

Hi {username},

You've requested to change your password.

Your verification code is: {code}

This code will expire in 15 minutes.

If you didn't request this, please secure your account immediately.
        '''
        mail.send(msg)
        print(f"✅ Password change OTP email sent to {email}")
        return True, None, None
    except Exception as e:
        print(f"❌ Failed to send password change OTP email: {e}")
        return False, str(e), code


@app.route("/api/user_stats")
def user_stats():
    """Get user statistics"""
    if "user" not in session:
        return jsonify({"total_trades": 0})
    
    username = session["user"]
    from models import get_trading_logs
    
    # Get trade logs (type='trade') for this user
    all_logs = get_trading_logs(username, limit=1000)
    trade_logs = [log for log in all_logs if log.get('type') == 'trade']
    
    return jsonify({
        "total_trades": len(trade_logs)
    })

# ---------------- MT5 CONNECTION ----------------
@app.route("/connect_mt5", methods=["GET", "POST"])
def connect_mt5():
    if "user" not in session:
        return redirect(url_for("login"))
    
    username = session["user"]
    user = get_user_by_username(username)
    
    if request.method == "POST":
        mt5_login = request.form.get("mt5_login", "").strip()
        mt5_password = request.form.get("mt5_password", "")
        mt5_server = request.form.get("mt5_server", "").strip()
        
        if not mt5_login or not mt5_password or not mt5_server:
            return render_template("connect_mt5.html", user=user, error="Please fill in all fields")
        
        try:
            mt5_login = int(mt5_login)
        except ValueError:
            return render_template("connect_mt5.html", user=user, error="MT5 Login must be a number")
        
        # Test the connection
        success, message = test_mt5_connection(mt5_login, mt5_password, mt5_server)
        
        if success:
            # Save credentials to database
            update_mt5_credentials(username, mt5_login, mt5_password, mt5_server)
            session["mt5_connected"] = True
            return redirect(url_for("index"))
        else:
            return render_template("connect_mt5.html", user=user, error=f"Connection failed: {message}")
    
    return render_template("connect_mt5.html", user=user)

@app.route("/disconnect_mt5")
def disconnect_mt5_route():
    if "user" not in session:
        return redirect(url_for("login"))
    
    username = session["user"]
    # Stop bot first if running
    stop_bot(username)
    disconnect_mt5(username)
    session["mt5_connected"] = False
    return redirect(url_for("profile"))

@app.route("/api/mt5_status")
def mt5_status():
    """Check if user has MT5 connected"""
    if "user" not in session:
        return jsonify({"connected": False})
    
    username = session["user"]
    creds = get_user_mt5_credentials(username)
    return jsonify({"connected": creds is not None})

# ---------------- TRADING LOGS ----------------
@app.route("/logs")
def logs():
    """View trading logs"""
    if "user" not in session:
        return redirect(url_for("login"))
    
    username = session["user"]
    trading_logs = get_trading_logs(username, limit=100)
    
    # Format logs for display
    formatted_logs = []
    for log in trading_logs:
        formatted_logs.append({
            'type': log.get('type', 'info'),
            'message': log.get('message', ''),
            'details': log.get('details', {}),
            'time': log.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if log.get('created_at') else ''
        })
    
    return render_template("logs.html", logs=formatted_logs)

# ---------------- LIVE CHARTS ----------------
@app.route("/charts")
def charts():
    """Live Market Charts page"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("charts.html", username=session["user"])

@app.route("/api/chart_data")
def api_chart_data():
    """Get candlestick data for chart"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    symbol = request.args.get('symbol', 'XAUUSD')
    timeframe = request.args.get('timeframe', 'M15')
    bars = int(request.args.get('bars', 200))
    
    username = session["user"]
    
    try:
        import MetaTrader5 as mt5
        from botlogic import get_broker_symbol
        
        # Always initialize MT5 first
        if not mt5.initialize():
            return jsonify({"success": False, "error": "MT5 not initialized - please open MetaTrader 5"})
        
        # Try to login with user's credentials if available
        creds = get_user_mt5_credentials(username)
        if creds:
            if not mt5.login(creds['login'], password=creds['password'], server=creds['server']):
                # Login failed but MT5 is initialized - try current session
                pass  # Continue with current MT5 session
        
        # Convert symbol to broker format (handles suffixes automatically)
        broker_symbol = get_broker_symbol(symbol)
        
        # Map timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
        
        # Try with broker symbol first, then original symbol
        rates = mt5.copy_rates_from_pos(broker_symbol, mt5_tf, 0, bars)
        
        if (rates is None or len(rates) == 0) and broker_symbol != symbol:
            # Try original symbol as fallback
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            return jsonify({"success": False, "error": f"No data for {symbol} (tried {broker_symbol})"})
        
        # Convert to chart format
        chart_data = []
        for rate in rates:
            chart_data.append({
                'time': int(rate['time']),
                'open': float(rate['open']),
                'high': float(rate['high']),
                'low': float(rate['low']),
                'close': float(rate['close'])
            })
        
        # Get spread
        symbol_info = mt5.symbol_info(broker_symbol) or mt5.symbol_info(symbol)
        spread = symbol_info.spread if symbol_info else 0
        
        return jsonify({
            "success": True,
            "data": chart_data,
            "spread": spread,
            "symbol": symbol,
            "broker_symbol": broker_symbol,
            "timeframe": timeframe
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ---------------- AI TOOLS ----------------
@app.route("/ai_tools")
def ai_tools():
    """AI Trading Tools page"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("ai_tools.html", username=session["user"])

# ---------------- TRADE SIGNALS ----------------
@app.route("/signals")
def signals():
    """Trade Signals page"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("signals.html", username=session["user"])

# ---------------- NEWS & EVENTS ----------------
@app.route("/news")
def news():
    """News & Events page"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("news.html", username=session["user"])

@app.route("/api/logs")
def api_logs():
    """API endpoint for trading logs"""
    if "user" not in session:
        return jsonify([])
    
    username = session["user"]
    trading_logs = get_trading_logs(username, limit=50)
    
    formatted_logs = []
    for log in trading_logs:
        formatted_logs.append({
            'type': log.get('type', 'info'),
            'message': log.get('message', ''),
            'details': log.get('details', {}),
            'time': log.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if log.get('created_at') else ''
        })
    
    return jsonify(formatted_logs)

@app.route("/clear_logs")
def clear_logs():
    """Clear user's trading logs"""
    if "user" not in session:
        return redirect(url_for("login"))
    
    username = session["user"]
    clear_trading_logs(username)
    return redirect(url_for("logs"))

# ---------------- AI INSIGHTS ENDPOINTS ----------------
@app.route("/api/ai_insights")
def api_ai_insights():
    """Get AI trading insights and recommendations"""
    if "user" not in session:
        return jsonify({"has_insights": False, "error": "Not logged in"})
    
    username = session["user"]
    insights = get_ai_insights(username)
    return jsonify(insights)

@app.route("/api/ai_sentiment")
def api_ai_sentiment():
    """Get AI market sentiment analysis"""
    if "user" not in session:
        return jsonify({"sentiment": "NEUTRAL", "confidence": 0})
    
    symbol = request.args.get("symbol", "XAUUSD")
    sentiment = ai_get_market_sentiment(symbol)
    return jsonify(sentiment)

@app.route("/api/ai_params")
def api_ai_params():
    """Get AI-optimized trading parameters"""
    if "user" not in session:
        return jsonify({})
    
    username = session["user"]
    params = get_ai_optimized_params(username)
    return jsonify(params)

# ---------------- NEWS & MARKET SCRAPING ENDPOINTS ----------------
@app.route("/api/news")
def api_news():
    """Get news analysis for a symbol"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbol = request.args.get("symbol", "XAUUSD")
    news_data = get_news_analysis(symbol, username)
    return jsonify(news_data)

@app.route("/api/calendar")
def api_calendar():
    """Get upcoming high-impact economic events"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    calendar_data = get_economic_calendar()
    return jsonify(calendar_data)

@app.route("/api/news/all")
def api_news_all():
    """Get news for all trading symbols"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbols_to_check = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]  # Key symbols (standard names)
    all_news = {}
    
    for symbol in symbols_to_check:
        all_news[symbol] = get_news_analysis(symbol, username)
    
    return jsonify({
        "symbols": all_news,
        "calendar": get_economic_calendar()
    })

# ---------------- AI LIVE SENTIMENT ENDPOINTS ----------------
@app.route("/api/sentiment/live")
def api_live_sentiment():
    """Get LIVE AI market sentiment for a symbol - refreshes with real data"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    symbol = request.args.get("symbol", "XAUUSD")
    force_refresh = request.args.get("refresh", "false").lower() == "true"
    
    sentiment = get_live_market_sentiment(symbol, force_refresh)
    return jsonify({
        "symbol": symbol,
        "sentiment": sentiment.get("sentiment", "NEUTRAL"),
        "confidence": sentiment.get("confidence", 0.5),
        "strength": sentiment.get("strength", "WEAK"),
        "trading_bias": sentiment.get("trading_bias", "WAIT"),
        "bias_reason": sentiment.get("bias_reason", ""),
        "short_term_outlook": sentiment.get("short_term_outlook", ""),
        "key_factors": sentiment.get("key_factors", []),
        "support_level": sentiment.get("support_level"),
        "resistance_level": sentiment.get("resistance_level"),
        "risk_events": sentiment.get("risk_events", [])
    })

@app.route("/api/sentiment/all")
def api_all_sentiments():
    """Get LIVE AI sentiment for all active symbols"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    user_symbols = get_user_symbols(username)
    
    # Get sentiment for user's symbols (or default to main ones)
    symbols = user_symbols if user_symbols else ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
    
    sentiments = get_all_live_sentiments(symbols)
    return jsonify({
        "sentiments": sentiments,
        "updated_at": __import__("datetime").datetime.now().isoformat()
    })

@app.route("/api/ai/entry")
def api_ai_entry():
    """Get AI entry point analysis for a symbol"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbol = request.args.get("symbol", "XAUUSD")

    entry = ai_find_entry_points(symbol, username)
    return jsonify({
        "symbol": symbol,
        "has_entry": entry.get("has_entry", False),
        "direction": entry.get("direction"),
        "confidence": entry.get("confidence"),
        "entry_price": entry.get("entry_price"),
        "stop_loss": entry.get("stop_loss"),
        "take_profit": entry.get("take_profit"),
        "risk_reward": entry.get("risk_reward"),
        "quality_score": entry.get("quality_score"),
        "confluences": entry.get("confluences", []),
        "reason": entry.get("reason"),
        "urgency": entry.get("urgency")
    })

@app.route("/api/ai/news-trade", methods=["POST"])
def api_ai_news_trade():
    """Execute an AI news-based trade"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    data = request.get_json() or {}
    symbol = data.get("symbol", "XAUUSD")
    
    # Handle null lot_size - let backend use intelligent lot sizing
    lot_size_raw = data.get("lot_size")
    lot_size = float(lot_size_raw) if lot_size_raw is not None else None

    result = ai_execute_news_trade(symbol, username, lot_size)
    return jsonify(result)

@app.route("/api/ai/entry-trade", methods=["POST"])
def api_ai_entry_trade():
    """Execute an AI entry-point trade"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    data = request.get_json() or {}
    symbol = data.get("symbol", "XAUUSD")
    
    # Handle null lot_size - let backend use intelligent lot sizing
    lot_size_raw = data.get("lot_size")
    lot_size = float(lot_size_raw) if lot_size_raw is not None else None
    
    result = ai_execute_entry_trade(symbol, username, lot_size)
    return jsonify(result)

# ---------------- BACKGROUND AUTO-SCAN API ENDPOINTS ----------------
@app.route("/api/ai/autoscan/start", methods=["POST"])
def api_autoscan_start():
    """Start background auto-scan for entry points"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    success = start_background_scan(username)
    return jsonify({
        "success": success,
        "message": "Background auto-scan started" if success else "Failed to start",
        "status": get_scan_status(username)
    })

@app.route("/api/ai/autoscan/stop", methods=["POST"])
def api_autoscan_stop():
    """Stop background auto-scan"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    success = stop_background_scan(username)
    return jsonify({
        "success": success,
        "message": "Background auto-scan stopped" if success else "Failed to stop",
        "status": get_scan_status(username)
    })

@app.route("/api/ai/autoscan/status")
def api_autoscan_status():
    """Get current auto-scan status"""
    if "user" not in session:
        return jsonify({"scanning": False, "error": "Not logged in"})
    
    username = session["user"]
    status = get_scan_status(username)
    return jsonify(status)

# ---------------- SIGNAL AUTO-EXECUTE ENDPOINTS ----------------
@app.route("/api/signals/autoexecute/start", methods=["POST"])
def api_signal_autoexecute_start():
    """Start background signal auto-execute"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    success = start_signal_auto_execute(username)
    return jsonify({
        "success": success,
        "message": "Signal auto-execute started" if success else "Failed to start",
        "status": get_signal_auto_execute_status(username)
    })

@app.route("/api/signals/autoexecute/stop", methods=["POST"])
def api_signal_autoexecute_stop():
    """Stop background signal auto-execute"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    success = stop_signal_auto_execute(username)
    return jsonify({
        "success": success,
        "message": "Signal auto-execute stopped" if success else "Failed to stop",
        "status": get_signal_auto_execute_status(username)
    })

@app.route("/api/signals/autoexecute/status")
def api_signal_autoexecute_status():
    """Get current signal auto-execute status"""
    if "user" not in session:
        return jsonify({"running": False, "error": "Not logged in"})
    
    username = session["user"]
    status = get_signal_auto_execute_status(username)
    return jsonify(status)

# ---------------- AI SESSION/TIMING ENDPOINTS ----------------
@app.route("/api/ai/session")
def api_ai_session():
    """Get AI analysis of current trading session quality"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbol = request.args.get("symbol", "XAUUSD")

    analysis = get_optimal_trading_time(symbol, username)
    return jsonify({
        "symbol": symbol,
        "should_trade_now": analysis.get("should_trade_now", True),
        "confidence": analysis.get("confidence", 0.5),
        "session_quality": analysis.get("current_session_quality", "UNKNOWN"),
        "recommendation": analysis.get("trading_recommendation", "TRADE_NOW"),
        "reason": analysis.get("reason", ""),
        "better_time_today": analysis.get("better_time_today"),
        "time_until_optimal": analysis.get("time_until_optimal"),
        "risk_level": analysis.get("risk_level", "MEDIUM"),
        "expected_volatility": analysis.get("expected_volatility", "NORMAL"),
        "best_hours": analysis.get("best_hours_today", []),
        "avoid_hours": analysis.get("avoid_hours_today", []),
        "special_notes": analysis.get("special_notes", [])
    })

@app.route("/api/ai/best-hours")
def api_ai_best_hours():
    """Get the best trading hours for today"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbol = request.args.get("symbol", "XAUUSD")

    result = get_best_trading_hours_today(symbol, username)
    return jsonify(result)

@app.route("/api/ai/should-trade")
def api_ai_should_trade():
    """Quick check if AI recommends trading now"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbol = request.args.get("symbol", "XAUUSD")

    should_trade, reason, quality = should_trade_this_session(symbol, username)
    return jsonify({
        "should_trade": should_trade,
        "reason": reason,
        "session_quality": quality
    })

# ---------------- CHART DATA ENDPOINTS ----------------
@app.route("/api/chart")
def api_chart():
    """Get candlestick/OHLC data for a symbol"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbol = request.args.get("symbol", "XAUUSD")
    timeframe = request.args.get("timeframe", "M5")
    bars = int(request.args.get("bars", 200))
    
    chart_data = get_chart_data(symbol, timeframe, bars, username)
    return jsonify(chart_data)

@app.route("/api/charts")
def api_charts():
    """Get chart data for multiple symbols"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbols = request.args.get("symbols", "XAUUSD,EURUSD,GBPUSD,BTCUSD").split(",")
    timeframe = request.args.get("timeframe", "M5")
    bars = int(request.args.get("bars", 100))
    
    charts_data = get_multi_chart_data(symbols, timeframe, bars, username)
    return jsonify(charts_data)

# ---------------- EXPLICIT TRADE SIGNALS ENDPOINT ----------------
@app.route("/api/signals")
def api_signals():
    """Get explicit trade signals for all symbols"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    symbols = get_user_symbols(username)
    
    signals = []
    for symbol in symbols[:6]:  # Max 6 symbols
        signal = generate_explicit_trade_signal(symbol, username)
        if signal:
            signals.append(signal)
    
    # Get current session
    session_name, session_data = get_current_session()
    
    return jsonify({
        "signals": signals,
        "session": session_name,
        "session_volatility": session_data.get('volatility', 'UNKNOWN') if session_data else 'OFF',
        "time": __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    })

@app.route("/api/signal/<symbol>")
def api_single_signal(symbol):
    """Get explicit trade signal for a specific symbol"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    signal = generate_explicit_trade_signal(symbol, username)
    
    if signal:
        return jsonify(signal)
    else:
        return jsonify({"error": "Could not generate signal", "symbol": symbol})

@app.route("/api/signal/execute", methods=["POST"])
def api_execute_signal():
    """Execute an explicit trade signal"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    data = request.get_json() or {}
    symbol = data.get("symbol", "XAUUSD")
    
    # Handle optional lot_size
    lot_size_raw = data.get("lot_size")
    lot_size = float(lot_size_raw) if lot_size_raw is not None else None
    
    result = execute_explicit_signal(symbol, username, lot_size)
    return jsonify(result)

# ---------------- LOSS PROTECTION ENDPOINT ----------------
@app.route("/api/loss-protection")
def api_loss_protection():
    """Get current loss protection status"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    status = get_loss_protection_status(username)
    return jsonify(status)

@app.route("/api/ai/loss-insights")
def api_ai_loss_insights():
    """Get AI loss pattern learning insights"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"})
    
    username = session["user"]
    
    from botlogic import (AI_LOSS_PATTERN_DATA, USER_STREAK_DATA, RECOVERY_MODE_ACTIVE,
                          get_recovery_status)
    
    # Get loss patterns
    loss_patterns = AI_LOSS_PATTERN_DATA.get(username, [])
    
    # Analyze patterns by symbol
    symbol_losses = {}
    for pattern in loss_patterns:
        sym = pattern.get('symbol', 'Unknown')
        if sym not in symbol_losses:
            symbol_losses[sym] = {'count': 0, 'total_loss': 0, 'patterns': []}
        symbol_losses[sym]['count'] += 1
        symbol_losses[sym]['total_loss'] += pattern.get('loss_amount', 0)
        symbol_losses[sym]['patterns'].append({
            'loss': pattern.get('loss_amount', 0),
            'time': pattern.get('time', 0),
            'context': pattern.get('context', {})
        })
    
    # Get streak data
    streak_data = USER_STREAK_DATA.get(username, {
        'current_streak': 0,
        'total_wins': 0,
        'total_losses': 0
    })
    
    # Get recovery status
    recovery = get_recovery_status(username)
    
    return jsonify({
        "total_loss_patterns": len(loss_patterns),
        "symbol_breakdown": symbol_losses,
        "streak": streak_data,
        "recovery_mode": recovery,
        "insights": generate_loss_insights(symbol_losses, streak_data)
    })

def generate_loss_insights(symbol_losses, streak_data):
    """Generate actionable insights from loss data"""
    insights = []
    
    # Worst performing symbols
    if symbol_losses:
        worst = max(symbol_losses.items(), key=lambda x: x[1]['total_loss'])
        insights.append(f"⚠️ Worst performer: {worst[0]} (${worst[1]['total_loss']:.2f} total loss, {worst[1]['count']} trades)")
    
    # Win rate
    total = streak_data.get('total_wins', 0) + streak_data.get('total_losses', 0)
    if total > 0:
        win_rate = streak_data.get('total_wins', 0) / total * 100
        insights.append(f"📊 Win rate: {win_rate:.1f}% ({streak_data.get('total_wins', 0)}W / {streak_data.get('total_losses', 0)}L)")
    
    # Current streak
    streak = streak_data.get('current_streak', 0)
    if streak > 0:
        insights.append(f"🔥 Current win streak: {streak}")
    elif streak < 0:
        insights.append(f"❄️ Current loss streak: {abs(streak)}")
    
    return insights

@app.route("/api/loss-protection/toggle", methods=["POST"])
def api_toggle_loss_protection():
    """Toggle loss protection on/off for user"""
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    username = session["user"]
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    
    new_status = set_loss_protection_enabled(username, enabled)
    return jsonify({"success": True, "enabled": new_status})

# ---------------- SYMBOL MANAGEMENT ENDPOINTS ----------------
@app.route("/api/symbols")
def api_get_symbols():
    """Get user's selected trading symbols"""
    if "user" not in session:
        return jsonify({"symbols": [], "error": "Not logged in"})
    
    username = session["user"]
    symbols = get_user_symbols(username)
    return jsonify({
        "symbols": symbols,
        "default_symbols": DEFAULT_SYMBOLS,
        "symbol_settings": SYMBOL_SETTINGS
    })

@app.route("/api/symbols/set", methods=["POST"])
def api_set_symbols():
    """Set user's trading symbols"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    data = request.get_json()
    symbols = data.get("symbols", [])
    
    if not symbols:
        return jsonify({"success": False, "error": "No symbols provided"})
    
    valid_symbols = set_user_symbols(username, symbols)
    return jsonify({
        "success": True,
        "symbols": valid_symbols,
        "message": f"Trading {len(valid_symbols)} symbols"
    })

@app.route("/api/symbols/add", methods=["POST"])
def api_add_symbol():
    """Add a symbol to user's trading list"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    
    if not symbol:
        return jsonify({"success": False, "error": "No symbol provided"})
    
    success, message = add_user_symbol(username, symbol)
    return jsonify({
        "success": success,
        "message": message,
        "symbols": get_user_symbols(username)
    })

@app.route("/api/symbols/remove", methods=["POST"])
def api_remove_symbol():
    """Remove a symbol from user's trading list"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in"})
    
    username = session["user"]
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    
    if not symbol:
        return jsonify({"success": False, "error": "No symbol provided"})
    
    success, message = remove_user_symbol(username, symbol)
    return jsonify({
        "success": success,
        "message": message,
        "symbols": get_user_symbols(username)
    })

@app.route("/api/symbols/available")
def api_available_symbols():
    """Get all available symbols from MT5"""
    if "user" not in session:
        return jsonify({"symbols": []})
    
    symbols = get_available_symbols()
    return jsonify({"symbols": symbols[:100]})  # Limit to 100 symbols

@app.route("/api/bot_status")
def api_bot_status():
    """Get detailed bot status including symbols"""
    if "user" not in session:
        return jsonify({"running": False, "symbols": []})
    
    username = session["user"]
    status = bot_status(username)
    return jsonify(status)

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("=" * 50, flush=True)
    print("🌐 Starting Flask server...", flush=True)
    
    # Check if running on VPS (production) or local (development)
    import os
    is_production = os.environ.get('PRODUCTION', 'false').lower() == 'true'
    
    if is_production:
        # VPS/Production mode - accessible from internet
        print("📍 PRODUCTION MODE - URL: http://0.0.0.0:5000", flush=True)
        print("🌍 Accessible from: http://YOUR_DOMAIN:5000", flush=True)
        print("=" * 50, flush=True)
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        # Local development mode
        print("📍 DEVELOPMENT MODE - URL: http://127.0.0.1:5000", flush=True)
        print("=" * 50, flush=True)
        app.run(debug=True, use_reloader=False)
