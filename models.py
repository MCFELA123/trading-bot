from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, NetworkTimeout, AutoReconnect
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import os
import random
import string
import time
import functools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- MONGODB RETRY DECORATOR ----------------
def mongodb_retry(max_retries=2, delay=1):
    """Decorator to retry MongoDB operations on connection failures"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ServerSelectionTimeoutError, ConnectionFailure, NetworkTimeout, AutoReconnect) as e:
                    last_error = e
                    if attempt < max_retries:
                        print(f"⚠️ MongoDB retry {attempt + 1}/{max_retries} for {func.__name__}: {str(e)[:80]}")
                        time.sleep(delay)
                        # Try to reconnect
                        try:
                            reconnect_mongodb()
                        except:
                            pass
                    else:
                        print(f"❌ MongoDB failed after {max_retries + 1} attempts: {func.__name__}")
            # Return None or empty result instead of crashing
            return None
        return wrapper
    return decorator

def reconnect_mongodb():
    """Force reconnect to MongoDB"""
    global client, db
    try:
        if client:
            client.close()
    except:
        pass
    client = None
    db = None
    get_db()  # Reinitialize

# ---------------- MONGODB CONNECTION ----------------
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'tradingbot')

# Connection settings for unstable networks
MONGO_CONNECT_TIMEOUT_MS = 10000  # 10 seconds
MONGO_SERVER_SELECTION_TIMEOUT_MS = 10000
MONGO_MAX_RETRIES = 3
MONGO_RETRY_DELAY = 2  # seconds

client = None
db = None

def get_db():
    """Get database connection (lazy initialization)"""
    global client, db
    try:
        if client is None:
            client = MongoClient(
                MONGODB_URI,
                connectTimeoutMS=MONGO_CONNECT_TIMEOUT_MS,
                serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
                retryWrites=True,
                retryReads=True
            )
            db = client[DB_NAME]
        return db
    except Exception as e:
        print(f"⚠️ MongoDB connection error: {str(e)[:80]}")
        return None

def init_db(app=None):
    """Initialize database connection with retry logic"""
    global client, db
    import time
    
    for attempt in range(1, MONGO_MAX_RETRIES + 1):
        try:
            print(f"🔄 MongoDB connection attempt {attempt}/{MONGO_MAX_RETRIES}...")
            client = MongoClient(
                MONGODB_URI,
                connectTimeoutMS=MONGO_CONNECT_TIMEOUT_MS,
                serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
                retryWrites=True,
                retryReads=True
            )
            db = client[DB_NAME]
            # Test connection
            client.admin.command('ping')
            print("✅ Connected to MongoDB successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Connection attempt {attempt} failed: {str(e)[:100]}")
            if attempt < MONGO_MAX_RETRIES:
                print(f"⏳ Retrying in {MONGO_RETRY_DELAY} seconds...")
                time.sleep(MONGO_RETRY_DELAY)
            else:
                print(f"❌ MongoDB connection failed after {MONGO_MAX_RETRIES} attempts")
                print("💡 Check your internet connection or try switching to a different network/DNS")
                return False
    return False

# ---------------- USER CLASS ----------------
class User:
    """User class for Flask-Login compatibility"""
    def __init__(self, user_data):
        self._id = user_data.get('_id')
        self.username = user_data.get('username')
        self.email = user_data.get('email')
        self.password_hash = user_data.get('password_hash')
        self.role = user_data.get('role', 'user')
        self.created_at = user_data.get('created_at', datetime.utcnow())
        # MT5 credentials
        self.mt5_login = user_data.get('mt5_login')
        self.mt5_password = user_data.get('mt5_password')
        self.mt5_server = user_data.get('mt5_server')
        self.mt5_connected = user_data.get('mt5_connected', False)
    
    @property
    def id(self):
        return str(self._id)
    
    @property
    def is_active(self):
        return True
    
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self._id)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def has_mt5_credentials(self):
        return bool(self.mt5_login and self.mt5_password and self.mt5_server)

# ---------------- USER HELPERS ----------------
def add_user(username, password, email=None, role="user"):
    """Add a new user to the database"""
    database = get_db()
    
    if email is None:
        email = f"{username}@example.com"
    
    # Check if username already exists
    if database.users.find_one({'username': username}):
        return None, "Username already exists"
    
    # Check if email already exists
    if database.users.find_one({'email': email}):
        return None, "Email already exists"
    
    user_data = {
        'username': username,
        'email': email,
        'password_hash': generate_password_hash(password),
        'role': role,
        'created_at': datetime.utcnow()
    }
    
    result = database.users.insert_one(user_data)
    user_data['_id'] = result.inserted_id
    
    return User(user_data), None

def verify_user(username_or_email, password):
    """Verify a user by username or email and password"""
    database = get_db()
    
    # Check by username first
    user_data = database.users.find_one({'username': username_or_email})
    
    # If not found by username, try email
    if not user_data:
        user_data = database.users.find_one({'email': username_or_email.lower()})
    
    if user_data:
        user = User(user_data)
        if user.check_password(password):
            return user
    return None

def get_user_by_id(user_id):
    """Get user by ID"""
    database = get_db()
    try:
        user_data = database.users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
    except:
        pass
    return None

def get_user_by_username(username):
    """Get user by username"""
    database = get_db()
    user_data = database.users.find_one({'username': username})
    if user_data:
        return User(user_data)
    return None

def get_user_by_email(email):
    """Get user by email"""
    database = get_db()
    user_data = database.users.find_one({'email': email})
    if user_data:
        return User(user_data)
    return None

# ---------------- TRADE HELPERS ----------------
def add_trade(user_id, symbol, order_type, lot, price, sl, tp):
    """Add a trade record for a user"""
    database = get_db()
    trade_data = {
        'user_id': user_id,
        'symbol': symbol,
        'order_type': order_type,
        'lot': lot,
        'price': price,
        'sl': sl,
        'tp': tp,
        'created_at': datetime.utcnow()
    }
    result = database.trades.insert_one(trade_data)
    trade_data['_id'] = result.inserted_id
    return trade_data

def get_user_trades(user_id):
    """Get all trades for a specific user"""
    database = get_db()
    trades = list(database.trades.find({'user_id': user_id}).sort('created_at', -1))
    return trades

# ---------------- MT5 CREDENTIALS ----------------
def update_mt5_credentials(username, mt5_login, mt5_password, mt5_server):
    """Update MT5 credentials for a user"""
    database = get_db()
    result = database.users.update_one(
        {'username': username},
        {'$set': {
            'mt5_login': int(mt5_login),
            'mt5_password': mt5_password,
            'mt5_server': mt5_server,
            'mt5_connected': True
        }}
    )
    return result.modified_count > 0

def get_user_mt5_credentials(username):
    """Get MT5 credentials for a user"""
    try:
        database = get_db()
        if database is None:
            return None
        user_data = database.users.find_one({'username': username})
        if user_data and user_data.get('mt5_login'):
            return {
                'login': user_data.get('mt5_login'),
                'password': user_data.get('mt5_password'),
                'server': user_data.get('mt5_server')
            }
    except (ServerSelectionTimeoutError, ConnectionFailure, NetworkTimeout, AutoReconnect) as e:
        print(f"⚠️ MongoDB connection error in get_user_mt5_credentials: {str(e)[:80]}")
    except Exception as e:
        print(f"⚠️ Error getting MT5 credentials: {str(e)[:80]}")
    return None

def disconnect_mt5(username):
    """Disconnect MT5 for a user"""
    database = get_db()
    result = database.users.update_one(
        {'username': username},
        {'$set': {
            'mt5_connected': False
        },
        '$unset': {
            'mt5_login': '',
            'mt5_password': '',
            'mt5_server': ''
        }}
    )
    return result.modified_count > 0

# ---------------- TRADING LOGS ----------------
def add_trading_log(username, log_type, message, details=None):
    """Add a trading log entry"""
    database = get_db()
    log_data = {
        'username': username,
        'type': log_type,  # 'trade', 'signal', 'error', 'info', 'bot'
        'message': message,
        'details': details or {},
        'created_at': datetime.utcnow()
    }
    result = database.trading_logs.insert_one(log_data)
    return result.inserted_id

def get_trading_logs(username, limit=100):
    """Get trading logs for a user"""
    database = get_db()
    logs = list(database.trading_logs.find(
        {'username': username}
    ).sort('created_at', -1).limit(limit))
    return logs

def get_all_trading_logs(limit=200):
    """Get all trading logs (admin only)"""
    database = get_db()
    logs = list(database.trading_logs.find().sort('created_at', -1).limit(limit))
    return logs

def clear_trading_logs(username):
    """Clear trading logs for a user"""
    database = get_db()
    result = database.trading_logs.delete_many({'username': username})
    return result.deleted_count

# ---------------- DEFAULT ADMIN ----------------
def create_default_admin():
    """Create default admin user if it doesn't exist"""
    database = get_db()
    if not database.users.find_one({'username': 'admin'}):
        user_data = {
            'username': 'admin',
            'email': 'admin@tradingbot.com',
            'password_hash': generate_password_hash('admin123'),
            'role': 'admin',
            'created_at': datetime.utcnow(),
            'mt5_login': 10009413572,
            'mt5_password': '@3BhJfGr',
            'mt5_server': 'MetaQuotes-Demo',
            'mt5_connected': True
        }
        database.users.insert_one(user_data)
        print("✅ Default admin user created (admin/admin123)")
        return True
    return False


# ---------------- EMAIL VERIFICATION ----------------
def generate_verification_code(length=6):
    """Generate a random numeric verification code"""
    return ''.join(random.choices(string.digits, k=length))

def store_pending_verification(username, email, password, code, expiry_minutes=10):
    """Store pending user registration with verification code"""
    database = get_db()
    
    # Remove any existing pending verification for this email
    database.pending_verifications.delete_many({'email': email})
    
    pending_data = {
        'username': username,
        'email': email,
        'password_hash': generate_password_hash(password),
        'code': code,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(minutes=expiry_minutes)
    }
    
    result = database.pending_verifications.insert_one(pending_data)
    return result.inserted_id

def verify_code_and_create_user(email, code):
    """Verify the code and create the user if valid"""
    database = get_db()
    
    # Find pending verification
    pending = database.pending_verifications.find_one({
        'email': email,
        'code': code,
        'expires_at': {'$gt': datetime.utcnow()}
    })
    
    if not pending:
        # Check if code exists but expired
        expired = database.pending_verifications.find_one({
            'email': email,
            'code': code
        })
        if expired:
            return None, "Verification code has expired. Please sign up again."
        return None, "Invalid verification code"
    
    # Check if username already exists (someone else may have taken it)
    if database.users.find_one({'username': pending['username']}):
        database.pending_verifications.delete_one({'_id': pending['_id']})
        return None, "Username has already been taken. Please sign up again with a different username."
    
    # Check if email already exists
    if database.users.find_one({'email': pending['email']}):
        database.pending_verifications.delete_one({'_id': pending['_id']})
        return None, "Email has already been registered. Please login or use a different email."
    
    # Create the user
    user_data = {
        'username': pending['username'],
        'email': pending['email'],
        'password_hash': pending['password_hash'],
        'role': 'user',
        'created_at': datetime.utcnow(),
        'email_verified': True
    }
    
    result = database.users.insert_one(user_data)
    user_data['_id'] = result.inserted_id
    
    # Remove pending verification
    database.pending_verifications.delete_one({'_id': pending['_id']})
    
    return User(user_data), None

def get_pending_verification(email):
    """Get pending verification for an email"""
    database = get_db()
    return database.pending_verifications.find_one({
        'email': email,
        'expires_at': {'$gt': datetime.utcnow()}
    })

def resend_verification_code(email):
    """Generate and update verification code for pending registration"""
    database = get_db()
    
    pending = database.pending_verifications.find_one({'email': email})
    if not pending:
        return None, "No pending registration found for this email"
    
    new_code = generate_verification_code()
    
    database.pending_verifications.update_one(
        {'email': email},
        {'$set': {
            'code': new_code,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(minutes=10)
        }}
    )
    
    return new_code, None

def cleanup_expired_verifications():
    """Remove expired pending verifications"""
    database = get_db()
    result = database.pending_verifications.delete_many({
        'expires_at': {'$lt': datetime.utcnow()}
    })
    return result.deleted_count


# ---------------- PASSWORD RESET ----------------
def store_password_reset(email, code, expiry_minutes=15):
    """Store password reset code for a user"""
    database = get_db()
    
    # Remove any existing reset codes for this email
    database.password_resets.delete_many({'email': email})
    
    reset_data = {
        'email': email,
        'code': code,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(minutes=expiry_minutes),
        'used': False
    }
    
    result = database.password_resets.insert_one(reset_data)
    return result.inserted_id

def verify_reset_code(email, code):
    """Verify the password reset code"""
    database = get_db()
    
    reset = database.password_resets.find_one({
        'email': email,
        'code': code,
        'expires_at': {'$gt': datetime.utcnow()},
        'used': False
    })
    
    if not reset:
        # Check if code exists but expired
        expired = database.password_resets.find_one({
            'email': email,
            'code': code
        })
        if expired:
            if expired.get('used'):
                return None, "This reset code has already been used"
            return None, "Reset code has expired. Please request a new one."
        return None, "Invalid reset code"
    
    return reset, None

def reset_user_password(email, code, new_password):
    """Reset user password after code verification"""
    database = get_db()
    
    # Verify code first
    reset, error = verify_reset_code(email, code)
    if error:
        return False, error
    
    # Update user password
    result = database.users.update_one(
        {'email': email},
        {'$set': {
            'password_hash': generate_password_hash(new_password),
            'updated_at': datetime.utcnow()
        }}
    )
    
    if result.modified_count == 0:
        return False, "Failed to update password. User not found."
    
    # Mark reset code as used
    database.password_resets.update_one(
        {'_id': reset['_id']},
        {'$set': {'used': True}}
    )
    
    return True, None

def resend_reset_code(email):
    """Generate and update password reset code"""
    database = get_db()
    
    # Check if user exists
    user = database.users.find_one({'email': email})
    if not user:
        return None, "No account found with this email"
    
    new_code = generate_verification_code()
    
    # Remove old codes and create new one
    database.password_resets.delete_many({'email': email})
    store_password_reset(email, new_code)
    
    return new_code, None


def change_user_password(username, current_password, new_password):
    """Change password for logged-in user (requires current password verification)"""
    database = get_db()
    
    # Find user
    user_doc = database.users.find_one({'username': username})
    if not user_doc:
        return False, "User not found"
    
    # Verify current password
    if not check_password_hash(user_doc.get('password_hash', ''), current_password):
        return False, "Current password is incorrect"
    
    # Validate new password
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters"
    
    if current_password == new_password:
        return False, "New password must be different from current password"
    
    # Update password
    result = database.users.update_one(
        {'username': username},
        {'$set': {
            'password_hash': generate_password_hash(new_password),
            'updated_at': datetime.utcnow()
        }}
    )
    
    if result.modified_count == 0:
        return False, "Failed to update password"
    
    return True, None


# ---------------- PASSWORD CHANGE OTP ----------------
def store_password_change_otp(username, code, expiry_minutes=15):
    """Store password change OTP code for a logged-in user"""
    database = get_db()
    
    # Remove any existing OTP codes for this user
    database.password_change_otps.delete_many({'username': username})
    
    otp_data = {
        'username': username,
        'code': code,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(minutes=expiry_minutes),
        'used': False
    }
    
    result = database.password_change_otps.insert_one(otp_data)
    return result.inserted_id


def verify_password_change_otp(username, code):
    """Verify the password change OTP code"""
    database = get_db()
    
    otp = database.password_change_otps.find_one({
        'username': username,
        'code': code,
        'expires_at': {'$gt': datetime.utcnow()},
        'used': False
    })
    
    if not otp:
        # Check if code exists but expired or already used
        expired = database.password_change_otps.find_one({
            'username': username,
            'code': code
        })
        if expired:
            if expired.get('used'):
                return None, "This OTP code has already been used"
            return None, "OTP code has expired. Please request a new one."
        return None, "Invalid OTP code"
    
    return otp, None


def complete_password_change_with_otp(username, code, new_password):
    """Complete password change after OTP verification"""
    database = get_db()
    
    # Verify OTP first
    otp, error = verify_password_change_otp(username, code)
    if error:
        return False, error
    
    # Validate new password
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters"
    
    # Update user password
    result = database.users.update_one(
        {'username': username},
        {'$set': {
            'password_hash': generate_password_hash(new_password),
            'updated_at': datetime.utcnow()
        }}
    )
    
    if result.modified_count == 0:
        return False, "Failed to update password. User not found."
    
    # Mark OTP code as used
    database.password_change_otps.update_one(
        {'_id': otp['_id']},
        {'$set': {'used': True}}
    )
    
    return True, None


def resend_password_change_otp(username):
    """Generate and update password change OTP code"""
    database = get_db()
    
    # Check if user exists
    user = database.users.find_one({'username': username})
    if not user:
        return None, "User not found"
    
    new_code = generate_verification_code()
    
    # Remove old codes and create new one
    database.password_change_otps.delete_many({'username': username})
    store_password_change_otp(username, new_code)
    
    return new_code, None
