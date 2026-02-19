import os
from dotenv import load_dotenv

# Load environment variables FIRST before reading them
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'supersecretkey123'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or 'your-api-key-here'
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL') or 'gpt-4o'
    OPENAI_MODEL_FAST = os.environ.get('OPENAI_MODEL_FAST') or 'gpt-4o-mini'
    
    # Email Configuration for Verification
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', 'False').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')  # Your email address
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')  # App password (not regular password)
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER') or os.environ.get('MAIL_USERNAME')
    
    # Verification Settings
    VERIFICATION_CODE_EXPIRY = int(os.environ.get('VERIFICATION_CODE_EXPIRY') or 600)  # 10 minutes
    
    # Development Mode - Set to True to show verification code on screen instead of email
    # DISABLED - verification codes will be sent via Gmail
    EMAIL_DEV_MODE = os.environ.get('EMAIL_DEV_MODE', 'False').lower() == 'true'
