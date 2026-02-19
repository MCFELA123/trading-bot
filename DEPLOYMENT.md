# Trading Bot Deployment Guide

## Render Deployment

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
gunicorn -c gunicorn.conf.py app:app
```

### Environment Variables

Add these in your Render dashboard:

```
MONGODB_URI=mongodb+srv://mcfela389:CAgqLXzp0N6qiPL5@cluster0.okvkw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0

SECRET_KEY=your-super-secret-key-change-this-in-production

DB_NAME=tradingbot

MAIL_SERVER=smtp.gmail.com
MAIL_PORT=465
MAIL_USE_TLS=False
MAIL_USE_SSL=True
MAIL_USERNAME=imohaniekan002@gmail.com
MAIL_PASSWORD=zyzz mthf ysif wgpl
MAIL_DEFAULT_SENDER=imohaniekan002@gmail.com

VERIFICATION_CODE_EXPIRY=600

EMAIL_DEV_MODE=False
```

### Default Login
- Username: `admin`
- Password: `admin123`

## Notes
- MetaTrader5 features are disabled on Linux/Render (Windows only)
- Email sending has a 15-second timeout to prevent worker timeouts
- Gunicorn worker timeout is set to 120 seconds
