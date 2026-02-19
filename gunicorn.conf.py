# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:10000"

# Worker processes
workers = 1  # Keep it low for free tier
worker_class = "sync"
worker_connections = 1000

# Timeouts
timeout = 120  # Increase timeout to 120 seconds (default is 30)
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "trading-bot"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Preload app
preload_app = False
